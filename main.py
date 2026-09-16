"""TMT-v2: P0-P4 recurrent byte model. Runtime arrays are private buffers."""
from __future__ import annotations
import itertools, math, os
from collections import deque
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import mlx.utils as util

BYTE_VOCAB, EOS, VOCAB_SIZE, EPS = 256, 256, 257, 1e-6
def logit(x):
    x=min(max(x,1e-6),1-1e-6); return math.log(x/(1-x))
def unit(x): return x/mx.sqrt(mx.sum(mx.square(x))+EPS)

class Encoder(nn.Module):
    def __init__(self,d):
        super().__init__(); self.embed=nn.Embedding(VOCAB_SIZE,d)
        self._embed_trace=mx.zeros((VOCAB_SIZE,d//2,2,2))
    def __call__(self,t): return self.embed(t)
    def reset_memory(self): self._embed_trace=mx.zeros(self._embed_trace.shape)

class Decoder(nn.Module):
    def __init__(self,d): super().__init__(); self.decode=nn.Linear(d,VOCAB_SIZE)
    def __call__(self,x): return self.decode(x)

class Layer(nn.Module):
    """Pairwise trace-compatible recurrent dynamics with raw, exact base angles."""
    def __init__(self,d,min_half,max_half):
        super().__init__()
        if d%2: raise ValueError("dim must be even")
        self.dim=d; n=d//2; nh=max(1,round(math.sqrt(n))); np=math.ceil(n/nh)
        half=[math.exp(math.log(min_half)+(math.log(max_half)-math.log(min_half))*i/max(nh-1,1)) for i in range(nh)]
        canonical=[2,3,4,5,8,12,16,24,32,48,64,96,128,256,512,2048]
        periods=canonical[:np] if np<=len(canonical) else [2*math.exp(math.log(1024)*i/max(np-1,1)) for i in range(np)]
        pairs=[(half[i%nh],periods[(i//nh)%np]) for i in range(n)]
        self.decay=mx.array([logit(2**(-1/h)) for h,_ in pairs])
        # Stability comes from rho, so phase is an unconstrained raw angle.
        self.phase=mx.array([2*math.pi/period for _,period in pairs])
        self.write_scale=mx.zeros((d,)); self.write_bias=mx.zeros((d,))
        self.phase_scale=mx.zeros((n,)); self.phase_bias=mx.zeros((n,))
        self.norm=nn.LayerNorm(d); self.weights=nn.Linear(d,d,bias=False); self.silu=nn.SiLU()
        self._state=mx.zeros((d,)); self._decay_trace=mx.zeros((d,)); self._phase_trace=mx.zeros((d,))
        self._write_scale_trace=mx.zeros((d,)); self._write_bias_trace=mx.zeros((d,))
        self._phase_scale_trace=mx.zeros((d,)); self._phase_bias_trace=mx.zeros((d,))
    @staticmethod
    def rotate(v,c,s):
        a,b=v[...,0::2],v[...,1::2]
        return mx.stack((c*a-s*b,s*a+c*b),axis=-1).reshape(v.shape)
    @staticmethod
    def j(v):
        a,b=v[...,0::2],v[...,1::2]
        return mx.stack((-b,a),axis=-1).reshape(v.shape)
    def dynamics(self,x):
        rho=mx.sigmoid(self.decay); pair_x=.5*(x[0::2]+x[1::2]); z=self.phase_scale*pair_x+self.phase_bias
        theta=self.phase+math.pi*mx.tanh(z)
        return rho,mx.cos(theta),mx.sin(theta),mx.sqrt(mx.maximum(1-mx.square(rho),EPS)),z,pair_x
    def step(self,x,state,dummy=None):
        rho,c,s,scale,z,pair_x=self.dynamics(x); pair=lambda z:mx.stack((z,z),axis=-1).reshape((self.dim,))
        gate=mx.sigmoid(self.write_scale*x+self.write_bias); rotated=self.rotate(state,c,s)
        nxt=pair(rho)*rotated+pair(scale)*gate*x
        if dummy is not None: nxt=nxt+dummy
        return x+self.silu(self.weights(self.norm(nxt))),nxt,(rho,c,s,scale,gate,rotated,z,pair_x)
    def history(self,trace,rho,c,s): return mx.stack((rho,rho),axis=-1).reshape((self.dim,))*self.rotate(trace,c,s)
    def phase_direct(self,rho,c,s,state,theta_derivative):
        pair=lambda z:mx.stack((z,z),axis=-1).reshape((self.dim,))
        return pair(rho*theta_derivative)*self.rotate(self.j(state),c,s)
    def next_decay_trace(self,x,rho,c,s,scale,gate,rotated):
        dr=rho*(1-rho); ds=-rho*dr/mx.maximum(scale,EPS); pair=lambda z:mx.stack((z,z),axis=-1).reshape((self.dim,))
        return self.history(self._decay_trace,rho,c,s)+pair(dr)*rotated+pair(ds)*gate*x
    def injection_diagonal(self,x,scale,gate):
        pair=lambda z:mx.stack((z,z),axis=-1).reshape((self.dim,))
        return pair(scale)*(gate+x*gate*(1-gate)*self.write_scale)
    def embedding_block(self,x,state,rho,c,s,scale,gate,z):
        """Exact local 2x2 ds_pair/dx_pair, including input-dependent rotation."""
        n=self.dim//2; pair=lambda q:mx.stack((q,q),axis=-1).reshape((self.dim,))
        diag=self.injection_diagonal(x,scale,gate).reshape((n,2))
        direct=mx.stack((mx.stack((diag[:,0],mx.zeros((n,))),axis=-1),mx.stack((mx.zeros((n,)),diag[:,1]),axis=-1)),axis=-2)
        sech2=1-mx.square(mx.tanh(z)); gamma=.5*math.pi*self.phase_scale*sech2
        v=(pair(rho)*self.rotate(self.j(state),c,s)).reshape((n,2))
        return direct+v[:,:,None]*gamma[:,None,None]
    def trace_block_history(self,trace,rho,c,s):
        a,b=trace[...,0,:],trace[...,1,:]
        return mx.stack((c[None,:,None]*a-s[None,:,None]*b,s[None,:,None]*a+c[None,:,None]*b),axis=-2)*rho[None,:,None,None]
    def reset_memory(self):
        self._state=mx.zeros((self.dim,)); self._decay_trace=mx.zeros((self.dim,)); self._phase_trace=mx.zeros((self.dim,))
        self._write_scale_trace=mx.zeros((self.dim,)); self._write_bias_trace=mx.zeros((self.dim,))
        self._phase_scale_trace=mx.zeros((self.dim,)); self._phase_bias_trace=mx.zeros((self.dim,))

class AssociativeMemory(nn.Module):
    """P3 delta memory: forget first, then correct its post-forget retrieval."""
    def __init__(self,d,m,half_life=1024):
        super().__init__(); self.memory_dim=m
        self.key=nn.Linear(d,m,bias=False); self.query=nn.Linear(d,m,bias=False)
        self.value=nn.Linear(d,m,bias=False); self.out=nn.Linear(m,d,bias=False)
        self.forget=nn.Linear(d,1); self.rate=nn.Linear(d,1)
        self.forget.bias=mx.array([logit(2**(-1/half_life))]); self.rate.bias=mx.array([logit(.4)])
        self._matrix=mx.zeros((m,m))
    def read(self,h,matrix=None): return (self._matrix if matrix is None else matrix)@unit(self.query(h))
    def next_matrix(self,h,matrix=None):
        matrix=self._matrix if matrix is None else matrix; k,v=unit(self.key(h)),mx.tanh(self.value(h))
        decayed=mx.sigmoid(self.forget(h))[0]*matrix; eta=.25*mx.sigmoid(self.rate(h))[0]
        err=v-decayed@k
        return decayed+eta*err[:,None]*k[None,:]
    def writer_loss(self,h,target,matrix=None):
        candidate=self.next_matrix(h,matrix); retrieved=candidate@unit(self.key(h))
        return 1-mx.sum(unit(self.out(retrieved))*unit(target))
    def reset_memory(self): self._matrix=mx.zeros((self.memory_dim,self.memory_dim))


class Model(nn.Module):
    def __init__(self,dim,layers,temp,lr,memory_dim=None,future_block=16,target_ema=.995,min_half_life=4,max_half_life=65536,latent_weight=.25,variance_weight=.02):
        super().__init__()
        if dim%2: raise ValueError("dim must be even")
        self.dim,self.layercount,self.temp=dim,layers,temp
        self.future_block,self.target_ema=future_block,target_ema
        self.horizons=(1,2,4,8,16)
        self.latent_weight,self.variance_weight,self.writer_weight=latent_weight,variance_weight,.10
        self.encoder,self.decoder=Encoder(dim),Decoder(dim)
        self.layers=[Layer(dim,min_half_life,max_half_life) for _ in range(layers)]
        self.predictors={f"h{h}":nn.Linear(dim,dim) for h in self.horizons}
        self.memory=AssociativeMemory(dim,memory_dim or max(16,min(64,dim//8)))
        self.optimizer=opt.AdamW(learning_rate=lr)
        self.memory_optimizer=opt.AdamW(learning_rate=lr)
        self._target_embed=mx.array(self.encoder.embed.weight)
        self._repr_buffer=mx.zeros((0,dim)); self._repr_capacity=32
    def buffer_parameter_invariant(self):
        names=" ".join(str(k) for k,_ in util.tree_flatten(self.parameters()))
        return not any(x in names for x in ("_state","_trace","_matrix","_target_embed","_repr_buffer"))
    def reset_memory(self):
        for layer in self.layers: layer.reset_memory()
        self.encoder.reset_memory(); self.memory.reset_memory()
        self._repr_buffer=mx.zeros((0,self.dim))
    def core(self,token,dummies=None,states_override=None,matrix_override=None):
        x=self.encoder(mx.array(token) if isinstance(token,int) else token)
        states=[]; dyn=[]; inputs=[]
        for i,layer in enumerate(self.layers):
            inputs.append(x); previous=layer._state if states_override is None else states_override[i]
            x,state,d=layer.step(x,previous,None if dummies is None else dummies[i])
            states.append(state); dyn.append(d)
        h=x+self.memory.out(self.memory.read(x,matrix_override))
        return self.decoder(h),h,states,dyn,inputs
    def commit(self,states,h):
        for layer,state in zip(self.layers,states): layer._state=mx.stop_gradient(state)
        self.memory._matrix=mx.stop_gradient(self.memory.next_matrix(h))
        mx.eval(*[x._state for x in self.layers],self.memory._matrix)
    def frozen_step(self,token):
        """P0 frozen weights, stateful memory. Never runs AdamW or eligibility updates."""
        logits,_,states,_,_=self.represent(token); return logits
    def represent(self,token):
        """Advance state and return the true post-associative-memory representation."""
        logits,h,states,dynamics,inputs=self.core(token); self.commit(states,h); return logits,h,states,dynamics,inputs
    def stateless_step(self,token):
        """P0 valid no-memory ablation that preserves live buffers."""
        old=[x._state for x in self.layers]; matrix=self.memory._matrix
        decay=[x._decay_trace for x in self.layers]; phase=[x._phase_trace for x in self.layers]; ws=[x._write_scale_trace for x in self.layers]; wb=[x._write_bias_trace for x in self.layers]; ps=[x._phase_scale_trace for x in self.layers]; pb=[x._phase_bias_trace for x in self.layers]
        embed_trace, repr_buffer = self.encoder._embed_trace, self._repr_buffer
        self.reset_memory(); logits,*_=self.core(token)
        for layer,state,dtrace,ptrace,wst,wbt,pst,pbt in zip(self.layers,old,decay,phase,ws,wb,ps,pb):
            layer._state,layer._decay_trace,layer._phase_trace,layer._write_scale_trace,layer._write_bias_trace,layer._phase_scale_trace,layer._phase_bias_trace=state,dtrace,ptrace,wst,wbt,pst,pbt
        self.memory._matrix, self.encoder._embed_trace = matrix, embed_trace
        self._repr_buffer = repr_buffer
        return logits
    def sample(self,logits):
        p=mx.softmax(logits); entropy=-mx.sum(p*mx.log(p+1e-8))/math.log(VOCAB_SIZE)
        return mx.random.categorical(logits/mx.maximum(.1,self.temp*(1-.5*entropy)).item()).item()
    def target(self,future,horizon):
        values=list(future)[:horizon]
        if not values: raise ValueError("future block is empty")
        weights=mx.array([math.exp(-(i+1)/max(horizon,1)) for i in range(len(values))])
        weights=weights/mx.sum(weights)
        return mx.stop_gradient(mx.sum(self._target_embed[mx.array(values)]*weights[:,None],axis=0))
    def latent_loss(self,h,future):
        usable=[horizon for horizon in self.horizons if len(future)>=horizon]
        return mx.mean(mx.stack([1-mx.sum(unit(self.predictors[f"h{horizon}"](h))*unit(self.target(future,horizon))) for horizon in usable]))
    def repr_loss(self,h):
        xs=mx.concatenate((self._repr_buffer,h[None,:]),axis=0)
        if xs.shape[0]<2: return mx.array(0.)
        return mx.mean(mx.maximum(0.,1-mx.sqrt(mx.var(xs,axis=0)+1e-4)))
    def train_step(self,current,target,future=None):
        """Online structured eligibility for every recurrent-dynamics parameter family."""
        if not 0<=current<VOCAB_SIZE or not 0<=target<VOCAB_SIZE: raise ValueError("invalid token")
        future=tuple(future or (target,)); params=self.trainable_parameters(); dummies=[mx.zeros((self.dim,)) for _ in self.layers]
        def lossfn(updated,ds):
            self.update(updated); logits,h,states,dyn,inputs=self.core(current,ds)
            byte=nn.losses.cross_entropy(logits[None,:],mx.array([target])).mean(); latent=self.latent_loss(h,future); variance=self.repr_loss(h)
            writer=self.memory.writer_loss(h,self.target(future,min(len(future),self.horizons[-1])))
            return byte+self.latent_weight*latent+self.variance_weight*variance+self.writer_weight*writer,(byte,latent,variance,writer,h,states,dyn,inputs)
        (loss,aux),(grads,dldstate)=mx.value_and_grad(lossfn,argnums=(0,1))(params,dummies)
        byte,latent,variance,writer,h,states,dyn,inputs=aux
        rho,c,s,scale,gate,rot,z,pair_x=dyn[0]; n=self.dim//2
        block_hist=self.layers[0].trace_block_history(self.encoder._embed_trace,rho,c,s)
        signal=dldstate[0].reshape((n,2))
        embed_hist=mx.sum(block_hist*signal[None,:,:,None],axis=2).reshape((VOCAB_SIZE,self.dim))
        grads["encoder"]["embed"]["weight"]=grads["encoder"]["embed"]["weight"]+embed_hist
        block_next=block_hist+(mx.arange(VOCAB_SIZE)==current)[:,None,None,None]*self.layers[0].embedding_block(inputs[0],self.layers[0]._state,rho,c,s,scale,gate,z)[None,:,:,:]
        traces=[]
        for i,layer in enumerate(self.layers):
            rho,c,s,scale,gate,rot,z,pair_x=dyn[i]; pair=lambda q:mx.stack((q,q),axis=-1).reshape((self.dim,))
            def add_vector(name,trace): grads["layers"][i][name]=grads["layers"][i][name]+dldstate[i]*layer.history(trace,rho,c,s)
            def add_pair(name,trace): grads["layers"][i][name]=grads["layers"][i][name]+mx.sum((dldstate[i]*layer.history(trace,rho,c,s)).reshape((-1,2)),axis=1)
            add_pair("decay",layer._decay_trace); add_pair("phase",layer._phase_trace)
            add_vector("write_scale",layer._write_scale_trace); add_vector("write_bias",layer._write_bias_trace)
            add_pair("phase_scale",layer._phase_scale_trace); add_pair("phase_bias",layer._phase_bias_trace)
            sech2=1-mx.square(mx.tanh(z)); phase_base=layer.phase_direct(rho,c,s,layer._state,mx.ones(z.shape))
            write_scale_direct=pair(scale)*gate*(1-gate)*mx.square(inputs[i]); write_bias_direct=pair(scale)*gate*(1-gate)*inputs[i]
            traces.append((
                layer.next_decay_trace(inputs[i],rho,c,s,scale,gate,rot),
                layer.history(layer._phase_trace,rho,c,s)+phase_base,
                layer.history(layer._write_scale_trace,rho,c,s)+write_scale_direct,
                layer.history(layer._write_bias_trace,rho,c,s)+write_bias_direct,
                layer.history(layer._phase_scale_trace,rho,c,s)+layer.phase_direct(rho,c,s,layer._state,math.pi*sech2*pair_x),
                layer.history(layer._phase_bias_trace,rho,c,s)+layer.phase_direct(rho,c,s,layer._state,math.pi*sech2),
            ))
        self.optimizer.update(self,grads)
        for layer,state,values in zip(self.layers,states,traces):
            layer._state=mx.stop_gradient(state); layer._decay_trace,layer._phase_trace,layer._write_scale_trace,layer._write_bias_trace,layer._phase_scale_trace,layer._phase_bias_trace=[mx.stop_gradient(x) for x in values]
        self.encoder._embed_trace=mx.stop_gradient(block_next); self.memory._matrix=mx.stop_gradient(self.memory.next_matrix(h))
        self._repr_buffer=mx.concatenate((self._repr_buffer,mx.stop_gradient(h)[None,:]),axis=0)[-self._repr_capacity:]; self._target_embed=mx.stop_gradient(self.target_ema*self._target_embed+(1-self.target_ema)*self.encoder.embed.weight)
        mx.eval(self.parameters(),self.optimizer.state,self.encoder._embed_trace,self._target_embed,self._repr_buffer,self.memory._matrix,*[x._state for x in self.layers],*[x._decay_trace for x in self.layers],*[x._phase_trace for x in self.layers])
        return {"loss":loss.item(),"byte":byte.item(),"latent":latent.item(),"variance":variance.item(),"writer":writer.item()}
    def memory_temporal_step(self,writes,queries,targets):
        """Short BPTT only through delta-memory updates, using future learned queries.

        Inputs are detached recurrent representations. This gives K,V,Q,forget,
        rate and out genuine delayed-retrieval credit without retaining the SSM graph.
        """
        if not (len(writes)==len(queries)==len(targets)) or not writes: raise ValueError("matched nonempty memory sequences required")
        params=self.memory.trainable_parameters(); initial=mx.stop_gradient(self.memory._matrix)
        def lossfn(updated):
            self.memory.update(updated); matrix=initial; losses=[]
            for write,query,target in zip(writes,queries,targets):
                matrix=self.memory.next_matrix(mx.stop_gradient(write),matrix)
                prediction=self.memory.out(self.memory.read(mx.stop_gradient(query),matrix))
                losses.append(1-mx.sum(unit(prediction)*unit(mx.stop_gradient(target))))
            return mx.mean(mx.stack(losses)),matrix
        (loss,matrix),grads=mx.value_and_grad(lossfn)(params)
        self.memory_optimizer.update(self.memory,grads); self.memory._matrix=mx.stop_gradient(matrix)
        mx.eval(self.memory.parameters(),self.memory_optimizer.state,self.memory._matrix)
        if not math.isfinite(loss.item()): raise FloatingPointError("non-finite temporal memory loss")
        return loss.item()
    def adaptive_patches(self,data,max_patch=16,entropy_threshold=.35):
        """P4 entropy scheduler; reports patch boundaries without claiming byte skipping."""
        patch=[]; ent=[]
        for token in data:
            logits=self.frozen_step(token); p=mx.softmax(logits)
            e=(-mx.sum(p*mx.log(p+1e-8))/math.log(VOCAB_SIZE)).item()
            patch.append(token); ent.append(e)
            if e>entropy_threshold or len(patch)>=max_patch:
                yield tuple(patch),sum(ent)/len(ent); patch=[]; ent=[]
        if patch: yield tuple(patch),sum(ent)/len(ent)
    def save(self,path):
        data={f"m.{k}":v for k,v in util.tree_flatten(self.parameters())}
        data.update({f"o.{k}":v for k,v in util.tree_flatten(self.optimizer.state)})
        data.update({"buffer.embed_trace":self.encoder._embed_trace,"buffer.target_embed":self._target_embed,"buffer.repr":self._repr_buffer,"buffer.memory":self.memory._matrix})
        for i,x in enumerate(self.layers):
            data[f"buffer.state.{i}"],data[f"buffer.decay_trace.{i}"],data[f"buffer.phase_trace.{i}"]=x._state,x._decay_trace,x._phase_trace
            data[f"buffer.write_scale_trace.{i}"],data[f"buffer.write_bias_trace.{i}"]=x._write_scale_trace,x._write_bias_trace
            data[f"buffer.phase_scale_trace.{i}"],data[f"buffer.phase_bias_trace.{i}"]=x._phase_scale_trace,x._phase_bias_trace
        tmp=path+".temporary.safetensors"; mx.save_safetensors(tmp,data,metadata={"format":"tmt-v3"}); os.replace(tmp,path)
    def load(self,path):
        if not os.path.exists(path): return False
        data=mx.load(path)
        if "buffer.target_embed" not in data:
            raise ValueError("legacy checkpoint is incompatible with TMT-v3; retrain instead of partially loading it")
        model={}; optimizer={}
        for k,v in data.items():
            if k.startswith("m."): model[k[2:]]=v
            elif k.startswith("o."): optimizer[k[2:]]=v
            elif k=="buffer.embed_trace": self.encoder._embed_trace=v
            elif k=="buffer.target_embed": self._target_embed=v
            elif k=="buffer.repr": self._repr_buffer=v
            elif k=="buffer.memory": self.memory._matrix=v
            elif k.startswith("buffer.state."): self.layers[int(k.rsplit(".",1)[1])]._state=v
            elif k.startswith("buffer.decay_trace."): self.layers[int(k.rsplit(".",1)[1])]._decay_trace=v
            elif k.startswith("buffer.phase_trace."): self.layers[int(k.rsplit(".",1)[1])]._phase_trace=v
            elif k.startswith("buffer.write_scale_trace."): self.layers[int(k.rsplit(".",1)[1])]._write_scale_trace=v
            elif k.startswith("buffer.write_bias_trace."): self.layers[int(k.rsplit(".",1)[1])]._write_bias_trace=v
            elif k.startswith("buffer.phase_scale_trace."): self.layers[int(k.rsplit(".",1)[1])]._phase_scale_trace=v
            elif k.startswith("buffer.phase_bias_trace."): self.layers[int(k.rsplit(".",1)[1])]._phase_bias_trace=v
        current=dict(util.tree_flatten(self.parameters()))
        required={f"buffer.phase_trace.{i}" for i in range(self.layercount)}
        if set(model)!=set(current) or any(model[k].shape!=current[k].shape for k in current) or not required.issubset(data):
            raise ValueError("checkpoint is not an exact TMT-v3 parameter/buffer match")
        self.update(util.tree_unflatten(list(model.items())))
        if optimizer: self.optimizer.state=util.tree_unflatten(list(optimizer.items()))
        return True

def continuous_bytes(files):
    files=[Path(x) for x in files]
    if not files: raise FileNotFoundError("no corpus files found")
    while True:
        for file in files:
            with file.open("rb") as f:
                while chunk:=f.read(65536): yield from chunk
            yield EOS
def stream_examples(stream,future_block):
    it=iter(stream); window=deque(itertools.islice(it,future_block+1))
    while len(window)>=2:
        current=window.popleft()
        yield current,window[0],tuple(itertools.islice(window,0,future_block))
        window.append(next(it))

class Runtime:
    def __init__(self,path,**kwargs): self.model,self.path,self.step=Model(**kwargs),path,0
    def train(self,glob_pattern="wikipedia_clean/**/wiki_*"):
        files=sorted(Path().glob(glob_pattern))
        for c,n,future in stream_examples(continuous_bytes(files),self.model.future_block):
            m=self.model.train_step(c,n,future); self.step+=1
            if self.step%100==0: print(f"{self.step}: loss={m['loss']:.4f} byte={m['byte']:.4f} latent={m['latent']:.4f}")
            if self.step%500==0: self.model.save(self.path)
    def chat(self):
        while True:
            text=input("\nUser >> "); logits=None
            for token in (*text.encode("utf-8"),EOS): logits=self.model.frozen_step(token)
            print("Model >> ",end="",flush=True)
            while True:
                token=self.model.sample(logits)
                if token==EOS: print(); break
                print(bytes((token,)).decode("utf-8",errors="replace"),end="",flush=True)
                logits=self.model.frozen_step(token)
    def __call__(self):
        mode=input("mode [train, chat, math] >> ").strip().lower(); self.model.load(self.path)
        try:
            if mode=="train": self.train()
            elif mode=="chat": self.chat()
            elif mode=="math":
                from math_diagnostics import main; main()
            else: raise ValueError("mode must be train, chat, or math")
        finally: self.model.save(self.path)
if __name__=="__main__":
    Runtime(path="tmt-v3.safetensors",dim=512,layers=16,temp=.75,lr=5e-4)()
