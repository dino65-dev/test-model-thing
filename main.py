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
        self._embed_trace=mx.zeros((VOCAB_SIZE,d))
    def __call__(self,t): return self.embed(t)
    def reset_memory(self): self._embed_trace=mx.zeros(self._embed_trace.shape)

class Decoder(nn.Module):
    def __init__(self,d): super().__init__(); self.decode=nn.Linear(d,VOCAB_SIZE)
    def __call__(self,x): return self.decode(x)

class Layer(nn.Module):
    """s_t = rho R(omega)s_(t-1) + sqrt(1-rho^2) gate(x_t)x_t."""
    def __init__(self,d,min_half,max_half):
        super().__init__()
        if d%2: raise ValueError("dim must be even")
        self.dim=d; n=d//2
        grid=lambda lo,hi:[math.exp(math.log(lo)+(math.log(hi)-math.log(lo))*i/max(n-1,1)) for i in range(n)]
        self.decay=mx.array([logit(2**(-1/h)) for h in grid(min_half,max_half)])
        self.phase=mx.array([logit(2/p) for p in grid(4,2048)])
        self.write=nn.Linear(d,d); self.norm=nn.LayerNorm(d)
        self.weights=nn.Linear(d,d,bias=False); self.silu=nn.SiLU()
        self._state=mx.zeros((d,)); self._decay_trace=mx.zeros((d,))
    @staticmethod
    def rotate(v,c,s):
        a,b=v[...,0::2],v[...,1::2]
        return mx.stack((c*a-s*b,s*a+c*b),axis=-1).reshape(v.shape)
    def dynamics(self):
        rho=mx.sigmoid(self.decay); w=math.pi*mx.sigmoid(self.phase)
        return rho,mx.cos(w),mx.sin(w),mx.sqrt(mx.maximum(1-mx.square(rho),EPS))
    def step(self,x,state,dummy=None):
        rho,c,s,scale=self.dynamics()
        pair=lambda z:mx.stack((z,z),axis=-1).reshape((self.dim,))
        gate=mx.sigmoid(self.write(x)); rotated=self.rotate(state,c,s)
        nxt=pair(rho)*rotated+pair(scale)*gate*x
        if dummy is not None: nxt=nxt+dummy
        return x+self.silu(self.weights(self.norm(nxt))),nxt,(rho,c,s,scale,gate,rotated)
    def history(self,rho,c,s):
        return mx.stack((rho,rho),axis=-1).reshape((self.dim,))*self.rotate(self._decay_trace,c,s)
    def next_trace(self,x,rho,c,s,scale,gate,rotated):
        dr=rho*(1-rho); ds=-rho*dr/mx.maximum(scale,EPS)
        pair=lambda z:mx.stack((z,z),axis=-1).reshape((self.dim,))
        return self.history(rho,c,s)+pair(dr)*rotated+pair(ds)*gate*x
    def reset_memory(self):
        self._state=mx.zeros((self.dim,)); self._decay_trace=mx.zeros((self.dim,))

class AssociativeMemory(nn.Module):
    """P3: bounded delta-rule fast-weight memory M <- lam*M+eta(v-Mk)kT."""
    def __init__(self,d,m):
        super().__init__(); self.memory_dim=m
        self.key=nn.Linear(d,m,bias=False); self.query=nn.Linear(d,m,bias=False)
        self.value=nn.Linear(d,m,bias=False); self.out=nn.Linear(m,d,bias=False)
        self.forget=nn.Linear(d,1); self.rate=nn.Linear(d,1)
        self._matrix=mx.zeros((m,m))
    def read(self,h): return self._matrix@unit(self.query(h))
    def next_matrix(self,h):
        k,v=unit(self.key(h)),mx.tanh(self.value(h))
        lam=mx.sigmoid(self.forget(h))[0]; eta=.25*mx.sigmoid(self.rate(h))[0]
        err=v-self._matrix@k
        return lam*self._matrix+eta*err[:,None]*k[None,:]
    def reset_memory(self): self._matrix=mx.zeros((self.memory_dim,self.memory_dim))

class Model(nn.Module):
    def __init__(self,dim,layers,temp,lr,memory_dim=None,future_block=16,target_ema=.995,min_half_life=4,max_half_life=65536,latent_weight=.25,variance_weight=.02):
        super().__init__()
        if dim%2: raise ValueError("dim must be even")
        self.dim,self.layercount,self.temp=dim,layers,temp
        self.future_block,self.target_ema=future_block,target_ema
        self.latent_weight,self.variance_weight=latent_weight,variance_weight
        self.encoder,self.decoder=Encoder(dim),Decoder(dim)
        self.layers=[Layer(dim,min_half_life,max_half_life) for _ in range(layers)]
        self.predictor=nn.Linear(dim,dim)
        self.memory=AssociativeMemory(dim,memory_dim or max(16,min(64,dim//8)))
        self.optimizer=opt.AdamW(learning_rate=lr)
        self._target_embed=mx.array(self.encoder.embed.weight)
        self._repr_buffer=mx.zeros((0,dim)); self._repr_capacity=32
    def buffer_parameter_invariant(self):
        names=" ".join(str(k) for k,_ in util.tree_flatten(self.parameters()))
        return not any(x in names for x in ("_state","_trace","_matrix","_target_embed","_repr_buffer"))
    def reset_memory(self):
        for layer in self.layers: layer.reset_memory()
        self.encoder.reset_memory(); self.memory.reset_memory()
        self._repr_buffer=mx.zeros((0,self.dim))
    def core(self,token,dummies=None):
        x=self.encoder(mx.array(token) if isinstance(token,int) else token)
        states=[]; dyn=[]; inputs=[]
        for i,layer in enumerate(self.layers):
            inputs.append(x); x,state,d=layer.step(x,layer._state,None if dummies is None else dummies[i])
            states.append(state); dyn.append(d)
        h=x+self.memory.out(self.memory.read(x))
        return self.decoder(h),h,states,dyn,inputs
    def commit(self,states,h):
        for layer,state in zip(self.layers,states): layer._state=mx.stop_gradient(state)
        self.memory._matrix=mx.stop_gradient(self.memory.next_matrix(h))
        mx.eval(*[x._state for x in self.layers],self.memory._matrix)
    def frozen_step(self,token):
        """P0 frozen weights, stateful memory. Never runs AdamW or eligibility updates."""
        logits,h,states,_,_=self.core(token); self.commit(states,h); return logits
    def stateless_step(self,token):
        """P0 valid no-memory ablation that preserves live buffers."""
        old=[x._state for x in self.layers]; matrix=self.memory._matrix
        embed_trace, repr_buffer = self.encoder._embed_trace, self._repr_buffer
        self.reset_memory(); logits,*_=self.core(token)
        for layer,state in zip(self.layers,old): layer._state=state
        self.memory._matrix, self.encoder._embed_trace = matrix, embed_trace
        self._repr_buffer = repr_buffer
        return logits
    def sample(self,logits):
        p=mx.softmax(logits); entropy=-mx.sum(p*mx.log(p+1e-8))/math.log(VOCAB_SIZE)
        return mx.random.categorical(logits/mx.maximum(.1,self.temp*(1-.5*entropy)).item()).item()
    def target(self,future):
        if not future: raise ValueError("future block is empty")
        return mx.stop_gradient(mx.mean(self._target_embed[mx.array(list(future))],axis=0))
    def repr_loss(self,h):
        xs=mx.concatenate((self._repr_buffer,h[None,:]),axis=0)
        if xs.shape[0]<2: return mx.array(0.)
        return mx.mean(mx.maximum(0.,1-mx.sqrt(mx.var(xs,axis=0)+1e-4)))
    def train_step(self,current,target,future=None):
        """P1 local traces; P2 contextual EMA target; P3 state update."""
        if not 0<=current<VOCAB_SIZE or not 0<=target<VOCAB_SIZE: raise ValueError("invalid token")
        future=tuple(future or (target,)); params=self.trainable_parameters()
        dummies=[mx.zeros((self.dim,)) for _ in self.layers]
        def lossfn(updated,ds):
            self.update(updated); logits,h,states,dyn,inputs=self.core(current,ds)
            byte=nn.losses.cross_entropy(logits[None,:],mx.array([target])).mean()
            latent=mx.mean(mx.square(unit(self.predictor(h))-unit(self.target(future))))
            variance=self.repr_loss(h)
            return byte+self.latent_weight*latent+self.variance_weight*variance,(byte,latent,variance,h,states,dyn,inputs)
        (loss,aux),(grads,dldstate)=mx.value_and_grad(lossfn,argnums=(0,1))(params,dummies)
        byte,latent,variance,h,states,dyn,inputs=aux
        rho,c,s,scale,gate,_=dyn[0]
        hist=mx.stack((rho,rho),axis=-1).reshape((self.dim,))[None,:]*Layer.rotate(self.encoder._embed_trace,c,s)
        grads["encoder"]["embed"]["weight"]=grads["encoder"]["embed"]["weight"]+hist*dldstate[0][None,:]
        direct=mx.stack((scale,scale),axis=-1).reshape((self.dim,))*gate*inputs[0]
        next_embed=hist+(mx.arange(VOCAB_SIZE)==current)[:,None]*direct
        traces=[]
        for i,layer in enumerate(self.layers):
            rho,c,s,scale,gate,rot=dyn[i]
            radial_history = dldstate[i]*layer.history(rho,c,s)
            grads["layers"][i]["decay"]=grads["layers"][i]["decay"]+mx.sum(radial_history.reshape((-1,2)),axis=1)
            traces.append(layer.next_trace(inputs[i],rho,c,s,scale,gate,rot))
        self.optimizer.update(self,grads)
        for layer,state,trace in zip(self.layers,states,traces):
            layer._state,layer._decay_trace=mx.stop_gradient(state),mx.stop_gradient(trace)
        self.encoder._embed_trace=mx.stop_gradient(next_embed)
        self.memory._matrix=mx.stop_gradient(self.memory.next_matrix(h))
        self._repr_buffer=mx.concatenate((self._repr_buffer,mx.stop_gradient(h)[None,:]),axis=0)[-self._repr_capacity:]
        self._target_embed=mx.stop_gradient(self.target_ema*self._target_embed+(1-self.target_ema)*self.encoder.embed.weight)
        mx.eval(self.parameters(),self.optimizer.state,self.encoder._embed_trace,self._target_embed,self._repr_buffer,self.memory._matrix,*[x._state for x in self.layers],*[x._decay_trace for x in self.layers])
        return {"loss":loss.item(),"byte":byte.item(),"latent":latent.item(),"variance":variance.item()}
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
            data[f"buffer.state.{i}"],data[f"buffer.decay_trace.{i}"]=x._state,x._decay_trace
        tmp=path+".temporary.safetensors"; mx.save_safetensors(tmp,data,metadata={"format":"tmt-v2"}); os.replace(tmp,path)
    def load(self,path):
        if not os.path.exists(path): return False
        data=mx.load(path)
        if "buffer.target_embed" not in data:
            raise ValueError("legacy TMT checkpoint is incompatible with TMT-v2; retrain instead of partially loading it")
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
        current=dict(util.tree_flatten(self.parameters()))
        good=[(k,v) for k,v in model.items() if k in current and current[k].shape==v.shape]
        if good: self.update(util.tree_unflatten(good))
        if len(good)==len(current) and optimizer: self.optimizer.state=util.tree_unflatten(list(optimizer.items()))
        return bool(good)

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
    Runtime(path="tmt-v2-4.5m.safetensors",dim=512,layers=16,temp=.75,lr=5e-4)()
