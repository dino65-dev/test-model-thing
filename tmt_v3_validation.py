"""Strict MLX regression tests for TMT-v3 recurrent mathematics."""
import math, os, tempfile
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.utils as util
from main import Model, Layer, EOS

OUT=Path("artifacts/tmt_v3_validation")
def finite(x): return math.isfinite(float(x))
def nrm(x): return mx.sqrt(mx.sum(mx.square(x))).item()

def pair_block_fd(seed):
    mx.random.seed(seed)
    layer=Layer(2,4,128)
    layer.write_scale=mx.array([.31,-.27]);layer.write_bias=mx.array([.1,-.2])
    layer.phase_scale=mx.array([.43]);layer.phase_bias=mx.array([-.17]);layer.phase=mx.array([1.1])
    x=mx.array([.7,-.4]); state=mx.array([.6,-.9]); rho,c,s,scale,z,px=layer.dynamics(x);gate=mx.sigmoid(layer.write_scale*x+layer.write_bias)
    analytic=layer.embedding_block(x,state,rho,c,s,scale,gate,z)
    eps=2e-4;columns=[]
    for j in range(2):
        delta=mx.array([eps if j==0 else 0,eps if j==1 else 0])
        _,plus,_=layer.step(x+delta,state);_,minus,_=layer.step(x-delta,state)
        columns.append((plus-minus)/(2*eps))
    numeric=mx.stack(columns,axis=-1);mx.eval(analytic,numeric)
    return mx.max(mx.abs(analytic-numeric)).item()

def phase_fd(seed):
    mx.random.seed(seed); layer=Layer(2,4,128)
    layer.phase=mx.array([1.2]);layer.phase_scale=mx.array([.33]);layer.phase_bias=mx.array([-.1])
    x=mx.array([.2,.6]); state=mx.array([.5,-.7]); q=mx.zeros((2,))
    for _ in range(25):
        rho,c,s,scale,z,px=layer.dynamics(x)
        q=layer.history(q,rho,c,s)+layer.phase_direct(rho,c,s,state,mx.ones((1,)))
        _,state,_=layer.step(x,state)
    e=1e-4
    def rollout(p):
        layer.phase=mx.array([p]);z=mx.array([.5,-.7])
        for _ in range(25):_,z,_=layer.step(x,z)
        return z
    plus,minus=rollout(1.2+e),rollout(1.2-e);numeric=(plus-minus)/(2*e);mx.eval(q,numeric)
    return mx.max(mx.abs(q-numeric)).item()

def bptt(seed):
    mx.random.seed(seed);m=Model(8,2,.7,1e-3,memory_dim=4,future_block=4);seq=[(1,2),(2,3),(3,4),(4,EOS)]
    def lossfn(params):
        m.update(params);states=[mx.zeros((m.dim,)) for _ in m.layers];matrix=mx.zeros(m.memory._matrix.shape);loss=mx.array(0.)
        for cur,tgt in seq:
            logits,h,states,_,_=m.core(cur,states_override=states,matrix_override=matrix)
            loss=loss+nn.losses.cross_entropy(logits[None,:],mx.array([tgt])).mean();matrix=m.memory.next_matrix(h,matrix)
        return loss/len(seq)
    loss,grads=mx.value_and_grad(lossfn)(m.trainable_parameters());flat=dict(util.tree_flatten(grads));mx.eval(loss,*flat.values())
    writer={str(k):nrm(v) for k,v in flat.items() if str(k).startswith("memory.") and any(x in str(k) for x in ("key","value","forget","rate"))}
    assert finite(loss.item()) and all(v>1e-10 for v in writer.values())
    return loss.item(),writer

def cycles():
    m=Model(32,1,.7,1e-3,memory_dim=4);layer=m.layers[0];rho,c,s,_,_,_=layer.dynamics(mx.zeros((32,)));mx.eval(rho,c,s)
    results={}
    for label,target,period in (("period2",math.pi,2),("period3",2*math.pi/3,3)):
        candidates=[i for i in range(16) if abs(layer.phase[i].item()-target)<1e-5];assert candidates
        i=max(candidates,key=lambda j:rho[j].item());theta=layer.phase[i].item()
        # angle closure after 1, 10, 100, 1000, 10000 cycles; rho is intentionally factored out.
        residual=max(abs(math.cos(period*theta*n)-1)+abs(math.sin(period*theta*n)) for n in (1,10,100,1000,10000))
        results[label]=residual
        assert residual<2e-3
    return results

def checkpoints():
    m=Model(8,2,.7,1e-3,memory_dim=4);m.train_step(1,2,(2,3,4,EOS))
    with tempfile.TemporaryDirectory() as d:
        good=os.path.join(d,"good.safetensors");m.save(good);assert Model(8,2,.7,1e-3,memory_dim=4).load(good)
        bad=os.path.join(d,"bad.safetensors");mx.save_safetensors(bad,{"m.encoder.embed.weight":m.encoder.embed.weight,"buffer.target_embed":m._target_embed})
        try: Model(8,2,.7,1e-3,memory_dim=4).load(bad)
        except ValueError:return "roundtrip and legacy rejection PASS"
        raise AssertionError("partial legacy checkpoint was accepted")

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    injection=[pair_block_fd(s) for s in (1,2,3)];phase=[phase_fd(s) for s in (1,2,3)]
    assert max(injection)<3e-3, injection;assert max(phase)<3e-3,phase
    loss,writer=bptt(7);cycle=cycles();check=checkpoints()
    m=Model(512,16,.75,5e-4);assert m.buffer_parameter_invariant();count=sum(v.size for _,v in util.tree_flatten(m.parameters()))
    online=m.train_step(65,66,tuple(range(67,82))+(EOS,));assert all(finite(v) for v in online.values()) and count==5952259
    report=f"""# TMT-v3 Strict MLX Validation
- Pair-block embedding Jacobian FD errors over three seeds: {injection}
- Base phase eligibility FD errors over three seeds: {phase}
- Exact tiny BPTT loss: {loss:.6f}
- BPTT memory-controller gradient norms: {writer}
- Raw-angle cycle closure: {cycle}
- Checkpoints: {check}
- Exact parameter count: {count}
- 512x16 online metrics: {online}
All assertions passed.
"""
    (OUT/"REPORT.md").write_text(report);print(report)
if __name__=="__main__":main()
