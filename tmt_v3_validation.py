"""Reproducible MLX validation for TMT-v3's mathematical contracts."""
import math
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.utils as util
from main import Model, Layer, EOS

OUT=Path("artifacts/tmt_v3_validation")
def norm(x): return mx.sqrt(mx.sum(mx.square(x))).item()

def injection_fd():
    layer=Layer(2,4,64)
    layer.write_scale=mx.array([.3,-.2]); layer.write_bias=mx.array([.1,-.15])
    x=mx.array([.7,-.4]); state=mx.zeros((2,))
    rho,c,s,scale,op=layer.dynamics(x); gate=mx.sigmoid(layer.write_scale*x+layer.write_bias)
    analytic=layer.injection_diagonal(x,scale,gate)
    eps=1e-4; finite=[]
    for i in range(2):
        plus=x+mx.array([eps if i==0 else 0,eps if i==1 else 0])
        minus=x-mx.array([eps if i==0 else 0,eps if i==1 else 0])
        _,sp,_=layer.step(plus,state); _,sm,_=layer.step(minus,state)
        finite.append(((sp[i]-sm[i])/(2*eps)).item())
    mx.eval(analytic)
    return max(abs(analytic[i].item()-finite[i]) for i in range(2))

def phase_fd():
    # The O(d) phase recurrence is checked against centered finite differences.
    d=.3; rho=1/(1+math.exp(-d)); phi=.2; x=(.4,-.7); q=(0.,0.); state=(0.,0.)
    def rollout(p):
        z=(0.,0.); w=math.pi/(1+math.exp(-p)); a=math.sqrt(1-rho*rho)
        for _ in range(32):
            z=(rho*(math.cos(w)*z[0]-math.sin(w)*z[1])+a*x[0],rho*(math.sin(w)*z[0]+math.cos(w)*z[1])+a*x[1])
        return z
    wp=math.pi*(1/(1+math.exp(-phi)))*(1-1/(1+math.exp(-phi))); w=math.pi/(1+math.exp(-phi)); a=math.sqrt(1-rho*rho)
    for _ in range(32):
        rs=(math.cos(w)*state[0]-math.sin(w)*state[1],math.sin(w)*state[0]+math.cos(w)*state[1])
        rq=(math.cos(w)*q[0]-math.sin(w)*q[1],math.sin(w)*q[0]+math.cos(w)*q[1])
        rj=(-rs[1],rs[0]); q=(rho*rq[0]+rho*wp*rj[0],rho*rq[1]+rho*wp*rj[1])
        state=(rho*rs[0]+a*x[0],rho*rs[1]+a*x[1])
    e=1e-5; plus=rollout(phi+e);minus=rollout(phi-e);fd=((plus[0]-minus[0])/(2*e),(plus[1]-minus[1])/(2*e))
    return max(abs(q[i]-fd[i]) for i in range(2))

def bptt_and_writer():
    m=Model(8,2,.7,1e-3,memory_dim=4,future_block=4); seq=[(1,2),(2,3),(3,4),(4,EOS)]
    def lossfn(p):
        m.update(p); states=[mx.zeros((m.dim,)) for _ in m.layers]; matrix=mx.zeros(m.memory._matrix.shape); loss=mx.array(0.)
        for cur,tgt in seq:
            logits,h,states,_,_=m.core(cur,states_override=states,matrix_override=matrix)
            loss=loss+nn.losses.cross_entropy(logits[None,:],mx.array([tgt])).mean()
            matrix=m.memory.next_matrix(h,matrix)
        return loss/len(seq)
    loss,grads=mx.value_and_grad(lossfn)(m.trainable_parameters())
    flat=dict(util.tree_flatten(grads)); mx.eval(loss,*flat.values())
    writer={k:norm(v) for k,v in flat.items() if str(k).startswith("memory.") and any(s in str(k) for s in ("key","value","forget","rate"))}
    return loss.item(),writer

def state_tracking():
    m=Model(32,1,.7,1e-3,memory_dim=4,min_half_life=4,max_half_life=65536)
    layer=m.layers[0]; x=mx.zeros((32,)); rho,c,s,_,_=layer.dynamics(x); mx.eval(rho,c,s)
    # cross-product positions 15*16 + 0/1 are persistent period-2 and period-3 modes
    values={}
    for label,index,period in (("parity",240,2),("mod3",241,3)):
        state=mx.zeros((32,)); state=state+((mx.arange(32)==2*index)[:,None] if False else mx.zeros((32,)))
        pair=mx.array([1.,0.]); rr=rho[index].item(); cc=c[index].item(); ss=s[index].item()
        z=pair
        for _ in range(period):z=mx.array([rr*(cc*z[0]-ss*z[1]),rr*(ss*z[0]+cc*z[1])])
        mx.eval(z); values[label]=math.sqrt((z[0].item()-rr**period)**2+z[1].item()**2)
    return values

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inj=injection_fd(); phase=phase_fd(); bptt,writer=bptt_and_writer(); state=state_tracking()
    m=Model(8,2,.7,1e-3,memory_dim=4); assert m.buffer_parameter_invariant()
    metrics=m.train_step(1,2,(2,3,4,5,EOS))
    text=f"""# TMT-v3 MLX Validation

- Buffer invariant: PASS
- Elementwise injection diagonal finite difference max error: {inj:.3e}
- Phase eligibility finite difference max error: {phase:.3e}
- Tiny actual-model BPTT loss: {bptt:.6f}
- Writer BPTT gradient norms: {writer}
- Persistent period-2/period-3 cycle residuals: {state}
- One real online train step: {metrics}

The tiny BPTT harness establishes exact differentiability through state and delta-memory updates. It is a baseline for future cosine/relative-error comparisons with the structured-local estimator; it does not mislabel that estimator as full RTRL.
"""
    (OUT/"REPORT.md").write_text(text)
    print(text)
if __name__=="__main__":main()
