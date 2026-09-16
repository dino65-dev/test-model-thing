"""Actual frozen-parameter structured-online-gradient versus exact-BPTT sweep."""
import csv, math
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.utils as util
from main import Model

OUT=Path("artifacts/gradient_fidelity")
class Capture:
    def __init__(self): self.state={};self.grads=None
    def update(self,module,grads): self.grads=grads

def flatvec(tree):
    values=[v.reshape((-1,)) for _,v in util.tree_flatten(tree)]
    return mx.concatenate(values) if values else mx.zeros((0,))
def metrics(a,b):
    mx.eval(a,b);na=mx.sqrt(mx.sum(a*a)).item();nb=mx.sqrt(mx.sum(b*b)).item()
    return (mx.sum(a*b).item()/(na*nb+1e-12),mx.sqrt(mx.sum((a-b)*(a-b))).item()/(na+1e-12),nb/(na+1e-12))
def exact(model,seq):
    model.reset_memory();params=model.trainable_parameters()
    def f(p):
        model.update(p);states=[mx.zeros((model.dim,)) for _ in model.layers];matrix=mx.zeros(model.memory._matrix.shape);loss=mx.array(0.)
        for current,target in seq:
            logits,h,states,_,_=model.core(current,states_override=states,matrix_override=matrix)
            loss=loss+nn.losses.cross_entropy(logits[None,:],mx.array([target])).mean();matrix=model.memory.next_matrix(h,matrix)
        return loss/len(seq)
    return mx.value_and_grad(f)(params)[1]
def online(model,seq,trace):
    model.reset_memory();old=model.optimizer;cap=Capture();model.optimizer=cap
    oldweights=(model.latent_weight,model.variance_weight,model.writer_weight);model.latent_weight=model.variance_weight=model.writer_weight=0.
    total={}
    try:
        for current,target in seq:
            if not trace:
                for layer in model.layers:
                    layer._decay_trace=mx.zeros((model.dim,));layer._phase_trace=mx.zeros((model.dim,))
                    layer._write_scale_trace=mx.zeros((model.dim,));layer._write_bias_trace=mx.zeros((model.dim,))
                    layer._phase_scale_trace=mx.zeros((model.dim,));layer._phase_bias_trace=mx.zeros((model.dim,))
                model.encoder._embed_trace=mx.zeros(model.encoder._embed_trace.shape)
            model.train_step(current,target,(target,))
            for k,v in util.tree_flatten(cap.grads): total[k]=total.get(k,mx.zeros(v.shape))+v
    finally:
        model.optimizer=old;model.latent_weight,model.variance_weight,model.writer_weight=oldweights
    return util.tree_unflatten([(k,v/len(seq)) for k,v in total.items()])
def groups(tree):
    out={}
    for k,v in util.tree_flatten(tree):
        s=str(k);name="embedding" if s.startswith("encoder") else "decay" if s.endswith(".decay") else "phase" if s.endswith(".phase") else "write" if "write_" in s else "phase_controller" if "phase_" in s else "memory" if s.startswith("memory") else "other"
        out.setdefault(name,[]).append(v.reshape((-1,)))
    return {k:mx.concatenate(v) for k,v in out.items()}
def svg(rows):
    w,h,p=760,400,50;xs=[r["T"] for r in rows];ys=[r["trace_cos"] for r in rows]+[r["instant_cos"] for r in rows]
    def q(x,y):return p+(x-min(xs))/(max(xs)-min(xs))*(w-2*p),h-p-(y-min(ys))/(max(ys)-min(ys)+1e-9)*(h-2*p)
    a=" ".join(f"{x:.1f},{y:.1f}" for x,y in [q(r["T"],r["trace_cos"]) for r in rows]);b=" ".join(f"{x:.1f},{y:.1f}" for x,y in [q(r["T"],r["instant_cos"]) for r in rows])
    (OUT/"cosine.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/><text x="50" y="28">TMT gradient fidelity</text><polyline fill="none" stroke="#2563eb" stroke-width="3" points="{a}"/><polyline fill="none" stroke="#dc2626" stroke-width="3" points="{b}"/><text x="560" y="55" fill="#2563eb">structured trace</text><text x="560" y="75" fill="#dc2626">instantaneous</text></svg>')
def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];details=[]
    for T in (2,4,8,16,32,64):
        mx.random.seed(19);m=Model(8,2,.7,1e-3,memory_dim=4);seq=[(i%8,(i+1)%8) for i in range(T)]
        e=exact(m,seq);tr=online(m,seq,True);ins=online(m,seq,False);ec,tc,ic=flatvec(e),flatvec(tr),flatvec(ins)
        c,r,n=metrics(ec,tc);ci,ri,ni=metrics(ec,ic);rows.append({"T":T,"trace_cos":c,"trace_rel":r,"trace_norm":n,"instant_cos":ci,"instant_rel":ri,"instant_norm":ni})
        ge,gt=groups(e),groups(tr)
        details.extend({"T":T,"family":name,"cos":metrics(ge[name],gt[name])[0],"rel":metrics(ge[name],gt[name])[1],"norm":metrics(ge[name],gt[name])[2]} for name in ge if name in gt)
    for row in rows:
        assert all(math.isfinite(float(v)) for v in row.values())
    with (OUT/"summary.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    with (OUT/"families.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=details[0]);w.writeheader();w.writerows(details)
    svg(rows);(OUT/"REPORT.md").write_text("# Actual gradient fidelity\n\n"+"\n".join(str(r) for r in rows)+"\n\nPer-family values: families.csv. These are measured approximation metrics, not claims of full RTRL.\n")
    print(rows)
if __name__=="__main__":main()
