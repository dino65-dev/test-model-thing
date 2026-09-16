"""Held-out byte NLL / bits-per-byte evaluator for an explicitly supplied corpus."""
import argparse, math
from pathlib import Path
import mlx.core as mx
from main import Model, EOS

def bytes_of(path):
    return list(Path(path).read_bytes())+[EOS]
def bpb(model,data,limit=None):
    model.reset_memory();loss=0.;n=0
    for cur,tgt in zip(data,data[1:]):
        logits=model.frozen_step(cur);loss+=(mx.logsumexp(logits)-logits[tgt]).item();n+=1
        if limit and n>=limit:break
    return loss/(n*math.log(2)),n
def main():
    p=argparse.ArgumentParser();p.add_argument("checkpoint");p.add_argument("validation");p.add_argument("--dim",type=int,default=512);p.add_argument("--layers",type=int,default=16);p.add_argument("--memory-dim",type=int);p.add_argument("--limit",type=int);a=p.parse_args()
    m=Model(a.dim,a.layers,.75,5e-4,memory_dim=a.memory_dim);m.load(a.checkpoint);value,n=bpb(m,bytes_of(a.validation),a.limit)
    print({"validation_bpb":value,"bytes":n})
if __name__=="__main__":main()
