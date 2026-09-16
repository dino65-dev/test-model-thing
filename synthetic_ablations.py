"""Synthetic recurrent-state ablations for parity, modulo, delayed XOR, and copy."""
import argparse, random
import mlx.core as mx
from main import Model

def sequence(task,length,delay,rng):
    bits=[rng.randrange(2) for _ in range(length)]
    if task=="parity":
        state=0;targets=[]
        for b in bits: state^=b;targets.append(state)
    elif task.startswith("mod"):
        base=int(task[3:]);state=0;targets=[]
        for b in bits: state=(state+b)%base;targets.append(state)
    elif task=="xor":
        targets=[0 if i<delay else bits[i]^bits[i-delay] for i in range(length)]
    elif task=="copy":
        targets=[0 if i<delay else bits[i-delay] for i in range(length)]
    else: raise ValueError(task)
    return bits,targets
def configure(m,complex_on,fast_memory_on):
    if not complex_on:
        for layer in m.layers: layer.phase=mx.zeros(layer.phase.shape);layer.phase_scale=mx.zeros(layer.phase_scale.shape);layer.phase_bias=mx.zeros(layer.phase_bias.shape)
    if not fast_memory_on: m.memory.out.weight=mx.zeros(m.memory.out.weight.shape)
def score(m,task,length,delay,episodes,seed):
    rng=random.Random(seed);right=total=0
    for _ in range(episodes):
        m.reset_memory();bits,targets=sequence(task,length,delay,rng)
        for b,y in zip(bits,targets):
            pred=mx.argmax(m.frozen_step(b)).item();right+=pred==y;total+=1
    return right/total
def run(task,steps,length,delay,complex_on,fast_memory_on,seed):
    rng=random.Random(seed);mx.random.seed(seed);m=Model(16,2,.7,2e-3,memory_dim=8,future_block=4);configure(m,complex_on,fast_memory_on)
    for _ in range(steps):
        m.reset_memory();bits,targets=sequence(task,length,delay,rng)
        for b,y in zip(bits,targets):m.train_step(b,y,(y,))
    return score(m,task,length,delay,64,seed+1)
def main():
    p=argparse.ArgumentParser();p.add_argument("--steps",type=int,default=64);p.add_argument("--length",type=int,default=16);p.add_argument("--delay",type=int,default=4);p.add_argument("--seed",type=int,default=7);a=p.parse_args()
    tasks=["parity","mod3","mod5","xor","copy"]
    for task in tasks:
        full=run(task,a.steps,a.length,a.delay,True,True,a.seed)
        no_phase=run(task,a.steps,a.length,a.delay,False,True,a.seed)
        no_memory=run(task,a.steps,a.length,a.delay,True,False,a.seed)
        print({"task":task,"complex_fast":full,"no_complex":no_phase,"no_fast_memory":no_memory})
if __name__=="__main__":main()
