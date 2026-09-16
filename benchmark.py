"""Frozen-representation CoLA evaluation for TMT-v2.

Train only the linear head on CoLA train; reset all recurrent/associative state
for every sentence; report the independent dev MCC.
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from main import Model

def cola(path):
    rows=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            p=line.rstrip("\n").split("\t")
            if len(p)==4: rows.append((p[3].encode("utf-8"),int(p[1])))
    return rows
def mcc(tp,tn,fp,fn):
    import math
    d=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn-fp*fn)/d if d else 0.
def representation(model,data):
    model.reset_memory()
    h=None
    for b in data:
        _,h,_,_,_=model.represent(b)
    return h if h is not None else mx.zeros((model.dim,))
def score(model,head,data):
    tp=tn=fp=fn=0
    for text,label in data:
        pred=mx.argmax(head(representation(model,text))).item()
        if pred==1 and label==1:tp+=1
        elif pred==0 and label==0:tn+=1
        elif pred==1:fp+=1
        else:fn+=1
    return mcc(tp,tn,fp,fn),(tp,tn,fp,fn)
def run():
    model=Model(dim=512,layers=16,temp=.75,lr=5e-4)
    model.load("tmt-v3.safetensors"); model.freeze()
    head=nn.Linear(model.dim,2); optimizer=opt.AdamW(learning_rate=1e-3)
    train=cola("CoLA/original/raw/in_domain_train.tsv")
    dev=cola("CoLA/original/raw/in_domain_dev.tsv")
    def lossfn(params,x,y):
        head.update(params); logits=head(x)
        return nn.losses.cross_entropy(logits[None,:],mx.array([y])).mean()
    for epoch in range(3):
        for text,label in train:
            x=representation(model,text)
            loss,grads=mx.value_and_grad(lossfn)(head.trainable_parameters(),x,label)
            optimizer.update(head,grads);mx.eval(head.parameters(),optimizer.state)
        result,counts=score(model,head,dev)
        print(f"epoch {epoch+1}: CoLA dev MCC={result:.4f}; TP,TN,FP,FN={counts}")
if __name__=="__main__":run()
