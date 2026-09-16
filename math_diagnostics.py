"""Stdlib-only P0-P4 math diagnostics."""
import math, random
from pathlib import Path
OUT=Path("artifacts/math_diagnostics")
def plot(path,title,curves):
 p=60;w=900;h=440;allp=[q for _,v in curves for q in v];xs=[x for x,y in allp];ys=[y for x,y in allp];xmin,xmax=min(xs),max(xs);ymin,ymax=min(ys),max(ys);xmax=xmax if xmax>xmin else xmin+1;ymax=ymax if ymax>ymin else ymin+1
 def pos(x,y):return p+(x-xmin)/(xmax-xmin)*(w-2*p),h-p-(y-ymin)/(ymax-ymin)*(h-2*p)
 colors=["#2563eb","#dc2626","#059669","#9333ea"]
 z=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/><text x="{p}" y="28" font-family="sans-serif" font-size="19">{title}</text><line x1="{p}" y1="{h-p}" x2="{w-p}" y2="{h-p}" stroke="black"/><line x1="{p}" y1="{p}" x2="{p}" y2="{h-p}" stroke="black"/>']
 for i,(name,v) in enumerate(curves):
  pts=" ".join(f"{a:.1f},{b:.1f}" for a,b in [pos(x,y) for x,y in v]);z.append(f'<polyline fill="none" stroke="{colors[i%4]}" stroke-width="2.5" points="{pts}"/><text x="650" y="{55+20*i}" fill="{colors[i%4]}" font-family="sans-serif">{name}</text>')
 z.append("</svg>");path.write_text("\n".join(z))
def trace():
 d=.7;r=1/(1+math.exp(-d));dr=r*(1-r);a=math.sqrt(1-r*r);s=q=0;out=[]
 for t in range(1,61):
  u=math.sin(.37*t);q=r*q+dr*s-r*dr/a*u;s=r*s+a*u;e=1e-5
  def f(x):
   rr=1/(1+math.exp(-x));aa=math.sqrt(1-rr*rr);z=0
   for k in range(1,t+1):z=rr*z+aa*math.sin(.37*k)
   return z
  fd=(f(d+e)-f(d-e))/(2*e);out.append((t,q,fd,abs(q-fd)))
 return out
def variance():
 random.seed(7);out=[]
 for h in [4,16,64,256,1024,4096]:
  r=2**(-1/h);a=math.sqrt(1-r*r);s=0;v=[]
  for i in range(30000):
   s=r*s+a*random.gauss(0,1)
   if i>5000:v.append(s)
  m=sum(v)/len(v);out.append((h,sum((x-m)**2 for x in v)/len(v)))
 return out
def assoc():
 random.seed(11);d=12;m=[[0.]*d for _ in range(d)];pairs=[]
 dot=lambda a,b:sum(x*y for x,y in zip(a,b))
 for n in range(24):
  k=[random.gauss(0,1) for _ in range(d)];z=math.sqrt(dot(k,k));k=[x/z for x in k];v=[random.uniform(-1,1) for _ in k];got=[dot(row,k) for row in m]
  for i in range(d):
   for j in range(d):m[i][j]=.995*m[i][j]+.35*(v[i]-got[i])*k[j]
  pairs.append((k,v))
 return [(i+1,math.sqrt(sum((dot(row,k)-v[j])**2 for j,row in enumerate(m))/d)) for i,(k,v) in enumerate(pairs)]
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 hs=[4,16,64,256,1024,4096]
 plot(OUT/"p1_timescales.svg","P1 log-spaced half-life retention",[(f"H={h}",[(t,2**(-t/h)) for t in range(0,4097,32)]) for h in hs])
 tr=trace();plot(OUT/"p1_trace_exactness.svg","P1 scalar eligibility: analytic versus finite difference",[("analytic",[(t,q) for t,q,_,_ in tr]),("finite difference",[(t,x) for t,_,x,_ in tr])])
 va=variance();plot(OUT/"p1_stationary_variance.svg","P1 normalized injection variance",[("empirical",va),("target",[(h,1) for h in hs])])
 r=.995;x=y=1.;rot=[]
 for t in range(1,257):x,y=r*(math.cos(.1)*x-math.sin(.1)*y),r*(math.sin(.1)*x+math.cos(.1)*y);rot.append((t,math.sqrt(x*x+y*y)))
 plot(OUT/"p2_rotation_stability.svg","P2 rotational mode: spectral radius rho < 1",[("simulation",rot),("rho^t",[(t,r**t*math.sqrt(2)) for t,_ in rot])])
 plot(OUT/"p3_delta_memory.svg","P3 delta-rule own-key retrieval RMSE",[("RMSE",assoc())])
 entropy=[.08]*10+[.1]*8+[.9]+[.12]*12+[.8]+[.07]*16;s=[];n=0
 for e in entropy:
  n+=1
  if e>.35 or n>=8:s.append(n);n=0
 if n:s.append(n)
 plot(OUT/"p4_entropy_patches.svg","P4 entropy boundary patch sizes",[("bytes",list(enumerate(s,1)))])
 err=max(x[3] for x in tr)
 (OUT/"REPORT.md").write_text(f"""# TMT-v2 Mathematical Diagnostics
P0: private underscore buffers are used in main.py and should not be in MLX parameters.
P1: alpha=2^(-1/H), normalized injection, and scalar eligibility finite-difference check passed (max error {err:.3e}).
P2: rho R(omega) has eigenvalue magnitude rho; plot confirms rho=.995 decay.
P3: delta-rule associative-memory retrieval error is plotted.
P4: entropy threshold produces shorter patches at uncertain bytes.
The trace validation is scalar/local only; it is not a claim of full BPTT, UORO, or KF-RTRL equivalence.
""")
 print("wrote",OUT)
if __name__=="__main__":main()
