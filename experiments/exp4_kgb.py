"""Variant D: kurtogram-style band selection (max kurtosis of band envelope over a dyadic-like grid)
   Variant E: kinematics-guided band selection (max defect-frequency envelope SNR over the grid)."""
import json, numpy as np
from scipy import signal
from dsp import *
from exp1_diag import records, windows
from sklearn.metrics import balanced_accuracy_score, f1_score
LABS=["Normal","IR","OR","Ball"]
BANDS=[(fc-bw/2,fc+bw/2) for bw in (800,1600,3200) for fc in np.arange(bw/2+200, 5600-bw/2+1, bw/4)]
def env(x,lo,hi):
    sos=signal.butter(4,[lo,hi],btype="bandpass",fs=FS,output="sos"); y=signal.sosfiltfilt(sos,x-x.mean())
    e=np.abs(signal.hilbert(y)); return e
rows=[]
for r in records():
    ws,rpm=windows(r); fr=rpm/60
    for w in ws:
        best_k=(-1,None); best_s=(-1e9,None)
        for lo,hi in BANDS:
            e=env(w,lo,hi); k=np.mean((e-e.mean())**4)/np.mean((e-e.mean())**2)**2
            n=len(e); E=np.abs(np.fft.rfft((e-e.mean())*np.hanning(n)))/n; f=np.fft.rfftfreq(n,1/FS)
            sc=fault_scores(f,E,fr)
            if k>best_k[0]: best_k=(k,sc)
            if max(sc.values())>best_s[0]: best_s=(max(sc.values()),sc)
        rows.append(dict(cls=r["cls"],load=r["load"],size=r["size"],D=best_k[1],E=best_s[1]))
def rule(sc,thr):
    b=max(sc,key=sc.get); return "Normal" if sc[b]<thr else {"BPFO":"OR","BPFI":"IR","BSF":"Ball"}[b]
out={}
tr=[d for d in rows if d["load"]==0]; te=[d for d in rows if d["load"]!=0]; yt=[d["cls"] for d in te]
for key in "DE":
    thr=max(np.arange(4,25.01,0.5),key=lambda t:balanced_accuracy_score([d["cls"] for d in tr],[rule(d[key],t) for d in tr]))
    p=[rule(d[key],thr) for d in te]
    tf=np.array([t!="Normal" for t in yt]); pf=np.array([x!="Normal" for x in p])
    out[key]=dict(thr=float(thr),acc=float(np.mean(np.array(p)==np.array(yt))),bacc=float(balanced_accuracy_score(yt,p)),
      f1=float(f1_score(yt,p,labels=LABS,average="macro",zero_division=0)),
      acc_noball=float(np.mean([a==b for a,b in zip(p,yt) if b!="Ball"])),
      tpr=float((tf&pf).sum()/tf.sum()),fpr=float((~tf&pf).sum()/(~tf).sum()))
    print(key,out[key])
json.dump(out,open("results/bandsel.json","w"),indent=1)
