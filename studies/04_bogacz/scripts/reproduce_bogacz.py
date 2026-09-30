"""Reproduce Bogacz boundary statistics and Figure 5."""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]; b=pd.read_csv(R/"data/BOGACZ_predictions.csv"); O=R/"results"; F=R/"figures"
O.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
m0=b.err_m0.mean(); mq=b.err_mq.mean()
rows=[("N_participants",b.sid.nunique(),20,0),("N_targets",len(b),60,0),("MAE_M0",m0,.224371,1e-6),
("MAE_Mq",mq,.256210,1e-6),("relative_reduction_pct",(1-mq/m0)*100,-14.1904,1e-3),
("N_improved",(b.improvement>0).sum(),35,0),("mean_improvement",b.improvement.mean(),-.031839,1e-6)]
for d,a,c,w,q in [(.5,.265164,.254571,15,.108048),(1,.235086,.222654,13,.108049),(2,.172863,.291405,7,.440641)]:
 x=b[np.isclose(b.D,d)]
 rows += [(f"D{d}_q",x.q_train.iloc[0],q,1e-6),(f"D{d}_MAE_M0",x.err_m0.mean(),a,1e-6),(f"D{d}_MAE_Mq",x.err_mq.mean(),c,1e-6),(f"D{d}_improved",(x.improvement>0).sum(),w,0)]
# Participant-cluster bootstrap: resample participants, retaining their three condition rows.
p=b.groupby("sid",sort=True).improvement.mean().to_numpy()
rng=np.random.default_rng(20260929); boot=p[rng.integers(0,len(p),size=(10000,len(p)))].mean(axis=1); lo,hi=np.quantile(boot,[.025,.975])
rows += [("cluster_bootstrap95_lo",lo,lo,0),("cluster_bootstrap95_hi",hi,hi,0)]
out=pd.DataFrame(rows,columns=["metric","recomputed","expected","tolerance"]); out["PASS"]=abs(out.recomputed-out.expected)<=out.tolerance
out.to_csv(O/"checks.csv",index=False)
pd.DataFrame({"statistic":["mean_improvement","CI_2.5","CI_97.5"],"value":[p.mean(),lo,hi]}).to_csv(O/"bootstrap_ci.csv",index=False)
ds=[.5,1,2]; mm0=[]; mmq=[]
for d in ds:
 x=b[np.isclose(b.D,d)]; mm0.append(x.err_m0.mean()); mmq.append(x.err_mq.mean())
xpos=np.arange(3); width=.36
fig,ax=plt.subplots(figsize=(8,6))
ax.bar(xpos-width/2,mm0,width,label="M0"); ax.bar(xpos+width/2,mmq,width,label="Mq")
ax.set_xticks(xpos,[str(d) for d in ds]); ax.set(xlabel="Delay D (s)",ylabel="Mean absolute error",title="Bogacz: cross-environment boundary")
ax.legend(); fig.tight_layout(); fig.savefig(F/"Figure5_Bogacz_boundary.png",dpi=200); plt.close(fig)
print(out.to_string(index=False)); print(f"\nBogacz: {out.PASS.sum()}/{len(out)} PASS; figure regenerated")
raise SystemExit(0 if out.PASS.all() else 1)
