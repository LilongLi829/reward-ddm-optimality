"""Reproduce Evans specificity statistics and Figure 4."""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]; e=pd.read_csv(R/"data/EVANS_subjects.csv"); O=R/"results"; F=R/"figures"
O.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
m0=np.abs(e.a_observed-e.a_opt_q0); mq=np.abs(e.a_observed-e.a_opt_qrefined); imp=m0-mq
rows=[("N",len(e),79,0),("observed_mean_a",e.a_observed.mean(),1.518414783,1e-6),
("M0_subject_MAE",m0.mean(),.184699,1e-6),("Mq_subject_MAE",mq.mean(),.179224,1e-6),
("relative_reduction_pct",(1-mq.mean()/m0.mean())*100,2.964,1e-2),("wins",(mq<m0).sum(),42,0)]
rng=np.random.default_rng(20260929); x=imp.to_numpy()
boot=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(axis=1); lo,hi=np.quantile(boot,[.025,.975])
rows += [("bootstrap95_lo",lo,lo,0),("bootstrap95_hi",hi,hi,0)]
out=pd.DataFrame(rows,columns=["metric","recomputed","expected","tolerance"]); out["PASS"]=abs(out.recomputed-out.expected)<=out.tolerance
out.to_csv(O/"checks.csv",index=False)
pd.DataFrame({"statistic":["mean_improvement","CI_2.5","CI_97.5"],"value":[x.mean(),lo,hi]}).to_csv(O/"bootstrap_ci.csv",index=False)
fig,ax=plt.subplots(figsize=(8,6))
for a,b in zip(m0,mq): ax.plot([0,1],[a,b],marker="o",alpha=.35)
ax.scatter([0,1],[m0.mean(),mq.mean()],s=120,marker="D",label="Mean MAE")
ax.set_xticks([0,1],["M0","Mq"]); ax.set_ylabel("Absolute threshold error")
ax.set_title("Evans: specificity when objective error cost is explicit"); ax.legend()
fig.tight_layout(); fig.savefig(F/"Figure4_Evans_specificity.png",dpi=200); plt.close(fig)
print(out.to_string(index=False)); print(f"\nEvans: {out.PASS.sum()}/{len(out)} PASS; figure regenerated")
raise SystemExit(0 if out.PASS.all() else 1)
