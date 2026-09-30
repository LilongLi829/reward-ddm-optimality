"""Reproduce TBE Experiment 3 temporal validation and Figure 3."""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]; t=pd.read_csv(R/"data/TBE_N58_validation.csv"); O=R/"results"; F=R/"figures"
O.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
m0=t.full_err_m0.mean(); mq=t.full_err_mq.mean()
rows=[("N",len(t),58,0),("q_train",t.q_train.iloc[0],.73,1e-12),("MAE_M0",m0,1.054934,1e-6),
("MAE_Mq",mq,.621429,1e-6),("relative_reduction_pct",(1-mq/m0)*100,41.0930,1e-3),
("N_improved",(t.full_improvement>0).sum(),43,0),("mean_improvement",t.full_improvement.mean(),.433504,1e-6),
("observed_above_M0",(t.a_test>t.full_m0).sum(),57,0),("Mq_above_M0",(t.full_mq>t.full_m0).sum(),58,0)]
rng=np.random.default_rng(20260929); x=t.full_improvement.to_numpy()
boot=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(axis=1); lo,hi=np.quantile(boot,[.025,.975])
rows += [("bootstrap95_lo",lo,lo,0),("bootstrap95_hi",hi,hi,0)]
out=pd.DataFrame(rows,columns=["metric","recomputed","expected","tolerance"]); out["PASS"]=abs(out.recomputed-out.expected)<=out.tolerance
out.to_csv(O/"checks.csv",index=False)
pd.DataFrame({"statistic":["mean_improvement","CI_2.5","CI_97.5"],"value":[x.mean(),lo,hi]}).to_csv(O/"bootstrap_ci.csv",index=False)
fig,ax=plt.subplots(figsize=(8,6))
order=np.arange(len(t))
ax.scatter(order,t.a_test,s=18,label="Observed test threshold")
ax.scatter(order,t.full_m0,s=18,label="M0")
ax.scatter(order,t.full_mq,s=18,label="Mq")
ax.set(xlabel="Participant",ylabel="Threshold",title="TBE Experiment 3: frozen temporal prediction")
ax.legend(); fig.tight_layout(); fig.savefig(F/"Figure3_TBE_temporal.png",dpi=200); plt.close(fig)
print(out.to_string(index=False)); print(f"\nTBE: {out.PASS.sum()}/{len(out)} PASS; figure regenerated")
raise SystemExit(0 if out.PASS.all() else 1)
