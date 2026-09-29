"""Reproduce Evans specificity-analysis statistics.

Reproduction begins from recovered posterior-derived participant inputs.
The script compares observed late thresholds with M0 and same-sample Mq
predictions; it does not rerun the historical multi-chain posterior fit.
"""
from pathlib import Path
import pandas as pd, numpy as np
R=Path(__file__).resolve().parents[1]; e=pd.read_csv(R/"data/EVANS_subjects.csv"); O=R/"results"; O.mkdir(exist_ok=True)
# Compare participant-level absolute prediction errors.
m0=np.abs(e.a_observed-e.a_opt_q0); mq=np.abs(e.a_observed-e.a_opt_qrefined)
rows=[("N",len(e),79,0),("observed_mean_a",e.a_observed.mean(),1.518414783,1e-6),
("M0_subject_MAE",m0.mean(),.184699,1e-6),("Mq_subject_MAE",mq.mean(),.179224,1e-6),
("relative_reduction_pct",(1-mq.mean()/m0.mean())*100,2.964,1e-2),("wins",(mq<m0).sum(),42,0)]
out=pd.DataFrame(rows,columns=["metric","recomputed","expected","tolerance"]); out["PASS"]=abs(out.recomputed-out.expected)<=out.tolerance
out.to_csv(O/"checks.csv",index=False); print(out.to_string(index=False)); print(f"\nEvans: {out.PASS.sum()}/{len(out)} PASS")
raise SystemExit(0 if out.PASS.all() else 1)
