"""Reproduce TBE Experiment 3 temporal-validation statistics.

The frozen table contains N=58 participant predictions. q was estimated on
Blocks 2-10 and then held fixed for the Blocks 11-20 test period. This script
recomputes participant-level absolute errors and directional counts.
"""
from pathlib import Path
import pandas as pd, numpy as np
R=Path(__file__).resolve().parents[1]; t=pd.read_csv(R/"data/TBE_N58_validation.csv"); O=R/"results"; O.mkdir(exist_ok=True)
# Participant-level held-out absolute errors.
m0=t.full_err_m0.mean(); mq=t.full_err_mq.mean()
rows=[("N",len(t),58,0),("q_train",t.q_train.iloc[0],.73,1e-12),("MAE_M0",m0,1.054934,1e-6),
("MAE_Mq",mq,.621429,1e-6),("relative_reduction_pct",(1-mq/m0)*100,41.0930,1e-3),
("N_improved",(t.full_improvement>0).sum(),43,0),("mean_improvement",t.full_improvement.mean(),.433504,1e-6),
("observed_above_M0",(t.a_test>t.full_m0).sum(),57,0),("Mq_above_M0",(t.full_mq>t.full_m0).sum(),58,0)]
out=pd.DataFrame(rows,columns=["metric","recomputed","expected","tolerance"]); out["PASS"]=abs(out.recomputed-out.expected)<=out.tolerance
out.to_csv(O/"checks.csv",index=False); print(out.to_string(index=False)); print(f"\nTBE: {out.PASS.sum()}/{len(out)} PASS")
raise SystemExit(0 if out.PASS.all() else 1)
