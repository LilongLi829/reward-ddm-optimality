"""Reproduce Bogacz cross-environment boundary statistics.

The input is the frozen participant-by-held-out-condition prediction table.
The script recomputes aggregate and delay-specific errors. The historical
generator of the archived q values is not claimed to be reconstructed.
"""
from pathlib import Path
import pandas as pd, numpy as np
R=Path(__file__).resolve().parents[1]; b=pd.read_csv(R/"data/BOGACZ_predictions.csv"); O=R/"results"; O.mkdir(exist_ok=True)
# Overall cross-environment prediction errors.
m0=b.err_m0.mean(); mq=b.err_mq.mean()
rows=[("N_participants",b.sid.nunique(),20,0),("N_targets",len(b),60,0),("MAE_M0",m0,.224371,1e-6),
("MAE_Mq",mq,.256210,1e-6),("relative_reduction_pct",(1-mq/m0)*100,-14.1904,1e-3),
("N_improved",(b.improvement>0).sum(),35,0),("mean_improvement",b.improvement.mean(),-.031839,1e-6)]
# Delay-specific checks.
for d,a,c,w,q in [(.5,.265164,.254571,15,.108048),(1,.235086,.222654,13,.108049),(2,.172863,.291405,7,.440641)]:
 x=b[np.isclose(b.D,d)]; rows += [(f"D{d}_q",x.q_train.iloc[0],q,1e-6),(f"D{d}_MAE_M0",x.err_m0.mean(),a,1e-6),(f"D{d}_MAE_Mq",x.err_mq.mean(),c,1e-6),(f"D{d}_improved",(x.improvement>0).sum(),w,0)]
out=pd.DataFrame(rows,columns=["metric","recomputed","expected","tolerance"]); out["PASS"]=abs(out.recomputed-out.expected)<=out.tolerance
out.to_csv(O/"checks.csv",index=False); print(out.to_string(index=False)); print(f"\nBogacz: {out.PASS.sum()}/{len(out)} PASS")
raise SystemExit(0 if out.PASS.all() else 1)
