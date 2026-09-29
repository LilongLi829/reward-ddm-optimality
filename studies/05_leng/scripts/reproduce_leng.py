"""Check Leng DDM-effect, value-mapping, and forward-prediction results.

The included tables are archived outputs from the analysis pipeline. The
script checks the reported DDM effects and confidence limits and the held-out
mapping/threshold prediction errors used in the manuscript.
"""
from pathlib import Path
import pandas as pd, numpy as np
R=Path(__file__).resolve().parents[1]; O=R/"results"; O.mkdir(exist_ok=True)
le=pd.read_csv(R/"data/LENG_DDM_effects.csv"); rows=[]
# Archived DDM incentive-effect targets and bootstrap limits.
targets={("Drift rate v","Reward"):(.109869,.044327,.175386),("Threshold a","Reward"):(-.048249,-.098626,.004626),
("Drift rate v","Penalty"):(.004052,-.071715,.080112),("Threshold a","Penalty"):(.074910,.007321,.139704)}
for key,(mu,lo,hi) in targets.items():
 x=le[(le.parameter==key[0])&(le.incentive==key[1])].iloc[0]
 rows += [(f"{key[1]}_{key[0]}_mean",x["mean"],mu,1e-6),(f"{key[1]}_{key[0]}_CIlo",x.boot_lo,lo,1e-6),(f"{key[1]}_{key[0]}_CIhi",x.boot_hi,hi,1e-6)]
# Held-out value-mapping errors.
lm=pd.read_csv(R/"data/LENG_mapping.csv"); lm=lm[lm.analysis=="Primary_all_participants"].set_index("model")
for m,e in [("Subjective",.326167),("Objective_proportional",.543403),("Stable",.326961)]: rows.append((f"mapping_{m}",lm.loc[m,"MAE_RP_equalweight"],e,1e-6))
# Forward threshold-prediction errors.
lf=pd.read_csv(R/"data/LENG_forward.csv"); lf=lf[lf.analysis=="Primary_all_65"].set_index("model")
for m,e in [("Subjective",.231201),("Objective_proportional",2.853353),("Stable",.235941)]: rows.append((f"forward_a_{m}",lf.loc[m,"MAE_a"],e,1e-6))
out=pd.DataFrame(rows,columns=["metric","recomputed","expected","tolerance"]); out["PASS"]=abs(out.recomputed-out.expected)<=out.tolerance
out.to_csv(O/"checks.csv",index=False); print(out.to_string(index=False)); print(f"\nLeng: {out.PASS.sum()}/{len(out)} PASS")
raise SystemExit(0 if out.PASS.all() else 1)
