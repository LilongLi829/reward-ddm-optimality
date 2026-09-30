"""Reproduce Leng archived effects/mapping checks and Figures 6–7."""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]; O=R/"results"; F=R/"figures"; O.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
le=pd.read_csv(R/"data/LENG_DDM_effects.csv"); rows=[]
targets={("Drift rate v","Reward"):(.109869,.044327,.175386),("Threshold a","Reward"):(-.048249,-.098626,.004626),
("Drift rate v","Penalty"):(.004052,-.071715,.080112),("Threshold a","Penalty"):(.074910,.007321,.139704)}
for key,(mu,lo,hi) in targets.items():
 x=le[(le.parameter==key[0])&(le.incentive==key[1])].iloc[0]
 rows += [(f"{key[1]}_{key[0]}_mean",x["mean"],mu,1e-6),(f"{key[1]}_{key[0]}_CIlo",x.boot_lo,lo,1e-6),(f"{key[1]}_{key[0]}_CIhi",x.boot_hi,hi,1e-6)]
lm=pd.read_csv(R/"data/LENG_mapping.csv"); lm=lm[lm.analysis=="Primary_all_participants"].set_index("model")
for m,e in [("Subjective",.326167),("Objective_proportional",.543403),("Stable",.326961)]: rows.append((f"mapping_{m}",lm.loc[m,"MAE_RP_equalweight"],e,1e-6))
lf=pd.read_csv(R/"data/LENG_forward.csv"); lf=lf[lf.analysis=="Primary_all_65"].set_index("model")
for m,e in [("Subjective",.231201),("Objective_proportional",2.853353),("Stable",.235941)]: rows.append((f"forward_a_{m}",lf.loc[m,"MAE_a"],e,1e-6))
out=pd.DataFrame(rows,columns=["metric","recomputed","expected","tolerance"]); out["PASS"]=abs(out.recomputed-out.expected)<=out.tolerance
out.to_csv(O/"checks.csv",index=False)
# Figure 6
plot=le.set_index(["incentive","parameter"])
labels=["Reward → drift v","Reward → threshold a","Penalty → drift v","Penalty → threshold a"]
keys=[("Reward","Drift rate v"),("Reward","Threshold a"),("Penalty","Drift rate v"),("Penalty","Threshold a")]
means=np.array([plot.loc[k,"mean"] for k in keys]); lo=np.array([plot.loc[k,"boot_lo"] for k in keys]); hi=np.array([plot.loc[k,"boot_hi"] for k in keys])
fig,ax=plt.subplots(figsize=(9,6)); y=np.arange(4)
ax.errorbar(means,y,xerr=[means-lo,hi-means],fmt="o",capsize=4); ax.axvline(0,linewidth=1)
ax.set_yticks(y,labels); ax.set_xlabel("Participant-level effect"); ax.set_title("Leng: incentive effects on DDM parameters"); ax.invert_yaxis()
fig.tight_layout(); fig.savefig(F/"Figure6_Leng_effects.png",dpi=200); plt.close(fig)
# Figure 7
models=["Subjective","Objective_proportional","Stable"]; x=np.arange(3); width=.36
fig,ax=plt.subplots(figsize=(9,6))
ax.bar(x-width/2,[lm.loc[m,"MAE_RP_equalweight"] for m in models],width,label="Value mapping MAE")
ax.bar(x+width/2,[lf.loc[m,"MAE_a"] for m in models],width,label="Forward threshold MAE")
ax.set_xticks(x,models,rotation=10); ax.set_ylabel("Mean absolute error"); ax.set_title("Leng: held-out mapping and forward prediction"); ax.legend()
fig.tight_layout(); fig.savefig(F/"Figure7_Leng_mapping.png",dpi=200); plt.close(fig)
print(out.to_string(index=False)); print(f"\nLeng: {out.PASS.sum()}/{len(out)} PASS; figures regenerated")
raise SystemExit(0 if out.PASS.all() else 1)
