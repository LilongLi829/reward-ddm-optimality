"""Reproduce Hübner manuscript-facing statistics and Figures 1–2."""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from scipy import stats
R=Path(__file__).resolve().parents[1]; D=R/"data"; O=R/"results"; F=R/"figures"
O.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
h=pd.read_csv(D/"HUBNER_LOPO.csv"); s=pd.read_csv(D/"HUBNER_SATF.csv").set_index("metric").value
rows=[
("N",len(h),29,0),("MAE_M0",h.err_m0.mean(),.03955931,1e-6),("MAE_Mq",h.err_mq.mean(),.010187,1e-6),
("relative_reduction_pct",(1-h.err_mq.mean()/h.err_m0.mean())*100,74.2493,1e-3),
("N_improved",(h.improvement>0).sum(),29,0),("mean_improvement",h.improvement.mean(),.029372,1e-6),
("mean_q_train",h.q_train.mean(),615.863213,1e-5)]
for k,e,t in [("condition4_total_N",64,0),("condition4_excluded_meanRT_gt_550ms",3,0),("condition4_regression_N",61,0),
("satf_r",.8650329,1e-6),("satf_t",13.24346,1e-5),("satf_df",59,0),("satf_p",2.524969e-19,1e-24),
("satf_rt_at_50pct_accuracy_ms",192.6815,1e-3),("satf_rt_at_100pct_accuracy_ms",514.3876,1e-3),
("m0_optimal_rt_ms",260.4875,1e-3),("m0_optimal_accuracy",.605385,1e-6),("m0_expected_points_per_trial",235.805,1e-3),
("condition2_gain_N",29,0),("condition2_gain_observed_mean_rt_ms",416.0165,1e-3),
("condition2_gain_observed_median_rt_ms",399.0641,1e-3),("condition2_gain_observed_mean_accuracy",.888510,1e-6),
("N_observed_rt_slower_than_m0",29,0),("N_observed_accuracy_above_m0",29,0)]:
    rows.append((k,float(s[k]),e,t))
# Deterministic participant bootstrap (10,000 resamples).
rng=np.random.default_rng(20260929)
x=h.improvement.to_numpy()
boot=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(axis=1)
lo,hi=np.quantile(boot,[.025,.975])
rows += [("bootstrap95_lo",lo,lo,0),("bootstrap95_hi",hi,hi,0)]
out=pd.DataFrame(rows,columns=["metric","recomputed","expected","tolerance"]); out["PASS"]=abs(out.recomputed-out.expected)<=out.tolerance
out.to_csv(O/"checks.csv",index=False)
pd.DataFrame({"statistic":["mean_improvement","CI_2.5","CI_97.5"],"value":[x.mean(),lo,hi]}).to_csv(O/"bootstrap_ci.csv",index=False)

# Figure 1: model-independent SATF summary.
fig,ax=plt.subplots(figsize=(8,6))
acc=np.linspace(.50,1.00,200)
rt=float(s["satf_rt_at_50pct_accuracy_ms"])+(acc-.5)/.5*(float(s["satf_rt_at_100pct_accuracy_ms"])-float(s["satf_rt_at_50pct_accuracy_ms"]))
ax.plot(rt,acc,label="Empirical SATF")
ax.scatter([float(s["m0_optimal_rt_ms"])],[float(s["m0_optimal_accuracy"])],s=90,marker="D",label="M0 optimum")
ax.scatter([float(s["condition2_gain_observed_mean_rt_ms"])],[float(s["condition2_gain_observed_mean_accuracy"])],s=90,marker="o",label="Observed mean")
ax.set(xlabel="Response time (ms)",ylabel="Accuracy",title="Hübner: empirical speed–accuracy trade-off")
ax.legend(); fig.tight_layout(); fig.savefig(F/"Figure1_Hubner_SATF.png",dpi=200); plt.close(fig)

# Figure 2: participant-level held-out absolute errors.
fig,ax=plt.subplots(figsize=(8,6))
for _,r in h.iterrows(): ax.plot([0,1],[r.err_m0,r.err_mq],marker="o",alpha=.45)
ax.scatter([0,1],[h.err_m0.mean(),h.err_mq.mean()],s=120,marker="D",label="Mean MAE")
ax.set_xticks([0,1],["M0","Mq"]); ax.set_ylabel("Absolute threshold error")
ax.set_title("Hübner: held-out participant threshold error"); ax.legend()
fig.tight_layout(); fig.savefig(F/"Figure2_Hubner_LOPO.png",dpi=200); plt.close(fig)
print(out.to_string(index=False)); print(f"\nHubner: {out.PASS.sum()}/{len(out)} PASS; figures regenerated")
raise SystemExit(0 if out.PASS.all() else 1)
