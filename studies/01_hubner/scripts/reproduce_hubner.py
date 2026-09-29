"""Reproduce Hübner manuscript-facing validation statistics.

Inputs
------
HUBNER_LOPO.csv : participant-level leave-one-participant-out predictions.
HUBNER_SATF.csv : archived speed-accuracy trade-off summary statistics.

The script recomputes prediction errors from participant rows and checks the
archived SATF quantities used in the manuscript.
"""
from pathlib import Path
import pandas as pd, numpy as np
R=Path(__file__).resolve().parents[1]; D=R/"data"; O=R/"results"; O.mkdir(exist_ok=True)
# Load participant-level LOPO predictions and archived SATF summaries.
h=pd.read_csv(D/"HUBNER_LOPO.csv"); s=pd.read_csv(D/"HUBNER_SATF.csv").set_index("metric").value
# Recompute cross-participant prediction metrics.
rows=[
("N",len(h),29,0),("MAE_M0",h.err_m0.mean(),.03955931,1e-6),("MAE_Mq",h.err_mq.mean(),.010187,1e-6),
("relative_reduction_pct",(1-h.err_mq.mean()/h.err_m0.mean())*100,74.2493,1e-3),
("N_improved",(h.improvement>0).sum(),29,0),("mean_improvement",h.improvement.mean(),.029372,1e-6),
("mean_q_train",h.q_train.mean(),615.863213,1e-5)]
# Check model-independent SATF and task-summary quantities.
for k,e,t in [("condition4_total_N",64,0),("condition4_excluded_meanRT_gt_550ms",3,0),("condition4_regression_N",61,0),
("satf_r",.8650329,1e-6),("satf_t",13.24346,1e-5),("satf_df",59,0),("satf_p",2.524969e-19,1e-24),
("satf_rt_at_50pct_accuracy_ms",192.6815,1e-3),("satf_rt_at_100pct_accuracy_ms",514.3876,1e-3),
("m0_optimal_rt_ms",260.4875,1e-3),("m0_optimal_accuracy",.605385,1e-6),("m0_expected_points_per_trial",235.805,1e-3),
("condition2_gain_N",29,0),("condition2_gain_observed_mean_rt_ms",416.0165,1e-3),
("condition2_gain_observed_median_rt_ms",399.0641,1e-3),("condition2_gain_observed_mean_accuracy",.888510,1e-6),
("N_observed_rt_slower_than_m0",29,0),("N_observed_accuracy_above_m0",29,0)]:
    rows.append((k,float(s[k]),e,t))
out=pd.DataFrame(rows,columns=["metric","recomputed","expected","tolerance"]); out["PASS"]=abs(out.recomputed-out.expected)<=out.tolerance
out.to_csv(O/"checks.csv",index=False); print(out.to_string(index=False)); print(f"\nHubner: {out.PASS.sum()}/{len(out)} PASS")
raise SystemExit(0 if out.PASS.all() else 1)
