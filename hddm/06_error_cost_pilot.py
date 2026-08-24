# 中文说明：本脚本是“主观错误成本”扩展的组水平 pilot。
# 目的不是估计新的心理参数，而是先检查：当错误被赋予额外主观成本时，
# 理论最优阈值 a* 是否会向奖励条件下实际观察到的 a_R 移动。
# 输入沿用 05 正式分析冻结下来的 posterior draws 和重建奖励标准；
# pilot 只使用这些 posterior draws 的组平均参数，因此不能替代 07 的 320-draw 正式分析。
# q=0 对应原始 intended-payoff 目标；更大的 q 表示非正确 trial 的额外主观成本更高。
# 注意：本 pilot 用 p_noncorrect=1-accuracy（包含 wrong 与 timeout）；正式 07 脚本会将二者分开，且只惩罚 wrong response。
# 代码中的随机种子、任务时长、奖励点数、阈值网格和计算语句均保持原样。
# 输出仅用于选择正式 sensitivity analysis 的合理 q 范围。

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import hddm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/home/jovyan/project")

POST_FILE = (
    ROOT / "results" / "optimality"
    / "posterior_uncertainty_320draw"
    / "selected_posterior_draws.csv"
)

CRIT_FILE = (
    ROOT / "results" / "optimality"
    / "posterior_uncertainty_320draw"
    / "reconstructed_reward_criteria.csv"
)

OUT_DIR = (
    ROOT / "results" / "optimality"
    / "error_cost_pilot"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Same experimental constants as 05_final_optimality_320draw.py
# 中文：这里完全沿用 05 脚本的实验时长和奖励常数，避免改变基准目标函数。
# ------------------------------------------------------------

CUE_S = 0.800
FIXATION_S = 0.400
FEEDBACK_CORRECT_S = 0.750
FEEDBACK_INCORRECT_S = 1.250
ITI_S = 0.300
OVERALL_RT_DEADLINE_S = 3.000
POINTS_PER_SUCCESS = 10.0

# ------------------------------------------------------------
# Pilot settings
# 中文：pilot 的 q 网格以“点数等价值”表示额外错误成本；q=0 必须退化为原分析。
# q = extra subjective cost of an incorrect/non-correct trial,
# expressed in point-equivalent units.
# q=0 exactly reduces to the original intended-payoff objective.
# ------------------------------------------------------------

Q_GRID = np.array(
    [0.0, 2.5, 5.0, 10.0, 20.0, 40.0],
    dtype=float
)

# Wider than the original search because error cost may move
# the optimal threshold upward toward observed a_R.
A_GRID = np.arange(0.15, 1.601, 0.025)

N_SIM = 30000
SEED = 20260822

# ------------------------------------------------------------
# Load frozen posterior draws and reconstructed criteria
# 中文：读取已经冻结的 320 个联合后验样本和按实验日志重建的奖励 RT 标准。
# ------------------------------------------------------------

post = pd.read_csv(POST_FILE)
crit = pd.read_csv(CRIT_FILE)

required = {"a_R", "v_RI", "v_RC", "t"}
missing = required - set(post.columns)

if missing:
    raise RuntimeError(
        f"Missing posterior columns: {sorted(missing)}"
    )

if "criterion_ms" not in crit.columns:
    raise RuntimeError(
        "criterion_ms not found in reconstructed_reward_criteria.csv"
    )

criteria_s = crit["criterion_ms"].to_numpy(dtype=float) / 1000.0

# Group-level posterior means for the pilot only
# 中文：这里只取组平均做快速 pilot；正式不确定性传播由 07 脚本逐 draw 完成。
a_R = float(post["a_R"].mean())
v_RI = float(post["v_RI"].mean())
v_RC = float(post["v_RC"].mean())
t = float(post["t"].mean())

print("\n=== GROUP-LEVEL PILOT PARAMETERS ===")
print(f"N posterior draws = {len(post)}")
print(f"a_R  = {a_R:.6f}")
print(f"v_RI = {v_RI:.6f}")
print(f"v_RC = {v_RC:.6f}")
print(f"t    = {t:.6f}")
print(f"N reward criteria = {len(criteria_s)}")

# ------------------------------------------------------------
# Simulation: copied conceptually from original 05 script
# 中文：模拟逻辑与 05 保持一致，只为后续效用函数增加错误成本项准备必要概率量。
# ------------------------------------------------------------

def simulate_condition(a, v, t, n_sim, seed, criteria_s):

    # Common random numbers across candidate thresholds
    # 中文：不同候选阈值重复使用同一随机数流，以降低 Monte Carlo 噪声对 a* 排序的影响。
    np.random.seed(seed)

    sim = hddm.generate.gen_rts(
        size=n_sim,
        method="cdf",
        v=float(v),
        a=float(a),
        t=float(t),
        z=0.5,
        sv=0.0,
        sz=0.0,
        st=0.0,
        range_=(-6, 6),
        dt=1e-3,
    )

    rt = sim["rt"].to_numpy(dtype=float)
    response = sim["response"].to_numpy(dtype=float)

    responded = rt < OVERALL_RT_DEADLINE_S
    correct = responded & (response == 1.0)

    effective_rt = np.minimum(
        rt,
        OVERALL_RT_DEADLINE_S
    )

    feedback_s = np.where(
        correct,
        FEEDBACK_CORRECT_S,
        FEEDBACK_INCORRECT_S
    )

    cycle_s = (
        CUE_S
        + FIXATION_S
        + effective_rt
        + feedback_s
        + ITI_S
    )

    sorted_correct_rt = np.sort(rt[correct])
    n = float(n_sim)

    # Probability of satisfying the intended reward rule:
    # 中文：奖励事件仍严格按 intended rule 定义为“正确且快于重建奖励标准”。
    # correct AND faster than reconstructed block criterion
    p_correct_fast = np.mean(
        np.searchsorted(
            sorted_correct_rt,
            criteria_s,
            side="left"
        ) / n
    )

    accuracy = float(np.mean(correct))

    return {
        "p_correct_fast": float(p_correct_fast),
        "accuracy": accuracy,
        "p_noncorrect": float(1.0 - accuracy),
        "mean_cycle_s": float(np.mean(cycle_s)),
    }


def evaluate_a(a):

    # Reward-incongruent
    # 中文：先单独模拟 Reward-Incongruent 条件。
    ri = simulate_condition(
        a=a,
        v=v_RI,
        t=t,
        n_sim=N_SIM,
        seed=SEED + 101,
        criteria_s=criteria_s,
    )

    # Reward-congruent
    # 中文：再单独模拟 Reward-Congruent 条件。
    rc = simulate_condition(
        a=a,
        v=v_RC,
        t=t,
        n_sim=N_SIM,
        seed=SEED + 202,
        criteria_s=criteria_s,
    )

    # Experiment has equal RI / RC weighting
    # 中文：实验中 RI/RC 等比例，因此两个条件按 0.5/0.5 加权。
    avg = {
        key: 0.5 * (ri[key] + rc[key])
        for key in ri
    }

    return {
        "a": float(a),
        **avg
    }


# ------------------------------------------------------------
# Simulate each threshold once.
# 中文：每个候选 a 只模拟一次，随后同一批模拟结果用于所有 q，保证 q 间比较公平。
# The same simulations can then be evaluated under every q.
# ------------------------------------------------------------

rows = []

print("\n=== SIMULATING THRESHOLD GRID ===")

for i, a in enumerate(A_GRID, start=1):

    result = evaluate_a(float(a))
    rows.append(result)

    print(
        f"{i:02d}/{len(A_GRID):02d}  "
        f"a={a:.3f}  "
        f"ACC={result['accuracy']:.4f}  "
        f"P(correct&fast)={result['p_correct_fast']:.4f}"
    )

curve = pd.DataFrame(rows)

# ------------------------------------------------------------
# Calculate utility for each subjective error cost q
# 中文：在不重跑 DDM 模拟的前提下，对每个 q 重新计算扩展效用率并寻找最优 a。
# ------------------------------------------------------------

summary_rows = []

for q in Q_GRID:

    col = f"utility_q_{q:g}"

    curve[col] = (
        POINTS_PER_SUCCESS * curve["p_correct_fast"]
        - q * curve["p_noncorrect"]
    ) / curve["mean_cycle_s"]

    best_idx = curve[col].idxmax()
    best = curve.loc[best_idx]

    a_star = float(best["a"])

    summary_rows.append({
        "q": float(q),
        "q_over_reward": float(q / POINTS_PER_SUCCESS),
        "a_star": a_star,
        "a_R_group": a_R,
        "gap_aR_minus_astar": a_R - a_star,
        "utility_star": float(best[col]),
        "accuracy_at_astar": float(best["accuracy"]),
        "p_correct_fast_at_astar":
            float(best["p_correct_fast"]),
    })

summary = pd.DataFrame(summary_rows)

summary["abs_gap"] = np.abs(
    summary["gap_aR_minus_astar"]
)

closest = summary.loc[
    summary["abs_gap"].idxmin()
]

# ------------------------------------------------------------
# Save
# 中文：保存完整曲线和 pilot 汇总，便于核对 q 网格是否覆盖可能的 a* 区域。
# ------------------------------------------------------------

curve_file = OUT_DIR / "error_cost_pilot_curve.csv"
summary_file = OUT_DIR / "error_cost_pilot_summary.csv"

curve.to_csv(curve_file, index=False)
summary.to_csv(summary_file, index=False)

# ------------------------------------------------------------
# Final output
# 中文：打印每个 q 的 a*(q) 与实际 a_R 的差距；这里只用于 pilot 判断。
# ------------------------------------------------------------

print("\n==============================================")
print("ERROR-COST PILOT RESULTS")
print("==============================================")

print(
    summary[
        [
            "q",
            "q_over_reward",
            "a_star",
            "a_R_group",
            "gap_aR_minus_astar",
            "accuracy_at_astar",
        ]
    ].to_string(index=False)
)

print("\nClosest tested q to observed a_R:")
print(
    f"q = {closest['q']:.3f} points "
    f"({closest['q_over_reward']:.3f} x reward)"
)
print(f"a*(q) = {closest['a_star']:.3f}")
print(f"a_R    = {a_R:.3f}")
print(
    f"a_R-a*(q) = "
    f"{closest['gap_aR_minus_astar']:.3f}"
)

print("\nSaved:")
print(curve_file)
print(summary_file)
