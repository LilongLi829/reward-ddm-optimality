#!/usr/bin/env python3
"""
07_error_cost_320draw.py

Formal posterior-sensitivity analysis extending the intended-payoff
reward-rate objective with an additional subjective cost for an
incorrect response.

Primary utility-rate objective:
    U(a; q) =
        [10 * P(correct and RT < criterion | a)
         - q_points * P(wrong response | a)]
        / E[T_trial | a]

where:
    q_points = q_relative * 10
    q_relative = 0 reproduces the original intended-payoff objective.

Important design choices:
- Uses the frozen 320 joint posterior draws already saved by the original
  formal analysis. It does NOT refit HDDM.
- Keeps v_RI, v_RC, t, z=.5 and task timing fixed within each posterior draw.
- Reward-incongruent and reward-congruent conditions are simulated separately
  and combined 50:50.
- Wrong responses and timeouts are recorded separately. The PRIMARY extra
  subjective cost is applied only to wrong responses.
- Uses common random numbers across candidate a values within each posterior
  draw.
- Saves an atomic checkpoint after every completed posterior draw.
- Supports interruption/resume.
- Produces final long-format posterior draws, summaries, probability summaries,
  run configuration, and an input hash manifest suitable for GitHub archiving.

Recommended workflow:
    # Smoke test: complete 3 new posterior draws
    python hddm/07_error_cost_320draw.py --max-new-draws 3

    # Continue from checkpoint
    python hddm/07_error_cost_320draw.py --resume

All output is written under:
    results/optimality/error_cost_320draw/
"""

# 这是论文中“主观错误成本敏感性分析”的正式 320 posterior-draw 脚本。
# 1) 不重新拟合 HDDM，而是复用 05 已冻结的 320 个联合后验样本。
# 2) 每个 draw 内固定 v_RI、v_RC、t、z=0.5 以及实验时长/奖励结构，只搜索阈值 a。
# 3) q_relative 是预先指定的 sensitivity weight，不是从当前数据估计出的心理参数。
# 4) q_points = 10 * q_relative；额外成本只施加到真正的 wrong response，timeout 单独记录。
# 5) Reward-Incongruent 与 Reward-Congruent 分开模拟，再按 0.5/0.5 合并。
# 6) 同一 posterior draw 的候选 a 使用 common random numbers，降低 Monte Carlo 比较噪声。
# 7) 每完成一个 draw 就原子化保存 checkpoint，可安全中断并用 --resume 继续。
# 8) q_relative=0 是关键基线复现检查，应该与 05 的 intended-payoff optimum 一致。
# 9) 最终输出包含逐 draw 结果、汇总、概率、validation、运行配置和输入哈希清单。

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import hddm
except Exception as exc:
    raise RuntimeError(
        "Could not import HDDM. Run this script inside the same Docker/HDDM "
        "environment used for the original analysis."
    ) from exc

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------
# Experiment-2 timing constants: identical to the original 05 analysis
# 实验时长常数与 05 完全一致，确保 q=0 可以作为严格的基线复现。
# ---------------------------------------------------------------------

CUE_S = 0.800
FIXATION_S = 0.400
FEEDBACK_CORRECT_S = 0.750
FEEDBACK_INCORRECT_S = 1.250
ITI_S = 0.300
OVERALL_RT_DEADLINE_S = 3.000
POINTS_PER_SUCCESS = 10.0

# Sensitivity values are relative to one 10-point successful reward.
# q_relative 以一次成功奖励（10 分）为单位；例如 q_relative=1 对应额外错误成本 10 分。
# q_relative = 1 means one wrong response has an extra subjective
# negative utility equal in magnitude to one successful 10-point reward.
DEFAULT_Q_RELATIVE = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--project-root", default="/home/jovyan/project")

    p.add_argument(
        "--source-subdir",
        default="posterior_uncertainty_320draw",
        help="Existing original optimality result directory."
    )

    p.add_argument(
        "--output-subdir",
        default="error_cost_320draw",
        help="New output directory under results/optimality."
    )

    p.add_argument(
        "--q-relative",
        nargs="+",
        type=float,
        default=DEFAULT_Q_RELATIVE,
        help=(
            "Extra wrong-response cost as multiples of one 10-point reward. "
            "Default: 0 .5 1 2 3 4"
        ),
    )

    # Preserve the original coarse grid exactly in the region relevant to q=0,
    # then extend upward for stronger error-cost sensitivity values.
    p.add_argument("--coarse-base-min", type=float, default=0.15)
    p.add_argument("--coarse-base-max", type=float, default=0.75)
    p.add_argument("--coarse-base-step", type=float, default=0.02)
    p.add_argument("--coarse-extend-min", type=float, default=0.80)
    p.add_argument("--coarse-extend-max", type=float, default=2.00)
    p.add_argument("--coarse-extend-step", type=float, default=0.05)

    p.add_argument("--fine-halfwidth", type=float, default=0.03)
    p.add_argument("--fine-step", type=float, default=0.005)

    # Same simulation sizes as the original 05 formal analysis.
    p.add_argument("--n-sim-coarse", type=int, default=15000)
    p.add_argument("--n-sim-fine", type=int, default=40000)

    # Wider re-estimation for fine-grid edge hits.
    p.add_argument("--refine-halfwidth", type=float, default=0.12)
    p.add_argument("--refine-step", type=float, default=0.005)
    p.add_argument("--n-sim-refine", type=int, default=80000)
    p.add_argument("--refine-global-min", type=float, default=0.05)
    p.add_argument("--refine-global-max", type=float, default=2.00)

    # Same seed as the original 05 formal analysis.
    p.add_argument("--seed", type=int, default=20260820)

    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the per-draw checkpoint."
    )

    p.add_argument(
        "--max-new-draws",
        type=int,
        default=0,
        help=(
            "Maximum number of NEW posterior draws to complete in this run. "
            "0 means all remaining draws. Useful for a 3-draw smoke test."
        ),
    )

    return p.parse_args()


# ---------------------------------------------------------------------
# File helpers
# 文件辅助函数负责哈希校验和原子写入，避免长时间运行中断时损坏 checkpoint。
# ---------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_json(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def grid_values(lo: float, hi: float, step: float) -> np.ndarray:
    return np.arange(lo, hi + step / 2.0, step, dtype=float)


def local_grid(
    center: float,
    halfwidth: float,
    step: float,
    global_lo: float,
    global_hi: float,
) -> np.ndarray:
    lo = max(float(global_lo), float(center) - float(halfwidth))
    hi = min(float(global_hi), float(center) + float(halfwidth))
    return grid_values(lo, hi, step)


def utility_rate(row: pd.Series | dict, q_points: float) -> float:
    numerator = (
        POINTS_PER_SUCCESS * float(row["p_correct_fast"])
        - float(q_points) * float(row["p_wrong"])
    )
    return numerator / float(row["mean_cycle_s"])


# ---------------------------------------------------------------------
# Simulation
# DDM 模拟阶段保留正确、错误、超时三类事件，为扩展目标函数提供独立概率量。
# ---------------------------------------------------------------------

def simulate_condition(
    *,
    a: float,
    v: float,
    t: float,
    n_sim: int,
    seed: int,
    criteria_s: np.ndarray,
) -> dict[str, float]:
    """
    Simulate one reward/congruency condition.

    "correct" reproduces the original formal accounting:
        responded before 3 s AND response == 1.

    The extra subjective cost is NOT applied here; we separately retain:
        p_wrong   = responded before 3 s but response != 1
        p_timeout = did not respond before 3 s

    This lets the primary sensitivity analysis penalize wrong responses only.
    """

    # Common random numbers across candidate thresholds within the draw.
    # 同一 draw 内不同 a 使用相同随机数种子，使 a 之间的效用差异更少受随机波动干扰。
    np.random.seed(seed)

    sim = hddm.generate.gen_rts(
        size=int(n_sim),
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
    wrong = responded & (response != 1.0)
    timeout = ~responded

    effective_rt = np.minimum(rt, OVERALL_RT_DEADLINE_S)

    # Preserve the original 05 trial-time accounting:
    # trial duration 的计算仍沿用 05；所有非正确 trial 使用错误反馈时长。
    # all non-correct trials receive the incorrect-feedback duration.
    feedback_s = np.where(
        correct,
        FEEDBACK_CORRECT_S,
        FEEDBACK_INCORRECT_S,
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

    # Average the probability of satisfying each reconstructed block criterion.
    # 对所有重建出的 block-specific reward criterion 求平均，复现实验奖励规则。
    p_correct_fast = np.mean(
        np.searchsorted(
            sorted_correct_rt,
            criteria_s,
            side="left",
        ) / n
    )

    return {
        "p_correct_fast": float(p_correct_fast),
        "accuracy": float(np.mean(correct)),
        "p_wrong": float(np.mean(wrong)),
        "p_timeout": float(np.mean(timeout)),
        "p_noncorrect": float(np.mean(~correct)),
        "mean_cycle_s": float(np.mean(cycle_s)),
    }


def evaluate_a(
    a: float,
    *,
    v_RI: float,
    v_RC: float,
    t: float,
    n_sim: int,
    seed: int,
    criteria_s: np.ndarray,
) -> dict[str, float]:
    """
    Simulate reward-incongruent and reward-congruent separately and average 50:50.
    """

    ri = simulate_condition(
        a=a,
        v=v_RI,
        t=t,
        n_sim=n_sim,
        seed=seed + 101,
        criteria_s=criteria_s,
    )

    rc = simulate_condition(
        a=a,
        v=v_RC,
        t=t,
        n_sim=n_sim,
        seed=seed + 202,
        criteria_s=criteria_s,
    )

    avg = {k: 0.5 * (ri[k] + rc[k]) for k in ri}

    return {
        "a": float(a),
        **avg,
    }


# ---------------------------------------------------------------------
# Summaries
# 把逐 draw 结果汇总成 posterior mean、SD、median、95% CrI 和方向概率。
# ---------------------------------------------------------------------

def summarize_vector(x: pd.Series) -> dict[str, float]:
    return {
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "median": float(x.median()),
        "q2.5": float(x.quantile(0.025)),
        "q97.5": float(x.quantile(0.975)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def build_summaries(res: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    probability_rows = []

    for q_rel, g in res.groupby("q_relative", sort=True):
        q_points = float(g["q_points"].iloc[0])

        for metric in [
            "a_star",
            "gap_aR_minus_astar",
            "utility_star",
            "utility_at_aR",
            "efficiency",
            "accuracy_at_astar",
            "p_wrong_at_astar",
            "p_timeout_at_astar",
        ]:
            ss = summarize_vector(g[metric])
            summary_rows.append({
                "q_relative": float(q_rel),
                "q_points": q_points,
                "metric": metric,
                **ss,
            })

        gap_positive = g["gap_aR_minus_astar"] > 0
        probability_rows.append({
            "q_relative": float(q_rel),
            "q_points": q_points,
            "quantity": "P(a_R > a_star | sampled posterior draws)",
            "value": float(gap_positive.mean()),
            "n_success": int(gap_positive.sum()),
            "n_draws": int(len(g)),
        })

        gap_negative = g["gap_aR_minus_astar"] < 0
        probability_rows.append({
            "q_relative": float(q_rel),
            "q_points": q_points,
            "quantity": "P(a_R < a_star | sampled posterior draws)",
            "value": float(gap_negative.mean()),
            "n_success": int(gap_negative.sum()),
            "n_draws": int(len(g)),
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(probability_rows)


# ---------------------------------------------------------------------
# One posterior draw
# 对一个联合 posterior draw 完成 coarse → fine → 必要时 refine 的完整最优阈值搜索。
# ---------------------------------------------------------------------

def analyse_one_draw(
    row: pd.Series,
    *,
    q_relative: list[float],
    coarse: np.ndarray,
    criteria_s: np.ndarray,
    args: argparse.Namespace,
    reference: pd.DataFrame | None,
) -> list[dict]:
    aid = int(row["analysis_id"])
    draw_seed = int(args.seed + aid * 10000)

    a_R = float(row["a_R"])
    v_RI = float(row["v_RI"])
    v_RC = float(row["v_RC"])
    t = float(row["t"])

    # ---------------------------
    # 1) Wide coarse search
    # 先在宽网格上为每个 q 找到大致最优区域。
    # ---------------------------
    coarse_rows = [
        evaluate_a(
            float(a),
            v_RI=v_RI,
            v_RC=v_RC,
            t=t,
            n_sim=int(args.n_sim_coarse),
            seed=draw_seed,
            criteria_s=criteria_s,
        )
        for a in coarse
    ]

    cg = pd.DataFrame(coarse_rows)

    coarse_peak_by_q: dict[float, float] = {}
    fine_grid_by_q: dict[float, np.ndarray] = {}

    for q_rel in q_relative:
        q_points = float(q_rel * POINTS_PER_SUCCESS)
        u = cg.apply(lambda r: utility_rate(r, q_points), axis=1)
        peak = float(cg.loc[u.idxmax(), "a"])

        if np.isclose(peak, coarse.min()) or np.isclose(peak, coarse.max()):
            raise RuntimeError(
                f"analysis_id={aid}, q_relative={q_rel}: "
                f"coarse optimum hit global edge at a={peak:.3f}. "
                "Expand the coarse grid before interpreting this draw."
            )

        coarse_peak_by_q[q_rel] = peak

        fine_grid_by_q[q_rel] = local_grid(
            peak,
            float(args.fine_halfwidth),
            float(args.fine_step),
            float(args.refine_global_min),
            float(args.refine_global_max),
        )

    # Simulate the union once, but restrict each q to its own local grid.
    # 把各 q 的局部候选 a 取并集后只模拟一次，再分别在各自网格内求最优。
    fine_union = np.unique(
        np.round(
            np.concatenate(list(fine_grid_by_q.values())),
            10,
        )
    )

    fine_rows = [
        evaluate_a(
            float(a),
            v_RI=v_RI,
            v_RC=v_RC,
            t=t,
            n_sim=int(args.n_sim_fine),
            seed=draw_seed + 5000,
            criteria_s=criteria_s,
        )
        for a in fine_union
    ]

    fg = pd.DataFrame(fine_rows)
    fg["_a_round"] = fg["a"].round(10)

    # Observed reward threshold under the same posterior draw.
    # 理论 a*(q) 与同一个 posterior draw 中的实际奖励阈值 a_R 配对比较。
    at_hat = evaluate_a(
        a_R,
        v_RI=v_RI,
        v_RC=v_RC,
        t=t,
        n_sim=int(args.n_sim_fine),
        seed=draw_seed + 5000,
        criteria_s=criteria_s,
    )

    records: list[dict] = []

    for q_rel in q_relative:
        q_points = float(q_rel * POINTS_PER_SUCCESS)
        q_grid = np.round(fine_grid_by_q[q_rel], 10)
        local = fg[fg["_a_round"].isin(set(q_grid.tolist()))].copy()

        local["utility_rate"] = local.apply(
            lambda r: utility_rate(r, q_points),
            axis=1,
        )

        best = local.loc[local["utility_rate"].idxmax()].copy()

        grid_min = float(q_grid.min())
        grid_max = float(q_grid.max())
        edge = bool(
            np.isclose(float(best["a"]), grid_min)
            or np.isclose(float(best["a"]), grid_max)
        )

        edge_refined = False
        refine_attempts = 0
        refine_grid_min = np.nan
        refine_grid_max = np.nan

        # ---------------------------
        # 2) Automatic wider refinement on edge hits
        # ---------------------------
        if edge:
            edge_refined = True
            center = float(coarse_peak_by_q[q_rel])
            half = float(args.refine_halfwidth)

            for attempt in range(1, 5):
                refine_attempts = attempt

                lo = max(
                    float(args.refine_global_min),
                    center - half,
                )
                hi = min(
                    float(args.refine_global_max),
                    center + half,
                )

                rgrid = grid_values(
                    lo,
                    hi,
                    float(args.refine_step),
                )

                rrows = [
                    evaluate_a(
                        float(a),
                        v_RI=v_RI,
                        v_RC=v_RC,
                        t=t,
                        n_sim=int(args.n_sim_refine),
                        seed=draw_seed + 9000,
                        criteria_s=criteria_s,
                    )
                    for a in rgrid
                ]

                rg = pd.DataFrame(rrows)
                rg["utility_rate"] = rg.apply(
                    lambda r: utility_rate(r, q_points),
                    axis=1,
                )

                best = rg.loc[rg["utility_rate"].idxmax()].copy()

                refine_grid_min = float(rgrid.min())
                refine_grid_max = float(rgrid.max())

                edge = bool(
                    np.isclose(float(best["a"]), refine_grid_min)
                    or np.isclose(float(best["a"]), refine_grid_max)
                )

                if not edge:
                    break

                if (
                    np.isclose(refine_grid_min, args.refine_global_min)
                    and np.isclose(refine_grid_max, args.refine_global_max)
                ):
                    break

                half *= 1.75

            # Re-estimate observed a_R at the higher refinement simulation size
            # for internally consistent utility efficiency.
            at_hat_q = evaluate_a(
                a_R,
                v_RI=v_RI,
                v_RC=v_RC,
                t=t,
                n_sim=int(args.n_sim_refine),
                seed=draw_seed + 9000,
                criteria_s=criteria_s,
            )
        else:
            at_hat_q = at_hat

        utility_star = utility_rate(best, q_points)
        utility_hat = utility_rate(at_hat_q, q_points)

        efficiency = (
            float(utility_hat / utility_star)
            if utility_star > 0
            else np.nan
        )

        # Optional validation against the frozen original q=0 result.
        ref_a_star = np.nan
        baseline_abs_diff = np.nan

        if reference is not None and np.isclose(q_rel, 0.0):
            rr = reference[reference["analysis_id"].astype(int) == aid]
            if len(rr) == 1 and "a_star_intended" in rr.columns:
                ref_a_star = float(rr.iloc[0]["a_star_intended"])
                baseline_abs_diff = abs(float(best["a"]) - ref_a_star)

        records.append({
            "analysis_id": aid,
            "chain": int(row["chain"]),
            "source_draw": int(row["source_draw"]),
            "a_R": a_R,
            "v_RI": v_RI,
            "v_RC": v_RC,
            "t": t,

            "q_relative": float(q_rel),
            "q_points": q_points,

            "a_star": float(best["a"]),
            "gap_aR_minus_astar": float(a_R - float(best["a"])),

            "utility_star": float(utility_star),
            "utility_at_aR": float(utility_hat),
            "efficiency": float(efficiency),

            "accuracy_at_astar": float(best["accuracy"]),
            "p_correct_fast_at_astar": float(best["p_correct_fast"]),
            "p_wrong_at_astar": float(best["p_wrong"]),
            "p_timeout_at_astar": float(best["p_timeout"]),
            "p_noncorrect_at_astar": float(best["p_noncorrect"]),
            "mean_cycle_s_at_astar": float(best["mean_cycle_s"]),

            "accuracy_at_aR": float(at_hat_q["accuracy"]),
            "p_correct_fast_at_aR": float(at_hat_q["p_correct_fast"]),
            "p_wrong_at_aR": float(at_hat_q["p_wrong"]),
            "p_timeout_at_aR": float(at_hat_q["p_timeout"]),
            "p_noncorrect_at_aR": float(at_hat_q["p_noncorrect"]),
            "mean_cycle_s_at_aR": float(at_hat_q["mean_cycle_s"]),

            "coarse_peak": float(coarse_peak_by_q[q_rel]),
            "fine_grid_min": grid_min,
            "fine_grid_max": grid_max,
            "fine_on_edge_after_all_refinement": bool(edge),

            "edge_refined": bool(edge_refined),
            "refine_attempts": int(refine_attempts),
            "refine_grid_min": refine_grid_min,
            "refine_grid_max": refine_grid_max,

            "original_q0_a_star_intended": ref_a_star,
            "q0_abs_difference_from_original": baseline_abs_diff,

            "seed_draw": draw_seed,
            "n_sim_coarse": int(args.n_sim_coarse),
            "n_sim_fine": int(args.n_sim_fine),
            "n_sim_refine": int(args.n_sim_refine),
        })

    return records


# ---------------------------------------------------------------------
# Main
# 主流程负责参数解析、冻结输入、断点续跑、逐 draw 分析、最终汇总与验证。
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    root = Path(args.project_root)
    source_dir = root / "results" / "optimality" / args.source_subdir
    out_dir = root / "results" / "optimality" / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_file = source_dir / "selected_posterior_draws.csv"
    criteria_file = source_dir / "reconstructed_reward_criteria.csv"
    original_final_file = source_dir / "posterior_optimality_draws_final.csv"

    for required in [selected_file, criteria_file]:
        if not required.exists():
            raise FileNotFoundError(required)

    selected = pd.read_csv(selected_file).sort_values("analysis_id").reset_index(drop=True)
    criteria = pd.read_csv(criteria_file)

    required_post = {
        "analysis_id", "a_R", "v_RI", "v_RC", "t", "chain", "source_draw"
    }
    missing = required_post - set(selected.columns)

    if missing:
        raise RuntimeError(
            f"selected_posterior_draws.csv missing columns: {sorted(missing)}"
        )

    if "criterion_ms" not in criteria.columns:
        raise RuntimeError(
            "reconstructed_reward_criteria.csv missing criterion_ms"
        )

    criteria_s = criteria["criterion_ms"].to_numpy(dtype=float) / 1000.0

    q_relative = sorted(set(float(x) for x in args.q_relative))

    if any(q < 0 for q in q_relative):
        raise ValueError("q_relative values must be non-negative.")

    # Original coarse region + extended region.
    # 保留 05 原 coarse 区间，同时向上扩展，以覆盖较大错误成本下可能上移的 a*。
    coarse_original = grid_values(
        args.coarse_base_min,
        args.coarse_base_max,
        args.coarse_base_step,
    )

    coarse_extension = grid_values(
        args.coarse_extend_min,
        args.coarse_extend_max,
        args.coarse_extend_step,
    )

    coarse = np.unique(
        np.round(
            np.concatenate([coarse_original, coarse_extension]),
            10,
        )
    )

    # Frozen reference results are optional but useful for q=0 validation.
    # 若原 05 逐 draw 结果存在，就逐项验证 q=0 的新 a* 是否与原结果一致。
    reference = (
        pd.read_csv(original_final_file)
        if original_final_file.exists()
        else None
    )

    # Save reproducibility metadata before the first simulation.
    # 在模拟开始前保存配置和输入 SHA-256，便于之后审计输入是否被改动。
    config = {
        "script": "hddm/07_error_cost_320draw.py",
        "project_root": str(root),
        "source_subdir": args.source_subdir,
        "output_subdir": args.output_subdir,
        "n_posterior_draws": int(len(selected)),
        "q_relative": q_relative,
        "q_points": [q * POINTS_PER_SUCCESS for q in q_relative],
        "utility_definition": (
            "[10*P(correct & RT<criterion) - q_points*P(wrong response)] "
            "/ E[T_trial]"
        ),
        "wrong_definition": "responded before 3 s AND response != 1",
        "timeout_definition": "RT >= 3 s",
        "q_cost_applied_to": "wrong responses only",
        "RI_RC_weighting": "0.5 / 0.5",
        "z": 0.5,
        "points_per_success": POINTS_PER_SUCCESS,
        "timing_seconds": {
            "reward_cue": CUE_S,
            "fixation": FIXATION_S,
            "feedback_correct": FEEDBACK_CORRECT_S,
            "feedback_incorrect": FEEDBACK_INCORRECT_S,
            "ITI": ITI_S,
            "overall_RT_deadline": OVERALL_RT_DEADLINE_S,
        },
        "coarse_original_grid": {
            "min": args.coarse_base_min,
            "max": args.coarse_base_max,
            "step": args.coarse_base_step,
        },
        "coarse_extension_grid": {
            "min": args.coarse_extend_min,
            "max": args.coarse_extend_max,
            "step": args.coarse_extend_step,
        },
        "fine": {
            "halfwidth": args.fine_halfwidth,
            "step": args.fine_step,
        },
        "simulation_sizes": {
            "coarse_per_congruency_per_a": args.n_sim_coarse,
            "fine_per_congruency_per_a": args.n_sim_fine,
            "refine_per_congruency_per_a": args.n_sim_refine,
        },
        "seed": args.seed,
    }

    atomic_write_json(
        config,
        out_dir / "error_cost_run_config.json",
    )

    manifest = {
        "selected_posterior_draws": {
            "relative_path": str(selected_file.relative_to(root)),
            "sha256": sha256_file(selected_file),
        },
        "reconstructed_reward_criteria": {
            "relative_path": str(criteria_file.relative_to(root)),
            "sha256": sha256_file(criteria_file),
        },
    }

    if original_final_file.exists():
        manifest["original_posterior_optimality_draws_final"] = {
            "relative_path": str(original_final_file.relative_to(root)),
            "sha256": sha256_file(original_final_file),
        }

    atomic_write_json(
        manifest,
        out_dir / "error_cost_input_manifest.json",
    )

    # Also copy the two small frozen input tables into the new directory
    # 中文：把两个关键冻结输入复制进结果目录，使 GitHub 中该结果文件夹可独立核查。
    # so the GitHub result folder is self-contained.
    frozen_inputs = out_dir / "frozen_inputs"
    frozen_inputs.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        selected_file,
        frozen_inputs / "selected_posterior_draws.csv",
    )
    shutil.copy2(
        criteria_file,
        frozen_inputs / "reconstructed_reward_criteria.csv",
    )

    checkpoint_file = out_dir / "error_cost_draws_checkpoint.csv"
    partial_summary_file = out_dir / "error_cost_summary_partial.csv"
    partial_probs_file = out_dir / "error_cost_probabilities_partial.csv"

    if checkpoint_file.exists() and not args.resume:
        raise RuntimeError(
            f"Checkpoint already exists: {checkpoint_file}\n"
            "Use --resume to continue, or deliberately remove the output "
            "directory before starting a new analysis."
        )

    if args.resume and checkpoint_file.exists():
        existing = pd.read_csv(checkpoint_file)

        expected_q = set(q_relative)
        completed_ids = set()

        for aid, g in existing.groupby("analysis_id"):
            got_q = set(np.round(g["q_relative"].astype(float), 10))
            want_q = set(np.round(list(expected_q), 10))
            if got_q == want_q:
                completed_ids.add(int(aid))

        print(f"\nResuming from checkpoint: {checkpoint_file}")
        print(f"Completed posterior draws: {len(completed_ids)}/{len(selected)}")
    else:
        existing = pd.DataFrame()
        completed_ids = set()

    remaining = selected[~selected["analysis_id"].astype(int).isin(completed_ids)].copy()

    if args.max_new_draws and args.max_new_draws > 0:
        remaining = remaining.head(int(args.max_new_draws))

    print("\n======================================================")
    print("FORMAL ERROR-COST POSTERIOR SENSITIVITY ANALYSIS")
    print("======================================================")
    print("Frozen posterior draws:", len(selected))
    print("Already completed:", len(completed_ids))
    print("New draws this invocation:", len(remaining))
    print("q_relative:", q_relative)
    print(
        f"Coarse grid: {coarse[0]:.3f}..{coarse[-1]:.3f} "
        f"({len(coarse)} candidate values)"
    )
    print(
        "n_sim coarse/fine/refine:",
        args.n_sim_coarse,
        args.n_sim_fine,
        args.n_sim_refine,
    )

    current = existing.copy()

    for j, (_, row) in enumerate(remaining.iterrows(), start=1):
        aid = int(row["analysis_id"])

        records = analyse_one_draw(
            row,
            q_relative=q_relative,
            coarse=coarse,
            criteria_s=criteria_s,
            args=args,
            reference=reference,
        )

        new_df = pd.DataFrame(records)

        if current.empty:
            current = new_df
        else:
            # In resume mode, replace any partial rows for this aid defensively.
            current = current[current["analysis_id"].astype(int) != aid]
            current = pd.concat([current, new_df], ignore_index=True)

        current = current.sort_values(
            ["analysis_id", "q_relative"]
        ).reset_index(drop=True)

        # Atomic checkpoint AFTER every completed posterior draw.
        atomic_to_csv(current, checkpoint_file)

        summary, probs = build_summaries(current)
        atomic_to_csv(summary, partial_summary_file)
        atomic_to_csv(probs, partial_probs_file)

        q0 = new_df[np.isclose(new_df["q_relative"], 0.0)]
        q4 = new_df[np.isclose(new_df["q_relative"], 4.0)]

        q0_text = (
            f"q0 a*={q0.iloc[0]['a_star']:.3f}"
            if len(q0)
            else "q0 n/a"
        )

        q4_text = (
            f"q4 a*={q4.iloc[0]['a_star']:.3f}"
            if len(q4)
            else "q4 n/a"
        )

        print(
            f"Completed new {j:03d}/{len(remaining):03d} | "
            f"analysis_id={aid:03d} | a_R={float(row['a_R']):.3f} | "
            f"{q0_text} | {q4_text} | "
            f"checkpoint rows={len(current)}"
        )

    # -------------------------------------------------------------
    # Current status
    # 检查每个 analysis_id 是否已经拥有完整的 q 网格，以判断是否真正完成。
    # -------------------------------------------------------------
    if checkpoint_file.exists():
        res = pd.read_csv(checkpoint_file)
    else:
        res = pd.DataFrame()

    fully_completed_ids = set()

    if not res.empty:
        want_q = set(np.round(q_relative, 10))

        for aid, g in res.groupby("analysis_id"):
            got_q = set(np.round(g["q_relative"].astype(float), 10))
            if got_q == want_q:
                fully_completed_ids.add(int(aid))

    n_complete = len(fully_completed_ids)
    all_complete = n_complete == len(selected)

    print("\n======================================================")
    print("RUN STATUS")
    print("======================================================")
    print(f"Completed posterior draws: {n_complete}/{len(selected)}")
    print("All complete:", all_complete)
    print("Checkpoint:", checkpoint_file)
    print("Partial summary:", partial_summary_file)

    if not all_complete:
        print(
            "\nThis is a valid checkpoint, NOT the final result yet.\n"
            "Continue with:\n"
            "  python hddm/07_error_cost_320draw.py --resume"
        )
        return

    # -------------------------------------------------------------
    # Finalize only when all 320 posterior draws are present
    # 只有 320 个 posterior draw 全部完成后才生成 final 文件，避免误把 partial 当正式结果。
    # -------------------------------------------------------------
    res = res.sort_values(
        ["analysis_id", "q_relative"]
    ).reset_index(drop=True)

    final_draws_file = out_dir / "error_cost_draws_final.csv"
    final_summary_file = out_dir / "error_cost_summary_final.csv"
    final_probs_file = out_dir / "error_cost_probabilities_final.csv"

    atomic_to_csv(res, final_draws_file)

    summary, probs = build_summaries(res)
    atomic_to_csv(summary, final_summary_file)
    atomic_to_csv(probs, final_probs_file)

    # q=0 reproduction diagnostic
    # 这是最重要的内部校验之一；q=0 应重现原 intended-payoff 的逐 draw a*。
    q0 = res[np.isclose(res["q_relative"], 0.0)].copy()

    validation = {
        "n_q0_draws": int(len(q0)),
        "mean_new_q0_a_star": float(q0["a_star"].mean()),
        "mean_original_q0_a_star": (
            float(q0["original_q0_a_star_intended"].mean())
            if q0["original_q0_a_star_intended"].notna().any()
            else None
        ),
        "mean_abs_difference_from_original": (
            float(q0["q0_abs_difference_from_original"].dropna().mean())
            if q0["q0_abs_difference_from_original"].notna().any()
            else None
        ),
        "max_abs_difference_from_original": (
            float(q0["q0_abs_difference_from_original"].dropna().max())
            if q0["q0_abs_difference_from_original"].notna().any()
            else None
        ),
        "remaining_edge_hits": int(
            res["fine_on_edge_after_all_refinement"].astype(bool).sum()
        ),
    }

    atomic_write_json(
        validation,
        out_dir / "error_cost_validation_final.json",
    )

    # Human-readable GitHub inventory.
    # 生成便于 GitHub 归档和人工核对的文件清单。
    inventory_lines = [
        "# Error-cost sensitivity analysis files",
        "",
        "Code:",
        "- hddm/06_error_cost_pilot.py",
        "- hddm/07_error_cost_320draw.py",
        "",
        "Formal result directory:",
        f"- results/optimality/{args.output_subdir}/",
        "",
        "Key files:",
        "- error_cost_draws_final.csv",
        "- error_cost_summary_final.csv",
        "- error_cost_probabilities_final.csv",
        "- error_cost_validation_final.json",
        "- error_cost_run_config.json",
        "- error_cost_input_manifest.json",
        "- frozen_inputs/selected_posterior_draws.csv",
        "- frozen_inputs/reconstructed_reward_criteria.csv",
        "",
        "Checkpoint retained for audit/resume:",
        "- error_cost_draws_checkpoint.csv",
        "",
        "Primary sensitivity objective:",
        "U(a;q) = [10*P(correct & RT<criterion) - q_points*P(wrong)] / E[T_trial]",
        "",
        "q_relative values:",
        ", ".join(str(x) for x in q_relative),
    ]

    inventory_path = out_dir / "GITHUB_UPLOAD_INVENTORY.md"
    inventory_path.write_text(
        "\n".join(inventory_lines) + "\n",
        encoding="utf-8",
    )

    print("\n======================================================")
    print("FINAL ANALYSIS COMPLETE")
    print("======================================================")
    print("Final draws:", final_draws_file)
    print("Final summary:", final_summary_file)
    print("Final probabilities:", final_probs_file)
    print("Validation:", out_dir / "error_cost_validation_final.json")
    print("GitHub inventory:", inventory_path)

    print("\nq-specific a* posterior summaries:")
    a_summary = summary[summary["metric"] == "a_star"][
        ["q_relative", "q_points", "mean", "q2.5", "q97.5"]
    ]
    print(a_summary.to_string(index=False))

    print("\nq-specific P(a_R > a*):")
    p_show = probs[
        probs["quantity"] == "P(a_R > a_star | sampled posterior draws)"
    ][
        ["q_relative", "q_points", "value", "n_success", "n_draws"]
    ]
    print(p_show.to_string(index=False))


if __name__ == "__main__":
    main()
