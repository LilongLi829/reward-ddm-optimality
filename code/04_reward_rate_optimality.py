#!/usr/bin/env python3
"""
04_reward_rate_optimality.py

Group-level reward-rate optimality analysis for Braeutigam Exp. 2 Stroop.

Uses the selected m_va HDDM model and the empirically reconstructed reward
criterion:
    criterion(i, block) = round(mean RT of ALL no-reward trials in the
                                 previous Stroop block)
    success = RT < criterion

Two payoff definitions are evaluated:
  1) logged   : +10 points when RT < criterion (matches raw log 'success')
  2) intended : +10 points when response is correct AND RT < criterion

The decision threshold a is varied while v and t are held at their fitted
reward-condition group posterior means. Congruency is averaged 50/50, matching
the balanced experimental design.

No p_outlier mixture is simulated: this analysis targets the latent DDM
process rather than contaminant trials.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    import hddm
except Exception as exc:
    raise RuntimeError(
        "Could not import HDDM. Run this script inside the same Docker/HDDM "
        "environment used for model fitting."
    ) from exc


EXCLUDED = {22, 25, 27}

# Experiment-2 trial timing (seconds)
CUE_S = 0.800
FIXATION_S = 0.400
FEEDBACK_CORRECT_S = 0.750
FEEDBACK_INCORRECT_S = 1.250
ITI_S = 0.300
OVERALL_RT_DEADLINE_S = 3.000
POINTS_PER_SUCCESS = 10.0

# Frozen HDDM coding used in this project
REWARD_NR = 0
REWARD_R = 1
CONG_I = 0
CONG_C = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--project-root",
        default="/home/jovyan/project",
        help="Project root inside Docker (default: /home/jovyan/project)",
    )
    p.add_argument("--a-min", type=float, default=0.30)
    p.add_argument("--a-max", type=float, default=2.50)
    p.add_argument("--a-step", type=float, default=0.01)
    p.add_argument(
        "--n-sim",
        type=int,
        default=50000,
        help="Simulated trials per congruency level at each a (default: 50000)",
    )
    p.add_argument("--seed", type=int, default=20260810)
    return p.parse_args()


def load_four_chain_group_means(formal_root: Path) -> dict[str, float]:
    """Average equal-length chain posterior means for required group nodes."""
    rows = []
    required = ["a(1)", "v(1.0)", "v(1.1)", "t"]

    for chain in range(1, 5):
        f = formal_root / "m_va" / f"chain_{chain:02d}" / "stats.csv"
        if not f.exists():
            raise FileNotFoundError(f"Missing stats file: {f}")
        s = pd.read_csv(f, index_col=0)
        missing = [x for x in required if x not in s.index]
        if missing:
            raise RuntimeError(f"Missing nodes in {f}: {missing}")
        if "mean" not in s.columns:
            raise RuntimeError(f"No 'mean' column in {f}")
        rows.append({name: float(s.loc[name, "mean"]) for name in required})

    x = pd.DataFrame(rows)
    out = x.mean(axis=0).to_dict()

    # Mapping from the frozen preprocessing:
    # v(1.0) = reward + incongruent; v(1.1) = reward + congruent.
    out["a_R"] = out.pop("a(1)")
    out["v_RI"] = out.pop("v(1.0)")
    out["v_RC"] = out.pop("v(1.1)")
    return out


def reconstruct_criteria(raw_file: Path) -> tuple[np.ndarray, pd.DataFrame]:
    d = pd.read_csv(raw_file, sep="\t")

    required = {
        "vpNum", "task", "blockNum", "trialNum", "rt", "reward", "success"
    }
    missing = required - set(d.columns)
    if missing:
        raise RuntimeError(f"Raw data missing columns: {sorted(missing)}")

    s = d[(d["task"] == "stroop") & (~d["vpNum"].isin(EXCLUDED))].copy()

    # First Stroop block per participant is practice in Exp. 2.
    first_block = s.groupby("vpNum")["blockNum"].transform("min")
    s["is_formal"] = s["blockNum"] != first_block

    # Mean RT of ALL no-reward trials in each Stroop block.
    bm = (
        s[s["reward"] == "no_reward"]
        .groupby(["vpNum", "blockNum"], as_index=False)["rt"]
        .mean()
        .rename(columns={"rt": "block_NR_mean_ms"})
        .sort_values(["vpNum", "blockNum"])
    )
    bm["criterion_exact_ms"] = (
        bm.groupby("vpNum")["block_NR_mean_ms"].shift(1)
    )
    # Empirically verified against all 5,328 formal reward trials.
    bm["criterion_ms"] = np.round(bm["criterion_exact_ms"])

    r = s[s["is_formal"] & (s["reward"] == "reward")].copy()
    r = r.merge(
        bm[["vpNum", "blockNum", "criterion_exact_ms", "criterion_ms"]],
        on=["vpNum", "blockNum"],
        how="left",
        validate="many_to_one",
    )

    if r["criterion_ms"].isna().any():
        bad = r[r["criterion_ms"].isna()][["vpNum", "blockNum"]].drop_duplicates()
        raise RuntimeError(f"Missing reconstructed criteria:\n{bad}")

    success_bool = r["success"].astype(str).str.lower().eq("true")
    predicted = r["rt"] < r["criterion_ms"]
    mismatches = int((success_bool != predicted).sum())

    print("\n=== REWARD CRITERION RECONSTRUCTION ===")
    print("Formal reward trials:", len(r))
    print("Participant-block criteria:", r[["vpNum", "blockNum"]].drop_duplicates().shape[0])
    print("round(previous-block all-NR mean) mismatches:", mismatches)

    if mismatches != 0:
        bad = r.loc[
            success_bool != predicted,
            ["vpNum", "blockNum", "trialNum", "rt", "criterion_exact_ms",
             "criterion_ms", "success"],
        ]
        print(bad.head(20).to_string(index=False))
        raise RuntimeError("Reward criterion no longer reproduces success exactly.")

    # One criterion per formal participant x Stroop block.
    crit = (
        r[["vpNum", "blockNum", "criterion_ms"]]
        .drop_duplicates()
        .sort_values(["vpNum", "blockNum"])
        .reset_index(drop=True)
    )

    criteria_s = crit["criterion_ms"].to_numpy(dtype=float) / 1000.0
    return criteria_s, crit


def simulate_condition(
    *,
    a: float,
    v: float,
    t: float,
    n_sim: int,
    seed: int,
    criteria_s: np.ndarray,
) -> dict[str, float]:
    """Simulate one reward x congruency DDM condition and summarize utility."""
    # Reset seed at every a for common-random-number smoothing across the a grid.
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

    # Apply the experiment's overall 3-s response deadline to the simulated path.
    responded = rt < OVERALL_RT_DEADLINE_S
    correct = responded & (response == 1.0)
    effective_rt = np.minimum(rt, OVERALL_RT_DEADLINE_S)

    feedback_s = np.where(
        correct,
        FEEDBACK_CORRECT_S,
        FEEDBACK_INCORRECT_S,
    )
    cycle_s = CUE_S + FIXATION_S + effective_rt + feedback_s + ITI_S

    # Deadline criterion c is always < 3 s here, but retain all simulations in the
    # denominator so probabilities are unconditional per presented reward trial.
    sorted_all_rt = np.sort(rt)
    sorted_correct_rt = np.sort(rt[correct])

    n = float(n_sim)
    p_fast_by_criterion = (
        np.searchsorted(sorted_all_rt, criteria_s, side="left") / n
    )
    p_correct_fast_by_criterion = (
        np.searchsorted(sorted_correct_rt, criteria_s, side="left") / n
    )

    return {
        "p_fast": float(np.mean(p_fast_by_criterion)),
        "p_correct_fast": float(np.mean(p_correct_fast_by_criterion)),
        "accuracy": float(np.mean(correct)),
        "mean_cycle_s": float(np.mean(cycle_s)),
        "mean_effective_rt_s": float(np.mean(effective_rt)),
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
    # Equal congruency weighting: Experiment-2 reward trials are balanced C/I.
    ri = simulate_condition(
        a=a, v=v_RI, t=t, n_sim=n_sim, seed=seed + 101,
        criteria_s=criteria_s,
    )
    rc = simulate_condition(
        a=a, v=v_RC, t=t, n_sim=n_sim, seed=seed + 202,
        criteria_s=criteria_s,
    )

    avg = {
        key: 0.5 * (ri[key] + rc[key])
        for key in ri.keys()
    }

    expected_points_logged = POINTS_PER_SUCCESS * avg["p_fast"]
    expected_points_intended = POINTS_PER_SUCCESS * avg["p_correct_fast"]

    return {
        "a": float(a),
        "p_fast_logged": avg["p_fast"],
        "p_correct_fast_intended": avg["p_correct_fast"],
        "accuracy": avg["accuracy"],
        "mean_effective_rt_ms": avg["mean_effective_rt_s"] * 1000.0,
        "mean_cycle_s": avg["mean_cycle_s"],
        "expected_points_logged": expected_points_logged,
        "expected_points_intended": expected_points_intended,
        "rr_logged_points_per_s": expected_points_logged / avg["mean_cycle_s"],
        "rr_intended_points_per_s": expected_points_intended / avg["mean_cycle_s"],
    }


def main() -> None:
    args = parse_args()
    root = Path(args.project_root)
    raw_file = root / "data" / "raw" / "RawData_Exp2.txt"
    formal_root = root / "results" / "hddm_formal"
    out = root / "results" / "optimality"
    out.mkdir(parents=True, exist_ok=True)

    if not raw_file.exists():
        raise FileNotFoundError(raw_file)

    params = load_four_chain_group_means(formal_root)
    criteria_s, criteria_table = reconstruct_criteria(raw_file)

    print("\n=== FITTED REWARD-CONDITION DDM PARAMETERS ===")
    print("Coding: reward 1 = reward; congruency 0 = incongruent, 1 = congruent")
    print(f"a_R observed posterior mean = {params['a_R']:.6f}")
    print(f"v_RI = {params['v_RI']:.6f}")
    print(f"v_RC = {params['v_RC']:.6f}")
    print(f"t     = {params['t']:.6f} s")
    print(f"criterion mean = {criteria_s.mean()*1000:.3f} ms")
    print(f"criterion min  = {criteria_s.min()*1000:.1f} ms")
    print(f"criterion max  = {criteria_s.max()*1000:.1f} ms")
    print(f"n criteria     = {len(criteria_s)}")

    a_grid = np.arange(
        args.a_min,
        args.a_max + args.a_step / 2.0,
        args.a_step,
        dtype=float,
    )

    print("\n=== REWARD-RATE GRID ===")
    print(
        f"a: {a_grid[0]:.2f} to {a_grid[-1]:.2f} by {args.a_step:.3f} "
        f"({len(a_grid)} values)"
    )
    print("Simulations per a x congruency:", args.n_sim)
    print("Payoffs: logged speed-only + intended correct-and-fast")

    rows = []
    for i, a in enumerate(a_grid, start=1):
        rows.append(
            evaluate_a(
                float(a),
                v_RI=params["v_RI"],
                v_RC=params["v_RC"],
                t=params["t"],
                n_sim=args.n_sim,
                seed=args.seed,
                criteria_s=criteria_s,
            )
        )
        if i == 1 or i % 10 == 0 or i == len(a_grid):
            print(f"Completed {i:3d}/{len(a_grid)}  a={a:.3f}")

    curve = pd.DataFrame(rows)

    best_logged = curve.loc[curve["rr_logged_points_per_s"].idxmax()].copy()
    best_intended = curve.loc[curve["rr_intended_points_per_s"].idxmax()].copy()

    # Evaluate the fitted threshold at its exact posterior mean, rather than only
    # at the nearest 0.01 grid location.
    at_hat = pd.Series(
        evaluate_a(
            params["a_R"],
            v_RI=params["v_RI"],
            v_RC=params["v_RC"],
            t=params["t"],
            n_sim=args.n_sim,
            seed=args.seed,
            criteria_s=criteria_s,
        )
    )

    summary = pd.DataFrame([
        {
            "payoff_rule": "logged_speed_only",
            "a_star": best_logged["a"],
            "rr_star_points_per_s": best_logged["rr_logged_points_per_s"],
            "a_hat_R": params["a_R"],
            "rr_at_a_hat_points_per_s": at_hat["rr_logged_points_per_s"],
            "a_hat_minus_a_star": params["a_R"] - best_logged["a"],
            "rr_efficiency_hat_over_star": (
                at_hat["rr_logged_points_per_s"] /
                best_logged["rr_logged_points_per_s"]
            ),
        },
        {
            "payoff_rule": "intended_correct_and_fast",
            "a_star": best_intended["a"],
            "rr_star_points_per_s": best_intended["rr_intended_points_per_s"],
            "a_hat_R": params["a_R"],
            "rr_at_a_hat_points_per_s": at_hat["rr_intended_points_per_s"],
            "a_hat_minus_a_star": params["a_R"] - best_intended["a"],
            "rr_efficiency_hat_over_star": (
                at_hat["rr_intended_points_per_s"] /
                best_intended["rr_intended_points_per_s"]
            ),
        },
    ])

    curve_file = out / "reward_rate_curve_group.csv"
    summary_file = out / "reward_rate_optimality_summary.csv"
    criteria_file = out / "reconstructed_reward_criteria.csv"
    params_file = out / "reward_ddm_group_parameters.csv"

    curve.to_csv(curve_file, index=False)
    summary.to_csv(summary_file, index=False)
    criteria_table.to_csv(criteria_file, index=False)
    pd.DataFrame([params]).to_csv(params_file, index=False)

    # Plot if matplotlib is available; failure to plot should not fail analysis.
    plot_file = out / "reward_rate_curves.png"
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(curve["a"], curve["rr_logged_points_per_s"], label="Logged payoff")
        ax.plot(curve["a"], curve["rr_intended_points_per_s"], label="Intended payoff")
        ax.axvline(params["a_R"], linestyle="--", linewidth=1, label="Observed a_R")
        ax.axvline(best_logged["a"], linestyle=":", linewidth=1, label="a* logged")
        ax.axvline(best_intended["a"], linestyle="-.", linewidth=1, label="a* intended")
        ax.set_xlabel("Decision threshold a")
        ax.set_ylabel("Expected points per second")
        ax.set_title("Reward-rate optimality: Experiment 2 Stroop")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_file, dpi=180)
        plt.close(fig)
    except Exception as exc:
        print("Plot warning:", repr(exc))

    print("\n============================================================")
    print("REWARD-RATE OPTIMALITY RESULTS")
    print("============================================================")
    print(f"Observed a_R (m_va, 4-chain group posterior mean): {params['a_R']:.6f}")

    print("\nLOGGED PAYOFF: +10 iff RT < reconstructed criterion")
    print(f"a*_logged = {best_logged['a']:.3f}")
    print(f"RR(a*)    = {best_logged['rr_logged_points_per_s']:.6f} points/s")
    print(f"RR(a_hat) = {at_hat['rr_logged_points_per_s']:.6f} points/s")
    print(
        "Efficiency RR(a_hat)/RR(a*) = "
        f"{at_hat['rr_logged_points_per_s']/best_logged['rr_logged_points_per_s']:.4f}"
    )
    print(f"a_hat - a* = {params['a_R'] - best_logged['a']:.6f}")

    print("\nINTENDED PAYOFF: +10 iff correct AND RT < criterion")
    print(f"a*_intended = {best_intended['a']:.3f}")
    print(f"RR(a*)      = {best_intended['rr_intended_points_per_s']:.6f} points/s")
    print(f"RR(a_hat)   = {at_hat['rr_intended_points_per_s']:.6f} points/s")
    print(
        "Efficiency RR(a_hat)/RR(a*) = "
        f"{at_hat['rr_intended_points_per_s']/best_intended['rr_intended_points_per_s']:.4f}"
    )
    print(f"a_hat - a* = {params['a_R'] - best_intended['a']:.6f}")

    boundary_flags = []
    if best_logged["a"] in (a_grid[0], a_grid[-1]):
        boundary_flags.append("logged optimum is on the search boundary")
    if best_intended["a"] in (a_grid[0], a_grid[-1]):
        boundary_flags.append("intended optimum is on the search boundary")
    if boundary_flags:
        print("\nWARNING:")
        for msg in boundary_flags:
            print(" -", msg)
        print("Expand --a-min/--a-max before interpreting that optimum.")

    print("\nSaved:")
    print(" ", curve_file)
    print(" ", summary_file)
    print(" ", criteria_file)
    print(" ", params_file)
    if plot_file.exists():
        print(" ", plot_file)

    print("\nOPTIMALITY GRID COMPLETE")


if __name__ == "__main__":
    main()
