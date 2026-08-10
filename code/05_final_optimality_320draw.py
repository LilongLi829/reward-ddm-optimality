#!/usr/bin/env python3
"""
05_final_optimality_320draw.py

Propagate m_va group-level HDDM posterior uncertainty into reward-rate
optimality quantities for Braeutigam Exp. 2 Stroop.

For each selected posterior draw, the script keeps the jointly sampled
reward-condition parameters:
    a_R      = a(1)
    v_RI     = v(1.0)
    v_RC     = v(1.1)
    t        = t

It then reconstructs the empirically verified blockwise reward criterion:
    criterion(i,b) = round(mean RT of ALL no-reward trials in previous
                           Stroop block)
    logged success = RT < criterion

Two payoff rules are optimized:
  logged   : +10 iff RT < criterion
  intended : +10 iff correct AND RT < criterion

A coarse grid is followed by local fine-grid refinement for each posterior
sample. Common random numbers are used across a values within each posterior
sample to smooth numerical comparisons.

This is a posterior-uncertainty analysis of the latent DDM process; the fixed
p_outlier contaminant mixture is not simulated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

try:
    import hddm
except Exception as exc:
    raise RuntimeError(
        "Could not import HDDM. Run inside the same Docker/HDDM environment "
        "used for model fitting."
    ) from exc

# Silence pandas-3 compatibility FutureWarnings emitted inside legacy HDDM
warnings.filterwarnings("ignore", category=FutureWarning)

EXCLUDED = {22, 25, 27}

# Experiment-2 trial timing (seconds)
CUE_S = 0.800
FIXATION_S = 0.400
FEEDBACK_CORRECT_S = 0.750
FEEDBACK_INCORRECT_S = 1.250
ITI_S = 0.300
OVERALL_RT_DEADLINE_S = 3.000
POINTS_PER_SUCCESS = 10.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default="/home/jovyan/project")
    p.add_argument(
        "--posterior-draws", type=int, default=320,
        help="Number of joint posterior draws, stratified across 4 chains (default 320).",
    )
    p.add_argument("--coarse-min", type=float, default=0.15)
    p.add_argument("--coarse-max", type=float, default=0.75)
    p.add_argument("--coarse-step", type=float, default=0.02)
    p.add_argument("--fine-halfwidth", type=float, default=0.03)
    p.add_argument("--fine-step", type=float, default=0.005)
    p.add_argument(
        "--n-sim-coarse", type=int, default=15000,
        help="Simulated trials per congruency and coarse a value (default 15000).",
    )
    p.add_argument(
        "--n-sim-fine", type=int, default=40000,
        help="Simulated trials per congruency and fine a value / a_hat (default 40000).",
    )
    p.add_argument("--seed", type=int, default=20260820)
    p.add_argument(
        "--output-subdir", default="posterior_uncertainty_320draw",
        help="Subdirectory under results/optimality for this final run.",
    )
    p.add_argument("--refine-halfwidth", type=float, default=0.12)
    p.add_argument("--refine-step", type=float, default=0.005)
    p.add_argument("--n-sim-refine", type=int, default=80000)
    p.add_argument("--refine-global-min", type=float, default=0.05)
    p.add_argument("--refine-global-max", type=float, default=1.00)
    p.add_argument(
        "--resume", action="store_true",
        help="Resume from saved selected draws/results if present.",
    )
    return p.parse_args()


def reconstruct_criteria(raw_file: Path) -> tuple[np.ndarray, pd.DataFrame]:
    d = pd.read_csv(raw_file, sep="\t")
    required = {"vpNum", "task", "blockNum", "trialNum", "rt", "reward", "success"}
    missing = required - set(d.columns)
    if missing:
        raise RuntimeError(f"Raw data missing columns: {sorted(missing)}")

    s = d[(d["task"] == "stroop") & (~d["vpNum"].isin(EXCLUDED))].copy()
    first_block = s.groupby("vpNum")["blockNum"].transform("min")
    s["is_formal"] = s["blockNum"] != first_block

    bm = (
        s[s["reward"] == "no_reward"]
        .groupby(["vpNum", "blockNum"], as_index=False)["rt"]
        .mean()
        .rename(columns={"rt": "block_NR_mean_ms"})
        .sort_values(["vpNum", "blockNum"])
    )
    bm["criterion_exact_ms"] = bm.groupby("vpNum")["block_NR_mean_ms"].shift(1)
    bm["criterion_ms"] = np.round(bm["criterion_exact_ms"])

    r = s[s["is_formal"] & (s["reward"] == "reward")].copy()
    r = r.merge(
        bm[["vpNum", "blockNum", "criterion_exact_ms", "criterion_ms"]],
        on=["vpNum", "blockNum"], how="left", validate="many_to_one",
    )
    if r["criterion_ms"].isna().any():
        raise RuntimeError("Missing reconstructed reward criteria.")

    success_bool = r["success"].astype(str).str.lower().eq("true")
    predicted = r["rt"] < r["criterion_ms"]
    mismatches = int((success_bool != predicted).sum())

    print("\n=== REWARD CRITERION RECONSTRUCTION ===")
    print("Formal reward trials:", len(r))
    print("round(previous-block all-NR mean) mismatches:", mismatches)
    if mismatches != 0:
        raise RuntimeError("Reward criterion no longer reproduces logged success exactly.")

    crit = (
        r[["vpNum", "blockNum", "criterion_ms"]]
        .drop_duplicates()
        .sort_values(["vpNum", "blockNum"])
        .reset_index(drop=True)
    )
    return crit["criterion_ms"].to_numpy(float) / 1000.0, crit


def load_combined_traces(formal_root: Path) -> pd.DataFrame:
    required = ["a(1)", "v(1.0)", "v(1.1)", "t"]
    pieces = []

    for chain in range(1, 5):
        path = formal_root / "m_va" / f"chain_{chain:02d}" / "model.hddm"
        if not path.exists():
            raise FileNotFoundError(path)
        print("Loading posterior:", path)
        m = hddm.load(str(path))
        tr = m.get_traces()
        missing = [c for c in required if c not in tr.columns]
        if missing:
            raise RuntimeError(f"Missing posterior nodes in chain {chain}: {missing}")
        x = tr[required].copy().reset_index(drop=True)
        x["chain"] = chain
        x["source_draw"] = np.arange(len(x), dtype=int)
        pieces.append(x)

    post = pd.concat(pieces, ignore_index=True)
    post = post.rename(columns={
        "a(1)": "a_R",
        "v(1.0)": "v_RI",
        "v(1.1)": "v_RC",
    })
    return post


def stratified_select(post: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("--posterior-draws must be positive")
    rng = np.random.default_rng(seed)
    base, rem = divmod(n, 4)
    out = []
    for chain in range(1, 5):
        k = base + (1 if chain <= rem else 0)
        pool = post[post["chain"] == chain]
        if k > len(pool):
            raise ValueError(f"Requested {k} draws from chain {chain}, only {len(pool)} available")
        idx = rng.choice(pool.index.to_numpy(), size=k, replace=False)
        out.append(post.loc[idx])
    sel = pd.concat(out, ignore_index=True)
    # Shuffle while remaining reproducible; each row gets a stable analysis id.
    sel = sel.iloc[rng.permutation(len(sel))].reset_index(drop=True)
    sel.insert(0, "analysis_id", np.arange(1, len(sel) + 1, dtype=int))
    return sel


def simulate_condition(*, a: float, v: float, t: float, n_sim: int,
                       seed: int, criteria_s: np.ndarray) -> dict[str, float]:
    # Common random numbers: reset to same condition-specific seed at every a
    # within a given posterior draw.
    np.random.seed(seed)
    sim = hddm.generate.gen_rts(
        size=n_sim, method="cdf", v=float(v), a=float(a), t=float(t), z=0.5,
        sv=0.0, sz=0.0, st=0.0, range_=(-6, 6), dt=1e-3,
    )

    rt = sim["rt"].to_numpy(dtype=float)
    response = sim["response"].to_numpy(dtype=float)
    responded = rt < OVERALL_RT_DEADLINE_S
    correct = responded & (response == 1.0)
    effective_rt = np.minimum(rt, OVERALL_RT_DEADLINE_S)

    feedback_s = np.where(correct, FEEDBACK_CORRECT_S, FEEDBACK_INCORRECT_S)
    cycle_s = CUE_S + FIXATION_S + effective_rt + feedback_s + ITI_S

    sorted_all_rt = np.sort(rt)
    sorted_correct_rt = np.sort(rt[correct])
    n = float(n_sim)

    p_fast = np.mean(np.searchsorted(sorted_all_rt, criteria_s, side="left") / n)
    p_correct_fast = np.mean(
        np.searchsorted(sorted_correct_rt, criteria_s, side="left") / n
    )

    return {
        "p_fast": float(p_fast),
        "p_correct_fast": float(p_correct_fast),
        "accuracy": float(np.mean(correct)),
        "mean_cycle_s": float(np.mean(cycle_s)),
    }


def evaluate_a(a: float, *, v_RI: float, v_RC: float, t: float,
               n_sim: int, seed: int, criteria_s: np.ndarray) -> dict[str, float]:
    ri = simulate_condition(
        a=a, v=v_RI, t=t, n_sim=n_sim, seed=seed + 101, criteria_s=criteria_s
    )
    rc = simulate_condition(
        a=a, v=v_RC, t=t, n_sim=n_sim, seed=seed + 202, criteria_s=criteria_s
    )
    avg = {k: 0.5 * (ri[k] + rc[k]) for k in ri}

    ep_logged = POINTS_PER_SUCCESS * avg["p_fast"]
    ep_intended = POINTS_PER_SUCCESS * avg["p_correct_fast"]
    return {
        "a": float(a),
        "rr_logged": ep_logged / avg["mean_cycle_s"],
        "rr_intended": ep_intended / avg["mean_cycle_s"],
        "accuracy": avg["accuracy"],
        "mean_cycle_s": avg["mean_cycle_s"],
    }


def grid_values(lo: float, hi: float, step: float) -> np.ndarray:
    return np.arange(lo, hi + step / 2.0, step, dtype=float)


def local_grid(center: float, halfwidth: float, step: float,
               global_lo: float, global_hi: float) -> np.ndarray:
    lo = max(global_lo, center - halfwidth)
    hi = min(global_hi, center + halfwidth)
    return grid_values(lo, hi, step)


def auto_refine_edge_hits(res: pd.DataFrame, selected: pd.DataFrame, *,
                          criteria_s: np.ndarray, args: argparse.Namespace) -> pd.DataFrame:
    """Automatically re-estimate any local fine-grid edge hits.

    Only flagged payoff rules are replaced. The wider refinement uses the same
    posterior draw and common-random-number logic as the main analysis. If an
    optimum still lands on the widened local edge, the window is expanded until
    either it is interior or the global refinement bounds are reached.
    """
    out = res.copy()
    sel = selected.set_index("analysis_id")
    flagged_ids = out.loc[
        out["fine_logged_on_edge"].astype(bool) |
        out["fine_intended_on_edge"].astype(bool),
        "analysis_id"
    ].astype(int).tolist()

    print("\n=== AUTOMATIC EDGE REFINEMENT ===")
    print("Flagged posterior draws:", len(flagged_ids))
    if not flagged_ids:
        return out

    print("Initial wider halfwidth:", args.refine_halfwidth)
    print("Refinement step:", args.refine_step)
    print("n_sim per congruency / a:", args.n_sim_refine)
    print("Global refinement bounds:", args.refine_global_min, args.refine_global_max)

    for j, aid in enumerate(flagged_ids, 1):
        p = sel.loc[aid]
        a_R = float(p["a_R"])
        v_RI = float(p["v_RI"])
        v_RC = float(p["v_RC"])
        t = float(p["t"])
        idx = out.index[out["analysis_id"].astype(int) == aid][0]
        draw_seed = args.seed + aid * 10000

        for rule in ["logged", "intended"]:
            edge_col = f"fine_{rule}_on_edge"
            if not bool(out.loc[idx, edge_col]):
                continue

            center = float(out.loc[idx, f"coarse_peak_{rule}"])
            half = float(args.refine_halfwidth)
            best = None
            grid = None
            edge = True

            for attempt in range(1, 5):
                lo = max(float(args.refine_global_min), center - half)
                hi = min(float(args.refine_global_max), center + half)
                grid = grid_values(lo, hi, float(args.refine_step))
                vals = [
                    evaluate_a(
                        float(a), v_RI=v_RI, v_RC=v_RC, t=t,
                        n_sim=int(args.n_sim_refine),
                        seed=draw_seed + 9000,
                        criteria_s=criteria_s,
                    )
                    for a in grid
                ]
                fg = pd.DataFrame(vals)
                metric = f"rr_{rule}"
                best = fg.loc[fg[metric].idxmax()]
                new_a = float(best["a"])
                edge = bool(
                    np.isclose(new_a, float(grid.min())) or
                    np.isclose(new_a, float(grid.max()))
                )
                if not edge:
                    break
                if (np.isclose(grid.min(), args.refine_global_min) and
                        np.isclose(grid.max(), args.refine_global_max)):
                    break
                half *= 1.75

            at_hat = evaluate_a(
                a_R, v_RI=v_RI, v_RC=v_RC, t=t,
                n_sim=int(args.n_sim_refine),
                seed=draw_seed + 9000,
                criteria_s=criteria_s,
            )
            assert best is not None and grid is not None
            new_a = float(best["a"])
            new_rr = float(best[f"rr_{rule}"])
            rr_hat = float(at_hat[f"rr_{rule}"])

            out.loc[idx, f"a_star_{rule}"] = new_a
            out.loc[idx, f"rr_star_{rule}"] = new_rr
            out.loc[idx, f"rr_hat_{rule}"] = rr_hat
            out.loc[idx, f"efficiency_{rule}"] = rr_hat / new_rr
            out.loc[idx, f"gap_hat_minus_star_{rule}"] = a_R - new_a
            out.loc[idx, edge_col] = edge
            out.loc[idx, f"{rule}_edge_refined"] = True
            out.loc[idx, f"{rule}_refine_grid_min"] = float(grid.min())
            out.loc[idx, f"{rule}_refine_grid_max"] = float(grid.max())

            print(
                f"Refined {j:03d}/{len(flagged_ids):03d} | id={aid:03d} "
                f"{rule}: a*={new_a:.3f} edge={edge} "
                f"grid=[{grid.min():.3f},{grid.max():.3f}]"
            )

    return out


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


def main() -> None:
    args = parse_args()
    root = Path(args.project_root)
    raw_file = root / "data" / "raw" / "RawData_Exp2.txt"
    formal_root = root / "results" / "hddm_formal"
    out = root / "results" / "optimality" / args.output_subdir
    out.mkdir(parents=True, exist_ok=True)

    selected_file = out / "selected_posterior_draws.csv"
    results_file = out / "posterior_optimality_draws.csv"
    summary_file = out / "posterior_optimality_summary.csv"
    probs_file = out / "posterior_optimality_probabilities.csv"
    criteria_file = out / "reconstructed_reward_criteria.csv"

    criteria_s, criteria_table = reconstruct_criteria(raw_file)
    criteria_table.to_csv(criteria_file, index=False)

    if args.resume and selected_file.exists():
        selected = pd.read_csv(selected_file)
        print("\nResuming selected posterior draws from:", selected_file)
    else:
        post = load_combined_traces(formal_root)
        selected = stratified_select(post, args.posterior_draws, args.seed)
        selected.to_csv(selected_file, index=False)
        print("\nSelected posterior draws:", len(selected))
        print(selected.groupby("chain").size().to_string())
        print("Saved:", selected_file)

    done_ids: set[int] = set()
    existing = None
    if args.resume and results_file.exists():
        existing = pd.read_csv(results_file)
        done_ids = set(existing["analysis_id"].astype(int).tolist())
        print("Already completed posterior draws:", len(done_ids))

    coarse = grid_values(args.coarse_min, args.coarse_max, args.coarse_step)
    new_rows = []

    print("\n=== POSTERIOR OPTIMALITY SETTINGS ===")
    print(f"Posterior draws: {len(selected)}")
    print(
        f"Coarse a grid: {coarse[0]:.3f}..{coarse[-1]:.3f} "
        f"by {args.coarse_step:.3f} ({len(coarse)} values)"
    )
    print(
        f"Fine: +/-{args.fine_halfwidth:.3f} around each coarse peak, "
        f"step {args.fine_step:.3f}"
    )
    print("n_sim coarse/fine:", args.n_sim_coarse, "/", args.n_sim_fine)

    for _, row in selected.iterrows():
        aid = int(row["analysis_id"])
        if aid in done_ids:
            continue

        draw_seed = args.seed + aid * 10000
        pars = {
            "a_R": float(row["a_R"]),
            "v_RI": float(row["v_RI"]),
            "v_RC": float(row["v_RC"]),
            "t": float(row["t"]),
        }

        coarse_rows = [
            evaluate_a(
                float(a), v_RI=pars["v_RI"], v_RC=pars["v_RC"], t=pars["t"],
                n_sim=args.n_sim_coarse, seed=draw_seed, criteria_s=criteria_s,
            )
            for a in coarse
        ]
        cg = pd.DataFrame(coarse_rows)
        c_logged = float(cg.loc[cg["rr_logged"].idxmax(), "a"])
        c_intended = float(cg.loc[cg["rr_intended"].idxmax(), "a"])

        fine_logged = local_grid(
            c_logged, args.fine_halfwidth, args.fine_step,
            args.coarse_min, args.coarse_max,
        )
        fine_intended = local_grid(
            c_intended, args.fine_halfwidth, args.fine_step,
            args.coarse_min, args.coarse_max,
        )
        fine = np.unique(np.round(np.concatenate([fine_logged, fine_intended]), 10))

        fine_rows = [
            evaluate_a(
                float(a), v_RI=pars["v_RI"], v_RC=pars["v_RC"], t=pars["t"],
                n_sim=args.n_sim_fine, seed=draw_seed + 5000, criteria_s=criteria_s,
            )
            for a in fine
        ]
        fg = pd.DataFrame(fine_rows)
        b_log = fg.loc[fg["rr_logged"].idxmax()]
        b_int = fg.loc[fg["rr_intended"].idxmax()]

        at_hat = evaluate_a(
            pars["a_R"], v_RI=pars["v_RI"], v_RC=pars["v_RC"], t=pars["t"],
            n_sim=args.n_sim_fine, seed=draw_seed + 5000, criteria_s=criteria_s,
        )

        rec = {
            "analysis_id": aid,
            "chain": int(row["chain"]),
            "source_draw": int(row["source_draw"]),
            **pars,
            "a_star_logged": float(b_log["a"]),
            "rr_star_logged": float(b_log["rr_logged"]),
            "rr_hat_logged": float(at_hat["rr_logged"]),
            "efficiency_logged": float(at_hat["rr_logged"] / b_log["rr_logged"]),
            "gap_hat_minus_star_logged": float(pars["a_R"] - b_log["a"]),
            "a_star_intended": float(b_int["a"]),
            "rr_star_intended": float(b_int["rr_intended"]),
            "rr_hat_intended": float(at_hat["rr_intended"]),
            "efficiency_intended": float(at_hat["rr_intended"] / b_int["rr_intended"]),
            "gap_hat_minus_star_intended": float(pars["a_R"] - b_int["a"]),
            "coarse_peak_logged": c_logged,
            "coarse_peak_intended": c_intended,
            "fine_logged_on_edge": bool(
                np.isclose(float(b_log["a"]), fine_logged.min()) or
                np.isclose(float(b_log["a"]), fine_logged.max())
            ),
            "fine_intended_on_edge": bool(
                np.isclose(float(b_int["a"]), fine_intended.min()) or
                np.isclose(float(b_int["a"]), fine_intended.max())
            ),
        }
        new_rows.append(rec)

        # Checkpoint after every draw.
        current_new = pd.DataFrame(new_rows)
        if existing is not None:
            current = pd.concat([existing, current_new], ignore_index=True)
        else:
            current = current_new
        current = current.sort_values("analysis_id")
        current.to_csv(results_file, index=False)

        print(
            f"Completed {len(done_ids) + len(new_rows):3d}/{len(selected)} | "
            f"id={aid:03d} chain={int(row['chain'])} | "
            f"a*_logged={rec['a_star_logged']:.3f} "
            f"a*_intended={rec['a_star_intended']:.3f} | "
            f"a_R={pars['a_R']:.3f}"
        )

    res = pd.read_csv(results_file).sort_values("analysis_id").reset_index(drop=True)

    # Automatically refine any local fine-grid edge hits for either payoff rule.
    refined = auto_refine_edge_hits(
        res, selected, criteria_s=criteria_s, args=args
    )
    refined_file = out / "posterior_optimality_draws_final.csv"
    refined.to_csv(refined_file, index=False)

    metrics = [
        "a_R",
        "a_star_logged", "gap_hat_minus_star_logged", "efficiency_logged",
        "a_star_intended", "gap_hat_minus_star_intended", "efficiency_intended",
    ]
    summary_rows = []
    for metric in metrics:
        ss = summarize_vector(refined[metric])
        summary_rows.append({"metric": metric, **ss})
    summary = pd.DataFrame(summary_rows)
    summary_file = out / "posterior_optimality_summary_final.csv"
    summary.to_csv(summary_file, index=False)

    probs = pd.DataFrame([
        {
            "quantity": "P(a_R > a*_logged | sampled posterior draws)",
            "value": float((refined["gap_hat_minus_star_logged"] > 0).mean()),
            "n_success": int((refined["gap_hat_minus_star_logged"] > 0).sum()),
            "n_draws": int(len(refined)),
        },
        {
            "quantity": "P(a_R > a*_intended | sampled posterior draws)",
            "value": float((refined["gap_hat_minus_star_intended"] > 0).mean()),
            "n_success": int((refined["gap_hat_minus_star_intended"] > 0).sum()),
            "n_draws": int(len(refined)),
        },
        {
            "quantity": "P(efficiency_intended < 1 | sampled posterior draws)",
            "value": float((refined["efficiency_intended"] < 1).mean()),
            "n_success": int((refined["efficiency_intended"] < 1).sum()),
            "n_draws": int(len(refined)),
        },
        {
            "quantity": "P(efficiency_intended < 0.9 | sampled posterior draws)",
            "value": float((refined["efficiency_intended"] < 0.9).mean()),
            "n_success": int((refined["efficiency_intended"] < 0.9).sum()),
            "n_draws": int(len(refined)),
        },
        {
            "quantity": "P(efficiency_logged < 1 | sampled posterior draws)",
            "value": float((refined["efficiency_logged"] < 1).mean()),
            "n_success": int((refined["efficiency_logged"] < 1).sum()),
            "n_draws": int(len(refined)),
        },
    ])
    probs_file = out / "posterior_optimality_probabilities_final.csv"
    probs.to_csv(probs_file, index=False)

    print("\n============================================================")
    print("FINAL 320-DRAW POSTERIOR OPTIMALITY SUMMARY")
    print("============================================================")
    print("Completed posterior draws:", len(refined))
    print("\nPosterior summaries:")
    print(summary.to_string(index=False))
    print("\nSampled-posterior probabilities / frequencies:")
    print(probs.to_string(index=False))

    print("\nNumerical diagnostics after automatic refinement:")
    print("logged fine-grid edge hits:", int(refined["fine_logged_on_edge"].astype(bool).sum()))
    print("intended fine-grid edge hits:", int(refined["fine_intended_on_edge"].astype(bool).sum()))

    if int(refined["fine_logged_on_edge"].astype(bool).sum()) > 0 or int(refined["fine_intended_on_edge"].astype(bool).sum()) > 0:
        print("WARNING: Some optima still lie on the GLOBAL refinement boundary.")

    print("\nSaved:")
    print(" ", selected_file)
    print(" ", results_file)
    print(" ", refined_file)
    print(" ", summary_file)
    print(" ", probs_file)
    print(" ", criteria_file)
    print("\nFINAL POSTERIOR OPTIMALITY COMPLETE")


if __name__ == "__main__":
    main()
