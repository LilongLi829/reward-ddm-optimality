#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03_ppc.py

Posterior predictive checks (PPC) for the formal HDDM models.

The script:
1) loads already-fitted model.hddm files (NO refitting);
2) generates posterior-predictive RT/response data separately for
   participant x reward x congruency cells;
3) summarizes predictive accuracy and RT distributions;
4) checks whether observed values fall inside the 95% posterior-predictive interval;
5) summarizes the observed reward/congruency RT effects and their predictive distributions.

Recommended first test:
    python hddm/03_ppc.py --model m_v --chains 1 --samples-per-chain 10

Formal run after the test succeeds:
    python hddm/03_ppc.py --model all --chains 1,2,3,4 --samples-per-chain 100

This script does not alter the saved MCMC chains.
"""

from __future__ import print_function

import os
import gc
import argparse
import numpy as np
import pandas as pd
import hddm

PROJECT = "/home/jovyan/project"
RESULT_ROOT = os.path.join(PROJECT, "results", "hddm_formal")
PPC_ROOT = os.path.join(RESULT_ROOT, "ppc")

VALID_MODELS = ["m_v", "m_a", "m_va"]
QUANTILES = [0.10, 0.30, 0.50, 0.70, 0.90]


def post_pred_gen_compat(
    model,
    groupby=None,
    samples=500,
    append_data=False,
    progress_bar=True,
):
    """
    Compatibility wrapper for Kabuki/HDDM posterior predictive generation.

    The installed Kabuki implementation finishes with:
        pd.concat(results, names=["node"])

    With the newer pandas version in this Docker image, that final concat can
    raise:
        ValueError: Length of names must match number of levels in MultiIndex

    The simulation itself has already succeeded at that point. This wrapper
    follows Kabuki's own algorithm but concatenates first without `names=`,
    then names the MultiIndex levels explicitly.
    """
    from kabuki.analyze import _post_pred_generate

    results = {}
    if groupby is None:
        iter_data = (
            (name, model.data.iloc[obs["node"].value.index])
            for name, obs in model.iter_observeds()
        )
    else:
        iter_data = model.data.groupby(groupby)

    total = len(model.get_observeds()) if groupby is None else None
    done = 0

    for name, data in iter_data:
        node = model.get_data_nodes(data.index)

        if node is None or not hasattr(node, "random"):
            continue

        if groupby is not None:
            new_name = str(node)
        else:
            new_name = name

        datasets = _post_pred_generate(
            node,
            samples=samples,
            data=data,
            append_data=append_data,
        )

        # This inner concat works in the installed environment; the failure
        # occurs only in Kabuki's final outer concat.
        results[new_name] = pd.concat(
            datasets,
            keys=list(range(len(datasets))),
            names=["sample"],
        )

        done += 1
        if progress_bar and (done == 1 or done % 20 == 0):
            if total is None:
                print("  posterior-predictive groups completed:", done)
            else:
                print("  posterior-predictive groups completed: {}/{}".format(done, total))

    if len(results) == 0:
        raise RuntimeError("No posterior predictive datasets were generated.")

    # Key compatibility fix: do not pass names=["node"] into pd.concat.
    out = pd.concat(
        list(results.values()),
        keys=list(results.keys()),
    )

    # Name levels after construction instead.
    names = list(out.index.names)
    if len(names) >= 1:
        names[0] = "node"
    if len(names) >= 2:
        names[1] = "sample"
    if len(names) >= 3 and names[2] is None:
        names[2] = "trial"
    out.index = out.index.set_names(names)

    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="all",
        choices=["all"] + VALID_MODELS,
        help="One model or all three."
    )
    p.add_argument(
        "--chains",
        default="1,2,3,4",
        help="Comma-separated chain numbers, e.g. 1 or 1,2,3,4."
    )
    p.add_argument(
        "--samples-per-chain",
        type=int,
        default=100,
        help="Posterior predictive samples generated per observed node in each chain."
    )
    return p.parse_args()


def load_model(model_name, chain):
    chain_dir = os.path.join(
        RESULT_ROOT, model_name, "chain_{:02d}".format(chain)
    )
    candidates = [
        os.path.join(chain_dir, "model.hddm"),
        os.path.join(chain_dir, "model"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise IOError("No saved HDDM model found in {}".format(chain_dir))
    print("Loading:", path)
    return hddm.load(path)


def detect_sim_columns(df):
    # With append_data=True, Kabuki joins generated data to the original data
    # using lsuffix="_sampled" for colliding columns.
    if "rt_sampled" in df.columns:
        rt_col = "rt_sampled"
    elif "rt" in df.columns:
        rt_col = "rt"
    else:
        raise KeyError("Could not find simulated RT column. Columns: {}".format(
            df.columns.tolist()
        ))

    if "response_sampled" in df.columns:
        resp_col = "response_sampled"
    elif "response" in df.columns:
        resp_col = "response"
    else:
        # HDDM posterior predictive RTs use sign to indicate boundary;
        # this fallback is only used if response is not returned.
        resp_col = None

    return rt_col, resp_col


def summarize_condition_rows(df, sample_id_cols):
    rows = []
    group_cols = sample_id_cols + ["reward", "congruency"]

    for keys, g in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        keymap = dict(zip(group_cols, keys))

        rt = np.abs(pd.to_numeric(g["_sim_rt"], errors="coerce").values.astype(float))
        response = pd.to_numeric(g["_sim_response"], errors="coerce").values.astype(float)

        valid = np.isfinite(rt) & np.isfinite(response)
        rt = rt[valid]
        response = response[valid]

        correct_rt = rt[response == 1]
        error_rt = rt[response == 0]

        row = dict(keymap)
        row["n_trials"] = int(len(rt))
        row["accuracy"] = float(np.mean(response == 1)) if len(rt) else np.nan
        row["correct_mean_rt_ms"] = (
            float(np.mean(correct_rt) * 1000.0) if len(correct_rt) else np.nan
        )
        row["error_mean_rt_ms"] = (
            float(np.mean(error_rt) * 1000.0) if len(error_rt) else np.nan
        )

        for q in QUANTILES:
            qname = int(round(q * 100))
            row["correct_q{:02d}_ms".format(qname)] = (
                float(np.quantile(correct_rt, q) * 1000.0)
                if len(correct_rt) else np.nan
            )
            row["error_q{:02d}_ms".format(qname)] = (
                float(np.quantile(error_rt, q) * 1000.0)
                if len(error_rt) else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_observed(data):
    d = data.copy()
    d["_sim_rt"] = pd.to_numeric(d["rt"], errors="coerce")
    d["_sim_response"] = pd.to_numeric(d["response"], errors="coerce")
    obs = summarize_condition_rows(d, sample_id_cols=[])
    return obs


def participant_marginal_effects(df, sample_id_cols):
    """
    Compute participant-marginal correct-RT effects, matching the logic used
    in the behavior reproduction:
      reward effect      = mean RT(no reward) - mean RT(reward)
      congruency effect  = mean RT(incongruent) - mean RT(congruent)
    First compute within participant, then average participants.
    """
    x = df.copy()
    x = x[pd.to_numeric(x["_sim_response"], errors="coerce") == 1].copy()
    x["_rt_ms"] = np.abs(pd.to_numeric(x["_sim_rt"], errors="coerce")) * 1000.0

    # Reward effect within participant
    reward_group = sample_id_cols + ["subj_idx", "reward"]
    rmeans = x.groupby(reward_group)["_rt_ms"].mean().unstack("reward")

    # Congruency effect within participant
    cong_group = sample_id_cols + ["subj_idx", "congruency"]
    cmeans = x.groupby(cong_group)["_rt_ms"].mean().unstack("congruency")

    # Expected coding: reward 0=NR, 1=R; congruency 0=C, 1=I.
    if 0 not in rmeans.columns or 1 not in rmeans.columns:
        raise ValueError("Reward coding is not 0/1 in PPC data.")
    if 0 not in cmeans.columns or 1 not in cmeans.columns:
        raise ValueError("Congruency coding is not 0/1 in PPC data.")

    r_eff = (rmeans[0] - rmeans[1]).rename("reward_rt_effect_ms")
    c_eff = (cmeans[1] - cmeans[0]).rename("congruency_rt_effect_ms")

    # Average participant effects within each predictive replicate.
    if sample_id_cols:
        r_out = r_eff.reset_index().groupby(sample_id_cols)["reward_rt_effect_ms"].mean()
        c_out = c_eff.reset_index().groupby(sample_id_cols)["congruency_rt_effect_ms"].mean()
        out = pd.concat([r_out, c_out], axis=1).reset_index()
    else:
        out = pd.DataFrame([{
            "reward_rt_effect_ms": float(r_eff.mean()),
            "congruency_rt_effect_ms": float(c_eff.mean()),
        }])

    return out


def compare_predictive_to_observed(obs_cond, pred_cond, model_name):
    metric_cols = [
        c for c in pred_cond.columns
        if c not in ["model", "chain", "sample", "reward", "congruency"]
    ]
    metric_cols = [
        c for c in metric_cols
        if c != "n_trials"
    ]

    rows = []
    for _, obsrow in obs_cond.iterrows():
        r = obsrow["reward"]
        c = obsrow["congruency"]
        pred_cell = pred_cond[
            (pred_cond["reward"] == r) &
            (pred_cond["congruency"] == c)
        ]

        for metric in metric_cols:
            if metric not in obsrow.index:
                continue
            observed = obsrow[metric]
            vals = pd.to_numeric(pred_cell[metric], errors="coerce").dropna().values
            if len(vals) == 0 or not np.isfinite(observed):
                continue

            lo, med, hi = np.quantile(vals, [0.025, 0.50, 0.975])
            rows.append({
                "model": model_name,
                "reward": int(r),
                "congruency": int(c),
                "metric": metric,
                "observed": float(observed),
                "pred_mean": float(np.mean(vals)),
                "pred_median": float(med),
                "pred_2.5": float(lo),
                "pred_97.5": float(hi),
                "observed_in_95": bool(lo <= observed <= hi),
                "abs_error_from_pred_mean": float(abs(observed - np.mean(vals))),
                "n_predictive_replicates": int(len(vals)),
            })

    return pd.DataFrame(rows)


def compare_effects(obs_eff, pred_eff, model_name):
    rows = []
    for metric in ["reward_rt_effect_ms", "congruency_rt_effect_ms"]:
        observed = float(obs_eff.iloc[0][metric])
        vals = pd.to_numeric(pred_eff[metric], errors="coerce").dropna().values
        lo, med, hi = np.quantile(vals, [0.025, 0.50, 0.975])
        rows.append({
            "model": model_name,
            "metric": metric,
            "observed": observed,
            "pred_mean": float(np.mean(vals)),
            "pred_median": float(med),
            "pred_2.5": float(lo),
            "pred_97.5": float(hi),
            "observed_in_95": bool(lo <= observed <= hi),
            "abs_error_from_pred_mean": float(abs(observed - np.mean(vals))),
            "n_predictive_replicates": int(len(vals)),
        })
    return pd.DataFrame(rows)


def run_model(model_name, chains, samples_per_chain):
    print("")
    print("=" * 78)
    print("PPC MODEL:", model_name)
    print("=" * 78)

    model_out = os.path.join(PPC_ROOT, model_name)
    os.makedirs(model_out, exist_ok=True)

    pred_condition_parts = []
    pred_effect_parts = []
    observed_condition = None
    observed_effects = None

    for chain in chains:
        print("")
        print("--- {} chain {:02d} ---".format(model_name, chain))
        model = load_model(model_name, chain)

        # Observed summaries only need to be calculated once.
        if observed_condition is None:
            observed_condition = summarize_observed(model.data)
            d_obs = model.data.copy()
            d_obs["_sim_rt"] = pd.to_numeric(d_obs["rt"], errors="coerce")
            d_obs["_sim_response"] = pd.to_numeric(d_obs["response"], errors="coerce")
            observed_effects = participant_marginal_effects(
                d_obs, sample_id_cols=[]
            )

        print(
            "Generating posterior predictive data:",
            samples_per_chain,
            "samples per participant x reward x congruency node"
        )

        ppc = post_pred_gen_compat(
            model,
            groupby=["subj_idx", "reward", "congruency"],
            samples=samples_per_chain,
            append_data=True,
            progress_bar=True,
        )

        flat = ppc.reset_index()
        rt_col, resp_col = detect_sim_columns(flat)

        flat["_sim_rt"] = pd.to_numeric(flat[rt_col], errors="coerce")
        if resp_col is not None:
            flat["_sim_response"] = pd.to_numeric(flat[resp_col], errors="coerce")
        else:
            flat["_sim_response"] = (flat["_sim_rt"] >= 0).astype(int)

        required = ["sample", "subj_idx", "reward", "congruency"]
        missing = [x for x in required if x not in flat.columns]
        if missing:
            raise KeyError(
                "PPC output is missing {}. Columns are: {}".format(
                    missing, flat.columns.tolist()
                )
            )

        flat["chain"] = int(chain)

        cond = summarize_condition_rows(
            flat, sample_id_cols=["chain", "sample"]
        )
        cond.insert(0, "model", model_name)
        pred_condition_parts.append(cond)

        eff = participant_marginal_effects(
            flat, sample_id_cols=["chain", "sample"]
        )
        eff.insert(0, "model", model_name)
        pred_effect_parts.append(eff)

        print(
            "Chain {:02d}: generated {} predictive rows; summarized {} replicates.".format(
                chain, len(flat), len(cond)
            )
        )

        del ppc, flat, model
        gc.collect()

    pred_condition = pd.concat(pred_condition_parts, ignore_index=True)
    pred_effects = pd.concat(pred_effect_parts, ignore_index=True)

    observed_condition.insert(0, "model", model_name)
    observed_effects.insert(0, "model", model_name)

    condition_compare = compare_predictive_to_observed(
        observed_condition.drop(columns=["model"]),
        pred_condition,
        model_name
    )
    effect_compare = compare_effects(
        observed_effects.drop(columns=["model"]),
        pred_effects,
        model_name
    )

    # Save model-specific PPC results.
    paths = {
        "observed_condition": os.path.join(model_out, "observed_condition_summary.csv"),
        "predictive_condition": os.path.join(model_out, "predictive_condition_replicates.csv"),
        "condition_compare": os.path.join(model_out, "condition_ppc_comparison.csv"),
        "observed_effects": os.path.join(model_out, "observed_behavioral_effects.csv"),
        "predictive_effects": os.path.join(model_out, "predictive_effect_replicates.csv"),
        "effect_compare": os.path.join(model_out, "effect_ppc_comparison.csv"),
    }

    observed_condition.to_csv(paths["observed_condition"], index=False)
    pred_condition.to_csv(paths["predictive_condition"], index=False)
    condition_compare.to_csv(paths["condition_compare"], index=False)
    observed_effects.to_csv(paths["observed_effects"], index=False)
    pred_effects.to_csv(paths["predictive_effects"], index=False)
    effect_compare.to_csv(paths["effect_compare"], index=False)

    print("")
    print("Behavioral-effect PPC:")
    print(effect_compare.to_string(index=False))
    print("")
    print("Condition-level coverage:")
    cov = condition_compare.groupby("metric")["observed_in_95"].agg(["sum", "count"])
    cov["coverage"] = cov["sum"] / cov["count"]
    print(cov.to_string())
    print("")
    print("Saved PPC results to:", model_out)

    return condition_compare, effect_compare


def main():
    args = parse_args()
    chains = [int(x.strip()) for x in args.chains.split(",") if x.strip()]
    models = VALID_MODELS if args.model == "all" else [args.model]

    if args.samples_per_chain < 1:
        raise ValueError("--samples-per-chain must be >= 1")

    os.makedirs(PPC_ROOT, exist_ok=True)

    print("HDDM version:", getattr(hddm, "__version__", "unknown"))
    print("Models:", models)
    print("Chains:", chains)
    print("Samples per chain:", args.samples_per_chain)
    print("PPC root:", PPC_ROOT)

    all_condition = []
    all_effect = []

    for model_name in models:
        c, e = run_model(model_name, chains, args.samples_per_chain)
        all_condition.append(c)
        all_effect.append(e)

    all_condition_df = pd.concat(all_condition, ignore_index=True)
    all_effect_df = pd.concat(all_effect, ignore_index=True)

    all_condition_path = os.path.join(PPC_ROOT, "all_models_condition_ppc_comparison.csv")
    all_effect_path = os.path.join(PPC_ROOT, "all_models_effect_ppc_comparison.csv")

    all_condition_df.to_csv(all_condition_path, index=False)
    all_effect_df.to_csv(all_effect_path, index=False)

    print("")
    print("=" * 78)
    print("PPC COMPLETE")
    print("=" * 78)
    print("Saved:")
    print(" ", all_condition_path)
    print(" ", all_effect_path)
    print("")
    print("Key effect comparison:")
    print(all_effect_df.to_string(index=False))


if __name__ == "__main__":
    main()
