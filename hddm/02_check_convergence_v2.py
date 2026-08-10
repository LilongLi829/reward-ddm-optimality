#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
02_check_convergence.py

Formal multi-chain convergence diagnostics for the three HDDM models:
    m_v, m_a, m_va

Expected directory:
    /home/jovyan/project/results/hddm_formal/
        m_v/chain_01/model.hddm ... chain_04/model.hddm
        m_a/chain_01/model.hddm ... chain_04/model.hddm
        m_va/chain_01/model.hddm ... chain_04/model.hddm

Outputs:
    results/hddm_formal/diagnostics/
        rhat_all.csv
        rhat_summary.csv
        dic_by_chain.csv
        convergence_report.txt

This script does NOT refit any model and does NOT alter any saved chain.
"""

from __future__ import print_function

import os
import sys
import math
import json
import numpy as np
import pandas as pd
import hddm

PROJECT = "/home/jovyan/project"
RESULT_ROOT = os.path.join(PROJECT, "results", "hddm_formal")
OUT_DIR = os.path.join(RESULT_ROOT, "diagnostics")

MODELS = ["m_v", "m_a", "m_va"]
CHAINS = [1, 2, 3, 4]

os.makedirs(OUT_DIR, exist_ok=True)


def load_models(model_name):
    models = []
    for chain in CHAINS:
        chain_dir = os.path.join(
            RESULT_ROOT, model_name, "chain_{:02d}".format(chain)
        )

        # Formal fitting script saved the serialized HDDM object as model.hddm.
        # Keep a fallback to "model" for compatibility with earlier runs.
        candidates = [
            os.path.join(chain_dir, "model.hddm"),
            os.path.join(chain_dir, "model"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)

        if path is None:
            raise IOError(
                "Missing saved model in {}. Checked: {}".format(
                    chain_dir, ", ".join(candidates)
                )
            )

        print("Loading:", path)
        models.append(hddm.load(path))
    return models


def manual_rhat_from_traces(models):
    """
    Classical Gelman-Rubin R-hat computed from the saved post-burn/thinned traces.
    Used as a fallback if kabuki's helper fails in the installed environment.
    """
    trace_frames = []
    for i, m in enumerate(models, 1):
        df = m.get_traces()
        if df is None or len(df) == 0:
            raise RuntimeError("Chain {} has no trace samples.".format(i))
        trace_frames.append(df)

    common_cols = set(trace_frames[0].columns)
    for df in trace_frames[1:]:
        common_cols &= set(df.columns)
    common_cols = sorted(common_cols)

    out = {}
    for col in common_cols:
        arrays = []
        good = True
        for df in trace_frames:
            x = pd.to_numeric(df[col], errors="coerce").dropna().values.astype(float)
            if len(x) < 2:
                good = False
                break
            arrays.append(x)

        if not good:
            continue

        n = min(len(x) for x in arrays)
        if n < 2:
            continue

        X = np.vstack([x[:n] for x in arrays])  # m chains x n draws
        m_chains = X.shape[0]

        chain_means = X.mean(axis=1)
        chain_vars = X.var(axis=1, ddof=1)

        W = chain_vars.mean()
        B = n * chain_means.var(ddof=1)

        if not np.isfinite(W) or W <= 0:
            # If every chain is exactly constant at the same value, define Rhat as 1.
            if np.allclose(X, X[0, 0], equal_nan=False):
                rhat = 1.0
            else:
                rhat = np.nan
        else:
            var_hat = ((n - 1.0) / n) * W + (B / n)
            rhat = math.sqrt(var_hat / W)

        out[col] = float(rhat)

    return out


def get_rhat(models):
    # Prefer HDDM/Kabuki's own implementation when available.
    try:
        from kabuki.analyze import gelman_rubin
        print("Computing R-hat with kabuki.analyze.gelman_rubin ...")
        gr = gelman_rubin(models)
        if not isinstance(gr, dict) or len(gr) == 0:
            raise RuntimeError("gelman_rubin returned no values")
        method = "kabuki.analyze.gelman_rubin"
        return gr, method
    except Exception as e:
        print("Kabuki Gelman-Rubin failed; using manual classical R-hat.")
        print("Reason:", repr(e))
        gr = manual_rhat_from_traces(models)
        method = "manual_classical_gelman_rubin"
        return gr, method


def read_dic(model_name, chain):
    path = os.path.join(
        RESULT_ROOT, model_name, "chain_{:02d}".format(chain), "dic.txt"
    )
    if not os.path.exists(path):
        return np.nan
    with open(path, "r") as f:
        txt = f.read().strip()
    # dic.txt may contain either just the number or a label + number.
    tokens = txt.replace(":", " ").split()
    for token in reversed(tokens):
        try:
            return float(token)
        except Exception:
            pass
    return np.nan


all_rhat_rows = []
summary_rows = []
dic_rows = []
report_lines = []

print("HDDM version:", getattr(hddm, "__version__", "unknown"))
print("Formal results root:", RESULT_ROOT)
print("Diagnostics output:", OUT_DIR)
print("")

for model_name in MODELS:
    print("=" * 72)
    print("MODEL:", model_name)
    print("=" * 72)

    models = load_models(model_name)
    gr, method = get_rhat(models)

    vals = []
    for param, value in gr.items():
        try:
            value = float(value)
        except Exception:
            continue
        vals.append(value)
        all_rhat_rows.append({
            "model": model_name,
            "parameter": str(param),
            "rhat": value,
            "method": method,
        })

    finite_vals = np.array([v for v in vals if np.isfinite(v)], dtype=float)

    if len(finite_vals) == 0:
        max_rhat = np.nan
        n_gt_101 = n_gt_105 = n_gt_110 = 0
        status = "NO_RHAT_VALUES"
    else:
        max_rhat = float(np.max(finite_vals))
        n_gt_101 = int(np.sum(finite_vals > 1.01))
        n_gt_105 = int(np.sum(finite_vals > 1.05))
        n_gt_110 = int(np.sum(finite_vals > 1.10))

        if n_gt_110 > 0:
            status = "CHECK_CONVERGENCE"
        elif n_gt_105 > 0:
            status = "PASS_CONVENTIONAL_BUT_REVIEW"
        else:
            status = "GOOD"

    summary_rows.append({
        "model": model_name,
        "n_parameters": len(finite_vals),
        "max_rhat": max_rhat,
        "n_rhat_gt_1.01": n_gt_101,
        "n_rhat_gt_1.05": n_gt_105,
        "n_rhat_gt_1.10": n_gt_110,
        "status": status,
        "method": method,
    })

    # DIC stability across the four independently run chains.
    dics = []
    for chain in CHAINS:
        dic = read_dic(model_name, chain)
        dics.append(dic)
        dic_rows.append({
            "model": model_name,
            "chain": chain,
            "dic": dic,
        })

    dics_finite = np.array([x for x in dics if np.isfinite(x)], dtype=float)
    dic_mean = float(np.mean(dics_finite)) if len(dics_finite) else np.nan
    dic_sd = float(np.std(dics_finite, ddof=1)) if len(dics_finite) > 1 else np.nan
    dic_range = (
        float(np.max(dics_finite) - np.min(dics_finite))
        if len(dics_finite) else np.nan
    )

    # Show worst parameters.
    worst = sorted(
        [(str(k), float(v)) for k, v in gr.items()
         if np.isfinite(float(v))],
        key=lambda kv: kv[1],
        reverse=True
    )[:15]

    print("")
    print("R-hat method:", method)
    print("Parameters checked:", len(finite_vals))
    print("Maximum R-hat:", "{:.6f}".format(max_rhat) if np.isfinite(max_rhat) else "NA")
    print("R-hat > 1.01:", n_gt_101)
    print("R-hat > 1.05:", n_gt_105)
    print("R-hat > 1.10:", n_gt_110)
    print("Status:", status)
    print("")
    print("Worst 15 R-hat values:")
    for p, r in worst:
        print("  {:60s} {:.6f}".format(p[:60], r))

    print("")
    print("DIC by chain:", [round(x, 6) if np.isfinite(x) else None for x in dics])
    print("DIC mean:", round(dic_mean, 6) if np.isfinite(dic_mean) else "NA")
    print("DIC SD:", round(dic_sd, 6) if np.isfinite(dic_sd) else "NA")
    print("DIC range:", round(dic_range, 6) if np.isfinite(dic_range) else "NA")
    print("")

    report_lines.extend([
        "MODEL: {}".format(model_name),
        "R-hat method: {}".format(method),
        "Parameters checked: {}".format(len(finite_vals)),
        "Maximum R-hat: {}".format(max_rhat),
        "R-hat > 1.01: {}".format(n_gt_101),
        "R-hat > 1.05: {}".format(n_gt_105),
        "R-hat > 1.10: {}".format(n_gt_110),
        "Status: {}".format(status),
        "DIC by chain: {}".format(dics),
        "DIC mean: {}".format(dic_mean),
        "DIC SD: {}".format(dic_sd),
        "DIC range: {}".format(dic_range),
        "Worst 15 R-hat values:",
    ])
    report_lines.extend(["  {} = {}".format(p, r) for p, r in worst])
    report_lines.append("")


rhat_df = pd.DataFrame(all_rhat_rows).sort_values(
    ["model", "rhat"], ascending=[True, False]
)
summary_df = pd.DataFrame(summary_rows)
dic_df = pd.DataFrame(dic_rows)

rhat_path = os.path.join(OUT_DIR, "rhat_all.csv")
summary_path = os.path.join(OUT_DIR, "rhat_summary.csv")
dic_path = os.path.join(OUT_DIR, "dic_by_chain.csv")
report_path = os.path.join(OUT_DIR, "convergence_report.txt")

rhat_df.to_csv(rhat_path, index=False)
summary_df.to_csv(summary_path, index=False)
dic_df.to_csv(dic_path, index=False)

with open(report_path, "w") as f:
    f.write("\n".join(report_lines))

print("=" * 72)
print("OVERALL SUMMARY")
print("=" * 72)
print(summary_df.to_string(index=False))
print("")
print("Saved:")
print(" ", rhat_path)
print(" ", summary_path)
print(" ", dic_path)
print(" ", report_path)
print("")
print("Interpretation rule used here:")
print("  R-hat > 1.10  -> convergence requires attention")
print("  1.05 < R-hat <= 1.10 -> passes the common 1.10 heuristic, but review")
print("  max R-hat <= 1.05 -> good for proceeding to trace/PPC checks")
