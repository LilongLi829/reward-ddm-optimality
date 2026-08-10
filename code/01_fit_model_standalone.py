#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone HDDM fitting script for the Bräutigam Experiment 2 Stroop project.

Target environment:
    HDDM 1.0.1RC
    PyMC 2.3.8

This file intentionally contains the model definitions directly so that it
does NOT import model_factory.py. This avoids local module/import-path issues.

Models
------
m_v:
    v depends on reward x congruency
    a and t estimated, condition-invariant

m_a:
    v depends on congruency
    a depends on reward
    t estimated, condition-invariant

m_va:
    v depends on reward x congruency
    a depends on reward
    t estimated, condition-invariant
"""

from __future__ import print_function

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd
import hddm


VALID_MODELS = ("m_v", "m_a", "m_va")

DEFAULT_DATA = (
    "/home/jovyan/project/data/processed/"
    "Braeutigam_Exp2_Stroop_HDDM.csv"
)

DEFAULT_RESULTS = "/home/jovyan/project/results/hddm"


def validate_data(data):
    required = ["subj_idx", "rt", "response", "reward", "congruency"]

    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(
            "Missing required columns: %s\nAvailable: %s"
            % (missing, list(data.columns))
        )

    if data[required].isnull().any().any():
        bad = data[required].isnull().sum()
        bad = bad[bad > 0]
        raise ValueError("Missing values in required columns:\n%s" % bad)

    responses = sorted(data["response"].unique().tolist())
    if not set(responses).issubset({0, 1}):
        raise ValueError(
            "response must contain only 0/1; found: %s" % responses
        )

    if len(data) != 10651:
        print(
            "WARNING: expected 10651 trials, found %d." % len(data)
        )

    if data["subj_idx"].nunique() != 37:
        print(
            "WARNING: expected 37 participants, found %d."
            % data["subj_idx"].nunique()
        )


def build_model(data, model_name, p_outlier):
    if model_name == "m_v":
        depends_on = {
            "v": ["reward", "congruency"],
        }

    elif model_name == "m_a":
        depends_on = {
            "v": "congruency",
            "a": "reward",
        }

    elif model_name == "m_va":
        depends_on = {
            "v": ["reward", "congruency"],
            "a": "reward",
        }

    else:
        raise ValueError(
            "Unknown model %r. Choose: %s"
            % (model_name, ", ".join(VALID_MODELS))
        )

    print("\n=== MODEL SPECIFICATION ===")
    print("Model:", model_name)
    print("depends_on:", depends_on)
    print("include: ['v', 'a', 't']")
    print("z: fixed at HDDM default 0.5")
    print("p_outlier:", p_outlier)

    model = hddm.HDDM(
        data,
        include=["v", "a", "t"],
        depends_on=depends_on,
        p_outlier=p_outlier,
        bias=False,
    )

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=VALID_MODELS
    )
    parser.add_argument(
        "--chain",
        required=True,
        type=int
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--burn",
        type=int,
        default=300
    )
    parser.add_argument(
        "--thin",
        type=int,
        default=2
    )
    parser.add_argument(
        "--p-outlier",
        type=float,
        default=0.05
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA
    )
    parser.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS
    )

    args = parser.parse_args()

    if args.chain < 1:
        raise ValueError("--chain must be >= 1")
    if args.samples <= 0:
        raise ValueError("--samples must be > 0")
    if args.burn < 0 or args.burn >= args.samples:
        raise ValueError("--burn must satisfy 0 <= burn < samples")
    if args.thin < 1:
        raise ValueError("--thin must be >= 1")

    chain_dir = os.path.join(
        args.results_root,
        args.model,
        "chain_%02d" % args.chain,
    )
    if not os.path.exists(chain_dir):
        os.makedirs(chain_dir)

    trace_path = os.path.join(chain_dir, "traces.db")
    model_path = os.path.join(chain_dir, "model")
    stats_path = os.path.join(chain_dir, "stats.csv")
    dic_path = os.path.join(chain_dir, "dic.txt")
    metadata_path = os.path.join(chain_dir, "run_metadata.json")

    print("=== HDDM RUN ===")
    print("HDDM version:", getattr(hddm, "__version__", "unknown"))
    print("Python:", sys.version.split()[0])
    print("Data:", args.data)
    print("Output:", chain_dir)
    print("Model:", args.model)
    print("Chain:", args.chain)
    print("Samples:", args.samples)
    print("Burn:", args.burn)
    print("Thin:", args.thin)
    print("p_outlier:", args.p_outlier)

    print("\nReading data...")
    data = pd.read_csv(args.data)
    validate_data(data)

    print("Rows:", len(data))
    print("Participants:", data["subj_idx"].nunique())
    print("Response counts:")
    print(data["response"].value_counts().sort_index())
    print("Reward x congruency:")
    print(pd.crosstab(data["reward"], data["congruency"]))

    print("\nBuilding hierarchical model...")
    model = build_model(data, args.model, args.p_outlier)
    print("Model build complete.")

    print("\nFinding starting values...")
    model.find_starting_values()
    print("Starting values complete.")

    print("\nSampling...")
    model.sample(
        args.samples,
        burn=args.burn,
        thin=args.thin,
        dbname=trace_path,
        db="pickle",
    )
    print("Sampling complete.")

    print("\nSaving model...")
    model.save(model_path)
    print("Saved:", model_path)

    print("\nSaving posterior statistics...")
    try:
        stats = model.gen_stats()
        stats.to_csv(stats_path)
        print("Saved:", stats_path)
    except Exception as exc:
        print("WARNING: could not save stats.csv:", repr(exc))

    print("\nSaving DIC...")
    try:
        dic_value = float(model.dic)
        with open(dic_path, "w") as f:
            f.write("%.10f\n" % dic_value)
        print("DIC:", dic_value)
        print("Saved:", dic_path)
    except Exception as exc:
        dic_value = None
        print("WARNING: could not compute/save DIC:", repr(exc))

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "hddm_version": getattr(hddm, "__version__", "unknown"),
        "python_version": sys.version,
        "data": args.data,
        "rows": int(len(data)),
        "participants": int(data["subj_idx"].nunique()),
        "model": args.model,
        "chain": int(args.chain),
        "samples": int(args.samples),
        "burn": int(args.burn),
        "thin": int(args.thin),
        "p_outlier": float(args.p_outlier),
        "dic": dic_value,
        "model_specification": {
            "m_v": "v ~ reward x congruency; a constant; t constant",
            "m_a": "v ~ congruency; a ~ reward; t constant",
            "m_va": "v ~ reward x congruency; a ~ reward; t constant",
        }[args.model],
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved:", metadata_path)

    print("\n=== RUN COMPLETE ===")
    print("Results directory:", chain_dir)


if __name__ == "__main__":
    main()
