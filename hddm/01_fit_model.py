#!/usr/bin/env python3
# 01_fit_model.py
# HDDM 1.0.x / PyMC2-compatible sequential-chain runner.
#
# Example pipeline test:
# python hddm/01_fit_model.py --model m_v --chain 1 --samples 1000 --burn 300 --thin 2

from __future__ import print_function
from pathlib import Path
import argparse
import json
import time

import hddm

from model_factory import build_model, VALID_MODELS

PROJECT = Path("/home/jovyan/project")
DATA = PROJECT / "data" / "processed" / "Braeutigam_Exp2_Stroop_HDDM.csv"
RESULTS = PROJECT / "results" / "hddm"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=VALID_MODELS)
    parser.add_argument("--chain", required=True, type=int)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--burn", type=int, default=300)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--p-outlier", type=float, default=0.05)
    parser.add_argument(
        "--skip-starting-values",
        action="store_true",
        help="Skip find_starting_values(); use only for debugging."
    )
    args = parser.parse_args()

    if args.samples <= args.burn:
        raise ValueError("--samples must be greater than --burn")
    if args.thin < 1:
        raise ValueError("--thin must be >= 1")
    if args.chain < 1:
        raise ValueError("--chain must be >= 1")

    out_dir = RESULTS / args.model / ("chain_%02d" % args.chain)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("HDDM version:", getattr(hddm, "__version__", "unknown"))
    print("Data:", DATA)
    print("Output:", out_dir)
    print("Model:", args.model)
    print("Chain:", args.chain)
    print("Samples:", args.samples)
    print("Burn:", args.burn)
    print("Thin:", args.thin)
    print("p_outlier:", args.p_outlier)

    data = hddm.load_csv(str(DATA))

    # Fail before building a large hierarchical model if the wrong CSV is read.
    assert len(data) == 10651
    assert data["subj_idx"].nunique() == 37

    print("\nBuilding hierarchical model...")
    model = build_model(data, args.model, p_outlier=args.p_outlier)

    if not args.skip_starting_values:
        print("\nFinding starting values...")
        model.find_starting_values()

    db_file = out_dir / "traces.db"
    model_file = out_dir / "model"
    stats_file = out_dir / "stats.csv"
    metadata_file = out_dir / "run_metadata.json"
    dic_file = out_dir / "dic.txt"

    metadata = {
        "model": args.model,
        "chain": args.chain,
        "samples": args.samples,
        "burn": args.burn,
        "thin": args.thin,
        "p_outlier": args.p_outlier,
        "n_trials": int(len(data)),
        "n_participants": int(data["subj_idx"].nunique()),
        "hddm_version": str(getattr(hddm, "__version__", "unknown"))
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nSampling...")
    t0 = time.time()
    model.sample(
        args.samples,
        burn=args.burn,
        thin=args.thin,
        dbname=str(db_file),
        db="pickle"
    )
    elapsed = time.time() - t0

    print("\nSaving model and summaries...")
    model.save(str(model_file))
    stats = model.gen_stats()
    stats.to_csv(str(stats_file))

    with open(str(dic_file), "w") as f:
        f.write("%.12f\n" % float(model.dic))

    metadata["elapsed_seconds"] = elapsed
    metadata["dic"] = float(model.dic)
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== COMPLETE ===")
    print("DIC:", model.dic)
    print("Elapsed seconds:", round(elapsed, 1))
    print("Model:", model_file)
    print("Trace DB:", db_file)
    print("Stats:", stats_file)
    print("Metadata:", metadata_file)

if __name__ == "__main__":
    main()
