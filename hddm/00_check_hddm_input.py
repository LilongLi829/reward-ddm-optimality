#!/usr/bin/env python3
# 00_check_hddm_input.py
# Sanity checks only. Does not fit a model.

from pathlib import Path
import sys
import pandas as pd
import hddm
import pymc

PROJECT = Path("/home/jovyan/project")
DATA = PROJECT / "data" / "processed" / "Braeutigam_Exp2_Stroop_HDDM.csv"

print("HDDM version:", getattr(hddm, "__version__", "unknown"))
print("PyMC version:", getattr(pymc, "__version__", "unknown"))
print("Reading:", DATA)

if not DATA.exists():
    raise FileNotFoundError("HDDM input not found: %s" % DATA)

d = pd.read_csv(str(DATA))

required = ["subj_idx", "rt", "response", "reward", "congruency"]
missing = [c for c in required if c not in d.columns]
if missing:
    raise ValueError("Missing required columns: %s" % missing)

print("\n=== SHAPE ===")
print("Rows:", len(d))
print("Participants:", d["subj_idx"].nunique())

print("\n=== COLUMN TYPES ===")
print(d[required].dtypes)

print("\n=== MISSING VALUES ===")
print(d[required].isnull().sum())

print("\n=== RESPONSE ===")
print(d["response"].value_counts().sort_index())

print("\n=== REWARD x CONGRUENCY ===")
print(pd.crosstab(d["reward"], d["congruency"]))

print("\n=== RT (seconds) ===")
print(d["rt"].describe())
print("RT < .200 s:", int((d["rt"] < .200).sum()))
print("RT >= 3.000 s:", int((d["rt"] >= 3.000).sum()))

print("\n=== PER-SUBJECT TRIAL COUNTS ===")
counts = d.groupby("subj_idx").size()
print(counts.describe())
print("Subjects not at 288 trials because of excluded no-response trials:")
print(counts[counts != 288])

# Hard replication checks.
assert len(d) == 10651, "Expected 10651 HDDM trials"
assert d["subj_idx"].nunique() == 37, "Expected 37 participants"
assert set(d["response"].unique()) <= {0, 1}, "response must be 0/1"
assert set(d["reward"].unique()) == {0, 1}, "reward must be 0/1"
assert set(d["congruency"].unique()) == {0, 1}, "congruency must be 0/1"
assert int((d["rt"] < .200).sum()) == 7, "Expected 7 RTs < 200 ms"
assert int((d["rt"] >= 3.000).sum()) == 0, "No 3000-ms no-response trials should remain"

print("\nALL HDDM INPUT CHECKS PASSED.")
