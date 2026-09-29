# Reproducibility levels

This repository avoids using the word *reproducible* as if all checks were equivalent.

## Level 1 — manuscript-number verification
`python run_all.py --mode verify`

This recomputes/checks the headline statistics from the frozen final participant-level/summary outputs. It is the release gate for consistency between the manuscript-facing result files and the reported values.

## Level 2 — study-level analysis rerun
Some study packages contain runnable analysis pipelines. Their local README files identify the entry point, inputs, outputs, and limitations. Because the five datasets were originally analyzed with heterogeneous software and archival structures, there is no claim that a single new wrapper can safely replace all original/frozen study pipelines.

## Level 3 — raw-source reconstruction
Raw-to-result reconstruction may require downloading third-party public datasets from their authoritative repositories. This release does not mirror third-party raw data when redistribution permission has not been explicitly confirmed.

The authoritative source, acquisition route, and integrity information are documented in `DATA_ACQUISITION.md`.

## Scientific release rule
Frozen final outputs are never silently overwritten by exploratory or newly rerun outputs. A changed analysis must receive a new version and be reconciled against `RESULT_MAP.md`.
