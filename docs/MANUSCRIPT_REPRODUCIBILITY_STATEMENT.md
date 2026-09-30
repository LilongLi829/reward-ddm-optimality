# Recommended reproducibility statement

All manuscript-facing numerical checks and all seven manuscript figures can be regenerated or independently checked from the archived analysis inputs and code distributed with this repository. Running `python run_all.py` executes all five study-specific reproduction scripts, rebuilds the combined manuscript result ledger, and regenerates Figures 1–7.

Because the analyses reuse third-party public datasets and, in some cases, recovered historical model outputs, the earliest reproducible stage differs across datasets. The repository does not equate manuscript-result reproduction with a fresh rerun of every original-author raw-data/model-fitting pipeline.

For Bogacz, reproduction begins from frozen participant-by-condition predictions because the exact generator of the historical reanalysis q values is not preserved. For Evans, reproduction begins from recovered posterior-derived participant inputs rather than a fresh rerun of the historical multi-chain posterior. These boundaries are documented and do not prevent regeneration of the manuscript-facing analyses from the archived inputs.

For participant bootstrap intervals in Hübner, TBE, Evans, and Bogacz, the release uses a fixed seed (20260929) and 10,000 resamples. The final manuscript should report these deterministic release values.
