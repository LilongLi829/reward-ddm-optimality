# Error avoidance in speeded decision making

Reproducible analysis repository for the manuscript by Lilong Li and Chuan-Peng Hu.

The repository is organized study by study. Each dataset has its own analysis input, reproduction script, result checks, figures, and methodological notes.

## Reproduce the reported analyses

Python 3 with the packages listed in `requirements.txt` is required.

```bash
python run_all.py
```

The command executes all five study-specific reproduction scripts and writes the combined verification summary to `verification/summary.csv`.

Individual analyses can also be run directly from their study directories:

```bash
python studies/01_hubner/scripts/reproduce_hubner.py
python studies/02_tbe_exp3/scripts/reproduce_tbe_exp3.py
python studies/03_evans/scripts/reproduce_evans.py
python studies/04_bogacz/scripts/reproduce_bogacz.py
python studies/05_leng/scripts/reproduce_leng.py
```

## Repository structure

- `studies/01_hubner/`: cross-participant predictive analysis and SATF robustness analysis.
- `studies/02_tbe_exp3/`: Experiment 3 temporal held-out validation (N = 58).
- `studies/03_evans/`: specificity analysis.
- `studies/04_bogacz/`: cross-environment boundary analysis.
- `studies/05_leng/`: DDM mechanism and value-mapping analyses.
- `manuscript/`: result ledger linking reported statistics to study-specific reproduction scripts.
- `docs/`: data provenance, acquisition, redistribution, and reproducibility documentation.
- `verification/`: combined verification output.

## Reproducibility scope

This repository supports reproduction and independent checking of the manuscript-facing analyses from the earliest reliable archived analysis stage for each dataset. It does not claim that every historical original-author raw-data/model-fitting pipeline can be rerun from the materials redistributed here.

Third-party raw data and original-author source archives are not redistributed when their file-level redistribution terms are unclear. `docs/DATA_ACQUISITION.md` identifies authoritative upstream sources.

For the Bogacz analysis, reproduction begins from frozen participant-by-held-out-condition predictions because the exact historical generator of the archived q values is not independently reconstructed. For the Evans analysis, reproduction begins from recovered posterior-derived participant inputs rather than a fresh rerun of the historical multi-chain posterior. These boundaries are documented to distinguish manuscript-result reproduction from historical pipeline reconstruction.

## Citation

Repository authors are listed in `CITATION.cff`. Cite the original studies when using their data or methods. DOI and source information are provided in `docs/DATA_ACQUISITION.md`.

## License

No repository-wide open-source license is asserted in this release. Third-party materials remain governed by their upstream terms.
