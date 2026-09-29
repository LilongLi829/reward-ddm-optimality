# Code annotation standard

All manuscript-facing analysis code should make the following information discoverable without reading the paper first:

1. **Purpose / evidence role** — core predictive, specificity, boundary, robustness, or archived.
2. **Input** — exact expected file(s), participant/sample scope, and preprocessing assumptions.
3. **Output** — filenames and the manuscript quantity produced.
4. **Frozen choices** — train/test split, block range, optimization grid/bounds, objective definition, and exclusions.
5. **Randomness** — seed and resampling count when stochastic procedures are used.
6. **Leakage boundary** — whether q/model parameters are train-only, held out, or same-sample.
7. **Interpretation boundary** — what the script does *not* establish.
8. **Provenance** — whether the implementation is original to this reanalysis, recovered from an archived project, or follows original-author code.

Comments should explain non-obvious scientific choices rather than narrating obvious syntax line by line. Archived/exploratory scripts must not be presented as manuscript-generating code.
