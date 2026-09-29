# Public release decision

Version 3.1.0 is deliberately a minimal manuscript-result reproduction package.

Third-party raw/source archives were removed from the distributable repository. The included CSV inputs are frozen/derived analysis inputs required by the manuscript verifier. Their role and earliest reproducible stage are documented rather than presented as original raw data.

Bogacz's official Oxford dataset page currently identifies the dataset terms as CC BY-SA 4.0; nevertheless, the raw dataset is not required for this minimal public release and is not bundled here. Other upstream datasets should be obtained from their official repositories using `DATA_ACQUISITION.md`.

Release acceptance criterion: `python run_all.py --mode manuscript` must return 58/58 PASS.
