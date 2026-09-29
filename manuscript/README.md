# Manuscript result audit

`MANUSCRIPT_RESULT_LEDGER.csv` links automated manuscript-facing checks to their study-specific reproduction scripts and archived analysis inputs.

The repository distinguishes quantities recomputed from participant-level frozen inputs from quantities checked against reliable archived summary inputs. This distinction prevents archived historical results from being presented as fresh raw-data/model-fitting reruns.

Several bootstrap confidence intervals for the Hübner, TBE, Evans, and Bogacz analyses require upstream or bootstrap-level inputs that are not redistributed in this minimal public repository. Those intervals should therefore be described as archived analysis results rather than freshly reproduced using this repository. Leng effect confidence intervals and Hübner SATF inferential summaries are directly checked from the archived analysis inputs included here.
