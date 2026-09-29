# Hübner — cross-participant predictive evidence

**Role in manuscript:** core predictive evidence plus a model-independent SATF check.

The archived participant-level LOPO table contains the observed threshold, the experimenter-defined M0 prediction, the Mq prediction, the training-only q estimate, and participant-level absolute errors. The SATF table contains the frozen summary values used in the manuscript.

Run `python scripts/reproduce_hubner.py` from this study directory. This release reproduces/checks manuscript-facing summaries from the archived analysis stage; it does not redistribute the upstream trial-level raw dataset.

Important interpretation: the held-out participant's own processing parameters enter the prediction, so this is cross-participant q generalization, not a claim of full new-person behavioral prediction.
