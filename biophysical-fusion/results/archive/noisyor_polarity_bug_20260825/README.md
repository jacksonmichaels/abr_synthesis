# `all_modalities__noisyor`, first run — the polarity bug. Archived 2026-08-26.

Kept as evidence, not as a result. **Macro CV 0.4956** (below chance), min 0.359.

`NoisyOrVariantNet` is monotone increasing in its evidence — every factor is in
(0,1), so adding a variant can only move the output one way. The first version
pointed that at **P(resistant)**, but this project encodes **R=0 / S=1**
(`training/multimodal._metrics`: `n_R = (y == 0).sum()`), so the logit is for the
SUSCEPTIBLE class. A monotone aggregator aimed at the wrong class is
structurally anti-predictive, and resistant isolates carry more variants, so it
came out below chance rather than merely bad.

A compounding second fault: the `-4.0` init on the final MLP layer put
`d/dz logsigmoid(-z) = -sigmoid(z)` at 0.018, attenuating every gradient into
the MLP by ~50x on top of BIG-TB's `lr = exp(-9) = 1.2e-4`. It never escaped its
initialization — train loss 0.3152 -> 0.3023 over 99 epochs, against
`additive`'s 0.2246 -> 0.0875 on the identical cell.

Fixed 2026-08-26 in `models/experimental_models.py`: the product is
P(susceptible), and the init is `-2.0`.
`tests/test_experimental_models.py::_test_noisyor_is_monotone_in_evidence` now
asserts the DIRECTION, not just the monotonicity, so this cannot regress
silently. The cell was resubmitted at the identical protocol; its replacement
lives back under `results/experiments/variant_aggregators_20260825/`.
