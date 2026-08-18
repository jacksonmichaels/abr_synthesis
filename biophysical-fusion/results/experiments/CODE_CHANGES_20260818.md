# Code changes behind `transformer_run` — 2026-08-18

What had to change before a transformer-vs-CNN comparison was possible at all.
Everything here is additive: **all defaults are unchanged**, and
`tests/test_transformer_encoder.py` asserts that explicitly (a `None`, an empty
dict, and an explicit `TRANSFORMER_DEFAULTS` must all give the identical model),
so `full_run` / `full_run_v2` / `alllocus_run_v2` checkpoints stay loadable and
their numbers stay reproducible.

## 1. `--arch mdcnn` can take a transformer (`models/net.py`)

It previously ignored the encoder choice entirely and printed
`encoders: n/a (mdcnn)` — its topology *was* the conv stack.

New `MDCNNTransformerTrunk`, a drop-in sibling of `MDCNNTrunk`: same
`(B, n_loci, C, L) -> (B, out_features)` contract, so `MDCNNNet`'s
channel-height grouping, zero-padding and dense head are untouched.

The design point: MD-CNN's layer 1 is already a `Conv2d(n_loci, 64, (channels,
12))` over locus-as-channel input, and making its stride equal its kernel width
along the position axis turns the identical op into a ViT patch embedding. So one
`Conv2d(n_loci, d_model, (channels, patch), stride=(1, patch))` both mixes every
locus at once — the property that makes this MD-CNN rather than late fusion — and
tokenizes the sequence. `+ learned pos -> TransformerEncoder -> mean-pool`
replaces the `Conv1d/pool` stack and the flatten.

Stated cost: the flatten is gone, so absolute position is coarsened to a pooled
summary — the same trade SetFusionNet makes.

Selected with `MDCNNNet(encoder="transformer")` via `MDCNN_TRUNKS`; `"cnn"`
remains the default and is asserted to build `MDCNNTrunk`.

**Uniform only.** mdcnn trunks group by channel height and a trunk can span
modalities (`dna` and `regulatory` are both 5-channel), so a per-modality mix has
no well-defined meaning. `_build_model` raises on a mix rather than half-applying
it; `--default-encoder` is the flag to use.

## 2. The transformer encoder is tunable (`models/net.py`)

`d_model=64, nhead=4, layers=2, dim_ff=128, patch=9, dropout=0.1` were hardcoded
in `TransformerEncoder` and reachable from nothing.

- New `TRANSFORMER_DEFAULTS`, the single place those values are written down.
- New `make_encoder(kind, C, L, transformer=None)`, which forwards overrides
  **only** to the transformer — `CNNEncoder` does not accept them, and a test
  asserts they are never passed to it.
- `MultiModalNet`, `MultiDrugNet` and `CisFusionNet` all gained a `transformer`
  argument.

This exists because the two encoders are not remotely parameter-comparable at the
defaults: `CNNEncoder` flattens (out_features `32 * L/9`), `TransformerEncoder`
mean-pools to `d_model`. A ~30x gap — see `transformer_run/README.md` for how the
sweep sizes around it.

## 3. Plumbing

- `training/multimodal.py`, `training/multidrug.py`: `transformer=` on
  `run_modal_cv` / `run_multidrug_cv` and `_build_model`, normalized the same way
  the setfusion knobs are (only values differing from the defaults are kept).
  Recorded in each result JSON under `transformer`, non-null whenever a
  transformer is actually in the model.
- `training/checkpoint.py`: `model_config` records it; `build_model_from_config`
  rebuilds with it. Read with `.get`, so a config written before the key existed
  still rebuilds at the defaults — tested both ways, including a weight-loading
  round-trip through a real serialize/parse cycle.
- `scripts/run_experiment.py`, `scripts/run_multidrug.py`: `--tf-d-model`,
  `--tf-nhead`, `--tf-layers`, `--tf-dim-ff`, `--tf-patch`, `--tf-dropout`. They
  **error** if passed without a transformer encoder selected, following the rule
  the setfusion flags set: silently ignoring a capacity flag would make a sweep
  arm look like it ran when it was really the control under a different folder
  name.
- `scripts/sbatch_all_runs.py`: passes the six `--tf-*` flags through, and gained
  `--default-encoder` (it previously had no way to set an encoder except a
  per-experiment `encoders` dict naming one modality at a time — so no way to
  make a whole cell uniformly transformer, which is the only form mdcnn accepts).

## 4. One reporting fix

`run_experiment.py` and `training/multimodal.py` both hardcoded
`encoders: n/a (mdcnn)`. True while the conv trunk was the only option; now it
would hide a transformer run behind a label saying the encoder did not matter.
They report the actual kind (`transformer for every mdcnn trunk`).

## Verification

- `tests/test_transformer_encoder.py` — **13/13**. Covers default-inertness, that
  a requested mdcnn transformer is really built (and is not the conv trunk's
  size), that a mixed request raises, that **every locus still reaches the
  output** (the MD-CNN property), that the knobs move the parameter count, that
  the transformer is length-agnostic where the CNN is not, and checkpoint
  round-trips with and without the new key.
- Pre-existing suites unchanged: `test_baseline_alignment` 8/8,
  `test_checkpoint` 34/34, `test_cisfusion` 16/16, `test_setfusion` **21/22**
  (the one failure predates all of this — `SetFusionNet` defaults drifted from
  the `full_run` configuration; whether the test or the defaults should move is a
  research call).
- End-to-end synthetic runs on all three architectures with
  `--default-encoder transformer`, checked for the recorded `transformer` config
  and a non-CNN parameter count.
