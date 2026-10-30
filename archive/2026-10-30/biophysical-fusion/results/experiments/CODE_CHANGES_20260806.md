# Code changes, 2026-08-06 — what changed and why

Written for whoever picks this up next. These changes were made to enable the
`joint_convergence/` and `joint_capacity/` runs submitted the same day. They came
out of an audit of `full_run/` (240 jobs, 2026-08-04) that asked two questions:
why is multi-drug performance lower than single-drug relative to its baseline,
and why does the regulatory modality underperform.

**Every new knob defaults to the value that produced `full_run`.** An unchanged
invocation of `sbatch_all_runs.py` still emits byte-identical commands, and
`MultiDrugNet(specs, drugs)` still builds a bit-identical model (verified: same
`state_dict` keys, same tensors under a fixed seed, same `Adam` at `exp(-9)`).
So `full_run` remains the valid control for everything below.

---

## 1. Model weights are now saved. They never were before.

**This was the critical finding.** `EarlyStopper.best_state` snapshotted the best
epoch into an in-memory dict, `restore()` loaded it back, and the process exited.
Nothing was ever written to disk — `scripts/trace_models.py:27` said so outright
("we never checkpoint"), and `results/experiments/` contained only json/png/csv.

Consequence: **every TEST metric in `full_run` was produced by a model that no
longer exists.** ~240 SLURM jobs of compute left numbers that cannot be
re-scored, recalibrated, inspected for attention/attribution, or deployed.

### What now happens

New module `training/checkpoint.py`. Weights go to the lab's shared large-storage
volume, *not* to `results/` (which is git-ignored workspace storage, and a full
grid of checkpoints is tens of GB):

```
/project/pi_mfiterau_umass_edu/abr_model_weights/{run_name}/{stem}/
    config.json     schema v1 — everything needed to rebuild the model
    isolates.txt    the exact isolate row order the model was fit on
    fold{k}.pt      state_dict per saved fold
```

`stem` is `{DRUG}__{tag}` (single-drug) or `multidrug__{tag}` (joint), matching
the json filenames in the run folder. The path constant is
`bigtb_ref.MODEL_WEIGHTS_DIR`; override with `--weights-dir` or
`$ABR_MODEL_WEIGHTS_DIR`. Each results folder also gets a
`weights_location.json` breadcrumb, and each result json gains a `weights_dir`
field, so a run and its weights can always be walked between in either direction.

`--save-weights {best,all,none}`, default **best** — the fold whose model is
scored on TEST, i.e. the one the reported numbers come from. `all` is ~5x the
bytes. Budget: one fold per cell across a full 240-run grid is ~5.1 GB. Note the
shared volume was **95% full (1.1 TB free)** when this was added, which is why
`best` is the default and not `all`.

### config.json carries what a state_dict cannot

1. **model** — arch, per-block `(name, modality, locus, channels, length,
   channel_names)` *in order*, encoder assignment, head shape, and `drug_names`
   in label-column order so `logits[:, i]` is attributable. Block ORDER alone
   decides what `forward(xs)` means; a bare `.pt` is unusable without it.
2. **data** — directories, resolved locus and regulatory-region lists in load
   order, and the loader switches (`per_modality_branch`, `all_regulatory`,
   `extra_loci`). This is what rebuilds the *inputs* the weights expect.
3. **split** — the test isolate IDs **verbatim**, not just "seed 42", so
   re-evaluation is immune to scikit-learn changing its shuffling.
4. **env** — git commit, torch version, argv.

```python
from training.checkpoint import load_model
model, cfg = load_model("joint_convergence/a0_control_multidrug_dna__late_fusion",
                        "multidrug__dna")
cfg["model"]["drug_names"]        # which logit column is which drug
```

### It is tested, not asserted

`tests/test_checkpoint.py` (30 checks) rebuilds from `config.json` alone — the
original model object is never reused — loads the weights, and requires
**bit-identical predictions**, for all four architectures, single-drug and joint,
and with the new capacity knobs on. End-to-end on the fixture data: a saved run
was reloaded in a fresh process, its TEST isolates re-located by ID from the
config, and it reproduced the reported TEST macro-AUC to 1e-12 and every
per-drug TEST AUC exactly.

A full or unwritable weights volume logs a warning and returns `None` rather
than raising — a 13-hour finished run must not be destroyed at the last step.

---

## 2. `--all-regulatory` is now reachable from `sbatch_all_runs.py`

The flag already existed on both entry points; the submission generator never
emitted it, so **no run in `full_run` ever used it**. By default regulatory
regions are intersected with the loaded coding loci
(`datasets/loader.py:180-181`, `datasets/multidrug.py:182-183`), which discards
most of the modality:

| drug | WHO regions | loaded | dropped (all present on disk) |
|---|---|---|---|
| ISONIAZID | 14 | **2** | **ahpC**, ndh, mshA, hadA, Rv1258c, … |
| PYRAZINAMIDE | 8 | **1** | clpC1, panD, rpsA, Rv1258c, sigE |
| KANAMYCIN | 7 | **1** | **eis**, whiB7, ccsA, bacA |
| ETHAMBUTOL | 10 | **2** | **ubiA**, embR |
| RIFAMPICIN | 11 | **2** | rpoA, mtrA/B, Rv2477c |

All 48 promoter FASTAs exist in `REAL_REGULATORY_DIR` with complete
17,943-isolate coverage. What survives the filter is mostly the promoter of the
CDS the DNA branch already models (rpoB promoter, gyrA promoter, pncA promoter),
which carries almost no resistance variation; what is dropped is the
mechanistically relevant set. The per-drug gains track mechanism, not modality:
ETO +0.084 and INH +0.03 (the only drugs whose surviving window — `inhA`, the
fabG1–inhA operon promoter carrying c-15t — *is* the mechanism), against PZA
−0.040 for late_fusion, where the lone pncA promoter is pure added parameters.

Not used by either run submitted today — it is the recommended **next**
submission, gated on `joint_convergence`'s answer (see that folder's README).

---

## 3. MDCNN trunk grouping (`--mdcnn-trunk-per-modality`)

`MDCNNNet` grouped blocks by channel count alone. `dna` and `regulatory` are
both 5-channel (A,C,T,G,gap), so a dna+regulatory run put all 16 promoter
windows in the **DNA** trunk, whose position axis is the longest CDS — rpoC at
4,066 bp. An 87–2,060 bp promoter window is then 95%+ zero padding while
occupying a full input channel of the layer-1 conv. This is why
`dna_regulatory__mdcnn` (joint CV 0.8926) scored *below* `dna__mdcnn` (0.8989).

The flag groups by `(modality, channels)` instead, giving promoters their own
trunk padded to their own 2,060 bp maximum. Default off; `full_run` reproduces.

---

## 4. Early-stopping monitor (`--monitor-min-n`, joint only)

`_macro_val_auc` averaged **unweighted** over all 11 drugs. LEVOFLOXACIN has 269
phenotyped isolates — ~43 per validation fold, ~15 resistant — so it contributed
a ninth of the stop signal and mostly noise; early stopping restored whichever
epoch LEVO got lucky on. `--monitor-min-n 500` drops such drugs from the *stop
signal only*; they are still trained on and still reported. `0` = `full_run`
behaviour. If the threshold would exclude every drug it is ignored **with a
printed message** rather than silently.

---

## 5. Optimizer and capacity knobs (all default to `full_run` values)

| flag | default | note |
|---|---|---|
| `--lr` | `exp(-9)` ≈ 1.2e-4 | was a module constant, unreachable without editing source |
| `--weight-decay` | 0 | any value > 0 switches Adam → **AdamW** (Adam's `weight_decay` couples into the adaptive step and does not decay weights as the name implies) |
| `--dropout` | 0 | after each dense-head hidden layer; the stack had no regularization at all |
| `--hidden` | 256 | dense-head width |
| `--per-drug-hidden` | 0 | joint only — see below |

`DenseHead` with `out_dim=11` read all eleven drugs off **one 256-d vector via
one shared linear layer**, while single-drug models got the same 256 units for
one task. Adding a modality widened `fc1`'s *input* without widening the shared
representation — the likely mechanical reason joint modality gains
(+0.005 late_fusion) collapsed against single-drug (+0.023–0.031).
`--per-drug-hidden k` gives each drug its own `hidden → k → 1` branch off the
shared trunk. Measured on the submitted `joint_capacity` jobs: k=64 costs
**+178,816 parameters (+0.39%)** on a 45.9M-parameter joint `late_fusion` net.

The same job logs give an empirical check on the "defaults unchanged" claim
above, stronger than the fixed-seed test: the `b3_reg` arm (dropout + AdamW,
which add no parameters) reports **45,898,955** for `dna_protein__late_fusion`
and **47,955,211** for `cisfusion` — equal to `full_run`'s counts to the
parameter.

`SetFusionNet` already has per-drug capacity (one learned query each), so it
takes only `--dropout`, mapped to a new `head_dropout` that is kept distinct
from its transformer dropout.

---

## Files touched

| file | change |
|---|---|
| `training/checkpoint.py` | **new** — save/load, config schema v1, rebuild |
| `tests/test_checkpoint.py` | **new** — 30 checks, bit-identical round trips |
| `bigtb_ref.py` | `MODEL_WEIGHTS_DIR` |
| `models/net.py` | `DenseHead` dropout + per-drug branches; head knobs threaded through all four nets; `MDCNNNet` trunk grouping + `from_blocks`; `SetFusionNet.head_dropout` |
| `training/multimodal.py` | `build_optimizer` (Adam/AdamW), lr/wd/head plumbing, checkpoint write |
| `training/multidrug.py` | same, plus `monitor_columns`; `_predict` empty-index path no longer reads `head.fc_out` (None with per-drug heads, absent on SetFusionNet) |
| `scripts/run_experiment.py`, `scripts/run_multidrug.py` | new flags + `data_config` provenance + weights pointer |
| `scripts/sbatch_all_runs.py` | plumbs `--all-regulatory`, `--min-delta`, `--monitor-min-n`, `--lr`, `--weight-decay`, `--dropout`, `--hidden`, `--per-drug-hidden`, `--mdcnn-trunk-per-modality`, `--save-weights`, `--weights-dir` |

Test suite after the changes: **70/70** (`test_baseline_alignment` 8,
`test_cisfusion` 16, `test_setfusion` 16, `test_checkpoint` 30).

---

## The one thing to read before trusting any joint comparison

Joint fold SD in `full_run` runs **0.003–0.030** against a joint-vs-MD-CNN gap of
**−0.0116**, on one seed and five folds. Most of the joint leaderboard is inside
its own noise, and the `mdcnn` joint row (SD 0.019–0.030) cannot be ranked at
all. The single-drug regulatory evidence is solid — 11 independent drug-runs,
consistent across three architectures — but joint per-drug deltas of ±0.005 are
not. **Multi-seeding the joint cells matters more than any architecture change
listed above**, and neither run submitted today does it; that is deliberate
(these two are one-variable-at-a-time diagnostics), but it is the gap to close
before anything joint is reported as a result.
