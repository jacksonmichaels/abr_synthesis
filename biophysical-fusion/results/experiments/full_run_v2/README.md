# full_run_v2 — the full sweep again, trained longer, with weights saved

Submitted **2026-08-06**, 240 SLURM jobs, reproducible via `submit.sh` in this
folder. Manifests: `slurm_logs/submitted_20260806_202903_820097.json` (220
single-drug) and `..._202925_997520.json` (20 joint).

## Why this run exists

`full_run` (2026-08-04) **checkpointed nothing** — `EarlyStopper` kept the best
epoch in an in-memory dict and the process exited. Every model behind its
headline numbers is gone: nothing can be re-scored, recalibrated, inspected for
attribution, or deployed. This re-runs the same grid so the project has a
baseline whose models actually exist on disk.

Weights and a rebuild config land in
`/project/pi_mfiterau_umass_edu/abr_model_weights/full_run_v2/{stem}/`. See
`../CODE_CHANGES_20260806.md` for the checkpoint format.

## What differs from `full_run`

**Inputs are identical** — same loci, same modality sets, same phenotypes, no
`--all-regulatory`, no `--extra-loci`. Only the training schedule changed, so
this is a baseline *replacement*, not a new experiment.

| setting | `full_run` | here | why |
|---|---|---|---|
| `--epochs` | 150 | **300** | 150 was a binding cap for the joint cells — `best_epoch` 120–148 on 40% of `late_fusion` folds, still climbing (fold 0: ep50 0.8727 → ep100 0.8969 → ep146 0.9100) |
| `--patience` | 15 | **30** | 15 fired spuriously: joint `all_modalities__mdcnn` fold 1 stopped at epoch 36 scoring 0.8609 while its siblings reached 0.918 |
| `--min-epochs` | 0 | **50** | warmup. SetFusionNet starts near-degenerate, sits at flat loss ~12 epochs, and patience fired before it escaped — that is what made `full_run`'s setfusion row (0.76–0.80) an artifact rather than an architecture verdict |
| weights | none | **best fold** | the fold scored on TEST, i.e. the model the reported numbers come from |

`--min-epochs` is applied to **every** architecture, not just setfusion. That is
safe because best-weight tracking still runs during warmup: a warmup can only
buy more epochs, never return a worse model than not warming up.

## The grid — unchanged from `full_run`

4 architectures (`late_fusion`, `mdcnn`, `setfusion`, `cisfusion`) × 5 modality
sets (`dna`, `dna_protein`, `dna_biophysical`, `dna_regulatory`,
`all_modalities`) = 20 cells. Single-drug is 11 jobs per cell (220); joint is one
job per cell (20).

```
full_run_v2/
  dna__late_fusion/            {DRUG}__{tag}.json (x11)  summary.csv  *_curves.png
  …                            (20 such folders)
  multidrug_dna__late_fusion/  multidrug__{tag}.json  multidrug_summary.csv
  …                            (20 such folders)
  weights_location.json        per-cell pointer into the weights volume
```

## Expected cost

`full_run` took ~144 GPU-hours at 150 epochs. Single-drug jobs are cheap
(median 0.1 h, max 2.9 h); the cost is the 20 joint jobs (`late_fusion` joint
median 9.3 h). At 300 epochs with `patience 30` and a 50-epoch warmup, expect
roughly **250–300 GPU-hours** and the joint `late_fusion` cells to run ~26 h
against their 48 h limit. Single-drug limit is 16 h.

Storage: **5.1 GB** of weights (one fold per run). `--save-weights all` would be
25.5 GB — the shared volume was at 95% (1.1 TB free) when this was submitted,
which is why `best` is what ran.

## Reading it — and the one thing to be careful about

**This run's numbers will not equal `full_run`'s, and its deltas against the
published SD-CNN / MD-CNN baselines may move.** Three settings changed at once,
deliberately, because the goal was a better baseline artifact rather than an
attribution experiment. If you need to know *which* of the three moved a given
cell, that is what `../joint_convergence/` isolates (one variable per arm, joint
scope). Do not read a `full_run_v2` − `full_run` difference as "epochs did this".

Expect `setfusion` to move the most: its `full_run` row was a known artifact and
this is the first run where every setfusion cell gets the warmup. `../setfusion_warmup/`
(same grid, `--min-epochs 50`, 150 epochs) is the matched comparison for that.

The standing caveat still applies: joint fold SD in `full_run` was 0.003–0.030 on
one seed against a −0.0116 gap, so single-seed joint cells cannot be ranked
against each other. Multi-seeding remains the prerequisite for reporting joint
results — see the closing section of `../CODE_CHANGES_20260806.md`.

## Loading a model

```python
from training.checkpoint import load_model
model, cfg = load_model("full_run_v2/multidrug_all_modalities__mdcnn",
                        "multidrug__dna+protein+biophysical+regulatory")
cfg["model"]["drug_names"]           # which logit column is which drug
cfg["split"]["test_isolate_ids"]     # the exact held-out isolates
```
