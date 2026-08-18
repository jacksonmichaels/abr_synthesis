# alllocus_run — is "joint wins" multi-task sharing, or just a bigger locus set?

Submitted **2026-08-06**, 221 SLURM jobs (the 20-cell single-drug grid × 11 drugs,
plus one resubmit). Manifests: `slurm_logs/manifests/submitted_20260806_1349*.json`
and `..._1604*.json`.

> ## This folder was reconstructed from SLURM logs on 2026-08-18
>
> The original `results/experiments/alllocus_run/` is **gone**. It is not in git
> (`results/` is git-ignored), `CLEANUP_REPORT.md` does not record deleting it,
> and only 9 stray checkpoint folders survive under
> `/project/pi_mfiterau_umass_edu/abr_model_weights/alllocus_run/`. What did
> survive is `slurm_logs/archive/20260806/` — and those logs print every number a
> `summary.csv` carries.
>
> `reconstruct_from_logs.py` in this folder re-derives the tables from them.
> **Read "What is and is not here" below before quoting anything from it.**

## The question

`full_run`'s §2/§4 comparison — joint models beat single-drug models — is
confounded, and the notebook says so in its own caveats. The two scopes do not
see the same input:

- **single-drug** uses SD-CNN's per-drug map (`tb.DRUG_TO_LOCI`): ISONIAZID gets
  `inhA+katG`, 2 loci.
- **joint** uses MD-CNN's rule, `datasets.loci_on_disk()`: every curated locus,
  19 of them, for every drug.

So "joint wins" could be multi-task sharing, or it could be that the joint model
simply gets 19 loci per drug where the single-drug model gets 2. This run
separates the two by giving the **single-drug** grid the same 19 loci
(`--loci eis embA embB embC ethA ethR fabG1 gid gyrA gyrB inhA katG pncA rpoB
rpoC rpsL rrl rrs tlyA`).

**`full_run` is the matched control, not `full_run_v2`.** Every job here differs
from its `full_run` counterpart in exactly one flag — `--loci`. Same 150 epochs,
patience 15, no warmup, batch 128, 5 splits, seed 0. `full_run_v2` changed three
training settings at once, so it is not comparable to this run.

## The answer: it depends on the architecture, and for the BIG-TB topology it is the loci

Macro CV AUC over the 11 drugs. `Δ loci` is this run minus `full_run`
single-drug; `Δ joint−SD19` is `full_run`'s joint cell minus this run — i.e. what
multi-task sharing is worth **once the locus universe is matched**.

| cell | SD 2–3 loci | SD 19 loci | Δ loci | JOINT 19 | Δ joint−SD19 | n |
|---|---:|---:|---:|---:|---:|---:|
| `dna__late_fusion` | 0.8779 | 0.8820 | +0.0041 | 0.9132 | **+0.0313** | 11 |
| `dna_regulatory__late_fusion` | 0.8854 | 0.8873 | +0.0018 | 0.9158 | **+0.0285** | 11 |
| `dna_biophysical__late_fusion` | 0.8875 | 0.8608 | −0.0267 | 0.9133 | **+0.0525** | 11 |
| `dna_protein__late_fusion` | 0.8644 | 0.8260 | −0.0384 | 0.8946 | +0.0686 | 7 |
| `all_modalities__late_fusion` | 0.8750 | 0.8246 | −0.0504 | 0.8844 | +0.0598 | 7 |
| `dna__mdcnn` | 0.8804 | 0.8982 | +0.0178 | 0.8989 | **+0.0007** | 11 |
| `dna_protein__mdcnn` | 0.8921 | 0.9036 | +0.0115 | 0.9142 | +0.0106 | 11 |
| `dna_biophysical__mdcnn` | 0.9105 | 0.9097 | −0.0009 | 0.9131 | +0.0034 | 10 |
| `dna_regulatory__mdcnn` | 0.8895 | 0.8992 | +0.0096 | 0.8926 | −0.0065 | 11 |
| `all_modalities__mdcnn` | 0.9049 | 0.8989 | −0.0060 | 0.9025 | +0.0036 | 11 |
| `dna__cisfusion` | 0.8717 | 0.8984 | +0.0267 | 0.8863 | −0.0121 | 11 |
| `dna_protein__cisfusion` | 0.8899 | 0.9059 | +0.0161 | 0.9228 | +0.0168 | 11 |
| `dna_biophysical__cisfusion` | 0.8900 | 0.9060 | +0.0160 | 0.9165 | +0.0105 | 11 |
| `dna_regulatory__cisfusion` | 0.8838 | 0.8935 | +0.0096 | 0.8902 | −0.0032 | 11 |
| `all_modalities__cisfusion` | 0.9022 | 0.9041 | +0.0018 | 0.9157 | +0.0117 | 11 |
| `dna__setfusion` | 0.8199 | 0.8287 | +0.0087 | 0.7644 | −0.0643 | 11 |
| `dna_protein__setfusion` | 0.8659 | 0.8218 | −0.0441 | 0.7775 | −0.0443 | 10 |
| `dna_biophysical__setfusion` | 0.8657 | 0.8188 | −0.0470 | 0.8000 | −0.0188 | 11 |
| `dna_regulatory__setfusion` | 0.8450 | 0.8261 | −0.0189 | 0.7632 | −0.0629 | 11 |
| `all_modalities__setfusion` | 0.8729 | 0.8162 | −0.0567 | 0.7718 | −0.0443 | 10 |

Every macro in a row is over the same drug set (`n`), so each row's deltas are
internally matched. Macros are **not comparable across rows with different `n`**:
the n=7 and n=10 cells are missing the largest drugs (see "What is and is not
here"), which shifts a macro by far more than any effect here.

Read by architecture:

- **`mdcnn` — BIG-TB's own topology, and the answer is the loci.** Multi-task
  sharing is worth **+0.0007** on DNA and −0.0065 to +0.0106 across the five
  modality sets, mean ≈ **+0.002**. Every one of those is inside the 0.003–0.030
  joint fold SD. Give the single-drug SD-CNN the 19-locus input and it matches the
  joint MD-CNN. The `full_run` "joint wins" margin for this architecture was the
  locus universe.
- **`cisfusion` — same story**, mean Δ ≈ +0.005, two of five cells negative.
- **`late_fusion` — a real joint advantage survives.** On the two clean n=11
  cells it is +0.031 and +0.029, well outside fold SD. This is the one
  architecture where multi-task training is doing something the locus set cannot
  explain. A plausible mechanism, untested: single-drug `late_fusion` on 19 loci
  is a 137k-wide flatten and ~36M parameters fit to one drug's labels, and the
  `Δ loci` column shows it *losing* ground as inputs widen — the joint model
  fits the same trunk against 11 label columns at once. If that is right the
  effect is regularization, not knowledge transfer, and adding dropout to the
  single-drug model should shrink it. `../joint_capacity/` tested the converse
  (regularizing the joint model) and it hurt.
- **`setfusion` — joint is worse than single-drug everywhere**, by 0.019–0.064.
  Consistent with the known early-stopping artifact in joint setfusion cells
  (`../setfusion_warmup/`); nothing here is an architecture verdict.

### `mdcnn`, DNA only, per drug

| drug | SD 2–3 loci | SD 19 loci | Δ loci | JOINT 19 | Δ joint−SD19 |
|---|---:|---:|---:|---:|---:|
| Amikacin | 0.8709 | 0.9033 | +0.0324 | 0.8895 | −0.0138 |
| Capreomycin | 0.8515 | 0.8652 | +0.0137 | 0.8711 | +0.0059 |
| Ethambutol | 0.9308 | 0.9486 | +0.0178 | 0.9442 | −0.0044 |
| Ethionamide | 0.6817 | 0.7928 | **+0.1111** | 0.7861 | −0.0067 |
| Isoniazid | 0.9151 | 0.9698 | **+0.0547** | 0.9714 | +0.0016 |
| Kanamycin | 0.8735 | 0.9436 | **+0.0701** | 0.9173 | −0.0263 |
| Levofloxacin | 0.9503 | 0.8042 | **−0.1461** | 0.8248 | +0.0206 |
| Moxifloxacin | 0.8562 | 0.8172 | −0.0390 | 0.8591 | +0.0419 |
| Pyrazinamide | 0.8540 | 0.9278 | **+0.0738** | 0.9264 | −0.0014 |
| Rifampicin | 0.9765 | 0.9791 | +0.0026 | 0.9773 | −0.0018 |
| Streptomycin | 0.9241 | 0.9285 | +0.0044 | 0.9207 | −0.0078 |
| **macro** | **0.8804** | **0.8982** | **+0.0178** | **0.8989** | **+0.0007** |

The macro hides two opposite effects. Extra loci help ETO (+0.111), KAN (+0.070),
PZA (+0.074) and INH (+0.055) — and **cost LEVOFLOXACIN 0.146** and MOXI 0.039,
the two fluoroquinolones. LFX has n=269, so ~54 isolates per fold against a
19-locus input: that is the smallest drug meeting the widest input, and the
direction is what overfitting looks like. Do not read the LFX number as
mechanism.

Note also that the SD-19 model is **no longer locus-matched to the SD-CNN
baseline**, so nothing in this run can be quoted against published SD-CNN
numbers. That comparison stays with `full_run` / `full_run_v2`.

## What is and is not here

Recovered: **210 of 220** (cell, drug) results, from the archived job logs.

**Present**, and verified equal to a real run's (see "Validation"):
`summary.csv` per cell — same columns as a genuine run; `cv_folds.csv` — per-fold
AUC, AUC-PR, sens, spec, `n_val`, `best_epoch`; `provenance.csv` — drug → job id,
SLURM state, elapsed, MaxRSS, arch, `n_params`, `n_test`, `test_model_fold`, and
the log path every number came from.

**Absent, and not reconstructible:**

| | why it matters |
|---|---|
| `{DRUG}__{tag}.json` — the canonical per-run artifact | deliberately **not** written. A JSON that looked genuine but lacked `cv_folds[i]["history"]`, `split`, `out_bias` etc. would break `training/curves.py` and `checkpoint.load_model` and invite being mistaken for real output. `build_full_run_viewer.py` reads `summary.csv` for every number it plots and touches the JSONs only to fill its `n_params` / `param_range` maps, so parameter counts are the one thing a generated viewer would show blank here — they are in `provenance.csv` instead. |
| per-epoch histories | no curve plots, and no way to check the 150-epoch cap beyond `best_epoch` (which says the cap was **not** binding for `mdcnn`: 10 of 275 folds ≥ epoch 140, 3.6%) |
| weights | 9 of 210 survive on the weights volume; nothing here can be re-scored, calibrated or attributed |
| `seconds` | **left empty on purpose.** A real run measures training time in-process; the logs do not print it. SLURM `Elapsed` covers the whole job including a multi-minute data load, so it is a different quantity — it is in `provenance.csv` as `job_elapsed`, not silently substituted into `seconds`. |
| `test_sens` / `test_spec` precision | the log prints 3 dp, a real `summary.csv` has 4 |
| `cv_auc_pr_mean` precision | no printed aggregate exists, so it is a mean of 4-dp per-fold values and can differ from a real run in the 4th decimal. `cv_auc_mean` / `cv_auc_std` are taken from the run's own printed line and are exact. |

**The 10 missing results** are in `missing.csv`, and they are not random — they
are the largest drugs meeting the widest inputs:

| cell | drugs | cause |
|---|---|---|
| `dna_protein__late_fusion` | INH, RIF, EMB, PZA | host **OUT_OF_MEMORY** at `--mem 64G` |
| `all_modalities__late_fusion` | INH, RIF, EMB, PZA | host **OUT_OF_MEMORY** at `--mem 64G` |
| `dna_protein__setfusion` | INH | **TIMEOUT** at 12 h |
| `all_modalities__setfusion` | INH | **TIMEOUT** at 12 h |

All four OOM drugs are the ≥12.9k-isolate ones; `late_fusion` on 19 loci × 2–4
modalities is the widest flatten in the grid. `KANAMYCIN` /
`all_modalities__late_fusion` additionally died on a **CUDA** OOM (job
`62639153`) and was resubmitted successfully as `62643407`; the reconstruction
takes the later job. **Every `mdcnn` and `cisfusion` cell is complete at 11/11** —
`mdcnn` is 3.87M parameters against `late_fusion`'s ~36M, and never came near the
limit.

## Validation

The parser was checked against a run whose real `summary.csv` files still exist,
before being trusted here:

```bash
python results/experiments/alllocus_run/reconstruct_from_logs.py --run full_run_v2 --check
# check: 220 rows across 20 cells, 0 mismatches beyond log precision, 387 within it
```

220 of 220 rows of `full_run_v2` reproduce exactly, across all 20 cells and every
field. That check found three real parser bugs on the way — the run reports
**population** std (ddof=0), the mean must be read from the printed aggregate
rather than averaged from 4-dp per-fold lines, and `late_fusion` cells carry no
per-locus block names so `genes` has to come from the manifest's `--loci`. All
three are fixed and commented in the script.

## Regenerating this folder

```bash
python results/experiments/alllocus_run/reconstruct_from_logs.py
```

Idempotent, reads only `slurm_logs/`, and depends on
`slurm_logs/archive/20260806/` staying in place. **Those logs are now the only
copy of this run.** `slurm_logs/README.md` describes the archive policy; this run
is the reason not to prune 20260806.

## If this run is worth having properly

Rerunning it is ~220 jobs. Two things must change: `--mem` above 64G for the
`late_fusion` cells (the 8 OOMs), and a wall clock above 12 h for single-drug
setfusion on ISONIAZID. Add `--save-weights best` — `full_run` saved nothing,
which is the same mistake `full_run_v2` exists to correct. Whether it is worth
the GPU time depends on whether the `late_fusion` joint advantage above is worth
pinning down; the `mdcnn` conclusion is already clear from what was recovered.
