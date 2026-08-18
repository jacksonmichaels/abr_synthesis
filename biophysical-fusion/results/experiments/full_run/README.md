# full_run — architecture × modality sweep

Submitted **2026-08-04**, 240 SLURM jobs, all at **150 epochs**. Every
architecture we have, crossed with every modality set, for both single-drug and
multi-drug prediction. Results land in the subfolders described below.

| | jobs | SLURM ids | resources |
|---|---|---|---|
| single-drug (one job per experiment × drug) | 220 | `62597933–62598172` | 48G, 1 GPU, 4 cpu, 8h |
| multi-drug (one job per experiment) | 20 | `62598173–62598195` | 64G, 1 GPU, 6 cpu, 16h |

Submission manifests (exact command per job): `../../../slurm_logs/submitted_20260804_1557*.json`.
Logs: `../../../slurm_logs/slurm-{experiment}_{drug}_*.out`.

## The grid — 4 architectures × 5 modality sets = 20 experiments

### Architectures

| `--arch` | class | what it does |
|---|---|---|
| `late_fusion` | `MultiModalNet` / `MultiDrugNet` | one encoder per feature block, outputs concatenated into a shared dense head. Our original net. |
| `mdcnn` | `MDCNNNet` | BIG-TB's own SD-CNN/MD-CNN topology: loci become **channels** on one zero-padded position axis, `Conv2d(n_loci, 64, (C,12))` mixes every locus at layer 1. |
| `setfusion` | `SetFusionNet` | one encoder **shared per modality**; each block becomes a token carrying learned (modality, locus) embeddings; transformer fusion + one attention query per drug. |
| `cisfusion` | `CisFusionNet` | promoter ⊕ CDS concatenated per locus into a cis-unit (+ a 6th channel marking which segment each column is), then per-branch encoders. |

`mdcnn`, `setfusion` and `cisfusion` all load **per-locus** blocks (implied by
`--arch`); `late_fusion` uses the default one-branch-per-modality layout.

### Modality sets

| set | modalities |
|---|---|
| `dna` | DNA only |
| `dna_protein` | DNA + protein |
| `dna_biophysical` | DNA + biophysical |
| `dna_regulatory` | DNA + regulatory |
| `all_modalities` | DNA + protein + biophysical + regulatory |

## Folders

40 subfolders, `{modality_set}__{arch}` for single-drug and
`multidrug_{modality_set}__{arch}` for multi-drug:

```
full_run/
  dna__late_fusion/              {DRUG}__{tag}.json  (x11)  summary.csv  {DRUG}__{tag}_curves.png
  dna__mdcnn/                    …
  dna__setfusion/                …
  dna__cisfusion/                …
  dna_protein__late_fusion/      …          (and the same 4 archs for every set)
  …
  multidrug_dna__late_fusion/    multidrug__{tag}.json  multidrug_summary.csv  *_curves.png
  …
  multidrug_all_modalities__cisfusion/
```

Per single-drug folder: one JSON per drug (CV folds, held-out test metrics,
per-epoch history, `arch`, `n_params`), a `summary.csv` across the 11 drugs, and
one curves PNG per drug. Per multi-drug folder: one JSON with per-drug **and**
macro CV/TEST metrics, plus `multidrug_summary.csv` (one row per drug + MACRO).

## Protocol (identical across every cell)

- 150 epochs, early stopping on **val AUC**, patience 15, min_delta 1e-4, best-weight restore
- 5-fold CV on the 80% training split; TEST = the best-val-AUC fold model scored once on the untouched 20%
- masked weighted BCE, alpha class weights fit on the training split only
- batch 128, Adam, LR `exp(-9)`, seed 0; splits seeded 42
- missing-phenotype isolates dropped before splitting

## Read this before comparing cells

- **The two scopes use different locus sets, by design.** Single-drug uses
  BIG-TB's per-drug `DRUG_TO_LOCI` (18-locus universe, no `fabG1` gene body,
  `--extra-loci` off), which keeps it locus-matched to the SD-CNN baseline.
  Multi-drug uses every curated locus on disk — **19, including `fabG1`** —
  which is MD-CNN's own rule. So a single-drug vs multi-drug comparison for
  INH/ETO is not locus-matched.
- **The fabG1 promoter is present in every `regulatory` cell**, under the name
  `inhA`: WHO delimits the *fabG1–inhA* operon promoter from fabG1's own start
  (tss 1,673,440) but files it under `inhA`, so `c-15t` is inside that window.
  This is separate from the fabG1 *gene body* above.
- **`cisfusion` without `regulatory` has no promoter to pair**, so
  `dna__cisfusion`, `dna_protein__cisfusion` and `dna_biophysical__cisfusion`
  degenerate to late fusion plus a constant segment channel. The cells that
  actually exercise it are `dna_regulatory__cisfusion` and
  `all_modalities__cisfusion`.
- **`setfusion`'s locus keying needs ≥2 modalities** to pair anything, so
  `dna__setfusion` is a weight-sharing ablation, not a pairing test. Also note
  it starts degenerate: at init every drug gets an identical logit (the fused
  tokens are near-collinear, pairwise cosine ≈0.997), so the per-drug queries
  have to learn their way apart. **This is what broke the setfusion cells in
  this run — see "Known problem with the setfusion cells" below.**
- **`mdcnn` groups blocks by channel count**, and DNA and regulatory are both
  5-channel — so in `dna_regulatory__mdcnn` and `all_modalities__mdcnn` the
  promoter windows are stacked as extra channels alongside the CDS loci in one
  trunk, rather than getting their own.
- **Regulatory cells carry one promoter window per loaded locus.** The region
  set is intersected with the loci, so `dna_regulatory` is 2 blocks per drug
  rather than the full WHO candidate list, and the joint runs carry 16 regions
  rather than 48. `--all-regulatory` is the opt-out.
- Test metrics are the best-of-5-folds model, i.e. favourably selected. Quote CV
  mean ± SD for comparisons; say so when quoting test.

## The sweep in five steps

The sweep varies two things at once — architecture and input modalities — so it
is read in two stages, each asked of both the single-drug and the multi-drug
task. Stage 1 (§1–§2) holds the inputs at what the baseline gets and isolates the
model/protocol; stage 2 (§3–§4) holds the architecture fixed and isolates the
modalities. Keeping them apart is the only way to attribute a margin.

| § | question | answer |
|---|---|---|
| **1** | DNA only, single-drug — do we beat the corrected SD-CNN? | **Yes, +0.0168** (10/11 drugs). The best cell is `mdcnn` — *BIG-TB's own topology* — so this is the training protocol alone, not the model. |
| **2** | DNA only, joint — do we beat the published MD-CNN? | **No, −0.0116** (−0.0011 excluding the flagged ETHIONAMIDE row). Parity at best. |
| **3** | What do the modalities add, single-drug? | **+0.026** on top of DNA, taking the best cell from **+0.0168 → +0.0412** vs SD-CNN. Roughly doubles a margin we already had. |
| **4** | What do the modalities add, joint? | **+0.036** on `cisfusion`, taking it from **−0.0385 → −0.0020** vs MD-CNN (+0.0087 excl ⚠). Closes the step-2 deficit to parity — not a decisive win. |
| **5** | Takeaways | below |

The asymmetry between §3 and §4 is the substantive finding: **modality choice
matters much more single-drug than joint**, because a joint model already sees
every locus for every drug and recovers by sharing what the single-drug model has
to be told explicitly.

## What the sweep found

1. **A single fixed configuration beats the leak-corrected SD-CNN on all 11
   drugs** — `all_modalities / mdcnn`, mean CV 0.9049 vs 0.8636, Δ +0.0412, with
   no per-drug selection (Fig C1). 16 of the 20 single-drug cells are ahead.
2. **The extra modalities help three drugs a lot and six drugs not at all**, and
   which three follows from the resistance mechanism: PYRAZINAMIDE +0.099 from
   protein/biophysical (*pncA* loss-of-function needs generalisation across
   unseen substitutions), ETHIONAMIDE +0.094 and ISONIAZID +0.042 from regulatory
   (the *fabG1–inhA* promoter). *rrs*-driven AMIKACIN/KANAMYCIN gain +0.001/+0.002
   — rRNA has no protein product, so those descriptors are undefined where it
   matters. RIFAMPICIN +0.000 (already 0.976). This is Fig 0 and it is the
   headline; the grid means average it away.
3. **The best architecture flips between scopes** (Fig D2): `mdcnn` wins
   single-drug (0.895), `late_fusion` wins joint (0.914). Mixing every locus at
   layer 1 is a good prior for 2–4 relevant loci and a bad one for 19.
4. **Joint ≈ MD-CNN parity**, best `dna_protein / cisfusion` macro CV 0.9228 vs
   0.9248 — and **+0.0087 ahead** once the flagged ETHIONAMIDE baseline row is
   dropped. That one suspect number decides parity-vs-behind.
5. **`setfusion`'s numbers here are an early-stopping artifact, not an
   architecture verdict** — see below.

## Known problem with the setfusion cells

SetFusionNet starts from a near-degenerate init: train loss sits flat (~0.2405)
for ~12 epochs, the monitored val AUC peaks *inside* that plateau, and
`patience=15` then fires around epoch 25 and restores weights from **before** the
network broke out. Across all 25 joint setfusion folds `best_epoch` is 3–12 while
the loss break is at epoch 8–20. The two folds that survived past it
(`dna_biophysical` folds 3/4, running to epochs 72 and 113) scored 0.811 and
0.8385 against a 0.78 cell mean.

Fixed by `EarlyStopper(min_epochs=...)` — a warmup that pins the patience counter
at zero until the given epoch. Best-weight tracking still runs, so a warmup can
never return a worse model. Exposed as `--min-epochs` on both entry points and on
`sbatch_all_runs.py`.

## Follow-up runs (submitted 2026-08-06)

| folder | jobs | question | how it differs |
|---|---|---|---|
| `../setfusion_warmup/` | 55 single-drug + 5 joint | Is setfusion's 0.78 real? | identical grid, `--min-epochs 50` |
| `../alllocus_run/` | 220 (20 cells × 11 drugs) | Is "joint wins" multi-task sharing or just the bigger input? | single-drug on the **joint 19-locus set**, so the two scopes are locus-matched. setfusion cells also carry `--min-epochs 50` so they stay matched to the warmup joint runs. |

Early returns on the warmup: of the first 27 finished drug-runs, **24 improved,
mean +0.0165 CV-AUC** (LEVO +0.091, MOXI +0.055, ETO +0.032), with folds now
running 64–106 epochs instead of stopping at 16–46.

## Viewing the results

`full_run_viewer.ipynb` (in this folder) reads all 40 run folders — plus the two
follow-up folders above once they exist — and is laid out as the five steps
above, each ending in a printed one-line ANSWER:

| cell | what it shows |
|---|---|
| §1 **Fig 1** | Δ vs SD-CNN per architecture, DNA only |
| §2 **Fig 2** | Δ vs MD-CNN per architecture, DNA only (with the excl-⚠ marker) |
| §3 **Fig 3** | the modality *ladder*: CV per modality set, one line per architecture, baseline as a rule and the region above it shaded |
| §3 **Fig 0** | the same gain broken out per drug — where it actually comes from |
| §4 **Fig 4** | the joint ladder |
| §5 | takeaways + a scorecard recomputed from the run folders |
| Appendix | the 20-cell leaderboards, arch × modality grids, **Fig C1** (one fixed config vs the baseline, no per-drug selection), Fig B/C, **Fig D2** (the architecture rank flip), Fig E/**E2**, and the follow-up decomposition |

Figures 3 and 4 are the ones to look at first: they answer "where did we cross
the baseline, and how much of that was the modalities" in one picture.

Two reading rules the notebook enforces mechanically: every ladder point is a
mean over **all 11 drugs** — a cell still running is drawn as a *gap* rather than
averaged over fewer drugs — and modality gains are always measured against the
**same architecture's** DNA-only cell.

Baselines are carried over unchanged from `notebooks/results_viewer.ipynb`. It
runs against partial results — rerun it as jobs land. Figures save to
`results/figures/full_run/`. Regenerate the notebook itself with
`python scripts/build_full_run_viewer.py`, then execute it (it ships executed).

## Monitoring

```bash
squeue -u $USER                                   # queue state
grep -l "FAILED\|Traceback" slurm_logs/*.out      # failures
python training/curves.py results/experiments/full_run/<folder>   # re-plot curves
```

Note `results/` is git-ignored, so this file is not versioned — the grid itself
lives in `scripts/sbatch_all_runs.py` (`MODALITY_SETS` × `GRID_ARCHS`).
