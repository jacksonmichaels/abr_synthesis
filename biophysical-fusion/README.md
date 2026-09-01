# Biophysical Fusion — ABR prediction

Genotype→phenotype antibiotic-resistance (R/S) prediction for *M. tuberculosis*,
built on top of the BIG-TB benchmark. The core question: does fusing amino-acid
**biophysical traits** (Kulkarni et al. 2026) onto BIG-TB's DNA CNN improve
binary resistance classification? The pipeline is **modality-selectable** — DNA,
protein, biophysical, and regulatory inputs can be combined in any subset.

Nothing here modifies `../Big-TB-benchmark/`; reference code is only *read*
(single point of contact: `bigtb_ref.py`).

## Overview

**The task.** Per-drug binary resistance (R/S) prediction for *M. tuberculosis*
from gene sequence — 11 antibiotics, 17,942 isolates, either one model per drug
(**single-drug**) or one model with 11 outputs (**joint**). Everything is scored
against the **BIG-TB benchmark** (Tasmin et al.), whose two published CNNs are
the reference points: **SD-CNN** (one drug at a time) and **MD-CNN** (13 drugs
from a shared trunk).

**What this project adds.** BIG-TB feeds its CNN a DNA one-hot and nothing else.
We ask what *derived and adjacent* views of the same locus are worth — the amino
acid translation, its physicochemical profile (Kulkarni et al. 2026), and the
WHO-catalogue promoter window upstream of it — and whether an architecture that
can **pair** those views beats one that merely concatenates them.

**Five axes, one runner.** Every run is a point in this grid, and every result
JSON records where it sat:

| axis | values |
|---|---|
| **modality** | `dna`, `protein`, `biophysical`, `regulatory` — any subset |
| **architecture** (`--arch`) | `late_fusion`, `mdcnn`, `setfusion`, `cisfusion`, `locusfusion`, + 6 experimental aggregators |
| **scope** | single-drug (`run_experiment.py`) / joint 11-output (`run_multidrug.py`) |
| **branch encoder** (`--encoders`) | `cnn` / `transformer` |
| **locus universe** | SD-CNN's per-drug map (2–3 loci) / every curated locus (19) |

**How a result is judged.** 5-fold CV AUC, macro-averaged over the 11 drugs
shared with the baselines. Held-out test AUC is recorded and reported because it
is what the papers publish, but it is one 20% split of one best-CV-fold model and
swings by ±0.05 on the small drugs — it is never the number to conclude from.
Single seed throughout, so **differences under ~0.01 are unresolved**, not small.

**Both baselines are handled, one of them corrected.** SD-CNN's published test
AUC is inflated by a crossval/assess `stratify` mismatch — ~80% of its "test"
isolates were in training (MOXI 0.886 → 0.825). We use leak-corrected numbers
re-scored from the authors' own saved models (`h1_repro/`,
`reference_docs/BASELINE_LEAK_FINDING.md`). MD-CNN derives train/test with one
identical call in both scripts and needs no correction; we reproduced it end to
end. `scripts/build_full_run_viewer.py` is the single source of both tables.

### Where it stands

Macro CV AUC over the 11 shared drugs, best architecture per row:

| | macro CV | |
|---|---:|---|
| SD-CNN, leak-corrected (published single-drug baseline) | 0.8636 | |
| MD-CNN, the authors' own run log (published joint baseline) | 0.9222 | |
| **our MD-CNN reproduction** | **0.9212** | −0.0010 — reproduced |
| single-drug, DNA only, locus-matched to SD-CNN | 0.8919 | **+0.028** vs SD-CNN, ahead on 11/11 |
| single-drug, all modalities, locus-matched | 0.9086 | **+0.045** vs SD-CNN, ahead on 11/11 |
| joint, 19 loci | 0.9210 | parity with MD-CNN |
| **single-drug, all modalities, 19 loci** | **0.9246** | best cell in the project |

Four findings sit underneath those rows.

**1. The modality gain is concentrated, and it lands where the resistance
mechanism predicts.** It is worth **+0.017** macro (0.8919 DNA-only → 0.9086 all
modalities), but averaging it that way destroys the result — two drugs carry
almost all of it and five get nothing:

Best gain over that drug's own DNA-only cell, averaged over `late_fusion` /
`mdcnn` / `cisfusion` (`setfusion` excluded — see finding 4):

| drug | DNA-only | modality that helps | gain | why |
|---|---:|---|---:|---|
| ETHIONAMIDE | 0.694 | **regulatory** | **+0.089** | the *fabG1–inhA* operon promoter (`c-15t`), structurally invisible to a CDS one-hot |
| ISONIAZID | 0.923 | **regulatory** | **+0.031** | *katG*, plus that same promoter |
| PYRAZINAMIDE | 0.910 | **protein / biophysical** | **+0.022** | *pncA* loss-of-function — hundreds of distinct inactivating substitutions, most unseen in training. "Does this substitution break the protein" generalises where memorising positions cannot. |
| CAPREOMYCIN | 0.855 | protein / biophysical | +0.013 | *rrs*/*rrl* **plus *tlyA***, which is protein-coding — so a small protein gain is expected here and not for AMK/KAN |
| EMB / LFX / STR / MOXI | — | marginal | +0.003 to +0.010 | point mutations whose positional signal a one-hot already captures — and inside single-seed noise |
| KANAMYCIN / AMIKACIN | 0.881 / 0.875 | **none apply** | +0.003 / +0.001 | target is *rrs*, 16S **rRNA** — no protein product for the descriptors to describe |
| RIFAMPICIN | 0.976 | **no headroom** | +0.001 | *rpoB* RRDR is a tight positional signal already at CV 0.976 |

> These are `full_run_v2` numbers and they are **substantially smaller than the
> `full_run` text still quoted in `build_full_run_viewer.py`** (PZA +0.099, ETO
> +0.094, INH +0.042). Nothing about the modalities changed — the corrected
> training schedule lifted the *DNA reference* (PZA DNA-only 0.824 → 0.910), and
> most of the old "modality gain" was undertrained-baseline. Only the two
> promoter drugs survive at a size worth reporting on one seed. The viewer's
> prose needs re-deriving against `full_run_v2`.

**2. Locus universe beats multi-task sharing.** "Joint models win" was
confounded — joint runs saw 19 loci, single-drug runs saw 2–3. Matched at 19
(`alllocus_run_v2`), the joint advantage is **≈0 or negative** under `mdcnn`
(+0.001 on DNA, −0.006 on all-modalities) and `cisfusion` (−0.008 / −0.006). It
survives only under `late_fusion` (+0.017 to +0.020) and even there it vanishes
at `all_modalities` (+0.001). What actually pays is the loci themselves:
**+0.016 to +0.027** for handing a single-drug model the full 19.

**3. Attribution share badly mis-ranks predictive value — and we falsified it
rather than believing it.** SHAP said the all-modalities SD-CNN spends ~1% of its
budget on DNA. Dropping DNA outright (`shap_followup/no_dna`) costs **≤0.009 CV
AUC on 7 of 11 drugs**, which is a correct, non-obvious, falsifiable prediction
about a trained model. The exceptions are exactly the rRNA-target drugs, where
the other three modalities are undefined on the locus that matters: KANAMYCIN
−0.362, AMIKACIN −0.322, CAPREOMYCIN −0.320 — with STREPTOMYCIN in between at
−0.059, which is what a drug with one protein-coding target (*rpsL*) and one
rRNA target (*rrs*) should do. Removing *regulatory* costs nothing
anywhere except ETHIONAMIDE (−0.079) and ISONIAZID (−0.030) — the two promoter
drugs again — while taking 6–23% of the budget.

**4. `setfusion` was never a capacity problem.** It is the smallest architecture
by two orders of magnitude (0.46 M params vs 46 M) and the weakest cell in every
sweep, but scaling four different width axes across 62 arms never closed the gap.
The diagnosis (`token_signal`) is that **0.14% of an encoded token varies with
the genotype** — the rest is a constant naming the locus — so attention is flat
at 1/8, the locus embedding is absorbed into the head's intercept, and plain
logistic regression on the encoder output beats the trained model by 0.011.
Fixing the *ratio* rather than the width works: `--token-norm keyed` or `--delta`
takes ISONIAZID from 0.9287 to **0.962**, level with `mdcnn`, at 1% of the
parameters.

## Status — what ran when

Newest first. Each entry names the run folder; every folder carries its own
`README.md` (the question, the grid, the caveats) and a `submit.sh` that
reproduces it.

| when | what | state |
|---|---|---|
| **2026-08-25** | six experimental **aggregators** built (`models/experimental_models.py`) — `catalogue` / `additive` / `noisyor` / `gatedpool` / `deepsets` / `fm`, sharing locusfusion's variant tokenizer | 🔨 **code only, zero training runs.** A controlled comparison of how to combine sparse evidence, on the premise that softmax normalisation is the wrong operator for a needle. Ships `variant_design_matrix()` for the sparse-linear / tree baseline this project has never had. `tests/test_experimental_models.py` 25/25. Items 6-7 under "Next". |
| **2026-08-25** | `--arch locusfusion` built — a variant-token, two-stage transformer (`models/locusfusion.py`) | 🔨 **code only, zero training runs.** One token per column that differs from H37Rv instead of one per patch of sequence; modalities fused within a locus, then loci fused; a learned `[WT]` sentinel per locus. `CODE_CHANGES_20260825.md`, `tests/test_locusfusion.py` 25/25. The run it wants is item 5 under "Next". |
| **2026-08-19** | `transformer_run` (165 jobs) and `alllocus_run_v2` (220 jobs) finish | ✅ complete, **not yet written up** — see "Next" |
| **2026-08-18** | MD-CNN reproduction completes (fold 4 + merge + eval, job `63188024`) | ✅ **macro CV 0.9212 vs the authors' 0.9222** over the 11 shared drugs; test 0.8857 vs 0.8840. Reproduced. |
| **2026-08-18** | `transformer_run` submitted — the `full_run_v2` grid with a self-attention encoder, at matched parameter count | needed new code first: a tunable transformer branch encoder and `MDCNNTransformerTrunk` (MD-CNN's layer 1 *is* a patch embedding once its stride equals its kernel width). `CODE_CHANGES_20260818.md`, `tests/test_transformer_encoder.py` 13/13. |
| **2026-08-18** | `alllocus_run_v2` submitted — the single-drug grid on all 19 loci, at the `full_run_v2` method | replaces `alllocus_run`, which ran at superseded settings, checkpointed nothing, lost 10 of 220 jobs, and then lost its results folder outright (rebuilt from SLURM logs) |
| **2026-08-13** | SHAP attribution built and run: `scripts/shap_attribution.py` (11 drugs, SD-CNN) + `scripts/shap_multidrug.py` (per-output attribution for the joint model), written up in `notebooks/shap_notebook.ipynb` | ✅ done. GradientExplainer, sign-flipped so positive = toward RESISTANT, every model re-scored against its own `summary.csv` before being explained. |
| **2026-08-13** | `shap_followup` (33 jobs) — the leave-one-out arm the ablation ladder was missing (`no_dna`, `no_regulatory`, `regulatory_only`) | ✅ complete; finding 3 above |
| **2026-08-13** | Repo cleanup — 154 MB → 57 MB, `scripts/` 20 files → 7, per-experiment tooling retired | ✅ `CLEANUP_REPORT.md`. Established the rule below: a script written for one experiment lives with that run and dies with its write-up. |
| **2026-08-06 → 08-11** | `setfusion_scaling` (62 arms) then `token_signal` (4 arms) — is setfusion small, or shaped wrong? | ✅ **shaped wrong**; finding 4 above |
| **2026-08-06** | `joint_convergence` (6 jobs) — were the joint models undertrained at 150 epochs, or is the LR too low? | ✅ **neither**. 400 epochs bought +0.000; `lr 1e-3` collapsed the cell to 0.735 (0.583 with regularization). BIG-TB's `exp(-9)` is correct. The cap was binding on `best_epoch` but worth ~nothing in AUC. |
| **2026-08-06** | `joint_capacity` (8 jobs) — does an 11-task head have enough capacity? | ✅ **not the bottleneck**. No arm beat its control: `--hidden 512` −0.004, per-drug heads −0.001, and dropout + weight decay clearly *hurt* (−0.046 cisfusion). |
| **2026-08-06** | `full_run_v2` (240 jobs) — **the project's baseline**. Same grid and identical inputs as `full_run`; 300 epochs / patience 30 / 50-epoch warmup, and `--save-weights best` | ✅ complete. Everything since is measured cell-for-cell against it. |
| **2026-08-06** | `alllocus_run` (221 jobs) — first attempt at the locus-vs-sharing question | ⚠️ superseded by `alllocus_run_v2`; results folder lost and partially rebuilt from SLURM logs |
| **2026-08-04** | `full_run` (240 jobs) — first complete 4 arch × 5 modality × 2 scope sweep, 150 epochs | ⚠️ **superseded**: checkpointed nothing, 150 was a binding cap on 40% of joint folds, patience 15 fired spuriously, and setfusion never escaped its degenerate init. Kept in full as the matched control for `alllocus_run` / `joint_*`. |
| **2026-07-28** | Baseline alignment — matched the reference protocol flag for flag (LR `exp(-9)`, batch 128, masked weighted BCE, per-drug inverse-frequency alpha on the train split, 5-fold shuffled KFold, `256→256→sigmoid` head) | ✅ `reference_docs/BASELINE_ALIGNMENT_CHANGES.md`. Found the **SD-CNN train/test leak** while doing it (`h1_repro/`, `BASELINE_LEAK_FINDING.md`) — the published single-drug baseline is not a fair target and every comparison since uses the corrected numbers. |
| **2026-07-01 → 07-20** | Pipeline build — the modality-aware data layer, the four architectures, the single- and multi-drug training engines | ✅ |

### Next

Where the project actually is: three large runs are finished and unanalysed.
In rough priority order —

1. **`alllocus_run_v2` × `full_run_v2` joint cells** — the locus-vs-sharing table
   is computable today and is the run's whole purpose. Its best cell,
   `all_modalities__mdcnn` at **0.9246**, is the highest number in the project
   and sits above the MD-CNN baseline; that needs stating carefully, since these
   models are no longer locus-matched to SD-CNN and so cannot be quoted against
   it.
2. **`transformer_run` × `full_run_v2`** — attention loses everywhere so far
   (DNA-only 0.802 vs 0.892 for the CNN; all-modalities 0.880 vs 0.909), but the
   parameter ratio runs 0.70–1.34 across cells, so only the `dna_protein` /
   `dna_biophysical` rows are a fair read. One job is missing (ETHAMBUTOL,
   `dna_protein__late_fusion` — the widest cell, the OOM/timeout risk its README
   called out).
3. **`build_full_run_viewer.py`'s prose is still `full_run`-era** and overstates
   the modality gains by 3–4× (see the note under finding 1). It is the single
   source of the baseline tables, so the numbers *it computes* are fine — it is
   the hardcoded narrative that needs re-deriving against `full_run_v2`.
4. **The `token_signal` fix has not been promoted.** `--token-norm keyed` is
   worth +0.033 on the drug the project's core question is about, from a
   one-line change, and it has only ever run on two drugs in one cell.
5. **`--arch locusfusion` is built and tested but has never been trained**, so
   it has no number attached to it and must not be quoted as if it did. It is
   the design that follows from findings 3 and 4 taken together: attribution
   share mis-ranks predictive value, and setfusion failed on the
   signal-to-constant *ratio* rather than on width — so tokenize the **variants**
   rather than the sequence. The run, matched to `full_run_v2` in everything but
   `--arch`:

   ```bash
   python scripts/run_experiment.py --modalities dna protein biophysical regulatory \
       --drugs all --arch locusfusion --epochs 300 --patience 30 --min-epochs 50 \
       --device cuda --run-name locusfusion_v1
   ```

   Judge it on three things, not one. **(a)** `cv_auc_mean` against the same cell
   in `full_run_v2` — the bar is `all_modalities__mdcnn` at 0.9086 (per-drug
   loci) / 0.9246 (19 loci). **(b)** the read-out attention must stop being
   uniform; flat 1/n_loci is the same collinearity that killed setfusion.
   **(c)** the attended tokens must be the *right* ones — `variant_report()`
   names the locus and alignment column behind each, so for ISONIAZID the top
   token should be katG codon 315 or the fabG1–inhA promoter `c-15t`. An arm
   that raises AUC while attending to neither improved for some other reason.
   Then the knobs in evidence order: `--lf-summary-norm`,
   `--lf-carry-variants 2`, `--lf-locus-encoder per_locus`, and `--lf-d-model`
   last — `setfusion_scaling` already showed width is not the lever.
6. **The sparse baseline has never been run, and it should come before item 5.**
   `variant_design_matrix()` exports the isolate × variant matrix; an L1-logistic
   and a LightGBM on it cost an afternoon and no GPU. If they reach ~0.92 that
   reframes the project, and it is the cheapest possible way to find out.
7. **The six experimental aggregators are built and tested, none trained.** They
   are a controlled comparison — one tokenizer, six ways of combining the tokens
   — so they want a single grid, not six separate runs. The three readings that
   matter: `catalogue` vs `additive` measures what featurising a variant buys on
   unseen substitutions; `deepsets` vs `gatedpool` vs `locusfusion` measures
   whether attention or its normalisation was ever the problem; `additive` vs
   `fm` measures whether epistasis is worth anything.

Standing gaps, unchanged: single seed everywhere (multi-seeding the headline
cells is a prerequisite for reporting any of this), attribution is unconverged
below `NSAMPLES` ~128, and alignment columns are not H37Rv coordinates outside
protein blocks.

### Working conventions

- **A script written for one experiment lives with that run and is deleted once
  the run is written up.** `scripts/` holds only what is general to the project.
  The run's `README.md` under `results/experiments/` is what preserves the
  finding — not the script that produced it.
- **Every run folder gets a `README.md` and a `submit.sh`** — the question it
  asks, its matched control, what it deliberately does *not* do, and the exact
  command that reproduces it.
- **`results/` and `slurm_logs/` are git-ignored**, so nothing in them is
  recoverable from git (this is how `alllocus_run` was lost). Write-ups and
  reproduction code are tracked; results and models never are.
- **Name the matched control explicitly.** A run that changes more than one thing
  against its control is not comparable to it, and several of the entries above
  exist only because that rule was broken once.

## Structure

```
datasets/                 modality-aware data layer (the single dataloader)
  loader.py               load_dataset(drug, modalities, …) -> DrugData; MODALITIES registry
  multidrug.py            load_multidrug_dataset(drugs, …) -> MultiDrugData (label MATRIX)
  dna.py                  one-hot nucleotides, gene loci concatenated        (N, 5, L)
  protein.py              one-hot amino acids, one block per gene            (N, 20, K)
  biophysical.py          MW / pI / hydrophobicity per residue, per gene     (N, 3, K)
  regulatory.py           per-drug regulatory regions, WHO-catalogue-driven      (N, 5, L)
  sequences.py            shared FASTA + phenotype loading, vectorized one-hot
  biochem.py              AA property tables, genetic code, translation, featurizers
  who_catalogue.py        WHO 2023 catalogue Tables 21 & 22 (candidate genes + promoter regions)
  base.py                 Modality / FeatureBlock / LoadContext interface
  fixtures.py             synthetic genotype/phenotype (+ regulatory) generator

models/                   architectures (everything re-exported from `models`)
  net.py                  ConvBranch / DenseHead, ENCODERS registry (cnn / transformer),
                          MultiModalNet + MultiDrugNet (late fusion), MDCNNNet (BIG-TB
                          topology), SetFusionNet (shared encoders, locus-keyed set
                          fusion), CisFusionNet (promoter ⊕ CDS cis-units)
  locusfusion.py          LocusFusionNet — one token per VARIANT, fused within a
                          locus then across loci; needs --delta input
  experimental_models.py  six variant-set AGGREGATORS on that same tokenizer —
                          catalogue / additive / noisyor / gatedpool / deepsets /
                          fm — plus variant_design_matrix() for the sparse-linear
                          and tree baselines that have never been run here
  legacy.py               earlier variants, not in the live path: DNAOnlyCNN /
                          LateFusionCNN / EarlyFusionCNN / CrossAttentionFusionCNN

training/                 training engines
  core.py                 masked weighted BCE + EarlyStopper (shared primitives)
  multimodal.py           single-drug mini-batched CV + held-out-test engine
  multidrug.py            multi-drug (MD-CNN style) engine, macro-AUC early stopping
  curves.py               per-epoch loss / val-metric plots; also a re-plot CLI

scripts/                  entry points (run from the project root). Kept
                          deliberately small: per-experiment sweep, analysis and
                          monitoring scripts are NOT added here — they live with
                          the run, and are deleted once it is written up.
  run_experiment.py       single-drug CLI — pick modalities, drugs, real/synthetic
  run_multidrug.py        multi-drug CLI — all drugs in one MultiDrugNet
  sbatch_all_runs.py      submit the experiment grid — one job per (experiment ×
                          drug), or one per experiment with --multidrug
  trace_models.py         push one real isolate through every net, diagram the dataflow
  build_full_run_viewer.py  builds <run>/full_run_viewer.ipynb for any full_run-style
                          sweep — ONE run, in depth
  build_overview.py       builds notebooks/overview.ipynb — project status:
                          one modality x model grid per task, across every run.
                          Tasks/models/modality sets are derived from the result
                          JSONs, so a new run needs no edit here
  build_datasets_overview.py  regenerates notebooks/datasets_overview.ipynb
  shap_attribution.py     per-drug SHAP for the single-drug models
  shap_multidrug.py       per-output SHAP for the joint model (on-target scoring)
  sbatch/trace_models.sh  standalone job script for the dataflow traces

notebooks/                all generated except one — edit the builder, not the .ipynb:
                          overview (status: task x modality x model),
                          datasets_overview (modality tour), shap_notebook (hand-
                          written), biophysical_properties_rdkit
tests/                    test_baseline_alignment.py — SD-CNN protocol checks
                          test_checkpoint.py / test_setfusion.py / test_cisfusion.py
                          test_transformer_encoder.py / test_locusfusion.py
                          test_experimental_models.py
bigtb_ref.py              imports BIG-TB utilities + real data paths (REAL_*)
bigtb_baselines.py        the two BIG-TB baseline tables — leak-corrected SD-CNN and
                          MD-CNN read from the authors' run log. Single source of
                          truth; both notebook builders import it, never copy it.
TODO.md                   status, next steps, open questions
results/                  run outputs (git-ignored). One folder per run, each with its
                          own README.md + submit.sh; results/archive/ = superseded runs
slurm_logs/               job logs (git-ignored); slurm_logs/manifests/ = the job_id →
                          run provenance record, see its README
diagrams/                 model/architecture figures
```

**Data flow:** `load_dataset(drug, modalities)` does the shared work once (resolve
loci, load aligned FASTAs, align isolates to phenotype), runs each requested
modality, and returns a `DrugData` bundle of feature **blocks**. Each block →
one branch of a `MultiModalCNN`. Adding a modality = one entry in
`datasets/loader.py:MODALITIES`.

## Environment

```bash
conda activate abr_env          # Python 3.12, torch 2.6.0+cu124, GPU-enabled
```

Real data lives on UMass Unity (`pi_annagreen` allocation); paths are wired into
`bigtb_ref.REAL_GENOTYPE_DIR` / `REAL_PHENOTYPE_CSV`. Synthetic mode needs no
cluster access.

## Running the pipeline

`scripts/run_experiment.py` runs BIG-TB's SD-CNN protocol (fixed 0.2 test split seed 42,
5-fold KFold on train, masked weighted BCE + alpha class weighting, sens/spec
threshold search) for each drug, mini-batched, and reports a held-out test
metric. Results go to `results/experiments/{run}/{DRUG}__{modality-tag}.json`
(+ `summary.csv`), named by drug **and** modality set.

```bash
# --- quick wiring check, no cluster needed (synthetic data, meaningless numbers) ---
python scripts/run_experiment.py --synthetic --modalities dna biophysical regulatory \
    --drugs ISONIAZID --epochs 5 --n-splits 3 --device cpu

# --- DNA-only baseline, all drugs, on the real data (GPU) ---
python scripts/run_experiment.py --modalities dna --drugs all --epochs 60 --device cuda --run-name dna_full

# --- every modality, every drug (pass 'all' to either flag) ---
python scripts/run_experiment.py --modalities all --drugs all --device cuda --run-name full_sweep

# --- the core experiment: DNA vs. DNA+biophysical on the multi-locus drugs ---
python scripts/run_experiment.py --modalities dna            --drugs ISONIAZID RIFAMPICIN --device cuda --run-name inh_rif_dna
python scripts/run_experiment.py --modalities dna biophysical --drugs ISONIAZID RIFAMPICIN --device cuda --run-name inh_rif_fusion

# --- everything, one drug (all four modalities where available) ---
python scripts/run_experiment.py --modalities dna protein biophysical regulatory \
    --drugs ISONIAZID --epochs 60 --device cuda --run-name inh_all

# --- pick which / how many loci to load ---
python scripts/run_experiment.py --modalities dna --drugs ISONIAZID --loci katG        # just katG
python scripts/run_experiment.py --modalities dna regulatory --drugs KANAMYCIN \
    --regulatory-loci eis                                                      # eis promoter only

# --- pick the model per modality (CNN vs Transformer) ---
python scripts/run_experiment.py --modalities dna protein biophysical --drugs ISONIAZID \
    --encoders dna=cnn protein=transformer biophysical=cnn --device cuda
```

Key flags: `--modalities` (any subset of `dna protein biophysical regulatory`,
or `all`), `--drugs` (default INH + RIF, or `all` for every drug), `--loci`
(which gene loci; default the drug's
`DRUG_TO_LOCI`), `--regulatory-loci` (which regulatory regions; default the
WHO regions **for the loaded loci** — see below), `--encoders` (`MODALITY=TYPE`, e.g.
`protein=transformer`) + `--default-encoder`, `--real`/`--synthetic` (default
real), `--epochs`,
`--n-splits`, `--batch-size`, `--device`, `--run-name`, `--tb` (log to
TensorBoard). Modalities/loci with no data for a drug (e.g. regulatory for
rifampicin) are dropped with a warning and the output tag reflects what was
actually used.

### Which loci a run sees

The two reference codebases choose loci two different ways, and the difference
is exactly one gene:

- **SD-CNN** uses a **per-drug** map (`tb.DRUG_TO_LOCI`) — INH = inhA+katG, etc.
  This is the default for `run_experiment.py`, and it keeps a single-drug run
  locus-matched to the SD-CNN baseline.
- **MD-CNN** ignores that map and feeds **every curated locus** to every drug
  (its flat `parameters/locus_order.py`, 19 entries). `run_multidrug.py` now
  defaults to the same rule on real data — `datasets.loci_on_disk(genotype_dir)`,
  every `*.fasta` there — so the multi-drug model is comparable to it and picks
  up any locus curated later for free. `--per-drug-loci` selects the per-drug
  union instead (18).

`fabG1` is the gene that falls between them: it has a FASTA on disk, WHO 2023
Table 21 lists it **tier 1** for INH and ETO (it heads the *fabG1–inhA* operon
whose promoter carries the dominant non-*katG* INH/ETO mechanism), but no drug
in `DRUG_TO_LOCI` names it. `datasets.EXTRA_LOCI` is the opt-in overlay:

```bash
python scripts/run_experiment.py --modalities dna --drugs ISONIAZID --extra-loci   # inhA katG fabG1
python scripts/run_multidrug.py  --modalities dna --drugs all                      # 19 loci (default)
python scripts/run_multidrug.py  --modalities dna --drugs all --per-drug-loci      # 18, the old union
```

`--extra-loci` is **off by default**: on, a single-drug run is no longer
locus-matched to SD-CNN, so it belongs to its own experiment (see the
`dna_mdcnn_extraloci` entry in `scripts/sbatch_all_runs.py`). Any requested
locus without a FASTA is reported and skipped, whatever the source.

**A run never carries more promoter windows than coding loci.** WHO's candidate
list per drug is much longer than `DRUG_TO_LOCI` (12 promoters for INH against 2
loci), and most of those promoters belong to genes whose CDS is never loaded —
`ahpC`, `ndh`, `mshA`, efflux loci. So the region set is intersected with the
loci actually loaded: INH gets `inhA` + `katG`, and the multi-drug union drops
from 48 regions to 16. `--all-regulatory` keeps the full WHO set; naming regions
with `--regulatory-loci` overrides both.

The strictness has one known cost: **KANAMYCIN keeps only `rrs` and loses the
`eis` promoter**, because `eis` is not one of its coding loci even though the
*eis* promoter is its best-known regulatory mechanism. AMIKACIN keeps it (`eis`
is a coding locus there), and so do multi-drug runs (it is one of the 19).

**The fabG1 promoter is a separate thing from the fabG1 gene body, and it is
already covered.** WHO files the *fabG1–inhA* operon promoter under `inhA`, but
keys it to tss=1,673,440 — fabG1's own CDS start (Rv1483 1673440–1674183; inhA
starts 762 bp later). So the `inhA` regulatory window *is* that promoter, and any
run including the `regulatory` modality already sees `c-15t`. `regulatory_msa`
emits the identical window under both names so `fabG1` can be requested on its
own (`--regulatory-loci fabG1`); `datasets.regulatory.REGION_ALIASES` keeps the
alias out of the per-drug defaults when `inhA` is present, so no run gets the
same 873 bp twice.

### Architectures (`--arch`)

Four network topologies, selected per run and recorded in each result JSON
(`arch`, `n_params`):

- **`late_fusion`** (default) — one encoder per feature block, all encoder
  outputs concatenated into a shared dense head. Per-block encoders are
  pluggable (`--encoders`, see below).
- **`mdcnn`** — BIG-TB's own SD-CNN / MD-CNN topology (`models.MDCNNNet`).
  Every locus becomes a **channel** on one shared zero-padded position axis and
  layer 1 is a `(n_channels x 12)` conv across all of them, so loci mix from the
  first layer instead of only at the flatten. On BIG-TB's 19-locus DNA input
  this reproduces their exact shapes: 73,024 layer-1 parameters, a 14,336-wide
  flatten, ~3.9 M parameters total (vs ~35 M for `late_fusion` on the same
  input, nearly all of it in one FC layer). `--arch mdcnn` implies per-locus
  branches — it must see the loci separately — and ignores `--encoders`.
  Mixed-modality runs get one trunk per channel-height group (DNA 5ch, protein
  20ch, …), since one kernel cannot span different channel heights.
- **`setfusion`** — `models.SetFusionNet`. One encoder **shared per modality**
  (every regulatory window goes through the same promoter encoder), each block
  becoming a token that carries learned modality and locus embeddings. A
  transformer fuses the set and one attention query per drug reads it out, so
  block count and order stop mattering and `dna:katG` can be matched with
  `regulatory:katG` by attention rather than by position. The cost is a pooled
  summary instead of a flatten: absolute position within a locus is coarsened to
  4 relative bins.
- **`cisfusion`** — `models.CisFusionNet`. Rebuilds the blocks into one branch
  per **locus**, concatenating that locus's promoter and CDS into a single
  nucleotide sequence with a 6th channel marking which segment each column
  belongs to, then encodes per branch. Without the promoter there is nothing to
  pair, so it needs the `regulatory` modality to be doing its job.
- **`locusfusion`** — `models.LocusFusionNet`. One token per **variant**, not
  per patch of sequence: it runs on reference-difference input and emits a token
  only where the isolate deviates from H37Rv, so 100% of a token varies with the
  genotype by construction. Two stages — all of a locus's modalities fuse into
  one locus representation (stage 1), then the locus representations fuse across
  genes (stage 2) — and each locus carries a learned `[WT]` sentinel, so **a
  susceptible isolate is the empty set**. Exact position survives (a sinusoidal
  encoding of a continuous codon coordinate, not a pooled bin), the input is
  O(variants) rather than O(sequence length), and the read-out attention names
  the locus and column it read. Implies `--delta`; see
  `results/experiments/CODE_CHANGES_20260825.md` for the design and the variant
  census it rests on.

**The experimental family** (`models/experimental_models.py`, 2026-08-25) — six
more `--arch` values that share `locusfusion`'s variant tokenizer and differ
**only** in how the variant set is aggregated. They exist because the census says
the problem is *sparse evidence aggregation*, not sequence encoding, and softmax
normalisation is the wrong operator for a needle: with one informative token and
thirteen neutral ones the weights must sum to 1, so attention has to spend mass
on the neutral tokens.

| `--arch` | aggregator | the question it asks |
|---|---|---|
| `catalogue` | learned scalar per exact variant id | how far does pure memorisation get? (= logistic regression on the variant matrix) |
| `additive` | `sum w(features)` | does featurising the variant buy generalisation to unseen substitutions? |
| `noisyor` | `1 − Π(1 − p_v)` | "susceptible unless something confers resistance", as an architecture |
| `gatedpool` | sigmoid gate, no softmax | is normalisation what broke attention here? |
| `deepsets` | sum + max + count | does attention buy anything at all over plain additivity? |
| `fm` | factorization machine | is epistasis worth anything, at O(T·k) instead of O(T²)? |

`catalogue` and `additive` are a matched pair: `catalogue` **cannot** score a
variant it never saw, `additive` scores it from features, and the gap between
them measures exactly the *pncA* generalization in finding 1. `additive` is also
its own attribution method — `contributions()` sums to the logit exactly.

`variant_design_matrix()` in the same module exports the sparse isolate × variant
matrix so an L1-logistic or a gradient-boosted tree can be fitted on it with no
GPU. **That baseline does not exist anywhere in this project and should be run
before any of the networks above.**

`mdcnn`, `setfusion`, `cisfusion`, `locusfusion` and all six experimental archs
require per-locus blocks; `--arch` implies that. `locusfusion` and the six also
imply `--delta`.

```bash
python scripts/run_experiment.py --modalities dna --drugs all --arch mdcnn --epochs 150 --device cuda
python scripts/run_experiment.py --modalities all --drugs all --arch locusfusion \
    --epochs 300 --patience 30 --min-epochs 50 --device cuda   # --delta implied
python scripts/run_multidrug.py  --modalities dna --drugs all --arch mdcnn --epochs 150 --device cuda
python scripts/sbatch_all_runs.py --experiments dna_mdcnn --drugs all --epochs 150   # one job per drug
```

### Models (per-modality encoders)

Each feature block is encoded independently, then all encoder outputs are
concatenated into a shared dense head (`models.MultiModalNet`, **late fusion**).
The per-block architecture is pluggable via the `ENCODERS` registry:

- **`cnn`** (default) — `ConvBranch` 1D-CNN, a strong local motif detector.
  Good default for DNA / regulatory / biophysical.
- **`transformer`** — ViT-style patch embedding (a strided conv chunks the
  sequence into ~L/9 tokens, keeping attention tractable on long sequences) +
  a small Transformer encoder, mean-pooled. Captures long-range interactions
  (e.g. epistatic residue pairs) a local CNN can miss — a candidate for protein.

Choose per modality with `--encoders MODALITY=TYPE` (all blocks of a modality
share its type); modalities not named use `--default-encoder` (cnn). The chosen
encoders are printed and recorded in each result JSON. Add a new per-block
encoder by registering it in `models.ENCODERS` — nothing else changes. These are
the per-block encoders used by `late_fusion`; `--encoders` is ignored by the
other three architectures, which define their own encoding.

### Multi-drug runs

`scripts/run_multidrug.py` trains ONE `MultiDrugNet` over the union of every
drug's loci with one sigmoid output per drug (BIG-TB's MD-CNN shape), so loci
that inform several drugs share features. Same flags as `run_experiment.py`;
results go to `results/experiments/{run}/multidrug__{tag}.json` (+
`multidrug_summary.csv`, one row per drug plus MACRO).

```bash
python scripts/run_multidrug.py --modalities dna --drugs all --device cuda --run-name multidrug_dna_all
sbatch scripts/sbatch/multidrug_all.sh        # the same, as a SLURM job
```

### Training curves (is the epoch cap enough?)

Every CV fold records its per-epoch train loss and early-stopping metric into
the result JSON (`cv_folds[i]["history"]`), and each run writes a
`*_curves.png` beside its JSON: train loss on the left, val AUC (or val loss)
on the right, one line per fold, best epoch dotted. If the val panel is still
climbing at the right edge, or the best epochs cluster near the cap, raise
`--epochs`. Re-plot a finished run without retraining:

```bash
python training/curves.py results/experiments/{run}          # a whole run folder
python training/curves.py results/experiments/{run}/ISONIAZID__dna.json
```

### Seeing what a model does to a sample

`scripts/trace_models.py` pushes ONE real isolate through every network we
train and writes, per model, a flow diagram (`.png`) plus a complete text trace
(`.txt`) into `diagrams/model_traces/`: input blocks -> per-branch stacks ->
fusion -> dense head -> logits, with the tensor shape, op signature (kernel,
stride, padding, channels), parameter count and output statistics at **every**
step. Twenty traces cover the live matrix — single-/multi-drug × single-/multi-
modal × every architecture. The seven variant-token traces (`*_locusfusion`,
`*_catalogue` and the four other aggregators) get their own `--delta` load and
open with the tokenizer instead of a conv stack: per block, how many of its
columns this isolate differs from H37Rv at, and the <=16 tokens that survive.

```bash
python scripts/trace_models.py --synthetic          # fast wiring check on fixtures
python scripts/trace_models.py                      # all traces, real data
python scripts/trace_models.py --traces sd_dna sd_dna_mdcnn --drug RIFAMPICIN
sbatch scripts/sbatch/trace_models.sh               # all 20 (the multi-drug load wants ~64G)
```

Weights are freshly initialised — we never checkpoint — so the logits are noise;
the shapes, parameter counts and dataflow are the point, and the sample is real
so the input side is exactly what the training jobs see. Figures show up to
`--max-branches` representative branches (one per modality first), and the
`.txt` always lists every branch.

### Running many jobs on SLURM

`scripts/sbatch_all_runs.py` submits **one sbatch job per (experiment × drug)**,
so each run gets its own `--mem`/GPU allocation and a single-drug memory
footprint — that isolation is what stops runs from crashing each other (the
all-drugs-in-one-process host-RAM growth that OOM-killed an interactive
`--drugs all`). Experiments (modality/encoder configs) are defined in the
`EXPERIMENTS` dict at the top of the script; all its per-drug jobs share one
`--run-name` folder.

```bash
python scripts/sbatch_all_runs.py --dry_run                       # print scripts, submit nothing
python scripts/sbatch_all_runs.py --experiments dna --drugs all   # DNA baseline, every drug
python scripts/sbatch_all_runs.py --mem 48G --time 06:00:00       # full sweep (all experiments × drugs)
```
Tune `--mem / --gpus / --time / --partition / --constraint` for the cluster.
Note: batch size caps **GPU** memory; the `Killed` OOM was **host** RAM during
data loading, which `--mem` (per isolated job) is what actually addresses.

### Live monitoring

Add `--tb`, then:

```bash
tensorboard --logdir results/experiments      # loss curves per fold/test + HPARAMS tab
```

### Inspecting the data

Open `notebooks/datasets_overview.ipynb` (runs on synthetic fixtures by default; set
`USE_REAL = True` for cluster data) for per-modality shapes, class balance,
one-hot heatmaps, gap/length distributions, and biophysical property histograms.

Or from Python:

```python
from datasets import load_dataset
from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV

data = load_dataset("ISONIAZID", ["dna", "biophysical"],
                    REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV)
data.n                # 17942
data.class_counts()   # {'R': 5825, 'S': 11611, 'missing': 506}
data.branch_specs()   # [(5, 3398), (3, 269), (3, 432)]
```

## Notes

- **Regulatory** regions are defined per-drug from the **WHO 2023 catalogue**
  (`datasets/who_catalogue.py`, Tables 21 & 22): the default set for a drug is
  its WHO candidate genes minus its primary coding loci. On the real data the
  available regions load — *fabG1* (fabG1–inhA operon promoter) for INH/ETO,
  *eis* (eis promoter) for KAN — the rest are skipped until their FASTAs are
  curated. Table-22 upstream coordinates + TSS are attached to each block as
  metadata (a hook for true promoter-slicing later). Override with
  `--regulatory-loci`.
- **Biophysical** property values are standard published tables (z-scored), a
  stand-in until Kulkarni et al.'s exact table is confirmed (see `TODO.md`).
- See `TODO.md` for status, the core DNA-vs-fusion comparison, and open
  methodological questions (lineage confounding, stratified CV).
