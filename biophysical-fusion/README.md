# Biophysical Fusion — ABR prediction

Genotype→phenotype antibiotic-resistance (R/S) prediction for *M. tuberculosis*,
built on top of the BIG-TB benchmark. The core question: does fusing amino-acid
**biophysical traits** (Kulkarni et al. 2026) onto BIG-TB's DNA CNN improve
binary resistance classification? The pipeline is **modality-selectable** — DNA,
protein, biophysical, and regulatory inputs can be combined in any subset.

Nothing here modifies `../Big-TB-benchmark/`; reference code is only *read*
(single point of contact: `bigtb_ref.py`).

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
                          sweep; also holds the BIG-TB baseline tables (single source)
  build_datasets_overview.py  regenerates notebooks/datasets_overview.ipynb
  sbatch/trace_models.sh  standalone job script for the dataflow traces

notebooks/                datasets_overview (modality tour, generated — edit the
                          builder, not the .ipynb), biophysical_properties_rdkit
tests/                    test_baseline_alignment.py — SD-CNN protocol checks
                          test_checkpoint.py / test_setfusion.py / test_cisfusion.py
bigtb_ref.py              imports BIG-TB utilities + real data paths (REAL_*)
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

`mdcnn`, `setfusion` and `cisfusion` all require per-locus blocks; `--arch`
implies that.

```bash
python scripts/run_experiment.py --modalities dna --drugs all --arch mdcnn --epochs 150 --device cuda
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
step. Eleven traces cover the live matrix — single-/multi-drug × single-/multi-
modal × all four architectures:

```bash
python scripts/trace_models.py --synthetic          # fast wiring check on fixtures
python scripts/trace_models.py                      # all traces, real data
python scripts/trace_models.py --traces sd_dna sd_dna_mdcnn --drug RIFAMPICIN
sbatch scripts/sbatch/trace_models.sh               # all 7 (the multi-drug load wants ~64G)
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
