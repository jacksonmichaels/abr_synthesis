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
  dna.py                  one-hot nucleotides, gene loci concatenated        (N, 5, L)
  protein.py              one-hot amino acids, one block per gene            (N, 20, K)
  biophysical.py          MW / pI / hydrophobicity per residue, per gene     (N, 3, K)
  regulatory.py           per-drug regulatory regions, WHO-catalogue-driven      (N, 5, L)
  sequences.py            shared FASTA + phenotype loading, vectorized one-hot
  biochem.py              AA property tables, genetic code, translation, featurizers
  who_catalogue.py        WHO 2023 catalogue Tables 21 & 22 (candidate genes + promoter regions)
  base.py                 Modality / FeatureBlock / LoadContext interface

models.py                 encoders (CNNEncoder / TransformerEncoder + ENCODERS registry),
                          MultiModalNet (generic late fusion, per-branch encoder choice),
                          also DNAOnlyCNN / LateFusionCNN / EarlyFusionCNN / CrossAttention
train_multimodal.py       generic mini-batched CV + held-out-test engine (any modality set)
run_experiment.py         CLI entry point — pick modalities, drugs, real/synthetic
eval_dna_cnn.py           standalone DNA-only baseline (kept for continuity)

fixtures.py               synthetic genotype/phenotype (+ regulatory) generator
datasets_overview.ipynb   visual tour of each modality's shapes & coverage
bigtb_ref.py              imports BIG-TB utilities + real data paths (REAL_*)
data.py                   thin legacy adapter over datasets/ (used by eval_dna_cnn)
TODO.md                   status, next steps, open questions
results/                  run outputs (git-ignored)
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

`run_experiment.py` runs BIG-TB's SD-CNN protocol (fixed 0.2 test split seed 42,
5-fold KFold on train, masked weighted BCE + alpha class weighting, sens/spec
threshold search) for each drug, mini-batched, and reports a held-out test
metric. Results go to `results/experiments/{run}/{DRUG}__{modality-tag}.json`
(+ `summary.csv`), named by drug **and** modality set.

```bash
# --- quick wiring check, no cluster needed (synthetic data, meaningless numbers) ---
python run_experiment.py --synthetic --modalities dna biophysical regulatory \
    --drugs ISONIAZID --epochs 5 --n-splits 3 --device cpu

# --- DNA-only baseline, all drugs, on the real data (GPU) ---
python run_experiment.py --modalities dna --drugs all --epochs 60 --device cuda --run-name dna_full

# --- every modality, every drug (pass 'all' to either flag) ---
python run_experiment.py --modalities all --drugs all --device cuda --run-name full_sweep

# --- the core experiment: DNA vs. DNA+biophysical on the multi-locus drugs ---
python run_experiment.py --modalities dna            --drugs ISONIAZID RIFAMPICIN --device cuda --run-name inh_rif_dna
python run_experiment.py --modalities dna biophysical --drugs ISONIAZID RIFAMPICIN --device cuda --run-name inh_rif_fusion

# --- everything, one drug (all four modalities where available) ---
python run_experiment.py --modalities dna protein biophysical regulatory \
    --drugs ISONIAZID --epochs 60 --device cuda --run-name inh_all

# --- pick which / how many loci to load ---
python run_experiment.py --modalities dna --drugs ISONIAZID --loci katG        # just katG
python run_experiment.py --modalities dna regulatory --drugs KANAMYCIN \
    --regulatory-loci eis                                                      # eis promoter only

# --- pick the model per modality (CNN vs Transformer) ---
python run_experiment.py --modalities dna protein biophysical --drugs ISONIAZID \
    --encoders dna=cnn protein=transformer biophysical=cnn --device cuda
```

Key flags: `--modalities` (any subset of `dna protein biophysical regulatory`,
or `all`), `--drugs` (default INH + RIF, or `all` for every drug), `--loci`
(which gene loci; default the drug's
`DRUG_TO_LOCI`), `--regulatory-loci` (which regulatory regions; default the
WHO-derived per-drug set), `--encoders` (`MODALITY=TYPE`, e.g.
`protein=transformer`) + `--default-encoder`, `--real`/`--synthetic` (default
real), `--epochs`,
`--n-splits`, `--batch-size`, `--device`, `--run-name`, `--tb` (log to
TensorBoard). Modalities/loci with no data for a drug (e.g. regulatory for
rifampicin) are dropped with a warning and the output tag reflects what was
actually used.

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
encoders are printed and recorded in each result JSON. Add a new architecture by
registering it in `models.ENCODERS` — nothing else changes. All-CNN reproduces
the previous behavior exactly, so existing results are unaffected.

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

Open `datasets_overview.ipynb` (runs on synthetic fixtures by default; set
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
