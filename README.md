# ABR synthesis: training and evaluation code

This repo predicts antibiotic resistance (R/S, 11 drugs) in *M. tuberculosis* from
genotype. It holds the code for a full training and evaluation run of four models:

| model | entry point | input |
|---|---|---|
| Stacked CNN (MD-CNN topology) | `stacked_cnn/train.py`, `stacked_cnn/train_multidrug.py` | aligned loci as one-hot channels: DNA, protein, biophysical, regulatory |
| LocusFusion v5_fixed | `locusfusion/train.py` | variant tokens from the 19 loci + promoter windows + binding-site proximity |
| GenomeFusion | `genomefusion/train.py` | variant tokens over the whole genome (6,776 loci), from the VCFs |
| LASSO | `lasso/lasso.py` | L1-logistic on the one-hot alignment |

All four read the same split files and report the same metrics: CV AUC over 5
folds, and a TEST AUC from the best CV fold's model.

## Data

Sai's curation folder is the source of record. `paths.py` points at it by
default: `/project/pi_annagreen_umass_edu/saishradha/project_data_curation`. It
supplies:

- the aligned gene MSAs (`genomic_data/aligned`, 64 loci including the curated 19)
- the phenotypes (`phenotype_data/master_resistance_table.csv`)
- the GenBank reference
- the lineage calls (`BIG_TB_isolates_with_lineages.csv` in her
  `Data-Curation-for-MTB__sm_lineage_aware_sdcnn`)

This repo adds only the inputs her folder doesn't cover:

| input | built by | lives in |
|---|---|---|
| regulatory promoter windows (WHO Table 22, start-codon anchored) | `msa/build_promoter_coords.py` → `msa/submit_msa.py` | `msa/windows/` (tracked); FASTAs in `regulatory_msa/aligned_atg/` |
| AlphaFold structures | `structures/fetch_af_structures.py` | `structures/alphafold/` (tracked) |
| binding-site distances (LocusFusion proximity channel) | `locusfusion/binding_sites/build_distances.py` | `locusfusion/binding_sites/` (tracked) |
| GenomeFusion references (RLC mask, essentiality, Mycobrowser) | `genomefusion_refs/fetch_refs.sh` | `genomefusion_refs/` (tracked) |
| GenomeFusion partition and token store | `genomefusion.partition`, `genomefusion.build_tokens` | `genomefusion_cache_v3/`, `genomefusion_tokens/` |
| held-out-lineage splits | `splits/build_holdout_splits.py` | `results/splits/` |

`/project` is not mounted on every compute node. To read from a local mirror of
her folder, set `ABR_SAI=<mirror>`. Every path in `paths.py` has an `ABR_*`
override.

Link the built inputs into a new clone from the shared workspace:

```bash
SRC=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis
mkdir -p regulatory_msa
ln -s $SRC/regulatory_msa/aligned_atg regulatory_msa/
ln -s $SRC/genomefusion_cache_v3 $SRC/genomefusion_tokens .
conda env create -f environment.yml          # abr_env
```

## Splits

The splits follow Sai's leave-one-major-lineage-out design. For held-out lineage
L (1–4):

- **test:** isolates whose lineage is exactly L
- **train:** every other isolate with a lineage call
- **excluded:** isolates with no lineage call (one isolate)

A (drug, L) cell exists only when train and test each have at least 50 R and 50 S.
Lineages 2 and 4 qualify for 10 drugs; lineages 1 and 3 qualify only for EMB,
INH, PZA, RIF and STR. LEVOFLOXACIN has no feasible cell.

Train is cut into 5 stratified CV folds (seed 42), because the networks early-stop
on a validation fold.

```bash
python splits/build_holdout_splits.py      # -> results/splits/heldout_{1..4}/{DRUG}.csv, MULTIDRUG.csv, summary.csv
```

## Running

Run from the repo root, through SLURM. `job.sbatch` wraps any command: it
activates `abr_env` and logs to `logs/`. Pass resources on the `sbatch` line.

```bash
mkdir -p logs
S=results/splits/heldout_2
GPU="-p gpu --gres=gpu:1 --constraint=vram23 -c 4"

# Stacked CNN: one drug, all four modalities, 19 loci. Paper protocol: 300/30/50.
sbatch $GPU --mem=128G -t 36:00:00 job.sbatch python stacked_cnn/train.py --drugs PYRAZINAMIDE \
    --modalities all --split-dir $S --epochs 300 --patience 30 --min-epochs 50 \
    --run-name cnn_L2/sd19 --loci eis embA embB embC ethA ethR fabG1 gid gyrA gyrB inhA katG pncA rpoB rpoC rpsL rrl rrs tlyA
#   per-drug loci: drop --loci.  Locus sets: --locus-set lab64 | who_t12_sai
#   multi-drug: stacked_cnn/train_multidrug.py --drugs all --split-file $S/MULTIDRUG.csv ...

# LocusFusion v5_fixed: builds its token cache on first use
sbatch $GPU --mem=32G -t 12:00:00 job.sbatch python -m locusfusion.train --drugs PYRAZINAMIDE \
    --split-dir $S --run v5_fixed_L2
sbatch $GPU --mem=48G -t 24:00:00 job.sbatch python -m locusfusion.train --drugs all \
    --split-dir $S --run md_v5_fixed_L2

# GenomeFusion. The locus-set arms are listed in genomefusion/train.py; --max-hours is required by convention.
sbatch $GPU --mem=48G -t 24:00:00 job.sbatch python -m genomefusion.train --drugs PYRAZINAMIDE \
    --split-dir $S --locus-set 19r --max-hours 20 --out results/genomefusion/gf19r_L2

# LASSO (CPU). --locus-set perdrug | all (the 19) | lab64 | who_t12_sai
sbatch -c 2 --mem=64G -t 12:00:00 job.sbatch python -m lasso.lasso --drugs PYRAZINAMIDE \
    --locus-set all --split-dir $S --out results/lasso/onehot_all_L2
```

Notes:

- A drug with no split file for L (see `results/splits/summary.csv`) has no cell
  for that held-out lineage.
- The whole-genome GenomeFusion arm sits near chance for its first ~20 epochs.
  Keep `--min-epochs` at 50 or more and `--max-hours` generous.
- GenomeFusion inputs, built once and from the VCFs, on CPU:
  - `python -m genomefusion.partition`
  - `python -m genomefusion.build_tokens --shard $SLURM_ARRAY_TASK_ID --n-shards 16`, as a 16-task array
  - then `--merge --n-shards 16`

## Outputs

| what | where |
|---|---|
| per-drug JSON (CV folds, TEST, config) + `summary.csv` | `results/<run>/` (CNN), `results/locusfusion/<run>/`, `results/genomefusion/<arm>/`, `--out` (LASSO) |
| weights | `model_weights_cnn/`, `model_weights_lf/`, `model_weights_gf/` |

## Layout

| path | what |
|---|---|
| `paths.py` | every data location |
| `bigtb.py` | the BIG-TB helpers we reuse (alpha weights, threshold, one-hot, DRUG_TO_LOCI), copied verbatim |
| `datasets/` | modality loaders (dna, protein, biophysical, regulatory), locus sets |
| `training/` | CV protocol, early stopping, metrics, checkpoints |
| `models/` | `stacked_cnn.py`, `locusfusion.py`, `genomefusion.py` |
| `stacked_cnn/`, `locusfusion/`, `genomefusion/`, `lasso/` | each model's runner (+ tokenizer for the two fusion models) |
| `splits/`, `msa/`, `structures/`, `genomefusion_refs/` | input builders |
