# ABR synthesis: LocusFusion and GenomeFusion

These models predict antibiotic resistance in *M. tuberculosis* from genotype.
There are 11 drugs, and each one is a binary R/S call.

- **LocusFusion** is a transformer over variant tokens from 19 resistance loci.
- **GenomeFusion** is the same design applied to the whole genome (6,776 loci).

The code is written for the Unity cluster.

## Layout

| path | what |
|---|---|
| `biophysical-fusion/canonical/` | **the two networks**: `locusfusion.py` and `genomefusion.py`. Edit these. |
| `biophysical-fusion/locusfusion_v4/` | LocusFusion tokenizer, training and SLURM launcher |
| `biophysical-fusion/genomefusion/` | GenomeFusion locus partition, tokenizer, training and SLURM launcher |
| `biophysical-fusion/datasets/`, `training/`, `models/` | shared data loading, training utilities and older models |
| `biophysical-fusion/scripts/` | data prep (lineage splits, lineage calls, AlphaFold fetch) and the older multi-modal runners |
| `regulatory_msa/`, `gene_msa/`, `genomefusion_refs/`, `structures/` | scripts and small reference files used to build model inputs |
| `mdcnn_eval/` | MD-CNN baseline |

## Setup

Clone into your own workspace and create the environment:

```bash
WS=/scratch3/workspace/$USER-abr        # from ws_allocate
git clone -b genomefusion https://github.com/jacksonmichaels/abr_synthesis.git "$WS"
cd "$WS"
conda env create -f biophysical-fusion/environment.yml    # creates abr_env
```

The data isn't in git. Link it from the shared workspace:

```bash
SRC=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis
for d in data_snapshot curated_fix Big-TB-benchmark genomefusion_tokens; do ln -s "$SRC/$d" .; done
for f in aligned aligned_atg promoter_windows_atg.json; do ln -s "$SRC/regulatory_msa/$f" regulatory_msa/; done
for d in genomefusion_cache genomefusion_cache_v3; do cp -r "$SRC/$d" .; done

mkdir -p biophysical-fusion/results/experiments/{lineage_split,locusfusion_v4}
ln -s "$SRC/biophysical-fusion/results/experiments/lineage_split/splits" biophysical-fusion/results/experiments/lineage_split/
ln -s "$SRC/biophysical-fusion/results/experiments/locusfusion_v4/splits" biophysical-fusion/results/experiments/locusfusion_v4/
```

Jobs bill the `pi_annagreen_umass_edu` account and activate `abr_env`.

## Making changes

| to change | edit |
|---|---|
| a network | `canonical/locusfusion.py` or `canonical/genomefusion.py` |
| token channels | `layout.py` and `tokenizer.py` in `locusfusion_v4/` or `genomefusion/` |
| which loci GenomeFusion uses | `genomefusion/partition.py` |
| training and early stopping | `train.py` in either package |
| experiment arms | `sbatch.py` in either package |

All paths above are under `biophysical-fusion/`. The training code imports the
networks from `canonical/`, so the next job you submit uses your edit. Each
package's `README.md` describes its design.

## Running

Run these from `biophysical-fusion/`, inside a job or an `salloc` session
(not on a login node):

```bash
python -m locusfusion_v4.tests.test_model
python -m genomefusion.tests.test_model
```

Submit through the launchers. Each arm runs 11 jobs, one per drug, on the
lineage split. Add `--dry-run` to print the job scripts without submitting.

```bash
python -m locusfusion_v4.sbatch --arms tokens        # build LocusFusion token caches (once, or after a tokenizer change)
python -m locusfusion_v4.sbatch --arms canary        # one PZA job
python -m locusfusion_v4.sbatch --arms v5_fixed      # the reference LocusFusion arm

python -m genomefusion.sbatch --arms canary          # one PZA job
python -m genomefusion.sbatch --arms gf19p           # GenomeFusion on the 19 LocusFusion loci
```

Each launcher copies the code into a snapshot before submitting, so editing
the code while jobs wait in the queue is safe. To build GenomeFusion tokens
from the VCFs, use `scripts/sbatch/genomefusion_tokens.sh`.

## Outputs

- **Results:** `results/experiments/<locusfusion_v4|genomefusion>/<arm>/`, one JSON per drug with CV and test metrics, plus test predictions
- **Logs:** `results/experiments/<package>/logs/`
- **Weights:** `<workspace>/model_weights_v4/` and `<workspace>/model_weights_gf/`
- **Loading GenomeFusion results into a DataFrame:** `genomefusion.results.load()`
