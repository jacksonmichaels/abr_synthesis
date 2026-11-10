# Getting started: LocusFusion and GenomeFusion

How to run, change and evaluate the two models on Unity. The work is on the
`genomefusion` branch.

## 1. Clone into your own workspace

The repo root is a workspace root, and the code finds its data next to the
clone. Don't work inside someone else's workspace.

```bash
ws_allocate abr 30                       # or reuse one of yours
WS=/scratch3/workspace/$USER-abr         # whatever ws_allocate printed
git clone -b genomefusion https://github.com/jacksonmichaels/abr_synthesis.git "$WS"
cd "$WS"
conda env create -f biophysical-fusion/environment.yml   # creates abr_env
```

The sbatch launchers activate `abr_env` and bill `pi_annagreen_umass_edu`,
so you need to be in that group.

## 2. Link the data

Data isn't in git. Link it from the shared copy, which is read-only for you:

```bash
SRC=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis
cd "$WS"
for d in data_snapshot curated_fix Big-TB-benchmark genomefusion_tokens; do ln -s "$SRC/$d" .; done
for f in aligned aligned_atg promoter_windows_atg.json; do ln -s "$SRC/regulatory_msa/$f" regulatory_msa/; done
for d in genomefusion_cache genomefusion_cache_v3; do cp -r "$SRC/$d" .; done   # GF partition caches (7 MB), copied because they get written to
```

| linked | what |
|---|---|
| `data_snapshot/` | genotypes, phenotypes, GenBank, lineage calls |
| `curated_fix/aligned19/` | the 19 loci with the MSA-insertion fix (v5_fixed's input) |
| `regulatory_msa/aligned_atg/` | promoter MSAs anchored on the start codon (v5_fixed's input) |
| `Big-TB-benchmark/` | upstream BIG-TB code that `bigtb_ref.py` imports |
| `genomefusion_tokens/` | GF tokens for every isolate (519 MB) |

LocusFusion token caches (`token_cache_v4/`) are not linked. Build your own
with `--arms tokens` (one CPU job). That also means a tokenizer change can't
read a stale cache. If you change the GF tokenizer, rebuild `genomefusion_tokens`
too (`scripts/sbatch/genomefusion_tokens.sh`). Weights, snapshots and results
are written into your own workspace.

## 3. Where to make changes

All of this is under `biophysical-fusion/`.

| change | file |
|---|---|
| LocusFusion network (v5_fixed) | `canonical/locusfusion.py` |
| GenomeFusion network | `canonical/genomefusion.py` |
| token channels | `locusfusion_v4/layout.py` + `tokenizer.py`, `genomefusion/layout.py` + `tokenizer.py` |
| which loci GF sees | `genomefusion/partition.py` |
| training loop, early stopping | `locusfusion_v4/train.py`, `genomefusion/train.py` |
| experiment arms (one arm = 11 drugs) | `locusfusion_v4/sbatch.py`, `genomefusion/sbatch.py` |

`canonical/README.md` has more detail. The paper's LocusFusion is v3
(`models/locusfusion_v2.py`); leave it alone unless you're reproducing the
paper.

## 4. Test, canary, sweep

Run everything from `biophysical-fusion/`, and only in a job or an `salloc`
session, never on a login node.

```bash
cd "$WS/biophysical-fusion"
python -m locusfusion_v4.tests.test_model        # seconds, synthetic
python -m genomefusion.tests.test_model

python -m locusfusion_v4.sbatch --arms tokens               # once: build LF token caches
python -m locusfusion_v4.sbatch --arms v5_fixed --dry-run   # print the job scripts
python -m genomefusion.sbatch --arms canary --dry-run       # PZA, one job
python -m genomefusion.sbatch --arms canary                 # submit the canary
python -m genomefusion.sbatch --arms gf19p                  # 11 jobs, after the canary looks right
```

Every launcher freezes the code it submits (`*_snapshot/<stamp>/`), so editing
the code while jobs are queued is safe.

- **Results:** `results/experiments/<package>/<arm>/<DRUG>.json`
- **Logs:** `.../logs/`
- **Loading GF results:** `genomefusion/results.py` (`load`)
- **Building the GF results notebook:** `scripts/build_genomefusion_results.py`

## 5. What to compare against

Everything is evaluated on the **lineage split** (Beijing gets its own fold;
5-fold CV, test held out). Significance is an exact Wilcoxon over the 55
per-drug folds.

| arm | what | CV AUC (mean over 11 drugs) |
|---|---|---|
| LF `v5_fixed` | 19 loci | 0.9147 |
| GF `gf19p` | the same 19 loci with promoters (GF's parity arm) | ties v5_fixed |
| GF whole genome | 6,776 loci | 0.8491 |

## Things that will bite you

- **The genome-scale GF arm needs about 20 warmup epochs.** Don't stop it
  early or judge it on early epochs.
- **A job can fail and still show COMPLETED.** Grep the `.out`/`.err` for
  `FAILED`/`Traceback` rather than trusting `sacct`. `/project` drops off
  some nodes, which is why the data is read from `data_snapshot/`.
- **`gf19` and `minus19` are superseded.** Use `gf19p` and `gf_minus19p`:
  `gf19` had no promoter bases, and `minus19` is contaminated.
- **The GF result JSON's `n_R`/`n_S` and `P_S_*` fields are mislabelled.**
  Recompute counts from the predictions CSV.
