# alllocus_run_v2 — the single-drug grid on all 19 loci, at the full_run_v2 method

Submitted **2026-08-18**, 220 SLURM jobs, reproducible via `submit.sh` in this
folder. Manifest: `slurm_logs/manifests/submitted_20260818_*.json`.

## The question, unchanged from `alllocus_run`

Is "joint models beat single-drug models" multi-task sharing, or just a bigger
locus set? The two scopes do not see the same input by design — single-drug uses
SD-CNN's per-drug map (`tb.DRUG_TO_LOCI`, 2–3 loci), joint uses MD-CNN's
drug-independent rule (`datasets.loci_on_disk()`, 19 loci). This gives the
single-drug grid the joint locus set so the two differ only in scope.

## Why this run exists when `alllocus_run` already answered it

`alllocus_run` (2026-08-06) ran at **`full_run`'s** settings — 150 epochs,
patience 15, no warmup — and three things are wrong with leaving the answer
there:

1. **It is measured against a superseded baseline.** `full_run_v2` replaced
   `full_run` as the project's baseline precisely because 150/15/no-warmup
   distorted results: 150 was a binding cap, patience 15 fired spuriously, and
   the missing warmup made the whole setfusion row an artifact. An answer stated
   against `full_run` inherits all of that.
2. **It checkpointed nothing.** Same mistake `full_run_v2` exists to correct.
3. **It lost 10 of 220 jobs** to resource limits, and not at random — the four
   largest drugs in the two widest `late_fusion` cells, which is exactly where
   the surviving `late_fusion` result was most in doubt.

And then its results folder was lost outright and had to be rebuilt from SLURM
logs (see `../alllocus_run/README.md`). So this is a replacement, not a
follow-up.

## What it is

Identical to `../full_run_v2/submit.sh` except for the locus set — that is the
whole point, and it makes **`full_run_v2` the matched control**:

| | `full_run_v2` single-drug | here |
|---|---|---|
| loci | per-drug, `DRUG_TO_LOCI` (2–3) | **all 19 curated** |
| epochs / patience / min-epochs | 300 / 30 / 50 | same |
| weights | `best` fold | same |
| grid | 5 modality sets × 4 archs × 11 drugs | same, 220 jobs |
| `--mem` / `-t` / constraint | 48G / 16 h / none | **128G / 36 h / vram23** |

Joint cells are deliberately absent: joint runs already use all 19 loci, so
`full_run_v2`'s `multidrug_*` cells **are** the joint arm of this comparison.
Nothing needs rerunning on that side.

### The grid is named explicitly, not `--experiments all`

Since 2026-08-13 `MODALITY_SETS` also carries the SHAP leave-one-out arms
(`no_dna`, `no_regulatory`, `regulatory_only`), so `--experiments all` is now
**32 cells / 352 jobs**, not 20 / 220. `submit.sh` lists the five `full_run_v2`
modality sets by name so the grid stays matched to its control. Anyone reusing
that script for a new run should check the same thing.

### Resources were raised, and why — measured, not guessed

Every number here comes from `../alllocus_run/*/provenance.csv`, recovered from
that run's logs. `full_run_v2`'s single-drug 48G / 16 h would reproduce its exact
failures.

- **`--mem 128G`** (was 64G on `alllocus_run`, 48G on `full_run_v2`). Every
  surviving job peaked at **59–62 GiB against a 64G request** — the whole grid
  was pressed against the ceiling — and the 8 that died were host
  `OUT_OF_MEMORY`. Peak RSS is nearly flat across drugs within a cell
  (`dna__late_fusion`: median 42.5, max 42.6 GiB over n=269 to n=17582), because
  the dominant cost is loading all 19 loci for all 17,942 isolates *before*
  filtering to one drug's phenotyped subset; the per-drug arrays are the term
  that pushed the four biggest drugs over. 388 GPU nodes have ≥128G, so this
  costs nothing in scheduling.
- **`-t 36:00:00`** (was 12 h). The worst job projects to ~11.4 h — scaling the
  measured elapsed by the epoch count the recovered `best_epoch` distribution
  implies under 300/30/50 (×1.32 overall). But the two `setfusion` ISONIAZID
  jobs **timed out at 12 h** and so never recorded a duration; their true
  300-epoch cost is unbounded by measurement, which is what 36 h is for.
- **`--constraint vram23`**. One job (`KANAMYCIN`,
  `all_modalities__late_fusion`) died on a **CUDA** OOM on a 10.9 GiB card,
  asking for 1.21 GiB with 4.5 GiB reserved-but-unallocated.
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in `sbatch_all_runs.py`
  targets exactly that fragmentation pattern — but it landed in `d52700a`, on
  2026-08-13, **after** `alllocus_run` ran, so it has never been tested on this
  input. Belt and braces. 146 nodes satisfy the constraint.

## Expected cost

**~440–460 GPU-hours.** 348 GPU-hours were measured across 210 completed jobs at
150 epochs; the projected epoch count under 300/30/50 is ×1.32, applied to the
training portion. Per architecture (measured → projected): `setfusion` 116 → 147,
`cisfusion` 112 → 139, `late_fusion` 80 → 106, `mdcnn` 40 → 49.

That is more than `full_run_v2`'s whole single-drug half, because every drug here
loads 19 loci instead of 2–3.

**Storage ~40 GB.** A 19-locus checkpoint is 139–228 MB against `full_run_v2`'s
7–69 MB, ×220 runs. The volume had 1.1 TB free (95% used) at submission. Weights
land in `/project/pi_mfiterau_umass_edu/abr_model_weights/alllocus_run_v2/`.

## What to compare, when it lands

Against `../full_run_v2/`, cell for cell — same method, only the loci differ:

- `Δ loci` = this run − `full_run_v2` single-drug. What the joint locus set buys
  a single-drug model.
- `Δ joint − SD19` = `full_run_v2`'s `multidrug_{cell}` − this run. **What
  multi-task sharing is worth once the locus universe is matched.** This is the
  number the run exists to produce.

`alllocus_run`'s answer, at the old settings, was that it depends on the
architecture: ≈+0.002 under `mdcnn` and ≈+0.005 under `cisfusion` (both inside
fold SD — gone), but **+0.029 to +0.031 surviving under `late_fusion`**. The two
things to check here are whether the `late_fusion` margin holds up at 300 epochs
with the four missing big drugs now present, and whether `setfusion` says
anything at all now that it finally gets the warmup.

Judge on CV, and remember the standing caveats: single-seed, and these models are
**no longer locus-matched to the SD-CNN baseline**, so nothing in this run can be
quoted against published SD-CNN numbers.
