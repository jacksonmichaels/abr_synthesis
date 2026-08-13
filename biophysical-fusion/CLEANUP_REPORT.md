# Cleanup report — `biophysical-fusion`

Survey date **2026-08-13**. Nothing in this report has been deleted or moved; it is
a proposal. Total tree: **154 MB**, of which `results/` is **136 MB** and the
actual source (`models/ datasets/ training/ tests/`) is **920 KB**.

Ordered by *do this first*, not by size.

---

## 0. Blocker — commit the reorg before deleting anything

`git status` shows the 2026-08-04 reorganization (top-level files moved into
`datasets/`, `training/`, `scripts/`, `notebooks/`, `tests/`) **staged but never
committed**, on top of ~21 source files that are not tracked at all:

| state | files |
|---|---|
| staged adds / renames, uncommitted | `models/legacy.py`, `training/__init__.py`, `training/curves.py`, `scripts/sweep_moxi.py`, `scripts/sbatch/multidrug_{all,dna,dna_mdcnn}.sh`, + 6 renames (`run_experiment.py`, `train_multimodal.py` → `training/multimodal.py`, `fixtures.py`, `test_baseline_alignment.py`, the two notebooks) |
| staged deletes, uncommitted | `biophysical.py`, `data.py`, `models.py`, `train.py`, `eval_dna_cnn.py`, `BASELINE_ALIGNMENT_CHANGES.md`, `scripts/results_viewer.ipynb` |
| **untracked entirely** | `training/checkpoint.py`, `datasets/cds.py`, `models/README.md`, `tests/test_{checkpoint,cisfusion,setfusion}.py`, 12 files in `scripts/`, `scripts/run_monitors/`, `notebooks/token_pca.ipynb` |

`training/checkpoint.py` is the module that fixed the "we never saved weights"
bug documented in `results/experiments/CODE_CHANGES_20260806.md` — it is the most
consequential file in the repo right now and it exists in exactly one place, on
disk. **Commit before any cleanup**, otherwise deletions are unrecoverable and
the reorg's rename history is lost.

Two commits is enough: one for the reorg-as-staged, one for the untracked
post-`full_run_v2` work.

---

## 1. Derived artifacts — 92 MB, all regenerable (largest win)

### 1a. Per-fold curve PNGs — **623 files, 85 MB**

Every `*_curves.png` in `results/experiments/` is rendered by
`training/curves.py:save_curves` from `cv_folds[i]["history"]` in the JSON
sitting next to it. The inputs are kept; the renders are not inputs to anything.

```
results/experiments/*/*/*_curves.png     # 623 files, 84.9 MB
```

`training/curves.py` already documents a re-plot CLI, so this is a pure cache.
Recommendation: **delete, and add `*_curves.png` to `.gitignore`** so they don't
accumulate again. If you'd rather not lose the at-a-glance view, keep them for
`full_run_v2` only (34 MB) and drop `full_run`, `setfusion_warmup`,
`setfusion_scaling` (51 MB).

### 1b. Generated viewer notebooks — **4 files, 6.6 MB**

| file | generator |
|---|---|
| `results/experiments/full_run/full_run_viewer.ipynb` (2.5 MB) | `scripts/build_full_run_viewer.py` |
| `results/experiments/full_run_v2/full_run_viewer.ipynb` (2.5 MB) | same, `<root>` arg |
| `results/experiments/followups_viewer.ipynb` (914 KB) | `scripts/build_followups_viewer.py` |
| `results/experiments/setfusion_warmup/setfusion_warmup_viewer.ipynb` (823 KB) | `scripts/build_setfusion_warmup_viewer.py` |

All four are build products with a committed generator. Delete; regenerate on
demand. (Also: `followups_viewer.ipynb` is the only one that sits at the
`experiments/` root rather than inside its run folder — if you keep it, move it
under a `followups/` directory for consistency.)

### 1c. `__pycache__` — 5 directories, 708 KB

`datasets/`, `models/`, `training/`, `scripts/`, and the project root. Ignored by
git but they clutter every `ls`. Delete.

---

## 2. Superseded outputs — ~4.2 MB, but the real cost is confusion

| path | size | why |
|---|---|---|
| `results/figures/fig1…fig5_*.{pdf,png}` (10 files, top level) | 1.1 MB | Dated **2026-07-29**, produced by `notebooks/results_viewer.ipynb` from the pre-`full_run` results. They sit at the top of `figures/` above the per-run subdirs, so they read as "the project's figures" when the current set is `figures/full_run_v2/`. **Most misleading artifact in the tree.** Move to `figures/_superseded/` or delete. |
| `results/figures/_superseded/` (14 files) | 1.6 MB | Already labelled superseded, untouched since 2026-07-29. Delete. |
| `results/archive/pre_c6_20260728/`, `results/archive/singledrug_20260728/` | 1.4 MB | Pre-`full_run` single-drug results, superseded twice over by `full_run` → `full_run_v2`. Delete unless a paper draft still cites them. |
| `results/figures/token_pca/` | empty | Empty directory. Delete. |
| `results/experiments/_smoke/` | 158 KB | 4-file smoke-test output from 2026-08-11. Delete; it is trivially reproducible and adds a fake entry to the experiments list. |

**Do NOT delete `results/experiments/full_run/`.**
`CODE_CHANGES_20260806.md` states every new knob defaults to the value that
produced `full_run`, making it the declared control for `joint_convergence`,
`joint_capacity`, and `full_run_v2`. Its JSON/CSV stay; only its curve PNGs
(1a) and viewer notebook (1b) go.

---

## 3. `scripts/` — 20 flat files, several one-offs

### Retire (verify first where noted)

| file | reason |
|---|---|
| `sweep.py` (181 lines) | Submits the grid via its own top-of-file lists. Superseded by `sbatch_all_runs.py`, which is what the README documents and what every run README cites for reproduction. Not mentioned anywhere in README/TODO. *Verify no run folder's `submit.sh` shells out to it before deleting.* |
| `sweep_moxi.py` | Self-described "one-off knob sweep … to diagnose the majority-class collapse". That diagnosis is closed — the monitor/patience/min-epochs knobs it explored are now `sbatch_all_runs.py` flags. |
| `rerun_regulatory.sh` | One-off re-run for the "intersect regulatory regions with loaded loci" rule change, already applied and superseded by `full_run_v2`. Hardcodes a `/scratch3/...` path. |
| `sbatch/multidrug_all.sh`, `sbatch/multidrug_dna.sh`, `sbatch/multidrug_dna_mdcnn.sh` | Hand-written joint job scripts, superseded by `sbatch_all_runs.py --multidrug` (which emits equivalent scripts and records a manifest). Keep `sbatch/trace_models.sh` — the README still points at it. |

### Consolidate

- **`run_status.py` vs `run_monitors/sweep_status.py`** — the newer one's docstring
  explicitly positions it as complementary (per-job vs per-run), which is fair,
  but one lives at the top of `scripts/` and the other in a subdirectory. Move
  `run_status.py` into `run_monitors/` so the two sit together.
- **`watch_setfusion_scaling.sh`** and **`analyze_setfusion_scaling.py`** and
  **`sweep_setfusion_scaling.py`** are one sweep's toolchain living in the shared
  `scripts/` namespace. Either move them next to
  `results/experiments/setfusion_scaling/submit.sh`, or group them (below).

### Proposed structure

`scripts/` currently mixes three unrelated kinds of thing. Grouping them makes
the entry points findable without reading 20 docstrings:

```
scripts/
  run_experiment.py  run_multidrug.py  sbatch_all_runs.py   # entry points
  sbatch/trace_models.sh
  figures/      build_*_viewer.py, data_shapes.py, param_scaling.py,
                trace_models.py, animate_locus_embedding.py
  monitors/     run_status.py, sweep_status.py
  sweeps/       sweep_setfusion_scaling.py, analyze_setfusion_scaling.py,
                watch_setfusion_scaling.sh, ablate_setfusion.py
```

---

## 4. `notebooks/` — 4 files, two problems

- **`results_viewer.ipynb` is superseded.** It produced the stale top-level
  figures in §2 and reads the pre-`full_run` layout of `results/experiments/`.
  Its replacement is `scripts/build_full_run_viewer.py`, which even carries the
  comment *"From `notebooks/results_viewer.ipynb`, unchanged"* over the copied
  baseline block. **The published BIG-TB baseline AUC table now exists in two
  places** — a genuine drift hazard, since a correction to one won't reach the
  other. Recommended: lift the baseline table into
  `datasets/` or a small `baselines.py`, have `build_full_run_viewer.py` import
  it, then retire the notebook.

- **`datasets_overview.ipynb` has two sources of truth.** It is generated by
  `scripts/build_datasets_overview.py`, but the `.ipynb` is also tracked in git
  and currently shows as modified — i.e. it is being hand-edited *and*
  regenerated. Pick one: either the notebook is the artifact (delete the
  builder) or the builder is (gitignore the notebook). Same question applies to
  the `build_*_viewer.py` family, but there the answer is clearly "the builder".

- **`token_pca.ipynb` is 1.1 MB** of embedded output — 40× the other notebooks.
  Strip outputs before committing (`nbstripout`, or a `.gitattributes` filter).
  It's also the only notebook that hardcodes an absolute
  `/home/jacksonmicha_umass_edu/...` project path instead of the relative
  `sys.path.insert(0, "..")` the other three use.

---

## 5. `slurm_logs/` — 1,303 files in one flat directory

Only ~4.5 MB, so this is a navigability problem, not a disk one.

| group | count | action |
|---|---|---|
| `submitted_*.json` manifests | 86 | **Keep all.** Load-bearing: `run_status.py` needs them for the job_id → run mapping (squeue job names are ambiguous across runs), and `full_run/README.md`, `full_run_v2/README.md`, `joint_convergence/README.md` cite them as provenance — including a hand-reconstructed one. Deleting these breaks the audit trail. |
| `archive_failed_scriptsmissing_20260804/` | 154 | Delete. Logs of jobs that failed for a known, fixed cause; already quarantined 9 days ago. |
| `*.out` from 2026-07-28 / 08-04 / 08-06 | 1,080 | Completed, written-up runs (`full_run`, `full_run_v2`, the follow-ups). Archive into `slurm_logs/archive/<date>/` — or delete, since each run's README carries the conclusions and the manifests carry the commands. |
| `*.out` from 2026-08-07 onward | 123 | **Keep.** `setfusion_scaling` is the live sweep (newest output 2026-08-12), and per project notes 616 single-drug jobs are deliberately still pending. |
| `watch_setfusion_scaling.log` | 1 | Keep while the sweep is live. |

Suggested end state: `slurm_logs/{manifests/, archive/<run>/, *.out}` — current
run's logs loose, everything else filed.

---

## 6. Documentation drift

- **`README.md`'s `scripts/` inventory lists 6 of 20 files.** Missing: every
  `build_*_viewer.py` except one, `data_shapes.py`, `param_scaling.py`,
  `run_status.py`, `run_monitors/`, `ablate_setfusion.py`,
  `animate_locus_embedding.py`, the three `setfusion_scaling` tools, `sweep.py`.
  It also lists `results_viewer` as a current notebook and omits `token_pca`.
- **`README.md` says `results/` is "git-ignored"** — true, but via the *parent*
  `abr_workspace/.gitignore`. There is no `.gitignore` in this repo directory,
  so the statement is unverifiable from inside the project. Add a local
  `.gitignore` (`__pycache__/`, `.ipynb_checkpoints/`, `*_curves.png`,
  `*_viewer.ipynb`, `results/`).
- **`datasets/biochem.py`'s docstring** says features were moved out of "the old
  top-level `biophysical.py` … that module is now a shim". That shim is a staged
  deletion — it no longer exists, and `datasets/biophysical.py` (a real modality,
  same name) does. Reword to avoid pointing at a deleted file with a colliding name.
- **`models/legacy.py`** is correctly documented as out of the live path, but
  `models/__init__.py` still re-exports it, so it's in every import. That's fine
  if `CrossAttentionFusionCNN` is still an open experiment (the docstring says it
  is) — flagging only so the choice stays deliberate.

---

## Summary

| step | reclaims | risk |
|---|---|---|
| 0. Commit staged reorg + untracked sources | — | **do first** |
| 1. Delete curve PNGs, viewer notebooks, `__pycache__` | **92 MB** | none — all regenerable |
| 2. Delete superseded figures / archives / `_smoke` | 4.2 MB | low — check paper drafts for `archive/` |
| 3. Retire 6 one-off scripts, regroup `scripts/` | ~35 KB | low — verify `sweep.py` unused |
| 4. Retire `results_viewer.ipynb`, de-duplicate baselines, strip notebook outputs | ~1 MB | medium — baselines must move before the notebook goes |
| 5. File `slurm_logs/`, keep all manifests | 1.6 MB | low — **never delete `submitted_*.json`** |
| 6. Refresh README inventory, add local `.gitignore` | — | none |

After steps 1–2 the tree drops from **154 MB to ~57 MB**, and `results/` contains
only inputs (JSON/CSV), run READMEs, and current figures.
