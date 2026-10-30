# Repo cleanup, 2026-08-13 — what was removed and why

Tree went from **154 MB to 57 MB**; `scripts/` from 20 files to 7; `notebooks/`
from 5 to 3.

Everything deleted here was **tracked in git at the time**, so all of it is one
command away — and the command does not need a hash, which matters because this
report was written before the deletions were committed:

```bash
# restore any path this cleanup removed, from whichever commit removed it
P=biophysical-fusion/scripts/sweep_setfusion_scaling.py
git checkout "$(git log --diff-filter=D -1 --format=%H -- "$P")^" -- "$P"
```

Anything under `results/` or `slurm_logs/` is a different matter: those trees are
git-ignored (see `.gitignore`), so nothing in them was ever recoverable this way.
That is how `results/experiments/alllocus_run/` came to be lost — see its
README, which was rebuilt from SLURM logs on 2026-08-18.

The organizing rule, applied throughout: **a script written for one experiment
lives with that run and is deleted once the run is written up.** `scripts/` holds
only what is general to the project. The run's `README.md` in
`results/experiments/` is what preserves the finding — not the script that
produced it.

## Deleted — regenerable output (95 MB)

- **623 `*_curves.png`** (85 MB). Rendered from `cv_folds[i]["history"]` in the
  result JSON beside each one. Verified all 623 regenerable before deleting, and
  round-tripped one:
  `python training/curves.py results/experiments/full_run_v2/dna__setfusion`.
  Now in `.gitignore`.
- `__pycache__/` (5 dirs), `results/figures/token_pca/` (empty),
  `results/experiments/_smoke/` (smoke-test output).

## Deleted — per-experiment scripts

| script | experiment it served |
|---|---|
| `sweep_setfusion_scaling.py`, `analyze_setfusion_scaling.py`, `watch_setfusion_scaling.sh` | `setfusion_scaling` |
| `ablate_setfusion.py`, `build_setfusion_warmup_viewer.py` | `setfusion_warmup` |
| `build_followups_viewer.py` | `followups` |
| `param_scaling.py`, `data_shapes.py`, `animate_locus_embedding.py` | one-off figure sets |
| `sweep_moxi.py` | 2026-07 MOXIFLOXACIN early-stopping knob sweep |
| `sweep.py`, `rerun_regulatory.sh` | superseded by `sbatch_all_runs.py` |
| `sbatch/multidrug_{all,dna,dna_mdcnn}.sh` | superseded by `sbatch_all_runs.py --multidrug` |
| `run_status.py`, `run_monitors/sweep_status.py` | sweep monitors, only useful in flight |

`scripts/` now holds `run_experiment.py`, `run_multidrug.py`,
`sbatch_all_runs.py`, `trace_models.py`, `build_full_run_viewer.py`,
`build_datasets_overview.py`, `sbatch/trace_models.sh`.

**`results/experiments/setfusion_scaling/submit.sh` no longer runs as written.**
Its README carries a restore command at the top for the 616 held-back
single-drug jobs.

## Deleted — notebooks

- `notebooks/results_viewer.ipynb` — superseded by `build_full_run_viewer.py`.
  It also held a **second copy of the BIG-TB baseline AUC tables**; that script
  is now their single source of truth (verified self-contained, then smoke-tested).
- `notebooks/token_pca.ipynb` — one-off analysis for `token_signal`.

## Moved / filed

- `results/figures/` held ten loose `fig1…fig5_*` files from 2026-07-29, sitting
  above the per-run subdirectories where they read as the current figure set.
  Those plus `figures/_superseded/` → `results/archive/figures_20260729/`
  (with a README on provenance). `figures/` now contains only per-run folders;
  the current set is `figures/full_run_v2/`.
- `slurm_logs/` was 1,303 files in one directory → `manifests/` (87 job_id→run
  provenance records), `archive/<submission-date>/` (1,144 logs of finished
  runs), and the live sweep's logs loose. Filed by the timestamp *in the
  filename*, not mtime, so a job submitted 08-06 that finished 08-07 stays with
  its own run. See `slurm_logs/README.md`.
- Deleted `slurm_logs/archive_failed_scriptsmissing_20260804/` (154 logs of jobs
  that failed on a since-fixed missing-script error).

## Code changed

- `sbatch_all_runs.py` writes manifests into `slurm_logs/manifests/` instead of
  the log root, so log housekeeping can't sweep up the only job_id→run mapping
  there is.
- `training/multimodal.py` and `datasets/biochem.py` docstrings no longer cite
  deleted files; `build_full_run_viewer.py` no longer claims its baselines are
  copied from a notebook that no longer exists.
- Added `biophysical-fusion/.gitignore` (previously only the parent repo had one,
  so the README's "results/ is git-ignored" was unverifiable from in here).
- `README.md` inventory rebuilt — it had listed 6 of 20 scripts and two
  notebooks that no longer exist.

## Not done, deliberately

- **`results/experiments/full_run/` kept in full.** `CODE_CHANGES_20260806.md`
  declares it the control for `full_run_v2`, `joint_convergence` and
  `joint_capacity`; only its curve PNGs went.
- **The four `*_viewer.ipynb` in `results/` kept.** They carry 2.3 MB of
  *executed* output each, not just generated source — deleting them would have
  cost the rendered analysis of each run for ~4% of the disk saving. Two of them
  no longer have a generator, which is fine: they are the record now.
- **`scripts/` left flat.** Every script resolves the project root as
  `Path(__file__).parent.parent`, so moving them into subdirectories would
  silently redirect every output path. Not worth it for 7 files.

## Pre-existing issue found, not fixed

`tests/test_setfusion.py` fails 1 of 22: *"defaults drifted from the full_run
configuration"* — `SetFusionNet` defaults are now `d_model=128, nhead=4,
layers=2, dim_ff=256, enc_width=64, bins=4, token_norm='none'`. Confirmed
failing at `HEAD` before this cleanup, so it predates it. Whether the test or
the defaults should move is a research call: the `setfusion_scaling` /
`token_signal` arms deliberately changed these, and `full_run_v2` is the control
they are measured against. The other three test files pass (8/8, 34/34, 16/16).
