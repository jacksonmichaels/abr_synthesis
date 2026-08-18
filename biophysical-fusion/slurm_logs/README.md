# slurm_logs/

Reorganized 2026-08-13. Previously 1,303 files in one flat directory.

```
manifests/          submitted_*.json — one per sbatch_all_runs.py invocation
archive/<date>/     .out logs from finished runs, by SUBMISSION date
*.out               logs of whatever sweep is currently in flight
```

## `manifests/` — do not delete

These are the **only** job_id → run mapping that exists. squeue job names are
`abr_{cell}_{drug}` with no run prefix, so `dna__late_fusion_ISONIAZID` is
ambiguous across `full_run` / `full_run_v2` / `alllocus_run`; only the manifest
disambiguates. The run READMEs cite them as provenance for the exact command
behind each job — including `submitted_20260806_200457_RECONSTRUCTED.json`,
which was rebuilt by hand after a second-resolution filename collision clobbered
four of `joint_convergence`'s six originals.

`sbatch_all_runs.py` writes new manifests straight into `manifests/`.

(The `run_status.py` / `sweep_status.py` monitors that used to read these were
retired 2026-08-13 along with the other per-sweep tooling; the manifests remain
the provenance record.)

## `archive/<date>/` — logs of finished runs

Filed by the submission timestamp embedded in each filename, not by mtime — a
job submitted 2026-08-06 that finished on the 7th belongs with its own run.

| folder | logs | run |
|---|---|---|
| `20260728/` | 110 | pre-`full_run` single-drug work |
| `20260804/` | 491 | `full_run` (240 jobs, submitted twice) |
| `20260806/` | 535 | **`alllocus_run`**, `full_run_v2`, `joint_convergence`, `joint_capacity` |
| `20260810/` | 62 | `setfusion_scaling` joint stage |
| `20260811/` | 8 | `token_signal` |
| `pre_full_run_multidrug/` | 8 | hand-written `multidrug_*` / `trace_models` jobs, no timestamp in filename |

Each of these runs has a `README.md` in its `results/experiments/` folder with
the conclusions, so the logs are kept for forensics only and are safe to delete
if space is ever needed — **with one exception**.

## `archive/20260806/` — do not prune

It holds the 221 `alllocus_run` logs, and they are the **only surviving copy of
that run**. Its `results/experiments/alllocus_run/` folder was lost (not in git,
not in `CLEANUP_REPORT.md`, only 9 of 210 checkpoints left on the weights
volume). `results/experiments/alllocus_run/reconstruct_from_logs.py` rebuilds the
run's tables from these logs and nothing else, so deleting them destroys the
result — including the finding that once the locus universe is matched,
multi-task sharing is worth ~+0.002 macro CV AUC under `mdcnn`.

Whether the same is true of the other archive folders is worth checking before
any future prune: a run whose `results/` folder still exists loses only
forensics, but this one loses the numbers.

Deleted 2026-08-13: `archive_failed_scriptsmissing_20260804/` (154 logs of jobs
that failed with a missing-script error, quarantined 2026-08-04 and since fixed).
