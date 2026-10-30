# locusfusion_v3 — full sweep (launched 2026-10-01)

**v3 = `locusfusion_v2 --lf-unknown-symbols`**: the v2 tokenizer (one merged
promoter+coding axis per locus, 16 tokens, H37Rv-frame translation, property
change from the reference) plus the uncalled-base channel (N in dna/regulatory,
`-`/`X` in protein), so an uncalled base makes a token. Result JSONs say
`arch: locusfusion_v2`; the folder is what marks them v3. `results_frame.py`
drops this run (DROP_RUNS) so it never pools into the v2 cells; read it with
`compare.py`.

## Cells (253 GPU jobs)

| arm | folders | matches |
|---|---|---|
| sd19 | `sd19_{modset}_unk__locusfusion_v2` | locusfusion_fixed/sd19_* (v1), alllocus_run_v2 (mdcnn) |
| sd | `sd_{modset}_unk__locusfusion_v2` | locusfusion_fixed/sd_*, full_run_v2 |
| ablation | `ablation_sd19_{modset}__locusfusion_v2` | v3 minus the uncalled-base channel, 19 loci (the v3 "tokenizer" row) |
| seeds | `s{1000..4000}_all_modalities_unk__locusfusion_v2` | seeds_19loci (seed 0 = sd19) |
| partitions | `cv{101..104}_all_modalities_unk__locusfusion_v2` | repartition_19loci/cv*_all_modalities__mdcnn |

Modsets: dna, dna_protein, dna_biophysical, dna_regulatory, all_modalities. 11
drugs. Protocol as every v2 run: 300 epochs / patience 30 / min 50 / best
weights / seed 0 / cv-seed 42. 19-loci jobs 128G/36h, per-drug 64G/24h, vram23.

## Isolation from the two things that broke earlier runs

- **Code**: jobs run from a frozen copy,
  `/scratch3/.../v3_snapshot/biophysical-fusion` (read-only source dirs;
  `BASE_COMMIT` + `working_tree.patch` record exactly what it is). Repo edits
  cannot reach running jobs (cf. the 09-24 draft-tokenizer contamination).
- **Data**: `/project` is missing on some GPU nodes. Jobs read a verified
  read-only copy in `/scratch3/.../data_snapshot/` through `ABR_GENOTYPE_DIR`,
  `ABR_PHENOTYPE_CSV`, `ABR_GENBANK` (new env overrides in `bigtb_ref.py` /
  `datasets/cds.py`, exported by `sbatch_all_runs.py --env`). Weights go to
  `/scratch3/.../model_weights_v3/`, not `/project/pi_mfiterau`.

## Supervision

`babysit.py` (SLURM CPU job, `babysit.sh`, 6 days) every 10 min: resubmits any
cell whose job left the queue without a result JSON (max 3 tries), watches the
canary (LEVOFLOXACIN, sd19 all_modalities) and on a non-mount failure writes
`ALERT.md` and stops resubmitting, writes `STATUS.md`, and reruns `compare.py`
hourly -> `SUMMARY.md`. State in `state.json`; restarting resumes.

- If ALERT.md appears: the queued jobs likely share the bug.
  `scancel` them (`squeue -u $USER -n` on `abr_*_unk__locusfusion_v2_*` and
  `abr_*__locusfusion_v2_*`), fix the snapshot, clear state.json, rerun
  `python babysit.py --once` then `sbatch babysit.sh`.

## compare.py

Rebuilds the paper's 14-row Holm family with v3 in the LocusFusion rows and the
ablation as the tokenizer row; `--as-v1` puts v1 back in those slots and
reproduces the paper exactly (checked 2026-10-01: +0.0103, +0.0117, +0.0095,
+0.0293, mdcnn − cisfusion Holm 0.056, whole genome 0.71). Pins the LASSOs to
`lasso_wholeseq_20260901` / `lasso_wholegenome_20260903`.

## CPU side jobs

- `scripts/token_census_v3.py` -> `results/analysis/token_census_v3/` (truncation
  under the v3 tokenizer; N tokens split out).
- `scripts/missingness_check.py` -> `results/analysis/missingness_v3/` (is the N
  signal batch or heteroresistance?).
