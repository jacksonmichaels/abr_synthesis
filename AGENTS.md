# Agent reference

Terse context for agents working in this repo. README.md has the run commands.

## Ground rules
- Shared HPC cluster (Unity). Login node: edit, import checks, `--help` only. Training, token building, VCF reads → SLURM (`job.sbatch`). Account `pi_annagreen_umass_edu`.
- Ask the user before submitting jobs.
- Env: `abr_env` (`environment.yml`). MSA extraction only: Sai's `make_msa` env.
- Run from repo root: `python -m <pkg>.<module>`.
- `/project` mount is flaky on some nodes → set `ABR_SAI` to a local mirror. A job can exit 0 after a failure; grep logs for `FAILED`.

## Data: what comes from where
| input | source | why it's here |
|---|---|---|
| gene MSAs (64 loci), phenotypes, GenBank, lineage calls | Sai's folder (`paths.SAI`) | source of record; shared with her benchmark |
| promoter windows (`msa/`) | built here from VCFs | not in her data; WHO Table-22 regions, start-codon anchored (TSS anchoring shifted katG by 481 bp) |
| AlphaFold structures, binding-site distances | built here | LocusFusion's per-drug site-proximity channel |
| `genomefusion_refs/` (RLC mask, DeJesus essentiality, Mycobrowser) | public downloads | GenomeFusion locus partition + locus features |
| GF partition (`genomefusion_cache_v3`) + token store | built here | whole-genome input; v3 = `19` with tlyA (not ahpC) |
| splits (`results/splits/heldout_L/`) | `splits/build_holdout_splits.py` | Sai's leave-one-major-lineage-out design |

All paths: `paths.py` (every one has an `ABR_*` override).

## Components
| path | what / why |
|---|---|
| `bigtb.py` | Big-TB helpers copied verbatim (alpha weights, threshold, DRUG_TO_LOCI). Avoids the TF-importing original. Do not edit; it defines the protocol. |
| `datasets/` | modality loaders: dna, protein, biophysical, regulatory; `loader.LOCUS_SETS`, `LOCI_19` |
| `training/multimodal.py` | the shared protocol: drop missing labels → split file → alpha on train → 5-fold CV, early stop on val AUC → TEST = best fold's model. LF/GF/LASSO reuse `_metrics`, `LR`, `build_optimizer`. |
| `models/stacked_cnn.py` | MD-CNN topology: loci stacked as channels. Baseline. |
| `models/locusfusion.py` + `locusfusion/` | 19-locus variant-token transformer (per-locus stage, then across-locus stage, then one query per drug). v5_fixed = synonymous tokens dropped, residue one-hot kept, site proximity on. |
| `models/genomefusion.py` + `genomefusion/` | LocusFusion over the whole genome (6,776 loci, one locus per base); tokens from VCFs |
| `lasso/lasso.py` | L1-logistic on the one-hot alignment; linear baseline on identical input |
| `splits/` | split builder |
| `msa/` | promoter-window MSA extraction (extractor, window builder, SLURM submitter) |

## Protocol facts
- Labels: R=0, S=1 in CNN/LASSO/LF (sigmoid = P(S)). GF encodes R=1 but uses the shared `_metrics`, so GF JSON `n_R`/`n_S` are swapped. AUC is unaffected.
- Results record `val_ids_sha1` per fold (not LASSO or the multi-drug CNN); pair models only on matching hashes.
- Significance between models: exact paired Wilcoxon over folds, Hodges-Lehmann CI, Holm correction.
- Split roles: `train` / `test` / `excluded` (no lineage call). Runners select rows by role.
- Held-out lineages 1 and 3 are feasible only for EMB/INH/PZA/RIF/STR; LEVO has no feasible cell.
- Defaults that matter: LF/GF 300 epochs, patience 30, `min_epochs` 50. The stacked CNN defaults to 60/15/0 → pass 300/30/50 for comparable runs.
- GenomeFusion whole-genome runs sit near chance for ~20 epochs before learning. Never judge one before epoch 30. Keep `--max-hours` generous.

## Gotchas
- `who_t12` needs 4 genes Sai doesn't have (whiB6, dnaA, PPE35, rpoA) → CNN/LASSO use `who_t12_sai` or `lab64`. GF is fine (genes from GenBank).
- Sai's `pepQ.fasta` is Rv3698, not WHO's pepQ (Rv2535c); `who_t12_sai` excludes it.
- Sai's 4 loci eis/embB/ethA/rrs were built with an MSA extractor that overwrote bases at insertions. Known, accepted for alignment with her data.
- LF token caches are keyed on input paths; a new data dir means a rebuild (~minutes, ~4 GB RAM).
- CNN and GF training are not bit-reproducible in the first CV fold of a process (old code too, even 1 thread + PYTHONHASHSEED=0); later folds are. Expect ~0.002 CV drift between identical runs.
- LASSO: liblinear runs with no `random_state`, so coefficients and AUCs drift slightly (~1e-6 AUC) between identical runs.
- GF regulated-locus edges are mostly a next-gene fallback, not WHO regulation. Don't describe them as catalogue knowledge.
- Checkpoints: `{"config", "state_dict"}`. Renaming parameters breaks loading old weights.
