# ABR synthesis workspace

**New here? Start with [GETTING_STARTED.md](GETTING_STARTED.md).**

The project is `biophysical-fusion/` (code, results, notebooks; see its README).
Everything else here is data, weights or frozen code that it reads, mostly by
absolute path. **Don't rename or regroup the folders below:** 49
`weights_location.json` files, about 15 scripts and the frozen snapshots
themselves have these paths baked in.

| Folder | What it is | Read by |
|---|---|---|
| `biophysical-fusion/` | the project | — |
| `paper/` | `paper_draft.tex`, `revision_plan.md` (the current manuscript edit list) | you |
| `data_snapshot/` | local copy of the genotype/phenotype/GenBank data + lineage calls (`ABR_*` env vars) | all data loading |
| `Big-TB-benchmark/` | upstream BIG-TB clone, read-only (own remote, git-ignored) | `bigtb_ref.py` |
| `mdcnn_eval/` | the MD-CNN reproduction (authors' code, our rerun) | paper's "MD-CNN rerun" number |
| `model_weights_v3/` | LocusFusion v3 checkpoints, random split | v3 predictions / SHAP |
| `model_weights_lineage/` | the paper grid's checkpoints on lineage-grouped splits | Shapley appendix, lineage analyses |
| `model_weights_v4/` | LocusFusion v4/v5 checkpoints, one folder per arm | `locusfusion_v4/` |
| `token_cache_v4/` | v4 tokenizer caches, keyed by hash (rebuildable) | v4 training + notebooks |
| `v3_snapshot/`, `v3md_snapshot/`, `lineage_snapshot/` | frozen code the v3 / multi-drug / lineage weights were trained with (`BASE_COMMIT` + `working_tree.patch`) | `predict_locusfusion_v3`, SHAP scripts |
| `v4_snapshot/<stamp>/` | frozen code for each v4 submission | `locusfusion_v4/sbatch.py` |
| `curated_fix/` | the 4 curated loci rebuilt after the MSA insertion bug; `aligned19/` = repaired set of 19 (symlinks) | v4 |
| `regulatory_msa/` | promoter MSAs: `aligned/` (TSS-anchored, v3/paper), `aligned_atg/` (start-codon-anchored, v5) + generators | data loading, v4 |
| `gene_msa/` | WHO tier 1+2 gene MSAs (`aligned_fixed/`) and genotypes | `who_t12` experiment |
| `structures/` | AlphaFold models (`alphafold/`, tracked) and drug-bound ribosome structures | structure modality, v4 site distance |
| `reference_docs/` | papers, advisor slides, the binding-site literature review | — |
| `archive/<date>/` | retired work at its original path; see each `MANIFEST.md` | — |
