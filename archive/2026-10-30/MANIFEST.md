# Archive, 2026-10-30

Workspace cleanup: LocusFusion (v3 paper, v4 in progress) and the paper's LASSO stay live;
everything below was moved here with its original path preserved under this folder.
Restore any entry with:

```bash
P=biophysical-fusion/notebooks/claims.ipynb   # path as listed below
mv "archive/2026-10-30/$P" "$P"
```

Notebooks and most of results/ are git-ignored, so this folder is the only copy of them.
The whole-genome / VCF LASSO line is here pending a from-scratch redo.


## workspace level

- `h1_repro`
- `wholegenome_cache.npz`
- `wholegenome_annot.npz`

## paper working files that revision_plan.md calls out of date

- `paper/paper_draft_revised.tex`
- `paper/edits.py`
- `paper/apply.py`
- `paper/render.py`

## stale project docs

- `biophysical-fusion/PAPER.md`
- `biophysical-fusion/PAPER_OUTLINE.tex`
- `biophysical-fusion/TODO.md`
- `biophysical-fusion/CLEANUP_REPORT.md`

## notebooks

- `biophysical-fusion/notebooks/results.ipynb`
- `biophysical-fusion/notebooks/claims.ipynb`
- `biophysical-fusion/notebooks/overview.ipynb`
- `biophysical-fusion/notebooks/cool_figures.ipynb`
- `biophysical-fusion/notebooks/datasets_overview.ipynb`
- `biophysical-fusion/notebooks/biophysical_properties_rdkit.ipynb`
- `biophysical-fusion/notebooks/structure_validation.ipynb`
- `biophysical-fusion/notebooks/tokenization_v2.ipynb`
- `biophysical-fusion/notebooks/presentation_figures.ipynb`
- `biophysical-fusion/notebooks/binding_site_visuals.ipynb`
- `biophysical-fusion/notebooks/wholegenome_lasso.ipynb`
- `biophysical-fusion/notebooks/lasso.ipynb`

## scripts

- `biophysical-fusion/scripts/build_results.py`
- `biophysical-fusion/scripts/build_claims.py`
- `biophysical-fusion/scripts/build_overview.py`
- `biophysical-fusion/scripts/build_cool_figures.py`
- `biophysical-fusion/scripts/build_datasets_overview.py`
- `biophysical-fusion/scripts/build_structure_validation.py`
- `biophysical-fusion/scripts/build_tokenization_v2.py`
- `biophysical-fusion/scripts/build_full_run_viewer.py`
- `biophysical-fusion/scripts/bigtb_paired_test.py`
- `biophysical-fusion/scripts/locusfusion_diagram.py`
- `biophysical-fusion/scripts/trace_models.py`
- `biophysical-fusion/scripts/build_talk_figures.py`
- `biophysical-fusion/scripts/build_talk_mockups.py`
- `biophysical-fusion/scripts/render_talk_figures.py`
- `biophysical-fusion/scripts/check_who_t12.py`
- `biophysical-fusion/scripts/token_census.py`
- `biophysical-fusion/scripts/token_census_v1.py`
- `biophysical-fusion/scripts/shap_multidrug.py`
- `biophysical-fusion/scripts/build_shap_v3_notebook.py`
- `biophysical-fusion/scripts/build_lasso.py`
- `biophysical-fusion/scripts/sbatch/build_cool_figures.sh`
- `biophysical-fusion/scripts/sbatch/build_structure_notebooks.sh`
- `biophysical-fusion/scripts/sbatch/test_structure_length.sh`
- `biophysical-fusion/scripts/sbatch/gate_who_t12.sh`
- `biophysical-fusion/scripts/sbatch/token_census.sh`
- `biophysical-fusion/scripts/sbatch/token_census_v1.sh`
- `biophysical-fusion/scripts/sbatch/trace_models.sh`
- `biophysical-fusion/scripts/sbatch/build_lasso_notebook.sh`

## experiments load_frame already skips (DROP_RUNS) + smoke output

- `biophysical-fusion/results/experiments/full_run`
- `biophysical-fusion/results/experiments/alllocus_run`
- `biophysical-fusion/results/experiments/joint_capacity`
- `biophysical-fusion/results/experiments/joint_convergence`
- `biophysical-fusion/results/experiments/setfusion_scaling`
- `biophysical-fusion/results/experiments/setfusion_warmup`
- `biophysical-fusion/results/experiments/token_signal`
- `biophysical-fusion/results/experiments/newmodels_pilot`
- `biophysical-fusion/results/experiments/seeds_19loci`
- `biophysical-fusion/results/experiments/repartition_19loci`
- `biophysical-fusion/results/experiments/_smoke_branched`
- `biophysical-fusion/results/experiments/_smoke_lf`
- `biophysical-fusion/results/experiments/_smoke_lf_md`
- `biophysical-fusion/results/experiments/locusfusion_v3_smoke`

## whole-genome / VCF LASSO line (to be redone from scratch)

- `biophysical-fusion/results/experiments/lasso_wholegenome_20260903`
- `biophysical-fusion/results/experiments/lasso_wholegenome_sweep_20260929`
- `biophysical-fusion/results/experiments/lasso_windows_20260903`
- `biophysical-fusion/results/experiments/lasso_vcf_loci_20260929`
- `biophysical-fusion/results/experiments/lasso_pooled_20260929`
- `biophysical-fusion/results/experiments/lasso_pooled_v2_20261001`

## loose experiment notes from Aug-Sep

- `biophysical-fusion/results/experiments/CODE_CHANGES_20260806.md`
- `biophysical-fusion/results/experiments/CODE_CHANGES_20260818.md`
- `biophysical-fusion/results/experiments/CODE_CHANGES_20260825.md`
- `biophysical-fusion/results/experiments/AGGREGATOR_FAMILY_20260827.md`
- `biophysical-fusion/results/experiments/SHAP_LOCUSFUSION_20260901.md`
- `biophysical-fusion/results/experiments/followups_viewer.ipynb`

## v1-era analyses

- `biophysical-fusion/results/analysis/shap`
- `biophysical-fusion/results/analysis/shap_locusfusion`
- `biophysical-fusion/results/analysis/shap_locusfusion_converge`
- `biophysical-fusion/results/analysis/shap_multidrug`
- `biophysical-fusion/results/analysis/token_census`
- `biophysical-fusion/results/analysis/token_census_v1`

## old figure sets

- `biophysical-fusion/results/figures/claims`
- `biophysical-fusion/results/figures/data_shapes`
- `biophysical-fusion/results/figures/followups`
- `biophysical-fusion/results/figures/full_run`
- `biophysical-fusion/results/figures/full_run_v2`
- `biophysical-fusion/results/figures/param_scaling`
- `biophysical-fusion/results/figures/paper`
- `biophysical-fusion/results/figures/setfusion_warmup`
- `biophysical-fusion/results/figures/presentation`
- `biophysical-fusion/results/figures/talk`
- `biophysical-fusion/results/figures/locusfusion_architecture.png`
- `biophysical-fusion/results/figures/locusfusion_architecture.svg`
- `biophysical-fusion/results/figures/locusfusion_shapes.png`
- `biophysical-fusion/results/figures/locusfusion_shapes.svg`

## Added later the same day: outside biophysical-fusion/

- `gene_msa/aligned` (pre-insertion-fix WHO t12 MSAs; `aligned_fixed/` is current)
- `gene_msa/slurm_logs`
- `regulatory_msa/slurm_logs`
- `worktrees/lf-token-offset-fix/` (former `.claude/worktrees/` checkout; see its README)

Deleted: `regulatory_msa/promoter_windows.before_20261022.json` (byte-identical to
`promoter_windows.json`), `regulatory_msa/__pycache__/`, and worktrees
`.claude/worktrees/{lasso-wholeseq,token-census}` (nothing unique; their branches remain).
