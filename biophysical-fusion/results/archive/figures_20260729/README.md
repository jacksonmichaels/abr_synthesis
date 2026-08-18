# Superseded figures, 2026-07-29

Filed here 2026-08-13 during the repo cleanup. Two sets, both from **before**
the `full_run` sweep (2026-08-04):

- `fig1_single_drug_vs_sdcnn`, `fig2_modality_gains`, `fig3_joint_vs_single`,
  `fig4_joint_vs_mdcnn`, `fig5_summary` — produced by
  `notebooks/results_viewer.ipynb`. These previously sat loose at the top of
  `results/figures/`, above the per-run subdirectories, where they read as the
  project's current figure set.
- `fig0_sd_vs_sdcnn_corrected`, `fig1_dumbbell_sd_to_md`,
  `fig2_delta_vs_fold_noise`, `fig3_gain_vs_difficulty`, `fig4_summary_bars`,
  `fig5_vs_published_mdcnn`, `fig6_vs_mdcnn_baseline` — the former
  `results/figures/_superseded/`.

Both sets were drawn from the single-drug results now in
`../singledrug_20260728/` and `../pre_c6_20260728/`.

**The current figure set is `results/figures/full_run_v2/`**, built by
`scripts/build_full_run_viewer.py`.
