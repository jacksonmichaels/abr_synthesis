# Talk figures: slide → file

Paths are relative to `results/figures/`. `talk/<notebook>/` figures are re-rendered from the notebook at slide
resolution (`scripts/render_talk_figures.py <notebook>`, text ×1.2); `talk/new/` is drawn by
`scripts/build_talk_figures.py`; `talk/mockups/` by `scripts/build_talk_mockups.py`. Rebuild both after any result changes.

| # | slide | file |
|---|---|---|
| 1 | TB resistance cartoon | `talk/mockups/s01_resistance_cartoon.png` |
| 3 | literature table | `talk/mockups/s03_literature_table.png` |
| 4 | data pipeline | `manuscript/fig0_pipeline.png` |
| 5 | what we vary | `talk/mockups/s05_axes.png` |
| 6 | model block diagrams | `talk/mockups/s06_models.png` |
| 6b | what the LASSO is | `talk/new/lasso_explainer.png` |
| 7 | one-hot vs variant tokens | `talk/mockups/s07_onehot_vs_tokens.png` |
| 8–9 | tokenization worked example | `manuscript/fig_tokenization.png` |
| 10 | LocusFusion architecture | `manuscript/fig1_architecture.png` |
| 12 | locus set | `manuscript/fig2_locus_size.png` (panel a) |
| 13 | where attribution lands | `shap_v3_alltest/shapley_alltest_locus_heatmap.png` or `manuscript/figA1_lasso_coefficients.png` |
| 14 | modalities | `manuscript/fig3_modalities.png` |
| 15 | model size | `manuscript/fig2_locus_size.png` (panel b) |
| 16 | LocusFusion per drug | `manuscript/fig4_locusfusion_per_drug.png` |
| 17 | uncalled-base channel, per drug | `talk/new/uncalled_channel_per_drug.png` |
| 18 | uncalled bases sit at resistance sites | `talk/new/uncalled_hotspots.png` |
| 19 | lineage split: design + result | `talk/new/lineage_vs_random.png` (backup: `talk/lineage_split/c4_0.png` split design, `c21_0.png` 19-locus gain) |
| 20 | PZA: S66* and A25A | `talk/modality_flips/c14_0.png` |
| 21 | pncA substitution scan | `talk/modality_flips/c19_0.png` (`c21_0.png` = synonymous along the gene) |
| 22 | structure modality (null) | `talk/new/structure_tokens.png` |
| 22b | how site distances were computed | `talk/mockups/s23_site_distance_method.png` |
| 23 | four drugs on the 16S | `talk/binding_site_visuals/c4_0.png`, `c6_0.png` |
| 24 | same variant, different distance | `talk/new/site_distance_by_drug.png` |
| 25 | %R by distance shell, 11 drugs | `talk/binding_site_visuals/c12_0.png` |
| 26 | EmbB in 3D | `presentation/embB_ethambutol_rotating.gif` (static: `talk/binding_site_visuals/c14_0.png`) |
| 27 | distance alone vs LocusFusion | `talk/binding_site_distance/c44_0.png` |
| 28 | summary | `talk/new/recap_effects.png` |
| 28b | next steps: drug-specific distance channel | `talk/mockups/s28_next_distance_channel.png` |
| 29 | questions (background) | `talk/binding_site_visuals/c4_0.png` or `presentation/embB_ethambutol_rotating.gif` |

`talk/binding_site_distance/c48_0.png` (§10) was drawn while the sd19 prediction sweep covered 6 of 11 drugs: do not use.
