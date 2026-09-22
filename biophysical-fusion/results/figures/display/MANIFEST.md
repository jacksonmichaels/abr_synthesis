# Display items — exported 2026-09-01

Standalone PNGs pulled out of the executed notebooks, so they can go straight
into slides. **These are exports, not sources.** Regenerate by rebuilding and
re-executing the notebook, then re-running the export:

    python scripts/build_results.py && jupyter nbconvert --to notebook \
        --execute --inplace notebooks/results.ipynb

`locusfusion_architecture.png` / `locusfusion_shapes.png` come from
`python scripts/locusfusion_diagram.py` and are copies of the same files in
`results/figures/` — that script reads every shape and parameter count off a
real `LocusFusionNet`, so it cannot drift from the model.

## From `notebooks/results.ipynb` (the presentation notebook)

| file | section | what it carries |
|---|---|---|
| `results_01` | §1 | headline tiles: 0.9246 established vs 0.9242 locusfusion, 6.7M vs 0.66M params |
| `results_02` | §2 | macro CV AUC by task and architecture, three panels |
| `results_03` | §3 | per-drug, best established single-drug cell vs SD-CNN and MD-CNN |
| `results_04` | §4 | per-drug modality gain — ETO/INH on regulatory, PZA on protein, rrs drugs flat |
| `results_05` | §5 | what the 19-locus set buys the established models |
| `results_06` | §6 | parameters vs AUC, established only |
| `results_07` | §7 | SHAP: columns carrying 90% of attribution vs columns given |
| `results_08` | §9 | the six aggregators at both locus settings |
| `results_09` | §10 | catalogue vs additive, per drug against cohort size |
| `results_10` | §11 | parameters vs AUC, BOTH families, 19 loci, four orders of magnitude |
| `results_11` | Part IV | the LocusFusionNet architecture diagram |
| `results_12` | §12 | locusfusion by modality set and task, and paired against mdcnn |

## From `notebooks/shap_locusfusion.ipynb`

| file | section | what it carries |
|---|---|---|
| `shap_lf_01`, `shap_lf_02` | §1 | attribution budget by modality, both cells |
| `shap_lf_03`, `shap_lf_04` | §2 | locus enrichment against the variant-token null; drug x locus matrix |
| `shap_lf_05` | §3 | SHAP against the model's own read-out attention |
| `shap_lf_06` | §4 | named variants: attribution vs how resistant the allele actually is |
| `shap_lf_07` | §5 | background sensitivity — wt vs train, rank agreement |
| `shap_lf_08` | §6 | per isolate, summed attribution vs logit gap (completeness) |
| `shap_lf_09` | §7 | what the tokenizer actually saw |
