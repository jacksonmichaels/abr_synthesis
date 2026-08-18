# shap_followup — testing what the SHAP attribution claims

Submitted **2026-08-13**, 33 SLURM jobs (3 modality sets x 11 drugs), `mdcnn`
single-drug only. Training settings are byte-identical to `full_run_v2`
(`--epochs 300 --patience 30 --min-epochs 50 --save-weights best`), so these
rows extend that ablation ladder rather than starting a new one.

## Why this exists

`notebooks/shap_notebook.ipynb` attributed the ISONIAZID SD-CNNs and found two
things that the existing runs cannot confirm or refute:

1. The all-modalities model gives its **DNA input 1.0%** of the attribution
   budget. It appears to re-read *katG* 315 through the protein channel, where
   the residue is one column of a 20-channel one-hot, instead of learning a
   3-nucleotide codon conjunction inside a 2,488-column alignment.
2. **Attribution share badly mis-ranks predictive value.** protein + biophysical
   take 89% of the budget and buy +0.014 CV AUC each; regulatory takes 6.5% and
   buys +0.032 — the largest single-modality gain.

The `full_run_v2` ladder (`dna`, `dna_protein`, `dna_biophysical`,
`dna_regulatory`, `all_modalities`) only ever ADDS to DNA, so nothing in it
removes a modality. That is the missing arm.

## The three cells

| cell | modalities | what it tests | prediction if the attribution is right |
|---|---|---|---|
| `no_dna__mdcnn` | protein + biophysical + regulatory | the 1.0% claim, in falsifiable form | CV AUC ~= `all_modalities` (0.9645 for INH) |
| `no_regulatory__mdcnn` | dna + protein + biophysical | is regulatory the only real gain? | drops toward `dna_protein` (~0.935) |
| `regulatory_only__mdcnn` | regulatory | how much does the promoter carry alone? | well below all of them, but above chance |

`no_dna` is the load-bearing one. If dropping DNA costs nothing, SHAP made a
correct and non-obvious prediction about a model, which is a much stronger claim
than a heatmap. If the model collapses, the 1.0% share was an artifact of
gradient magnitude and every attribution-share number in the notebook needs a
caveat.

## What was NOT run, and why

The first plan was to retrain `dna` with the regulatory FASTAs appended as extra
loci, to separate "regulatory adds information" from "regulatory gets its own
branch". **That experiment is vacuous for `mdcnn`.** `MDCNNNet` groups blocks by
CHANNEL COUNT, and dna and regulatory are both 5-channel (A,C,T,G,gap), so with
`mdcnn_trunk_per_modality=False` — the default, and what every `full_run_v2`
config records — the promoter windows are ALREADY stacked as extra channels in
the same DNA trunk, zero-padded to the longest CDS. `dna_regulatory__mdcnn` *is*
the proposed experiment, and it scores 0.9523. The separate-branch explanation
was already ruled out before the job was written.

Worth knowing for the other architectures: the same grouping rule means
`dna_regulatory` gets no modality separation under `mdcnn` unless
`--mdcnn-trunk-per-modality` is passed, which no recorded run uses.

## Reading the results

`scripts/shap_attribution.py` writes attribution shares to
`results/analysis/shap/`; the CV AUCs land here in `{cell}__mdcnn/summary.csv`.
`notebooks/shap_notebook.ipynb` section 9 joins the two into the
share-vs-delta-AUC table, which is the actual deliverable.
