# Biophysical Fusion — TODO

Isolated workspace. Nothing here modifies `Big-TB-benchmark/`; we only read
reference code from it. Goal: build a foundation good enough to launch the
first experiment (biophysical-property fusion on top of BIG-TB's DNA CNN)
and to extend later with the other modalities/models sketched in
`reference_docs/experimental_plan.pdf`.

## 1. Hypothesis (from reference_docs/)

- BIG-TB's best single-drug model is the DNA one-hot CNN, AUC 0.8753
  (`dna-tasks/SD-CNN/model_training/run_SDCNN_ccp_crossval.py` +
  `.../parameters/tb_cnn_codebase.py`). Best protein model (one-hot CNN) only
  reaches AUC 0.7889 — DNA beats protein throughout BIG-TB's sweep.
- Kulkarni et al. 2026 fused amino-acid **molecular weight, isoelectric
  point, and hydrophobicity (Eisenberg scale)** as a 3×K matrix (K = protein
  length) through an independent conv branch, alongside nucleotide input, to
  predict MICs. This improved on nucleotide-only models and "forced the
  network to identify the severe functional damage of nonsense truncations
  vs. missense changes."
- **Test**: does the same biophysical branch improve BIG-TB's DNA CNN on
  binary resistance classification? Start with Isoniazid (katG+inhA) and
  Rifampicin (rpoB) — one multi-locus gene pair, one single-locus gene.
- Secondary goal (per experimental_plan.pdf): stand up a modality/model
  foundation general enough that later experiments — lineage vectors,
  richer protein features, cross-attention fusion, late-fusion with
  modality dropout, adversarial lineage decoupling, MIC+ABR multi-task,
  causal probing — are additive, not rewrites.

## 2. Key BIG-TB reference files

| Purpose | File |
|---|---|
| DNA CNN architecture + training loop | `dna-tasks/SD-CNN/model_training/run_SDCNN_ccp_crossval.py`, `.../parameters/tb_cnn_codebase.py` |
| DNA one-hot encoding, gene concat, drug→loci map | `protein-tasks/esm_models/data_utils.py` (`BASE_TO_COLUMN`, `DRUG_TO_LOCI`, `get_one_hot`, `create_X`) |
| Protein one-hot CNN (simplest fusion target for alignment sanity checks) | `protein-tasks/one_hot_encoded/cnn_model.py`, `cnn_utils.py` |
| Aligned-FASTA → CDS → protein translation (gap-aware) | `protein-tasks/protein_translation/` (`data_preprocessing.py`, `gene_coordinate_map.ipynb`) |
| Per-isolate CSV schema (`Filename`, `Sequence`, `Phenotype`, `Protein_Sequence`, `Frameshift_Mutation`) | `Big-TB-benchmark/README.md` §Dataset Overview |
| SHAP interpretability pattern to eventually reuse | `protein-tasks/esm_models/shap_esm.py`, `dna-tasks/SD-CNN/interpretability/run_interpret.py` |

## 3. Blocker to resolve before any real training run

No genotype/phenotype data exists in this checkout — `dna-tasks/data/genotype/`
and `dna-tasks/data/phenotype/` are literally placeholder files ("Download
this data"), and `protein-tasks/` scripts hardcode
`/project/pi_annagreen_umass_edu/mahbuba/Data-Curation-for-MTB/...` (an HPC
path we don't have here). **We cannot validate any pipeline end-to-end
without either cluster access or a synthetic fixture set.** Plan: build a
small synthetic-data generator first (see Phase 0) so the pipeline and model
shapes can be verified locally, then run for real on the cluster.

## 4. Foundation build order (no code yet — this is the plan)

**Phase 0 — synthetic fixtures**
- Tiny fake aligned-FASTA + phenotype CSV generator (a few genes, ~50
  isolates, deliberate gaps/frameshifts) so every later step is testable
  without the real dataset.

**Phase 1 — biophysical property table + alignment utility**
- Pin a 20-AA lookup table (MW / pI / Eisenberg hydrophobicity) as a static
  module, not computed on the fly.
- Build the nucleotide-position → codon → AA → property-vector mapper that
  walks the *aligned* (gapped) DNA sequence used by the DNA CNN, so the
  biophysical channel lines up 1:1 with existing one-hot positions per gene,
  per isolate. Reuse `protein_translation`'s CDS/gap logic rather than
  re-deriving it.
- Define the sentinel for gap / stop / frameshift positions explicitly.

**Phase 2 — modality module**
- One small interface all modalities implement (isolate ids in → array out,
  known length, known channel count), so DNA one-hot, biophysical, and
  future modalities (lineage vector, protein embedding, etc.) are
  interchangeable inputs to a fusion model rather than each model hardcoding
  its own loader.

**Phase 3 — fusion model**
- Two-branch CNN: existing DNA one-hot branch (mirrors `tb_cnn_codebase.py`'s
  conv stack) + new biophysical branch (mirrors `ProteinCNN1x1`'s stem+conv
  pattern), fused per the fusion-point decision below, single sigmoid head.

**Phase 4 — training/eval harness**
- Per-drug binary classification, 5-fold CV, AUC / AUC-PR / sens / spec at
  optimal threshold — matching BIG-TB's reported metrics so results are
  directly comparable to the 0.8753 baseline.

**Phase 5 — first experiment**
- Run Isoniazid + Rifampicin, DNA-only baseline vs. DNA+biophysical, compare.

**Phase 6 — extensibility pass**
- Only after Phase 5 has a result: slot in the next modality from
  `experimental_plan.pdf` (lineage vector is the cheapest next one) to prove
  the Phase 2 interface actually generalizes.

## 5. Design decisions

1. **Framework: PyTorch.** DECIDED. The DNA branch will be a fresh PyTorch
   port of `tb_cnn_codebase.py`'s conv stack (not a literal reuse of the
   Keras model), since we need it in the same autograd graph as the
   biophysical branch and as future modalities/fusion strategies
   (cross-attention, modality dropout, adversarial heads).

2. **Fusion point: late-concat as the default, but build both.** DECIDED.
   Two model variants share the same DNA-branch and biophysical-branch
   building blocks:
   - `LateFusionCNN` (default/primary) — DNA branch and biophysical branch
     each get their own conv stack, pooled/flattened separately, then
     concatenated before the dense head. Matches Kulkarni's description
     ("passed through independent convolutions") and is faithful to the
     paper's ablation (branch can be dropped cleanly).
   - `EarlyFusionCNN` (ablation variant) — biophysical channels
     concatenated onto the 5-channel one-hot before the first conv, single
     shared conv stack.
   Design the branch modules so both models are thin wrappers around the
   same `DNABranch`/`BiophysicalBranch` classes rather than duplicated code.

3. & 4. **Biophysical property table + normalization: use Kulkarni et al.'s
   actual method.** DECIDED (direction), BLOCKED on source material. Our
   `Literature Review.pdf` notes only name the Eisenberg (1984) scale for
   hydrophobicity and don't capture Kulkarni et al. 2026's exact MW/pI
   table or normalization scheme — need the paper itself (or its
   supplementary materials) to pin this down rather than substituting a
   different reproducible-but-not-their table. **Open question for you:**
   do you have the Kulkarni et al. 2026 PDF (or supplement) already, or
   should this be tracked down (e.g. via web search) before Phase 1 starts?
   Until resolved, Phase 1 (property table + alignment utility) is blocked;
   everything else in Phase 0 is not.

   Standing default so this doesn't fully block the rest of the plan:
   Biopython-derived tables (`IUPACData.protein_weights` for MW,
   `Bio.SeqUtils.IsoelectricPoint`'s EMBOSS pKa set for pI, hardcoded
   Eisenberg 1984 for hydrophobicity) + per-channel z-score, swapped out
   for Kulkarni's actual values/scheme as soon as we have them.

5. **Gap/stop/frameshift sentinel: encode as gaps.** DECIDED. Positions
   past a gap, indel, or premature stop get the same treatment as the DNA
   branch's own gap channel (`BASE_TO_COLUMN['-']`) — i.e. the biophysical
   branch emits a zero vector at any position where the DNA one-hot branch
   is also encoding a gap, rather than a separate out-of-range sentinel.
   Keeps the two branches positionally and semantically consistent.

6. **CV protocol: replicate BIG-TB's existing methods exactly.** DECIDED,
   specifically *because* multi-drug is a planned future step — keep the
   masked multi-weighted-BCE loss, the fixed random train/test split before
   CV, and plain (non-stratified) `KFold`, even though single-drug binary
   doesn't strictly need the masking. Output layer sized for multi-drug
   from the start (single-drug runs are just a 1-column label matrix
   through the same masked-loss path) so extending to multi-drug later is
   a data-shape change, not an architecture change.

7. **Multi-locus genes: shared branch.** DECIDED. Isoniazid's katG+inhA
   (and similar multi-locus drugs) get one concatenated biophysical conv
   stack across the gene-concatenated length axis, mirroring exactly how
   the DNA one-hot branch already concatenates multi-locus genes in
   `create_X`.

8. **HPC path — found two, access pending.** DECIDED (path identified,
   access not yet granted). Both existing BIG-TB pipelines hardcode paths
   under `/project/pi_annagreen_umass_edu/` (reads as a UMass Unity cluster
   allocation):
   - DNA CNN data (what we need): `genotype_input_directory:
     /project/pi_annagreen_umass_edu/saishradha/project_data_curation/genomic_data/aligned/`,
     `phenotype_file: /project/pi_annagreen_umass_edu/saishradha/project_data_curation/phenotype_data/master_resistance_table.csv`
     (from `dna-tasks/SD-CNN/model_training/parameter_files/optimized_epochs/RIF_ccp_epoch_60.txt`)
   - Protein-tasks data: `/project/pi_annagreen_umass_edu/mahbuba/Data-Curation-for-MTB/protein-tasks/data/latest/...`
   You don't have Unity access under this allocation yet but will get it.
   **Phase 5 (real training) stays blocked until access is granted; Phases
   0–4 (synthetic fixtures, property table, modality interface, model code,
   training harness) don't depend on it and can proceed now.**

## 6. Non-goals for this pass

Not building yet: cross-attention fusion, late-fusion+modality-dropout,
adversarial lineage decoupling, MIC+ABR multi-task, causal probing via
latent injection. These are real candidates from `experimental_plan.pdf`
but come after Phase 5 has a working biophysical-fusion result to build on.
