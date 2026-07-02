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
  binary resistance classification? Start with Isoniazid (inhA+katG) and
  Rifampicin (rpoB+rpoC) — both are multi-locus in BIG-TB's `DRUG_TO_LOCI`
  (corrected 2026-07-02: an earlier note called Rifampicin single-locus, but
  the reference map pairs rpoB with rpoC, matching Kulkarni's own
  rpoB/rpoC example). Single-locus drugs in the map are e.g. Pyrazinamide
  (pncA) / Kanamycin (rrs) if a true single-gene case is wanted.
- Secondary goal (per experimental_plan.pdf): stand up a modality/model
  foundation general enough that later experiments — lineage vectors,
  richer protein features, cross-attention fusion, late-fusion with
  modality dropout, adversarial lineage decoupling, MIC+ABR multi-task,
  causal probing — are additive, not rewrites.

## 2. Key BIG-TB reference files

| Purpose | File |
|---|---|
| DNA CNN training loop | `dna-tasks/SD-CNN/model_training/run_SDCNN_ccp_crossval.py` |
| DNA CNN architecture, one-hot encoding, drug→loci map, alpha weighting, threshold search — corrected 2026-07-01, an earlier note misattributed these to `protein-tasks/esm_models/data_utils.py` | `dna-tasks/SD-CNN/model_training/parameters/tb_cnn_codebase.py` (`BASE_TO_COLUMN`, `DRUG_TO_LOCI`, `get_one_hot`, `sequence_dictionary`, `make_genotype_df`, `rs_encoding_to_numeric`, `alpha_mat`, `get_threshold_val`) |
| Protein one-hot CNN (conv-stack shape reused for `ConvBranch`) | `protein-tasks/one_hot_encoded/cnn_model.py`, `cnn_utils.py` |
| Aligned-FASTA → CDS → protein translation (gap-aware) | `protein-tasks/protein_translation/` (`data_preprocessing.py`, `gene_coordinate_map.ipynb`) — not yet wired in, our own codon-walk in `biophysical.py` is a interim stand-in |
| Per-isolate CSV schema (`Filename`, `Sequence`, `Phenotype`, `Protein_Sequence`, `Frameshift_Mutation`) | `Big-TB-benchmark/README.md` §Dataset Overview |
| SHAP interpretability pattern to eventually reuse | `protein-tasks/esm_models/shap_esm.py`, `dna-tasks/SD-CNN/interpretability/run_interpret.py` |

## 3. Blocker to resolve before any real training run

Reference code is now present (updated 2026-07-02): the `Big-TB-benchmark/`
tree is populated, so `bigtb_ref.py` imports `tb_cnn_codebase` for real and
the smoke test runs (see Phase 4). What's still missing is the real
**genotype/phenotype data** — `dna-tasks/data/genotype/` and
`dna-tasks/data/phenotype/master_resistance_table.csv` are still placeholder
files ("Download this data"), and `protein-tasks/` scripts hardcode
`/project/pi_annagreen_umass_edu/mahbuba/Data-Curation-for-MTB/...` (an HPC
path we don't have here). **We cannot validate any pipeline end-to-end on
*real* data without cluster access; the synthetic fixture set (Phase 0) lets
us verify pipeline + model shapes locally in the meantime.** Real runs move
to the cluster once Unity access lands (§5.8).

## 4. Foundation build order

**Phase 0 — synthetic fixtures. IMPLEMENTED (`fixtures.py`).**
Fake aligned-FASTA + phenotype CSV generator, gene length/isolate count
configurable, deliberate gaps standing in for indels/frameshifts.

**Phase 1 — biophysical property table + alignment utility. IMPLEMENTED
(`biophysical.py`), table values themselves still a placeholder.**
20-AA lookup table (MW / pI / Eisenberg hydrophobicity, z-scored) as a
static module. `translate_seq` degaps the aligned nucleotide sequence then
translates in triplets with a hardcoded standard genetic-code table,
stopping at the first stop codon (truncates the protein like a real
nonsense mutation would) — matches Kulkarni et al. 2026 Methods
("Featurizing amino acid sequences": translated from the MSA, then
left-aligned, i.e. *not* kept in gap-preserving positional correspondence
with the nucleotide branch — this revised an earlier draft of decision #5
that assumed gap-preserving alignment). `biophysical_matrix` then maps the
translated protein string to a (3, K) property matrix, zero-padded.
Does **not** yet reuse `protein_translation`'s gap/CDS logic (tightly
coupled to real coordinate files we don't have locally); revisit once real
data is in hand.

**Phase 2 — dataset construction. IMPLEMENTED (`data.py`), as plain
functions, not a class-based interface.**
`build_dataset(drug, genotype_dir, phenotype_csv)` reuses
`tb_cnn_codebase.sequence_dictionary`/`get_one_hot`/`rs_encoding_to_numeric`
directly. One local fix needed: `tb_cnn_codebase`'s path handling
(`filename.split("/")[-1]`) assumes POSIX paths and silently breaks on
Windows backslash paths from `glob`/`os.path.join` — `_load_genotype_df`
in `data.py` routes around it by always handing `sequence_dictionary` a
forward-slash path, without touching the reference file.

**Phase 3 — fusion models. IMPLEMENTED (`models.py`).**
Shared `ConvBranch` (mirrors `ProteinCNN1x1`'s stem+conv+pool stack) +
`DenseHead`, composed three ways: `LateFusionCNN` (default — one DNA
branch over the concatenated locus, one biophysical `ConvBranch` **per
gene/protein** per decision #7's revision, flatten-and-concat into a
2×256-node dense head, confirmed against Kulkarni et al. 2026's Model
design Methods almost verbatim), `EarlyFusionCNN` (ablation, not from the
paper), and `CrossAttentionFusionCNN` (experimental_plan.pdf's
"Asymmetrical Cross-attention" — DNA branch as Query, the per-gene
biophysical branches concatenated along the sequence axis as Key/Value,
via `nn.MultiheadAttention`).

**Phase 4 — training/eval harness. IMPLEMENTED (`train.py`).**
`run_cv(...)` replicates BIG-TB's SD-CNN protocol per decision #6: fixed
train/test split, plain KFold on train, masked weighted BCE (torch port of
`masked_multi_weighted_bce`), `tb.alpha_mat`/`tb.get_threshold_val` reused
directly for class weighting and sens/spec threshold search.

**Smoke test: `run_experiment.py`.** Runs the full Phase 0→4 pipeline
against synthetic data for all three models — confirms the wiring works,
numbers are meaningless (random data, 5 epochs). Re-verified passing
2026-07-02 in the `abr_env` conda env now that `Big-TB-benchmark/` is
populated (all three models train + score without error; example run:
`dna_X (60, 5, 240)`, `bio_Xs rpoB (60,3,20) / rpoC (60,3,31)`).

**Model data-flow diagrams: `gen_model_diagrams.py` → `diagrams/`.** Added
2026-07-02. Traces every step of each model's forward pass on dummy tensors
and emits both a `.csv` (step / stage / operation / notes / output_shape)
and an editable `.svg` (colored flowchart, one grouped box per step) for
`late_fusion`, `early_fusion`, `cross_attention_fusion`. The final logits
shape in each is asserted against the model's real output, so the diagrams
can't silently drift from `models.py`. Shapes use a representative
Rifampicin scenario (rpoB+rpoC); regenerate after any `models.py` change.

**Phase 5 — first experiment (blocked on real data, see #8).**
Run Isoniazid + Rifampicin, DNA-only baseline vs. DNA+biophysical, compare.

**Phase 6 — extensibility pass.**
Only after Phase 5 has a result: slot in the next modality from
`experimental_plan.pdf` (lineage vector is the cheapest next one).

### Environment (updated 2026-07-02)
Project env is the **`abr_env` conda env** — use it for all Python here
(`conda activate abr_env`; base miniforge env has no torch). It has
Python 3.12 and `torch 2.6.0+cu124` with working GPU (`cuda.is_available()`
True on the machine's NVIDIA RTX 2080 Ti, Turing sm_75). Also installed, all
needed to import `tb_cnn_codebase.py` as-is (it pulls in `Bio.SeqIO`,
`sparse`, `ipdb`, `h5py`, `yaml`, and `tensorflow` at module level even
though we only use a handful of its functions): `numpy`, `pandas`,
`scikit-learn`, `biopython`, `sparse`, `ipdb`, `h5py`, `pyyaml`,
`tensorflow` (CPU build — the reused utilities don't need GPU TF, and CPU
avoids CUDA-library conflicts with torch's cu124 wheels; the actual GPU
training runs through PyTorch).

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
   actual method.** DECIDED (architecture confirmed from the full paper,
   added 2026-07-01), STILL PARTIALLY BLOCKED on exact numeric values. The
   full-text Methods (`reference_docs/Kulkarni et al. - 2026 - ...pdf`,
   "Featurizing amino acid sequences") confirms three features — molecular
   weight (g/mol), isoelectric point, hydrophobicity (Eisenberg scale) — as
   a 3×K matrix per protein, "inspired by EVEscape" (Thadani et al. 2023,
   *Nature*, ref 28) — but the main text does not give the literal MW/pI
   numbers or a normalization scheme (no "z-score"/"standardize"/"min-max"
   anywhere in the paper). Likely explanation: MW and pI each have one
   essentially-standard chemistry definition (unlike hydrophobicity, where
   many competing scales exist — which is presumably *why* only that one
   needed a named scale). Current implementation (`biophysical.py`) uses
   standard published tables — average residue mass for MW, free-amino-acid
   pI, Eisenberg (1984) for hydrophobicity — z-scored per channel. Still
   worth asking the authors directly (§6) or checking EVEscape's own
   supplement, but this is now a low-confidence-but-plausible match rather
   than a guess.

5. **Gap/stop/frameshift representation: translate-then-left-align, not
   gap-preserving.** REVISED 2026-07-01 (supersedes the original
   "encode as gaps, positionally consistent with the DNA branch" draft) —
   the paper's Methods are explicit: protein sequences are translated from
   the MSA then "left-aligned (i.e., not a multiple sequence alignment) and
   ... padded to the same length in the same manner as the nucleotide
   sequences." I.e. the biophysical branch does **not** stay in
   per-codon positional correspondence with the DNA one-hot branch —
   `translate_seq` (`biophysical.py`) degaps first (so an indel shifts the
   downstream reading frame, like a real frameshift) and stops at the first
   stop codon (truncating like a real nonsense mutation), then the
   resulting protein string is padded independently. Inferred (not stated
   at this granularity in the paper) and worth confirming with the authors:
   that stop-codon truncation, rather than continuing translation with a
   placeholder, is the right read of "distinguish nonsense from missense."

6. **CV protocol: replicate BIG-TB's existing methods exactly.** DECIDED,
   specifically *because* multi-drug is a planned future step — keep the
   masked multi-weighted-BCE loss, the fixed random train/test split before
   CV, and plain (non-stratified) `KFold`, even though single-drug binary
   doesn't strictly need the masking. Output layer sized for multi-drug
   from the start (single-drug runs are just a 1-column label matrix
   through the same masked-loss path) so extending to multi-drug later is
   a data-shape change, not an architecture change.

7. **Multi-locus genes: separate branch per gene (biophysical only).**
   REVISED 2026-07-01 (reverses the original "shared branch" draft, which
   was made before we had the full paper). The paper's Methods are
   explicit: "Individual proteins were encoded as separate channels, even
   if they are translated from the same locus (i.e., rpoB and rpoC
   proteins are in two separate channels in the amino acid block, while the
   corresponding gene sequences are concatenated into a single contiguous
   locus in the nucleotide block)." So: the **DNA branch** still
   concatenates multi-locus genes into one locus (that part of the
   original decision was right) — but the **biophysical branch** gets one
   `ConvBranch` per gene/protein (`LateFusionCNN`/`CrossAttentionFusionCNN`
   in `models.py` use `nn.ModuleList` over `bio_lens`, one length per
   gene), each flattened independently before concatenation.

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

## 6. Questions for author meeting

**Kulkarni et al. 2026 (biophysical fusion / MIC paper) — updated
2026-07-01 after reading the full paper (`reference_docs/Kulkarni et al. -
2026 - ...pdf`); items resolved by the paper are struck through, not
deleted, so it's visible what we already checked:**
1. Exact MW / isoelectric point numeric values and normalization scheme
   for the 3×K input — the paper names the properties and cites EVEscape
   (Thadani et al. 2023, *Nature*) as inspiration but never states the
   literal table or a normalization step. Still the one real gap.
2. ~~Exact fusion point~~ — ANSWERED: two independent conv blocks (same
   architecture as each other), each flattened, concatenated with lineage
   SNPs, through two 256-node dense layers. `LateFusionCNN` matches this.
3. Whether stop-codon truncation (translate up to, not past, the first
   stop) is the intended way the amino-acid branch "distinguishes nonsense
   from missense" — the paper states the *effect* (pncA nonsense vs.
   missense distinguishability, Supplementary Fig. 3) but not this
   mechanical detail; we inferred it.
4. ~~Shared or per-gene branch for multi-locus genes~~ — ANSWERED:
   per-gene/per-protein separate channels, confirmed explicitly with the
   rpoB/rpoC example.
5. Which drug lost performance from adding lineage, and why — ANSWERED for
   MIC models (pyrazinamide, due to small dataset + lineage skew) but
   they *also* say lineage "did not significantly alter performance" for
   rifampicin/isoniazid specifically — worth asking whether that's still
   their recommendation for a binary (not MIC) task before we add our own
   lineage branch (Phase 6).
6. They do have MIC labels for 8 drugs (14,834 isolates, Supplementary
   Data 1/2) — worth asking directly whether any could be shared,
   independent of BIG-TB's binary R/S labels, for the "predict both MIC
   and ABR" idea from `experimental_plan.pdf`.

**Tasmin & Mohanty 2026 / BIG-TB authors:**
7. Was the SD-CNN's train/test-split-then-plain-`KFold` (not stratified,
   despite importing `StratifiedKFold`) intentional? Notably, Kulkarni et
   al. 2026 explicitly use *stratified* 5-fold CV ("stratified by binary
   resistance phenotype") for the closely related architecture — worth
   asking whether BIG-TB's non-stratified split was a deliberate choice or
   should not be treated as "the protocol to replicate" for the reported
   0.8753 baseline.
8. Confirm `/project/pi_annagreen_umass_edu/` Unity paths in the SD-CNN
   parameter files (§5.8) are still the current location of the real
   genotype/phenotype data, and what's needed to get access provisioned.
9. Is `mycobrowser_h37rv_genes_v4.csv` / the WHO 2023 catalogue still the
   reference version to build against, or is there a newer curation in
   progress we should target instead?

## 7. Non-goals for this pass

Not building yet: late-fusion+modality-dropout, adversarial lineage
decoupling, MIC+ABR multi-task, causal probing via latent injection.
(Cross-attention fusion moved out of this list 2026-07-01 —
`CrossAttentionFusionCNN` is implemented in `models.py`.) The rest are real
candidates from `experimental_plan.pdf` but come after Phase 5 has a
working biophysical-fusion result to build on.
