# Biophysical Fusion — TODO

Isolated workspace. We only *read* reference code from `Big-TB-benchmark/`;
nothing here modifies it. Goal: test whether a biophysical-property branch
improves BIG-TB's DNA CNN on binary resistance, then extend to other
modalities.

## The experiment

BIG-TB's best single-drug model is the DNA one-hot CNN (AUC 0.8753).
Kulkarni et al. 2026 fused amino-acid **molecular weight, isoelectric point,
and Eisenberg hydrophobicity** as a 3×K matrix through a separate conv branch
alongside the nucleotide input. **Test: does that branch improve BIG-TB's DNA
CNN on binary R/S classification?** Start with Isoniazid (inhA+katG) and
Rifampicin (rpoB+rpoC).

## Status

**Done**
- Synthetic fixtures, biophysical property table + translation, dataset
  builder, fusion models, CV/eval harness — all implemented and passing the
  synthetic smoke test (`run_experiment.py`).
- Models: `MultiModalNet` (generic late fusion, **per-branch encoder choice** via
  `models.ENCODERS` — `cnn` / `transformer`); `run_experiment.py --encoders
  dna=cnn protein=transformer …`. Also `LateFusionCNN` / `EarlyFusionCNN` /
  `CrossAttentionFusionCNN`; diagrams (`gen_model_diagrams.py` → `diagrams/`).
- Real data wired in (Unity access, 2026-07-14): `bigtb_ref.REAL_GENOTYPE_DIR`
  / `REAL_PHENOTYPE_CSV`, joined on `New_ID` like BIG-TB's
  `make_geno_pheno_dataset` (~17.9k isolates).
- **DNA-only baseline** (`eval_dna_cnn.py`, `models.DNAOnlyCNN`): full real-data
  eval per drug, mini-batched, held-out test metric. First result — ISONIAZID
  CV fold-0 AUC 0.926. Full 11-drug run done via TensorBoard.

**Data layer (refactored 2026-07-20)**
- All modality loading lives in the **`datasets/`** package, one file per
  modality (`dna.py`, `protein.py`, `biophysical.py`, `regulatory.py`) behind a
  single `datasets.load_dataset(drug, modalities, geno_dir, pheno_csv)` →
  `DrugData` bundle of model-ready blocks. Shared substrate: `sequences.py`
  (FASTA/label loading, one-hot), `biochem.py` (AA science), `base.py`
  (`Modality`/`FeatureBlock`). Add a modality = one entry in `loader.MODALITIES`.
- `models.MultiModalCNN` late-fuses any set of blocks (subsumes DNAOnly/Late
  fusion). `train_multimodal.run_modal_cv` is the generic mini-batched CV+test
  engine. `run_experiment.py --modalities dna biophysical …` runs any subset and
  names outputs `results/experiments/{run}/{DRUG}__{modality-tag}.json`.
- `data.py` is now a thin legacy adapter over `datasets/` (keeps
  `eval_dna_cnn.py` working); old top-level `biophysical.py` is a shim →
  `datasets.biochem`.
- `datasets_overview.ipynb` visualizes each modality's shapes/coverage (runs on
  synthetic fixtures by default; `USE_REAL=True` for cluster data).
- **Locus selection**: `load_dataset(..., loci=[...], regulatory_loci=[...])` /
  `run_experiment.py --loci --regulatory-loci` pick which & how many loci each
  modality loads; defaults = `DRUG_TO_LOCI` (genes) and the WHO-derived set (regulatory).
- Regulatory regions are WHO-2023-catalogue-driven (`datasets/who_catalogue.py`,
  Tables 21 & 22): per-drug default = WHO candidate genes − coding loci. Real
  data loads what exists — *fabG1* for INH/ETO, *eis* for KAN; Table-22
  upstream coords/TSS ride along as block metadata for future promoter-slicing.

**Next**
1. DNA+biophysical fusion on real data: `run_experiment.py --modalities dna
   biophysical --drugs ISONIAZID RIFAMPICIN --device cuda`.
2. DNA-only vs. DNA+biophysical comparison — the core result.
3. Then: slot in the next modality (lineage vector is cheapest).

## Open questions / risks

- **Biophysical table values**: Kulkarni names the 3 properties but not the
  literal MW/pI numbers or normalization. Current `biophysical.py` uses
  standard published tables, z-scored per channel — plausible but unconfirmed.
- **Stop-codon truncation**: we translate up to (not past) the first stop to
  distinguish nonsense from missense. Inferred, not stated in the paper.
- **Lineage confounding**: population structure correlates with both variants
  and phenotype — a multimodal model can shortcut through it. Decide on
  lineage-stratified eval / decoupling before adding a lineage branch.
- **CV protocol**: we replicate BIG-TB's split-then-plain-KFold. Kulkarni uses
  *stratified* CV — worth revisiting whether non-stratified is really "the
  protocol to replicate."

## Key reference files (in `Big-TB-benchmark/`)

- DNA CNN + one-hot + drug→loci map + alpha weighting + threshold search:
  `dna-tasks/SD-CNN/model_training/parameters/tb_cnn_codebase.py`
- DNA CNN training loop: `dna-tasks/SD-CNN/model_training/run_SDCNN_ccp_crossval.py`
- Protein CNN (conv-stack shape reused for `ConvBranch`):
  `protein-tasks/one_hot_encoded/cnn_model.py`
- Gap-aware translation (not yet wired in): `protein-tasks/protein_translation/`

## Environment

Use the **`abr_env` conda env** (`conda activate abr_env`; base has no torch).
Python 3.12, torch 2.6.0+cu124, GPU works (RTX 2080 Ti). `tb_cnn_codebase.py`
pulls in biopython/sparse/h5py/tensorflow (CPU) at import — all installed.
