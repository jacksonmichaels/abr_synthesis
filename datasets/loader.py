"""
The single-drug dataloader. Given a drug and a set of modality names, it does
the shared work once (resolve loci, load the aligned FASTAs, align isolates to
the phenotype), runs each requested modality, and returns a ``DrugData`` bundle
of model-ready feature blocks (one per locus/region) plus labels and metadata.

    from datasets.loader import load_dataset
    data = load_dataset("ISONIAZID", ["dna", "biophysical"],
                        genotype_dir, phenotype_csv, regulatory_dir)
    data.branch_specs()     # [(5, L_inhA), (5, L_katG), (3, K_inhA), (3, K_katG)]
    data.arrays()           # aligned (N, C, L) arrays, one per block

Also home to the locus-set policy (``drug_loci``, ``LOCUS_SETS``,
``loci_on_disk``) shared with the multi-drug loader, LASSO and GenomeFusion.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

import bigtb as tb
from .base import FeatureBlock, LoadContext
from .biophysical import BiophysicalModality
from .dna import DNAModality
from .protein import ProteinModality
from .regulatory import DRUG_TO_REGULATORY, RegulatoryModality
from .sequences import load_phenotype, load_sequence_df, numeric_labels

# registry order is block order and the order of a run's modality tag
MODALITIES = {
    "dna": DNAModality,
    "protein": ProteinModality,
    "biophysical": BiophysicalModality,
    "regulatory": RegulatoryModality,
}

DRUG_TO_LOCI = tb.DRUG_TO_LOCI

# Which gene loci a drug loads. "bigtb" is BIG-TB's per-drug DRUG_TO_LOCI (the
# SD-CNN rule). "lab64" is every locus in saishradha's genomic_data/aligned
# folder (the 19 curated + 45 WHO candidates), given to EVERY drug.
# "who_t12" is the drug's WHO 2023 Table-21 tier 1 + tier 2 candidate genes,
# tier 1 first (GenomeFusion resolves these from GenBank; four of them -- dnaA,
# PPE35, rpoA, whiB6 -- have no FASTA in her folder). "who_t12_sai" is who_t12
# limited to loci she provides; pepQ is left out because the locus_sets sweep
# that defined this set used a different pepQ extraction (Rv2535c) than her
# pepQ.fasta (Rv3698).
LOCUS_SETS = ("bigtb", "who_t12", "lab64", "who_t12_sai")
# The 19 curated resistance loci (the union of DRUG_TO_LOCI plus fabG1), sorted:
# the locus set of every "19-locus" run, and of LocusFusion and GenomeFusion's `19`.
LOCI_19 = ("eis", "embA", "embB", "embC", "ethA", "ethR", "fabG1", "gid", "gyrA", "gyrB",
           "inhA", "katG", "pncA", "rpoB", "rpoC", "rpsL", "rrl", "rrs", "tlyA")
LAB64_LOCI = (
    "aftA", "aftB", "ahpC", "atpE", "bacA", "ccsA", "clpC1", "ddn", "eis", "embA",
    "embB", "embC", "embR", "ethA", "ethR", "fabG1", "fbiA", "fbiB", "fbiC", "fgd1",
    "furA", "gid", "glpK", "gyrA", "gyrB", "hadA", "inhA", "katG", "lpqB", "mmpL5",
    "mmpS5", "mshA", "mtrA", "mtrB", "ndh", "nusG", "panD", "pepQ", "pncA", "rplC",
    "rpoB", "rpoC", "rpsA", "rpsL", "rrl", "rrs", "Rv0010c", "Rv0565c", "Rv0678",
    "Rv1129c", "Rv1258c", "Rv1979c", "Rv2477c", "Rv2680", "Rv2681", "Rv2752c",
    "Rv2983", "Rv3083", "Rv3236c", "sigE", "tlyA", "tsnR", "ubiA", "whiB7")
_WHO_T12_SAI_LOCI = tuple(g for g in LAB64_LOCI if g != "pepQ")


def drug_loci(drug, locus_set="bigtb"):
    """Gene loci for one drug under ``locus_set`` (see LOCUS_SETS)."""
    if locus_set not in LOCUS_SETS:
        raise ValueError(f"unknown locus_set {locus_set!r}; choose from {LOCUS_SETS}")
    if locus_set == "lab64":
        return list(LAB64_LOCI)
    if locus_set in ("who_t12", "who_t12_sai"):
        from .regulatory import DRUG_NAME_TO_WHO
        from .who_catalogue import genes_for_drug
        genes = dict.fromkeys(g for g, _ in genes_for_drug(DRUG_NAME_TO_WHO[drug.upper()]))
        return [g for g in genes if locus_set == "who_t12" or g in _WHO_T12_SAI_LOCI]
    return list(DRUG_TO_LOCI.get(drug.upper(), []))


def resolve_modalities(modalities):
    """Validate and put in registry order (dedupes)."""
    unknown = [m for m in modalities if m not in MODALITIES]
    if unknown:
        raise ValueError(
            f"Unknown modalit{'y' if len(unknown) == 1 else 'ies'} {unknown}. "
            f"Available: {list(MODALITIES)}")
    return [m for m in MODALITIES if m in set(modalities)]


def modality_tag(modalities):
    """Compact, stable tag for output filenames, e.g. 'dna+biophysical'."""
    return "+".join(m for m in MODALITIES if m in modalities)


@dataclass
class DrugData:
    drug: str
    modalities: List[str]              # modalities that actually produced blocks
    requested: List[str]               # what the caller asked for
    blocks: List[FeatureBlock]
    y: np.ndarray                      # (N,) 0=R 1=S -1=missing
    isolate_ids: List[str]
    loci: List[str]
    gene_order: List[str]              # gene loci actually found on disk
    dropped: List[str] = field(default_factory=list)  # requested but empty

    @property
    def n(self):
        return len(self.isolate_ids)

    def arrays(self):
        return [b.array for b in self.blocks]

    def branch_specs(self):
        return [b.spec() for b in self.blocks]

    def modality_tag(self):
        return modality_tag(self.modalities)


def load_dataset(drug, modalities, genotype_dir, phenotype_csv, regulatory_dir,
                 loci=None, locus_set="bigtb", verbose=True):
    """Build a DrugData bundle for one drug over the requested modalities.

    loci      : gene loci for dna/protein/biophysical; default
                ``drug_loci(drug, locus_set)``.
    Regulatory regions are the drug's WHO set (DRUG_TO_REGULATORY) intersected
    with the loaded loci, so a run never carries more promoter windows than
    coding loci. Requested loci / regions with no FASTA on disk are skipped
    with a warning."""
    drug = drug.upper()
    modalities = resolve_modalities(modalities)

    user_loci = loci is not None
    loci = list(loci) if user_loci else drug_loci(drug, locus_set)
    regulatory_loci = [r for r in DRUG_TO_REGULATORY.get(drug, []) if r in set(loci)]

    mods = [MODALITIES[m]() for m in modalities]
    needs_genes = any(m.uses_genes for m in mods)

    df_phenos = load_phenotype(phenotype_csv)

    # --- establish the shared isolate axis ---------------------------------
    gene_seqs, gene_order = (load_sequence_df(genotype_dir, loci)
                             if needs_genes and loci else (None, []))
    if needs_genes and verbose:
        missing = [g for g in loci if g not in gene_order]
        if missing:
            src = "--loci" if user_loci else "the default locus set"
            print(f"  [load] {drug}: gene loci from {src} not found on disk, "
                  f"skipped: {missing}")
    if gene_seqs is None or gene_seqs.empty:
        raise ValueError(f"No gene loci on disk for {drug} in {genotype_dir}: {loci}")
    isolates = list(gene_seqs.index.intersection(df_phenos.index))
    gene_seqs = gene_seqs.reindex(isolates)

    df_phenos = df_phenos.loc[isolates]
    ctx = LoadContext(drug=drug, loci=loci, regulatory_loci=regulatory_loci,
                      isolates=isolates, gene_seqs=gene_seqs,
                      genotype_dir=str(genotype_dir), regulatory_dir=str(regulatory_dir))

    # --- run each modality --------------------------------------------------
    blocks, used, dropped = [], [], []
    for mod in mods:
        mod_blocks = mod.build(ctx)
        if mod.name == "regulatory" and verbose:
            loaded = {b.name.split(":", 1)[1] for b in mod_blocks}
            miss = [r for r in regulatory_loci if r not in loaded]
            if miss:
                print(f"  [load] {drug}: regulatory regions from the WHO default set "
                      f"not found on disk, skipped: {miss}")
        if mod_blocks:
            blocks.extend(mod_blocks)
            used.append(mod.name)
        else:
            dropped.append(mod.name)
            if verbose:
                print(f"  [load] {drug}: modality '{mod.name}' produced no data "
                      f"— dropped.")

    y = numeric_labels(df_phenos, drug)
    return DrugData(drug=drug, modalities=used, requested=modalities, blocks=blocks,
                    y=y, isolate_ids=isolates, loci=loci, gene_order=gene_order,
                    dropped=dropped)
