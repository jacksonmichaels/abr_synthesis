"""
Multi-drug dataset: load a set of loci once and attach a per-isolate label
MATRIX (one column per drug) instead of a single-drug vector. Every drug is
predicted simultaneously from the same genotype, so loci shared across drugs
(rrs, gyrA, inhA, …) are loaded once.

    from datasets.multidrug import load_multidrug_dataset
    data = load_multidrug_dataset(None, ["dna"], geno_dir, pheno_csv, reg_dir, loci)
    data.drugs            # ['ISONIAZID', 'RIFAMPICIN', ...]  (label-column order)
    data.branch_specs()   # one (5, L_g) spec per locus block
    data.Y                # (N, n_drugs) float: 0=R 1=S -1=missing

Reuses the single-drug modality classes and isolate-axis logic; each locus and
region is its own block.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

import bigtb as tb
from .base import FeatureBlock, LoadContext
from .loader import (DRUG_TO_LOCI, MODALITIES, drug_loci, modality_tag,
                     resolve_modalities)
from .regulatory import DRUG_TO_REGULATORY
from .sequences import load_phenotype, load_sequence_df

# Canonical drug order for the label-matrix columns (the DRUG_TO_LOCI keys).
ALL_DRUGS = list(DRUG_TO_LOCI)


def union_loci(drugs, locus_set="bigtb"):
    """Ordered, de-duplicated union of every drug's ``drug_loci``."""
    out = []
    for d in drugs:
        for g in drug_loci(d, locus_set):
            if g not in out:
                out.append(g)
    return out


def union_regulatory(drugs):
    """Ordered, de-duplicated union of every drug's WHO regulatory regions."""
    out = []
    for d in drugs:
        for r in DRUG_TO_REGULATORY.get(d.upper(), []):
            if r not in out:
                out.append(r)
    return out


@dataclass
class MultiDrugData:
    drugs: List[str]                   # label-column order
    modalities: List[str]              # modalities that produced blocks
    requested: List[str]
    blocks: List[FeatureBlock]
    Y: np.ndarray                      # (N, n_drugs) 0=R 1=S -1=missing
    isolate_ids: List[str]
    loci: List[str]                    # gene loci requested
    gene_order: List[str]              # ...and actually found on disk
    dropped: List[str] = field(default_factory=list)

    @property
    def n(self):
        return len(self.isolate_ids)

    def arrays(self):
        return [b.array for b in self.blocks]

    def branch_specs(self):
        return [b.spec() for b in self.blocks]

    def modality_tag(self):
        return modality_tag(self.modalities)


def load_multidrug_dataset(drugs, modalities, genotype_dir, phenotype_csv,
                           regulatory_dir, loci, verbose=True):
    """Build a MultiDrugData bundle predicting ``drugs`` (None = all 11, in
    ALL_DRUGS order) from ``loci``. Regulatory regions are the union of the
    drugs' WHO sets intersected with ``loci``."""
    drugs = [d.upper() for d in (drugs if drugs else ALL_DRUGS)]
    modalities = resolve_modalities(modalities)

    loci = list(loci)
    regulatory_loci = [r for r in union_regulatory(drugs) if r in set(loci)]

    mods = [MODALITIES[m]() for m in modalities]
    needs_genes = any(m.uses_genes for m in mods)

    df_phenos = load_phenotype(phenotype_csv)

    gene_seqs, gene_order = (load_sequence_df(genotype_dir, loci)
                             if needs_genes and loci else (None, []))
    if needs_genes and verbose:
        missing = [g for g in loci if g not in gene_order]
        if missing:
            print(f"  [load-md] gene loci not found on disk, skipped: {missing}")
    if gene_seqs is None or gene_seqs.empty:
        raise ValueError(f"No gene loci on disk in {genotype_dir}: {loci}")
    isolates = list(gene_seqs.index.intersection(df_phenos.index))
    gene_seqs = gene_seqs.reindex(isolates)

    df_phenos = df_phenos.loc[isolates]
    ctx = LoadContext(drug="+".join(drugs), loci=loci,
                      regulatory_loci=regulatory_loci, isolates=isolates,
                      gene_seqs=gene_seqs, genotype_dir=str(genotype_dir),
                      regulatory_dir=str(regulatory_dir))

    blocks, used, dropped = [], [], []
    for mod in mods:
        mod_blocks = mod.build(ctx)
        if mod_blocks:
            # Downcast one-hot blocks (dna/protein/regulatory are 0/1) to uint8 as
            # EACH modality is built — before the next modality allocates its own
            # float32 arrays — so the load-time peak stays one-modality-sized.
            # biophysical is continuous -> left float32. training.multimodal._batch
            # casts back to float per batch.
            for b in mod_blocks:
                if b.modality in ("dna", "protein", "regulatory") and b.array.dtype != np.uint8:
                    b.array = b.array.astype(np.uint8)
            blocks.extend(mod_blocks)
            used.append(mod.name)
        else:
            dropped.append(mod.name)
            if verbose:
                print(f"  [load-md] modality '{mod.name}' produced no data — dropped.")

    # label matrix (N, n_drugs), columns in `drugs` order, via the BIG-TB encoder
    _, Y = tb.rs_encoding_to_numeric(df_phenos, drugs)
    Y = Y.astype(np.float32)

    if verbose:
        print(f"  [load-md] drugs={len(drugs)} loci={len(gene_order)} "
              f"blocks={len(blocks)} isolates={len(isolates)}")
    return MultiDrugData(drugs=drugs, modalities=used, requested=modalities,
                         blocks=blocks, Y=Y, isolate_ids=isolates, loci=loci,
                         gene_order=gene_order, dropped=dropped)
