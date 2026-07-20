"""
The single dataloader. Given a drug and a set of modality names, it does the
shared work once (resolve loci, load the aligned FASTAs, align isolates to the
phenotype), runs each requested modality, and returns a ``DrugData`` bundle of
model-ready feature blocks plus labels and metadata.

    from datasets import load_dataset, MODALITIES
    data = load_dataset("ISONIAZID", ["dna", "biophysical"],
                        genotype_dir, phenotype_csv)
    data.branch_specs()     # [(5, L), (3, K_inhA), (3, K_katG)]
    data.arrays()           # aligned (N, C, L) arrays, one per branch

Adding a modality is a one-line entry in MODALITIES; nothing else changes.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from bigtb_ref import tb
from .base import FeatureBlock, LoadContext
from .biophysical import BiophysicalModality
from .dna import DNAModality
from .protein import ProteinModality
from .regulatory import DRUG_TO_REGULATORY, RegulatoryModality
from .sequences import load_phenotype, load_sequence_df, numeric_labels

# the registry — the whole point of the refactor: any subset can be requested
MODALITIES = {
    "dna": DNAModality,
    "protein": ProteinModality,
    "biophysical": BiophysicalModality,
    "regulatory": RegulatoryModality,
}

DRUG_TO_LOCI = tb.DRUG_TO_LOCI


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

    def class_counts(self):
        return {"R": int((self.y == 0).sum()),
                "S": int((self.y == 1).sum()),
                "missing": int((self.y == -1).sum())}

    def modality_tag(self):
        """Compact, stable tag for output filenames, e.g. 'dna+biophysical'."""
        return "+".join(m for m in MODALITIES if m in self.modalities)

    def summary(self):
        return {
            "drug": self.drug,
            "modalities": self.modalities,
            "dropped": self.dropped,
            "n_isolates": self.n,
            "genes": self.gene_order,
            "class_counts": self.class_counts(),
            "blocks": [
                {"name": b.name, "modality": b.modality,
                 "shape": tuple(b.array.shape), "channels": b.channel_names,
                 "note": b.note}
                for b in self.blocks
            ],
        }


def _resolve(modalities):
    unknown = [m for m in modalities if m not in MODALITIES]
    if unknown:
        raise ValueError(
            f"Unknown modalit{'y' if len(unknown) == 1 else 'ies'} {unknown}. "
            f"Available: {list(MODALITIES)}")
    # keep registry order for determinism, dedupe
    return [m for m in MODALITIES if m in set(modalities)]


def load_dataset(drug, modalities, genotype_dir, phenotype_csv,
                 regulatory_dir=None, loci=None, regulatory_loci=None, verbose=True):
    """Build a DrugData bundle for one drug over the requested modalities.

    loci             : which gene loci to load for dna/protein/biophysical.
                       Default = DRUG_TO_LOCI[drug] (the current behavior). Pass
                       a subset/reordering to control which — and how many — are
                       loaded, e.g. loci=['katG'].
    regulatory_loci  : which regulatory regions to load. Default = the WHO-derived
                       DRUG_TO_REGULATORY[drug]. Pass a list to override.
    Requested loci with no FASTA on disk are skipped (warned when explicit)."""
    drug = drug.upper()
    modalities = _resolve(modalities)
    regulatory_dir = regulatory_dir or genotype_dir

    user_loci = loci is not None
    loci = list(loci) if user_loci else list(DRUG_TO_LOCI.get(drug, []))
    user_reg = regulatory_loci is not None
    regulatory_loci = (list(regulatory_loci) if user_reg
                       else list(DRUG_TO_REGULATORY.get(drug, [])))

    mods = [MODALITIES[m]() for m in modalities]
    needs_genes = any(m.uses_genes for m in mods)

    df_phenos = load_phenotype(phenotype_csv)

    # --- establish the shared isolate axis ---------------------------------
    gene_seqs, gene_order = (load_sequence_df(genotype_dir, loci)
                             if needs_genes and loci else (None, []))
    if needs_genes and user_loci and verbose:
        missing = [g for g in loci if g not in gene_order]
        if missing:
            print(f"  [load] {drug}: gene loci not found on disk, skipped: {missing}")

    if gene_seqs is not None and not gene_seqs.empty:
        isolates = list(gene_seqs.index.intersection(df_phenos.index))
        gene_seqs = gene_seqs.reindex(isolates)
    else:
        # regulatory-only (no gene modality / no gene loci): bootstrap the
        # isolate axis from the regulatory region FASTAs.
        reg_df, _ = load_sequence_df(regulatory_dir, regulatory_loci)
        if reg_df.empty:
            raise ValueError(
                f"No sequence data for {drug} with modalities {modalities} "
                f"(no gene loci found and no regulatory regions).")
        isolates = list(reg_df.index.intersection(df_phenos.index))

    df_phenos = df_phenos.loc[isolates]
    ctx = LoadContext(drug=drug, loci=loci, regulatory_loci=regulatory_loci,
                      isolates=isolates, gene_seqs=gene_seqs,
                      genotype_dir=str(genotype_dir), regulatory_dir=str(regulatory_dir))

    # --- run each modality --------------------------------------------------
    blocks, used, dropped = [], [], []
    for mod in mods:
        mod_blocks = mod.build(ctx)
        if mod.name == "regulatory" and user_reg and verbose:
            loaded = {b.name.split(":", 1)[1] for b in mod_blocks}
            miss = [r for r in regulatory_loci if r not in loaded]
            if miss:
                print(f"  [load] {drug}: regulatory regions not found on disk, "
                      f"skipped: {miss}")
        if mod_blocks:
            blocks.extend(mod_blocks)
            used.append(mod.name)
        else:
            dropped.append(mod.name)
            if verbose:
                print(f"  [load] {drug}: modality '{mod.name}' produced no data "
                      f"— dropped.")

    if not blocks:
        raise ValueError(f"No modality produced data for {drug} "
                         f"(requested {modalities}).")

    y = numeric_labels(df_phenos, drug)
    return DrugData(drug=drug, modalities=used, requested=modalities, blocks=blocks,
                    y=y, isolate_ids=isolates, loci=loci, gene_order=gene_order,
                    dropped=dropped)
