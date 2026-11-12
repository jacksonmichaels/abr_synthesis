"""
Protein-sequence modality: one-hot amino acids (20 channels), one block PER
GENE. Each isolate's aligned nucleotide sequence is cut to the CDS window
(datasets.cds), translated (biochem.translate_seq: degap, read in frame, stop at
the first stop codon), one-hot encoded and left-aligned/padded to that gene's
max protein length.
"""
from typing import List

from . import biochem, cds
from .base import FeatureBlock, LoadContext, Modality
from .sequences import stack_padded

_WARNED = set()


def locus_residues(ctx: LoadContext, g: str):
    """Per-isolate protein strings for one gene, shared by protein and
    biophysical. Returns ``(proteins, width)``.

    Residue k is the isolate's own k-th codon (degapped, translated in the
    isolate's frame), so an indel shifts everything after it."""
    seqs = ctx.gene_seqs.fillna("")
    window = cds.cds_columns(ctx.genotype_dir, g)
    if window is None and g not in _WARNED:
        _WARNED.add(g)
        print(f"  [load] {g}: no CDS found in the reference record — "
              f"translating the whole aligned window (rRNA loci have no "
              f"protein; this keeps the block shape unchanged)")
    proteins = [biochem.translate_seq(
        cds.cds_slice(ctx.genotype_dir, g, seqs.at[iso, g], window))
        for iso in ctx.isolates]
    k_max = max((len(p) for p in proteins), default=1) or 1
    return proteins, k_max


class ProteinModality(Modality):
    name = "protein"
    uses_genes = True

    def build(self, ctx: LoadContext) -> List[FeatureBlock]:
        seqs = ctx.gene_seqs
        gene_cols = [g for g in ctx.loci if g in seqs.columns]
        if not gene_cols:
            return []
        blocks = []
        for g in gene_cols:
            proteins, width = locus_residues(ctx, g)
            array = stack_padded([biochem.one_hot_aa(p, pad_to=width) for p in proteins])
            blocks.append(FeatureBlock(
                name=f"protein:{g}",
                modality=self.name,
                array=array,
                channel_names=list(biochem.AMINO_ACIDS),
                note=f"gene {g}, CDS window translated + left-aligned",
            ))
        return blocks
