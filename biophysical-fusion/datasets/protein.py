"""
Protein-sequence modality: one-hot amino acids (20 channels), one block PER
GENE. Each isolate's aligned nucleotide sequence is translated (biochem.
translate_seq: degap, read in frame, stop at the first stop codon) then
one-hot encoded and left-aligned/padded to that gene's max protein length.

Separate block per gene mirrors Kulkarni et al. 2026's "individual proteins are
separate channels even if translated from the same locus" (see TODO.md) — the
same reason biophysical traits are per-gene.
"""
from typing import List

from . import biochem
from .base import FeatureBlock, LoadContext, Modality
from .sequences import stack_padded


class ProteinModality(Modality):
    name = "protein"
    uses_genes = True

    def build(self, ctx: LoadContext) -> List[FeatureBlock]:
        seqs = ctx.gene_seqs
        gene_cols = [g for g in ctx.loci if g in seqs.columns]
        if not gene_cols:
            return []
        seqs = seqs.fillna("")
        blocks = []
        for g in gene_cols:
            proteins = [biochem.translate_seq(seqs.at[iso, g]) for iso in ctx.isolates]
            k_max = max((len(p) for p in proteins), default=1) or 1
            array = stack_padded([biochem.one_hot_aa(p, pad_to=k_max) for p in proteins])
            blocks.append(FeatureBlock(
                name=f"protein:{g}",
                modality=self.name,
                array=array,
                channel_names=list(biochem.AMINO_ACIDS),
                note=f"gene {g}, translated + left-aligned",
            ))
        return blocks
