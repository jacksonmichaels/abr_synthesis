"""
Biophysical-traits modality: the (3, K) amino-acid property matrix — molecular
weight, isoelectric point, Eisenberg hydrophobicity — one block PER GENE. This
is the modality the whole fusion experiment is about (Kulkarni et al. 2026).

Same translation as the protein modality; the only difference is that each
amino acid maps to its 3 biophysical properties (biochem.biophysical_matrix)
instead of a one-hot identity.
"""
from typing import List

from . import biochem
from .base import FeatureBlock, LoadContext, Modality
from .sequences import stack_padded


class BiophysicalModality(Modality):
    name = "biophysical"
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
            array = stack_padded([biochem.biophysical_matrix(p, pad_to=k_max) for p in proteins])
            blocks.append(FeatureBlock(
                name=f"biophysical:{g}",
                modality=self.name,
                array=array,
                channel_names=list(biochem.PROPERTY_NAMES),
                note=f"gene {g}, z-scored per property",
            ))
        return blocks
