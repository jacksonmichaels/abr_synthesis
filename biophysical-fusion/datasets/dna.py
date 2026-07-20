"""
DNA-sequence modality: one-hot nucleotides, the drug's gene loci concatenated
into a single contiguous locus (matches Kulkarni et al. 2026's nucleotide
featurization and BIG-TB's SD-CNN input). One block, 5 channels (A,C,T,G,-).
"""
from typing import List

from .base import FeatureBlock, LoadContext, Modality
from .sequences import NT_CHANNELS, one_hot_nt, stack_padded


class DNAModality(Modality):
    name = "dna"
    uses_genes = True

    def build(self, ctx: LoadContext) -> List[FeatureBlock]:
        seqs = ctx.gene_seqs
        gene_cols = [g for g in ctx.loci if g in seqs.columns]
        if not gene_cols:
            return []
        seqs = seqs[gene_cols].fillna("")
        mats = []
        for iso in ctx.isolates:
            full = "".join(seqs.at[iso, g] for g in gene_cols)
            mats.append(one_hot_nt(full).T)  # (L,5) -> (5,L)
        array = stack_padded(mats)
        return [FeatureBlock(
            name="dna",
            modality=self.name,
            array=array,
            channel_names=list(NT_CHANNELS),
            note=f"genes concatenated: {'+'.join(gene_cols)}",
        )]
