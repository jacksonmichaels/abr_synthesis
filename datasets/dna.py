"""
DNA-sequence modality: one-hot nucleotides, 5 channels (A,C,T,G,-), one block
PER LOCUS. The stacked CNN stacks those blocks as input channels, so it has to
see each locus separately.
"""
from typing import List

from . import cds
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
        blocks = []
        for g in gene_cols:
            array = stack_padded([one_hot_nt(seqs.at[iso, g]).T for iso in ctx.isolates])
            blocks.append(FeatureBlock(
                name=f"dna:{g}",
                modality=self.name,
                array=array,
                channel_names=list(NT_CHANNELS),
                note=f"locus {g}",
                meta=cds.block_frame(ctx.genotype_dir, g, array.shape[2],
                                     translated=True),
            ))
        return blocks
