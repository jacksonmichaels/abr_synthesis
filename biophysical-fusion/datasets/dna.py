"""
DNA-sequence modality: one-hot nucleotides, the drug's gene loci concatenated
into a single contiguous locus (matches Kulkarni et al. 2026's nucleotide
featurization and BIG-TB's SD-CNN input). One block, 5 channels (A,C,T,G,-).
"""
from typing import List

import numpy as np

from .base import FeatureBlock, LoadContext, Modality
from .sequences import (NT_CHANNELS, delta_one_hot_nt, one_hot_nt,
                        reference_row, stack_padded)


class DNAModality(Modality):
    name = "dna"
    uses_genes = True

    # Zero every column that matches the H37Rv reference (see
    # datasets/sequences.py). Shape and alphabet are unchanged; only the
    # cohort-constant columns go. Default off — it changes the input.
    delta = False

    # When True, emit ONE block per locus (a separate branch each) instead of a
    # single block with all loci one-hot-concatenated. The multi-drug loader
    # sets this so each locus gets its own CNN encoder (MD-CNN-style); the
    # single-drug path leaves it False and keeps the concatenated SD-CNN input.
    per_locus = False

    def build(self, ctx: LoadContext) -> List[FeatureBlock]:
        seqs = ctx.gene_seqs
        gene_cols = [g for g in ctx.loci if g in seqs.columns]
        if not gene_cols:
            return []
        seqs = seqs[gene_cols].fillna("")

        refs = ({g: reference_row(ctx.genotype_dir, g) for g in gene_cols}
                if self.delta else {})
        if self.delta:
            missing = [g for g, r in refs.items() if not r]
            if missing:
                print(f"  [load] delta encoding: no H37Rv row for {missing} — "
                      f"those loci stay plain one-hot")

        def encode(seq, ref):
            return (delta_one_hot_nt(seq, ref) if self.delta else one_hot_nt(seq)).T

        suffix = " (delta vs H37Rv)" if self.delta else ""
        if self.per_locus:
            blocks = []
            for g in gene_cols:
                mats = [encode(seqs.at[iso, g], refs.get(g)) for iso in ctx.isolates]
                blocks.append(FeatureBlock(
                    name=f"dna:{g}",
                    modality=self.name,
                    array=stack_padded(mats),
                    channel_names=list(NT_CHANNELS),
                    note=f"locus {g}{suffix}",
                ))
            return blocks

        mats = []
        for iso in ctx.isolates:
            if self.delta:
                # encode each locus against ITS OWN reference and concatenate the
                # arrays. Concatenating the reference strings first would go
                # wrong the moment one locus has no H37Rv row: the empty string
                # shifts every downstream locus out of register and the mask
                # would zero the wrong columns.
                mats.append(np.concatenate(
                    [encode(seqs.at[iso, g], refs.get(g)) for g in gene_cols], axis=1))
            else:
                full = "".join(seqs.at[iso, g] for g in gene_cols)
                mats.append(encode(full, None))  # (L,5) -> (5,L)
        array = stack_padded(mats)
        return [FeatureBlock(
            name="dna",
            modality=self.name,
            array=array,
            channel_names=list(NT_CHANNELS),
            note=f"genes concatenated: {'+'.join(gene_cols)}{suffix}",
        )]
