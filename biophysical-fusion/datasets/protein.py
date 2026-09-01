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

import numpy as np

from . import biochem, cds, tokens
from .base import FeatureBlock, LoadContext, Modality
from .sequences import delta_zero_columns, reference_row, stack_padded

_WARNED = set()


class ProteinModality(Modality):
    name = "protein"
    uses_genes = True

    # see datasets/sequences.py — zero the residues matching the translated
    # H37Rv protein, so the block carries the substitutions and nothing else.
    delta = False

    # One symbol id per residue instead of the 20-channel one-hot; see
    # datasets/tokens.py. Residues past this isolate's own protein become
    # ``AA_UNK`` rather than an all-zero column, so a premature stop is a
    # deviation from the reference instead of silence.
    variant_tokens = False

    def build(self, ctx: LoadContext) -> List[FeatureBlock]:
        seqs = ctx.gene_seqs
        gene_cols = [g for g in ctx.loci if g in seqs.columns]
        if not gene_cols:
            return []
        seqs = seqs.fillna("")
        blocks = []

        if self.variant_tokens:
            for g in gene_cols:
                window = cds.cds_columns(ctx.genotype_dir, g)
                proteins = [biochem.translate_seq(
                    cds.cds_slice(ctx.genotype_dir, g, seqs.at[iso, g], window))
                    for iso in ctx.isolates]
                ref_nt = reference_row(ctx.genotype_dir, g)
                ref_aa = (biochem.translate_seq(
                    cds.cds_slice(ctx.genotype_dir, g, ref_nt, window))
                    if ref_nt else "")
                k_max = max((len(p) for p in proteins), default=1) or 1
                ids = np.stack([tokens.aa_symbol_ids(p, pad_to=k_max)
                                for p in proteins])[:, None, :]
                blocks.append(FeatureBlock(
                    name=f"protein:{g}",
                    modality=self.name,
                    array=ids,
                    channel_names=["symbol_id"],
                    note=f"gene {g}, CDS window translated "
                         f"(symbol ids, H37Rv reference in column_meta)",
                    column_meta=tokens.protein_column_meta(ref_aa, k_max),
                ))
            return blocks

        for g in gene_cols:
            window = cds.cds_columns(ctx.genotype_dir, g)
            if window is None and g not in _WARNED:
                _WARNED.add(g)
                print(f"  [load] {g}: no CDS found in the reference record — "
                      f"translating the whole aligned window (rRNA loci have no "
                      f"protein; this keeps the block shape unchanged)")
            proteins = [biochem.translate_seq(
                cds.cds_slice(ctx.genotype_dir, g, seqs.at[iso, g], window))
                for iso in ctx.isolates]
            # the reference protein comes through the identical CDS-slice +
            # translate path, so residue k of the reference is residue k of every
            # isolate wherever no indel has shifted the frame
            ref_nt = reference_row(ctx.genotype_dir, g) if self.delta else None
            ref_aa = (biochem.translate_seq(
                cds.cds_slice(ctx.genotype_dir, g, ref_nt, window)) if ref_nt else None)
            if self.delta and not ref_aa:
                print(f"  [load] delta encoding: no usable H37Rv protein for {g} "
                      f"— stays plain one-hot")
            k_max = max((len(p) for p in proteins), default=1) or 1
            array = stack_padded([
                delta_zero_columns(biochem.one_hot_aa(p, pad_to=k_max), p, ref_aa)
                if self.delta else biochem.one_hot_aa(p, pad_to=k_max)
                for p in proteins])
            blocks.append(FeatureBlock(
                name=f"protein:{g}",
                modality=self.name,
                array=array,
                channel_names=list(biochem.AMINO_ACIDS),
                note=f"gene {g}, CDS window translated + left-aligned"
                     + (" [delta vs H37Rv]" if self.delta else ""),
            ))
        return blocks
