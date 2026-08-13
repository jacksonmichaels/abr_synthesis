"""
Biophysical-traits modality: the (3, K) amino-acid property matrix — molecular
weight, isoelectric point, Eisenberg hydrophobicity — one block PER GENE. This
is the modality the whole fusion experiment is about (Kulkarni et al. 2026).

Same translation as the protein modality; the only difference is that each
amino acid maps to its 3 biophysical properties (biochem.biophysical_matrix)
instead of a one-hot identity.
"""
from typing import List

from . import biochem, cds
from .base import FeatureBlock, LoadContext, Modality
from .sequences import delta_zero_columns, reference_row, stack_padded

_WARNED = set()


class BiophysicalModality(Modality):
    name = "biophysical"
    uses_genes = True

    # see datasets/sequences.py. Note this makes the block a signed CHANGE in
    # each property at the substituted residues (the z-scored value of the new
    # residue, zero elsewhere) rather than the absolute property profile.
    delta = False

    def build(self, ctx: LoadContext) -> List[FeatureBlock]:
        seqs = ctx.gene_seqs
        gene_cols = [g for g in ctx.loci if g in seqs.columns]
        if not gene_cols:
            return []
        seqs = seqs.fillna("")
        blocks = []
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
            ref_nt = reference_row(ctx.genotype_dir, g) if self.delta else None
            ref_aa = (biochem.translate_seq(
                cds.cds_slice(ctx.genotype_dir, g, ref_nt, window)) if ref_nt else None)
            if self.delta and not ref_aa:
                print(f"  [load] delta encoding: no usable H37Rv protein for {g} "
                      f"— stays plain properties")
            k_max = max((len(p) for p in proteins), default=1) or 1
            array = stack_padded([
                delta_zero_columns(biochem.biophysical_matrix(p, pad_to=k_max), p, ref_aa)
                if self.delta else biochem.biophysical_matrix(p, pad_to=k_max)
                for p in proteins])
            blocks.append(FeatureBlock(
                name=f"biophysical:{g}",
                modality=self.name,
                array=array,
                channel_names=list(biochem.PROPERTY_NAMES),
                note=f"gene {g}, CDS window, z-scored per property"
                     + (" [delta vs H37Rv]" if self.delta else ""),
            ))
        return blocks
