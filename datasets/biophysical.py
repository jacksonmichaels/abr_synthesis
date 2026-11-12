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
from .protein import locus_residues
from .sequences import delta_zero_columns, stack_padded


class BiophysicalModality(Modality):
    name = "biophysical"
    uses_genes = True

    # see datasets/sequences.py. Note this makes the block a signed CHANGE in
    # each property at the substituted residues (the z-scored value of the new
    # residue, zero elsewhere) rather than the absolute property profile.
    delta = False

    # which codon grid residue k is on — see datasets.protein.locus_residues. A
    # stop codon has no properties, so it is the zero vector here; the protein
    # block's STOP channel is what carries it.
    frame = "isolate"

    def build(self, ctx: LoadContext) -> List[FeatureBlock]:
        seqs = ctx.gene_seqs
        gene_cols = [g for g in ctx.loci if g in seqs.columns]
        if not gene_cols:
            return []
        blocks = []
        for g in gene_cols:
            proteins, ref_aa, width, frame = locus_residues(ctx, g, self.frame, self.delta)
            if self.delta and not ref_aa:
                print(f"  [load] delta encoding: no usable H37Rv protein for {g} "
                      f"— stays plain properties")
            array = stack_padded([
                delta_zero_columns(biochem.biophysical_matrix(p, pad_to=width), p, ref_aa)
                if self.delta else biochem.biophysical_matrix(p, pad_to=width)
                for p in proteins])
            blocks.append(FeatureBlock(
                name=f"biophysical:{g}",
                modality=self.name,
                array=array,
                channel_names=list(biochem.PROPERTY_NAMES),
                # see ProteinModality.build -- the reference residue per codon,
                # which is what `ref_biophysical` and the delta are read from
                meta=({"ref_residues": ref_aa}
                      if (frame == "reference" and ref_aa) else {}),
                note=(f"gene {g}, one residue per H37Rv codon, z-scored per property"
                      if frame == "reference"
                      else f"gene {g}, CDS window, z-scored per property")
                     + (" [delta vs H37Rv]" if self.delta else ""),
            ))
        return blocks
