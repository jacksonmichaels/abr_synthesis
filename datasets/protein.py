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

from . import biochem, cds
from .base import FeatureBlock, LoadContext, Modality
from .sequences import delta_zero_columns, reference_row, stack_padded

_WARNED = set()


FRAMES = ("isolate", "reference")


def locus_residues(ctx: LoadContext, g: str, frame: str, delta: bool):
    """Per-isolate residue strings for one gene, shared by protein and biophysical.

    Returns ``(residues, ref_residues, width, frame)``. ``frame`` is echoed back
    because a locus with no H37Rv row cannot be read on the reference grid and
    falls back to the isolate frame.

    * ``"isolate"`` — the original featurization, byte-identical to every run
      before 2026-09-17: degap, translate in the isolate's own frame, stop at the
      first stop, left-align, pad to the longest. Residue k is the isolate's k-th
      codon, so an indel shifts everything after it.
    * ``"reference"`` — ``biochem.translate_reference_frame``: one residue per
      H37Rv codon, stops kept as ``'*'``, fixed width. Residue k is reference
      codon k in every isolate. models.locusfusion_v2 requires this.
    """
    if frame not in FRAMES:
        raise ValueError(f"unknown protein frame {frame!r}; choose from {list(FRAMES)}")
    seqs = ctx.gene_seqs.fillna("")
    window = cds.cds_columns(ctx.genotype_dir, g)
    if window is None and g not in _WARNED:
        _WARNED.add(g)
        print(f"  [load] {g}: no CDS found in the reference record — "
              f"translating the whole aligned window (rRNA loci have no "
              f"protein; this keeps the block shape unchanged)")
    if frame == "reference":
        ref_nt = reference_row(ctx.genotype_dir, g)
        if ref_nt:
            residues = [biochem.translate_reference_frame(seqs.at[iso, g], ref_nt, window)
                        for iso in ctx.isolates]
            ref_aa = biochem.translate_reference_frame(ref_nt, ref_nt, window)
            return residues, (ref_aa if delta else None), max(len(ref_aa), 1), frame
        print(f"  [load] {g}: no H37Rv row, so no reference codon grid — "
              f"translated in the isolate frame instead")
        frame = "isolate"
    proteins = [biochem.translate_seq(
        cds.cds_slice(ctx.genotype_dir, g, seqs.at[iso, g], window))
        for iso in ctx.isolates]
    # the reference protein comes through the identical CDS-slice + translate
    # path, so residue k of the reference is residue k of every isolate wherever
    # no indel has shifted the frame
    ref_nt = reference_row(ctx.genotype_dir, g) if delta else None
    ref_aa = (biochem.translate_seq(
        cds.cds_slice(ctx.genotype_dir, g, ref_nt, window)) if ref_nt else None)
    k_max = max((len(p) for p in proteins), default=1) or 1
    return proteins, ref_aa, k_max, frame


class ProteinModality(Modality):
    name = "protein"
    uses_genes = True

    # see datasets/sequences.py — zero the residues matching the translated
    # H37Rv protein, so the block carries the substitutions and nothing else.
    delta = False

    # which codon grid residue k is on — see locus_residues. "reference" adds a
    # 21st channel, STOP, since a stop codon there is a residue, not an ending.
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
                      f"— stays plain one-hot")
            stop = frame == "reference"
            unk = self.unknown_symbols
            array = stack_padded([
                delta_zero_columns(biochem.one_hot_aa(p, pad_to=width, with_stop=stop,
                                                      with_unknown=unk), p, ref_aa)
                if self.delta else biochem.one_hot_aa(p, pad_to=width, with_stop=stop,
                                                      with_unknown=unk)
                for p in proteins])
            blocks.append(FeatureBlock(
                name=f"protein:{g}",
                modality=self.name,
                array=array,
                channel_names=biochem.residue_channels(with_stop=stop, with_unknown=unk),
                note=(f"gene {g}, one residue per H37Rv codon (stops kept)" if stop
                      else f"gene {g}, CDS window translated + left-aligned")
                     + (" [delta vs H37Rv]" if self.delta else ""),
                # The reference residue of every codon, so a model reading a
                # delta-encoded block can say what the codon read BEFORE the
                # substitution (and, with biochem.AA_PROPERTY, what it was
                # worth biophysically). Only meaningful on the reference grid.
                meta={"ref_residues": ref_aa} if (stop and ref_aa) else {},
            ))
        return blocks
