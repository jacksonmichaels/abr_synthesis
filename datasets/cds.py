"""
Where the coding sequence actually starts inside an aligned locus FASTA.

The aligned gene FASTAs are not bare CDS: they carry flanking sequence, and the
CDS does not begin at column 0. katG's record is 2,423 bp around a 2,223 bp CDS,
so translating from column 0 reads the WRONG FRAME and dies at the first stop —
20 residues for the reference isolate, against a real protein of 740. Eleven of
the seventeen protein-coding loci were affected that way.

This module finds the CDS once per locus, in ALIGNMENT COLUMN space, so the
protein and biophysical modalities can slice every isolate to the same window
before translating:

  * take the reference record (MT_H37Rv) out of the locus FASTA,
  * degap it and locate the GenBank CDS nucleotides inside it (exact match),
  * map that span back to alignment columns, which all isolates share.

Slicing by COLUMN (not by degapped offset) is what keeps isolates comparable:
an isolate with an insertion upstream still has its start codon in the same
column, because that is what an alignment guarantees.

Loci with no CDS in the reference — rrs and rrl are rRNA — return None, and the
caller falls back to translating the whole record. There is no protein there to
get right; the fallback only keeps those blocks existing and unchanged.
"""
import functools
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq

from paths import GENBANK  # H37Rv record the alignments were cut from

REFERENCE_IDS = ("MT_H37Rv", "H37Rv", "NC_000962.3")


@functools.lru_cache(maxsize=4)
def _cds_by_gene(genbank=GENBANK, feature="CDS"):
    """gene name -> nucleotides of its `feature` (CDS, or rRNA), from H37Rv."""
    out = {}
    try:
        rec = next(SeqIO.parse(genbank, "genbank"))
    except Exception:
        return out
    for f in rec.features:
        if f.type != feature:
            continue
        for key in ("gene", "locus_tag"):
            name = f.qualifiers.get(key, [None])[0]
            if name and name not in out:
                out[name] = str(f.extract(rec.seq))
    return out


def _reference_row(fasta):
    """The aligned reference sequence in a locus FASTA, or None."""
    best = None
    for rec in SeqIO.parse(fasta, "fasta"):
        if rec.id in REFERENCE_IDS or "H37Rv" in rec.id:
            return str(rec.seq)
        if best is None:
            best = None          # only the reference anchors the CDS reliably
    return best


@functools.lru_cache(maxsize=256)
def cds_columns(seq_dir, locus, feature="CDS"):
    """(start_col, end_col) of the CDS within `locus`'s alignment, or None.

    End is exclusive. Cached per (directory, locus). ``feature="rRNA"`` locates
    an rRNA gene the same way -- rrs and rrl have no CDS, but they do have a
    real start, and the record carries flank around it (rrs: 100 bp before)."""
    matches = sorted(Path(seq_dir).glob(f"{locus}*.fasta"))
    if not matches:
        return None
    ref = _reference_row(matches[0].as_posix())
    cds = _cds_by_gene(feature=feature).get(locus)
    if not ref or not cds:
        return None

    # degapped reference, plus the column each degapped base came from
    bases, cols = [], []
    for col, ch in enumerate(ref):
        if ch != "-":
            bases.append(ch.upper())
            cols.append(col)
    flat = "".join(bases)
    i = flat.find(cds.upper())
    if i < 0:                                  # try the other orientation
        from Bio.Seq import Seq
        i = flat.find(str(Seq(cds).reverse_complement()).upper())
        if i < 0:
            return None
    return cols[i], cols[i + len(cds) - 1] + 1


def cds_slice(seq_dir, locus, aligned_seq, window=None):
    """`aligned_seq` cut to the locus's CDS columns (unchanged if unknown)."""
    window = window if window is not None else cds_columns(seq_dir, locus)
    if window is None:
        return aligned_seq
    start, end = window
    return aligned_seq[start:end]


# --- placing a block's columns in the genome --------------------------------
# The aligned FASTAs were cut from this same H37Rv record, so a locus's
# reference row degaps to an EXACT genome substring and can be located by
# search rather than by alignment. That is what lets a gene record and a
# promoter window be put on one coordinate system: measured over the 16 loci
# with both, the promoter is anywhere from 30 bp inside the CDS to 732 bp
# upstream of it (inhA's Table-22 row is really the fabG1 promoter), and 10 of
# them OVERLAP the gene record's own flank by 30-130 bp -- so "the promoter
# window ends where the CDS begins" is wrong in both directions, and the
# overlap means one changed base can otherwise produce two tokens.


@functools.lru_cache(maxsize=1)
def _genome(genbank=GENBANK):
    """(forward, reverse-complement) H37Rv genome strings, or (None, None)."""
    try:
        rec = next(SeqIO.parse(genbank, "genbank"))
    except Exception:
        return None, None
    fwd = str(rec.seq).upper()
    return fwd, str(Seq(fwd).reverse_complement())


@functools.lru_cache(maxsize=512)
def genome_span(seq_dir, locus):
    """(start, end, strand) of a locus's aligned reference row in H37Rv, or None.

    Half-open, always in FORWARD genome coordinates. ``'+'`` means the row
    degaps to ``genome[start:end]``; ``'-'`` means it degaps to that span's
    reverse complement, i.e. the record is stored in the gene's own 5'->3'
    direction (katG, pncA, eis, ethA, gid are the '-' loci here).
    """
    matches = sorted(Path(seq_dir).glob(f"{locus}*.fasta"))
    if not matches:
        return None
    ref = _reference_row(matches[0].as_posix())
    fwd, rc = _genome()
    if not ref or not fwd:
        return None
    flat = ref.replace("-", "").upper()
    i = fwd.find(flat)
    if i >= 0:
        return i, i + len(flat), "+"
    j = rc.find(flat)
    if j >= 0:
        return len(fwd) - (j + len(flat)), len(fwd) - j, "-"
    return None


def block_frame(seq_dir, locus, length, translated=False):
    """Where a block's alignment columns sit, for a variant-token tokenizer.

    ``{ref_gap_cols, genome_start, genome_end, strand}``, plus
    ``{cds_start, cds_end, has_cds}`` when ``translated`` -- the gene's window
    in this block's own column space. For a protein-coding gene that is the CDS
    and ``has_cds`` is True. For rrs and rrl it is the rRNA gene and ``has_cds``
    is False: the window still gives the locus its origin (the gene's first
    base), but there is no reading frame, so a tokenizer must not read residues
    there -- whatever datasets/protein.py's whole-record fallback translates.
    The gap columns turn a column into a count of reference bases, and the
    genome span turns that count into a genome position; between them any two
    blocks of the same locus can be placed on one axis.

    Empty when there is no reference row to measure against, which puts
    models.locusfusion_v2 on its fallback path.
    """
    ref = None
    matches = sorted(Path(seq_dir).glob(f"{locus}*.fasta"))
    if matches:
        ref = _reference_row(matches[0].as_posix())
    if not ref:
        return {}
    meta = {"ref_gap_cols": [i for i, ch in enumerate(ref[:length]) if ch == "-"]}
    span = genome_span(seq_dir, locus)
    if span:
        meta.update(genome_start=int(span[0]), genome_end=int(span[1]), strand=span[2])
    if translated:
        window = cds_columns(seq_dir, locus)
        has_cds = window is not None
        if not has_cds:
            window = (cds_columns(seq_dir, locus, feature="rRNA")
                      or (0, min(len(ref), length)))
        meta.update(cds_start=int(window[0]), cds_end=int(window[1]), has_cds=has_cds)
    return meta
