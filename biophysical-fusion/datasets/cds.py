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

# read-only reference used to build the alignments in the first place
GENBANK = ("/project/pi_annagreen_umass_edu/saishradha/project_data_curation/"
           "make_msa/genbank_reference_data/genome.gbff")
REFERENCE_IDS = ("MT_H37Rv", "H37Rv", "NC_000962.3")


@functools.lru_cache(maxsize=1)
def _cds_by_gene(genbank=GENBANK):
    """gene name -> CDS nucleotides, from the H37Rv GenBank record."""
    out = {}
    try:
        rec = next(SeqIO.parse(genbank, "genbank"))
    except Exception:
        return out
    for f in rec.features:
        if f.type != "CDS":
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
def cds_columns(seq_dir, locus):
    """(start_col, end_col) of the CDS within `locus`'s alignment, or None.

    End is exclusive. Cached per (directory, locus)."""
    matches = sorted(Path(seq_dir).glob(f"{locus}*.fasta"))
    if not matches:
        return None
    ref = _reference_row(matches[0].as_posix())
    cds = _cds_by_gene().get(locus)
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
