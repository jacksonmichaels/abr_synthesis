"""
The H37Rv reference sequence and its codon grid, read from the GenBank record.

    from genomefusion.reference import load_reference
    ref = load_reference()          # uppercase bytes, length 4,411,532
"""
from pathlib import Path
import re

import numpy as np

from genomefusion.partition import CACHE, GENBANK

COMPLEMENT = bytes.maketrans(b"ACGTNacgtn", b"TGCANtgcan")


def read_origin(path=GENBANK):
    """The ORIGIN block of a GenBank record as uppercase bytes."""
    txt = Path(path).read_text()
    i = txt.rindex("\nORIGIN")
    body = txt[txt.index("\n", i + 1):]
    seq = re.sub(r"[^A-Za-z]", "", body.split("//")[0])
    return seq.upper().encode()


def load_reference(cache=CACHE, genbank=GENBANK):
    path = Path(cache) / "reference.npy"
    if path.exists():
        return np.load(path).tobytes()
    seq = read_origin(genbank)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.frombuffer(seq, np.uint8))
    return seq


def codon_positions(locus, partition):
    """(n_codons, 3) int64 of genome positions per codon of a CDS locus.

    Row k is reference codon k's three bases in translation order, so a minus-
    strand gene's positions run downwards. Positions the locus does not own
    (an overlap given to the neighbour) are -1.
    """
    sl = slice(locus.start, locus.end)
    pos = np.arange(locus.start, locus.end, dtype=np.int64)
    cod = partition.codon_of[sl]
    ph = partition.phase_of[sl]
    own = (partition.locus_of[sl] == locus.index) & (cod >= 0)
    n = int(cod[own].max()) + 1 if own.any() else 0
    out = np.full((n, 3), -1, np.int64)
    out[cod[own], ph[own]] = pos[own]
    return out
