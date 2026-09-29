"""
Amino-acid biochemistry: property tables, the genetic code, translation, and
the featurizers the protein / biophysical modalities are built on. This is the
single source of truth for anything that turns a nucleotide sequence into
amino-acid–level features. Not to be confused with ``datasets/biophysical.py``,
which is the traits *modality* built on top of these tables.

PLACEHOLDER property table pending Kulkarni et al. 2026's exact values /
normalization (see TODO.md) — standard published tables used as stand-ins:
  - molecular weight  : average residue mass (Da)
  - isoelectric point : free amino acid pI
  - hydrophobicity    : Eisenberg (1984) normalized consensus scale
Swap the three dicts below for Kulkarni's real table once we have it; the
translation / featurization logic does not change.
"""
import numpy as np
from Bio.Data import CodonTable
from Bio.Seq import Seq

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {a: i for i, a in enumerate(AMINO_ACIDS)}
# The 20 residues plus STOP, for the reference-frame translation below: there a
# stop codon is a residue like any other (a nonsense mutation at codon k), not
# the end of the protein, so it needs a channel of its own.
STOP = "*"
RESIDUES_WITH_STOP = AMINO_ACIDS + STOP
RESIDUE_WITH_STOP_TO_INDEX = {a: i for i, a in enumerate(RESIDUES_WITH_STOP)}

# --- RDKit-derived amino-acid properties (paste into datasets/biochem.py) ---

_MW = {
    "A": 71.079, "C": 103.146, "D": 115.088, "E": 129.115, "F": 147.177,
    "G": 57.052, "H": 137.142, "I": 113.16, "K": 128.175, "L": 113.16,
    "M": 131.2, "N": 114.104, "P": 97.117, "Q": 128.131, "R": 156.189,
    "S": 87.078, "T": 101.105, "V": 99.133, "W": 186.214, "Y": 163.176
} 

_PI = {
    "A": 5.97, "C": 5.32, "D": 2.99, "E": 3.29, "F": 5.97, "G": 5.97,
    "H": 7.8, "I": 5.97, "K": 10.07, "L": 5.97, "M": 5.97, "N": 5.97,
    "P": 5.97, "Q": 5.97, "R": 11.04, "S": 5.97, "T": 5.97, "V": 5.97,
    "W": 5.97, "Y": 5.91
} 

_HYDRO_EISENBERG = {
    "A": -0.582, "C": -0.672, "D": -1.127, "E": -0.737, "F": 0.641,
    "G": -0.97, "H": -0.636, "I": 0.444, "K": -0.473, "L": 0.444, "M": 0.151,
    "N": -1.726, "P": -0.177, "Q": -1.336, "R": -1.338, "S": -1.609,
    "T": -1.221, "V": 0.054, "W": 1.122, "Y": 0.347
}

PROPERTY_NAMES = ("molecular_weight", "isoelectric_point", "hydrophobicity")


def _zscore(table):
    vals = np.array([table[a] for a in AMINO_ACIDS])
    mean, std = vals.mean(), vals.std()
    return {a: (table[a] - mean) / std for a in AMINO_ACIDS}


_MW_Z = _zscore(_MW)
_PI_Z = _zscore(_PI)
_HYDRO_Z = _zscore(_HYDRO_EISENBERG)

# per-AA (mw, pi, hydrophobicity), all z-scored over the 20 canonical AAs
AA_PROPERTY = {
    a: np.array([_MW_Z[a], _PI_Z[a], _HYDRO_Z[a]], dtype=np.float32)
    for a in AMINO_ACIDS
}
N_PROPERTIES = len(PROPERTY_NAMES)

# Standard genetic code (NCBI translation table 1), sourced from Biopython
# rather than hand-typed: forward_table maps each sense codon -> single-letter
# AA, and the three stop codons are added as '*'. Same DNA-codon -> AA mapping
# as before, now from a validated reference.
_STANDARD_CODE = CodonTable.unambiguous_dna_by_id[1]
CODON_TABLE = {
    **_STANDARD_CODE.forward_table,
    **{codon: "*" for codon in _STANDARD_CODE.stop_codons},
}


def translate_codon(codon: str) -> str:
    """Single-letter AA, or '-' for a gap/incomplete/stop/ambiguous codon."""
    codon = codon.upper()
    if len(codon) != 3 or "-" in codon:
        return "-"
    return CODON_TABLE.get(codon, "-")


def translate_seq(aligned_nt_seq: str) -> str:
    """
    Nucleotide (gapped, aligned) -> protein string, matching Kulkarni et al.
    2026's amino-acid featurization: translated from the MSA, then treated as
    left-aligned (not gap-preserving vs. the DNA branch). Gaps are stripped
    before translating (so an indel shifts the reading frame downstream, like a
    real frameshift), and translation stops at the first stop codon, truncating
    the protein exactly like a real nonsense mutation would.

    Translation itself goes through Biopython's ``Seq.translate`` (NCBI standard
    code, ``to_stop=True``) — the validated engine — rather than a hand-rolled
    loop. Any trailing 1-2 nt that don't complete a codon are dropped; ambiguous
    codons (e.g. containing N) become 'X', which the downstream featurizers treat
    as an unknown residue (zero column).
    """
    degapped = aligned_nt_seq.replace("-", "")
    whole_codons = degapped[: len(degapped) - len(degapped) % 3]
    if not whole_codons:
        return ""
    return str(Seq(whole_codons).translate(to_stop=True))


def translate_reference_frame(aligned_nt_seq: str, ref_row: str, window=None) -> str:
    """
    Nucleotide (gapped, aligned) -> one residue per REFERENCE codon.

    ``translate_seq`` degaps first, so its residue k is the isolate's own k-th
    codon: an indel upstream slides every residue after it off the reference
    numbering, and a frameshift turns the rest of the protein into hundreds of
    "substitutions" at positions whose bases never changed. This reads the
    isolate on the H37Rv codon grid instead — codon k is the isolate's bases at
    the three alignment columns holding reference codon k's bases — so residue k
    means the same codon in every isolate, and a residue can only differ from the
    reference where one of its own three columns did.

    Symbols: one of the 20 residues; ``'*'`` for a stop codon (a nonsense
    mutation is a residue here, not the end of the protein); ``'-'`` for a codon
    that lost a base to a deletion (the deletion itself is a DNA gap, carried by
    the nucleotide block); ``'X'`` for an ambiguous codon. Bases the isolate
    carries at columns where the REFERENCE has a gap — insertions — are not part
    of any reference codon and are skipped; the nucleotide block carries them.

    ``window`` is ``(start_col, end_col)`` in alignment columns, the CDS as
    ``datasets.cds.cds_columns`` returns it; None reads the whole record, which
    is what the rRNA loci (no CDS) fall back to. The terminal stop codon of the
    reference is included, so a stop-loss is visible too.
    """
    ref_row = ref_row.upper()
    start, end = window if window is not None else (0, len(ref_row))
    end = min(end, len(ref_row))
    cols = [c for c in range(start, end) if ref_row[c] != "-"]
    seq = aligned_nt_seq.upper()
    out = []
    for k in range(len(cols) // 3):
        codon = "".join(seq[c] if c < len(seq) else "-" for c in cols[3 * k:3 * k + 3])
        if "-" in codon:
            out.append("-")
        else:
            out.append(CODON_TABLE.get(codon, "X"))
    return "".join(out)


def biophysical_matrix(protein_seq: str, pad_to: int = None) -> np.ndarray:
    """
    (3, K) property matrix for one protein sequence, K = pad_to (if given) else
    len(protein_seq). Positions past the sequence end — padding, or where
    translation stopped early on a premature stop — get the zero vector.
    """
    k = pad_to if pad_to is not None else len(protein_seq)
    out = np.zeros((N_PROPERTIES, k), dtype=np.float32)
    for i, aa in enumerate(protein_seq[:k]):
        if aa in AA_PROPERTY:
            out[:, i] = AA_PROPERTY[aa]
    return out


def one_hot_aa(protein_seq: str, pad_to: int = None,
               with_stop: bool = False) -> np.ndarray:
    """
    (20, K) one-hot amino-acid matrix, K = pad_to (if given) else
    len(protein_seq). Unknown residues / padding are the all-zero column (same
    convention the DNA one-hot uses for unknown bases).

    ``with_stop=True`` adds a 21st channel for ``'*'`` (``RESIDUES_WITH_STOP``
    order), for residue strings from ``translate_reference_frame``. The default
    is byte-identical to what every existing run was trained on.
    """
    k = pad_to if pad_to is not None else len(protein_seq)
    alphabet = RESIDUE_WITH_STOP_TO_INDEX if with_stop else AA_TO_INDEX
    out = np.zeros((len(alphabet), k), dtype=np.float32)
    for i, aa in enumerate(protein_seq[:k]):
        j = alphabet.get(aa)
        if j is not None:
            out[j, i] = 1.0
    return out


def residue_channels(with_stop: bool = False):
    """The channel names `one_hot_aa` emits, in order."""
    return list(RESIDUES_WITH_STOP if with_stop else AMINO_ACIDS)


def upsample_to_nt(bio: np.ndarray) -> np.ndarray:
    """Repeat each AA-position column 3x (AA resolution -> ~nucleotide
    resolution). Used only by EarlyFusionCNN's channel-stack ablation."""
    return np.repeat(bio, 3, axis=-1)
