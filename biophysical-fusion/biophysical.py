"""
Phase 1 — amino-acid biophysical property table + nucleotide-to-property
alignment.

PLACEHOLDER table pending Kulkarni et al. 2026's exact values/normalization
(see TODO.md #5.3-4) — standard published tables used as stand-ins:
  - molecular weight   : average residue mass (Da)
  - isoelectric point   : free amino acid pI
  - hydrophobicity      : Eisenberg (1984) normalized consensus scale
    (this is the one channel the literature notes do pin down explicitly)
Swap the three dicts below for Kulkarni's real table once we have it; the
translation/alignment logic does not change.
"""
import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

_MW = {
    "A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 103.1388,
    "Q": 128.1307, "E": 129.1155, "G": 57.0519, "H": 137.1411, "I": 113.1594,
    "L": 113.1594, "K": 128.1741, "M": 131.1926, "F": 147.1766, "P": 97.1167,
    "S": 87.0782, "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326,
}

_PI = {
    "A": 6.00, "R": 10.76, "N": 5.41, "D": 2.77, "C": 5.07,
    "Q": 5.65, "E": 3.22, "G": 5.97, "H": 7.59, "I": 6.02,
    "L": 5.98, "K": 9.74, "M": 5.74, "F": 5.48, "P": 6.30,
    "S": 5.68, "T": 5.60, "W": 5.89, "Y": 5.66, "V": 5.96,
}

_HYDRO_EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}


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
N_PROPERTIES = 3

# Standard genetic code (DNA codons -> single-letter AA, '*' = stop),
# compact table construction (same layout Biopython/NCBI table 1 uses).
_bases = "TCAG"
_codons = [b1 + b2 + b3 for b1 in _bases for b2 in _bases for b3 in _bases]
_aas = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE = dict(zip(_codons, _aas))


def translate_codon(codon: str) -> str:
    """Single-letter AA, or '-' for a gap/incomplete/stop/ambiguous codon."""
    codon = codon.upper()
    if len(codon) != 3 or "-" in codon:
        return "-"
    return CODON_TABLE.get(codon, "-")


def translate_seq(aligned_nt_seq: str) -> str:
    """
    Nucleotide (gapped, aligned) -> protein sequence, matching Kulkarni et
    al. 2026's amino-acid featurization (Methods, "Featurizing amino acid
    sequences"): translated from the MSA, then treated as left-aligned —
    *not* kept in gap-preserving positional correspondence with the DNA
    one-hot branch (that revises an earlier draft of decision #5). Gaps are
    stripped before translating (so an indel naturally shifts the reading
    frame downstream, the way a real frameshift would), and translation
    stops at the first stop codon, truncating the protein exactly like a
    real nonsense mutation would.
    """
    degapped = aligned_nt_seq.replace("-", "")
    aa = []
    for i in range(0, len(degapped) - 2, 3):
        a = translate_codon(degapped[i:i + 3])
        if a in ("*", "-"):
            break
        aa.append(a)
    return "".join(aa)


def biophysical_matrix(protein_seq: str, pad_to: int = None) -> np.ndarray:
    """
    (3, K) property matrix for one protein sequence, K = pad_to (if given)
    else len(protein_seq). Positions past the sequence end — padding, or
    where translation stopped early on a premature stop — get the zero
    vector (decision #5: same "encode as gap" convention as the DNA
    branch's own gap channel).
    """
    k = pad_to if pad_to is not None else len(protein_seq)
    out = np.zeros((N_PROPERTIES, k), dtype=np.float32)
    for i, aa in enumerate(protein_seq[:k]):
        if aa in AA_PROPERTY:
            out[:, i] = AA_PROPERTY[aa]
    return out


def upsample_to_nt(bio: np.ndarray) -> np.ndarray:
    """Repeat each AA-position column 3x. Used only by EarlyFusionCNN (our
    own ablation, not literally from the paper) to get a biophysical array
    to nucleotide-length resolution for channel-stacking; since the
    biophysical branch is no longer codon-position-aligned to the DNA
    branch (see translate_seq), this is a rough approximation, not a true
    per-position mapping."""
    return np.repeat(bio, 3, axis=-1)
