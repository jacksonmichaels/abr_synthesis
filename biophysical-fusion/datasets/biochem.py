"""
Amino-acid biochemistry: property tables, the genetic code, translation, and
the featurizers the protein / biophysical modalities are built on. This is the
single source of truth for anything that turns a nucleotide sequence into
amino-acid–level features (moved here from the old top-level ``biophysical.py``
so all data handling lives under ``datasets/``; that module is now a shim).

PLACEHOLDER property table pending Kulkarni et al. 2026's exact values /
normalization (see TODO.md) — standard published tables used as stand-ins:
  - molecular weight  : average residue mass (Da)
  - isoelectric point : free amino acid pI
  - hydrophobicity    : Eisenberg (1984) normalized consensus scale
Swap the three dicts below for Kulkarni's real table once we have it; the
translation / featurization logic does not change.
"""
import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {a: i for i, a in enumerate(AMINO_ACIDS)}

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

# Standard genetic code (DNA codons -> single-letter AA, '*' = stop).
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
    Nucleotide (gapped, aligned) -> protein string, matching Kulkarni et al.
    2026's amino-acid featurization: translated from the MSA, then treated as
    left-aligned (not gap-preserving vs. the DNA branch). Gaps are stripped
    before translating (so an indel shifts the reading frame downstream, like a
    real frameshift), and translation stops at the first stop codon, truncating
    the protein exactly like a real nonsense mutation would.
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


def one_hot_aa(protein_seq: str, pad_to: int = None) -> np.ndarray:
    """
    (20, K) one-hot amino-acid matrix, K = pad_to (if given) else
    len(protein_seq). Unknown residues / padding are the all-zero column (same
    convention the DNA one-hot uses for unknown bases).
    """
    k = pad_to if pad_to is not None else len(protein_seq)
    out = np.zeros((len(AMINO_ACIDS), k), dtype=np.float32)
    for i, aa in enumerate(protein_seq[:k]):
        j = AA_TO_INDEX.get(aa)
        if j is not None:
            out[j, i] = 1.0
    return out


def upsample_to_nt(bio: np.ndarray) -> np.ndarray:
    """Repeat each AA-position column 3x (AA resolution -> ~nucleotide
    resolution). Used only by EarlyFusionCNN's channel-stack ablation."""
    return np.repeat(bio, 3, axis=-1)
