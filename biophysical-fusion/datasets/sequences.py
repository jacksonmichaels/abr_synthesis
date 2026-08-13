"""
Shared nucleotide-sequence substrate for every sequence-derived modality
(DNA, protein, biophysical all start from the same aligned per-gene FASTAs;
regulatory reads region FASTAs through the same loader).

Everything that touches the BIG-TB reference codebase or the on-disk FASTA /
phenotype layout lives here, so the individual modality files stay small and
only describe *their own* featurization.
"""
import functools
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

from bigtb_ref import tb
from .cds import REFERENCE_IDS

# Vectorized equivalent of tb.get_one_hot: same BASE_TO_COLUMN mapping
# (A,C,T,G,- -> 0..4), unknown bases (N, lowercase, etc.) left all-zero.
# tb.get_one_hot is a per-character Python loop; over ~18k isolates x several
# thousand bp that dominates load time, so we do the identical mapping with a
# 256-entry lookup table instead. Semantics are byte-for-byte the same.
BASE_TO_COLUMN = tb.BASE_TO_COLUMN
NT_CHANNELS = list(BASE_TO_COLUMN.keys())          # ['A','C','T','G','-']
_BASE_LUT = np.full(256, -1, dtype=np.int64)
for _b, _col in BASE_TO_COLUMN.items():
    _BASE_LUT[ord(_b)] = _col


def one_hot_nt(seq: str) -> np.ndarray:
    """(L, 5) one-hot, matching tb.get_one_hot exactly but vectorized."""
    codes = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
    cols = _BASE_LUT[codes]
    oh = np.zeros((len(codes), len(BASE_TO_COLUMN)), dtype=np.float32)
    valid = cols >= 0
    oh[np.nonzero(valid)[0], cols[valid]] = 1.0
    return oh


# --- reference-difference ("delta") encoding --------------------------------
# M. tuberculosis is effectively clonal: two isolates differ at a handful of
# positions in a 2.6 kb gene. A plain one-hot therefore spends ~99.9% of its
# columns restating sequence that is identical in every isolate, and the
# measured consequence is severe — in the full_run_v2 setfusion cell the
# per-isolate part of an encoded token is 0.14% of its magnitude (spread 0.00090
# against a norm of 0.6654), so the fusion transformer sees essentially the same
# vector for every sample and its attention collapses to uniform.
#
# Delta encoding keeps the one-hot alphabet and shape but ZEROES every column
# where the isolate matches the H37Rv reference, so what reaches the CNN is the
# variants and nothing else. No information that distinguishes isolates is lost:
# the columns removed are constant across the cohort and therefore carry no
# discriminative signal. The base an isolate actually has at a differing
# position is still one-hot encoded, so "which substitution" survives.
#
# The reference is the real MT_H37Rv record shipped in each alignment (verified
# present in both the genotype and regulatory dirs), NOT a consensus over the
# cohort — a consensus would be fitted on test isolates too.

@functools.lru_cache(maxsize=512)
def reference_row(seq_dir, locus):
    """The aligned H37Rv row for a locus, or None if the FASTA has no reference.

    Same alignment column space as every other record in the file, so a
    position-by-position comparison is well defined.
    """
    matches = sorted(Path(seq_dir).glob(f"{locus}*.fasta"))
    if not matches:
        return None
    for rec in SeqIO.parse(matches[0].as_posix(), "fasta"):
        if rec.id in REFERENCE_IDS or "H37Rv" in rec.id:
            return str(rec.seq).upper()
    return None


def _match_mask(seq, ref):
    """Boolean mask over the overlap: True where `seq` matches `ref`."""
    n = min(len(seq), len(ref))
    if n == 0:
        return np.zeros(0, dtype=bool), 0
    a = np.frombuffer(seq[:n].upper().encode("ascii", "replace"), dtype=np.uint8)
    b = np.frombuffer(ref[:n].encode("ascii", "replace"), dtype=np.uint8)
    return a == b, n


def delta_one_hot_nt(seq, ref):
    """(L, 5) one-hot with reference-matching columns zeroed.

    Positions past the end of the reference are left encoded — they cannot be
    compared, and dropping them would silently discard real sequence.
    """
    oh = one_hot_nt(seq)
    if not ref:
        return oh
    same, n = _match_mask(seq, ref)
    if n:
        oh[:n][same] = 0.0
    return oh


def delta_zero_columns(mat, seq, ref):
    """Zero the columns of a (C, K) feature matrix where `seq` matches `ref`.

    The residue-level counterpart of ``delta_one_hot_nt``, shared by the protein
    (20-channel one-hot) and biophysical (3-channel property) featurizers, both
    of which are indexed by amino-acid position.
    """
    if not ref:
        return mat
    same, n = _match_mask(seq, ref)
    n = min(n, mat.shape[1])
    if n:
        mat[:, :n][:, same[:n]] = 0.0
    return mat


def load_sequence_df(seq_dir, loci):
    """
    Per-locus aligned-sequence table: index = isolate id, one column per locus,
    values = the aligned nucleotide string. Works for both gene loci and
    regulatory-region loci — both follow the same ``{locus}*.fasta`` layout.

    Reuses tb.sequence_dictionary but hands it a forward-slash path (its
    ``filename.split('/')[-1]`` breaks on Windows backslashes) and forces the
    column name to the locus: tb names the column ``filename.split('_')[0]``,
    which yields e.g. ``"rpoB.fasta"`` for the real underscore-less files and
    never matches the locus downstream. Loci with no matching FASTA are skipped;
    returns (df, found_loci).
    """
    seq_dir = Path(seq_dir)
    dfs, found = [], []
    for locus in loci:
        matches = sorted(seq_dir.glob(f"{locus}*.fasta"))
        if not matches:
            continue
        df = tb.sequence_dictionary(matches[0].as_posix())
        df.columns = [locus]
        dfs.append(df)
        found.append(locus)
    if not dfs:
        return pd.DataFrame(), []
    joined = dfs[0].join(dfs[1:], how="outer") if len(dfs) > 1 else dfs[0]
    return joined, found


def load_phenotype(phenotype_csv):
    """Phenotype table indexed on New_ID (the sample-accession join key),
    blanks filled with '-1' so tb.rs_encoding_to_numeric reads them as missing.
    """
    return pd.read_csv(phenotype_csv, index_col="New_ID", dtype=str).fillna("-1")


def numeric_labels(df_phenos, drug):
    """(N,) float labels for one drug: 0=resistant, 1=susceptible, -1=missing
    (via tb.rs_encoding_to_numeric, matching BIG-TB exactly)."""
    _, y_array = tb.rs_encoding_to_numeric(df_phenos, [drug.upper()])
    return y_array[:, 0].astype(np.float32)


def pad_cl(arr, target_len):
    """Right-pad / truncate a (C, L) array to target_len along the length axis."""
    c, l = arr.shape
    if l == target_len:
        return arr
    if l > target_len:
        return arr[:, :target_len]
    out = np.zeros((c, target_len), dtype=arr.dtype)
    out[:, :l] = arr
    return out


def stack_padded(mats):
    """Stack a list of (C, L_i) arrays into (N, C, Lmax), right-padding each to
    the max length. Empty input -> a (0, C, 1) array is avoided by the caller."""
    lmax = max((m.shape[1] for m in mats), default=1) or 1
    return np.stack([pad_cl(m, lmax) for m in mats]).astype(np.float32)
