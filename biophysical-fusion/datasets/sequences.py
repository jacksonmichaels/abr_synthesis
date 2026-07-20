"""
Shared nucleotide-sequence substrate for every sequence-derived modality
(DNA, protein, biophysical all start from the same aligned per-gene FASTAs;
regulatory reads region FASTAs through the same loader).

Everything that touches the BIG-TB reference codebase or the on-disk FASTA /
phenotype layout lives here, so the individual modality files stay small and
only describe *their own* featurization.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from bigtb_ref import tb

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
