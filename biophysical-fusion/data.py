"""
Phase 2 — dataset construction. Plain functions (not a class-based
modality interface — kept minimal) that build the DNA one-hot array and
the biophysical array for a given drug, reusing BIG-TB's own gene-loading
and one-hot utilities directly rather than re-implementing them.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from bigtb_ref import tb
from biophysical import biophysical_matrix, translate_seq, upsample_to_nt

DRUG_TO_LOCI = tb.DRUG_TO_LOCI


def _load_genotype_df(genotype_dir, loci):
    """Reimplements tb.make_genotype_df's per-locus join, but hands
    tb.sequence_dictionary a forward-slash path explicitly: that function
    extracts the gene name via `filename.split("/")[-1]`, which silently
    breaks on the backslash paths glob.glob/os.path.join produce on
    Windows. Still reuses tb.sequence_dictionary itself, just not the
    Windows-unsafe path plumbing around it."""
    dfs = []
    for gene in loci:
        matches = sorted(Path(genotype_dir).glob(f"{gene}*.fasta"))
        if matches:
            dfs.append(tb.sequence_dictionary(matches[0].as_posix()))
    if not dfs:
        raise ValueError(f"No loci {loci} found in genotype directory {genotype_dir}")
    return dfs[0].join(dfs[1:], how="outer")


def _pad(arr, target_len):
    """Right-pad a (C, L) array with zeros to target_len along axis 1."""
    c, l = arr.shape
    if l >= target_len:
        return arr[:, :target_len]
    out = np.zeros((c, target_len), dtype=arr.dtype)
    out[:, :l] = arr
    return out


def build_dataset(drug, genotype_dir, phenotype_csv):
    """
    Returns:
        dna_X       : (N, 5, L)          one-hot DNA, genes concatenated
                                          into one contiguous locus (matches
                                          Kulkarni et al. 2026's nucleotide
                                          featurization)
        bio_Xs      : list[(N, 3, K_g)]  one biophysical array PER GENE,
                                          each independently padded to that
                                          gene's own max protein length —
                                          individual proteins are separate
                                          channels even when their genes are
                                          concatenated for dna_X (paper's
                                          amino-acid featurization; see
                                          TODO.md #7)
        gene_order  : list[str]          gene name for each entry in bio_Xs
        y           : (N,) float, 0=resistant 1=susceptible -1=missing
                                          (matches tb.rs_encoding_to_numeric)
        isolate_ids : list[str]
    """
    drug = drug.upper()
    loci = DRUG_TO_LOCI[drug]
    df_genos = _load_genotype_df(genotype_dir, loci)

    df_phenos = pd.read_csv(phenotype_csv, index_col="New_ID", dtype=str).fillna("-1")
    isolates = list(df_genos.index.intersection(df_phenos.index))
    df_genos = df_genos.loc[isolates]
    df_phenos = df_phenos.loc[isolates]

    gene_cols = [g for g in loci if g in df_genos.columns]

    dna_rows = []
    protein_seqs = {g: [] for g in gene_cols}
    for iso in isolates:
        full_seq = "".join(df_genos.at[iso, g] for g in gene_cols)
        dna_rows.append(tb.get_one_hot(full_seq).T)  # (L, 5) -> (5, L)
        for g in gene_cols:
            protein_seqs[g].append(translate_seq(df_genos.at[iso, g]))

    l_dna = max(a.shape[1] for a in dna_rows)
    dna_X = np.stack([_pad(a, l_dna) for a in dna_rows]).astype(np.float32)

    bio_Xs = []
    for g in gene_cols:
        k_max = max((len(s) for s in protein_seqs[g]), default=1) or 1
        bio_Xs.append(np.stack([
            biophysical_matrix(s, pad_to=k_max) for s in protein_seqs[g]
        ]).astype(np.float32))

    _, y_array = tb.rs_encoding_to_numeric(df_phenos, [drug])
    y = y_array[:, 0].astype(np.float32)

    return dna_X, bio_Xs, gene_cols, y, isolates


def concat_upsampled(bio_Xs, target_len):
    """Concatenate the per-gene biophysical arrays along the length axis
    and upsample x3, for EarlyFusionCNN's channel-stack ablation. Padded/
    truncated to target_len (== dna_X's length)."""
    combined = np.concatenate(bio_Xs, axis=2)
    up = upsample_to_nt(combined)
    n, c, l = up.shape
    if l >= target_len:
        return up[:, :, :target_len]
    out = np.zeros((n, c, target_len), dtype=up.dtype)
    out[:, :, :l] = up
    return out
