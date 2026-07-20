"""
Legacy dataset adapter.

The modality-aware loader now lives in the ``datasets/`` package
(``datasets.load_dataset``). This module is a thin backward-compatibility shim
that keeps ``build_dataset(...)`` / ``concat_upsampled(...)`` — the signatures
``eval_dna_cnn.py`` and ``train.py`` were written against — working, but
implemented entirely on top of ``datasets`` so there is a single source of
truth. New code should call ``datasets.load_dataset`` directly.
"""
import numpy as np

from datasets import DRUG_TO_LOCI, load_dataset  # noqa: F401 (re-exported)
from datasets.biochem import upsample_to_nt


def build_dataset(drug, genotype_dir, phenotype_csv, compute_bio=True):
    """
    DNA one-hot (+ optional per-gene biophysical) arrays for one drug.

    Returns (dna_X, bio_Xs, gene_order, y, isolate_ids), matching the original
    contract: dna_X (N,5,L); bio_Xs list[(N,3,K_g)] per gene (empty if
    compute_bio=False); y (N,) with 0=R/1=S/-1=missing.
    """
    mods = ["dna", "biophysical"] if compute_bio else ["dna"]
    data = load_dataset(drug, mods, genotype_dir, phenotype_csv, verbose=False)
    dna_X = next(b.array for b in data.blocks if b.modality == "dna")
    bio_Xs = [b.array for b in data.blocks if b.modality == "biophysical"]
    return dna_X, bio_Xs, data.gene_order, data.y, data.isolate_ids


def concat_upsampled(bio_Xs, target_len):
    """Concatenate per-gene biophysical arrays along length, upsample x3 to
    ~nucleotide resolution, pad/truncate to target_len — EarlyFusionCNN's
    channel-stack ablation input."""
    combined = np.concatenate(bio_Xs, axis=2)
    up = upsample_to_nt(combined)
    n, c, l = up.shape
    if l >= target_len:
        return up[:, :, :target_len]
    out = np.zeros((n, c, target_len), dtype=up.dtype)
    out[:, :, :l] = up
    return out
