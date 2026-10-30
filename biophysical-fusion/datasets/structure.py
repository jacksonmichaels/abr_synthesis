"""
3D structure of each coding locus, from AlphaFold DB models of the H37Rv
proteins -- the ``structure`` modality, and the token-token distances
``locusfusion_v2`` reads off it.

``scripts/fetch_af_structures.py`` writes one ``{locus}.npz`` per protein-coding
locus: C-alpha coordinates and pLDDT indexed by H37Rv CODON k, the same index a
v2 token carries (``variant_report(...)["codon"]``). A token's position in space
is therefore a table lookup, and a locus's token-token distance matrix is
``|ca[codon_i] - ca[codon_j]|``.

**The modality carries no per-isolate data.** Geometry is a property of the
locus, not of the isolate: every isolate is placed on H37Rv's fold. So each
block's array is ``(N, 0, n_codons)`` -- an isolate axis so the loader, the CV
split and the batcher treat it like any other block, and zero channels so it
costs nothing to copy -- and the coordinates ride on ``FeatureBlock.meta``, the
way ``ref_residues`` does on the protein block. That also puts them in a run's
config.json (``training.checkpoint.block_records``), so a checkpoint rebuilds
without the structure files.

Masked (``mask`` False, distance 0) wherever either token has no coordinate:
padding slots, promoter / flank / insertion columns (codon -1), every column of
rrs and rrl (no reading frame, codon -1; no block is emitted for them), the
terminal stop codon, and any residue the model does not cover. Two tokens in the
same codon are 0 A apart and UNmasked -- they are one residue.

Only WITHIN-locus distances exist: each model is one chain, so there is no
geometry between genes.
"""
import functools
import warnings
from pathlib import Path

import numpy as np
import torch

from .base import FeatureBlock, Modality


def _default_dir():
    from bigtb_ref import REAL_STRUCTURE_DIR
    return REAL_STRUCTURE_DIR


@functools.lru_cache(maxsize=64)
def load_structure(locus, root=None):
    """``{ca (n_codons,3), plddt (n_codons,), ...}`` for a locus, or None.

    None for a locus with no file -- rRNA, or not fetched -- whose tokens are
    then all masked."""
    path = Path(root or _default_dir()) / f"{locus}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def residue_distances(locus, root=None):
    """(n_codons, n_codons) C-alpha distance matrix in A, NaN where unmapped."""
    s = load_structure(locus, root)
    if s is None:
        return None
    ca = s["ca"]
    return np.linalg.norm(ca[:, None] - ca[None], axis=-1)


# --- block meta <-> arrays ----------------------------------------------------
# JSON has no NaN, so a codon with no coordinate is None in meta.

def structure_meta(s):
    """The ``FeatureBlock.meta`` of one locus's structure block.

    With heavy atoms and PAE in the file (fetched since 2026-10-01) it also
    carries ``heavy`` (per codon, its heavy-atom coordinates, or None) and
    ``pae_bad`` ([i, j] codon pairs inside CONTACT_REACH whose PAE fails the
    gate) -- the sparse form, so config.json stays small. Older files give
    neither, and a model that asks for them refuses to build."""
    ok = ~np.isnan(s["ca"]).any(1)
    meta = {"ca": [[round(float(v), 3) for v in xyz] if good else None
                   for xyz, good in zip(s["ca"], ok)],
            "model_id": str(s["model_id"]), "afdb_version": int(s["afdb_version"])}
    if "heavy" in s:
        meta["heavy"] = [[[round(float(v), 3) for v in xyz] for xyz in atoms
                          if not np.isnan(xyz).any()] if good else None
                         for atoms, good in zip(s["heavy"], ok)]
    if "pae" in s:
        meta["pae_bad"] = pae_bad_pairs(s["ca"], s["pae"])
    return meta


def meta_arrays(meta):
    """(ca (n,3) float32 zero where missing, has (n,) bool)."""
    ca = meta["ca"]
    has = np.array([x is not None for x in ca], dtype=bool)
    xyz = np.array([x if x is not None else (0.0, 0.0, 0.0) for x in ca], dtype=np.float32)
    return xyz.reshape(-1, 3), has


# --- heavy atoms and the contact gate (2026-10-01) ------------------------------
# Two residues can only touch if some heavy atom of one is within CONTACT_WINDOW
# of some heavy atom of the other. No heavy atom sits further than
# SIDECHAIN_REACH from its own C-alpha, so beyond CONTACT_REACH (C-alpha to
# C-alpha) no contact is possible -- that is the distance half of the gate. The
# other half is AlphaFold's own 2D confidence: the pair's predicted aligned
# error, the larger of PAE[i, j] and PAE[j, i], must be <= PAE_MAX.
SIDECHAIN_REACH = 7.41          # A; Arg NH, the furthest over all 17 AFDB v6 models
                                # (then Trp 6.85, Tyr 6.62, Lys 6.52)
CONTACT_WINDOW = 5.0            # A; heavy-atom contact distance
CONTACT_REACH = 2 * SIDECHAIN_REACH + CONTACT_WINDOW     # 19.82 A
PAE_MAX = 10.0                  # A


def pae_bad_pairs(ca, pae):
    """[i, j] (i <= j) codon pairs within CONTACT_REACH whose PAE > PAE_MAX.

    Pairs beyond the reach fail the gate on distance alone, so they are not
    listed; the model checks the reach itself from the C-alpha."""
    D = np.linalg.norm(ca[:, None] - ca[None], axis=-1)
    P = np.fmax(pae, pae.T)
    i, j = np.nonzero(np.triu((D <= CONTACT_REACH) & ~(P <= PAE_MAX)))
    return [[int(a), int(b)] for a, b in zip(i, j) if not np.isnan(D[a, b])]


def meta_heavy(meta):
    """(atoms (n, A, 3) float32 zero-padded, atom_ok (n, A) bool), A = the most
    heavy atoms any residue has (14 for Trp)."""
    heavy = meta["heavy"]
    A = max((len(a) for a in heavy if a), default=1)
    atoms = np.zeros((len(heavy), A, 3), np.float32)
    atom_ok = np.zeros((len(heavy), A), bool)
    for k, a in enumerate(heavy):
        if a:
            atoms[k, :len(a)] = a
            atom_ok[k, :len(a)] = True
    return atoms, atom_ok


def meta_pae_bad(meta, n):
    """(n, n) bool, True where the pair fails the PAE half of the gate."""
    bad = np.zeros((n, n), bool)
    for i, j in meta["pae_bad"]:
        bad[i, j] = bad[j, i] = True
    return bad


def heavy_min_distances(codon, located, atoms, atom_ok):
    """(B, K, K) closest heavy-atom distance in A between the tokens' residues,
    0 where either is not ``located`` (``pair_distances``'s mask still decides
    what is real). The same residue is 0 A."""
    n, A, _ = atoms.shape
    B, K = codon.shape
    cc = codon.clamp(0, n - 1)
    xyz = atoms[cc].reshape(B, K * A, 3)
    ok = (atom_ok[cc] & located.unsqueeze(-1)).reshape(B, K * A)
    d = torch.cdist(xyz, xyz)
    d = d.masked_fill(~(ok.unsqueeze(2) & ok.unsqueeze(1)), float("inf"))
    d = d.reshape(B, K, A, K, A).amin(dim=(2, 4))
    return torch.where(torch.isfinite(d), d, torch.zeros_like(d))


# --- token distances ------------------------------------------------------------

def pair_distances(codon, valid, ca, has):
    """-> (dist, mask, located), (B, K, K), (B, K, K), (B, K).

    ``codon`` (B, K) long, -1 for a column with no codon; ``valid`` (B, K) bool;
    ``ca`` (n_codons, 3) and ``has`` (n_codons,) as from ``meta_arrays``. The one
    implementation both the model and the validation notebook use.
    """
    n = ca.shape[0]
    cc = codon.clamp(0, n - 1)
    located = valid & (codon >= 0) & (codon < n) & has[cc]
    xyz = ca[cc] * located.unsqueeze(-1).to(ca.dtype)
    mask = located.unsqueeze(2) & located.unsqueeze(1)
    dist = torch.cdist(xyz, xyz) * mask.to(ca.dtype)
    return dist, mask, located


def token_distances(codon, valid, locus, root=None):
    """(dist, mask), each (B, K, K), for one locus's K variant tokens, read from
    the structure files."""
    B, K = codon.shape
    s = load_structure(locus, root)
    if s is None:
        return torch.zeros(B, K, K), torch.zeros(B, K, K, dtype=torch.bool)
    ca, has = meta_arrays(structure_meta(s))
    dist, mask, _ = pair_distances(codon, valid, torch.as_tensor(ca), torch.as_tensor(has))
    return dist, mask


def report_distances(report, max_variants=16, root=None):
    """(dist, mask) of shape (B, n_loci, max_variants, max_variants), from
    ``LocusFusionV2Net.variant_report``. Slot j is variant token j+1 in the
    model (slot 0 there is [NULL], which has no position and is not included).
    """
    B = report[0]["codon"].shape[0]
    D = torch.zeros(B, len(report), max_variants, max_variants)
    M = torch.zeros(B, len(report), max_variants, max_variants, dtype=torch.bool)
    for li, r in enumerate(report):
        k = r["codon"].shape[1]
        d, m = token_distances(r["codon"], r["valid"], r["locus"], root)
        D[:, li, :k, :k], M[:, li, :k, :k] = d, m
    return D, M


# --- the modality ---------------------------------------------------------------

class StructureModality(Modality):
    """One ``structure:{locus}`` block per protein-coding locus with a model.

    Zero channels: see the module docstring. A locus is skipped (warned) when
    its structure is not on the codon grid the tokenizer uses -- the file's
    codon count must equal the H37Rv CDS's -- so a stale or mismatched file
    cannot silently shift every coordinate."""
    name = "structure"
    uses_genes = True

    def __init__(self, root=None):
        self.root = root

    def build(self, ctx):
        from .cds import _cds_by_gene
        if ctx.gene_seqs is None:
            return []
        cds = _cds_by_gene()
        blocks, skipped = [], []
        for locus in ctx.gene_seqs.columns:
            s = load_structure(locus, self.root)
            if s is None:
                continue                                  # rRNA / not fetched
            n_codons = s["ca"].shape[0]
            if locus not in cds or len(cds[locus]) // 3 != n_codons:
                skipped.append(locus)
                continue
            blocks.append(FeatureBlock(
                name=f"structure:{locus}", modality="structure",
                array=np.zeros((len(ctx.isolates), 0, n_codons), dtype=np.float32),
                note=f"structure:{locus}: {s['model_id']} C-alpha on {n_codons} "
                     "H37Rv codons (no per-isolate channels)",
                meta=structure_meta(s)))
        if skipped:
            warnings.warn(f"structure: {skipped} skipped -- the structure file's "
                          "codon count does not match the H37Rv CDS. Re-run "
                          "scripts/fetch_af_structures.py.", stacklevel=2)
        return blocks


# --- candidate token encodings (for comparison; the model uses closeness) -------
# Each maps a slot-aligned (…, K, K) distance matrix in A and its mask to the
# channel blocks a token would carry: row i of each block goes to variant token i.
# `locus` is only read by the protein-relative encodings. Candidates under
# discussion, not a settled choice -- see notebooks/tokenization_v2.ipynb §6.

@functools.lru_cache(maxsize=64)
def _locus_pair_distances(locus, root=None):
    """Sorted C-alpha distances over every residue pair of a locus (A)."""
    D = residue_distances(locus, root)
    d = D[np.triu_indices_from(D, 1)]
    return torch.as_tensor(np.sort(d[~np.isnan(d)]), dtype=torch.float32)


def _percentile_closeness(d, m, locus):
    """1 - (share of this protein's residue pairs closer than d): 0.5 is the
    protein's own median distance, so every protein uses the full 0-1 range."""
    ref = _locus_pair_distances(locus)
    return (1 - torch.searchsorted(ref, d.contiguous()).float() / len(ref)) * m


def _linear_closeness(d, m, locus):
    """1 - d / (this protein's largest C-alpha distance)."""
    return (1 - d / _locus_pair_distances(locus)[-1]).clamp(min=0) * m


ENCODINGS = {   # name -> (channels, encode(dist, mask, locus) -> [channel blocks])
    "distance + mask":         (32, lambda d, m, l: [d / 10 * m, m.float()]),
    "distance only":           (16, lambda d, m, l: [d / 10 * m]),
    "closeness exp(-d/10)":    (16, lambda d, m, l: [torch.exp(-d / 10) * m]),
    "closeness exp(-d/30)":    (16, lambda d, m, l: [torch.exp(-d / 30) * m]),
    "closeness 1/(1+d/10)":    (16, lambda d, m, l: [1 / (1 + d / 10) * m]),
    "closeness linear 1-d/dmax": (16, lambda d, m, l: [_linear_closeness(d, m, l)]),
    "closeness percentile":    (16, lambda d, m, l: [_percentile_closeness(d, m, l)]),
    "contact d<8 Å":           (16, lambda d, m, l: [((d < 8) & m).float()]),
}
CURRENT_ENCODING = "closeness exp(-d/10)"   # what LocusFusionV2Net computes
