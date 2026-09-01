"""
Shared machinery for the variant-token models, plus the legacy dense token
layout the aggregator family still uses.

Two token layouts live in this project.

**The current one** (``models/locusfusion.py``, and ``datasets/tokens.py`` on
the data side) is discrete: a token is a symbol id, a reference symbol id, a
codon phase and an exact reference-codon coordinate. Nothing is one-hot, nothing
is duplicated, and the coordinate is computed from the CDS annotation and the
reference gap pattern rather than guessed at.

**The legacy one** — ``SLOTS`` / ``C_TOK`` / the ``F_*`` flag offsets below — is
the 42-float fixed-slot vector that ``models/experimental_models.py``'s six
aggregators were measured on (``results/experiments/AGGREGATOR_FAMILY_20260827.md``
and the two ``variant_aggregators`` runs). It is kept here, unchanged, so those
results stay reproducible: porting them to the new layout is a separate piece of
work with its own control run, and silently changing their input would make the
recorded numbers unattributable.

New work should use the discrete layout. ``SLOTS`` carries a known defect that
the new one fixes — its coordinate for a nucleotide token is ``column / 3``,
which is 33-37 codons past where the amino-acid stream puts the same residue
before reference gaps drift it further (see ``datasets/tokens.py`` for the
measurements).
"""
import math

import torch

# --- the LEGACY dense token layout (experimental_models.py only) ------------
SLOTS = {                       # modality -> (start, stop) in the feature vector
    "dna":         (0, 5),      # A C T G -   at one alignment column
    "protein":     (5, 25),     # 20 amino acids at one codon
    "biophysical": (25, 28),    # MW / pI / hydrophobicity of the new residue
    "regulatory":  (28, 33),    # A C T G -   at one promoter column
}
F_IS_NT, F_IS_AA, F_IS_REG, F_IS_WT = 33, 34, 35, 36
F_GAP, F_UNCOVERED = 37, 38
F_PHASE = 39                    # 3 dims: alignment column % 3
C_TOK = 42

GAP_CHANNEL = 4                 # datasets.sequences.NT_CHANNELS == [A, C, T, G, -]

# --- shared by both layouts -------------------------------------------------
# Which blocks share a coordinate system. The unit is CODONS throughout so the
# streams are commensurable: `nt` and `reg` are nucleotide columns mapped onto
# the reference's codon numbering, `aa` is already codon-resolution.
STREAMS = {"dna": "nt", "protein": "aa", "biophysical": "aa", "regulatory": "reg"}


def sinusoid(coord, dims, min_wavelength=1 / 3, max_wavelength=4096.0):
    """(..., ) continuous coordinate -> (..., dims) sinusoidal encoding.

    Continuous rather than table-indexed on purpose: a learned table would need
    one row per column per locus (1,355 x 128 x 19 = 3.3M parameters) to say
    what zero parameters say here.

    The wavelength band is set to the data rather than to the transformer
    default. Coordinates are codon numbers, they run to ~1,355 for the longest
    locus, and the finest step that means anything is 1/3 of a codon. The old
    default band (2*pi to 2*pi*10^4 codons) spent its bottom half on wavelengths
    longer than any locus and never resolved a codon phase at the top; this one
    spans exactly [1/3, 4096] codons, which is why `pos_dims` can be halved
    without losing resolution.
    """
    half = dims // 2
    i = torch.arange(half, device=coord.device, dtype=coord.dtype)
    # geometric sweep from the shortest wavelength to the longest
    ratio = math.log(max_wavelength / min_wavelength)
    wavelengths = min_wavelength * torch.exp(ratio * i / max(half - 1, 1))
    ang = coord.unsqueeze(-1) * (2 * math.pi / wavelengths)
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


def sinusoid_legacy(coord, dims):
    """The original band (2*pi to 2*pi*10^4 codons), kept bit-for-bit.

    ``models/experimental_models.py``'s measured runs used this; changing the
    frequencies underneath them would make ``variant_aggregators_*`` no longer
    reproducible. Only that file should call it.
    """
    half = dims // 2
    freqs = torch.exp(-math.log(10000.0)
                      * torch.arange(half, device=coord.device, dtype=coord.dtype) / half)
    ang = coord.unsqueeze(-1) * freqs
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


def select_variants(occ, k):
    """Occupancy mask (B, L) -> the first `k` occupied columns per isolate.

    Returns ``(idx, valid, n_occ)``: ``idx`` (B, k) long — occupied columns in
    ASCENDING position, padded out with unoccupied ones; ``valid`` (B, k) bool
    marking which of those are real; ``n_occ`` (B,) the true count before the
    cap, which is what the [WT] token reports and what the uncovered-locus test
    reads.

    The ranking trick: score an occupied column ``L+1-p`` and an unoccupied one
    ``-p``. Every occupied score exceeds every unoccupied one, and within each
    group the score decreases with position, so a single ``topk`` yields exactly
    "occupied first, then in positional order".
    """
    B, L = occ.shape
    n_occ = occ.sum(1)
    k = max(1, min(int(k), L))
    ar = torch.arange(L, device=occ.device, dtype=torch.float32)
    score = occ.to(torch.float32) * (L + 1) - ar
    idx = score.topk(k, dim=1).indices                     # (B, k)
    return idx, occ.gather(1, idx), n_occ
