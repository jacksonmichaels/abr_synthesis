"""
Variant tokens: one symbol id per alignment column, plus the static per-column
metadata a model needs to place that symbol in the genome.

This replaces the dense 42-float feature vector `models/locusfusion.py` used to
build per token. A variant is a *discrete* event — "katG codon 315 is now
Threonine where the reference has Serine" — and the smallest faithful encoding
of that is three small integers and one coordinate, not a 42-wide sparse float
vector with duplicated flags. Everything the old layout carried is either kept
here or was provably redundant:

===========================  ==========================================
old feature slot             where it went
===========================  ==========================================
``dna`` (5) / ``regulatory``  the symbol id: the two never co-occur on one
  (5) one-hot                 token (they are different coordinate streams),
                              so they shared nothing but width
``protein`` (20) one-hot      the symbol id
``biophysical`` (3)           kept, as the only continuous field — see below
``F_IS_NT/AA/REG``            implied by which range the symbol id falls in
``F_IS_WT``                   symbol id ``WT``
``F_GAP``                     a literal copy of the one-hot's ``-`` channel;
                              that is symbol id ``NT_GAP``
``F_PHASE`` (3)               ``phase``, and now computed correctly (below)
``F_UNCOVERED``               a per-LOCUS constant broadcast onto every token;
                              it belongs on the [WT] token's statistics
===========================  ==========================================

and one field is new: the **reference symbol** at the same column. A variant is
a substitution, so it has two ends; the old layout carried only the alt. Given
``(locus, column)`` the reference is a constant the model could in principle
memorize, but there is no reason to make it: it is free to look up.

The coordinate
--------------

The old code placed a DNA token at ``column / 3 - offset[locus]`` with a LEARNED
per-locus ``offset``, and a protein token at its codon index. Both are wrong,
for reasons that are arithmetic rather than arguable:

* The aligned FASTAs are not bare CDS. The coding sequence starts at column
  100-112 in 10 of the 17 protein-coding loci, so ``column / 3`` is 33-37 codons
  past where the protein stream puts the same residue before anything else
  happens.
* The reference row carries gaps INSIDE the CDS window (pncA 304, gid 177,
  ethA 167, rpoB 96, katG 53), so the true map is not linear in the column
  index at all and no single scalar can express it.

Measured against the canonical resistance codons, the DNA token for katG S315
landed at coordinate 357.3 while its own protein token landed at 314; rpoB S450
at 501.7 against 449; pncA S65 at 126.7 against 64. And the learned scalar that
was supposed to absorb this did not: read off a trained ISONIAZID checkpoint
(``newmodels_full/sd_all_modalities__locusfusion``, fold 3) it held
``[-0.0107, +0.0081]`` for (inhA, katG) against a true katG offset of 37 — it
initialises at zero and enters the model through a sinusoid whose top wavelength
is 6.3 codons, so its gradient is oscillatory and it never moves.

So compute it instead. For a coding locus, walk the reference row and count
DEGAPPED reference bases from the CDS start; ``coord = n / 3`` and
``phase = n % 3``, negative upstream. That is the H37Rv codon number — the same
coordinate the WHO catalogue names a mutation by — it costs zero parameters, and
it puts a DNA token and a protein token for the same residue on the same axis.

The coordinate is **0-based** — codon *k* is ``coord == k``, matching residue
*k* of the protein block — while the WHO catalogue numbers residues from 1. So
katG S315T sits at ``coord == 314.33`` (the .33 is the codon phase: the second
base of the codon). Anything reporting to a human should add the 1.

The remaining approximation is unchanged and is stated where it lives: protein
codon *k* is the k-th codon of the ISOLATE's degapped CDS, so it equals
reference codon *k* only when no indel sits upstream. That holds for the
overwhelming majority of a clonal cohort and cannot be fixed here; it is a
property of how ``datasets/protein.py`` translates.
"""
import functools

import numpy as np

from . import cds
from .biochem import AMINO_ACIDS
from .sequences import reference_row

# --- the symbol vocabulary --------------------------------------------------
# One flat vocabulary shared by the reference slot and the alt slot, so "the
# reference has Serine" and "the isolate has Serine" index the same row of two
# different tables. The stream (nucleotide / amino acid / promoter) is implied
# by which range an id falls in, which is why no stream flag is needed.
PAD = 0                 # padding, and every column past a block's real extent
WT = 1                  # the [WT] sentinel token, one per locus

NT_BASES = "ACTG-"      # matches datasets.sequences.NT_CHANNELS
NT0 = 2                 # 2..6   nucleotide A C T G -
NT_UNK = NT0 + len(NT_BASES)                    # 7   N, or any unresolved base
REG0 = NT_UNK + 1       # 8..12  promoter-window A C T G -
REG_UNK = REG0 + len(NT_BASES)                  # 13
AA0 = REG_UNK + 1       # 14..33 amino acids, in AMINO_ACIDS order
AA_UNK = AA0 + len(AMINO_ACIDS)                 # 34  stop, X, or past the end
N_SYMBOLS = AA_UNK + 1                          # 35

NT_GAP = NT0 + NT_BASES.index("-")
REG_GAP = REG0 + NT_BASES.index("-")

# phase is the codon position a nucleotide column sits at, 0/1/2. Amino-acid
# tokens are already codon-resolution, so they get their own value rather than a
# meaningless 0.
PHASE_NA = 3
N_PHASES = 4

# The coordinate streams, and which modalities share one. A stream is a set of
# blocks that are co-indexed, so their occupancies union and their features land
# on one token.
STREAMS = {"dna": "nt", "protein": "aa", "biophysical": "aa", "regulatory": "reg"}

_NT_LUT = np.full(256, NT_UNK, dtype=np.int8)
_REG_LUT = np.full(256, REG_UNK, dtype=np.int8)
for _i, _b in enumerate(NT_BASES):
    _NT_LUT[ord(_b)] = NT0 + _i
    _NT_LUT[ord(_b.lower())] = NT0 + _i
    _REG_LUT[ord(_b)] = REG0 + _i
    _REG_LUT[ord(_b.lower())] = REG0 + _i

_AA_INDEX = {a: AA0 + i for i, a in enumerate(AMINO_ACIDS)}


def symbol_names():
    """id -> human-readable name, for reports and tests."""
    names = ["<pad>", "[WT]"]
    names += [f"nt {b}" for b in NT_BASES] + ["nt ?"]
    names += [f"reg {b}" for b in NT_BASES] + ["reg ?"]
    names += [f"aa {a}" for a in AMINO_ACIDS] + ["aa ?"]
    return names


def nt_symbol_ids(seq, stream="nt", pad_to=None):
    """Aligned nucleotide string -> (L,) int8 symbol ids.

    Unresolved bases (N and friends) become ``NT_UNK`` rather than the all-zero
    column ``one_hot_nt`` gives them. That distinction is the whole point: under
    the old encoding an N where the reference has a base was indistinguishable
    from a match, so a failed base call read as wild type.
    """
    lut = _NT_LUT if stream == "nt" else _REG_LUT
    codes = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
    ids = lut[codes]
    if pad_to is not None:
        out = np.full(pad_to, PAD, dtype=np.int8)
        n = min(pad_to, ids.size)
        out[:n] = ids[:n]
        return out
    return ids


def aa_symbol_ids(protein, pad_to=None):
    """Translated protein string -> (K,) int8 symbol ids.

    Residues past the end of this isolate's protein are ``AA_UNK``, not
    ``PAD``: a translation that stopped early is a real deviation from the
    reference (a premature stop), and padding it to look like absent data is
    how truncations became invisible.
    """
    k = pad_to if pad_to is not None else len(protein)
    out = np.full(k, AA_UNK, dtype=np.int8)
    for i, aa in enumerate(protein[:k]):
        out[i] = _AA_INDEX.get(aa, AA_UNK)
    return out


# --- static per-column metadata ---------------------------------------------

def _degapped_offsets(ref, start):
    """Signed count of non-gap reference bases from column `start`.

    ``n[c]`` is the number of degapped reference bases in ``[start, c)`` for
    ``c >= start`` and ``-|[c, start)|`` for ``c < start``. A column where the
    reference itself is a gap does not advance the count, so an insertion
    relative to H37Rv shares the coordinate of the reference base that follows
    it — which is what an insertion *is*. Those columns are still tellable
    apart, because their reference symbol is the gap id.
    """
    solid = np.fromiter((ch != "-" for ch in ref), dtype=bool, count=len(ref))
    cum = np.concatenate([[0], np.cumsum(solid)])      # cum[c] = solid in [0, c)
    return (cum[:-1] - cum[start]).astype(np.int32)


def _meta(coord, phase, ref_ids):
    return {"coord": coord.astype(np.float32),
            "phase": phase.astype(np.int8),
            "ref_id": ref_ids.astype(np.int8)}


@functools.lru_cache(maxsize=256)
def _gene_column_meta(seq_dir, locus):
    ref = reference_row(seq_dir, locus)
    if not ref:
        return None
    window = cds.cds_columns(seq_dir, locus)
    # rRNA loci (rrs, rrl) have no CDS. Count from the record start instead, so
    # the coordinate is still an exact degapped reference position and still in
    # the same units; there is simply no codon for it to be a codon number of.
    start = window[0] if window else 0
    n = _degapped_offsets(ref, start)
    return _meta(n / 3.0, np.mod(n, 3), nt_symbol_ids(ref, "nt"))


@functools.lru_cache(maxsize=256)
def _region_column_meta(seq_dir, region):
    """Promoter windows have no CDS, so the coordinate is degapped reference
    distance UPSTREAM of the window's 3' end — negative, in codon units to stay
    commensurable with the coding streams. The window carries a 30 bp flank
    whose orientation is not recorded here, so the zero point is the end of the
    extracted window rather than the transcription start; that is a per-region
    constant, and unlike the nt/aa case no other stream has to agree with it.
    """
    ref = reference_row(seq_dir, region)
    if not ref:
        return None
    n = _degapped_offsets(ref, len(ref))
    return _meta(n / 3.0, np.mod(n, 3), nt_symbol_ids(ref, "reg"))


def gene_column_meta(seq_dir, locus, length):
    """Per-column ``{coord, phase, ref_id}`` for a gene locus, cut to `length`."""
    return _cut(_gene_column_meta(str(seq_dir), locus), length)


def region_column_meta(seq_dir, region, length):
    """Per-column ``{coord, phase, ref_id}`` for a promoter window."""
    return _cut(_region_column_meta(str(seq_dir), region), length)


def protein_column_meta(ref_protein, length):
    """Per-residue metadata for the amino-acid stream.

    Residue *k* is already a codon coordinate, so ``coord = k`` with no map and
    no phase. Past the end of the reference protein the reference symbol is
    ``AA_UNK`` — those residues exist in some isolate but not in H37Rv.
    """
    k = np.arange(length, dtype=np.float32)
    return _meta(k, np.full(length, PHASE_NA),
                 aa_symbol_ids(ref_protein or "", pad_to=length))


def _cut(meta, length):
    """Trim or zero-extend metadata to a block's actual padded length."""
    if meta is None:
        return None
    have = meta["coord"].shape[0]
    if have == length:
        return meta
    if have > length:
        return {k: v[:length] for k, v in meta.items()}
    pad = length - have
    return {"coord": np.concatenate([meta["coord"], np.zeros(pad, np.float32)]),
            "phase": np.concatenate([meta["phase"], np.full(pad, PHASE_NA, np.int8)]),
            "ref_id": np.concatenate([meta["ref_id"], np.full(pad, PAD, np.int8)])}
