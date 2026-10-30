"""
Variant-token transformer, second tokenizer (``--arch locusfusion_v2``).

The architecture is ``locusfusion``'s, unchanged: fuse a locus's tokens into one
locus representation, then let the loci talk to each other, then one learned
query per drug. Read ``models/locusfusion.py`` for why that shape. What changes
here is the tokenizer, and only the tokenizer.

**One token per VARIANT POSITION, not per (variant, stream).** v1 tokenized each
modality's coordinate stream separately: a locus offered up to 16 nucleotide
tokens, 16 residue tokens and 16 promoter tokens, and the `aa` tokens were
deliberately NOT paired with the `nt` tokens because the protein block is indexed
by codon of the isolate's own degapped CDS while the DNA block is in alignment-
column space, so the two could only be related through a LEARNED per-locus offset
— good enough to place a token on a shared axis, useless for a gather.

v2 takes the offset from the data instead. ``datasets.cds.block_frame`` places
every nucleotide block in the H37Rv genome and gives a coding record its CDS
window; it rides along on ``FeatureBlock.meta``, so a DNA column has an exact
codon index and the residue it codes for can simply be read off the protein
block. That makes the fusion positional rather than attentional:

* **DNA is the only source of tokens.** Occupancy comes from the nucleotide
  blocks alone (``dna`` in the CDS record, ``regulatory`` in the promoter
  window). Protein and biophysical never create a token — they are looked up at
  the codon the DNA column falls in and ride along as extra channels. A variant
  is a base that changed; what it did to the translation is a property of it.
* **16 tokens per LOCUS, not per locus per stream.** The cap is now over the
  whole gene, and the promoter window and the coding record share one position
  axis whose origin is the start codon. Both are placed by their real H37Rv
  coordinates (``datasets.cds.block_frame``), not by assuming the promoter abuts
  the CDS: it sits anywhere from overlapping the record to 732 bp upstream of
  it, and where the two overlap the promoter's copy of those columns is
  dropped, so within a locus a base is one token however many of its blocks
  carry it. Across loci that is deliberately NOT done: neighbouring genes share
  4-130 bp of record and promoter sequence (fabG1 x the inhA window, ethA x
  ethR, embA x embB/embC, gyrA x gyrB, rpoB x rpoC), and a base there is a token
  in EACH locus that carries it, so every gene's summary is built with its own
  regulatory context — c-15t reads as upstream of fabG1 in fabG1's locus and of
  inhA in inhA's. The cost is that stage 2 sees that base twice.
* **A token is one base pair, in every representation it has.** Its slots hold
  the new base, the residue that codon now translates to (for a synonymous
  substitution, the unchanged H37Rv residue, read from the reference because the
  delta-encoded protein block is zero there), that residue's biophysical
  properties, the change in those properties from H37Rv's residue, and the codon
  phase. A
  base with no codon — promoter, flank, an insertion column, or anywhere in the
  rRNA genes *rrs* and *rrl*, which have a start but no reading frame — carries
  its base and position and nothing else.

The flag channels are gone with the streams that needed them. ``is_nt`` /
``is_aa`` / ``is_reg`` said which stream a token came from; a token is no longer
from a stream, and coding vs promoter is already the difference between the
``dna`` and ``regulatory`` slots being occupied. ``is_gap`` was a copy of the
``-`` channel of the nucleotide one-hot, which is still there. The per-token
``uncovered`` flag was a locus-level fact broadcast onto every token; the
``[NULL]`` token already carries the occupied FRACTION it was thresholded from,
so the information survives without the threshold or its knob.

**The ``[NULL]`` token.** Every locus has one learned token that is always
present, whatever the isolate carries. It plays two roles. It is what a locus
looks like when nothing in it differs from H37Rv — such a locus presents
``[NULL]`` and nothing else, so the model scores deviations from the reference
rather than re-deriving the reference from thousands of constant columns. And it
is the locus's summary slot, the way BERT's ``[CLS]`` is: stage 1 reads the
locus representation back off it. It says nothing biological about the
reference; H37Rv is the coordinate system the deviations are measured in, not a
claim about what a susceptible strain looks like. (v1 called this token
``[WT]``; the rename is terminology only — the mechanism is the same.)

**Structure (optional, ``--modalities ... structure``).** A token can also say
where it sits on the H37Rv fold: its row of the locus's slot-aligned C-alpha
closeness matrix over the (up to 16) variant tokens -- channel j of token i is
exp(-d / STRUCT_LENGTH) for the distance d from token i to token j, 1 for the
same residue and 0 where either token has no residue -- from an AlphaFold DB
model (``datasets/structure.py``). The encoding is tentative. Like
protein and biophysical it is read at the token's codon and never makes a token;
unlike them it has no per-isolate channels -- every isolate is placed on the
reference fold -- so its block is zero channels wide and the coordinates ride on
the block's meta. Within a locus only: each model is one chain. The channels are
APPENDED after the fixed C_TOK (see ``c_tok``), so a model without structure is
unchanged.

What stays from v1: ``--delta`` input, per-locus blocks, that per-locus token
and its count/coverage statistics, the keyed summary norm, ``carry_variants``,
the sinusoidal coordinate, the per-locus FiLM adapter, the drug-query read-out.

**Residues on the reference codon grid.** The protein and biophysical blocks
this model reads are translated codon by codon on H37Rv's grid
(``load_dataset(protein_frame="reference")``, ``biochem.translate_reference_frame``):
residue k is the isolate's bases at the three alignment columns holding reference
codon k. The default isolate-frame translation degaps first, so an indel slid
every residue after it off the reference numbering and a frameshift became
hundreds of phantom substitutions at unchanged bases. On the reference grid a
residue can only change where one of its own columns did. Measured over all
17,942 ISONIAZID isolates at katG: 80 carried a changed residue no DNA token could
reach in the isolate frame (85 counting the 16-token cap), and 2 do on the
reference grid — both over the cap, neither a frame error
(``notebooks/tokenization_v2.ipynb`` §5). An indel itself is its gap or insertion token; a codon that lost
a base carries no residue. And STOP is a residue — the 21st protein
channel — so a nonsense mutation (20 katG isolates) no longer looks identical to
a synonymous change on its token.

NOTE ON THE NAME: the 2026-09-01 run of the ORIGINAL architecture on the fixed
tokenizer used to be called ``locusfusion_v2``; it was renamed to
``results/experiments/locusfusion_fixed/`` so that this name is free for the
model. A run write-up from before 2026-09-17 may still use the old name.
"""
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .locusfusion import _select_variants
from .net import NO_LOCUS, KeyedTokenNorm, parse_block_key
from datasets.biochem import AA_PROPERTY, N_PROPERTIES, RESIDUES_WITH_STOP
from datasets.structure import (CONTACT_REACH, heavy_min_distances, meta_arrays,
                                meta_heavy, meta_pae_bad, pair_distances)

# --- the token feature vector ----------------------------------------------
# One fixed layout whatever subset of modalities a run loads: a modality that is
# absent simply leaves its slot zero, which is what makes two runs' attention
# maps comparable. Every channel is a named quantity a human can read off — a
# base, a residue, a property — which is the point of this model.
#
# A token says what the position reads NOW and how that differs from H37Rv:
# `biophysical` is the isolate's residue's properties and `D_BIO` the change
# (isolate - H37Rv), so "S -> N, hydrophobicity -1.4" is one subtraction the
# model does not have to learn.
SLOTS = {                       # modality -> (start, stop) in the feature vector
    "dna":         (0, 5),      # A C T G -   the new base, at one CDS column
    "protein":     (5, 26),     # 20 residues + STOP: what that reference codon
                                #   reads in this isolate
    "biophysical": (26, 29),    # MW / pI / hydrophobicity of that residue
    "regulatory":  (29, 34),    # A C T G -   the new base, at one promoter column
}
D_BIO = (34, 37)                # those three properties, isolate MINUS H37Rv
F_IS_NULL = 37                  # the per-locus [NULL] token (see the module docstring)
F_PHASE = 38                    # 3 dims: position within the codon (0/1/2)
C_TOK = 41



def token_layout(unknown_symbols=False, ref_bio=False):
    """The token feature layout for one model. Both options are opt-in
    (2026-10-01, the ablation of the 2026-09-24 48-channel draft); with both off
    this is exactly the module constants above (C_TOK = 41).

      unknown_symbols  dna / regulatory slots gain an N channel (6), protein gains
                       '-' deleted and 'X' unreadable (23) -- the blocks must be
                       loaded with load_dataset(unknown_symbols=True), and an
                       uncalled base then makes a token
      ref_bio          the H37Rv residue's three properties get their own slot,
                       right before D_BIO (both on = the draft's 48 channels)
    """
    nt = 6 if unknown_symbols else 5
    pr = 23 if unknown_symbols else 21
    o, slots = 0, {}
    for name, width in (("dna", nt), ("protein", pr), ("biophysical", 3),
                        ("regulatory", nt)):
        slots[name] = (o, o + width)
        o += width
    ref = None
    if ref_bio:
        ref = (o, o + 3)
        o += 3
    d_bio = (o, o + 3)
    o += 3
    f_null = o
    f_phase = o + 1
    o += 4
    order = list(RESIDUES_WITH_STOP) + (["-", "X"] if unknown_symbols else [])
    props = np.zeros((pr, N_PROPERTIES), dtype=np.float32)
    for i, r in enumerate(order):
        if r in AA_PROPERTY:
            props[i] = AA_PROPERTY[r]
    return {"slots": slots, "ref_bio": ref, "d_bio": d_bio, "f_is_null": f_null,
            "f_phase": f_phase, "c_tok": o, "nt_channels": nt, "protein_channels": pr,
            "unknown_symbols": bool(unknown_symbols), "residue_properties": props}


# --- the structure channels, appended when a `structure` block is loaded -----
# The locus's C-alpha geometry over its own variant tokens, SLOT-ALIGNED: token i
# carries row i of the (V x V) matrix, V = max_variants, so channel C_TOK + j of
# token i is how close it is to token j (entry (3, 5) is token 3 -> token 5, and
# token 5's channel 3 holds the same number).
#
# PROXIMITY, not distance: exp(-d / STRUCT_LENGTH). The same residue reads 1.0,
# adjacent residues (3.8 A) 0.68, and it decays to 0 across the fold -- so 0 is
# both "very far" and "no geometry here", and the mask channel the distance form
# needed (0 A is a REAL distance: two changed bases in one codon) is not needed.
# One row of V channels per token, never two.
#
# Zero for everything without a residue: promoter, flank, insertion and rRNA
# tokens, the stop codon, padding slots and [NULL] -- and for every other token's
# entry pointing at them.
#
# Appended, not slotted into the fixed layout above, so a model built without
# structure keeps C_TOK = 41.
STRUCT_LENGTH = 10.0            # A; the decay length of the proximity channels
#
# Three opt-in knobs (2026-10-01); the defaults are the behaviour above.
#   struct_dist     'ca' (C-alpha) | 'heavy' (closest heavy atoms of the two
#                   residues, FeatureBlock.meta['heavy'])
#   struct_gate     'none' | 'contact': a pair is real only if its C-alpha are
#                   within CONTACT_REACH (2 x the longest side chain + a 5 A
#                   heavy-atom window) AND AlphaFold's PAE for it is <= PAE_MAX
#                   (datasets/structure.py)
#   struct_invalid  the value a non-real entry reads. 0 by default, which is
#                   also "very far"; -1 keeps "no geometry / failed the gate"
#                   apart from both "far" (-> 0) and "same residue" (1).
STRUCT_DISTS = ("ca", "heavy")
STRUCT_GATES = ("none", "contact")


def c_tok(max_variants, structure):
    """Token feature width: C_TOK, plus max_variants with structure."""
    return C_TOK + (int(max_variants) if structure else 0)


# The protein block this model reads is on the H37Rv codon grid
# (load_dataset(protein_frame="reference")): 21 channels, one column per
# reference codon. The isolate-frame block every other arch reads is 20 channels
# and left-aligned, and reading it here would misplace every residue after an
# indel — so construction checks both the channel count and the width.
PROTEIN_CHANNELS = SLOTS["protein"][1] - SLOTS["protein"][0]
NT_CHANNELS = SLOTS["dna"][1] - SLOTS["dna"][0]
GAP_CHANNEL = 4                 # datasets.sequences.NT_CHANNELS == [A, C, T, G, -]

# residue -> its three z-scored properties, on the protein block's channel order.
# STOP has none, so a nonsense mutation reads as "lost the reference residue's
# properties" on the delta channels.
_RES_ORDER = list(RESIDUES_WITH_STOP)
RESIDUE_PROPERTIES = np.zeros((PROTEIN_CHANNELS, N_PROPERTIES), dtype=np.float32)
for _i, _r in enumerate(_RES_ORDER):
    if _r in AA_PROPERTY:
        RESIDUE_PROPERTIES[_i] = AA_PROPERTY[_r]

# Occupancy is read off these, in this order: the promoter window sits upstream
# of the coding record, so concatenating them in this order makes the merged
# index axis run 5' -> 3'.
NT_MODALITIES = ("regulatory", "dna")
# ...and these are looked up at the codon a DNA token falls in. They never
# create a token of their own.
AA_MODALITIES = ("protein", "biophysical")
# ...and so is this one, but it has no per-isolate channels at all: its block is
# (N, 0, n_codons) and the fold rides on FeatureBlock.meta (datasets/structure.py).
STRUCTURE_MODALITIES = ("structure",)

LOCUSFUSION_V2_DEFAULTS = {
    # A  token width, shared by both stages, both embeddings and the readout
    "d_model": 128, "nhead": 4, "dropout": 0.1,
    # B  stage 1 — within one locus, over its own variant tokens
    "enc_layers": 2, "enc_dim_ff": 256,
    # C  stage 2 — across loci, over the locus summaries
    "fusion_layers": 2, "fusion_dim_ff": 256,
    # D  the tokenizer. `max_variants` is now per LOCUS, not per (locus, stream):
    #    every modality of a gene shares these 16 slots because they describe the
    #    same 16 base pairs. The census in README.md puts the 99th percentile at
    #    <=7 changed columns per locus for 17 of 19 loci and 26 for the two rRNA
    #    genes. Overflow keeps the FIRST 16 in positional order (deterministic;
    #    one-hot columns have no norm to rank by).
    "max_variants": 16, "pos_dims": 32,
    # E  per-locus specialization: 'shared' | 'adapter' (FiLM) | 'per_locus'
    "locus_encoder": "adapter",
    # F  standardise each locus summary across the batch, per locus, before
    #    stage 2 sees it. See LOCUSFUSION_DEFAULTS for the measurement.
    "summary_norm": "keyed",
    # G  how many of a locus's own variant tokens are handed up to stage 2
    #    alongside its summary. 0 = summaries only.
    "carry_variants": 0,
    # H  the decay length (A) of the structure closeness channels,
    #    exp(-d / struct_length). Read only when a `structure` block is loaded.
    #    A config that does not record it rebuilds at this default.
    "struct_length": STRUCT_LENGTH,
    #    ...which distance, which gate, and what a non-real entry reads (see
    #    STRUCT_LENGTH). The defaults reproduce the 2026-09-24 encoding.
    "struct_dist": "ca", "struct_gate": "none", "struct_invalid": 0.0,
    # I  opt-in tokenizer ablations (token_layout); both off = C_TOK 41
    "unknown_symbols": False, "ref_bio": False,
}

def _sinusoid(coord, dims, min_wavelength=1 / 3, max_wavelength=4096.0):
    """(..., ) coordinate in CODONS -> (..., dims) sinusoidal encoding.

    The band is set to the data, not to the transformer default. Coordinates run
    to ~1,355 codons for the longest locus and the finest step that means
    anything is 1/3 of a codon (a DNA column inside its codon), so the band spans
    exactly that. v1's default (2*pi to 2*pi*10^4 codons) spent its bottom half
    on wavelengths longer than any locus and never resolved a codon phase at the
    top, which is why it needed twice the dimensions to say less.
    """
    half = dims // 2
    i = torch.arange(half, device=coord.device, dtype=coord.dtype)
    ratio = math.log(max_wavelength / min_wavelength)
    wavelengths = min_wavelength * torch.exp(ratio * i / max(half - 1, 1))
    ang = coord.unsqueeze(-1) * (2 * math.pi / wavelengths)
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


LOCUS_ENCODERS = ("shared", "adapter", "per_locus")
SUMMARY_NORMS = ("none", "keyed")


class LocusFusionV2Net(nn.Module):
    """Variant-token transformer, one token per changed base pair.

    Pipeline, per isolate::

        per locus L:
            [ regulatory block | dna block ]   one merged column axis, 5' -> 3',
                                               each column placed by genome position
                --occupancy (nucleotides only, --> the columns that differ
                  minus the promoter's copy
                  of what the record covers)
                --gather-->                      <= max_variants tokens
        token = tok_proj(new base | new residue | its biophysics | codon phase)
              + pos_proj(sinusoid(genome distance from the start codon, in codons))
              + locus_emb[L]  (+ FiLM adapter[L])
        [NULL]_L = null_emb[L] + null_proj(variant count, occupied fraction)

        stage 1:  TransformerEncoder over {[NULL]_L, variants of L} -> z_L = out[[NULL]]
        stage 2:  TransformerEncoder over {z_L for every locus}     -> fused
        readout:  one learned query per drug, cross-attending the fused set

    Same ``forward(xs)`` contract as every other arch — a list of (B, C_i, L_i)
    block tensors in ``branch_specs`` order. ``block_frames`` is one dict per
    block, in the same order — ``{ref_gap_cols, genome_start, genome_end,
    strand}`` plus ``{cds_start, cds_end}`` for a coding record, built by
    ``datasets.cds.block_frame`` and carried on the block's ``meta``, which
    ``from_blocks`` reads for you. Without the genome span a locus falls back to
    "the promoter window abuts the start codon", which is wrong by up to 732 bp
    and cannot detect the overlap, so the fallback warns.
    """

    bio_input = "blocks"        # forward takes the list of block tensors as-is

    def __init__(self, block_keys, branch_specs, n_drugs=1, drug_names=None,
                 d_model=128, nhead=4, dropout=0.1,
                 enc_layers=2, enc_dim_ff=256,
                 fusion_layers=2, fusion_dim_ff=256,
                 max_variants=16, pos_dims=32,
                 locus_encoder="adapter", summary_norm="keyed", carry_variants=0,
                 struct_length=STRUCT_LENGTH,
                 struct_dist="ca", struct_gate="none", struct_invalid=0.0,
                 unknown_symbols=False, ref_bio=False,
                 block_frames=None,
                 hidden=256, head_dropout=0.0, per_drug_hidden=0, out_bias=None):
        super().__init__()
        if not branch_specs:
            raise ValueError("LocusFusionV2Net needs at least one branch spec")
        if len(block_keys) != len(branch_specs):
            raise ValueError("block_keys must match branch_specs length "
                             f"({len(block_keys)} vs {len(branch_specs)})")
        if d_model % nhead:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        if locus_encoder not in LOCUS_ENCODERS:
            raise ValueError(f"unknown locus_encoder {locus_encoder!r}; "
                             f"choose from {list(LOCUS_ENCODERS)}")
        if summary_norm not in SUMMARY_NORMS:
            raise ValueError(f"unknown summary_norm {summary_norm!r}; "
                             f"choose from {list(SUMMARY_NORMS)}")
        if not struct_length > 0:
            raise ValueError(f"struct_length must be > 0 A, got {struct_length}")
        if struct_dist not in STRUCT_DISTS:
            raise ValueError(f"unknown struct_dist {struct_dist!r}; choose from {list(STRUCT_DISTS)}")
        if struct_gate not in STRUCT_GATES:
            raise ValueError(f"unknown struct_gate {struct_gate!r}; choose from {list(STRUCT_GATES)}")
        if pos_dims % 2:
            raise ValueError(f"pos_dims must be even, got {pos_dims}")
        if drug_names:
            n_drugs = len(drug_names)

        self.layout = token_layout(unknown_symbols, ref_bio)
        self.unknown_symbols, self.ref_bio = bool(unknown_symbols), bool(ref_bio)
        block_keys = [(m, l or NO_LOCUS) for m, l in block_keys]
        known = set(SLOTS) | set(STRUCTURE_MODALITIES)
        unknown = sorted({m for m, _ in block_keys} - known)
        if unknown:
            raise ValueError(f"LocusFusionV2Net has no feature slot for modalities "
                             f"{unknown}; known: {sorted(known)}")
        block_frames = list(block_frames or [{}] * len(block_keys))
        if len(block_frames) != len(block_keys):
            raise ValueError("block_frames must match block_keys length "
                             f"({len(block_frames)} vs {len(block_keys)})")

        # --- group the flat block list into loci ----------------------------
        self.loci = list(dict.fromkeys(l for _, l in block_keys))
        if self.loci == [NO_LOCUS]:
            warnings.warn(
                "LocusFusionV2Net: every block keyed to <none> — you passed the "
                "merged per-modality blocks, so there is exactly one 'locus' and "
                "the whole two-stage structure collapses. Load with "
                "per_modality_branch=False for per-locus blocks.", stacklevel=2)
        plan = []
        for locus in self.loci:
            mine = [(i, m, int(length))
                    for i, ((m, l), (_c, length)) in enumerate(zip(block_keys, branch_specs))
                    if l == locus]
            nt, offset = [], 0
            for want in NT_MODALITIES:          # promoter first: it is upstream
                for i, m, length in mine:
                    if m == want:
                        nt.append({"index": i, "modality": m, "length": length,
                                   "channels": int(branch_specs[i][0]),
                                   "offset": offset, "frame": block_frames[i] or {}})
                        offset += length
            if not nt:
                raise ValueError(
                    f"{locus}: locusfusion_v2 tokenizes NUCLEOTIDE positions, and "
                    f"this locus has none — its blocks are "
                    f"{sorted(m for _, m, _ in mine)}. Load 'dna' (and/or "
                    f"'regulatory'); protein and biophysical are read at the "
                    f"codon a DNA variant falls in, and cannot stand alone.")
            plan.append({
                "locus": locus,
                "nt": nt,
                "aa": [{"index": i, "modality": m, "length": length,
                        "channels": int(branch_specs[i][0])}
                       for i, m, length in mine if m in AA_MODALITIES],
                "structure": next(({"index": i, "length": length,
                                   "frame": block_frames[i] or {}}
                                  for i, m, length in mine
                                  if m in STRUCTURE_MODALITIES), None),
                "total": offset,
                "reg_len": sum(b["length"] for b in nt if b["modality"] == "regulatory"),
            })
        self._plan = plan
        # one lookup per locus, over the MERGED [promoter | coding] column axis:
        # where that column sits in codons from the start codon (`coord`), which
        # codon of the protein block it belongs to (`codon`, -1 outside the
        # translated window) and where inside that codon it falls (`phase`, -1
        # likewise). Precomputed because it is fixed per locus and integer —
        # v1 could only approximate it with a learned per-locus offset.
        self.frames, unplaced, unref, has_ref = {}, [], [], []
        for li, p in enumerate(plan):
            coord, codon, phase, keep, note = self._frame_maps(p)
            # H37Rv's residue at every codon of this locus, as an index into the
            # protein block's channel order. It is what the D_BIO delta is taken
            # against, and what fills `protein` where the isolate's codon did NOT
            # change (a delta-encoded block is zero there, which would otherwise
            # read as "no residue" rather than "the same residue").
            ref_res = self._ref_residues(p, block_frames)
            if p["aa"] and ref_res is None:
                unref.append(p["locus"])
            self.register_buffer(
                f"ref_res_{li}",
                torch.as_tensor(ref_res if ref_res is not None
                                else np.full(1, -1, dtype=np.int64)),
                persistent=False)
            has_ref.append(ref_res is not None)
            # the fold, per codon: C-alpha (zero where missing) and whether it
            # exists. A locus with no structure block gets a one-row table that
            # locates nothing.
            sb = p["structure"]
            if sb is not None and sb["frame"].get("ca") is not None:
                ca, has_ca = meta_arrays(sb["frame"])
                if len(ca) != sb["length"]:
                    raise ValueError(
                        f"{p['locus']}: structure block is {sb['length']} codons wide "
                        f"but its meta carries {len(ca)} coordinates")
            elif sb is not None:
                raise ValueError(
                    f"{p['locus']}: structure block has no coordinates on its meta "
                    "(FeatureBlock.meta['ca'], attached by datasets.structure). "
                    "A config written without them cannot rebuild the fold.")
            else:
                ca = np.zeros((1, 3), np.float32)
                has_ca = np.zeros(1, bool)
            self.register_buffer(f"struct_ca_{li}", torch.as_tensor(ca), persistent=False)
            self.register_buffer(f"struct_has_{li}", torch.as_tensor(has_ca), persistent=False)
            # heavy atoms and the PAE half of the gate, only when asked for
            for need, key in ((struct_dist == "heavy", "heavy"),
                              (struct_gate == "contact", "pae_bad")):
                if need and sb is not None and sb["frame"].get(key) is None:
                    raise ValueError(
                        f"{p['locus']}: struct_dist={struct_dist!r} / struct_gate="
                        f"{struct_gate!r} needs FeatureBlock.meta[{key!r}], which this "
                        "structure block does not carry. Re-run "
                        "scripts/fetch_af_structures.py (heavy atoms + PAE).")
            if struct_dist == "heavy":
                atoms, atom_ok = (meta_heavy(sb["frame"]) if sb is not None
                                  else (np.zeros((1, 1, 3), np.float32), np.zeros((1, 1), bool)))
                self.register_buffer(f"struct_atoms_{li}", torch.as_tensor(atoms), persistent=False)
                self.register_buffer(f"struct_atom_ok_{li}", torch.as_tensor(atom_ok), persistent=False)
            if struct_gate == "contact":
                bad = meta_pae_bad(sb["frame"], len(ca)) if sb is not None else np.zeros((1, 1), bool)
                self.register_buffer(f"struct_pae_bad_{li}", torch.as_tensor(bad), persistent=False)
            self.register_buffer(f"coord_map_{li}", torch.as_tensor(coord), persistent=False)
            self.register_buffer(f"codon_map_{li}", torch.as_tensor(codon), persistent=False)
            self.register_buffer(f"phase_map_{li}", torch.as_tensor(phase), persistent=False)
            self.register_buffer(f"keep_map_{li}", torch.as_tensor(keep), persistent=False)
            p["kept"] = int(keep.sum())
            self.frames[p["locus"]] = note
            if not note["genome_anchored"]:
                unplaced.append(p["locus"])
            sb = p["structure"]
            if sb is not None and not (note["genome_anchored"] and note["has_cds"]
                                       and sb["length"] == note["n_codons"]):
                # the same check the protein block gets: coordinates indexed by
                # anything but the tokenizer's codon would put every token on
                # the wrong atom, silently
                raise ValueError(
                    f"{p['locus']}: structure block is {sb['length']} codons wide, "
                    f"but this locus's coding record "
                    + (f"has {note['n_codons']} reference codons"
                       if note["genome_anchored"] and note["has_cds"]
                       else "has no genome-anchored CDS")
                    + " — the fold is not on the tokenizer's H37Rv codon grid.")
            for b in p["nt"]:
                if b["channels"] != self.layout["nt_channels"]:
                    raise ValueError(
                        f"{p['locus']}: {b['modality']} block has {b['channels']} channels, "
                        f"this model reads {self.layout['nt_channels']} -- "
                        + ("A C T G - N: load with unknown_symbols=True."
                           if self.unknown_symbols else
                           "A C T G -: this block was loaded with unknown_symbols=True; "
                           "build the model with unknown_symbols=True too."))
            for b in p["aa"]:
                want = self.layout["protein_channels"]
                if b["modality"] == "protein" and b["channels"] != want:
                    raise ValueError(
                        f"{p['locus']}: protein block has {b['channels']} channels, this "
                        f"model reads {want} (20 residues + STOP"
                        + (" + deleted + unreadable" if self.unknown_symbols else "")
                        + ") on the H37Rv codon grid. Load with protein_frame='reference' "
                        "(the runners imply it for --arch locusfusion_v2)"
                        + (" and unknown_symbols=True." if self.unknown_symbols else "."))
                # a frame-anchored coding record says how many reference codons
                # there are; an isolate-frame block is as wide as its longest
                # translation instead, which is the silent failure worth catching
                if (note["genome_anchored"] and note["has_cds"]
                        and b["length"] != note["n_codons"]):
                    raise ValueError(
                        f"{p['locus']}: {b['modality']} block is {b['length']} residues "
                        f"wide but the reference CDS has {note['n_codons']} codons — "
                        "it is not on the H37Rv codon grid. Load with "
                        "protein_frame='reference'.")
        if unref:
            warnings.warn(
                f"LocusFusionV2Net: no reference residues for {unref} — those loci "
                "lose the property-delta (D_BIO) channels, and a "
                "synonymous or property-neutral codon reads as an empty residue "
                "slot. They ride on FeatureBlock.meta['ref_residues'], which the "
                "protein and biophysical modalities attach on the reference "
                "codon grid.", stacklevel=2)
        if unplaced:
            warnings.warn(
                f"LocusFusionV2Net: no genome span for {unplaced} — those loci fall "
                "back to 'the promoter window abuts the start codon', which is "
                "wrong by up to 732 bp on the real data and cannot drop the "
                "columns a gene record and its promoter window share. Load with "
                "datasets.cds.block_frame metadata (any current loader does).",
                stacklevel=2)
        self.max_variants = int(max_variants)
        self.tokens_per_locus = 1 + self.max_variants
        self.carry_variants = min(int(carry_variants), self.max_variants)
        self.struct_length = float(struct_length)
        self.struct_dist, self.struct_gate = struct_dist, struct_gate
        self.struct_invalid = float(struct_invalid)

        self._has_ref = has_ref
        self.use_structure = any(p["structure"] is not None for p in plan)
        self.c_tok = (self.layout["c_tok"]
                      + (self.max_variants if self.use_structure else 0))
        self.f_struct = self.layout["c_tok"]                    # proximity row
        self.register_buffer("residue_properties",
                             torch.as_tensor(self.layout["residue_properties"]),
                             persistent=False)

        # --- token embedding -------------------------------------------------
        n_loci = len(self.loci)
        self.tok_proj = nn.Linear(self.c_tok, d_model)
        self.pos_proj = nn.Linear(pos_dims, d_model)
        self.null_proj = nn.Linear(2, d_model)       # log1p(count) / occupied frac
        self.locus_emb = nn.Embedding(n_loci, d_model)
        self.null_emb = nn.Embedding(n_loci, d_model)
        self.tok_norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.locus_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.null_emb.weight, std=0.02)
        self.pos_dims = int(pos_dims)

        # --- stage 1: within one locus ---------------------------------------
        self.locus_encoder = locus_encoder
        if locus_encoder == "adapter":
            self.film_scale = nn.Parameter(torch.zeros(n_loci, d_model))
            self.film_shift = nn.Parameter(torch.zeros(n_loci, d_model))
        else:
            self.film_scale = self.film_shift = None
        n_enc = n_loci if locus_encoder == "per_locus" else 1
        self.encoders = nn.ModuleList(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, nhead, enc_dim_ff, dropout=dropout,
                                           batch_first=True, norm_first=True),
                enc_layers, enable_nested_tensor=False)
            for _ in range(n_enc))

        # --- stage 2: across loci --------------------------------------------
        self.summary_norm = summary_norm
        self.summ_norm = (KeyedTokenNorm(n_loci, d_model)
                          if summary_norm == "keyed" else None)
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, fusion_dim_ff, dropout=dropout,
                                       batch_first=True, norm_first=True),
            fusion_layers, enable_nested_tensor=False)

        # --- readout: one learned query per drug -----------------------------
        self.drug_queries = nn.Parameter(torch.zeros(n_drugs, d_model))
        nn.init.trunc_normal_(self.drug_queries, std=0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.head_drop = nn.Dropout(float(head_dropout)) if head_dropout else nn.Identity()
        self.per_drug_hidden = int(per_drug_hidden) if n_drugs > 1 else 0
        if self.per_drug_hidden:
            k = self.per_drug_hidden
            self.drug_hidden = nn.ModuleList(nn.Linear(hidden, k) for _ in range(n_drugs))
            self.drug_out = nn.ModuleList(nn.Linear(k, 1) for _ in range(n_drugs))
            self.fc_out = None
        else:
            self.drug_hidden = self.drug_out = None
            self.fc_out = nn.Linear(hidden, 1)
        if out_bias is not None:
            with torch.no_grad():
                for layer in (self.drug_out or [self.fc_out]):
                    layer.bias.fill_(float(out_bias))

        self.block_keys = block_keys
        self.drug_names = list(drug_names) if drug_names else None
        self.n_drugs = n_drugs
        self.d_model = d_model
        self._occupancy_checked = False

    @staticmethod
    def _frame_maps(p):
        """(coord, codon, phase, keep, note) over one locus's merged column axis.

        Every column is placed by its GENOME position. A block's reference row
        degaps to an exact genome substring (``datasets.cds.block_frame``), so
        column -> count of reference bases -> genome position is exact, and the
        gene record and its promoter window land on one axis measured from the
        start codon rather than assumed to abut it. Three things follow:

        * a promoter variant sits at its true distance — inhA's window is 732 bp
          upstream of the inhA CDS, being the fabG1 promoter, not adjacent to it;
        * a column the record already covers is dropped from the promoter block
          (``keep``), so the 30-130 bp the two share on 10 of 16 loci produce one
          token, not two. That includes the 30 bp of CDS most WHO windows reach
          into: the record covers them, so the record's token carries their base
          and residue, and the kept promoter columns are all upstream of it;
        * the kept axis is monotonic in coordinate, 5' -> 3'.

        Codon and phase count reference bases from the start codon, which is the
        index `datasets/protein.py` translates on. Without a genome span the
        fallback is v1's convention: the promoter counts backwards from the end
        of its own window and nothing is deduplicated.
        """
        total, reg_len = p["total"], p["reg_len"]
        coord = np.zeros(total, dtype=np.float32)
        codon = np.full(total, -1, dtype=np.int64)
        phase = np.full(total, -1, dtype=np.int64)
        keep = np.ones(total, dtype=bool)
        note = {"genome_anchored": False, "cds_start": None, "has_cds": None,
                "n_codons": None, "duplicate_columns": 0, "promoter_offset": None}

        def bases_before(b):
            """(n, is_base): reference bases strictly before each column."""
            L = b["length"]
            is_base = np.ones(L, dtype=np.int64)
            gaps = np.asarray(b["frame"].get("ref_gap_cols", ()), dtype=np.int64)
            if gaps.size:
                is_base[gaps[gaps < L]] = 0
            return np.concatenate([[0], np.cumsum(is_base)])[:L], is_base

        def positions(b):
            """(genome position per column, is_base), or (None, is_base)."""
            n, is_base = bases_before(b)
            f = b["frame"]
            if f.get("genome_start") is None:
                return None, is_base
            return (f["genome_start"] + n if f["strand"] == "+"
                    else f["genome_end"] - 1 - n), is_base

        rec = next((b for b in p["nt"] if b["modality"] == "dna"), None)
        prom = next((b for b in p["nt"] if b["modality"] == "regulatory"), None)

        # --- the origin: the start codon of this locus's coding record -------
        rec_pos = rec_is_base = None
        cds_len = 0
        if rec is not None:
            rec_pos, rec_is_base = positions(rec)
            n, _ = bases_before(rec)
            f = rec["frame"]
            start = int(f.get("cds_start", 0))
            end = int(f.get("cds_end", rec["length"]))
            # an rRNA gene (rrs, rrl) has a start -- the origin -- but no reading
            # frame: its tokens carry a base and a position and nothing else,
            # whatever the protein block's whole-record fallback translated.
            # Frames written without the key predate it and were all coding.
            has_cds = bool(f.get("has_cds", True))
            n_full = np.concatenate([n, [n[-1] + rec_is_base[-1]]])  # bases before col, 0..L
            cds_len = int(n_full[min(end, rec["length"])] - n_full[min(start, rec["length"])]) \
                if has_cds else 0
            note["cds_start"] = start
            note["has_cds"] = has_cds
            # reference codons in the translated window, the terminal stop
            # included — the width of a reference-frame protein block
            n_win = int(n_full[min(end, rec["length"])] - n_full[min(start, rec["length"])])
            note["n_codons"] = n_win // 3
            rel = n - n[min(start, rec["length"] - 1)]          # bases from the start codon
            lo, L = rec["offset"], rec["length"]
            coord[lo:lo + L] = rel / 3.0
            inside = ((np.arange(L) >= start) & (np.arange(L) < end) & (rec_is_base > 0)
                      & has_cds)
            codon[lo:lo + L] = np.where(inside, rel // 3, -1)
            phase[lo:lo + L] = np.where(inside, rel % 3, -1)
            if rec_pos is not None:
                note["genome_anchored"] = True
                g0 = rec_pos[min(start, L - 1)]
                strand = rec["frame"]["strand"]

        # --- the promoter, placed against that origin ------------------------
        if prom is not None:
            lo, L = prom["offset"], prom["length"]
            prom_pos, prom_is_base = positions(prom)
            anchored = (rec_pos is not None and prom_pos is not None
                        and prom["frame"]["strand"] == rec["frame"]["strand"])
            if anchored:
                off = (prom_pos - g0) if strand == "+" else (g0 - prom_pos)
                coord[lo:lo + L] = off / 3.0
                # the WHO window's 3' flank reaches into the CDS on most loci;
                # those columns are coding and are read as such
                inside = (off >= 0) & (off < cds_len) & (prom_is_base > 0)
                codon[lo:lo + L] = np.where(inside, off // 3, -1)
                phase[lo:lo + L] = np.where(inside, off % 3, -1)
                # one base, one token: drop what the record already covers. Gap
                # columns go too -- a gap sits at the position of the base after
                # it, so one inside the shared span is the record's to report,
                # and keeping it would put a promoter token inside the record's
                # stretch of the axis.
                f = rec["frame"]
                dup = (prom_pos >= f["genome_start"]) & (prom_pos < f["genome_end"])
                keep[lo:lo + L] = ~dup
                note["duplicate_columns"] = int(dup.sum())
                note["promoter_offset"] = float(off[0] / 3.0)
            else:
                note["genome_anchored"] = False
                coord[lo:lo + L] = (np.arange(L, dtype=np.float32) - L) / 3.0
        return coord, codon, phase, keep, note

    @staticmethod
    def _ref_residues(p, block_frames):
        """H37Rv's residue per codon for one locus, as protein-channel indices.

        Reads ``meta['ref_residues']`` (the string `datasets.protein` attaches on
        the reference codon grid) off whichever aa block carries it. -1 marks a
        codon with no residue, and a locus with no aa block gets None.
        """
        for b in p["aa"]:
            meta = block_frames[b["index"]] or {}
            ref = meta.get("ref_residues")
            if not ref:
                continue
            idx = np.full(b["length"], -1, dtype=np.int64)
            for k, res in enumerate(ref[:b["length"]]):
                j = _RES_ORDER.index(res) if res in _RES_ORDER else -1
                idx[k] = j
            return idx
        return None

    @classmethod
    def from_blocks(cls, blocks, **kwargs):
        """Build straight from loader FeatureBlocks (``data.blocks``).

        Each block's frame rides along on its ``meta`` (see
        ``datasets.cds.block_frame``); an explicit ``block_frames=`` wins."""
        kwargs.setdefault("block_frames",
                          [dict(getattr(b, "meta", None) or {}) for b in blocks])
        return cls([parse_block_key(b.name) for b in blocks],
                   [b.spec() for b in blocks], **kwargs)

    # -- tokenization --------------------------------------------------------

    def _locus_tokens(self, xs):
        """-> (feat, coord, valid, stats) each (B, n_loci, tokens_per_locus, ...).

        Slot 0 of every locus is its [NULL] token and is always valid, so no
        stage-1 attention row is ever fully masked. Slots 1.. are that locus's
        changed base pairs, in 5'->3' order over the merged
        ``[promoter | coding record]`` column axis.
        """
        B = xs[0].shape[0]
        dev, dt = xs[0].device, xs[0].dtype
        T, n_loci = self.tokens_per_locus, len(self.loci)
        feat = torch.zeros(B, n_loci, T, self.c_tok, device=dev, dtype=dt)
        coord = torch.zeros(B, n_loci, T, device=dev, dtype=dt)
        valid = torch.zeros(B, n_loci, T, dtype=torch.bool, device=dev)
        stats = torch.zeros(B, n_loci, 2, device=dev, dtype=dt)

        SL = self.layout["slots"]
        for li, p in enumerate(self._plan):
            # occupancy over the merged axis, from the NUCLEOTIDE blocks only
            occ = torch.cat([xs[b["index"]].abs().sum(1) > 0 for b in p["nt"]], dim=1)
            occ &= getattr(self, f"keep_map_{li}")    # one base, one token
            idx, ok, n_occ = _select_variants(occ, self.max_variants)   # (B, k)
            k = idx.shape[1]
            # everything about WHERE a column sits is a table lookup (see
            # _frame_maps): codons from the start codon, which codon of the
            # protein block, and the position inside it.
            coord[:, li, 1:1 + k] = getattr(self, f"coord_map_{li}")[idx].to(dt)
            codon = getattr(self, f"codon_map_{li}")[idx]
            ph = getattr(self, f"phase_map_{li}")[idx]
            coding = codon >= 0

            f = feat[:, li, 1:1 + k]
            for b in p["nt"]:                       # the base that changed
                lo, hi = SL[b["modality"]]
                x = xs[b["index"]]
                here = (idx >= b["offset"]) & (idx < b["offset"] + b["length"])
                local = (idx - b["offset"]).clamp_(0, b["length"] - 1)
                f[:, :, lo:hi] = x.transpose(1, 2).gather(
                    1, local.unsqueeze(-1).expand(-1, -1, x.shape[1])
                ) * here.unsqueeze(-1).to(dt)

            # What that codon codes for, in this isolate and in H37Rv.
            #
            # A delta-encoded aa block is zero wherever the residue did NOT
            # change, which is indistinguishable from "no residue here". So the
            # slots are filled from the reference there: a synonymous change
            # reads "S, still S", and `d_biophysical` — the only channel that
            # separates it from a real substitution — is zero.
            ref_res = getattr(self, f"ref_res_{li}")          # (n_codons,) or (1,)
            props = self.residue_properties
            has_ref = bool(self._has_ref[li])
            known = coding.clone()
            if has_ref:
                cc = codon.clamp(0, ref_res.numel() - 1)
                ref_id = torch.where(coding, ref_res[cc], torch.zeros_like(cc))
                known = coding & (ref_res[cc] >= 0)
                ref_props = props[ref_id] * known.unsqueeze(-1).to(dt)
            # Which residues changed is read off the PROTEIN block, never off
            # the properties: STOP and a deleted codon have all-zero properties
            # legitimately, so "the property row is zero" does not mean "nothing
            # happened here". Without a protein block the two are not separable,
            # and a property-less residue then reads as unchanged.
            aa = {b["modality"]: b for b in p["aa"]}
            got = {}
            for mod, b in aa.items():
                x = xs[b["index"]]
                here = coding & (codon < b["length"])
                kk = codon.clamp(0, b["length"] - 1)
                got[mod] = x.transpose(1, 2).gather(
                    1, kk.unsqueeze(-1).expand(-1, -1, x.shape[1])
                ) * here.unsqueeze(-1).to(dt)
            if "protein" in got:
                changed = got["protein"].abs().sum(-1, keepdim=True) > 0
            elif "biophysical" in got:
                changed = got["biophysical"].abs().sum(-1, keepdim=True) > 0
            else:
                changed = None
            alt_props = None
            if changed is not None and has_ref:
                # A gap token is a base the isolate LOST, so its codon does not
                # read as the reference no matter what the (delta-zero) aa block
                # says -- restating the reference residue there would report an
                # unchanged codon at a deletion. Substitutions only; the deleted
                # codon keeps empty residue slots and a delta of -ref_props, the
                # same "the reference residue is gone" a stop gain reports.
                base = f[:, :, SL["dna"][0]:SL["dna"][1]]
                sub = (base.sum(-1) > 0) & (base[..., GAP_CHANNEL] == 0)
                if self.unknown_symbols:            # an uncalled base is no substitution
                    sub &= base[..., NT_CHANNELS] == 0
                fill = ~changed & known.unsqueeze(-1) & sub.unsqueeze(-1)
                if "protein" in got:
                    ref_oh = F.one_hot(ref_id, num_classes=self.layout["protein_channels"]).to(dt)
                    got["protein"] = torch.where(
                        fill, ref_oh * known.unsqueeze(-1).to(dt), got["protein"])
                if "biophysical" in got:
                    got["biophysical"] = torch.where(fill, ref_props, got["biophysical"])
            for mod, v in got.items():
                lo, hi = SL[mod]
                f[:, :, lo:hi] = v
            if "biophysical" in got:
                alt_props = got["biophysical"]
            elif "protein" in got:
                # no biophysical block loaded: the residue's properties are a
                # table lookup away, so the delta is still available
                alt_props = got["protein"] @ props
            # what the substitution did to those properties -- the channels that
            # make a token read as a CHANGE rather than as a state. The reference
            # values themselves are not carried: they are a constant of the
            # position, and the isolate's own properties are already on the token.
            db, fp = self.layout["d_bio"], self.layout["f_phase"]
            if has_ref and alt_props is not None:
                f[:, :, db[0]:db[1]] = (alt_props - ref_props) * \
                    known.unsqueeze(-1).to(dt)
            if self.layout["ref_bio"] is not None and has_ref and coding.any():
                rb = self.layout["ref_bio"]         # --lf-ref-bio: H37Rv's own properties
                f[:, :, rb[0]:rb[1]] = ref_props * known.unsqueeze(-1).to(dt)
            f[:, :, fp:fp + 3] = F.one_hot(
                ph.clamp(0, 2), num_classes=3).to(dt) * coding.unsqueeze(-1).to(dt)

            # Where this token sits on the fold, relative to the locus's other
            # variant tokens: row i of the (k x k) C-alpha distance matrix goes
            # to variant token i, so channel j of token i is its distance to
            # token j -- the same 5'->3' token order on both axes. `ok` keeps
            # padding out of it.
            if p["structure"] is not None:
                dist, mask = self._struct_pairs(li, codon, ok, dt)
                a = self.f_struct
                f[:, :, a:a + self.max_variants] = self.struct_invalid
                f[:, :, a:a + k] = torch.where(
                    mask, torch.exp(-dist / self.struct_length),
                    torch.full_like(dist, self.struct_invalid))

            feat[:, li, 1:1 + k] = f * ok.unsqueeze(-1).to(dt)
            coord[:, li, 1:1 + k] = coord[:, li, 1:1 + k] * ok.to(dt)
            valid[:, li, 1:1 + k] = ok
            # [NULL]: how much of this locus deviates at all. The fraction is what
            # v1 thresholded into an `uncovered` flag -- an all-gap record differs
            # from the reference at EVERY column and must not read as a locus
            # that matches it.
            valid[:, li, 0] = True
            feat[:, li, 0, self.layout["f_is_null"]] = 1.0
            stats[:, li, 0] = torch.log1p(n_occ.to(dt))
            stats[:, li, 1] = n_occ.to(dt) / max(p["kept"], 1)
        return feat, coord, valid, stats

    def _struct_pairs(self, li, codon, ok, dt):
        """(dist (B, k, k) in A, mask (B, k, k)): what the proximity rows are cut
        from, under this model's struct_dist and struct_gate."""
        dist, mask, located = pair_distances(
            codon, ok, getattr(self, f"struct_ca_{li}").to(dt),
            getattr(self, f"struct_has_{li}"))
        if self.struct_gate == "contact":
            bad = getattr(self, f"struct_pae_bad_{li}")
            cc = codon.clamp(0, bad.shape[0] - 1)
            mask = mask & (dist <= CONTACT_REACH) & ~bad[cc.unsqueeze(2), cc.unsqueeze(1)]
        if self.struct_dist == "heavy":
            dist = heavy_min_distances(codon, located, getattr(self, f"struct_atoms_{li}").to(dt),
                                       getattr(self, f"struct_atom_ok_{li}")) * mask.to(dt)
        return dist, mask

    def _check_occupancy(self, valid):
        """Warn once if the input does not look delta-encoded.

        A dense one-hot occupies every column, the cap then keeps the first
        `max_variants` of them, and the model silently becomes "read the head of
        each locus" — a failure that produces plausible numbers, so it is worth
        a loud warning rather than a docstring."""
        self._occupancy_checked = True
        filled = valid[:, :, 1:].to(torch.float32).mean().item()
        if filled > 0.9:
            warnings.warn(
                f"LocusFusionV2Net: {100 * filled:.0f}% of variant slots are occupied. "
                "This architecture expects reference-difference input (--delta); "
                "on a plain one-hot every column differs from nothing and the "
                "tokenizer just keeps the first max_variants columns of each "
                "locus. Re-run with --delta.", stacklevel=3)

    # -- forward -------------------------------------------------------------

    def forward(self, xs, return_attn=False):
        """xs: list of (B, C_i, L_i) delta-encoded block tensors."""
        if len(xs) != len(self.block_keys):
            raise ValueError(f"expected {len(self.block_keys)} blocks, got {len(xs)}")
        feat, coord, valid, stats = self._locus_tokens(xs)
        if not self._occupancy_checked:
            self._check_occupancy(valid)
        B, n_loci, T, _ = feat.shape
        li = torch.arange(n_loci, device=feat.device)

        tok = self.tok_proj(feat) + self.pos_proj(_sinusoid(coord, self.pos_dims))
        tok = tok + self.locus_emb(li).unsqueeze(0).unsqueeze(2)
        tok[:, :, 0] = tok[:, :, 0] + self.null_emb(li).unsqueeze(0) + self.null_proj(stats)
        if self.film_scale is not None:
            tok = tok * (1.0 + self.film_scale[li].unsqueeze(0).unsqueeze(2)) \
                + self.film_shift[li].unsqueeze(0).unsqueeze(2)
        tok = self.tok_norm(tok) * valid.unsqueeze(-1).to(tok.dtype)

        # --- stage 1: within-locus ------------------------------------------
        pad = ~valid
        if self.locus_encoder == "per_locus":
            z = torch.stack([enc(tok[:, i], src_key_padding_mask=pad[:, i])
                             for i, enc in enumerate(self.encoders)], dim=1)
        else:
            z = self.encoders[0](tok.reshape(B * n_loci, T, self.d_model),
                                 src_key_padding_mask=pad.reshape(B * n_loci, T))
            z = z.reshape(B, n_loci, T, self.d_model)

        # --- stage 2: across loci -------------------------------------------
        m = self.carry_variants
        fused_in = z[:, :, 0]                                   # (B, n_loci, d)
        if self.summ_norm is not None:
            fused_in = self.summ_norm(fused_in, li)
        fused_pad = torch.zeros(B, n_loci, dtype=torch.bool, device=z.device)
        if m:
            fused_in = torch.cat([fused_in, z[:, :, 1:1 + m].reshape(B, n_loci * m, -1)], 1)
            fused_pad = torch.cat([fused_pad, pad[:, :, 1:1 + m].reshape(B, n_loci * m)], 1)
        fused = self.fusion(fused_in, src_key_padding_mask=fused_pad)

        # --- readout: one query per drug over the fused set -------------------
        q = self.drug_queries.unsqueeze(0).expand(B, -1, -1)
        pooled, attn = self.pool_attn(q, fused, fused, key_padding_mask=fused_pad,
                                      need_weights=return_attn, average_attn_weights=True)
        h = self.head_drop(F.relu(self.fc1(self.norm(pooled))))  # (B, n_drugs, hidden)
        if self.fc_out is not None:
            logits = self.fc_out(h).squeeze(-1)
        else:
            logits = torch.cat(
                [out(F.relu(hid(h[:, j]))) for j, (hid, out)
                 in enumerate(zip(self.drug_hidden, self.drug_out))], dim=1)
        return (logits, attn) if return_attn else logits

    # -- interpretability ----------------------------------------------------

    @torch.no_grad()
    def variant_report(self, xs):
        """What the tokenizer actually saw, per locus, for one batch.

        ``[{locus, columns, region, valid, codon, coord, n_variants,
        occupied_frac}]``
        — one row per locus, each field (B, max_variants) except the last two.
        With a structure block, also ``struct_dist`` (A) and ``struct_mask``,
        each (B, k, k): the distance matrix the token rows were cut from.
        ``columns`` is in the merged axis; ``region`` is 'promoter' or 'cds' and
        ``column_in_block`` re-indexes into whichever block that is, so an
        attention map from ``forward(..., return_attn=True)`` reads as "drug j
        attended to katG alignment column 944, codon 315" rather than to token 7.
        """
        out = []
        for p in self._plan:
            li = self.loci.index(p["locus"])
            occ = torch.cat([xs[b["index"]].abs().sum(1) > 0 for b in p["nt"]], dim=1)
            occ &= getattr(self, f"keep_map_{li}")
            idx, ok, n_occ = _select_variants(occ, self.max_variants)
            in_reg = idx < p["reg_len"]
            codon = getattr(self, f"codon_map_{li}")[idx]
            extra = {}
            if p["structure"] is not None:
                dist, mask = self._struct_pairs(li, codon, ok, torch.float32)
                extra = {"struct_dist": dist, "struct_mask": mask}
            out.append({
                "locus": p["locus"],
                "modalities": [b["modality"] for b in p["nt"] + p["aa"]],
                "columns": idx,
                "column_in_block": torch.where(in_reg, idx, idx - p["reg_len"]),
                "region": torch.where(in_reg, 0, 1),        # 0 promoter, 1 cds
                "valid": ok,
                "codon": getattr(self, f"codon_map_{li}")[idx],
                "coord": getattr(self, f"coord_map_{li}")[idx],
                "n_variants": n_occ,
                "occupied_frac": n_occ.to(torch.float32) / max(p["kept"], 1),
                **extra,
            })
        return out
