"""
Hierarchical locus-fusion transformer (``--arch locusfusion``).

Two ideas, and everything here follows from them.

**1. Fuse at the gene, then across genes.** Every other architecture in this
project fuses the modalities and the loci in the SAME step — `late_fusion` at
one flatten, `mdcnn` at layer 1, `setfusion` in one transformer over one token
per block. This one is two stages: all of *rpoB*'s evidence (its CDS symbols,
its translation, that translation's biophysical profile, its promoter window) is
fused into a single **locus representation** first, and only those 19 locus
representations then talk to each other. A resistance mechanism is a property of
a gene; a resistance *phenotype* is a property of the set of genes.

**2. A token per VARIANT, not a token per patch.** This is the fix for the
failure that `results/experiments/token_signal` diagnosed and that every
transformer arm in this project has died of. *M. tuberculosis* is clonal: the
measured median isolate differs from H37Rv at **0-3 columns of a 2.5 kb gene**
(census in `README.md`). A patch-embedding transformer therefore spends ~99.9%
of its tokens restating sequence that is identical in every isolate, and the
measured consequence was an encoded token whose per-isolate part is 0.14% of its
magnitude, attention pinned at exactly uniform, and a linear probe on the encoder
output beating the trained model.

So do not tokenize the sequence. Tokenize the **difference from the reference**:
a column becomes a token only where the isolate's symbol differs from H37Rv's.
100% of a token varies with the genotype by construction.

That is also the shape of the hypothesis the biology suggests and the reason
this net is built the way it is: **a susceptible isolate is the empty set.**
Each locus gets a learned ``[WT]`` sentinel token — the wild-type null — and
every variant token is evidence against it. A pan-susceptible isolate presents
19 sentinels and nothing else; the model's job is to score deviations, not to
re-derive what a sensitive strain looks like from 12,000 constant columns.

What a token is
---------------

A variant is a discrete event, so a token is three small integers and one
coordinate — not a 42-wide sparse float vector::

    alt    symbol id of what the ISOLATE has here      (35-symbol vocabulary)
    ref    symbol id of what H37Rv has here            (same vocabulary)
    phase  codon position 0/1/2, or "not applicable"   (amino-acid tokens)
    coord  exact H37Rv codon number, fractional        (a float)

embedded as ``alt_emb[alt] + ref_emb[ref] + phase_emb[phase] +
pos_proj(sinusoid(coord)) + locus_emb[locus]``, plus a 3-dim projection of the
biophysical properties when that modality is loaded and a single learned vector
when the locus failed to assemble. The vocabulary and the coordinate map live in
``datasets/tokens.py``, which is also where the two things this replaced are
written up: the old layout's duplicated flags, and the coordinate bug.

The coordinate is the point. It is the H37Rv codon number, computed from the
CDS annotation and the reference gap pattern, so a DNA token and a protein token
for the same residue land on the same axis — "katG 315" is 314.0 whether it
arrives through the nucleotide stream or the protein stream. The previous
version placed the nucleotide token at ``column/3`` minus a learned per-locus
scalar, which put katG S315 at 357.3 against its protein token's 314, and read
``[-0.0107, +0.0081]`` for that scalar off a fully trained checkpoint.

Three things fall out of the variant-token design, all of them things the pooled
architectures gave up:

* **Exact position survives**, to the nucleotide — better than `mdcnn`, which
  pools 9-fold, and unlike `setfusion`, which coarsens a locus to 4 relative
  bins.
* **The input collapses.** 19 loci x up to 4,066 columns becomes a median of
  ~14 tokens per isolate. Attention over 32 tokens is free where attention over
  1,300 patches was not, so the size question that dominated `transformer_run`
  stops being the binding constraint.
* **Attribution is free and exact.** ``forward(..., return_attn=True)`` returns
  which token each drug read, and a token IS a variant with a locus, a
  coordinate and a ref->alt pair — checkable against the WHO catalogue without
  SHAP.

What it cannot see: anything constant across the cohort (which carries no
discriminative signal, so this is a real restriction with no measured cost), and
sequence *context* around a variant beyond what the position encoding says.

Requires symbol-id blocks (``load_dataset(variant_tokens=True)``) and PER-LOCUS
blocks; both runners set them from ``--arch locusfusion``. The one modality with
no symbol-id form is biophysical, deliberately: it is the modality whose claim is
that three properties stand in for the residue identity, so handing it that
identity would answer its own ablation. In a ``dna+biophysical`` cell the
amino-acid stream therefore carries properties and coordinates but no symbol,
and a residue whose properties are unchanged stays invisible there.
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import tokens as tok

from .net import NO_LOCUS, KeyedTokenNorm, parse_block_key
from .variant_tokens import STREAMS, select_variants, sinusoid

LOCUSFUSION_DEFAULTS = {
    # A  token width, shared by both stages, both embeddings and the readout
    "d_model": 128, "nhead": 4, "dropout": 0.1,
    # B  stage 1 — within one locus, over its own variant tokens
    "enc_layers": 2, "enc_dim_ff": 256,
    # C  stage 2 — across loci, over the locus summaries
    "fusion_layers": 2, "fusion_dim_ff": 256,
    # D  the tokenizer. `max_variants` is per (locus, coordinate-stream): the
    #    census in README.md puts the 99th percentile at <=7 columns per locus
    #    for 17 of 19 loci and 26 for the two rRNA genes, so 16 covers >99% of
    #    (isolate, locus) pairs. Overflow keeps the FIRST 16 in positional
    #    order (deterministic, and a symbol id has no magnitude to rank by).
    #    `pos_dims` is 32 rather than 64 because the sinusoid band is now fitted
    #    to the data — [1/3, 4096] codons, see variant_tokens.sinusoid — instead
    #    of spending half its range on wavelengths longer than any locus.
    "max_variants": 16, "pos_dims": 32, "uncovered_frac": 0.5,
    # E  per-locus specialization. 'shared' = one stage-1 encoder for every
    #    locus, identity carried only by `locus_emb`; 'adapter' = the same
    #    encoder plus a per-locus FiLM (scale+shift) on its input, 2*d_model
    #    parameters per locus; 'per_locus' = a separate stage-1 encoder per
    #    locus (19x the stage-1 weights, and it cannot batch the loci together).
    "locus_encoder": "adapter",
    # F  standardise each locus summary across the batch, with statistics kept
    #    per locus, before stage 2 sees it. ON by default, and the measurement
    #    below is why. A locus summary is read off the [WT] slot, and for a
    #    wild-type isolate that slot's input is IDENTICAL in every isolate — so
    #    at init only 1.6% of the summary varies with the genotype (measured,
    #    400 real INH isolates), the same shape of failure `token_signal`
    #    diagnosed in setfusion at 0.14%. Subtracting the per-locus mean deletes
    #    the constant, which is what took setfusion's ISONIAZID cell from 0.9287
    #    to 0.962 at 1% of the parameters. The constant here is MEANINGFUL (it is
    #    the wild-type null) but it is still constant, so it can carry no
    #    per-isolate signal and downstream affine layers absorb it either way.
    #    Statistics are batch statistics collected on TRAINING batches only, as
    #    in any BatchNorm — eval uses the running estimates, so nothing about the
    #    held-out split leaks.
    "summary_norm": "keyed",
    # G  how many of a locus's own variant tokens are handed up to stage 2
    #    alongside its summary. 0 = summaries only. >0 lets cross-locus
    #    attention see individual variants (rpoB + rpoC compensatory pairs),
    #    at the cost of a larger stage-2 token set.
    "carry_variants": 0,
}

LOCUS_ENCODERS = ("shared", "adapter", "per_locus")
SUMMARY_NORMS = ("none", "keyed")

# Modalities that arrive as symbol ids, (B, 1, L). Biophysical is the exception
# and arrives as its 3 property channels; see the module docstring.
ID_MODALITIES = ("dna", "protein", "regulatory")
N_BIO = 3


class LocusFusionNet(nn.Module):
    """Variant-token transformer: fuse modalities within a locus, then loci.

    Pipeline, per isolate::

        per locus L, per coordinate stream (nt / aa / reg):
            symbol ids (B, 1, len)  --alt != ref-->  the columns that differ
                                    --gather-->      <= max_variants tokens
        token  = alt_emb + ref_emb + phase_emb + pos_proj(sinusoid(coord))
               + locus_emb  (+ bio_proj(properties))  (+ uncovered_emb)
        [WT]_L = wt_emb[L] + wt_proj(variant count, coverage, uncovered)

        stage 1:  TransformerEncoder over {[WT]_L, variants of L}   -> z_L = out[[WT]]
        stage 2:  TransformerEncoder over {z_L for every locus}     -> fused
        readout:  one learned query per drug, cross-attending the fused set

    Same ``forward(xs)`` contract as every other arch — a list of (B, C_i, L_i)
    block tensors in ``branch_specs`` order — so the existing trainers drive it
    unchanged. ``block_keys`` are ``(modality, locus)`` pairs and ``column_meta``
    is the per-block ``{coord, phase, ref_id}`` the loader attached;
    ``from_blocks`` takes all three straight off the loader's FeatureBlocks.
    """

    bio_input = "blocks"        # forward takes the list of block tensors as-is

    def __init__(self, block_keys, branch_specs, column_meta=None,
                 n_drugs=1, drug_names=None,
                 d_model=128, nhead=4, dropout=0.1,
                 enc_layers=2, enc_dim_ff=256,
                 fusion_layers=2, fusion_dim_ff=256,
                 max_variants=16, pos_dims=32, uncovered_frac=0.5,
                 locus_encoder="adapter", summary_norm="keyed", carry_variants=0,
                 hidden=256, head_dropout=0.0, per_drug_hidden=0, out_bias=None):
        super().__init__()
        if not branch_specs:
            raise ValueError("LocusFusionNet needs at least one branch spec")
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
        if pos_dims % 2:
            raise ValueError(f"pos_dims must be even, got {pos_dims}")
        if drug_names:
            n_drugs = len(drug_names)

        block_keys = [(m, l or NO_LOCUS) for m, l in block_keys]
        unknown = sorted({m for m, _ in block_keys} - set(STREAMS))
        if unknown:
            raise ValueError(f"LocusFusionNet has no coordinate stream for "
                             f"modalities {unknown}; known: {sorted(STREAMS)}")
        column_meta = list(column_meta) if column_meta else [None] * len(block_keys)
        if len(column_meta) != len(block_keys):
            raise ValueError("column_meta must match block_keys length "
                             f"({len(column_meta)} vs {len(block_keys)})")
        for (modality, _l), (channels, _len) in zip(block_keys, branch_specs):
            want = 1 if modality in ID_MODALITIES else N_BIO
            if channels != want:
                raise ValueError(
                    f"LocusFusionNet expects {want} channel(s) for a "
                    f"{modality!r} block and got {channels}. The symbol-id "
                    f"layout is what this architecture reads — load with "
                    f"variant_tokens=True (both runners set it from "
                    f"--arch locusfusion).")

        # --- group the flat block list into loci, and each locus into the
        # coordinate streams its blocks share -------------------------------
        self.loci = list(dict.fromkeys(l for _, l in block_keys))
        if self.loci == [NO_LOCUS]:
            warnings.warn(
                "LocusFusionNet: every block keyed to <none> — you passed the "
                "merged per-modality blocks, so there is exactly one 'locus' "
                "and the whole two-stage structure collapses. Load with "
                "per_modality_branch=False for per-locus blocks.", stacklevel=2)
        # plan[locus_index] = [(stream_name, [(block_index, modality), ...], length), ...]
        plan = []
        for locus in self.loci:
            streams = {}
            for i, ((modality, l), (_c, length)) in enumerate(zip(block_keys, branch_specs)):
                if l != locus:
                    continue
                s = STREAMS[modality]
                members, prev_len = streams.setdefault(s, ([], length))
                if length != prev_len:
                    raise ValueError(
                        f"{locus}: blocks {[m for _, m in members] + [modality]} share "
                        f"the {s!r} coordinate stream but have different lengths "
                        f"({prev_len} vs {length}); they must be co-indexed")
                members.append((i, modality))
            plan.append([(s, members, length)
                         for s, (members, length) in streams.items()])
        self._plan = plan

        # --- the per-column constants, as buffers ----------------------------
        # coord / phase / ref_id are identical for every isolate, so they belong
        # to the model, not to the batch. One entry per (locus, stream), keyed by
        # a flat name because ModuleDict/BufferDict keys cannot contain dots.
        self._meta_keys = {}
        for li, streams in enumerate(plan):
            for si, (stream, members, length) in enumerate(streams):
                meta = self._pick_meta(column_meta, members)
                key = f"m{li}_{si}"
                if meta is None:
                    # No reference available (a fixture without an H37Rv row, or
                    # a biophysical-only amino-acid stream). Residue index IS the
                    # codon coordinate, and with no reference symbol the stream
                    # falls back to "any non-zero property" occupancy.
                    coord = torch.arange(length, dtype=torch.float32)
                    phase = torch.full((length,), tok.PHASE_NA, dtype=torch.long)
                    ref = torch.full((length,), tok.AA_UNK if stream == "aa"
                                     else tok.PAD, dtype=torch.long)
                else:
                    coord = torch.as_tensor(meta["coord"], dtype=torch.float32)
                    phase = torch.as_tensor(meta["phase"], dtype=torch.long)
                    ref = torch.as_tensor(meta["ref_id"], dtype=torch.long)
                # PERSISTENT on purpose. The reference symbols and the codon
                # coordinate are part of what the trained model means, not a
                # property of whatever dataset is loaded next: a checkpoint
                # rebuilt through training/checkpoint.py gets its keys and specs
                # from the config and would otherwise come back with the
                # fallback map, quietly moving every token. They cost a few
                # hundred KB and make a checkpoint self-contained.
                self.register_buffer(f"coord_{key}", coord)
                self.register_buffer(f"phase_{key}", phase)
                self.register_buffer(f"ref_{key}", ref)
                self._meta_keys[(li, si)] = key

        # every locus contributes [WT] + max_variants per stream; the token axis
        # is padded to the widest locus so all loci run through stage 1 in ONE
        # batched call instead of a python loop over 19 transformers.
        self.max_variants = int(max_variants)
        self.tokens_per_locus = 1 + self.max_variants * max(len(p) for p in plan)
        self.uncovered_frac = float(uncovered_frac)
        self.carry_variants = min(int(carry_variants), self.tokens_per_locus - 1)

        # --- token embedding -------------------------------------------------
        n_loci = len(self.loci)
        self.alt_emb = nn.Embedding(tok.N_SYMBOLS, d_model, padding_idx=tok.PAD)
        self.ref_emb = nn.Embedding(tok.N_SYMBOLS, d_model, padding_idx=tok.PAD)
        self.phase_emb = nn.Embedding(tok.N_PHASES, d_model)
        self.pos_proj = nn.Linear(pos_dims, d_model)
        self.locus_emb = nn.Embedding(n_loci, d_model)
        self.wt_emb = nn.Embedding(n_loci, d_model)
        self.wt_proj = nn.Linear(3, d_model)         # count / coverage / uncovered
        # An all-gap record differs from the reference at EVERY column, so
        # without this a locus that simply failed to assemble reads as the
        # most-mutated isolate in the cohort. 14-91 isolates per locus are in
        # that state (README census); they are not wild type and must not look
        # it. One learned vector, added to every token of an uncovered locus —
        # the old layout spent a whole feature slot broadcasting one bit.
        self.uncovered_emb = nn.Parameter(torch.zeros(d_model))
        self.has_bio = any(m == "biophysical" for m, _ in block_keys)
        self.bio_proj = nn.Linear(N_BIO, d_model) if self.has_bio else None
        self.tok_norm = nn.LayerNorm(d_model)
        for emb in (self.alt_emb, self.ref_emb, self.phase_emb,
                    self.locus_emb, self.wt_emb):
            nn.init.trunc_normal_(emb.weight, std=0.02)
        with torch.no_grad():                        # padding_idx rows stay zero
            self.alt_emb.weight[tok.PAD].zero_()
            self.ref_emb.weight[tok.PAD].zero_()
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
        # keyed by LOCUS: the summary of katG is standardised against other
        # isolates' katG summaries, never against inhA's. See the note on
        # `summary_norm` in LOCUSFUSION_DEFAULTS for why this is on by default.
        self.summary_norm = summary_norm
        self.summ_norm = (KeyedTokenNorm(n_loci, d_model)
                          if summary_norm == "keyed" else None)
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, fusion_dim_ff, dropout=dropout,
                                       batch_first=True, norm_first=True),
            fusion_layers, enable_nested_tensor=False)

        # --- readout: one learned query per drug (setfusion's, kept because it
        # is the interpretable one — here the keys are individual variants) ----
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
    def _pick_meta(column_meta, members):
        """The metadata for a stream: whichever member block carries it.

        Only the symbol-id modalities attach one, so an amino-acid stream with
        protein loaded takes protein's (which has the reference residues) and
        one with only biophysical loaded gets None.
        """
        for i, modality in members:
            if modality in ID_MODALITIES and column_meta[i] is not None:
                return column_meta[i]
        return None

    @classmethod
    def from_blocks(cls, blocks, **kwargs):
        """Build straight from loader FeatureBlocks (``data.blocks``)."""
        return cls([parse_block_key(b.name) for b in blocks],
                   [b.spec() for b in blocks],
                   column_meta=[b.column_meta for b in blocks], **kwargs)

    # -- tokenization --------------------------------------------------------

    def _stream_tokens(self, xs, li, si, stream, members, length):
        """One coordinate stream of one locus -> the tokens it contributes.

        `members` may hold several blocks (protein + biophysical are co-indexed
        by construction — same translate path, same k_max), in which case the
        symbol comes from the id block and the properties ride along at the same
        selected columns. That is the gene-level fusion: one token per changed
        residue carrying every modality's view of it.

        Occupancy is ``alt != ref`` wherever a reference is known, which is what
        makes an unresolved base call visible: ``N`` against a reference ``C``
        is a difference, where under the old all-zero-column encoding it was
        indistinguishable from a match.
        """
        key = self._meta_keys[(li, si)]
        coord_map = getattr(self, f"coord_{key}")
        phase_map = getattr(self, f"phase_{key}")
        ref_map = getattr(self, f"ref_{key}")

        ids = bio = None
        for i, modality in members:
            if modality in ID_MODALITIES:
                ids = xs[i][:, 0].to(torch.long)               # (B, len)
            else:
                bio = xs[i]                                    # (B, 3, len)

        if ids is not None:
            occ = ids != ref_map.unsqueeze(0)
            # padding past a block's real extent is not a variant
            occ &= ids != tok.PAD
        else:
            # biophysical-only stream: no symbol to compare, so fall back to the
            # delta encoding's own signal — a residue whose properties moved.
            occ = bio.abs().sum(1) > 0
        idx, valid, n_occ = select_variants(occ, self.max_variants)

        alt = ids.gather(1, idx) if ids is not None else torch.full_like(
            idx, tok.AA_UNK if stream == "aa" else tok.PAD)
        ref = ref_map[idx]
        phase = phase_map[idx]
        coord = coord_map[idx]
        props = (bio.transpose(1, 2).gather(1, idx.unsqueeze(-1).expand(-1, -1, N_BIO))
                 if bio is not None else None)
        return idx, alt, ref, phase, coord, props, valid, n_occ, length

    def _locus_tokens(self, xs):
        """-> per-locus token fields, each shaped (B, n_loci, tokens_per_locus, ...).

        Slot 0 of every locus is its [WT] sentinel and is always valid, so no
        stage-1 attention row is ever fully masked.
        """
        B = xs[0].shape[0]
        dev = xs[0].device
        dt = torch.float32
        T = self.tokens_per_locus
        n_loci = len(self.loci)
        alt = torch.full((B, n_loci, T), tok.PAD, dtype=torch.long, device=dev)
        ref = torch.full((B, n_loci, T), tok.PAD, dtype=torch.long, device=dev)
        phase = torch.full((B, n_loci, T), tok.PHASE_NA, dtype=torch.long, device=dev)
        coord = torch.zeros(B, n_loci, T, device=dev, dtype=dt)
        props = (torch.zeros(B, n_loci, T, N_BIO, device=dev, dtype=dt)
                 if self.has_bio else None)
        valid = torch.zeros(B, n_loci, T, dtype=torch.bool, device=dev)
        stats = torch.zeros(B, n_loci, 3, device=dev, dtype=dt)

        for li, streams in enumerate(self._plan):
            cursor = 1
            n_var = torch.zeros(B, device=dev, dtype=dt)
            frac = torch.zeros(B, device=dev, dtype=dt)
            for si, (stream, members, length) in enumerate(streams):
                _idx, a, r, p, c, pr, v, n_occ, length = self._stream_tokens(
                    xs, li, si, stream, members, length)
                k = a.shape[1]
                sl = slice(cursor, cursor + k)
                alt[:, li, sl] = a
                ref[:, li, sl] = r
                phase[:, li, sl] = p
                coord[:, li, sl] = c
                if props is not None and pr is not None:
                    props[:, li, sl] = pr
                valid[:, li, sl] = v
                cursor += k
                n_var = n_var + n_occ.to(dt)
                frac = torch.maximum(frac, n_occ.to(dt) / length)
            uncovered = (frac > self.uncovered_frac).to(dt)
            stats[:, li, 0] = torch.log1p(n_var)   # true count, BEFORE the cap
            stats[:, li, 1] = frac
            stats[:, li, 2] = uncovered
            valid[:, li, 0] = True
            alt[:, li, 0] = tok.WT
            ref[:, li, 0] = tok.WT
        return alt, ref, phase, coord, props, valid, stats

    def _embed(self, alt, ref, phase, coord, props, valid, stats, li):
        """The token vector: four lookups, one projection, and a coordinate.

        Every term here is information the token actually carries. The layout
        this replaced spent 42 float slots to say the same thing, nine of them
        on flags derivable from the rest (see datasets/tokens.py) and the widest
        two on nucleotide and promoter one-hots that can never both be set,
        because a token belongs to exactly one coordinate stream.
        """
        tokens = (self.alt_emb(alt) + self.ref_emb(ref) + self.phase_emb(phase)
                  + self.pos_proj(sinusoid(coord, self.pos_dims))
                  + self.locus_emb(li).unsqueeze(0).unsqueeze(2))
        if props is not None:
            tokens = tokens + self.bio_proj(props)
        tokens = tokens + (stats[:, :, 2:3] * self.uncovered_emb).unsqueeze(2)
        tokens[:, :, 0] = tokens[:, :, 0] + self.wt_emb(li).unsqueeze(0) \
            + self.wt_proj(stats)
        if self.film_scale is not None:
            tokens = tokens * (1.0 + self.film_scale[li].unsqueeze(0).unsqueeze(2)) \
                + self.film_shift[li].unsqueeze(0).unsqueeze(2)
        return self.tok_norm(tokens) * valid.unsqueeze(-1).to(tokens.dtype)

    def _check_occupancy(self, valid):
        """Warn once if nearly every slot is a variant.

        With symbol-id blocks this should be impossible on real data — the
        census puts the median isolate at 0-3 differences per locus — so a full
        token set means the reference metadata did not load and every column is
        reading as a difference. That produces plausible numbers, which is worth
        a loud warning rather than a docstring."""
        self._occupancy_checked = True
        filled = valid[:, :, 1:].to(torch.float32).mean().item()
        if filled > 0.9:
            warnings.warn(
                f"LocusFusionNet: {100 * filled:.0f}% of variant slots are occupied. "
                "Every column is reading as different from the reference, which "
                "on a clonal cohort means the per-column reference ids did not "
                "load (no H37Rv row in the alignment?). Check that the loader "
                "ran with variant_tokens=True and that column_meta reached "
                "from_blocks.", stacklevel=3)

    # -- forward -------------------------------------------------------------

    def forward(self, xs, return_attn=False):
        """xs: list of (B, C_i, L_i) block tensors, symbol ids except biophysical."""
        if len(xs) != len(self.block_keys):
            raise ValueError(f"expected {len(self.block_keys)} blocks, got {len(xs)}")
        alt, ref, phase, coord, props, valid, stats = self._locus_tokens(xs)
        if not self._occupancy_checked:
            self._check_occupancy(valid)
        B, n_loci, T = alt.shape
        li = torch.arange(n_loci, device=alt.device)
        tokens = self._embed(alt, ref, phase, coord, props, valid, stats, li)

        # --- stage 1: within-locus ------------------------------------------
        pad = ~valid
        if self.locus_encoder == "per_locus":
            z = torch.stack([enc(tokens[:, i], src_key_padding_mask=pad[:, i])
                             for i, enc in enumerate(self.encoders)], dim=1)
        else:
            z = self.encoders[0](tokens.reshape(B * n_loci, T, self.d_model),
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

        ``[{locus, stream, modalities, columns, coord, ref, alt, valid,
        n_variants, uncovered}]`` — the token set spelled out, so an attention
        map from ``forward(..., return_attn=True)`` reads as "drug j attended to
        katG codon 314, Ser->Thr" rather than to token 7. ``coord`` is the H37Rv
        codon number, which is the coordinate the WHO catalogue names a mutation
        by, so ``datasets/who_catalogue.py`` can be joined against it directly.
        """
        names = tok.symbol_names()
        out = []
        for li, streams in enumerate(self._plan):
            for si, (stream, members, length) in enumerate(streams):
                idx, alt, ref, phase, coord, _props, valid, n_occ, length = \
                    self._stream_tokens(xs, li, si, stream, members, length)
                out.append({
                    "locus": self.loci[li], "stream": stream,
                    "modalities": [m for _, m in members],
                    "columns": idx, "coord": coord, "phase": phase,
                    "ref": ref, "alt": alt,
                    "ref_name": [[names[i] for i in row] for row in ref.tolist()],
                    "alt_name": [[names[i] for i in row] for row in alt.tolist()],
                    "valid": valid, "n_variants": n_occ,
                    "uncovered": (n_occ.float() / length) > self.uncovered_frac,
                })
        return out
