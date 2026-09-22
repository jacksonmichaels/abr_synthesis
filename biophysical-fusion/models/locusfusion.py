"""
Hierarchical locus-fusion transformer (``--arch locusfusion``).

Two ideas, and everything here follows from them.

**1. Fuse at the gene, then across genes.** Every other architecture in this
project fuses the modalities and the loci in the SAME step — `late_fusion` at
one flatten, `mdcnn` at layer 1, `setfusion` in one transformer over one token
per block. This one is two stages: all of *rpoB*'s evidence (its CDS one-hot,
its translation, that translation's biophysical profile, its promoter window)
is fused into a single **locus representation** first, and only those 19 locus
representations then talk to each other. A resistance mechanism is a property
of a gene; a resistance *phenotype* is a property of the set of genes.

**2. A token per VARIANT, not a token per patch.** This is the fix for the
failure that `results/experiments/token_signal` diagnosed and that every
transformer arm in this project has died of. *M. tuberculosis* is clonal: the
measured median isolate differs from H37Rv at **0-3 columns of a 2.5 kb gene**
(census in `README.md`). A patch-embedding transformer therefore spends ~99.9%
of its tokens restating sequence that is identical in every isolate, and the
measured consequence was an encoded token whose per-isolate part is 0.14% of
its magnitude, attention pinned at exactly uniform, and a linear probe on the
encoder output beating the trained model.

So do not tokenize the sequence. Tokenize the **difference from the reference**:
run on ``--delta`` input (`datasets.sequences.delta_one_hot_nt`, which zeroes
every column matching the real MT_H37Rv record) and emit one token per column
that survives. A token then exists *only* where the genotype deviates from wild
type, so 100% of it varies with the genotype by construction.

That is also the shape of the hypothesis the biology suggests and the reason
this net is built the way it is: **a susceptible isolate is the empty set.**
Each locus gets a learned ``[WT]`` sentinel token — the wild-type null — and
every variant token is evidence against it. A pan-susceptible isolate presents
19 sentinels and nothing else; the model's job is to score deviations, not to
re-derive what a sensitive strain looks like from 12,000 constant columns.

Three things fall out of that, all of them things the pooled architectures gave
up:

* **Exact position survives.** A variant token carries its own coordinate
  through a sinusoidal encoding, so "column 315 of katG" is preserved to the
  nucleotide — better than `mdcnn`, which pools 9-fold, and unlike
  `setfusion`, which coarsens a locus to 4 relative bins.
* **The input collapses.** 19 loci x up to 4,066 columns becomes a median of
  ~14 tokens per isolate. Attention over 32 tokens is free where attention over
  1,300 patches was not, so the size question that dominated
  `transformer_run` stops being the binding constraint.
* **Attribution is free and exact.** ``forward(..., return_attn=True)`` returns
  which token each drug read, and a token IS a variant with a locus and a
  coordinate — checkable against the WHO catalogue without SHAP.

What it cannot see: anything constant across the cohort (which carries no
discriminative signal, so this is a real restriction with no measured cost),
and sequence *context* around a variant beyond what the position encoding says.

Requires ``--delta`` and PER-LOCUS blocks. Given dense (non-delta) input every
column is occupied, the cap keeps the first ``max_variants`` columns, and the
model degenerates to reading the head of each block — the constructor cannot
detect that, so ``forward`` checks occupancy on the first batch and warns.
"""
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .net import NO_LOCUS, KeyedTokenNorm, parse_block_key

# --- the token feature vector ----------------------------------------------
# One fixed layout, whatever subset of modalities a run loads: a modality that
# is absent simply leaves its slot zero. Dead input dims cost 128 weights each
# in `tok_proj` and buy a token vector whose meaning does not change between
# cells, which is what makes two runs' attention maps comparable.
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

# Which blocks share a coordinate system, and how their index maps onto it.
# The unit is CODONS throughout so the streams are roughly commensurable:
#   nt   dna, at alignment-column resolution           coord = col/3 - offset[locus]
#   aa   protein + biophysical, at codon resolution    coord = k
#   reg  the promoter window, upstream of the CDS      coord = (q - L)/3   (negative)
#
# "Roughly" is doing real work and is the one approximation in this model.
# `datasets/protein.py` translates the CDS window after DEGAPPING each isolate,
# so codon k is the k-th codon of that isolate's own degapped CDS, while the DNA
# block stays in shared alignment-column space. The two agree exactly when no
# indel sits upstream of the variant — the overwhelming majority of a clonal
# cohort — and drift by the indel length when one does. We therefore do NOT fuse
# nt and aa tokens by position: they stay separate tokens that the within-locus
# attention may pair, and `coord_offset` is a LEARNED per-locus scalar so the
# constant part of the offset (where the CDS starts inside the aligned record —
# katG's is 2,223 bp of CDS inside a 2,488 bp record) is fitted rather than
# assumed.
STREAMS = {"dna": "nt", "protein": "aa", "biophysical": "aa", "regulatory": "reg"}

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
    #    order (deterministic, and one-hot columns have no norm to rank by).
    "max_variants": 16, "pos_dims": 64, "uncovered_frac": 0.5,
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


def _sinusoid(coord, dims):
    """(..., ) continuous coordinate -> (..., dims) sinusoidal encoding.

    Continuous rather than table-indexed on purpose: the coordinate is in codon
    units and a DNA column lands on k, k+1/3 or k+2/3, so the fractional part
    carries the codon phase; and a learned table would need one row per column
    per locus (1,355 x 128 x 19 = 3.3M parameters) to say what 0 parameters say
    here. Wavelengths run 2*pi to 2*pi*10^4 codons, which resolves a 1/3-codon
    step at the top and spans the longest locus at the bottom.
    """
    half = dims // 2
    freqs = torch.exp(-math.log(10000.0)
                      * torch.arange(half, device=coord.device, dtype=coord.dtype) / half)
    ang = coord.unsqueeze(-1) * freqs
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


def _select_variants(occ, k):
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


class LocusFusionNet(nn.Module):
    """Variant-token transformer: fuse modalities within a locus, then loci.

    Pipeline, per isolate::

        per locus L, per coordinate stream (nt / aa / reg):
            delta block (B, C, len)  --occupancy--> the columns that differ
                                     --gather-->    <= max_variants tokens
        token = tok_proj(feature slots + flags)
              + pos_proj(sinusoid(coord))
              + locus_emb[L]  (+ FiLM adapter[L])
        [WT]_L = wt_emb[L] + wt_proj(variant count, coverage, uncovered)

        stage 1:  TransformerEncoder over {[WT]_L, variants of L}   -> z_L = out[[WT]]
        stage 2:  TransformerEncoder over {z_L for every locus}     -> fused
        readout:  one learned query per drug, cross-attending the fused set

    Same ``forward(xs)`` contract as every other arch — a list of (B, C_i, L_i)
    block tensors in ``branch_specs`` order — so the existing trainers drive it
    unchanged. ``block_keys`` are ``(modality, locus)`` pairs; ``from_blocks``
    takes them straight off the loader's FeatureBlocks.
    """

    bio_input = "blocks"        # forward takes the list of block tensors as-is

    def __init__(self, block_keys, branch_specs, n_drugs=1, drug_names=None,
                 d_model=128, nhead=4, dropout=0.1,
                 enc_layers=2, enc_dim_ff=256,
                 fusion_layers=2, fusion_dim_ff=256,
                 max_variants=16, pos_dims=64, uncovered_frac=0.5,
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
        unknown = sorted({m for m, _ in block_keys} - set(SLOTS))
        if unknown:
            raise ValueError(f"LocusFusionNet has no feature slot for modalities "
                             f"{unknown}; known: {sorted(SLOTS)}")

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
        # every locus contributes [WT] + max_variants per stream; the token axis
        # is padded to the widest locus so all loci run through stage 1 in ONE
        # batched call instead of a python loop over 19 transformers.
        self.max_variants = int(max_variants)
        self.tokens_per_locus = 1 + self.max_variants * max(len(p) for p in plan)
        self.uncovered_frac = float(uncovered_frac)
        self.carry_variants = min(int(carry_variants), self.tokens_per_locus - 1)

        # --- token embedding -------------------------------------------------
        n_loci = len(self.loci)
        self.tok_proj = nn.Linear(C_TOK, d_model)
        self.pos_proj = nn.Linear(pos_dims, d_model)
        self.wt_proj = nn.Linear(3, d_model)         # count / coverage / uncovered
        self.locus_emb = nn.Embedding(n_loci, d_model)
        self.wt_emb = nn.Embedding(n_loci, d_model)
        # the constant part of the nt->codon offset: where the CDS starts inside
        # the aligned record. Learned rather than read off `datasets.cds`, which
        # would mean plumbing a column offset through FeatureBlock for a number
        # the model can fit from 19 scalars.
        self.coord_offset = nn.Parameter(torch.zeros(n_loci))
        self.tok_norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.locus_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.wt_emb.weight, std=0.02)
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

    @classmethod
    def from_blocks(cls, blocks, **kwargs):
        """Build straight from loader FeatureBlocks (``data.blocks``)."""
        return cls([parse_block_key(b.name) for b in blocks],
                   [b.spec() for b in blocks], **kwargs)

    # -- tokenization --------------------------------------------------------

    def _stream_tokens(self, xs, li, stream, members, length):
        """One coordinate stream of one locus -> (features, coord, valid, n_occ).

        `members` may hold several blocks (protein + biophysical are co-indexed
        by construction — same translate path, same k_max), in which case the
        occupancy is their UNION and each block writes into its own feature
        slot at the same selected columns. That is the gene-level fusion: one
        token per changed residue carrying every modality's view of it.
        """
        blocks = [xs[i] for i, _ in members]
        B = blocks[0].shape[0]
        occ = torch.zeros(B, length, dtype=torch.bool, device=blocks[0].device)
        for x in blocks:
            occ |= x.abs().sum(1) > 0
        idx, valid, n_occ = _select_variants(occ, self.max_variants)
        k = idx.shape[1]

        feat = blocks[0].new_zeros(B, k, C_TOK)
        gather = idx.unsqueeze(-1)
        for x, (_i, modality) in zip(blocks, members):
            lo, hi = SLOTS[modality]
            feat[:, :, lo:hi] = x.transpose(1, 2).gather(
                1, gather.expand(-1, -1, x.shape[1]))

        pos = idx.to(feat.dtype)
        if stream == "aa":
            feat[:, :, F_IS_AA] = 1.0
            coord = pos
        else:
            flag = F_IS_NT if stream == "nt" else F_IS_REG
            feat[:, :, flag] = 1.0
            # the '-' channel of the nucleotide one-hot: an alignment gap where
            # the reference has a base, i.e. a deletion or a coverage hole. The
            # model needs to be able to tell those from substitutions.
            lo, _hi = SLOTS["dna" if stream == "nt" else "regulatory"]
            feat[:, :, F_GAP] = feat[:, :, lo + GAP_CHANNEL]
            feat[:, :, F_PHASE:F_PHASE + 3] = F.one_hot(
                (idx % 3), num_classes=3).to(feat.dtype)
            coord = pos / 3.0
            if stream == "nt":
                coord = coord - self.coord_offset[li]
            else:
                coord = coord - length / 3.0          # upstream: negative codons
        feat = feat * valid.unsqueeze(-1).to(feat.dtype)
        coord = coord * valid.to(coord.dtype)
        return feat, coord, valid, n_occ, length

    def _locus_tokens(self, xs):
        """-> (feat, coord, valid) each shaped (B, n_loci, tokens_per_locus, ...).

        Slot 0 of every locus is its [WT] sentinel and is always valid, so no
        stage-1 attention row is ever fully masked.
        """
        B = xs[0].shape[0]
        dev, dt = xs[0].device, xs[0].dtype
        T = self.tokens_per_locus
        n_loci = len(self.loci)
        feat = torch.zeros(B, n_loci, T, C_TOK, device=dev, dtype=dt)
        coord = torch.zeros(B, n_loci, T, device=dev, dtype=dt)
        valid = torch.zeros(B, n_loci, T, dtype=torch.bool, device=dev)
        stats = torch.zeros(B, n_loci, 3, device=dev, dtype=dt)

        for li, streams in enumerate(self._plan):
            cursor = 1
            n_var = torch.zeros(B, device=dev, dtype=dt)
            frac = torch.zeros(B, device=dev, dtype=dt)
            for stream, members, length in streams:
                f, c, v, n_occ, length = self._stream_tokens(xs, li, stream, members, length)
                k = f.shape[1]
                feat[:, li, cursor:cursor + k] = f
                coord[:, li, cursor:cursor + k] = c
                valid[:, li, cursor:cursor + k] = v
                cursor += k
                n_var = n_var + n_occ.to(dt)
                frac = torch.maximum(frac, n_occ.to(dt) / length)
            uncovered = (frac > self.uncovered_frac).to(dt)
            # An all-gap record differs from the reference at EVERY column, so
            # without this flag a locus that simply failed to assemble reads as
            # a hypervariant one. 14-91 isolates per locus are in that state
            # (README census); they are not wild type and must not look it.
            feat[:, li, :, F_UNCOVERED] = uncovered.unsqueeze(-1)
            stats[:, li, 0] = torch.log1p(n_var)
            stats[:, li, 1] = frac
            stats[:, li, 2] = uncovered
            valid[:, li, 0] = True
            feat[:, li, 0, F_IS_WT] = 1.0
        return feat, coord, valid, stats

    def _check_occupancy(self, valid):
        """Warn once if the input does not look delta-encoded.

        A dense one-hot occupies every column, the cap then keeps the first
        `max_variants` of them, and the model silently becomes "read the head of
        each block" — a failure that produces plausible numbers, so it is worth
        a loud warning rather than a docstring."""
        self._occupancy_checked = True
        filled = valid[:, :, 1:].to(torch.float32).mean().item()
        if filled > 0.9:
            warnings.warn(
                f"LocusFusionNet: {100 * filled:.0f}% of variant slots are occupied. "
                "This architecture expects reference-difference input (--delta); "
                "on a plain one-hot every column differs from nothing and the "
                "tokenizer just keeps the first max_variants columns of each "
                "block. Re-run with --delta.", stacklevel=3)

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
        tok[:, :, 0] = tok[:, :, 0] + self.wt_emb(li).unsqueeze(0) + self.wt_proj(stats)
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

        ``[{locus, stream, modalities, columns (B, k), valid (B, k),
        n_variants (B,), uncovered (B,)}]`` — the token set spelled out, so an
        attention map from ``forward(..., return_attn=True)`` can be read as
        "drug j attended to katG alignment column 944" rather than to token 7.
        """
        out = []
        for li, streams in enumerate(self._plan):
            for stream, members, length in streams:
                blocks = [xs[i] for i, _ in members]
                occ = torch.zeros(blocks[0].shape[0], length, dtype=torch.bool,
                                  device=blocks[0].device)
                for x in blocks:
                    occ |= x.abs().sum(1) > 0
                idx, valid, n_occ = _select_variants(occ, self.max_variants)
                out.append({
                    "locus": self.loci[li], "stream": stream,
                    "modalities": [m for _, m in members],
                    "columns": idx, "valid": valid, "n_variants": n_occ,
                    "uncovered": (n_occ.float() / length) > self.uncovered_frac,
                })
        return out
