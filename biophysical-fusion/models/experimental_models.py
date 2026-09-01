"""
Experimental architectures — the aggregation question, not the encoding question.

Everything in ``net.py`` and ``locusfusion.py`` answers "how do I read a genomic
block". These six answer a different one, and they exist because of what the
project's own measurements say the problem actually is.

## The reframing

Censused over 19 loci x 17,943 isolates against each alignment's ``MT_H37Rv``
row, **the median isolate differs from the reference at 0-3 columns of a 2.5 kb
gene, and at 14 columns across all 19 loci.** Once you tokenize those
differences (which ``LocusFusionNet`` does), there is no long sequence left to
encode. The task is:

> given a set of ~14 deviations from wild type, most of them neutral lineage
> markers, decide R/S — from 11.5 k labelled training isolates.

That is **sparse evidence aggregation**, not sequence encoding. CNN /
transformer / SSM are all answers to the encoding question. The aggregator is
what is left, and softmax attention is a specifically bad choice for it:

**Softmax normalises, so it is a RELATIVE selector.** With one informative token
and thirteen neutral ones the weights must sum to 1, so it cannot say "this
token, absolutely, whatever else is present" — it has to spend mass on the
neutral tokens. That is the mechanism behind the flat 1/8 attention
`results/experiments/token_signal` measured; it is a structural property of the
operator, not a training failure. A needle detector wants an **absolute,
monotone** aggregator: adding a neutral token must not dilute the signal.

So: same tokenizer, six different aggregators, deliberately far apart.

| `--arch` | aggregator | the question it asks |
|---|---|---|
| `catalogue` | learned scalar per exact variant id | how far does pure memorisation get? (= logistic regression on the variant matrix) |
| `additive` | sum of `w(features)` | does featurising the variant buy generalisation to substitutions never seen in training? |
| `noisyor` | `1 - prod(1 - p_v)` | "susceptible unless something confers resistance", as an architecture |
| `gatedpool` | sigmoid gate, no softmax | is normalisation the thing that broke attention here? |
| `deepsets` | sum + max + count | does attention buy anything at all over plain additivity? |
| `fm` | factorization machine | is epistasis worth anything, priced at O(T*k) instead of O(T^2)? |

`catalogue` and `additive` are a matched pair and the difference between them is
a measurement: `catalogue` **cannot** score a variant it never saw (its weight
stays at its zero init), `additive` scores it from position + substitution +
biophysical change. That difference is exactly the *pncA* / PYRAZINAMIDE
mechanism — hundreds of distinct inactivating substitutions, most unseen in
training — which is where the project's biophysical gain comes from.

## The baseline that is not a network

Before any of this, `variant_design_matrix()` builds the sparse isolate x
variant matrix these models tokenize, so an L1-logistic or a gradient-boosted
tree can be fitted on it in a few lines and no GPU. There is no sparse-linear or
tree baseline anywhere in this project, and for TB AMR from a variant matrix
those are the canonical strong methods (TB-Profiler and Mykrobe are catalogue
lookups and are clinically competitive). `token_signal` already found plain
logistic regression on setfusion's own encoder output beating the trained model
by 0.011, which points straight here. If that baseline reaches ~0.92 it is the
most valuable result available, because it reframes the project.

## Shared substrate

All six consume the same **flat** variant token set — every block's variants
concatenated into one set per isolate, with no locus hierarchy. That is a
deliberate contrast with `LocusFusionNet`, which groups by locus and fuses in
two stages; here the whole point is to test aggregators, so the grouping is held
flat and constant across all six.

The token feature layout is imported from ``locusfusion`` rather than redefined
(``SLOTS``, ``C_TOK``, the flag offsets), so a contribution or gate from any of
these models is directly comparable to a `locusfusion` attention weight.

**All six require ``--delta`` and per-locus blocks**, for the same reason
`locusfusion` does: on dense input every column is occupied, the cap keeps the
first ``max_variants`` columns of each block, and the model silently degenerates
while still producing plausible numbers.
"""
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .locusfusion import (C_TOK, F_GAP, F_IS_AA, F_IS_NT, F_IS_REG, F_PHASE,
                          F_UNCOVERED, GAP_CHANNEL, NO_LOCUS, SLOTS, STREAMS,
                          _select_variants, _sinusoid, parse_block_key)

# Shared across every model here. Model-specific knobs live in each class's
# ``KNOBS`` and are rejected by the factory if you pass one to a model that does
# not use it — silently ignoring a knob would make a sweep arm look like it ran
# when it was really the control under a different folder name.
EXPERIMENTAL_DEFAULTS = {
    # the tokenizer, identical in meaning to LOCUSFUSION_DEFAULTS' so the two
    # families are comparable cell for cell
    "max_variants": 16, "pos_dims": 64, "uncovered_frac": 0.5,
    # the shared variant embedding every model but `catalogue` reads
    "d_model": 128, "dropout": 0.1,
    # model-specific (see each class's KNOBS)
    "fm_rank": 8, "residual_catalogue": False,
}


# ---------------------------------------------------------------------------
# Shared substrate: one flat variant token set per isolate
# ---------------------------------------------------------------------------

class VariantSet(nn.Module):
    """Delta blocks -> a flat, padded set of variant tokens. No parameters.

    Returns a dict, all tensors ``(B, T, ...)`` with ``T = n_blocks *
    max_variants``:

      ``feat``   (B, T, C_TOK)  the locusfusion feature layout
      ``valid``  (B, T)  bool   which slots are real variants
      ``vid``    (B, T)  long   EXACT variant identity, (block, column, alt),
                                offset into one global vocabulary
      ``coord``  (B, T)  float  codon coordinate (negative upstream)
      ``locus``  (B, T)  long
      ``modality``(B, T) long
      ``uncovered`` (B, n_blocks)  a block that differs from the reference
                                everywhere is a record that failed to assemble,
                                not a hypervariant one
      ``n_occ``  (B, n_blocks)  the TRUE variant count, before the cap

    ``vid`` is exact rather than hashed: the vocabulary is
    ``sum_b(L_b * C_b)`` — about 51 k for an isoniazid cell and 527 k for all 19
    loci with every modality — which is small enough to index directly, so
    `catalogue` has no collisions to explain away.
    """

    def __init__(self, block_keys, branch_specs, max_variants=16,
                 uncovered_frac=0.5):
        super().__init__()
        self.block_keys = [(m, l or NO_LOCUS) for m, l in block_keys]
        self.specs = [(int(c), int(length)) for c, length in branch_specs]
        self.max_variants = int(max_variants)
        self.uncovered_frac = float(uncovered_frac)
        self.loci = list(dict.fromkeys(l for _, l in self.block_keys))
        self.modalities = list(dict.fromkeys(m for m, _ in self.block_keys))
        self._locus_ix = {l: i for i, l in enumerate(self.loci)}
        self._mod_ix = {m: i for i, m in enumerate(self.modalities)}
        # one contiguous id range per block: [offset_b, offset_b + L_b*C_b)
        offsets, total = [], 0
        for c, length in self.specs:
            offsets.append(total)
            total += length * c
        self.register_buffer("_offsets", torch.tensor(offsets, dtype=torch.long),
                             persistent=False)
        self.vocab_size = total
        self.n_blocks = len(self.specs)
        self.tokens = self.n_blocks * self.max_variants
        self._occupancy_checked = False

    def forward(self, xs):
        if len(xs) != self.n_blocks:
            raise ValueError(f"expected {self.n_blocks} blocks, got {len(xs)}")
        B = xs[0].shape[0]
        dev, dt = xs[0].device, xs[0].dtype
        k = self.max_variants
        feat = torch.zeros(B, self.tokens, C_TOK, device=dev, dtype=dt)
        valid = torch.zeros(B, self.tokens, dtype=torch.bool, device=dev)
        vid = torch.zeros(B, self.tokens, dtype=torch.long, device=dev)
        coord = torch.zeros(B, self.tokens, device=dev, dtype=dt)
        locus = torch.zeros(B, self.tokens, dtype=torch.long, device=dev)
        modality = torch.zeros(B, self.tokens, dtype=torch.long, device=dev)
        uncov = torch.zeros(B, self.n_blocks, device=dev, dtype=dt)
        n_occ_all = torch.zeros(B, self.n_blocks, device=dev, dtype=dt)

        for b, (x, (mod, loc), (c, length)) in enumerate(
                zip(xs, self.block_keys, self.specs)):
            lo, hi = b * k, (b + 1) * k
            occ = x.abs().sum(1) > 0                          # (B, L)
            idx, v, n_occ = _select_variants(occ, k)
            kk = idx.shape[1]
            col = x.transpose(1, 2).gather(
                1, idx.unsqueeze(-1).expand(-1, -1, c))       # (B, kk, c)
            s0, s1 = SLOTS[mod]
            feat[:, lo:lo + kk, s0:s1] = col
            # exact identity: which column, and which channel carries the change
            alt = col.abs().argmax(-1)
            vid[:, lo:lo + kk] = self._offsets[b] + idx * c + alt
            stream = STREAMS[mod]
            if stream == "aa":
                feat[:, lo:lo + kk, F_IS_AA] = 1.0
                crd = idx.to(dt)
            else:
                feat[:, lo:lo + kk, F_IS_NT if stream == "nt" else F_IS_REG] = 1.0
                feat[:, lo:lo + kk, F_GAP] = col[:, :, GAP_CHANNEL].abs()
                feat[:, lo:lo + kk, F_PHASE:F_PHASE + 3] = F.one_hot(
                    idx % 3, num_classes=3).to(dt)
                crd = idx.to(dt) / 3.0
                if stream == "reg":
                    crd = crd - length / 3.0        # upstream: negative codons
            coord[:, lo:lo + kk] = crd * v.to(dt)
            valid[:, lo:lo + kk] = v
            locus[:, lo:lo + kk] = self._locus_ix[loc]
            modality[:, lo:lo + kk] = self._mod_ix[mod]
            n_occ_all[:, b] = n_occ.to(dt)
            u = (n_occ.to(dt) / length > self.uncovered_frac).to(dt)
            uncov[:, b] = u
            feat[:, lo:hi, F_UNCOVERED] = u.unsqueeze(-1)

        feat = feat * valid.unsqueeze(-1).to(dt)
        feat[:, :, F_UNCOVERED] = uncov.repeat_interleave(k, dim=1)
        vid = vid * valid.to(vid.dtype)
        if not self._occupancy_checked:
            self._occupancy_checked = True
            filled = valid.to(torch.float32).mean().item()
            if filled > 0.9:
                warnings.warn(
                    f"{100 * filled:.0f}% of variant slots are occupied. These "
                    "architectures expect reference-difference input (--delta); "
                    "on a plain one-hot the tokenizer just keeps the first "
                    "max_variants columns of each block.", stacklevel=3)
        return {"feat": feat, "valid": valid, "vid": vid, "coord": coord,
                "locus": locus, "modality": modality, "uncovered": uncov,
                "n_occ": n_occ_all}


class VariantEmbedding(nn.Module):
    """One variant token -> (B, T, d_model). Shared by every model but `catalogue`.

    ``feature projection + sinusoid(coordinate) + locus + modality``, then a
    LayerNorm, then zeroed at padded slots so a pooled sum is not polluted.
    The coordinate encoding is the parameter-free continuous one from
    ``locusfusion`` — a nucleotide column lands at k, k+1/3 or k+2/3, so the
    fractional part is the codon phase and exact position survives.
    """

    def __init__(self, n_loci, n_modalities, d_model=128, pos_dims=64):
        super().__init__()
        if pos_dims % 2:
            raise ValueError(f"pos_dims must be even, got {pos_dims}")
        self.feat_proj = nn.Linear(C_TOK, d_model)
        self.pos_proj = nn.Linear(pos_dims, d_model)
        self.locus_emb = nn.Embedding(n_loci, d_model)
        self.modality_emb = nn.Embedding(n_modalities, d_model)
        nn.init.trunc_normal_(self.locus_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.modality_emb.weight, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self.pos_dims = int(pos_dims)
        self.out_features = d_model

    def forward(self, v):
        h = (self.feat_proj(v["feat"])
             + self.pos_proj(_sinusoid(v["coord"], self.pos_dims))
             + self.locus_emb(v["locus"]) + self.modality_emb(v["modality"]))
        return self.norm(h) * v["valid"].unsqueeze(-1).to(h.dtype)


class _VariantModel(nn.Module):
    """Base: tokenizer + (optional) embedding + the trainer's contract.

    Subclasses implement ``aggregate(h, v) -> (B, n_drugs)`` logits. Everything
    else — ``forward(xs)``, ``from_blocks``, ``n_drugs``/``drug_names``,
    ``out_bias`` — is here so the six differ ONLY in the aggregator, which is the
    whole point of the family.
    """

    bio_input = "blocks"          # forward takes the list of block tensors as-is
    KNOBS = ()                    # model-specific knob names, checked by the factory
    needs_embedding = True        # False -> no VariantEmbedding is built at all
    uses_hidden = True            # False -> the model has no hidden layer to size

    def __init__(self, block_keys, branch_specs, n_drugs=1, drug_names=None,
                 out_bias=None, max_variants=16, pos_dims=64, uncovered_frac=0.5,
                 d_model=128, dropout=0.1):
        super().__init__()
        if not branch_specs:
            raise ValueError("needs at least one branch spec")
        if len(block_keys) != len(branch_specs):
            raise ValueError("block_keys must match branch_specs length "
                             f"({len(block_keys)} vs {len(branch_specs)})")
        if drug_names:
            n_drugs = len(drug_names)
        keys = [(m, l or NO_LOCUS) for m, l in block_keys]
        unknown = sorted({m for m, _ in keys} - set(SLOTS))
        if unknown:
            raise ValueError(f"no feature slot for modalities {unknown}; "
                             f"known: {sorted(SLOTS)}")
        if len({l for _, l in keys}) == 1 and keys[0][1] == NO_LOCUS:
            warnings.warn(
                f"{type(self).__name__}: every block keyed to <none> — you passed "
                "the merged per-modality blocks, so locus identity is a no-op. "
                "Load with per_modality_branch=False.", stacklevel=2)
        self.tok = VariantSet(keys, branch_specs, max_variants=max_variants,
                              uncovered_frac=uncovered_frac)
        self.emb = (VariantEmbedding(len(self.tok.loci), len(self.tok.modalities),
                                     d_model=d_model, pos_dims=pos_dims)
                    if self.needs_embedding else None)
        self.drop = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        # a missing gene is not a wild-type gene, and no other architecture in
        # this project distinguishes them. One weight per (block, drug).
        self.uncovered_w = nn.Parameter(torch.zeros(self.tok.n_blocks, n_drugs))
        self.bias = nn.Parameter(torch.zeros(n_drugs))
        if out_bias is not None:
            with torch.no_grad():
                self.bias.fill_(float(out_bias))
        self.n_drugs = int(n_drugs)
        self.drug_names = list(drug_names) if drug_names else None
        self.d_model = int(d_model)

    @classmethod
    def from_blocks(cls, blocks, **kwargs):
        return cls([parse_block_key(b.name) for b in blocks],
                   [b.spec() for b in blocks], **kwargs)

    def aggregate(self, h, v):
        raise NotImplementedError

    def forward(self, xs):
        v = self.tok(xs)
        h = self.drop(self.emb(v)) if self.emb is not None else None
        return self.aggregate(h, v) + v["uncovered"] @ self.uncovered_w


# ---------------------------------------------------------------------------
# 1. catalogue — pure memorisation. The control.
# ---------------------------------------------------------------------------

class CatalogueNet(_VariantModel):
    """``logit_j = bias_j + sum_v w[vid_v, j]``.

    A learned scalar per EXACT variant identity, summed. This is logistic
    regression on the isolate x variant presence matrix, expressed as a module
    so it trains in the existing loop against the existing protocol — and it is
    a WHO-style resistance catalogue, learned rather than curated.

    It is here as the **control for generalisation**, not as a contender. The
    weight table is zero-initialised, so a substitution that never appears in
    training contributes exactly nothing at test time: `catalogue` can only
    recognise variants it has already seen. `AdditiveVariantNet` is the same
    aggregator with `w` computed from the variant's FEATURES instead of its id,
    so the gap between the two is a direct measurement of what featurisation
    buys — which is the *pncA* loss-of-function question (hundreds of distinct
    inactivating substitutions, most unseen in training).

    Zero init also means an untrained model is exactly the base rate, and the
    contribution of every variant is readable off the table by name.
    """

    needs_embedding = False
    uses_hidden = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w = nn.Embedding(self.tok.vocab_size, self.n_drugs)
        nn.init.zeros_(self.w.weight)

    def aggregate(self, _h, v):
        w = self.w(v["vid"]) * v["valid"].unsqueeze(-1).to(self.bias.dtype)
        return self.bias + w.sum(1)

    @torch.no_grad()
    def catalogue_entries(self, top=25, drug=0):
        """The learned catalogue, largest effect first: ``[(vid, weight), ...]``.

        Pair with ``VariantSet``'s id arithmetic (``vid - offset[b]`` divided by
        the block's channel count gives the alignment column) to read an entry as
        "katG column 944 -> +2.1 toward RESISTANT".
        """
        w = self.w.weight[:, drug]
        val, ix = w.abs().topk(min(top, w.numel()))
        return [(int(i), float(w[i])) for i in ix]


# ---------------------------------------------------------------------------
# 2. additive — featurised additive variant-effect model.
# ---------------------------------------------------------------------------

class AdditiveVariantNet(_VariantModel):
    """``logit_j = bias_j + sum_v w_j(features_v)``, ``w = MLP(embedding)``.

    The same additive aggregator as `catalogue`, but the per-variant weight is
    computed from **what the variant is** — locus, exact position, which base or
    residue it became, and the biophysical consequence — rather than looked up by
    identity. So a substitution never seen in training still gets a weight, from
    its features.

    That is the whole argument for this model. The strongest biophysical result
    in the project is *pncA* loss of function for PYRAZINAMIDE, where hundreds of
    distinct inactivating substitutions exist and most are absent from training;
    "does this substitution break the protein" generalises where memorising
    positions cannot. This is the smallest architecture that can express it.

    Three further properties, all consequences of staying additive:

    * **It is exactly interpretable, with no attribution machinery.**
      ``contributions(xs)`` returns the signed contribution of every token, and
      they sum to the logit. No SHAP, no sampling error, no convergence question
      — which matters given finding 3 (attribution share badly mis-ranks
      predictive value).
    * **A susceptible isolate is the empty sum**, so the model's prior for "no
      deviations from H37Rv" is the learned bias, i.e. the base rate.
    * **It forbids epistasis by construction.** That is a real restriction — see
      `FactorizedInteractionNet` for the cheapest way to relax it — and it is
      also what makes it sample-efficient at 11.5 k training isolates.

    ``residual_catalogue=True`` adds `catalogue`'s exact-identity table on top,
    giving a hybrid that memorises the variants it has seen and featurises the
    ones it has not. That is likely the strongest single model in this file, and
    it is off by default so the clean measurement comes first.
    """

    KNOBS = ("residual_catalogue",)

    def __init__(self, *args, residual_catalogue=False, hidden=256, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, hidden), nn.ReLU(),
            nn.Linear(hidden, self.n_drugs))
        # zero the last layer so an untrained model sits exactly at the base
        # rate and every variant starts with no effect — same convention as
        # `catalogue`, which keeps the two comparable at init
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.residual_catalogue = bool(residual_catalogue)
        if self.residual_catalogue:
            self.w = nn.Embedding(self.tok.vocab_size, self.n_drugs)
            nn.init.zeros_(self.w.weight)
        else:
            self.w = None

    def _per_token(self, h, v):
        w = self.mlp(h)
        if self.w is not None:
            w = w + self.w(v["vid"])
        return w * v["valid"].unsqueeze(-1).to(w.dtype)

    def aggregate(self, h, v):
        return self.bias + self._per_token(h, v).sum(1)

    @torch.no_grad()
    def contributions(self, xs):
        """``(B, T, n_drugs)`` signed per-variant contributions to the logit.

        They sum to ``logit - bias - uncovered term`` exactly — an additive model
        needs no attribution method, it *is* one.
        """
        v = self.tok(xs)
        return self._per_token(self.emb(v), v)


# ---------------------------------------------------------------------------
# 3. noisyor — multiple-instance learning. "Susceptible unless."
# ---------------------------------------------------------------------------

class NoisyOrVariantNet(_VariantModel):
    """``P(R) = 1 - (1 - p_0) * prod_v (1 - p_v)``, ``p_v = sigmoid(MLP(h_v))``.

    The biology as an architecture. Resistance is conferred if **any** variant
    confers it, which is a noisy-OR over independent causes and is exactly the
    hypothesis "a susceptible isolate is the one with nothing in it, and anything
    else is resistant". A wild-type isolate has an empty product and falls back
    to the learned background ``p_0`` — the base rate.

    Why this rather than a sum:

    * **It is absolute and monotone.** Every ``p_v`` is judged on its own; adding
      a neutral variant cannot dilute a resistance variant the way a softmax
      must. That is the property softmax attention structurally cannot have.
    * **It saturates.** Two independent resistance mutations do not give twice
      the logit, they give "resistant, twice over" — which is what an additive
      model gets wrong when an isolate carries several.
    * ``p_v`` reads directly as **"probability this variant confers resistance"**,
      comparable against the WHO catalogue's own confidence gradings.

    **The label convention is load-bearing here, and the first version got it
    wrong.** This project encodes **R=0, S=1** (`training/multimodal._metrics`:
    ``n_R = (y == 0).sum()``), so a higher logit means *more likely
    SUSCEPTIBLE*. A noisy-OR is monotone increasing in its evidence, so if it is
    pointed at P(resistant) it is structurally forced to push every isolate that
    carries variants toward the wrong class — and since resistant isolates carry
    more variants, it comes out anti-predictive. Measured, before the fix:
    macro CV **0.4956**, below chance, with train loss moving 0.3152 -> 0.3023
    over 99 epochs against `additive`'s 0.2246 -> 0.0875 on the same cell.

    So the product is the **susceptible** probability, which is what "susceptible
    unless something confers resistance" actually says::

        log P(S) = logsigmoid(-z_0) + sum_v logsigmoid(-z_v)  =  S
        logit    = S - log(-expm1(S))

    Computed in log space throughout, so nothing ever evaluates ``1 - p`` in
    float. Anything reusing this class under a P(resistant) convention must
    negate the return value.

    Two restrictions, both structural and both worth stating before the run:

    * **It is monotone increasing in evidence.** Every factor is in (0, 1), so
      adding a variant can only raise P(R) — this model literally cannot learn a
      protective variant. For resistance that is the right prior (resistance is
      conferred, susceptibility is its absence), but a lineage marker that
      happens to correlate with susceptibility is invisible to it.
    * **It assumes the causes are independent**, so it cannot express
      compensatory epistasis (rpoB + rpoC) or dose-dependence — two resistance
      mutations are "resistant", not "more resistant".
    """

    def __init__(self, *args, hidden=256, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, hidden), nn.ReLU(),
            nn.Linear(hidden, self.n_drugs))
        # Start every variant near p ~ 0 so an untrained model is the background
        # rate alone and training has to argue a variant up -- but only NEAR.
        # d/dz logsigmoid(-z) = -sigmoid(z), which at z=-4 attenuates every
        # gradient into the MLP by ~50x, and on top of BIG-TB's lr = exp(-9) =
        # 1.2e-4 that is what left the first run stuck at its initial loss.
        # -2.0 keeps p_v ~ 0.12 at init, a 7x weaker attenuation.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, -2.0)

    def aggregate(self, h, v):
        z = self.mlp(h)                                        # (B, T, n_drugs)
        # log(1 - p_v), zeroed where the slot is padding (log(1-0) = 0)
        log_not = F.logsigmoid(-z) * v["valid"].unsqueeze(-1).to(z.dtype)
        # `bias` is the background z_0: the wild-type log-odds
        S = F.logsigmoid(-self.bias) + log_not.sum(1)          # log P(susceptible)
        S = S.clamp(max=-1e-6)                                 # keep -expm1 > 0
        # R=0 / S=1, so the logit is for the SUSCEPTIBLE class: every variant can
        # only lower it. See the class docstring -- the opposite sign is what
        # made this the one architecture in the family to score below chance.
        return S - torch.log(-torch.expm1(S))


# ---------------------------------------------------------------------------
# 4. gatedpool — attention with the softmax removed.
# ---------------------------------------------------------------------------

class GatedPoolNet(_VariantModel):
    """``pooled_j = sum_v sigmoid(g_j(h_v)) * V(h_v)``, plus a max branch.

    The minimal edit to the thing that failed. `setfusion` and `locusfusion`
    read out with a softmax attention query; this keeps the per-drug gate and
    deletes the normalisation, so the gate is an **absolute** relevance score
    rather than a share of a fixed budget. With one informative token among
    thirteen neutral ones a softmax must spend weight on the neutral tokens; a
    sigmoid gate can open for one and stay shut for the rest.

    The max branch answers "did any single variant fire hard", which is the
    needle question, while the gated sum answers "how much total evidence is
    there" — the two disagree exactly when an isolate carries several weak
    variants, and keeping both lets the head decide.

    Run this against `locusfusion` to attribute its result: if `gatedpool`
    closes a gap that `deepsets` does not, normalisation was the problem; if
    `deepsets` matches it, the gating was never doing anything.
    """

    def __init__(self, *args, hidden=256, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.d_model
        self.gate = nn.Linear(d, self.n_drugs)
        self.value = nn.Linear(d, d)
        self.norm = nn.LayerNorm(2 * d)
        self.head = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def aggregate(self, h, v):
        mask = v["valid"].unsqueeze(-1).to(h.dtype)
        g = torch.sigmoid(self.gate(h)) * mask                 # (B, T, n_drugs)
        val = self.value(h) * mask                             # (B, T, d)
        # (B, n_drugs, d): each drug pools the values under its own gate
        pooled = torch.einsum("btj,btd->bjd", g, val)
        gated = val.unsqueeze(2) * g.unsqueeze(-1)             # (B, T, n_drugs, d)
        gated = gated.masked_fill(~v["valid"][:, :, None, None], float("-inf"))
        peak = gated.max(dim=1).values                         # (B, n_drugs, d)
        peak = torch.nan_to_num(peak, neginf=0.0)              # all-wild-type isolate
        x = self.norm(torch.cat([pooled, peak], dim=-1))
        return self.bias + self.head(x).squeeze(-1)

    @torch.no_grad()
    def gates(self, xs):
        """``(B, T, n_drugs)`` absolute relevance per variant — the un-normalised
        analogue of an attention map, and the thing to compare against
        `locusfusion`'s, which is forced to sum to 1."""
        v = self.tok(xs)
        return torch.sigmoid(self.gate(self.emb(v))) * v["valid"].unsqueeze(-1)


# ---------------------------------------------------------------------------
# 5. deepsets — no attention at all.
# ---------------------------------------------------------------------------

class DeepSetsVariantNet(_VariantModel):
    """``rho([sum phi(h_v), max phi(h_v), count])``.

    The ablation the project has never run: **does attention buy anything over
    plain permutation-invariant aggregation?** No gate, no query, no
    normalisation — just a per-token MLP, a sum, a max and the variant count,
    then a readout.

    It is the honest null model for this family. `setfusion` and `locusfusion`
    both spend most of their parameters on machinery whose job is to decide
    *which* tokens matter; if summing them and taking a max does as well, that
    machinery is not earning its place, and the finding is worth more than a
    fourth-decimal AUC gain would be.

    The count is included deliberately: "how many deviations from H37Rv does this
    isolate carry" is a real feature (lineage divergence, assembly quality) that
    a pure sum conflates with effect size, and it is the one thing the
    tokenizer's cap can distort — so it is fed in from ``n_occ``, the TRUE count
    before the cap.
    """

    def __init__(self, *args, hidden=256, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.d_model
        self.phi = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.rho = nn.Sequential(
            nn.LayerNorm(2 * d + 1), nn.Linear(2 * d + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, self.n_drugs))

    def aggregate(self, h, v):
        mask = v["valid"].unsqueeze(-1).to(h.dtype)
        p = self.phi(h) * mask
        summed = p.sum(1)
        peak = p.masked_fill(~v["valid"].unsqueeze(-1), float("-inf")).max(1).values
        peak = torch.nan_to_num(peak, neginf=0.0)
        count = torch.log1p(v["n_occ"].sum(1, keepdim=True))
        return self.bias + self.rho(torch.cat([summed, peak, count], dim=-1))


# ---------------------------------------------------------------------------
# 6. fm — pairwise interactions, priced linearly.
# ---------------------------------------------------------------------------

class FactorizedInteractionNet(_VariantModel):
    """Factorization machine over featurised variants.

    ``logit = bias + sum_v w_v + W2 . [ 0.5 * ((sum_v e_v)^2 - sum_v e_v^2) ]``

    The classical trick: the second term is **every pairwise interaction**
    ``sum_{u<v} <e_u, e_v>``, computed in ``O(T * rank)`` instead of ``O(T^2)``.
    So this asks "is epistasis worth anything here?" without paying attention's
    quadratic cost and without attention's normalisation problem.

    It is the right way to relax `additive`'s hard restriction. Compensatory
    resistance is real — *rpoC* mutations that restore fitness lost to *rpoB*
    RRDR mutations are the textbook case, and both loci are in the 19 — but the
    project's own finding 1 says the signal is dominated by a handful of specific
    positions, so interactions should be second-order. An FM prices them
    accordingly: ``rank`` extra dimensions per variant, nothing more.

    Factors are computed from the variant's FEATURES, not from a per-identity
    table, so an unseen substitution still participates in interactions — the
    same reason `additive` beats `catalogue` on unseen variants, applied one
    order up. A per-identity factor table would be
    ``vocab x rank`` = 4 M parameters at 19 loci, almost all of it never trained.
    """

    KNOBS = ("fm_rank",)
    uses_hidden = False           # first order + rank-k interactions, no MLP

    def __init__(self, *args, fm_rank=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.first = nn.Linear(self.d_model, self.n_drugs)
        self.factor = nn.Linear(self.d_model, int(fm_rank))
        self.second = nn.Linear(int(fm_rank), self.n_drugs)
        nn.init.zeros_(self.first.weight); nn.init.zeros_(self.first.bias)
        nn.init.zeros_(self.second.weight); nn.init.zeros_(self.second.bias)
        nn.init.normal_(self.factor.weight, std=0.01)
        self.fm_rank = int(fm_rank)

    def aggregate(self, h, v):
        mask = v["valid"].unsqueeze(-1).to(h.dtype)
        w = (self.first(h) * mask).sum(1)                      # first order
        e = self.factor(h) * mask                              # (B, T, rank)
        inter = 0.5 * (e.sum(1) ** 2 - (e ** 2).sum(1))        # all pairs, O(T*rank)
        return self.bias + w + self.second(inter)


# ---------------------------------------------------------------------------
# registry + the non-neural baseline
# ---------------------------------------------------------------------------

EXPERIMENTAL_MODELS = {
    "catalogue": CatalogueNet,
    "additive": AdditiveVariantNet,
    "noisyor": NoisyOrVariantNet,
    "gatedpool": GatedPoolNet,
    "deepsets": DeepSetsVariantNet,
    "fm": FactorizedInteractionNet,
}


def make_experimental(name, block_keys, branch_specs, **kwargs):
    """Build one experimental model, rejecting knobs it does not use.

    ``EXPERIMENTAL_DEFAULTS`` is the union over the whole family, so the callers
    pass it straight through and a member simply ignores the entries that are not
    its own. What is NOT ignored is a knob that has been **moved off its
    default** and belongs to a different member: that raises, for the same reason
    the setfusion and locusfusion flag groups are guarded — an arm that quietly
    ran as its own control is worse than a crashed job.

    So the rule is *changed-and-foreign*, not merely *foreign*. Passing the whole
    defaults dict to `--arch deepsets` is fine; passing ``fm_rank=16`` to it is
    an error naming `fm`.
    """
    if name not in EXPERIMENTAL_MODELS:
        raise ValueError(f"unknown experimental model {name!r}; "
                         f"choose from {sorted(EXPERIMENTAL_MODELS)}")
    cls = EXPERIMENTAL_MODELS[name]
    shared = {"max_variants", "pos_dims", "uncovered_frac", "d_model", "dropout"}
    allowed = shared | set(cls.KNOBS)
    bad = sorted(k for k, v in kwargs.items()
                 if k in EXPERIMENTAL_DEFAULTS and k not in allowed
                 and v != EXPERIMENTAL_DEFAULTS[k])
    if bad:
        owners = sorted(n for n, c in EXPERIMENTAL_MODELS.items()
                        if set(bad) & set(c.KNOBS))
        raise ValueError(f"--arch {name} does not use {bad}; "
                         f"{'those knobs belong to ' + str(owners) if owners else 'no model uses them'}")
    kwargs = {k: v for k, v in kwargs.items()
              if k not in EXPERIMENTAL_DEFAULTS or k in allowed}
    if not cls.needs_embedding:
        # `catalogue` reads a weight straight off the variant id, so it builds no
        # embedding and these would be dead kwargs rather than knobs
        for dead in ("d_model", "pos_dims", "dropout"):
            kwargs.pop(dead, None)
    if not cls.uses_hidden:
        kwargs.pop("hidden", None)
    return cls(block_keys, branch_specs, **kwargs)


@torch.no_grad()
def variant_design_matrix(xs, block_keys, branch_specs, max_variants=64):
    """The sparse isolate x variant matrix, for the baseline that is not a network.

    Returns ``(scipy.sparse.csr_matrix, feature_names)`` — one column per
    (block, alignment column, alt) actually observed, so an L1-logistic or a
    gradient-boosted tree can be fitted directly::

        X, names = variant_design_matrix(arrays, keys, specs)
        LogisticRegression(penalty="l1", solver="liblinear", C=0.1).fit(X[tr], y[tr])

    **Fit this before believing any of the networks above.** There is no
    sparse-linear or tree baseline anywhere in this project, and for TB AMR from
    a variant matrix they are the canonical strong methods — `token_signal`
    already found plain logistic regression on setfusion's own representations
    beating the trained model by 0.011. If this reaches ~0.92 it reframes the
    project, and that is worth more than any architecture here.

    ``max_variants`` is raised well above the training default on purpose: this
    is a one-off feature extraction, not a forward pass, so there is no reason to
    truncate the tail.
    """
    from scipy import sparse                                   # optional dependency

    tok = VariantSet([(m, l or NO_LOCUS) for m, l in block_keys], branch_specs,
                     max_variants=max_variants)
    v = tok(list(xs))
    vid, valid = v["vid"].cpu().numpy(), v["valid"].cpu().numpy()
    rows, cols = np.nonzero(valid)
    ids = vid[rows, cols]
    seen = {vid_: j for j, vid_ in enumerate(sorted(set(ids.tolist())))}
    X = sparse.csr_matrix(
        (np.ones(len(ids), dtype=np.float32),
         (rows, np.array([seen[i] for i in ids]))),
        shape=(valid.shape[0], len(seen)))
    offsets = tok._offsets.tolist()
    names = []
    for vid_ in sorted(seen):
        b = max(i for i, off in enumerate(offsets) if off <= vid_)
        c = tok.specs[b][0]
        rel = vid_ - offsets[b]
        mod, loc = tok.block_keys[b]
        names.append(f"{mod}:{loc}@{rel // c}={rel % c}")
    return X, names
