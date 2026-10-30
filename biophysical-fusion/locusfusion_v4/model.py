"""
LocusFusion v4: v3's two-stage attention over v4's pre-built tokens.

The network is v3's (models/locusfusion_v2.py, `LocusFusionV2Net`) with the
tokenizer taken out -- tokens arrive ready-made from tokenizer.TokenSet:

    per locus L:
        token = tok_proj(26 base channels | site proximity per drug)
              + pos_proj(sinusoid(codons from the start codon))
              + locus_emb[L]   (+ FiLM adapter[L])
        [NULL]_L = null_emb[L] + null_proj(log1p(changed columns), occupied fraction)
    stage 1:  TransformerEncoder over {[NULL]_L, variants of L} -> z_L = out[[NULL]]
    stage 2:  TransformerEncoder over {z_L for every locus}     -> fused
    readout:  one learned query per drug, cross-attending the fused set

**Site proximity, per drug.** A token carries one channel per drug in
`dist_drugs`: exp(-d / 10 A) for its distance d to that drug's binding site,
-1 where there is none (promoter, a locus with no site for that drug, a base
with no residue). A single-drug model reads one channel, its own drug's; the
multi-drug model reads all eleven, so a token says how close it is to every
drug's site and drug j's query can weigh channel j. The distances are a table
lookup (`prox`, a buffer built from binding_sites/distances.npz), indexed by
the token's `site_pos`; nothing about them is learned or per isolate. [NULL]
reads -1 on every proximity channel.

Defaults are v3's (654,977 parameters at 45 input channels; v4's narrower
token changes only tok_proj).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.locusfusion_v2 import LOCUS_ENCODERS, SUMMARY_NORMS, _sinusoid
from models.net import KeyedTokenNorm

from .layout import C_BASE, DIST_LENGTH, DIST_MISSING, SLOTS

V4_DEFAULTS = {
    "d_model": 128, "nhead": 4, "dropout": 0.1,
    "enc_layers": 2, "enc_dim_ff": 256,
    "fusion_layers": 2, "fusion_dim_ff": 256,
    "pos_dims": 32, "locus_encoder": "adapter", "summary_norm": "keyed",
    "hidden": 256,
}


def proximity_table(dist_drugs, loci, distances, length=DIST_LENGTH):
    """(n_dist, n_loci, P + 1) float32: exp(-d / length) per drug / locus /
    position, DIST_MISSING where there is no distance. Column P is the
    'no position' slot a token with site_pos -1 reads."""
    P = max([len(d) for d in distances.values()] + [1])
    tab = np.full((len(dist_drugs), len(loci), P + 1), DIST_MISSING, np.float32)
    for i, drug in enumerate(dist_drugs):
        for j, locus in enumerate(loci):
            d = distances.get((drug, locus))
            if d is not None:
                tab[i, j, :len(d)] = np.where(np.isfinite(d), np.exp(-d / length), DIST_MISSING)
    return tab


class LocusFusionV4Net(nn.Module):
    def __init__(self, loci, kept, drugs, prox=None, dist_drugs=(), max_variants=16,
                 c_base=C_BASE,
                 d_model=128, nhead=4, dropout=0.1, enc_layers=2, enc_dim_ff=256,
                 fusion_layers=2, fusion_dim_ff=256, pos_dims=32,
                 locus_encoder="adapter", summary_norm="keyed", hidden=256, out_bias=None):
        super().__init__()
        if locus_encoder not in LOCUS_ENCODERS:
            raise ValueError(f"locus_encoder must be one of {LOCUS_ENCODERS}")
        if summary_norm not in SUMMARY_NORMS:
            raise ValueError(f"summary_norm must be one of {SUMMARY_NORMS}")
        if d_model % nhead or pos_dims % 2:
            raise ValueError("d_model must divide by nhead and pos_dims must be even")
        self.loci, self.drugs, self.dist_drugs = list(loci), list(drugs), list(dist_drugs)
        n_loci, n_drugs, n_dist = len(self.loci), len(self.drugs), len(self.dist_drugs)
        if prox is None:
            prox = np.full((n_dist, n_loci, 1), DIST_MISSING, np.float32)
        prox = torch.as_tensor(np.asarray(prox, np.float32))
        if tuple(prox.shape[:2]) != (n_dist, n_loci):
            raise ValueError(f"prox is {tuple(prox.shape)}, want ({n_dist}, {n_loci}, P+1)")
        self.register_buffer("prox", prox)
        self.register_buffer("kept", torch.as_tensor(np.asarray(kept, np.float32)).clamp_min(1))
        self.max_variants, self.pos_dims = int(max_variants), int(pos_dims)
        self.c_base = int(c_base)              # C_BASE, or C_BASE_RES with --keep-residue
        self.c_tok = self.c_base + n_dist
        self.config = dict(loci=self.loci, drugs=self.drugs, dist_drugs=self.dist_drugs,
                           max_variants=self.max_variants, c_base=self.c_base,
                           d_model=d_model, nhead=nhead,
                           dropout=dropout, enc_layers=enc_layers, enc_dim_ff=enc_dim_ff,
                           fusion_layers=fusion_layers, fusion_dim_ff=fusion_dim_ff,
                           pos_dims=pos_dims, locus_encoder=locus_encoder,
                           summary_norm=summary_norm, hidden=hidden, out_bias=out_bias)

        self.tok_proj = nn.Linear(self.c_tok, d_model)
        self.pos_proj = nn.Linear(pos_dims, d_model)
        self.null_proj = nn.Linear(2, d_model)
        self.locus_emb = nn.Embedding(n_loci, d_model)
        self.null_emb = nn.Embedding(n_loci, d_model)
        self.tok_norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.locus_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.null_emb.weight, std=0.02)

        self.locus_encoder = locus_encoder
        if locus_encoder == "adapter":
            self.film_scale = nn.Parameter(torch.zeros(n_loci, d_model))
            self.film_shift = nn.Parameter(torch.zeros(n_loci, d_model))
        else:
            self.film_scale = self.film_shift = None

        def encoder(layers, ff):
            return nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, nhead, ff, dropout=dropout,
                                           batch_first=True, norm_first=True),
                layers, enable_nested_tensor=False)
        self.encoders = nn.ModuleList(encoder(enc_layers, enc_dim_ff) for _ in
                                      range(n_loci if locus_encoder == "per_locus" else 1))
        self.summ_norm = KeyedTokenNorm(n_loci, d_model) if summary_norm == "keyed" else None
        self.fusion = encoder(fusion_layers, fusion_dim_ff)

        self.drug_queries = nn.Parameter(torch.zeros(n_drugs, d_model))
        nn.init.trunc_normal_(self.drug_queries, std=0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc_out = nn.Linear(hidden, 1)
        if out_bias is not None:
            with torch.no_grad():
                self.fc_out.bias.fill_(float(out_bias))

    def tokens(self, feat, coord, valid, site_pos):
        """Batch tensors -> (x, coord, valid) with the [NULL] slot prepended and
        the per-drug proximity channels appended: (B, L, 1 + V, c_tok)."""
        B, L, V, _ = feat.shape
        dev, dt = feat.device, feat.dtype
        if self.dist_drugs:
            P = self.prox.shape[-1] - 1
            idx = site_pos.long()
            idx = torch.where(idx < 0, torch.full_like(idx, P), idx.clamp(max=P))
            tbl = self.prox.permute(1, 2, 0)                          # (L, P+1, n_dist)
            li = torch.arange(L, device=dev).view(1, L, 1).expand(B, L, V)
            dist = tbl[li, idx].to(dt)                               # (B, L, V, n_dist)
            feat = torch.cat([feat, dist], -1)
        null = torch.zeros(B, L, 1, self.c_tok, device=dev, dtype=dt)
        null[..., SLOTS["is_null"][0]] = 1.0
        null[..., self.c_base:] = DIST_MISSING
        x = torch.cat([null, feat], 2)
        # padding slots are masked out of attention; zero them so nothing leaks
        x[:, :, 1:] = x[:, :, 1:] * valid.unsqueeze(-1).to(dt)
        coord = torch.cat([torch.zeros(B, L, 1, device=dev, dtype=dt), coord * valid.to(dt)], 2)
        valid = torch.cat([torch.ones(B, L, 1, device=dev, dtype=torch.bool), valid], 2)
        return x, coord, valid

    def forward(self, feat, coord, valid, site_pos, n_occ, return_attn=False):
        x, coord, valid = self.tokens(feat, coord, valid, site_pos)
        B, L, T, _ = x.shape
        li = torch.arange(L, device=x.device)
        n_occ = n_occ.to(x.dtype)
        stats = torch.stack([torch.log1p(n_occ), n_occ / self.kept], -1)       # (B, L, 2)

        tok = self.tok_proj(x) + self.pos_proj(_sinusoid(coord, self.pos_dims))
        tok = tok + self.locus_emb(li).view(1, L, 1, -1)
        tok[:, :, 0] = tok[:, :, 0] + self.null_emb(li).unsqueeze(0) + self.null_proj(stats)
        if self.film_scale is not None:
            tok = tok * (1.0 + self.film_scale.view(1, L, 1, -1)) + self.film_shift.view(1, L, 1, -1)
        tok = self.tok_norm(tok) * valid.unsqueeze(-1).to(tok.dtype)

        pad = ~valid
        if self.locus_encoder == "per_locus":
            z = torch.stack([enc(tok[:, i], src_key_padding_mask=pad[:, i])
                             for i, enc in enumerate(self.encoders)], 1)
        else:
            z = self.encoders[0](tok.reshape(B * L, T, -1),
                                 src_key_padding_mask=pad.reshape(B * L, T)).reshape(B, L, T, -1)
        fused = z[:, :, 0]
        if self.summ_norm is not None:
            fused = self.summ_norm(fused, li)
        fused = self.fusion(fused)

        q = self.drug_queries.unsqueeze(0).expand(B, -1, -1)
        pooled, attn = self.pool_attn(q, fused, fused, need_weights=return_attn)
        logits = self.fc_out(F.relu(self.fc1(self.norm(pooled)))).squeeze(-1)   # (B, n_drugs)
        return (logits, attn) if return_attn else logits
