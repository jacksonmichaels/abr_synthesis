"""
GenomeFusion -- the canonical model file. Edit the network here.

Training imports it through genomefusion/model.py, so a change here is what the
next run trains. Partition, tokens and batching live in genomefusion/
(partition.py, tokenizer.py, data.py); the token layout is genomefusion/layout.py.

    train:     python -m genomefusion.train ...
    sweep:     python -m genomefusion.sbatch --arms gf19p --dry-run   (gf19p is the
               19-locus parity arm, the genome arm needs ~20 warmup epochs)
    tests:     python -m genomefusion.tests.test_model

Checkpoints store {"config": model.config, "state_dict": ...}; renaming or
adding a parameter here means old weights no longer load strictly.

GenomeFusion: LocusFusion over the whole genome.

Two transformer stages, as in v3-v5, but sparse. Only the loci an isolate
actually changes enter the network, and a locus holds as many tokens as the
isolate has changes in it.

  1. token embedding: linear C -> d, plus a sinusoid of the position within the
     locus, plus the locus embedding predicted from that locus's features;
  2. WITHIN each active locus: a [NULL] token carrying the locus features, the
     token count, the net frame offset and the out-of-frame fraction, then a
     shared 2-layer encoder with a per-locus FiLM; the locus vector is the
     output at [NULL]. Pairs are PACKED and bucketed by token count, so the
     encoder pads each bucket only to its own width;
  3. ACROSS the active loci: a 2-layer encoder over the locus vectors plus one
     learned REFERENCE token, with a padding mask;
  4. readout: one learned query per drug attends over the loci and the
     reference token, then an MLP -> one logit per drug.

The reference token is what makes the sparse form behave: a susceptible isolate
may have no relevant locus active at all, and without it the softmax would have
to put every drug query's weight on some lineage SNP.

Locus identity is NOT a free embedding. At 6,776 loci that would be 2.7M
parameters, most of them for loci that vary in a handful of isolates. The
embedding and the FiLM scale/shift are predicted from the locus feature vector
(region kind, functional category, essentiality, length, position, WHO-19
membership), plus a small learned residual with its own dropout.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from genomefusion.layout import DIST_MISSING, N_NULL_STATS

GF_DEFAULTS = dict(d_model=128, nhead=4, dropout=0.1, enc_layers=2, enc_dim_ff=256,
                   fusion_layers=2, fusion_dim_ff=256, pos_dims=32, hidden=256,
                   residual_scale=0.1, residual_dropout=0.3)
#: pairs are padded up to the next of these widths, so a locus with one token
#: never sits in a tensor sized for the busiest locus in the batch
BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def _sinusoid(x, dims):
    half = dims // 2
    freq = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32)
                     * (-np.log(10_000.0) / max(half - 1, 1)))
    a = x.unsqueeze(-1).float() * freq
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


def proximity_table(dist_drugs, loci, distances, length=10.0):
    """(n_drugs, genome_len) is far too big; this is the per-locus fallback used
    when a token has no site index: one value per (drug, locus)."""
    out = np.full((len(dist_drugs), len(loci)), DIST_MISSING, np.float32)
    for i, d in enumerate(dist_drugs):
        for j, L in enumerate(loci):
            key = f"{d}|{L.gene}"
            if key in distances:
                out[i, j] = 0.0
    return out


class GenomeFusionNet(nn.Module):
    def __init__(self, locus_features, drugs, c_tok, n_dist=0, site_table=None,
                 d_model=128, nhead=4, dropout=0.1, enc_layers=2, enc_dim_ff=256,
                 fusion_layers=2, fusion_dim_ff=256, pos_dims=32, hidden=256,
                 residual_scale=0.1, residual_dropout=0.3, out_bias=None,
                 regulated_lid=None):
        super().__init__()
        if d_model % nhead or pos_dims % 2:
            raise ValueError("d_model must divide by nhead and pos_dims must be even")
        self.drugs = list(drugs)
        self.n_loci, self.n_feat = locus_features.shape
        self.c_tok, self.n_dist = int(c_tok), int(n_dist)
        self.c_in = self.c_tok + self.n_dist
        self.d_model, self.pos_dims = d_model, pos_dims
        self.config = dict(drugs=self.drugs, c_tok=self.c_tok, n_dist=self.n_dist,
                           n_loci=self.n_loci, n_feat=self.n_feat, d_model=d_model,
                           nhead=nhead, dropout=dropout, enc_layers=enc_layers,
                           enc_dim_ff=enc_dim_ff, fusion_layers=fusion_layers,
                           fusion_dim_ff=fusion_dim_ff, pos_dims=pos_dims, hidden=hidden,
                           residual_scale=residual_scale, residual_dropout=residual_dropout,
                           out_bias=out_bias)

        self.register_buffer("locus_feat", torch.as_tensor(locus_features, dtype=torch.float32))
        if site_table is None:
            site_table = np.zeros((max(self.n_dist, 1), 1), np.float32) + DIST_MISSING
        self.register_buffer("site_table", torch.as_tensor(np.asarray(site_table, np.float32)))

        # locus identity, predicted from features (+ a small learned residual)
        self.feat_proj = nn.Sequential(nn.Linear(self.n_feat, d_model), nn.GELU(),
                                       nn.Linear(d_model, d_model))
        self.film_proj = nn.Linear(self.n_feat, 2 * d_model)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)
        self.locus_residual = nn.Embedding(self.n_loci, d_model)
        nn.init.trunc_normal_(self.locus_residual.weight, std=0.02)
        self.residual_scale = float(residual_scale)
        self.residual_drop = nn.Dropout(residual_dropout)

        # The known-relationship edge: locus -> the locus it regulates, or -1.
        # A promoter's identity picks up the learned residual of the GENE it
        # acts on, so "the fabG1 promoter" is tied to fabG1 specifically
        # rather than being one of 3,027 interchangeable intergenic segments.
        # Gated by its own scalar, initialised at 0, so the model starts
        # exactly where it would without the edge and has to earn the term.
        if regulated_lid is None:
            regulated_lid = np.full(self.n_loci, -1, np.int64)
        regulated_lid = np.asarray(regulated_lid, np.int64)
        if regulated_lid.shape != (self.n_loci,):
            raise ValueError(f"regulated_lid must be ({self.n_loci},), got {regulated_lid.shape}")
        self.register_buffer("regulated_lid", torch.as_tensor(regulated_lid, dtype=torch.long))
        self.has_edge = bool((regulated_lid >= 0).any())
        self.reg_gate = nn.Parameter(torch.zeros(1))
        self.config["n_regulated_edges"] = int((regulated_lid >= 0).sum())

        self.tok_proj = nn.Linear(self.c_in, d_model)
        self.pos_proj = nn.Linear(pos_dims, d_model)
        self.null_proj = nn.Linear(N_NULL_STATS, d_model)
        self.null_emb = nn.Parameter(torch.zeros(d_model))
        self.tok_norm = nn.LayerNorm(d_model)

        def encoder(layers, ff):
            return nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, nhead, ff, dropout=dropout,
                                           batch_first=True, norm_first=True),
                layers, enable_nested_tensor=False)
        self.encoder = encoder(enc_layers, enc_dim_ff)
        self.fusion = encoder(fusion_layers, fusion_dim_ff)
        self.summ_norm = nn.LayerNorm(d_model)
        self.reference = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.reference, std=0.02)

        self.drug_queries = nn.Parameter(torch.zeros(len(self.drugs), d_model))
        nn.init.trunc_normal_(self.drug_queries, std=0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc_out = nn.Linear(hidden, 1)
        if out_bias is not None:
            with torch.no_grad():
                self.fc_out.bias.fill_(float(out_bias))

    # ---- locus identity ----------------------------------------------------
    def locus_vectors(self, lid):
        """(n, d) identity for compact locus ids; lid < 0 is the reference slot."""
        safe = lid.clamp_min(0)
        emb = self.feat_proj(self.locus_feat[safe])
        emb = emb + self.residual_scale * self.residual_drop(self.locus_residual(safe))
        if self.has_edge:
            tgt = self.regulated_lid[safe]                     # the gene it regulates
            edge = self.residual_drop(self.locus_residual(tgt.clamp_min(0)))
            emb = emb + self.reg_gate * torch.where(
                (tgt >= 0).unsqueeze(-1), edge, torch.zeros_like(edge))
        return torch.where((lid >= 0).unsqueeze(-1), emb, torch.zeros_like(emb))

    def film(self, lid):
        safe = lid.clamp_min(0)
        g = self.film_proj(self.locus_feat[safe])
        scale, shift = g.chunk(2, -1)
        return 1.0 + scale, shift

    # ---- stage 1: within each active locus ---------------------------------
    def encode_pairs(self, b):
        """Packed (isolate, locus) pairs -> (P, d) locus vectors."""
        P, dev = b["P"], self.drug_queries.device
        d = self.d_model
        if P == 0:
            return torch.zeros(0, d, device=dev)
        lid = b["pair_lid"]
        lvec = self.locus_vectors(lid)
        scale, shift = self.film(lid)

        null = (self.null_emb.unsqueeze(0) + self.null_proj(b["pair_stats"]) + lvec)
        if b["tok_pair"].numel() == 0:
            x = self.tok_norm(null * scale + shift).unsqueeze(1)
            return self.encoder(x)[:, 0]

        x = b["tok_feat"].to(torch.float32)
        if self.n_dist:
            idx = b["tok_site"]
            ok = idx >= 0
            tab = self.site_table                       # (n_dist, n_sites)
            safe = idx.clamp(0, tab.shape[1] - 1)
            dist = tab[:, safe].t()                     # (N, n_dist)
            dist = torch.where(ok.unsqueeze(-1), dist, torch.full_like(dist, DIST_MISSING))
            x = torch.cat([x, dist], -1)
        tok = self.tok_proj(x) + self.pos_proj(_sinusoid(b["tok_coord"], self.pos_dims))
        tok = tok + lvec[b["tok_pair"]]

        n = b["pair_n"]
        out = torch.zeros(P, d, device=dev)
        # bucket pairs by token count; each bucket pads only to its own width
        for lo, hi in zip((0,) + BUCKETS[:-1], BUCKETS):
            sel = torch.nonzero((n > lo) & (n <= hi), as_tuple=True)[0]
            if not sel.numel():
                continue
            out[sel] = self._run_bucket(tok, b["tok_pair"], sel, hi, null, scale, shift)
        big = torch.nonzero(n > BUCKETS[-1], as_tuple=True)[0]
        if big.numel():
            out[big] = self._run_bucket(tok, b["tok_pair"], big, int(n[big].max()),
                                        null, scale, shift)
        return out

    def _run_bucket(self, tok, tok_pair, sel, width, null, scale, shift):
        """Run the shared encoder on the pairs in `sel`, padded to `width` tokens."""
        dev = tok.device
        d = self.d_model
        P_b = sel.numel()
        slot = torch.full((len(null),), -1, dtype=torch.long, device=dev)
        slot[sel] = torch.arange(P_b, device=dev)
        keep = slot[tok_pair] >= 0
        idx = torch.nonzero(keep, as_tuple=True)[0]
        rows = slot[tok_pair[idx]]
        # position of each token inside its own pair
        order = torch.argsort(rows, stable=True)
        idx, rows = idx[order], rows[order]
        counts = torch.bincount(rows, minlength=P_b)
        starts = torch.cumsum(counts, 0) - counts
        within = torch.arange(len(rows), device=dev) - starts[rows]
        ok = within < width
        idx, rows, within = idx[ok], rows[ok], within[ok]

        x = torch.zeros(P_b, width + 1, d, device=dev)
        pad = torch.ones(P_b, width + 1, dtype=torch.bool, device=dev)
        x[:, 0] = null[sel]
        pad[:, 0] = False
        x[rows, within + 1] = tok[idx]
        pad[rows, within + 1] = False
        x = x * scale[sel].unsqueeze(1) + shift[sel].unsqueeze(1)
        x = self.tok_norm(x) * (~pad).unsqueeze(-1)
        return self.encoder(x, src_key_padding_mask=pad)[:, 0]

    # ---- stage 2 + readout --------------------------------------------------
    def forward(self, b, return_attn=False):
        B, Lmax, dev = b["B"], b["Lmax"], self.drug_queries.device
        d = self.d_model
        pair_vec = self.encode_pairs(b)
        fused = torch.zeros(B, Lmax, d, device=dev)
        lvalid = torch.zeros(B, Lmax, dtype=torch.bool, device=dev)
        fused[:, 0] = self.reference
        lvalid[:, 0] = True
        if b["P"]:
            fused[b["pair_bi"], b["pair_slot"]] = pair_vec
            lvalid[b["pair_bi"], b["pair_slot"]] = True
        fused = self.summ_norm(fused) * lvalid.unsqueeze(-1)
        pad = ~lvalid
        fused = self.fusion(fused, src_key_padding_mask=pad)

        q = self.drug_queries.unsqueeze(0).expand(B, -1, -1)
        pooled, attn = self.pool_attn(q, fused, fused, key_padding_mask=pad,
                                      need_weights=return_attn)
        logits = self.fc_out(F.relu(self.fc1(self.norm(pooled)))).squeeze(-1)
        return (logits, attn) if return_attn else logits
