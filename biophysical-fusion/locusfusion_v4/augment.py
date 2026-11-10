"""
Synthetic synonymous substitutions, inserted into a batch on the GPU.

Why: on an H37Rv background the v3 pyrazinamide models give P(R) 0.14, and one
synonymous pncA change moves that to a mean of 0.52-0.62
(results/analysis/pza_substitution_scan). The token already says the codon
did not change; the training data just has too few synonymous changes for the
model to learn that it should not care. So the training fold is shown isolates
with extra synonymous changes, and the loss asks for the same answer:

    loss = BCE(clean) + BCE(augmented copies, same labels)
         + lambda * mean |logit(augmented) - stopgrad(logit(clean))|

(train.py). This module only makes the copies.

A copy gets k ~ U{1..k_max} synthetic tokens. Each is drawn locus-first --
a locus uniformly among the coding loci, then one of its candidates uniformly
-- so a short gene (pncA, 418 candidates) is augmented as often as rpoB (3,000+).
A draw is rejected, and up to `tries` more are made, when its locus has no free
slot (the token cap is never exceeded and no real variant is displaced) or the
isolate already carries a change in that codon (a second change there could
make it non-synonymous). The candidate table (tokenizer.synonymous_candidates)
already excludes the start and stop codons, bases another locus also carries,
and augment_exclusions.json.

The copy is a valid isolate in every respect: the token lands in a free slot
(slot order is irrelevant -- stage 1 has no positional index other than the
coordinate), carries its real coordinate, codon phase and binding-site
position, and the locus's changed-column count goes up by one, as it would for
a real isolate with that change.
"""
import numpy as np
import torch

BATCH_KEYS = ("feat", "coord", "valid", "codon", "site_pos", "n_occ")


class SynonymousAugmenter:
    def __init__(self, cand, device, locus_weight="uniform", tries=4):
        loc = np.asarray(cand["locus"], np.int64)
        if len(loc) == 0:
            raise ValueError("no synonymous candidates in this token set")
        order = np.argsort(loc, kind="stable")
        t = lambda a, dt: torch.as_tensor(np.asarray(a)[order], dtype=dt, device=device)  # noqa: E731
        self.locus = t(loc, torch.long)
        self.codon = t(cand["codon"], torch.long)
        self.coord = t(cand["coord"], torch.float32)
        self.site_pos = t(cand["site_pos"], torch.long)
        self.feat = t(cand["feat"], torch.float32)
        loci, start, count = np.unique(loc[order], return_index=True, return_counts=True)
        self.loci = torch.as_tensor(loci, device=device)
        self.start = torch.as_tensor(start, device=device)
        self.count = torch.as_tensor(count, device=device)
        if locus_weight not in ("uniform", "length"):
            raise ValueError("locus_weight is 'uniform' or 'length'")
        self.locus_weight, self.tries = locus_weight, int(tries)

    def _draw(self, B, gen):
        """(B, tries) candidate indices."""
        dev = self.locus.device
        if self.locus_weight == "length":
            return torch.randint(len(self.locus), (B, self.tries), device=dev, generator=gen)
        j = torch.randint(len(self.loci), (B, self.tries), device=dev, generator=gen)
        u = torch.rand(B, self.tries, device=dev, generator=gen)
        return self.start[j] + torch.minimum((u * self.count[j]).long(), self.count[j] - 1)

    @torch.no_grad()
    def __call__(self, batch, k_max=2, gen=None, exact_k=None):
        """batch: dict of BATCH_KEYS tensors (B, L, V, ...) -> (augmented copy, n inserted (B,)).

        k ~ U{1..k_max} per isolate, or exactly `exact_k` (the robustness probe)."""
        out = {k: batch[k].clone() for k in BATCH_KEYS}
        feat, coord, valid, codon, site, n_occ = (out[k] for k in BATCH_KEYS)
        B, L, V = valid.shape
        dev = valid.device
        if exact_k is not None:
            k = torch.full((B,), int(exact_k), device=dev)
            rounds = int(exact_k)
        else:
            k = torch.randint(1, int(k_max) + 1, (B,), device=dev, generator=gen)
            rounds = int(k_max)
        inserted = torch.zeros(B, dtype=torch.long, device=dev)
        ar = torch.arange(B, device=dev)
        for r in range(rounds):
            c = self._draw(B, gen)                                   # (B, T)
            l = self.locus[c]
            v = valid[ar[:, None], l]                                # (B, T, V)
            free = ~v.all(-1)
            clash = ((codon[ar[:, None], l] == self.codon[c].unsqueeze(-1)) & v).any(-1)
            ok = free & ~clash & (r < k).unsqueeze(1)
            has = ok.any(1)
            if not has.any():
                continue
            pick = ok.float().argmax(1)
            rows = ar[has]
            cc = c[rows, pick[has]]
            ll = self.locus[cc]
            slot = (~valid[rows, ll]).float().argmax(-1)             # first free slot
            feat[rows, ll, slot] = self.feat[cc].to(feat.dtype)
            coord[rows, ll, slot] = self.coord[cc].to(coord.dtype)
            valid[rows, ll, slot] = True
            codon[rows, ll, slot] = self.codon[cc].to(codon.dtype)
            site[rows, ll, slot] = self.site_pos[cc].to(site.dtype)
            n_occ[rows, ll] += 1
            inserted[rows] += 1
        return out, inserted
