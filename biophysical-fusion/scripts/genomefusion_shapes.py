"""Measure GenomeFusion's real batch shapes, for the architecture figure.

    cd biophysical-fusion && python scripts/genomefusion_shapes.py

Writes results/figures/genomefusion/batch_shapes.json.

Everything here comes from the token store's OFFSET arrays (`tok_off`,
`sum_off`) and the locus column, never from the expanded feature matrix, so it
costs a few seconds and a few hundred MB rather than the ~2 GB the trainer
spends -- light enough for a login node.

The batches are cut by the real sampler (`length_grouped_batches`) at the real
`--locus-budget`, so B and Lmax are the shapes a training step actually sees:
isolates are grouped by active-locus count and the budget caps batch x width,
which is why the widest batch holds fewer isolates than the median one.
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from genomefusion.data import length_grouped_batches  # noqa: E402
from genomefusion.model import BUCKETS, GF_DEFAULTS  # noqa: E402
from genomefusion.partition import WORKSPACE, load_or_build  # noqa: E402
from genomefusion.train import base_19r, locus_subset  # noqa: E402

OUT = PROJECT / "results" / "figures" / "genomefusion"
TOKENS = WORKSPACE / "genomefusion_tokens" / "tokens.npz"


def main(tokens=TOKENS, batch_size=32, budget=32768, seed=0):
    P = load_or_build()
    z = np.load(tokens, allow_pickle=False)
    tok_locus, sum_locus = z["locus"], z["sum_locus"]
    tok_off, sum_off = z["tok_off"], z["sum_off"]
    n_iso = len(sum_off) - 1
    tok_iso = np.repeat(np.arange(n_iso), np.diff(tok_off))
    sum_iso = np.repeat(np.arange(n_iso), np.diff(sum_off))

    out = {"meta": {"n_isolates": n_iso, "batch_size": batch_size, "locus_budget": budget,
                    "seed": seed, "n_loci": len(P.loci),
                    "n_feat": int(P.feature_matrix().shape[1]),
                    "d_model": GF_DEFAULTS["d_model"], "pos_dims": GF_DEFAULTS["pos_dims"],
                    "hidden": GF_DEFAULTS["hidden"], "buckets": list(BUCKETS),
                    "tokens": str(tokens)}}
    for name, loci in (("genome", locus_subset(P, "genome")), ("19r", base_19r(P))):
        keep = np.zeros(len(P.loci), bool)
        keep[np.asarray(loci, np.int64)] = True
        n_active = np.bincount(sum_iso[keep[sum_locus]], minlength=n_iso)
        n_tok = np.bincount(tok_iso[keep[tok_locus]], minlength=n_iso)
        batches = length_grouped_batches(n_active, np.arange(n_iso), batch_size,
                                         np.random.default_rng(seed), locus_budget=budget)
        widths = np.array([n_active[b].max() + 1 for b in batches])
        rec = {"n_loci": int(len(loci)),
               "active_mean": round(float(n_active.mean()), 1),
               "active_max": int(n_active.max()),
               "tokens_mean": round(float(n_tok.mean()), 1),
               "n_batches": len(batches)}
        for lab, b in (("median", batches[int(np.argsort(widths)[len(widths) // 2])]),
                       ("widest", batches[int(np.argmax(widths))])):
            rec[lab] = {"B": len(b), "Lmax": int(n_active[b].max()) + 1,
                        "P": int(n_active[b].sum()), "N": int(n_tok[b].sum())}
        # tokens per (isolate, locus) pair on the median batch -> bucket occupancy
        b = batches[int(np.argsort(widths)[len(widths) // 2])]
        sel = keep[tok_locus] & np.isin(tok_iso, b)
        key = tok_locus[sel].astype(np.int64) + (tok_iso[sel].astype(np.int64) << 32)
        _, c = np.unique(key, return_counts=True)
        occ, lo = {}, 0
        for hi in BUCKETS:
            n = int(((c > lo) & (c <= hi)).sum())
            if n:
                occ[str(hi)] = n
            lo = hi
        if (c > BUCKETS[-1]).any():
            occ[f">{BUCKETS[-1]}"] = int((c > BUCKETS[-1]).sum())
        rec["buckets"] = occ
        rec["pair_tokens"] = {"median": int(np.median(c)), "mean": round(float(c.mean()), 2),
                              "max": int(c.max())}
        out[name] = rec
        m = rec["median"]
        print(f"{name:7s} {rec['n_loci']:>6,} loci | median batch B={m['B']} Lmax={m['Lmax']} "
              f"P={m['P']:,} N={m['N']:,} | pair tokens med {rec['pair_tokens']['median']} "
              f"max {rec['pair_tokens']['max']} | buckets {occ}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "batch_shapes.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT / 'batch_shapes.json'}")
    return out


if __name__ == "__main__":
    main()
