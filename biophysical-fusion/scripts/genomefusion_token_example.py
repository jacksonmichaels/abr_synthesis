"""Pick one real isolate's tokens for the tokenization figure.

    cd biophysical-fusion && python scripts/genomefusion_token_example.py

Writes results/figures/genomefusion/token_example.json. Reads one token shard (not
the whole store) and takes the channel values from the trainer's own
`BatchBuilder._expand`, so the figure cannot disagree with what the model sees.
Light enough for a login node: one 50 MB shard plus the cached partition.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from genomefusion.data import TOKEN_FIELDS, BatchBuilder  # noqa: E402
from genomefusion.layout import C_BASE, CONSEQUENCES, FRAMESHIFT, NT_SYMBOLS, SLOTS  # noqa: E402
from genomefusion.partition import WORKSPACE, load_or_build  # noqa: E402

OUT = PROJECT / "results" / "figures" / "genomefusion" / "token_example.json"
KATG_S315, FABG1_C15 = 2155167, 1673424       # 0-based genome positions
SHARD = WORKSPACE / "genomefusion_tokens" / "shard_0.npz"


def main():
    P = load_or_build()
    z = np.load(SHARD, allow_pickle=False)
    T = {k: z[k] for k in TOKEN_FIELDS}
    off = np.concatenate([[0], np.cumsum(z["n_tok"])])
    sum_off = np.concatenate([[0], np.cumsum(z["n_sum"])])
    start = np.asarray([L.start for L in P.loci], np.int64)
    expander = SimpleNamespace(store=SimpleNamespace(tok=T), c_tok=C_BASE, locus_start=start)

    picks, best = None, None
    for i in range(len(off) - 1):
        r = np.arange(off[i], off[i + 1])
        snp = r[(T["pos"][r] == KATG_S315) & (T["span"][r] == 1)]
        prom = r[(T["pos"][r] == FABG1_C15) & (T["span"][r] == 1)]
        fs = r[T["consequence"][r] == FRAMESHIFT]
        runs = r[(T["span"][r] >= 10) & (T["base"][r] == NT_SYMBOLS.index("-"))]
        if len(snp) and len(prom) and len(fs) and len(runs):
            n = int(off[i + 1] - off[i])
            if best is None or n < best:          # the plainest isolate that shows all four
                best, picks = n, (i, int(snp[0]), int(prom[0]), int(fs[0]), int(runs[0]))
    if picks is None:
        raise SystemExit(f"no isolate in {SHARD.name} carries all four token kinds")
    i, *rows = picks

    feat = BatchBuilder._expand(expander, np.asarray(rows), None)[0]
    name = lambda L: L.gene or L.name
    in_locus = lambda L: int((T["locus"][off[i]:off[i + 1]] == L.index).sum())
    tokens = []
    for kind, row, f in zip(("substitution", "promoter", "frameshift", "deletion"), rows, feat):
        L = P.loci[int(T["locus"][row])]
        c = int(T["consequence"][row])
        tokens.append(dict(kind=kind, row=row, locus=name(L), pos=int(T["pos"][row]),
                           locus_kind=L.kind, locus_tokens=in_locus(L),
                           locus_start=int(L.start), locus_end=int(L.end),
                           base=NT_SYMBOLS[int(T["base"][row])], span=int(T["span"][row]),
                           codon=int(T["codon"][row]),
                           consequence=CONSEQUENCES[c] if c >= 0 else None,
                           truncation=round(float(T["truncation"][row]), 3),
                           up_plus=int(T["up_plus"][row]), up_minus=int(T["up_minus"][row]),
                           feat=[round(float(v), 3) for v in f]))
    loc = z["sum_locus"][sum_off[i]:sum_off[i + 1]]
    counts = np.bincount(T["locus"][off[i]:off[i + 1]], minlength=len(P.loci))
    busiest = P.loci[int(counts.argmax())]
    out = dict(isolate=str(z["ids"][i]), shard=SHARD.name, n_tokens=int(off[i + 1] - off[i]),
               n_active_loci=int(len(set(loc.tolist()))), n_loci=len(P.loci),
               busiest=dict(locus=name(busiest), n=int(counts.max())),
               slots={k: list(v) for k, v in SLOTS.items()},
               tokens=tokens)
    OUT.write_text(json.dumps(out, indent=1))
    show = {k: v for k, v in out.items() if k not in ("tokens", "slots")}
    print(show)
    for t in tokens:
        print({k: v for k, v in t.items() if k != "feat"})


if __name__ == "__main__":
    main()
