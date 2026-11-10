"""Where GenomeFusion's regulatory relationships come from, and how each edge is decided.

    python scripts/genomefusion_regulatory_extract.py      # no model; seconds

Writes results/figures/genomefusion/regulatory_trace.json for
scripts/genomefusion_method_figures.py. Replays `Partition.regulated_index` so every
edge is labelled with the branch that produced it, and fails if the replay disagrees
with regulated_index on any of the 6,776 loci.
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from genomefusion.partition import PROMOTERS, WORKSPACE, load  # noqa: E402
from genomefusion.train import locus_subset  # noqa: E402

CACHE_V3 = WORKSPACE / "genomefusion_cache_v3"
OUT = PROJECT / "results" / "figures" / "genomefusion" / "regulatory_trace.json"
WINDOWS = {"fabG1-inhA": (1672800, 1675100), "katG": (2153700, 2157000)}
UPSTREAM_LENGTH = 50.0


def name(L):
    return L.gene or L.name


def main():
    P = load(CACHE_V3 / "partition.npz")
    L = P.loci
    prom = json.loads(Path(PROMOTERS).read_text())
    ri = P.regulated_index()
    by_gene = {}
    for i, x in enumerate(L):
        if x.gene and x.kind != "intergenic":
            by_gene.setdefault(x.gene, i)
    tag_named = {x.name: x.gene for x in L}

    # ---- replay regulated_index, recording the branch -------------------------------
    branch = []
    for i, x in enumerate(L):
        got, why = -1, "none"
        for g in x.regulates:
            if g and g in by_gene and by_gene[g] != i:
                got, why = by_gene[g], "who"
                break
        if got < 0:
            dn = x.downstream or ("", "")
            if dn[0] and dn[0] in by_gene and by_gene[dn[0]] != i:
                got, why = by_gene[dn[0]], "plus"
            elif len(dn) > 1 and dn[1] and dn[1] in by_gene and by_gene[dn[1]] != i:
                got, why = by_gene[dn[1]], "minus"
        assert got == int(ri[i]), (i, x.name, got, int(ri[i]))
        branch.append(why)
    branch = np.array(branch)
    print("replay matches regulated_index on all", len(L), "loci", flush=True)

    def start_codon(t):
        return t.start if t.strand > 0 else t.end - 1

    def dist_to(i, t):
        """Bases from a locus to the target's start codon, on the side the rule looks."""
        x, tt = L[i], L[t]
        return (start_codon(tt) - x.end) if tt.strand > 0 else (x.start - start_codon(tt))

    kinds = np.array([x.kind for x in L])
    named = np.array([bool(x.gene) for x in L])
    # genes by strand, every CDS whether named or not
    plus = sorted((x.start, i) for i, x in enumerate(L) if x.kind == "cds" and x.strand > 0)
    minus = sorted((x.end - 1, i) for i, x in enumerate(L) if x.kind == "cds" and x.strand < 0)
    ps, ms = np.array([a for a, _ in plus]), np.array([a for a, _ in minus])

    rows = []
    for i, x in enumerate(L):
        j = int(np.searchsorted(ps, x.end, side="left"))
        k = int(np.searchsorted(ms, x.start, side="right")) - 1
        dp = int(ps[j] - x.end) if j < len(ps) else None
        dm = int(x.start - ms[k]) if k >= 0 else None
        rows.append((dp, dm))

    ig = [i for i, x in enumerate(L) if x.kind == "intergenic"]
    dn_edge = [i for i in range(len(L)) if branch[i] in ("plus", "minus")]
    dist = {b: [dist_to(i, int(ri[i])) for i in range(len(L)) if branch[i] == b] for b in ("plus", "minus")}

    # intergenic loci given a plus-strand target although a minus-strand gene's start codon is nearer
    wrong_side = []
    for i in ig:
        if branch[i] != "plus":
            continue
        dp, dm = rows[i]
        d_t = dist_to(i, int(ri[i]))
        if dm is not None and dm < d_t:
            wrong_side.append((i, d_t, dm))
    wrong_side.sort(key=lambda r: r[1] - r[2], reverse=True)
    ig_edge_d = [dist_to(i, int(ri[i])) for i in ig if branch[i] in ("plus", "minus")]

    def example(i):
        x = L[i]
        return dict(locus=name(x), kind=x.kind, start=int(x.start), end=int(x.end), strand=int(x.strand),
                    is_regulatory=bool(x.is_regulatory), regulates=list(x.regulates), reg_bases=int(x.reg_bases),
                    downstream=list(x.downstream or ()), branch=str(branch[i]),
                    edge_to=name(L[int(ri[i])]) if ri[i] >= 0 else None,
                    edge_dist=dist_to(i, int(ri[i])) if ri[i] >= 0 and branch[i] != "who" else None,
                    index=i)

    # ---- real windows -----------------------------------------------------------------
    def window(lo, hi):
        ids = [i for i, x in enumerate(L) if x.end > lo and x.start < hi]
        ids.sort(key=lambda i: L[i].start)
        regions = []
        for g, w in prom["windows"].items():
            a, b = sorted((int(w["core_start"]), int(w["core_end"])))
            if b >= lo and a <= hi:
                regions.append(dict(gene=g, start=a, end=b + 1, **{k: w[k] for k in w if k in ("tss", "atg", "strand", "rv", "tag", "segments", "lo_off", "hi_off")}))
        return dict(lo=lo, hi=hi, loci=[example(i) for i in ids], regions=regions)

    # ---- the values the model gets for the c-15t locus and its target ------------------
    F = P.feature_matrix(with_regulated=True)
    fn = P.feature_names
    c15 = next(i for i, x in enumerate(L) if x.name == "Rv1482c-Rv1483")
    fab = by_gene["fabG1"]
    def nonzero(i):
        return [dict(name=fn[j], value=round(float(F[i, j]), 4)) for j in np.flatnonzero(F[i])]
    pos = 1673424
    upstream = dict(pos=pos, up_plus=int(P.up_plus[pos]), up_minus=int(P.up_minus[pos]),
                    ch_plus=round(float(np.exp(-abs(P.up_plus[pos]) / UPSTREAM_LENGTH)), 4),
                    ch_minus=round(float(np.exp(-abs(P.up_minus[pos]) / UPSTREAM_LENGTH)), 4))

    # ---- which edges survive in each locus set -----------------------------------------
    sets = {}
    for s in ("19", "19p", "19r", "minus19p", "genome"):
        keep = locus_subset(P, s)
        mask = np.zeros(len(L), bool)
        mask[keep] = True
        inside = ri[keep] >= 0
        live = inside & mask[np.clip(ri[keep], 0, None)]
        sets[s] = dict(loci=int(len(keep)), have_target=int(inside.sum()), edges_kept=int(live.sum()),
                       who_edges=int((live & (branch[keep] == "who")).sum()))

    q = lambda a, v: float(np.percentile(a, v))    # noqa: E731
    counts = {b: int((branch == b).sum()) for b in ("who", "plus", "minus", "none")}
    by_kind = {k: {b: int(((kinds == k) & (branch == b)).sum()) for b in ("who", "plus", "minus", "none")}
               for k in ("cds", "rna", "intergenic")}
    # WHO tags that lost a named gene to the first-hit rule
    multi = [example(i) for i, x in enumerate(L) if len(x.regulates) > 1]
    lost = []
    for i, x in enumerate(L):
        names_ = [g for g in x.regulates if g in by_gene and by_gene[g] != i]
        if len(names_) > 1:
            lost.append(dict(locus=name(x), regulates=list(x.regulates), kept=names_[0], dropped=names_[1:]))
    # WHO names with no locus to point at
    unresolved = sorted({g for x in L for g in x.regulates if g not in by_gene})
    unnamed_cds = int(((kinds == "cds") & ~named).sum())
    tagged = np.array([bool(x.is_regulatory) for x in L])
    tagged_by_branch = {b: int((tagged & (branch == b)).sum()) for b in ("who", "plus", "minus", "none")}
    # bases of a region counted once per WHO row naming it (fabG1 and inhA are one region, listed twice)
    n_ids = {g: (int(w["core_start"]), int(w["core_end"])) for g, w in prom["windows"].items()}
    dup_regions = [g for g in n_ids if sum(v == n_ids[g] for v in n_ids.values()) > 1]
    source_row = {g: {k: prom["windows"][g][k] for k in ("who_region_from", "rv", "sense", "anchor", "atg", "tss", "core_start", "core_end", "segments", "pad")}
                  for g in ("inhA", "katG")}

    OUT.write_text(json.dumps(dict(
        promoters_file=str(PROMOTERS), n_regions=len(prom["windows"]), pad=prom.get("pad"), chrom=prom.get("chrom"),
        counts=counts, by_kind=by_kind, n_loci=len(L), n_regulatory=int(sum(x.is_regulatory for x in L)),
        tagged_by_branch=tagged_by_branch, duplicate_regions=dup_regions, source_row=source_row,
        unnamed_cds=unnamed_cds, named_cds=int(((kinds == "cds") & named).sum()),
        multi_tag=len(multi), first_hit_lost=lost, unresolved_names=unresolved,
        dist_quantiles={b: dict(median=q(v, 50), p95=q(v, 95), max=int(max(v)), n=len(v)) for b, v in dist.items()},
        intergenic_downstream=dict(n=len(ig_edge_d), median=q(ig_edge_d, 50), p95=q(ig_edge_d, 95), max=int(max(ig_edge_d)),
                                   over_500=int(sum(d > 500 for d in ig_edge_d)), over_2000=int(sum(d > 2000 for d in ig_edge_d)),
                                   hist=[h.tolist() for h in np.histogram(np.log10(np.maximum(ig_edge_d, 1)), bins=30)]),
        intergenic_plus_with_nearer_minus=dict(n=len(wrong_side), examples=[
            dict(**example(i), chosen_dist=d, nearer_minus_dist=m) for i, d, m in wrong_side[:6]]),
        windows={k: window(*v) for k, v in WINDOWS.items()},
        features=dict(names=fn, c15=nonzero(c15), fabG1=nonzero(fab), own_width=len(P.own_feature_names),
                      c15_index=c15, fabG1_index=fab),
        upstream=upstream, sets=sets), indent=None))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
