"""How often are the who_t12r locus pairs a GI / regulatory attention bias would
touch actually active in the same isolate? Training isolates only; trains nothing.

An across-locus bias between loci i and j can only act in an isolate that carries
a variant in both, because unchanged loci never enter the network.

    python scripts/gi_coactivity.py --xlsx X --out DIR

Writes to DIR:
    gi_pairs.csv          every measured gene pair in each drug's set
    regulatory_pairs.csv  promoter<->target gene, and promoters inheriting a GI
    drug_summary.csv      per-drug summary, including the stop rule
"""
import argparse
import html
import itertools
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.loader import drug_loci  # noqa: E402
from genomefusion.data import Labels, split_rows  # noqa: E402
from genomefusion.partition import WORKSPACE, load_or_build  # noqa: E402
from genomefusion.train import ALL_DRUGS, REAL_PHENOTYPE_CSV, locus_subset  # noqa: E402

N_PAIRS = 290_703                   # 762 genes including self-pairs
ETO_ACTIVE_MEAN = 1.419996656       # the gf_who_t12r ETO canary's active_loci_mean
RPOB_RPOC_GI2 = -7.331
STOP_MIN_ISOLATES = 50
STOP_MIN_DRUGS = 3
KEEP = ("orf1", "orf2", "self", "possible_polar", "gi_score_experiment1", "gi_score_experiment2",
        "joint_prob_negative", "joint_prob_positive", "joint_prob_discordant", "joint_call")
CELL = re.compile(r'<c r="([A-Z]+)\d+"([^>]*?)(?:/>|>(?:<f>.*?</f>)?<v>([^<]*)</v></c>)')
ROW = re.compile(r"<row [^>]*?>(.*?)</row>", re.S)


def to_rv(tag):
    return "Rv" + tag[4:] if tag.startswith("RVBD") else tag


def read_gi(path):
    """{(Rv a, Rv b) sorted: record}, non-self rows, streamed from the xlsx."""
    z = zipfile.ZipFile(path)
    shared = [html.unescape("".join(re.findall(r"<t[^>]*>([^<]*)</t>", m)))
              for m in re.findall(r"<si>(.*?)</si>", z.read("xl/sharedStrings.xml").decode(), re.S)]
    hdr, out, n = None, {}, 0
    with z.open("xl/worksheets/sheet2.xml") as f:
        buf = b""
        while True:
            chunk = f.read(1 << 22)
            if not chunk:
                break
            buf += chunk
            end = buf.rfind(b"</row>")
            if end < 0:
                continue
            for m in ROW.finditer(buf[:end + 6].decode()):
                d = {c: (shared[int(v)] if 't="s"' in a and v else v)
                     for c, a, v in CELL.findall(m.group(1))}
                if hdr is None:
                    hdr = {v: k for k, v in d.items()}
                    continue
                n += 1
                r = {k: d.get(hdr[k], "") for k in KEEP}
                if r["self"] == "1":
                    continue
                a, b = to_rv(r["orf1"]), to_rv(r["orf2"])
                out[(a, b) if a < b else (b, a)] = {
                    "polar": r["possible_polar"] == "1",
                    "gi1": float(r["gi_score_experiment1"]),
                    "gi2": float(r["gi_score_experiment2"]),
                    "p_any": float(r["joint_prob_negative"]) + float(r["joint_prob_positive"])
                    + float(r["joint_prob_discordant"]),
                    "call": r["joint_call"]}
            buf = buf[end + 6:]
    if n != N_PAIRS:
        raise AssertionError(f"expected {N_PAIRS} rows, read {n}")
    return out


class Store:
    """Only what split_rows and Labels read, without expanding 15.8M tokens."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.ids = [str(x) for x in z["ids"]]
        self.index = {s: i for i, s in enumerate(self.ids)}
        self.sum_locus = z["sum_locus"]
        self.sum_off = z["sum_off"]

    def __len__(self):
        return len(self.ids)


def active_matrix(store, set_ids):
    """(n_isolates, n_set) bool: isolate carries a variant in that locus."""
    col = np.full(int(store.sum_locus.max()) + 1, -1, np.int64)
    col[set_ids] = np.arange(len(set_ids))
    iso = np.repeat(np.arange(len(store)), np.diff(store.sum_off))
    c = col[store.sum_locus]
    keep = c >= 0
    A = np.zeros((len(store), len(set_ids)), bool)
    A[iso[keep], c[keep]] = True
    return A


def counts(A, y, i, j):
    co = A[:, i] & A[:, j]
    return {"n_a": int(A[:, i].sum()), "n_b": int(A[:, j].sum()), "co": int(co.sum()),
            "co_R": int((co & (y == 1)).sum()), "co_S": int((co & (y == 0)).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default=str(WORKSPACE / "genomefusion_tokens" / "tokens.npz"))
    ap.add_argument("--cache", default=str(WORKSPACE / "genomefusion_cache_v2"))
    ap.add_argument("--split-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gi = read_gi(args.xlsx)
    got = gi[("Rv0667", "Rv0668")]["gi2"]
    if abs(got - RPOB_RPOC_GI2) > 1e-6:
        raise AssertionError(f"rpoB/rpoC GI exp2 {got}, expected {RPOB_RPOC_GI2}")
    screened = {x for k in gi for x in k}
    print(f"[gi] {len(gi):,} pairs, {len(screened)} genes; rpoB/rpoC check ok", flush=True)

    p = load_or_build(args.cache)
    store = Store(args.tokens)
    loci = p.loci
    by_name = {}
    for i, L in enumerate(loci):
        if L.kind != "intergenic":
            for key in (L.gene, L.name):
                if key:
                    by_name.setdefault(key, i)

    def get_gi(i, j):
        a, b = loci[i].name, loci[j].name
        return gi.get((a, b) if a < b else (b, a))

    gi_rows, reg_rows, drug_rows = [], [], []
    for drug in ALL_DRUGS:
        set_ids = np.asarray(locus_subset(p, "who_t12r", drug=drug), np.int64)
        named = {int(i) for i in locus_subset(p, "who_t12", drug=drug)}
        A_all = active_matrix(store, set_ids)
        mean_all = float(A_all.sum(1).mean())
        if drug == "ETHIONAMIDE" and abs(mean_all - ETO_ACTIVE_MEAN) > 1e-6:
            raise AssertionError(f"ETO active mean {mean_all}, canary had {ETO_ACTIVE_MEAN}")

        y_all = Labels(store, [drug], REAL_PHENOTYPE_CSV).Y[:, 0]
        train, _, _ = split_rows(store, str(Path(args.split_dir) / f"{drug}.csv"))
        train = train[y_all[train] != -1]
        A, y = A_all[train], y_all[train]
        slot = {int(x): k for k, x in enumerate(set_ids)}
        base = {"drug": drug, "n_train": len(train), "n_R": int((y == 1).sum()),
                "n_S": int((y == 0).sum())}

        # every measured gene pair in the set
        any_called, any_measured = np.zeros(len(y), bool), np.zeros(len(y), bool)
        n_called, n_called_50 = 0, 0
        for a, b in itertools.combinations(set_ids.tolist(), 2):
            g = get_gi(a, b)
            if g is None:
                continue
            both = a in named and b in named
            i, j = slot[a], slot[b]
            c = counts(A, y, i, j)
            called = g["call"] != "no-interaction"
            gi_rows.append({**base, "a": loci[a].gene or loci[a].name,
                            "b": loci[b].gene or loci[b].name, "both_who_named": both,
                            "called": called, **g, **c})
            if both:
                co = A[:, i] & A[:, j]
                any_measured |= co
                if called:
                    any_called |= co
                    n_called += 1
                    n_called_50 += c["co"] >= STOP_MIN_ISOLATES

        # promoter loci: the target itself, and any gene the target has a called GI with
        any_edge = np.zeros(len(y), bool)
        n_prom = n_edge = 0
        for s in set_ids.tolist():
            L = loci[s]
            if L.kind != "intergenic" or not L.regulates:
                continue
            n_prom += 1
            for t_name in L.regulates:
                t = by_name.get(t_name)
                if t is None:
                    continue
                if t in slot:
                    n_edge += 1
                    c = counts(A, y, slot[s], slot[t])
                    any_edge |= A[:, slot[s]] & A[:, slot[t]]
                    reg_rows.append({**base, "kind": "promoter_target", "promoter": L.name,
                                     "target": t_name, "partner": t_name, "call": "", "p_any": "",
                                     "polar": "", **c})
                for h in set_ids.tolist():
                    if h == t or loci[h].kind == "intergenic" or loci[t].name not in screened:
                        continue
                    g = get_gi(t, h)
                    if g is None or g["call"] == "no-interaction":
                        continue
                    reg_rows.append({**base, "kind": "promoter_inherits_gi", "promoter": L.name,
                                     "target": t_name, "partner": loci[h].gene or loci[h].name,
                                     "call": g["call"], "p_any": g["p_any"], "polar": g["polar"],
                                     **counts(A, y, slot[s], slot[h])})

        n_act = A.sum(1)
        drug_rows.append({
            **base, "n_loci": len(set_ids), "active_mean_all": round(mean_all, 3),
            "active_mean_train": round(float(n_act.mean()), 3), "active_max": int(n_act.max()),
            "frac_ge2_active": round(float((n_act >= 2).mean()), 4),
            "called_named_pairs": n_called, "called_named_pairs_ge50": n_called_50,
            "frac_any_called_coactive": round(float(any_called.mean()), 4),
            "frac_any_measured_coactive": round(float(any_measured.mean()), 4),
            "promoter_loci": n_prom, "promoter_target_edges": n_edge,
            "frac_any_promoter_target_coactive": round(float(any_edge.mean()), 4)})
        print(f"[gi] {drug} done", flush=True)

    gi_df = pd.DataFrame(gi_rows)
    reg_df = pd.DataFrame(reg_rows)
    sum_df = pd.DataFrame(drug_rows)
    gi_df.to_csv(out / "gi_pairs.csv", index=False)
    reg_df.to_csv(out / "regulatory_pairs.csv", index=False)
    sum_df.to_csv(out / "drug_summary.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_rows", 500)
    print(sum_df.to_string(index=False))
    print("\ncalled pairs among WHO-named genes")
    cols = ["drug", "a", "b", "call", "polar", "p_any", "gi1", "gi2", "n_a", "n_b", "co", "co_R", "co_S"]
    print(gi_df[gi_df.called & gi_df.both_who_named][cols].to_string(index=False))
    print("\npromoter <-> target")
    cols = ["drug", "promoter", "target", "n_a", "n_b", "co", "co_R", "co_S"]
    print(reg_df[reg_df.kind == "promoter_target"][cols].to_string(index=False))
    print("\npromoter inheriting a called GI")
    cols = ["drug", "promoter", "target", "partner", "call", "polar", "n_a", "n_b", "co", "co_R", "co_S"]
    print(reg_df[reg_df.kind == "promoter_inherits_gi"][cols].to_string(index=False))
    passing = int((sum_df.called_named_pairs_ge50 > 0).sum())
    print(f"\nSTOP RULE: {passing} drugs have a called WHO-named pair co-active in >= "
          f"{STOP_MIN_ISOLATES} training isolates; need >= {STOP_MIN_DRUGS}: "
          f"{'PASS' if passing >= STOP_MIN_DRUGS else 'STOP'}")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
