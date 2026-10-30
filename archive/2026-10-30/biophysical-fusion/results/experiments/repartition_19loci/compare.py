"""locusfusion_v2 − mdcnn at 19 loci, all_modalities, across CV partitions.

    python results/experiments/repartition_19loci/compare.py

Reads whatever has landed. A partition counts only once both models have all 11
drugs with 5 folds; partial partitions are listed and skipped.

Four read-outs:

1. **Pairing.** Both models must hold out the same isolates in every fold. Each
   result records `val_ids_sha1` per fold; they are compared wherever both sides
   have them (results written before 2026-09-18 do not, and partition 42 is
   paired by construction there).
2. **Per partition.** Macro CV AUC of each model, the macro Δ, and the §13 test
   (`results_helpers.corrected_t`, Nadeau-Bengio) run on that partition alone —
   i.e. what a single-split analysis would have concluded, five times over.
3. **Is the single-split test calibrated?** The SD of the macro Δ across
   partitions is partition noise, measured. The Nadeau-Bengio SE from one
   partition is the formula's estimate of it (it also prices training-set
   sampling, so it should be at least as large). Ratio > 1 means the §13 tests
   are too optimistic. With 5 partitions the SD is itself uncertain; its 95% CI
   is shown. For mdcnn, the SD across the 5 seeds of seeds_19loci (training
   noise, partition fixed) is shown next to its SD across partitions.
4. **The test on all 25 folds.** Corrected repeated k-fold t (Bouckaert & Frank
   2004): per drug, the r·k fold differences with variance inflated by
   (1/(r·k) + n_test/n_train); pooled over drugs with Satterthwaite df as in §13.
   TOST at ±0.01 on the same SE. One pre-specified comparison, so no Holm.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP.parent.parent))
from results_helpers import K, MARGIN, RHO, corrected_t  # noqa: E402

DRUGS = ["AMIKACIN", "CAPREOMYCIN", "ETHAMBUTOL", "ETHIONAMIDE", "ISONIAZID",
         "KANAMYCIN", "LEVOFLOXACIN", "MOXIFLOXACIN", "PYRAZINAMIDE",
         "RIFAMPICIN", "STREPTOMYCIN"]
MODELS = ["locusfusion_v2", "mdcnn"]
# partition 42 is read from the seed-0 runs that already exist
BASE = {"locusfusion_v2": EXP / "locusfusion_v2" / "sd19_all_modalities__locusfusion_v2",
        "mdcnn": EXP / "seeds_19loci" / "s0_all_modalities__mdcnn"}
SEEDS_MDCNN = [EXP / "seeds_19loci" / f"s{s}_all_modalities__mdcnn"
               for s in (0, 1000, 2000, 3000, 4000)]


def cell(cell_dir):
    """drug -> (5 fold AUCs, 5 fold fingerprints or None) for one cell folder."""
    out = {}
    for p in Path(cell_dir).glob("*.json"):
        if p.name == "weights_location.json":
            continue
        j = json.loads(p.read_text())
        f = j.get("cv_folds") or []
        aucs = [x["auc"] for x in f if x.get("auc") is not None]
        if j.get("drug") in DRUGS and len(aucs) == K:
            fps = [x.get("val_ids_sha1") for x in f]
            out[j["drug"]] = (np.asarray(aucs), fps if all(fps) else None)
    return out


def load():
    """partition -> model -> {drug: (aucs, fingerprints)}, complete partitions only."""
    dirs = {42: dict(BASE)}
    for d in sorted(HERE.glob("cv*_all_modalities__*")):
        if d.name.startswith("canary"):
            continue
        cv, model = int(d.name.split("_")[0][2:]), d.name.rsplit("__", 1)[1]
        dirs.setdefault(cv, {})[model] = d
    done, partial = {}, []
    for cv, by in sorted(dirs.items()):
        got = {m: cell(by[m]) if m in by else {} for m in MODELS}
        n = {m: len(got[m]) for m in MODELS}
        if all(v == len(DRUGS) for v in n.values()):
            done[cv] = got
        else:
            partial.append(f"partition {cv}: " + ", ".join(f"{m} {v}/11" for m, v in n.items()))
    return done, partial


def auc_frame(got):
    return pd.DataFrame({d: a for d, (a, _) in got.items()}).T.loc[DRUGS]


def repeated_t(diffs):
    """Corrected repeated k-fold t on a list of drug x fold Δ frames (one per partition)."""
    X = pd.concat(diffs, axis=1)                    # drug x (r*k)
    rk = X.shape[1]
    v = (1 / rk + RHO) * X.var(axis=1, ddof=1)
    se = np.sqrt(v.sum()) / len(X)
    df = v.sum() ** 2 / ((v ** 2).sum() / (rk - 1))
    M = X.values.mean()
    h = stats.t.ppf(0.975, df) * se
    return dict(delta=M, lo=M - h, hi=M + h, df=df, se=se,
                p=2 * stats.t.sf(abs(M / se), df),
                p_eq=max(stats.t.sf((M + MARGIN) / se, df), stats.t.cdf((M - MARGIN) / se, df)))


def sd_ci(x):
    """SD of x with a chi-square 95% CI."""
    n, s = len(x), np.std(x, ddof=1)
    lo = np.sqrt((n - 1) * s ** 2 / stats.chi2.ppf(0.975, n - 1))
    hi = np.sqrt((n - 1) * s ** 2 / stats.chi2.ppf(0.025, n - 1))
    return s, lo, hi


def main():
    done, partial = load()
    for line in partial:
        print("incomplete --", line)
    if not done:
        print("no complete partitions yet")
        return
    P = sorted(done)

    # 1. pairing
    checked = mismatched = unverifiable = 0
    for cv in P:
        a, b = done[cv]["locusfusion_v2"], done[cv]["mdcnn"]
        for d in DRUGS:
            fa, fb = a[d][1], b[d][1]
            if fa is None or fb is None:
                unverifiable += 1
            elif fa == fb:
                checked += 1
            else:
                mismatched += 1
                print(f"PAIRING BROKEN: partition {cv}, {d}")
    print(f"\npairing: {checked} drug-partitions verified by fold fingerprint, "
          f"{mismatched} mismatched, {unverifiable} unverifiable (older results)")
    if mismatched:
        print("  -> the paired tests below are not valid until this is fixed")

    # 2. per partition
    diffs, rows = [], []
    for cv in P:
        lf, md = auc_frame(done[cv]["locusfusion_v2"]), auc_frame(done[cv]["mdcnn"])
        d = lf - md
        diffs.append(d)
        r = corrected_t(d)
        se = np.sqrt(((1 / K + RHO) * d.var(axis=1, ddof=1)).sum()) / len(DRUGS)
        rows.append(dict(partition=cv, locusfusion_v2=lf.values.mean(), mdcnn=md.values.mean(),
                         delta=d.values.mean(), ahead=int((d.mean(axis=1) > 0).sum()),
                         nb_se=se, p=r["p"], p_tost=r["p_eq"]))
    t = pd.DataFrame(rows).set_index("partition")
    print("\nper partition (single-split §13 test on each)")
    print(t.to_string(formatters={
        "locusfusion_v2": "{:.4f}".format, "mdcnn": "{:.4f}".format,
        "delta": "{:+.4f}".format, "ahead": "{}/11".format, "nb_se": "{:.4f}".format,
        "p": "{:.4f}".format, "p_tost": "{:.4f}".format}))
    print(f"  single splits calling it different (p < 0.05): {int((t.p < 0.05).sum())}/{len(t)}")

    # 3. calibration
    if len(P) > 1:
        s, lo, hi = sd_ci(t.delta.values)
        pred = t.nb_se.mean()
        print("\npartition noise, measured vs the single-split formula")
        print(f"  SD of macro Δ across {len(P)} partitions: {s:.4f}  (95% CI {lo:.4f}-{hi:.4f})")
        print(f"  mean Nadeau-Bengio SE from one partition: {pred:.4f}")
        print(f"  ratio measured / formula: {s / pred:.2f}  "
              f"(> 1: single-split tests too optimistic; <= 1: conservative, as intended)")
        for m in MODELS:
            sm, lm, hm = sd_ci(t[m].values)
            print(f"  {m:<15} macro CV SD across partitions {sm:.4f} (95% CI {lm:.4f}-{hm:.4f})")
        seeds = [cell(p) for p in SEEDS_MDCNN]
        if all(len(x) == len(DRUGS) for x in seeds):
            ss = np.std([auc_frame(x).values.mean() for x in seeds], ddof=1)
            print(f"  mdcnn           macro CV SD across 5 training seeds {ss:.4f} "
                  f"(seeds_19loci; partition fixed)")

    # 4. the test on every fold
    r = repeated_t(diffs)
    print(f"\nlocusfusion_v2 − mdcnn on {len(P)} partitions x {K} folds (corrected repeated k-fold t)")
    print(f"  Δ {r['delta']:+.4f}   95% CI [{r['lo']:+.4f}, {r['hi']:+.4f}]   df {r['df']:.1f}")
    print(f"  difference p = {r['p']:.4g}   TOST (±{MARGIN}) p = {r['p_eq']:.4g}")
    if len(P) < 5:
        print(f"  ({len(P)} of 5 partitions -- interim)")


if __name__ == "__main__":
    main()
