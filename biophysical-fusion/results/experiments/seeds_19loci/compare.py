"""locusfusion − mdcnn at 19 loci, all_modalities, across seeds.

    python results/experiments/seeds_19loci/compare.py

Reads whatever has landed. A seed counts only once both models have all 11
drugs with 5 folds; partial seeds are listed and skipped.

Three read-outs:

1. **Code drift.** Seed 0 here against the original seed-0 runs
   (`locusfusion_fixed/sd19_all_modalities__locusfusion`,
   `alllocus_run_v2/all_modalities__mdcnn`). Same seed, same folds, current
   code; a macro gap well beyond ~0.002 means the code changed the model and the
   original grid is not the same experiment as this one.
2. **Per-seed macro Δ** and its SD across seeds -- the seed noise §17 could not see.
3. **The test.** Every seed shares one CV partition, so there are two noise
   sources and neither is visible to a test on the other:
     * seed: SD of the per-seed macro Δ, over S seeds -> var_seed = s² / S
     * partition: Nadeau-Bengio corrected variance of the seed-AVERAGED
       drug x fold Δ matrix (as in results.ipynb §17) -> var_fold
   SE = sqrt(var_seed + var_fold). This double-counts a little -- the
   seed-averaged fold Δ still carries seed noise / S -- so it is conservative.
   df by Satterthwaite over the two components. TOST at ±0.01 on the same SE.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
DRUGS = ["AMIKACIN", "CAPREOMYCIN", "ETHAMBUTOL", "ETHIONAMIDE", "ISONIAZID",
         "KANAMYCIN", "LEVOFLOXACIN", "MOXIFLOXACIN", "PYRAZINAMIDE",
         "RIFAMPICIN", "STREPTOMYCIN"]
MODELS = ["locusfusion", "mdcnn"]
K, RHO, MARGIN = 5, 1 / 4, 0.01
ORIGINAL = {"locusfusion": EXP / "locusfusion_fixed" / "sd19_all_modalities__locusfusion",
            "mdcnn": EXP / "alllocus_run_v2" / "all_modalities__mdcnn"}


def folds_in(cell_dir):
    """drug -> 5 fold AUCs, for every drug JSON in one cell folder."""
    out = {}
    for p in Path(cell_dir).glob("*.json"):
        if p.name == "weights_location.json":
            continue
        j = json.loads(p.read_text())
        aucs = [f["auc"] for f in j.get("cv_folds") or [] if f.get("auc") is not None]
        if j.get("drug") in DRUGS and len(aucs) == K:
            out[j["drug"]] = np.asarray(aucs)
    return out


def load():
    """seed -> model -> drug x fold DataFrame, complete seeds only."""
    seeds, partial = {}, []
    for d in sorted(HERE.glob("s*_all_modalities__*")):
        seed, model = int(d.name.split("_")[0][1:]), d.name.rsplit("__", 1)[1]
        seeds.setdefault(seed, {})[model] = folds_in(d)
    done = {}
    for s, by in sorted(seeds.items()):
        n = {m: len(by.get(m, {})) for m in MODELS}
        if all(v == len(DRUGS) for v in n.values()):
            done[s] = {m: pd.DataFrame(by[m]).T.loc[DRUGS] for m in MODELS}
        else:
            partial.append(f"seed {s}: " + ", ".join(f"{m} {v}/11" for m, v in n.items()))
    return done, partial


def main():
    done, partial = load()
    for line in partial:
        print("incomplete --", line)
    if not done:
        print("no complete seeds yet")
        return

    # 1. drift
    if 0 in done:
        print("\ncode drift: seed 0 rerun − original seed-0 run (macro CV AUC)")
        for m in MODELS:
            orig = pd.DataFrame(folds_in(ORIGINAL[m])).T.loc[DRUGS]
            new = done[0][m]
            d = new.mean(axis=1) - orig.mean(axis=1)
            print(f"  {m:<12} original {orig.values.mean():.4f}  rerun {new.values.mean():.4f}"
                  f"  Δ {d.mean():+.4f}  (max |per-drug Δ| {d.abs().max():.4f})")

    # 2. per seed
    S = sorted(done)
    diff = {s: done[s]["locusfusion"] - done[s]["mdcnn"] for s in S}
    per_seed = pd.Series({s: diff[s].values.mean() for s in S})
    print("\nper-seed macro CV AUC")
    print(f"  {'seed':>6}{'locusfusion':>13}{'mdcnn':>9}{'Δ':>9}{'drugs ahead':>13}")
    for s in S:
        lf, md = done[s]["locusfusion"].values.mean(), done[s]["mdcnn"].values.mean()
        ahead = int((diff[s].mean(axis=1) > 0).sum())
        print(f"  {s:>6}{lf:>13.4f}{md:>9.4f}{lf - md:>+9.4f}{ahead:>10}/11")
    if len(S) > 1:
        print(f"  SD of macro Δ across seeds: {per_seed.std(ddof=1):.4f}   "
              f"(n = {len(S)})")

    # 3. test
    avg = sum(diff.values()) / len(S)                      # drug x fold, seed-averaged
    v_d = (1 / K + RHO) * avg.var(axis=1, ddof=1)
    var_fold = v_d.sum() / len(DRUGS) ** 2
    df_fold = v_d.sum() ** 2 / ((v_d ** 2).sum() / (K - 1))
    M = avg.values.mean()
    if len(S) > 1:
        var_seed = per_seed.var(ddof=1) / len(S)
        df_seed = len(S) - 1
        var = var_seed + var_fold
        df = var ** 2 / (var_seed ** 2 / df_seed + var_fold ** 2 / df_fold)
    else:
        var_seed, var, df = float("nan"), var_fold, df_fold
    se = np.sqrt(var)
    h = stats.t.ppf(0.975, df) * se
    p = 2 * stats.t.sf(abs(M / se), df)
    p_eq = max(stats.t.sf((M + MARGIN) / se, df), stats.t.cdf((M - MARGIN) / se, df))
    print(f"\nlocusfusion − mdcnn, averaged over {len(S)} seed(s)")
    print(f"  Δ {M:+.4f}   95% CI [{M - h:+.4f}, {M + h:+.4f}]   df {df:.1f}")
    print(f"  SE² = seed {var_seed:.2e} + partition {var_fold:.2e}")
    print(f"  difference p = {p:.4g}   TOST (±{MARGIN}) p = {p_eq:.4g}")
    if len(S) < 5:
        print(f"  ({len(S)} of 5 seeds -- interim)")


if __name__ == "__main__":
    main()
