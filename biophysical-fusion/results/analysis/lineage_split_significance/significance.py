"""Significance tests for the lineage-split retrain (results/experiments/lineage_split).

Two parts, one Holm family:
  A. within the lineage split: the paper's fold-paired tests (exact Wilcoxon over
     55 fold differences, HL 95% CI), for every paper row whose cells were retrained.
     LASSO is the 19-locus one-hot only.
  B. lineage vs random: folds don't pair across partitions, so pair on drugs
     (exact Wilcoxon over 11 per-drug differences). Each row is a difference of
     a contrast (or a raw cell) between the two splits.

Run from biophysical-fusion/:  python results/analysis/lineage_split_significance/significance.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from results_frame import _fold_aucs, STD_MODSETS as M5
from results_helpers import WilcoxonComparisons, fold_matrix, hodges_lehmann, holm, _PD, _L19
from bigtb_baselines import ALL_DRUGS

EXP = PROJECT / "results" / "experiments"
OUT = Path(__file__).resolve().parent
A, EST3 = "all_modalities", ["mdcnn", "cisfusion", "late_fusion"]


def read_dir(d, task, model, modset):
    rows = []
    for p in sorted(d.glob("*.json")):
        if p.name == "weights_location.json":
            continue
        j = json.loads(p.read_text())
        if "cv_auc_mean" not in j:
            continue
        rows.append(dict(task=task, model=model, modset=modset, drug=j["drug"],
                         fold_aucs=_fold_aucs(j), cv=j["cv_auc_mean"],
                         n_params=j.get("n_params"), src=str(d.relative_to(EXP))))
    return rows


def build(split):
    rows = []
    for task, pre, rand in [(_PD, "sd_", "full_run_v2"), (_L19, "sd19_", "alllocus_run_v2")]:
        for m in M5:
            for a in EST3:
                d = EXP / ("lineage_split" if split == "lineage" else rand) / (
                    f"{pre}{m}__{a}" if split == "lineage" else f"{m}__{a}")
                rows += read_dir(d, task, a, m)
            d = EXP / ("lineage_split" if split == "lineage" else "locusfusion_v3") / f"{pre}{m}_unk__locusfusion_v2"
            rows += read_dir(d, task, "locusfusion", m)
    lasso = (EXP / "lineage_split/lasso_onehot_all" if split == "lineage"
             else EXP / "lasso_wholeseq_20260901/onehot_all")
    rows += read_dir(lasso, _L19, "lasso_onehot", "dna")
    return pd.DataFrame(rows)


cache = OUT / "fold_aucs.pkl"
if cache.exists():
    lin, ran = pd.read_pickle(cache)
else:
    lin, ran = build("lineage"), build("random")
    pd.to_pickle((lin, ran), cache)
for name, f in [("lineage", lin), ("random", ran)]:
    dup = f.duplicated(["task", "model", "modset", "drug"]).sum()
    print(f"{name}: {len(f)} cells, {dup} duplicates, "
          f"{(f.fold_aucs.map(len) != 5).sum()} without 5 folds")

# ---- A. within the lineage split, fold-paired (paper rows that were retrained)
c = WilcoxonComparisons(lin, ALL_DRUGS)
G = "A: lineage split, fold-paired"
c.folds(G, "lasso_onehot − mdcnn · 19 loci", (_L19, "lasso_onehot", "dna"), (_L19, "mdcnn", A))
c.folds(G, "locusfusion − lasso_onehot · 19 loci", (_L19, "locusfusion", A), (_L19, "lasso_onehot", "dna"))
c.folds(G, "locusfusion − mdcnn · 19 loci · 5 modsets", (_L19, "locusfusion", M5), (_L19, "mdcnn", M5))
c.folds(G, "established: 19 − per-drug loci · 5 modsets", (_L19, EST3, M5), (_PD, EST3, M5))
c.folds(G, "locusfusion: 19 − per-drug loci · 5 modsets", (_L19, "locusfusion", M5), (_PD, "locusfusion", M5))
c.folds(G, "established: all_modalities − dna · per-drug loci", (_PD, EST3, A), (_PD, EST3, "dna"))
npar = lin[(lin.task == _L19) & lin.model.isin(EST3)].groupby("model").n_params.median()
order = sorted(EST3, key=npar.get)
for i, a in enumerate(order):
    for b in order[i + 1:]:
        c.folds(G, f"{a} ({npar[a] / 1e6:.1f}M) − {b} ({npar[b] / 1e6:.1f}M) · 19 loci · 5 modsets",
                (_L19, a, M5), (_L19, b, M5))
rows = c.rows


# ---- B. lineage vs random, drug-paired
def per_drug(frame, a, b=None):
    v = fold_matrix(frame, ALL_DRUGS, *a).mean(axis=1)
    return v - fold_matrix(frame, ALL_DRUGS, *b).mean(axis=1) if b else v


def across(label, a, b=None):
    per = per_drug(lin, a, b) - per_drug(ran, a, b)
    hl, lo, hi = hodges_lehmann(per.values)
    rows.append(dict(group="B: lineage − random, drug-paired", comparison=label, n=len(per),
                     delta=per.mean(), hl=hl, lo=lo, hi=hi,
                     wins=f"{int((per > 0).sum())}/{len(per)}",
                     p=stats.wilcoxon(per.values, method="exact").pvalue))


across("Δ[locusfusion: 19 − per-drug loci · 5 modsets]", (_L19, "locusfusion", M5), (_PD, "locusfusion", M5))
across("Δ[established: 19 − per-drug loci · 5 modsets]", (_L19, EST3, M5), (_PD, EST3, M5))
across("Δ[locusfusion − mdcnn · 19 loci · 5 modsets]", (_L19, "locusfusion", M5), (_L19, "mdcnn", M5))
across("Δ[locusfusion − lasso_onehot · 19 loci]", (_L19, "locusfusion", A), (_L19, "lasso_onehot", "dna"))
across("locusfusion · 19 loci · all_modalities", (_L19, "locusfusion", A))
across("mdcnn · 19 loci · all_modalities", (_L19, "mdcnn", A))
across("lasso_onehot · 19 loci", (_L19, "lasso_onehot", "dna"))

t = pd.DataFrame(rows)
t["p_holm"] = holm(t.p)
t["verdict"] = np.where(t.p_holm < 0.05, "significant", "not significant")
t.to_csv(OUT / "significance.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 60)
print(t[["group", "comparison", "n", "delta", "hl", "lo", "hi", "wins", "p", "p_holm", "verdict"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
