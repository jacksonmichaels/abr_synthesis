"""The paper's Table 1 family and in-text numbers, recomputed on the lineage-grouped split.

Lineage split is now the paper's base evaluation setting. Reuses the cell loader of
../lineage_split_significance/significance.py (its fold_aucs.pkl), and adds the per-drug
LASSO. Holm is over the 11 Table 1 rows that exist on the lineage split (the uncalled-base
ablation, whole-genome LASSO and delta-encoding LASSO were not run on it).

Run from biophysical-fusion/:  python results/analysis/lineage_paper_family/paper_family.py
(> paper_family.log). scripts/build_display_figures.py imports load_lineage(), family(),
drugmean() and spearman_cells() from here, so the manuscript figures and tables use
exactly these numbers.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from results_frame import _fold_aucs, STD_MODSETS as M5
from results_helpers import WilcoxonComparisons, fold_matrix, holm, _PD, _L19
from bigtb_baselines import ALL_DRUGS

EXP = PROJECT / "results" / "experiments"
OUT = Path(__file__).resolve().parent
A, EST3 = "all_modalities", ["mdcnn", "cisfusion", "late_fusion"]



def load_lineage():
    """The lineage-split cells: significance.py's fold_aucs.pkl plus the per-drug LASSO."""
    lin, _ = pd.read_pickle(PROJECT / "results/analysis/lineage_split_significance/fold_aucs.pkl")
    rows = []
    for p in sorted((EXP / "lineage_split/lasso_onehot_perdrug").glob("*.json")):
        j = json.loads(p.read_text())
        rows.append(dict(task=_PD, model="lasso_onehot", modset="dna", drug=j["drug"],
                         fold_aucs=_fold_aucs(j), cv=j["cv_auc_mean"], n_params=j.get("n_params"),
                         src="lineage_split/lasso_onehot_perdrug"))
    lin = pd.concat([lin, pd.DataFrame(rows)], ignore_index=True)
    assert not lin.duplicated(["task", "model", "modset", "drug"]).any()
    return lin


def family(lin):
    """Table 1 on the lineage split: 11 fold-paired Wilcoxon rows, Holm over the 11."""
    c = WilcoxonComparisons(lin, ALL_DRUGS)
    G = "paper family, lineage split"
    c.folds(G, "lasso − stacked · 19", (_L19, "lasso_onehot", "dna"), (_L19, "mdcnn", A))
    c.folds(G, "lasso − stacked · per-drug", (_PD, "lasso_onehot", "dna"), (_PD, "mdcnn", A))
    c.folds(G, "LF − lasso · 19", (_L19, "locusfusion", A), (_L19, "lasso_onehot", "dna"))
    c.folds(G, "LF − stacked · 19 · 5 modsets", (_L19, "locusfusion", M5), (_L19, "mdcnn", M5))
    c.folds(G, "locus set · established · 5 modsets", (_L19, EST3, M5), (_PD, EST3, M5))
    c.folds(G, "locus set · LF · 5 modsets", (_L19, "locusfusion", M5), (_PD, "locusfusion", M5))
    c.folds(G, "locus set · lasso", (_L19, "lasso_onehot", "dna"), (_PD, "lasso_onehot", "dna"))
    c.folds(G, "modalities · established · all − dna · per-drug", (_PD, EST3, A), (_PD, EST3, "dna"))
    c.folds(G, "stacked − late · 19 · 5 modsets", (_L19, "mdcnn", M5), (_L19, "late_fusion", M5))
    c.folds(G, "stacked − cis · 19 · 5 modsets", (_L19, "mdcnn", M5), (_L19, "cisfusion", M5))
    c.folds(G, "cis − late · 19 · 5 modsets", (_L19, "cisfusion", M5), (_L19, "late_fusion", M5))
    t = pd.DataFrame(c.rows)
    t["p_holm"] = holm(t.p)
    t["verdict"] = np.where(t.p_holm < 0.05, "significant", "not significant")
    return t


def drugmean(lin, task, model, ms):
    """Per-drug mean fold AUC over the (model, modset) cells."""
    return fold_matrix(lin, ALL_DRUGS, task, model, ms).mean(axis=1)


def spearman_cells(lin):
    """The 20 network cells at 19 loci (mean CV AUC, median n_params) and Spearman
    rho over all of them and over the 15 established-model cells."""
    cells = lin[(lin.task == _L19) & lin.model.isin(EST3 + ["locusfusion"]) & lin.modset.isin(M5)]
    g = cells.groupby(["model", "modset"]).agg(cv=("cv", "mean"), n=("n_params", "median"))
    e = g.loc[EST3]
    return g, stats.spearmanr(g.n, g.cv), stats.spearmanr(e.n, e.cv)


def main():
    lin = load_lineage()
    t = family(lin)
    t.to_csv(OUT / "paper_family_lineage.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print(t[["comparison", "n", "delta", "hl", "lo", "hi", "wins", "p", "p_holm", "verdict"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    dm = lambda task, model, ms: drugmean(lin, task, model, ms)

    print("\n== macro CV AUC (all_modalities; LASSO dna)")
    for task in (_PD, _L19):
        for m in ["locusfusion"] + EST3 + ["lasso_onehot"]:
            ms = "dna" if m == "lasso_onehot" else A
            print(f"{task:28s} {m:14s} {dm(task, m, ms).mean():.4f}")

    print("\n== per-model locus-set gain (all_modalities)")
    for m in EST3 + ["locusfusion"]:
        d = dm(_L19, m, A) - dm(_PD, m, A)
        print(f"{m:12s} {d.mean():+.4f}  {(d > 0).sum()}/11")
    print("per-model locus-set gain, 5 modsets:")
    for m in EST3 + ["locusfusion"]:
        d = dm(_L19, m, M5) - dm(_PD, m, M5)
        print(f"{m:12s} {d.mean():+.4f}  {(d > 0).sum()}/11")

    print("\n== modality gain over dna, per drug (established mean / LF)")
    for task in (_PD, _L19):
        for who, ms_models in [("established", EST3), ("LF", "locusfusion")]:
            base = dm(task, ms_models, "dna")
            tab = pd.DataFrame({ms: dm(task, ms_models, ms) - base for ms in M5[1:]})
            print(f"-- {who} | {task}"); print(tab.round(4).to_string()); print("macro", tab.mean().round(4).to_dict())

    print("\n== LF − LASSO and LF − stacked per drug, 19 loci")
    print(pd.DataFrame({"LF-lasso": dm(_L19, "locusfusion", A) - dm(_L19, "lasso_onehot", "dna"),
                        "LF-stacked(5ms)": dm(_L19, "locusfusion", M5) - dm(_L19, "mdcnn", M5)}).round(4).to_string())

    print("\n== Spearman: 20 network cells at 19 loci vs n_params")
    g, r, re_ = spearman_cells(lin)
    print("all", len(g), r)
    print("established", len(g.loc[EST3]), re_)
    print(g.round(4).to_string())

    print("\n== LASSO support (cv_support_mean), lineage vs random, 19 loci")
    for name, d in [("lineage", EXP / "lineage_split/lasso_onehot_all"), ("random", EXP / "lasso_wholeseq_20260901/onehot_all")]:
        v = [json.loads(p.read_text()).get("cv_support_mean") for p in d.glob("*.json")]
        v = [x for x in v if x is not None]; print(name, len(v), np.mean(v))


if __name__ == "__main__":
    main()
