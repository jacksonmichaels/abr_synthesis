"""Collect the locus_sets sweep (and its lineage_split controls) into one frame,
and run the fold-paired significance family.

    from collect import load, significance, status
    F = load()              # one row per (setting, locus_set, model, modset, drug)
    status(F)               # cells done / expected
    significance(F)         # Wilcoxon table, Holm over every row that could run

Columns: setting ('single'|'multi'), locus_set ('bigtb'|'bigtb19'|'who_t12'|'lab64'),
model ('locusfusion'|'mdcnn'|'lasso'), modset, drug, fold_aucs (5 CV AUCs), cv,
test, n_params, n_loci, src. `task` = f"{setting}:{locus_set}" for results_helpers.

Single-drug controls are lineage_split/{sd,sd19}_* (per-drug BIG-TB loci / 19 loci)
and lineage_split/lasso_onehot_{perdrug,all}; they were built on data_snapshot's
FASTAs, the new arms on locus_sets/aligned (repaired eis/embB/ethA/rrs). Multi-drug
controls (bigtb19) were trained in this sweep on the repaired FASTAs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from results_frame import _fold_aucs, STD_MODSETS as M5          # noqa: E402
from results_helpers import WilcoxonComparisons                  # noqa: E402
from bigtb_baselines import ALL_DRUGS                            # noqa: E402

EXP = PROJECT / "results" / "experiments"
HERE = EXP / "locus_sets"
LIN = EXP / "lineage_split"
LOCUS_SETS = ["bigtb", "bigtb19", "who_t12", "lab64"]
NEW_SETS = ["who_t12", "lab64"]
LABEL = {"bigtb": "Big-TB per-drug", "bigtb19": "Big-TB 19 loci",
         "who_t12": "WHO tier 1+2", "lab64": "64 loci"}
MODEL_LABEL = {"locusfusion": "LocusFusion", "mdcnn": "stacked CNN", "lasso": "LASSO"}
SUFFIX = {"locusfusion": "_unk__locusfusion_v2", "mdcnn": "__mdcnn"}


def _single_dir(d, locus_set, model, modset):
    rows = []
    if not d.is_dir():
        return rows
    for p in sorted(d.glob("*.json")):
        if p.name == "weights_location.json" or p.name.startswith("multidrug"):
            continue
        j = json.loads(p.read_text())
        if "cv_auc_mean" not in j:
            continue
        rows.append(dict(setting="single", locus_set=locus_set, model=model, modset=modset,
                         drug=j["drug"], fold_aucs=_fold_aucs(j), cv=j["cv_auc_mean"],
                         test=(j.get("test") or {}).get("auc"), n_params=j.get("n_params"),
                         n_loci=len(j.get("genes") or ()), src=str(d.relative_to(EXP))))
    return rows


def _multi_dir(d, locus_set, model, modset):
    rows = []
    if not d.is_dir():
        return rows
    for p in sorted(d.glob("multidrug__*.json")):
        j = json.loads(p.read_text())
        if "cv_per_drug_auc" not in j:
            continue
        for drug in j["drugs"]:
            folds = tuple(f["per_drug"][drug]["auc"] for f in j["cv_folds"]
                          if f.get("per_drug", {}).get(drug, {}).get("auc") is not None)
            rows.append(dict(setting="multi", locus_set=locus_set, model=model, modset=modset,
                             drug=drug, fold_aucs=folds, cv=j["cv_per_drug_auc"][drug],
                             test=j["test_per_drug"][drug]["auc"], n_params=j.get("n_params"),
                             n_loci=len(j.get("loci") or ()), src=str(d.relative_to(EXP))))
    return rows


def load():
    rows = []
    for model, suf in SUFFIX.items():
        for m in M5:
            rows += _single_dir(LIN / f"sd_{m}{suf}", "bigtb", model, m)
            rows += _single_dir(LIN / f"sd19_{m}{suf}", "bigtb19", model, m)
            for s in NEW_SETS:
                rows += _single_dir(HERE / f"{s}_sd_{m}{suf}", s, model, m)
            for s in ["bigtb19"] + NEW_SETS:
                rows += _multi_dir(HERE / f"{s}_multidrug_{m}{suf}", s, model, m)
    for s, d in [("bigtb", LIN / "lasso_onehot_perdrug"), ("bigtb19", LIN / "lasso_onehot_all"),
                 ("who_t12", HERE / "lasso_onehot_who_t12"), ("lab64", HERE / "lasso_onehot_lab64")]:
        rows += _single_dir(d, s, "lasso", "dna")
    F = pd.DataFrame(rows, columns=["setting", "locus_set", "model", "modset", "drug",
                                    "fold_aucs", "cv", "test", "n_params", "n_loci", "src"])
    F["task"] = F.setting + ":" + F.locus_set
    dup = F.duplicated(["setting", "locus_set", "model", "modset", "drug"])
    if dup.any():
        raise ValueError(f"{dup.sum()} duplicate cells:\n{F[dup]}")
    return F


def expected():
    """Every cell the sweep + its controls should produce."""
    rows = []
    for model in SUFFIX:
        for m in M5:
            for s in LOCUS_SETS:
                rows += [("single", s, model, m, d) for d in ALL_DRUGS]
            for s in ["bigtb19"] + NEW_SETS:
                rows += [("multi", s, model, m, d) for d in ALL_DRUGS]
    for s in LOCUS_SETS:
        rows += [("single", s, "lasso", "dna", d) for d in ALL_DRUGS]
    return pd.DataFrame(rows, columns=["setting", "locus_set", "model", "modset", "drug"])


def status(F):
    """Cells done / expected per (setting, model, locus_set), modsets x drugs pooled."""
    E = expected()
    keys = ["setting", "model", "locus_set"]
    done = (E.merge(F[keys + ["modset", "drug"]], how="left", indicator=True)
             .assign(done=lambda t: t._merge == "both"))
    t = done.groupby(keys).done.agg(["sum", "size"])
    return (t["sum"].astype(str) + "/" + t["size"].astype(str)).unstack("locus_set")[
        [s for s in LOCUS_SETS if s in t.index.get_level_values("locus_set")]]


def macro(F, value="cv"):
    """Macro (mean over the 11 drugs) per (setting, model, modset, locus_set);
    NaN unless all 11 drugs are in."""
    g = F.groupby(["setting", "model", "modset", "locus_set"])[value]
    m = g.mean().where(g.count() == len(ALL_DRUGS))
    return m.unstack("locus_set").reindex(columns=[s for s in LOCUS_SETS])


def significance(F):
    """Fold-paired exact Wilcoxon (11 drugs x 5 folds = 55) per row, HL 95% CI,
    Holm over the rows that ran. Network rows average the five modality sets
    fold-by-fold (the paper's convention); LASSO is dna only. Single- vs
    multi-drug runs sit on different partitions, so those rows pair on drugs."""
    W = WilcoxonComparisons(F, ALL_DRUGS)
    pending = []

    def run(fn, group, label, *ab):
        try:
            fn(group, label, *ab)
        except ValueError:
            pending.append((group, label))

    nets = [("locusfusion", M5), ("mdcnn", M5), ("lasso", "dna")]
    for model, ms in nets:
        G = f"single-drug · {MODEL_LABEL[model]}"
        for a, b in [("who_t12", "bigtb"), ("lab64", "bigtb19"), ("lab64", "who_t12")]:
            run(W.folds, G, f"{LABEL[a]} − {LABEL[b]}",
                (f"single:{a}", model, ms), (f"single:{b}", model, ms))
    for s in NEW_SETS:
        run(W.folds, "single-drug · model", f"LocusFusion − stacked CNN · {LABEL[s]}",
            (f"single:{s}", "locusfusion", M5), (f"single:{s}", "mdcnn", M5))
    for model in ["locusfusion", "mdcnn"]:
        G = f"multi-drug · {MODEL_LABEL[model]}"
        for a, b in [("who_t12", "bigtb19"), ("lab64", "bigtb19"), ("lab64", "who_t12")]:
            run(W.folds, G, f"{LABEL[a]} − {LABEL[b]}",
                (f"multi:{a}", model, M5), (f"multi:{b}", model, M5))
        for s in ["bigtb19"] + NEW_SETS:
            run(W.drugs_only, f"multi − single · {MODEL_LABEL[model]}", LABEL[s],
                (f"multi:{s}", model, M5), (f"single:{s}", model, M5))
    t = W.table() if W.rows else pd.DataFrame()
    return t, pd.DataFrame(pending, columns=["group", "comparison"])
