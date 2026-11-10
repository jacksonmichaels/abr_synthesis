"""GenomeFusion results, in the schema `locus_sets/collect.py` uses.

    from genomefusion.results import load, fold_provenance
    G = load()                      # one row per (locus_set, drug)
    fold_provenance()               # the val-ID hashes, GF beside the baselines

So that one frame can hold GenomeFusion and the locus_sets baselines:

    import collect                  # results/experiments/locus_sets
    F = pd.concat([collect.load(), load()], ignore_index=True)

Columns match `collect.load()` exactly -- setting, locus_set, model, modset,
drug, fold_aucs, cv, test, n_params, n_loci, src, task -- so
`results_helpers.fold_matrix` and `WilcoxonComparisons` work on the union with
no special cases.

Two deliberate differences from the baseline rows:

`modset` is **gf_token**, not one of the five standard modality sets. The
57-channel GenomeFusion token is not any of them: it carries base, residue,
biophysical and delta-biophysical values, an 8-way consequence, truncation,
region, upstream distance, span, phase and flags in one vector. The nearest
baseline cell is `all_modalities`, and that is the comparison the notebook
makes, but it is not a modality-matched pair and must not be reported as one.

`locus_set` is **gf19** (the 19 v3/v5 loci, a parity arm) or **genome** (all
6,776 loci of the strict one-locus-per-base partition).

Rows whose run kept fewer than 5 folds are returned with the folds they have.
`fold_matrix` requires all 5 and will refuse them, which is the intended
behaviour: a 1-fold canary cannot enter a 55-fold paired test.
"""
import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
EXP = PROJECT / "results" / "experiments"
GF = EXP / "genomefusion"

# run folder -> the locus universe it trained on
LADDER = (64, 128, 256, 512, 1024, 2048, 4096, 6776)

ARMS = {"gf19": "gf19", "gf19r": "gf19r", "gf19r_nofeat": "gf19r_nofeat", "gf19p": "gf19p", "canary": "genome",
        "gf_genome": "genome", "gf_minus19p": "minus19p",
        "gf_minus19": "minus19",
        "gf_who_t12": "gf_who_t12", "gf_who_t12r": "gf_who_t12r",
        "gf_lab64": "gf_lab64", "gf_lab64r": "gf_lab64r",
        **{f"size{n}": f"size{n}" for n in LADDER}}

LABEL = {"gf19": "19 loci, no promoter bases", "gf19r": "19 loci + WHO regulatory",
         "gf19r_nofeat": "19 loci + WHO regulatory, no copied target features",
         "gf19p": "19 loci + 100bp flank (parity)",
         "genome": "whole genome",
         "minus19p": "genome minus the 19 and their promoters",
         "minus19": "genome minus the 19 (KEEPS their promoters)",
         "gf_who_t12": "WHO tier 1+2 genes, no promoter bases",
         "gf_who_t12r": "WHO tier 1+2 genes + WHO regulatory",
         "gf_lab64": "lab64 genes, no promoter bases",
         "gf_lab64r": "lab64 genes + WHO regulatory",
         **{f"size{n}": f"{n:,} loci" for n in LADDER}}


def ladder_frame(F=None):
    """The size ladder as one frame: n_loci vs CV AUC, for the dose-response.

    Rungs are nested by construction (one ladder seed), so a decline along
    `n_loci` is the cost of carrying more loci rather than a content change.
    """
    F = load() if F is None else F
    L = F[F.arm.str.startswith("size") | F.arm.isin(["gf19r", "gf19"])].copy()
    L["rung"] = L.n_loci
    return L.sort_values(["rung", "drug"])

# The baseline cells this notebook compares against, all on the same lineage
# folds. Keys are (locus_set, model) in the locus_sets frame.
BASELINE_DIRS = {
    ("bigtb19", "locusfusion_v5", "v5_fixed"):
        EXP / "locusfusion_v4/sd19_v5_fixed",
    ("bigtb19", "locusfusion", "dna_regulatory"):
        EXP / "lineage_split/sd19_dna_regulatory_unk__locusfusion_v2",
    ("bigtb19", "locusfusion", "all_modalities"):
        EXP / "lineage_split/sd19_all_modalities_unk__locusfusion_v2",
    ("bigtb19", "locusfusion", "dna"):
        EXP / "lineage_split/sd19_dna_unk__locusfusion_v2",
    ("bigtb19", "mdcnn", "dna_regulatory"):
        EXP / "lineage_split/sd19_dna_regulatory__mdcnn",
    ("bigtb19", "lasso", "dna"):
        EXP / "lineage_split/lasso_onehot_all",
}

# 19-locus, lineage-split runs the locus_sets frame leaves out, so a "best model"
# claim can be made against every cell. folder glob -> model id in the frame.
OTHER_19_LOCUS = {"locusfusion_v4/sd19_*": "locusfusion_v5",
                  "lineage_split/sd19_*__cisfusion": "cisfusion",
                  "lineage_split/sd19_*__late_fusion": "late_fusion"}


def load_other_19_locus():
    """LocusFusion v4/v5, cis-fusion and late-fusion on the 19 loci and lineage
    folds, one row per (model, modset, drug), in the same schema as `load`.

    `modset` is the folder name without its sd19_ prefix or __model suffix; v4/v5
    arms are not modality sets, so "v5_fixed" names the arm itself.
    """
    rows = []
    for pattern, model in OTHER_19_LOCUS.items():
        for d in sorted(EXP.glob(pattern)):
            modset = d.name.removeprefix("sd19_").split("__")[0]
            for p in sorted(d.glob("*.json")):
                j = json.loads(p.read_text())
                if "cv_auc_mean" not in j or "cv_folds" not in j:
                    continue
                rows.append(dict(
                    setting="single", locus_set="bigtb19", model=model, modset=modset,
                    drug=j["drug"], fold_aucs=_fold_aucs(j), cv=j["cv_auc_mean"],
                    test=(j.get("test") or {}).get("auc"), n_params=j.get("n_params"),
                    src=str(d.relative_to(EXP)), task="single:bigtb19",
                    n_folds=len(_fold_aucs(j))))
    F = pd.DataFrame(rows)
    dup = F.duplicated(["model", "modset", "drug"])
    if dup.any():
        raise ValueError(f"{dup.sum()} duplicate cells:\n{F[dup]}")
    return F


def _fold_aucs(j):
    """The per-fold CV AUCs at full precision. GenomeFusion always writes
    `cv_folds`, one entry per fold it actually ran."""
    return tuple(f["auc"] for f in (j.get("cv_folds") or [])
                 if f.get("auc") is not None)


def _fold_shas(j):
    return tuple(f.get("val_ids_sha1") for f in (j.get("cv_folds") or []))


def load(arms=None):
    """One row per (locus_set, drug) over the GenomeFusion run folders."""
    rows = []
    for folder, locus_set in ARMS.items():
        if arms is not None and folder not in arms:
            continue
        d = GF / folder
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            j = json.loads(p.read_text())
            if "cv_auc_mean" not in j:
                continue
            rows.append(dict(
                setting="single", locus_set=locus_set, model="genomefusion",
                modset="gf_token", drug=j["drug"], fold_aucs=_fold_aucs(j),
                cv=j["cv_auc_mean"], test=(j.get("test") or {}).get("auc"),
                n_params=j.get("n_params"), n_loci=j.get("n_loci"),
                src=str(d.relative_to(EXP)), arm=folder,
                n_folds=len(_fold_aucs(j)),
                active_loci_mean=j.get("active_loci_mean"),
                seconds=j.get("seconds"), host=j.get("host"),
                n_isolates=j.get("n_isolates"),
                split_file=(j.get("split_file") or "").split("/")[-1]))
    F = pd.DataFrame(rows, columns=[
        "setting", "locus_set", "model", "modset", "drug", "fold_aucs", "cv",
        "test", "n_params", "n_loci", "src", "task", "arm", "n_folds",
        "active_loci_mean", "seconds", "host", "n_isolates", "split_file"])
    F["task"] = F.setting + ":" + F.locus_set
    dup = F.duplicated(["locus_set", "model", "modset", "drug"])
    if dup.any():
        raise ValueError(f"{dup.sum()} duplicate cells:\n{F[dup]}")
    return F


def fold_provenance(drug="PYRAZINAMIDE"):
    """The val-ID hash of every fold, GenomeFusion beside each baseline.

    This is the load-bearing check for the whole comparison: the 55-fold
    Wilcoxon is only valid if fold *i* is the same held-out isolates in every
    cell. The hashes are of the validation IDs, so equal hashes mean the same
    isolates in the same fold.

    LASSO writes no per-fold hash (it keeps `c_path` instead of `cv_folds`), so
    its row reports the split directory it recorded instead -- weaker evidence,
    and the notebook says so.
    """
    rows = []
    for arm, locus_set in ARMS.items():
        p = GF / arm / f"{drug}.json"
        if not p.exists():
            continue
        j = json.loads(p.read_text())
        rows.append(dict(cell=f"GenomeFusion · {LABEL[locus_set]}",
                         n_isolates=j.get("n_isolates"),
                         shas=_fold_shas(j),
                         evidence="per-fold val-ID hash"))
    for (locus_set, model, modset), d in BASELINE_DIRS.items():
        hits = sorted(d.glob(f"{drug}__*.json")) or sorted(d.glob(f"{drug}.json"))
        if not hits:
            continue
        j = json.loads(hits[0].read_text())
        shas = _fold_shas(j)
        rows.append(dict(
            cell=f"{model} ({modset}) · 19 loci",
            n_isolates=j.get("n_isolates"),
            shas=shas,
            evidence="per-fold val-ID hash" if shas else
                     f"split_dir={(j.get('split') or {}).get('split_dir')}"))
    return pd.DataFrame(rows)


def warmup_curve(drug="PYRAZINAMIDE", arm="canary", fold=0):
    """(epoch, train_loss, val_auc) for one fold -- the convergence trace.

    GenomeFusion's whole-genome arm spends ~20 epochs near chance before it
    locates the drug's locus, so a run cut short looks like a failure. This is
    the evidence for the `min_epochs >= 50` floor in the protocol.
    """
    j = json.loads((GF / arm / f"{drug}.json").read_text())
    h = j["cv_folds"][fold]["history"]
    return pd.DataFrame({"epoch": range(1, len(h["val_auc"]) + 1),
                         "train_loss": h["train_loss"],
                         "val_auc": h["val_auc"]})
