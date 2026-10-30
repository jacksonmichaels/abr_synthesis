"""locusfusion_v3 against the paper's comparisons.

v3 = locusfusion_v2 + --lf-unknown-symbols (an uncalled base makes a token).
This rebuilds the paper's 14-row Holm family (SIGNIFICANCE_TESTS.md) with v3 in
every LocusFusion row, the v3 tokenizer ablation (v3 minus the uncalled-base
channel) standing in for v1's tokenizer-fix row, and then the unadjusted
extras: v3 against v1, modality effects, seeds and repartitions.

    python results/experiments/locusfusion_v3/compare.py            # v3
    python results/experiments/locusfusion_v3/compare.py --as-v1    # self-check:
        # puts v1 (locusfusion_fixed) in the v3 slots; rows 4-8 must reproduce
        # the paper (+0.0103, +0.0117, ..., +0.0293)

Writes SUMMARY.md and family.csv / extras.csv next to this file. Rows whose
cells are incomplete are listed as missing, not dropped silently.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT))
from results_frame import EXP, load_frame, strip_task_prefix  # noqa: E402
from results_helpers import WilcoxonComparisons, fold_matrix, hodges_lehmann  # noqa: E402

PD, L19, WG = "single-drug", "single-drug, all loci", "single-drug, whole genome"
A = "all_modalities"
MODS5 = ["dna", "dna_protein", "dna_biophysical", "dna_regulatory", A]
EST3 = ["mdcnn", "cisfusion", "late_fusion"]
V3, V3X, V1 = "locusfusion_v3", "locusfusion_v3_nounk", "locusfusion"


def _fold_aucs(j):
    return tuple(f["auc"] for f in j.get("cv_folds") or [] if f.get("auc") is not None)


def read_cells(run, prefixes, model):
    """{prefix: rows} for results/experiments/<run>/<prefix><modset>[_unk]__locusfusion_v2."""
    rows = []
    for d in sorted((EXP / run).glob("*__locusfusion_v2")):
        pre = next((p for p in prefixes if d.name.startswith(p)), None)
        if pre is None:
            continue
        modset = d.name[len(pre):].rsplit("__", 1)[0].removesuffix("_unk")
        for p in d.glob("*.json"):
            if p.name == "weights_location.json":
                continue
            j = json.loads(p.read_text())
            if "cv_auc_mean" not in j:
                continue
            n_loci = len(j.get("genes") or j.get("loci") or [])
            rows.append(dict(task=prefixes[pre], modset=modset, model=model,
                             fold_aucs=_fold_aucs(j), drug=j["drug"], cv=j["cv_auc_mean"],
                             cv_sd=j.get("cv_auc_std"), n_params=j.get("n_params"),
                             n_loci=n_loci, run=f"{run}/{d.name}", arch="locusfusion_v2"))
    return pd.DataFrame(rows)


def build_frame(as_v1=False):
    frame, _ = load_frame()
    # the paper's own filter: only the fixed-tokenizer locusfusion is "locusfusion"
    old = (frame.model == V1) & (frame.run == "newmodels_full") & (frame.task != "multi-drug")
    frame_old = frame[old].assign(model="locusfusion_old")
    frame = frame[~old]
    # the paper's LASSOs are the 09-01 alignment run and the 09-03 whole-genome
    # run; later LASSO sweeps (e.g. lasso_wholegenome_sweep_20260929) would add a
    # second cell to the same (task, model) and are out of scope
    lasso = frame.model.str.startswith("lasso")
    frame = frame[~lasso | frame.run.isin(["lasso_wholeseq_20260901",
                                           "lasso_wholegenome_20260903"])]
    if as_v1:                     # self-check: v1 in the v3 slots
        v3 = frame[frame.model == V1].assign(model=V3)
        v3x = frame_old.assign(model=V3X)
    else:
        v3 = read_cells("locusfusion_v3", {"sd19_": L19, "sd_": PD}, V3)
        v3x = read_cells("locusfusion_v3", {"ablation_sd19_": L19}, V3X)
    parts = [frame, frame_old] + [x for x in (v3, v3x) if len(x)]
    return pd.concat(parts, ignore_index=True)


def family(frame, drugs):
    """The paper's 14 rows (SIGNIFICANCE_TESTS.md), LocusFusion -> v3."""
    W = WilcoxonComparisons(frame, drugs)
    missing = []

    def row(group, label, a, b):
        try:
            W.folds(group, label, a, b)
        except ValueError as e:
            missing.append(f"{group}: {label} -- {e}")

    row("LASSO vs. CNN", "lasso_onehot − mdcnn(all_modalities), 19 loci",
        (L19, "lasso_onehot", "dna"), (L19, "mdcnn", A))
    row("LASSO vs. CNN", "lasso_onehot − mdcnn(all_modalities), per-drug loci",
        (PD, "lasso_onehot", "dna"), (PD, "mdcnn", A))
    row("LASSO vs. CNN", "lasso onehot − delta, 19 loci",
        (L19, "lasso_onehot", "dna"), (L19, "lasso_delta", "dna"))
    row("LocusFusion v3", "v3(all_modalities) − lasso_onehot, 19 loci",
        (L19, V3, A), (L19, "lasso_onehot", "dna"))
    for m in MODS5:               # 5-modset rows are tested over the modset-mean
        pass
    row("LocusFusion v3", "v3 − mdcnn, 19 loci, 5 modality sets",
        (L19, V3, MODS5), (L19, "mdcnn", MODS5))
    row("LocusFusion v3", "v3 − v3 without the uncalled-base channel, 19 loci, 5 modality sets",
        (L19, V3, MODS5), (L19, V3X, MODS5))
    row("Locus set", "established: 19 − per-drug loci, 5 modality sets",
        (L19, EST3, MODS5), (PD, EST3, MODS5))
    row("Locus set", "v3: 19 − per-drug loci, 5 modality sets",
        (L19, V3, MODS5), (PD, V3, MODS5))
    row("Locus set", "lasso_onehot: 19 − per-drug loci",
        (L19, "lasso_onehot", "dna"), (PD, "lasso_onehot", "dna"))
    row("Locus set", "lasso_delta: whole genome − 19 loci",
        (WG, "lasso_genome", "dna"), (L19, "lasso_delta", "dna"))
    row("Modalities", "established: all_modalities − dna, per-drug loci",
        (PD, EST3, A), (PD, EST3, "dna"))
    row("Model size", "mdcnn − cisfusion, 19 loci, 5 modality sets",
        (L19, "mdcnn", MODS5), (L19, "cisfusion", MODS5))
    row("Model size", "mdcnn − late_fusion, 19 loci, 5 modality sets",
        (L19, "mdcnn", MODS5), (L19, "late_fusion", MODS5))
    row("Model size", "late_fusion − cisfusion, 19 loci, 5 modality sets",
        (L19, "late_fusion", MODS5), (L19, "cisfusion", MODS5))
    t = W.table() if W.rows else pd.DataFrame()
    return t, missing


def extras(frame, drugs):
    """Unadjusted rows: no Holm, no verdict."""
    W = WilcoxonComparisons(frame, drugs)
    missing = []

    def row(group, label, a, b):
        try:
            W.folds(group, label, a, b)
        except ValueError as e:
            missing.append(f"{group}: {label} -- {e}")

    for task, tname in ((L19, "19 loci"), (PD, "per-drug loci")):
        row("v3 vs v1", f"v3 − v1, {tname}, 5 modality sets", (task, V3, MODS5), (task, V1, MODS5))
        for m in MODS5:
            row("v3 vs v1", f"v3 − v1, {tname}, {m}", (task, V3, m), (task, V1, m))
        row("v3 vs mdcnn", f"v3 − mdcnn, {tname}, all_modalities", (task, V3, A), (task, "mdcnn", A))
        row("v3 vs mdcnn", f"v3 − mdcnn, {tname}, dna", (task, V3, "dna"), (task, "mdcnn", "dna"))
        for m in MODS5[1:]:
            row("v3 modalities", f"v3: {m} − dna, {tname}", (task, V3, m), (task, V3, "dna"))
    for m in MODS5:
        row("v3 tokenizer ablation", f"v3 − no-uncalled-base, 19 loci, {m}",
            (L19, V3, m), (L19, V3X, m))
    row("v3 vs LASSO", "v3(all_modalities) − lasso_onehot, per-drug loci",
        (PD, V3, A), (PD, "lasso_onehot", "dna"))
    t = W.table().drop(columns=["p_holm", "verdict"]) if W.rows else pd.DataFrame()
    return t, missing


def robustness(drugs):
    """Seed spread and repartition comparison for v3 at 19 loci, all_modalities."""
    out = []
    seeds = {}
    for s in (0, 1000, 2000, 3000, 4000):
        cell = ("sd19_" if s == 0 else f"s{s}_") + "all_modalities_unk__locusfusion_v2"
        r = {}
        for p in (EXP / "locusfusion_v3" / cell).glob("*.json"):
            j = json.loads(p.read_text())
            if "cv_auc_mean" in j:
                r[j["drug"]] = j["cv_auc_mean"]
        seeds[s] = r
    full = [s for s, r in seeds.items() if set(drugs) <= set(r)]
    if len(full) >= 2:
        M = pd.DataFrame({s: seeds[s] for s in full}).loc[drugs]
        out.append(f"- Seeds complete: {full}. Macro CV per seed: "
                   + ", ".join(f"{s}: {M[s].mean():.4f}" for s in full)
                   + f". SD of the macro across seeds {M.mean().std(ddof=1):.4f}; "
                   f"mean per-drug SD across seeds {M.std(axis=1, ddof=1).mean():.4f}.")
    else:
        out.append(f"- Seeds: only {full} complete for all 11 drugs -- not summarised.")

    for c in (101, 102, 103, 104):
        v3, md = {}, {}
        for store, folder in ((v3, EXP / "locusfusion_v3" / f"cv{c}_all_modalities_unk__locusfusion_v2"),
                              (md, EXP / "repartition_19loci" / f"cv{c}_all_modalities__mdcnn")):
            for p in folder.glob("*.json"):
                j = json.loads(p.read_text())
                if "cv_auc_mean" in j:
                    store[j["drug"]] = _fold_aucs(j)
        if not (set(drugs) <= set(v3) and set(drugs) <= set(md)):
            out.append(f"- Partition {c}: incomplete (v3 {len(v3)}/11, mdcnn {len(md)}/11).")
            continue
        diff = np.array([np.array(v3[d]) - np.array(md[d]) for d in drugs])
        x = diff.ravel()
        hl, lo, hi = hodges_lehmann(x)
        per = diff.mean(axis=1)
        p = stats.wilcoxon(x, method="exact").pvalue
        out.append(f"- Partition {c}: v3 − mdcnn {per.mean():+.4f} [HL {hl:+.4f}, "
                   f"{lo:+.4f}, {hi:+.4f}], {int((per > 0).sum())}/11, p={p:.2g} (unadjusted).")
    return out


def coverage():
    """How many drugs each v3 cell has, so a missing job is visible."""
    lines = []
    root = EXP / "locusfusion_v3"
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        n = len([p for p in d.glob("*.json")
                 if p.name != "weights_location.json"])
        lines.append(f"| {d.name} | {n}/11 |")
    return lines


def fmt(t, adjusted):
    if t.empty:
        return "(no complete rows)"
    cols = ["group", "comparison", "delta", "hl", "lo", "hi", "wins", "p"]
    cols += ["p_holm", "verdict"] if adjusted else []
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in t[cols].iterrows():
        v = [r.group, r.comparison, f"{r.delta:+.4f}", f"{r.hl:+.4f}", f"{r.lo:+.4f}",
             f"{r.hi:+.4f}", r.wins, f"{r.p:.2g}"]
        if adjusted:
            v += [f"{r.p_holm:.2g}", r.verdict]
        lines.append("| " + " | ".join(map(str, v)) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-v1", action="store_true", help="self-check against the paper")
    args = ap.parse_args()
    from bigtb_baselines import ALL_DRUGS
    drugs = list(ALL_DRUGS)
    frame = build_frame(args.as_v1)
    fam, miss_f = family(frame, drugs)
    ext, miss_e = extras(frame, drugs)
    if args.as_v1:
        print(fmt(fam, True))
        print("\nmissing:", *miss_f, sep="\n  ")
        return
    fam.to_csv(HERE / "family.csv", index=False)
    ext.to_csv(HERE / "extras.csv", index=False)
    md = ["# locusfusion_v3 results", "",
          "v3 = `locusfusion_v2 --lf-unknown-symbols`. Generated by `compare.py`; "
          "do not edit by hand.", "",
          "## Paper family (Holm over these rows)", "", fmt(fam, True), ""]
    if miss_f:
        md += ["Missing rows (incomplete cells -- the Holm family is SMALLER than 14 "
               "until these land, so its p-values are not final):", ""]
        md += [f"- {m}" for m in miss_f] + [""]
    md += ["## Unadjusted extras", "", fmt(ext, False), ""]
    if miss_e:
        md += ["Missing:", ""] + [f"- {m}" for m in miss_e] + [""]
    md += ["## Robustness (19 loci, all_modalities)", ""] + robustness(drugs) + [""]
    md += ["## Coverage", "", "| cell | drugs |", "|---|---|"] + coverage()
    (HERE / "SUMMARY.md").write_text("\n".join(md) + "\n")
    print(f"wrote {HERE / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
