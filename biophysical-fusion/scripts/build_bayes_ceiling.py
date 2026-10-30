"""Builds notebooks/analysis/bayes_ceiling.ipynb -- the DNA AUC ceiling (with and without
promoter windows) against our lineage-split results: LocusFusion v3, the v4/v5
ablation arms, the stacked CNN and the LASSO, single-drug and multi-drug,
per-drug loci and 19 loci.

    python scripts/build_bayes_ceiling.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/analysis/bayes_ceiling.ipynb

Reads results/analysis/bayes_ceiling/ (written by scripts/bayes_ceiling.py,
job scripts/sbatch/bayes_ceiling.sh) and the result JSONs / test-prediction
CSVs already on disk. Trains and scores nothing.
"""
import json
from pathlib import Path

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": t.strip("\n").splitlines(keepends=True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})


md(r"""
# How close are we to the DNA ceiling?

Every model here sees a deterministic function of the isolate's aligned DNA.
Protein and biophysical channels are computed from it. So isolates with
identical sequences must get the same score, and no model can beat the ranking
that orders genotype groups by their resistant fraction. That ranking's AUC on
an evaluation set is the **ceiling**: the maximum AUC on that set for any model
of that input. It falls below 1 only where isolates share a genotype but
disagree on phenotype.

The ceiling is computed for every combination of:
- **task:** the single-drug lineage splits, and the one multi-drug split;
- **locus set:** per-drug loci, and all 19;
- **input:** `dna` (gene-locus FASTAs only) and `dna+promoter` (plus the promoter
  windows the regulatory modality loads, which are DNA outside the gene records);
- **eval set:** TEST and CV folds 0–4, each scored exactly as the models were.

Built by `scripts/bayes_ceiling.py` from the same files the models trained on.
""")

code(r"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from IPython.display import display

PROJECT = next(p for p in [Path.cwd(), *Path.cwd().parents,
                           Path("/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion")]
               if (p / "paper_style.py").exists())
sys.path.insert(0, str(PROJECT))
from paper_style import MODEL, NEUTRAL, ramp, apply
apply()
pd.set_option("display.width", 240, "display.max_columns", 60, "display.max_rows", 200,
              "display.float_format", "{:.4f}".format)

CEIL = PROJECT / "results" / "analysis" / "bayes_ceiling"
LS = PROJECT / "results" / "experiments" / "lineage_split"
V4 = PROJECT / "results" / "experiments" / "locusfusion_v4"
DRUGS = ["ISONIAZID", "RIFAMPICIN", "ETHAMBUTOL", "PYRAZINAMIDE", "STREPTOMYCIN",
         "KANAMYCIN", "AMIKACIN", "CAPREOMYCIN", "LEVOFLOXACIN", "MOXIFLOXACIN", "ETHIONAMIDE"]
AB = {"ISONIAZID": "INH", "RIFAMPICIN": "RIF", "ETHAMBUTOL": "EMB", "PYRAZINAMIDE": "PZA",
      "STREPTOMYCIN": "STR", "KANAMYCIN": "KAN", "AMIKACIN": "AMI", "CAPREOMYCIN": "CAP",
      "LEVOFLOXACIN": "LFX", "MOXIFLOXACIN": "MXF", "ETHIONAMIDE": "ETO"}
CV = [f"cv{k}" for k in range(5)]
KEYS = ["task", "locus_set", "input", "drug"]

ceil = pd.read_csv(CEIL / "ceiling.csv")
iso = pd.read_csv(CEIL / "isolates.csv.gz")
meta = json.loads((CEIL / "meta.json").read_text())
print(meta["built"], "|", len(ceil), "ceiling rows | inputs", meta["inputs"])
""")

md(r"""
## The ceiling

The macro mean over the 11 drugs comes first, then per drug. `singleton` is the
share of TEST isolates whose genotype occurs only once in TEST. The ceiling
ranks those perfectly by construction, so where `singleton` is high the ceiling
sits near 1 and is not evidence of headroom.
""")

code(r"""
t = ceil[ceil.eval_set == "test"].set_index(KEYS)
v = ceil[ceil.eval_set.isin(CV)].groupby(KEYS)
summ = pd.DataFrame({"ceiling_test": t["oracle_auc"], "ceiling_cv": v["oracle_auc"].mean(),
                     "noise_test": t["noise_auc"], "noise_cv": v["noise_auc"].mean(),
                     "singleton_test": t["frac_singleton"], "genotypes_test": t["n_genotypes"],
                     "n_test": t["n"]}).reset_index()
summ["drug"] = pd.Categorical(summ["drug"], DRUGS)
summ = summ.sort_values(KEYS)
display(summ.groupby(["task", "locus_set", "input"])[
    ["ceiling_test", "ceiling_cv", "noise_test", "noise_cv", "singleton_test"]].mean())
""")

code(r"""
for task in ["single", "multi"]:
    print(f"{task}-drug: ceiling AUC (CV mean | TEST)")
    w = summ[summ.task == task].pivot_table(index="drug", columns=["locus_set", "input"],
                                            values=["ceiling_cv", "ceiling_test"], observed=True)
    display(w.round(4))
print("TEST singleton share")
display(summ.pivot_table(index="drug", columns=["task", "locus_set", "input"],
                         values="singleton_test", observed=True).round(3))
""")

md(r"""
Adding promoters can only raise the ceiling, and so can going from per-drug
loci to 19, because each finer genotype key splits groups further. At per-drug
loci the promoter windows matter most where a known promoter mechanism exists,
for example INH and ETO (fabG1–inhA c-15t) and KAN/AMI (eis).
""")

md(r"""
## The results compared

Every model is scored against the ceiling of its **own** input. The
all-modalities cells, v4/v5 and the multi-drug arms load promoter windows, so
they are compared with `dna+promoter`. The DNA-only cells and the LASSO (a
one-hot of the gene DNA) are compared with `dna`. AUCs come from the result
JSONs, TEST and per-fold CV. Fold `k` in a JSON is CV fold `k` of the split, and
`n_match` checks that both were scored on the same isolates.

| key | cell | input |
|---|---|---|
| LF v3 | `lineage_split/{sd,sd19}_all_modalities_unk__locusfusion_v2` | dna+promoter |
| LF v3 DNA | `lineage_split/{sd,sd19}_dna_unk__locusfusion_v2` | dna |
| stacked CNN | `lineage_split/{sd,sd19}_all_modalities__mdcnn` | dna+promoter |
| stacked CNN DNA | `lineage_split/{sd,sd19}_dna__mdcnn` | dna |
| LASSO | `lineage_split/lasso_onehot_{perdrug,all}` (CV mean only) | dna |
| v4 noaug / aug / nodist / nosyn, v5, v5 nodist | `locusfusion_v4/sd19_*` | dna+promoter |
| md noaug / aug / nodist / nosyn | `locusfusion_v4/md_*` (multi-drug) | dna+promoter |

The v5 arms on repaired DNA (`v5_fixdna`, `v5_atg`, `v5_fixed`) are left out,
because their input files differ from the ones the ceiling is computed on.
""")

code(r"""
rows = []

def add(model, task, ls, inp, drug, eval_set, auc, n=np.nan):
    rows.append(dict(model=model, task=task, locus_set=ls, input=inp, drug=drug,
                     eval_set=eval_set, auc=auc, n=n))

def from_sd_json(model, ls, inp, path, drug):
    if not path.exists():
        print("missing", path.relative_to(PROJECT)); return
    d = json.loads(path.read_text())
    folds = [f for f in d.get("cv_folds", []) if isinstance(f, dict) and "auc" in f]
    for f in folds:
        add(model, "single", ls, inp, drug, f"cv{f['fold']}", f["auc"], f.get("n", np.nan))
    if not folds:
        add(model, "single", ls, inp, drug, "cvmean", d["cv_auc_mean"])
    add(model, "single", ls, inp, drug, "test", d["test"]["auc"], d["test"].get("n", np.nan))

ALL = "dna+protein+biophysical+regulatory"
SD_ARMS = {"v4 noaug": "sd19_noaug", "v4 aug": "sd19_aug", "v4 nodist": "sd19_nodist",
           "v4 nosyn": "sd19_nosyn", "v5": "sd19_v5", "v5 nodist": "sd19_v5_nodist"}
MD_ARMS = {"md noaug": "md_noaug", "md aug": "md_aug", "md nodist": "md_nodist", "md nosyn": "md_nosyn"}
for drug in DRUGS:
    for ls, pre in [("perdrug", "sd"), ("19", "sd19")]:
        from_sd_json("LF v3", ls, "dna+promoter", LS / f"{pre}_all_modalities_unk__locusfusion_v2" / f"{drug}__{ALL}.json", drug)
        from_sd_json("LF v3 DNA", ls, "dna", LS / f"{pre}_dna_unk__locusfusion_v2" / f"{drug}__dna.json", drug)
        from_sd_json("stacked CNN", ls, "dna+promoter", LS / f"{pre}_all_modalities__mdcnn" / f"{drug}__{ALL}.json", drug)
        from_sd_json("stacked CNN DNA", ls, "dna", LS / f"{pre}_dna__mdcnn" / f"{drug}__dna.json", drug)
    from_sd_json("LASSO", "perdrug", "dna", LS / "lasso_onehot_perdrug" / f"{drug}__dna-onehot-perdrug.json", drug)
    from_sd_json("LASSO", "19", "dna", LS / "lasso_onehot_all" / f"{drug}__dna-onehot-all.json", drug)
    for name, arm in SD_ARMS.items():
        from_sd_json(name, "19", "dna+promoter", V4 / arm / f"{drug}.json", drug)

for name, arm in MD_ARMS.items():
    d = json.loads((V4 / arm / "MULTIDRUG.json").read_text())
    for f in d["cv_folds"]:
        for drug, r in f["per_drug"].items():
            add(name, "multi", "19", "dna+promoter", drug, f"cv{f['fold']}", r["auc"], r["n"])
    for drug, r in d["per_drug"].items():
        add(name, "multi", "19", "dna+promoter", drug, "test", r["test"]["auc"], r["test"]["n"])

res = pd.DataFrame(rows)
cvm = (res[res.eval_set.isin(CV)].groupby(["model"] + KEYS)["auc"].agg(["mean", "size"]).reset_index())
assert (cvm["size"] == 5).all(), cvm[cvm["size"] != 5]
res = pd.concat([res, cvm.assign(eval_set="cvmean", auc=cvm["mean"]).drop(columns=["mean", "size"])],
                ignore_index=True)

c = ceil[ceil.eval_set != "all"][KEYS + ["eval_set", "n", "oracle_auc", "noise_auc"]]
c = pd.concat([c, c[c.eval_set.isin(CV)].groupby(KEYS)[["oracle_auc", "noise_auc"]]
               .mean().reset_index().assign(eval_set="cvmean", n=np.nan)])
cmp = res.merge(c, on=KEYS + ["eval_set"], how="left", suffixes=("_model", "_ceil"))
cmp["gap"] = cmp["oracle_auc"] - cmp["auc"]
cmp["n_match"] = cmp["n_model"].isna() | (cmp["n_model"] == cmp["n_ceil"])
print("models:", sorted(cmp.model.unique()))
print("eval sets with mismatched n:", int((~cmp.n_match).sum()), "| missing ceiling:", int(cmp.oracle_auc.isna().sum()))
over = cmp[cmp["auc"] > cmp["oracle_auc"] + 1e-9]
print(f"model AUC above its own-input ceiling on {len(over)} eval sets (must be 0):")
display(over)
""")

md(r"""
**Tie-breaking.** The ceiling scores a pair of identical-genotype isolates as a
tie (0.5). A network's GPU floats are not exactly reproducible across batches,
so it can give identical genotypes scores that differ by ~1e-8. sklearn's AUC
treats that as an ordering, which can land a few thousandths above the ceiling
by chance on a small fold. The v5 TEST predictions show within-group score
ranges up to 1.8e-8. To size this, each flagged eval set is re-scored with the
ceiling's ties broken at random (1,000 draws), and the 99th percentile is
reported. Before the promoter key was added, the per-drug INH and ETO
all-modalities cells sat 0.01–0.06 above the gene-only ceiling. That excess was
input, not tie-breaking.
""")

code(r"""
rng = np.random.default_rng(0)
def tiebreak_q(task, ls, inp, drug, ev, q=0.99, draws=1000):
    g = iso[(iso.task == task) & (iso.locus_set == ls) & (iso.input == inp) & (iso.drug == drug) & (iso.eval_set == ev)]
    y, s = g["y"].to_numpy(), g["frac_R"].to_numpy()
    aucs = [roc_auc_score(y, s + rng.uniform(0, 1e-9, len(s))) for _ in range(draws)]
    return float(np.quantile(aucs, q))

if len(over):
    over = over.copy()
    over["tiebreak_q99"] = [tiebreak_q(r.task, r.locus_set, r.input, r.drug, r.eval_set) for r in over.itertuples()]
    over["within_band"] = over["auc"] <= over["tiebreak_q99"]
    display(over[["model", "locus_set", "input", "drug", "eval_set", "n_model", "auc", "oracle_auc", "tiebreak_q99", "within_band"]])
""")

md(r"""
## Gap to the ceiling, macro over 11 drugs

Gap = own-input ceiling − model, on the same eval set. Rows are grouped by task
and locus set.
""")

code(r"""
ORDER = ["LASSO", "stacked CNN DNA", "stacked CNN", "LF v3 DNA", "LF v3",
         "v4 noaug", "v4 aug", "v4 nodist", "v4 nosyn", "v5", "v5 nodist",
         "md noaug", "md aug", "md nodist", "md nosyn"]
macro = (cmp[cmp.eval_set.isin(["cvmean", "test"])]
         .groupby(["task", "locus_set", "input", "model", "eval_set"])
         .agg(model_auc=("auc", "mean"), ceiling=("oracle_auc", "mean"), gap=("gap", "mean"))
         .reset_index())
mt = macro.pivot_table(index=["task", "locus_set", "model", "input"], columns="eval_set",
                       values=["model_auc", "ceiling", "gap"])
mt = mt.reindex(sorted(mt.index, key=lambda k: (k[0] != "single", k[1] != "perdrug", ORDER.index(k[2]))))
display(mt.round(4))
""")

md(r"""
## The v4 ablations against v3 and the ceiling

These are the 19-locus single-drug arms. Each is paired with LF v3 on the same
55 CV folds (11 drugs × 5) using an exact Wilcoxon signed-rank test, with the
Hodges–Lehmann shift as the effect size. `share of gap` is the fraction of
v3's CV gap to the ceiling that the arm closes.
""")

def_hl = r"""
def hl(x):
    x = np.asarray(x); w = (x[:, None] + x[None, :]) / 2
    return float(np.median(w[np.triu_indices(len(x))]))

fold = cmp[(cmp.task == "single") & (cmp.locus_set == "19") & cmp.eval_set.isin(CV)]
base = fold[fold.model == "LF v3"].set_index(["drug", "eval_set"])
abl = []
for m in ["LF v3", "LASSO", "stacked CNN", "v4 noaug", "v4 aug", "v4 nodist", "v4 nosyn", "v5", "v5 nodist"]:
    s = fold[fold.model == m].set_index(["drug", "eval_set"])
    tm = cmp[(cmp.task == "single") & (cmp.locus_set == "19") & (cmp.model == m) & (cmp.eval_set == "test")]
    cvm_ = cmp[(cmp.task == "single") & (cmp.locus_set == "19") & (cmp.model == m) & (cmp.eval_set == "cvmean")]
    row = dict(model=m, cv_auc=cvm_["auc"].mean(), test_auc=tm["auc"].mean(),
               cv_ceiling=cvm_["oracle_auc"].mean(), cv_noise_ceiling=cvm_["noise_auc"].mean(),
               cv_gap=cvm_["gap"].mean(), test_gap=tm["gap"].mean())
    if len(s) and m != "LF v3":
        dlt = (s["auc"] - base["auc"]).dropna()
        row.update(n_folds=len(dlt), hl_vs_v3=hl(dlt), wins=int((dlt > 0).sum()),
                   p_vs_v3=wilcoxon(dlt, method="exact" if len(dlt) <= 60 else "auto").pvalue)
    abl.append(row)
abl = pd.DataFrame(abl)
v3gap = abl.loc[abl.model == "LF v3", "cv_gap"].iloc[0]
abl["share_of_gap"] = (v3gap - abl["cv_gap"]) / v3gap
display(abl.set_index("model"))
"""
code(def_hl)

md(r"""
The LASSO has no per-fold AUCs in its JSON, so its Wilcoxon columns are empty
and it is shown on its CV mean. The ceilings here (`dna+promoter`, 19 loci)
sit near 1 because about half of the TEST isolates have a unique genotype.
The shared-genotype comparison below is the informative one for these arms.
""")

code(r"""
SERIES = {"LASSO": MODEL["lasso"], "stacked CNN": MODEL["cnn"], "stacked CNN DNA": ramp(MODEL["cnn"], 3)[1],
          "LF v3": MODEL["locusfusion"], "LF v3 DNA": ramp(MODEL["locusfusion"], 4)[2],
          "v4 nosyn": ramp(MODEL["locusfusion"], 4)[1], "v5": ramp(MODEL["locusfusion"], 4)[3]}
MARK = {"LASSO": "^", "stacked CNN": "s", "stacked CNN DNA": "s", "LF v3": "o", "LF v3 DNA": "o",
        "v4 nosyn": "v", "v5": "D"}
MD_COL = dict(zip(MD_ARMS, ramp(MODEL["locusfusion"], 5)[:4]))

def dot_panel(ax, task, ls, eval_set, models, colors):
    y = {d: i for i, d in enumerate(DRUGS[::-1])}
    for inp, col, lab in [("dna", NEUTRAL["muted"], "ceiling, gene DNA"),
                          ("dna+promoter", NEUTRAL["ink"], "ceiling, gene DNA + promoters")]:
        cl = c[(c.task == task) & (c.locus_set == ls) & (c.input == inp) & (c.eval_set == eval_set)].set_index("drug")["oracle_auc"]
        ax.scatter([cl.get(d, np.nan) for d in y], list(y.values()), marker="|", s=280, color=col, lw=2.2,
                   label=lab, zorder=4)
    sub = cmp[(cmp.task == task) & (cmp.locus_set == ls) & (cmp.eval_set == eval_set)]
    for m in models:
        s = sub[sub.model == m].set_index("drug")["auc"]
        if len(s):
            ax.scatter([s.get(d, np.nan) for d in y], list(y.values()), s=28, color=colors[m],
                       marker=MARK.get(m, "o"), label=m, zorder=5, alpha=0.9)
    ax.set_yticks(list(y.values()), [AB[d] for d in y])
    ax.set_xlabel(f"AUC ({'TEST' if eval_set == 'test' else 'CV mean'})")
    ax.grid(axis="x", lw=0.6); ax.spines[["top", "right"]].set_visible(False)

SD_MODELS = ["LASSO", "stacked CNN DNA", "stacked CNN", "LF v3 DNA", "LF v3", "v4 nosyn", "v5"]
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharey=True)
for j, ev in enumerate(["cvmean", "test"]):
    lab = "CV mean" if ev == "cvmean" else "TEST"
    dot_panel(axes[0, j], "single", "perdrug", ev, SD_MODELS, SERIES)
    axes[0, j].set_title(f"Single-drug, per-drug loci, {lab}")
    dot_panel(axes[1, j], "single", "19", ev, SD_MODELS, SERIES)
    axes[1, j].set_title(f"Single-drug, 19 loci, {lab}")
h, l = axes[1, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 1))
plt.show()
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
for j, ev in enumerate(["cvmean", "test"]):
    dot_panel(axes[j], "multi", "19", ev, list(MD_ARMS), MD_COL)
    axes[j].set_title(f"Multi-drug, 19 loci, {'CV mean' if ev == 'cvmean' else 'TEST'}")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, fontsize=8)
fig.tight_layout(rect=(0, 0.1, 1, 1))
plt.show()
""")

md(r"""
The multi-drug ceiling is computed on the multi-drug split, whose TEST/CV
isolates differ from the single-drug splits, so its numbers are not
interchangeable with the single-drug ones. Its per-drug-loci rows give the
ceiling for a drug head restricted to its own loci.
""")

md(r"""
## Where the ceiling is measurable: isolates that share a genotype

TEST is restricted to isolates whose genotype occurs at least twice in TEST, so
on that subset the ceiling is a real bound, not one met by construction. The
model is rescored on the same isolates from its saved TEST predictions, which
exist for the v4/v5 19-locus arms and the multi-drug arms. Genotype groups use
the model's own input (19 loci + promoters). Rescored full-TEST AUCs are first
checked against the JSONs.
""")

code(r"""
def test_preds(path, drug):
    p = pd.read_csv(path)
    p = p[p[f"y_{drug}"] >= 0]
    return pd.DataFrame({"New_ID": p["New_ID"], "p_R": 1 - p[f"P_S_{drug}"], "y_model": 1 - p[f"y_{drug}"]})

PRED = [(n, "single", V4 / a, lambda d: f"{d}_test_predictions.csv") for n, a in SD_ARMS.items()]
PRED += [(n, "multi", V4 / a, lambda d: "MULTIDRUG_test_predictions.csv") for n, a in MD_ARMS.items()]
drows = []
for model, task, folder, fname in PRED:
    for drug in DRUGS:
        f = folder / fname(drug)
        if not f.exists():
            continue
        p = test_preds(f, drug)
        g = iso[(iso.task == task) & (iso.locus_set == "19") & (iso.input == "dna+promoter")
                & (iso.drug == drug) & (iso.eval_set == "test")]
        m = g.merge(p, on="New_ID", how="inner")
        assert len(m) == len(g) == len(p), (model, drug, len(m), len(g), len(p))
        assert (m["y"] == m["y_model"]).all(), (model, drug)
        dup = m[m["size"] >= 2]
        ok = dup["y"].nunique() == 2
        ref = cmp[(cmp.model == model) & (cmp.drug == drug) & (cmp.eval_set == "test")]["auc"]
        drows.append(dict(model=model, task=task, drug=drug, test_auc=roc_auc_score(m["y"], m["p_R"]),
                          json_auc=ref.iloc[0], dup_n=len(dup), dup_share=len(dup) / len(m),
                          dup_R=int(dup["y"].sum()),
                          model_dup_auc=roc_auc_score(dup["y"], dup["p_R"]) if ok else np.nan,
                          ceiling_dup_auc=roc_auc_score(dup["y"], dup["frac_R"]) if ok else np.nan))
dup = pd.DataFrame(drows)
dup["dup_gap"] = dup["ceiling_dup_auc"] - dup["model_dup_auc"]
print("max |rescored - JSON| TEST AUC:", float((dup.test_auc - dup.json_auc).abs().max()))
print("macro over drugs with both classes in the subset:")
display(dup.groupby(["task", "model"])[["dup_share", "model_dup_auc", "ceiling_dup_auc", "dup_gap"]].mean()
        .reindex([m for m in ORDER if m in set(dup.model)], level=1))
display(dup.pivot_table(index="drug", columns="model", values="dup_gap").reindex(DRUGS).round(4))
""")

code(r"""
fig, ax = plt.subplots(figsize=(6.2, 5.8))
cols = {**SERIES, **dict(zip(SD_ARMS, ramp(MODEL["locusfusion"], 7)[:6])), **MD_COL}
for m in [m for m in ["v4 nosyn", "v5", *MD_ARMS] if m in set(dup.model)]:
    s = dup[dup.model == m]
    ax.scatter(s["ceiling_dup_auc"], s["model_dup_auc"], color=cols[m], s=32, label=m, alpha=0.9,
               marker="o" if m.startswith("md") else MARK.get(m, "o"))
lo = np.nanmin(dup[["ceiling_dup_auc", "model_dup_auc"]].to_numpy()) - 0.02
ax.plot([lo, 1], [lo, 1], color=NEUTRAL["muted"], lw=1, label="model = ceiling")
ax.set_xlim(lo, 1.005); ax.set_ylim(lo, 1.005)
ax.set_xlabel("ceiling AUC, shared-genotype TEST isolates")
ax.set_ylabel("model AUC, same isolates")
ax.set_title("Shared-genotype TEST subset, 19 loci + promoters, per drug")
ax.grid(lw=0.6); ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=8, loc="lower right")
plt.show()
""")

md(r"""
## The noise-model ceiling

This extrapolates the ceiling to every isolate, singletons included. It assumes
that a genotype fixes the true phenotype, so any disagreement inside a duplicate
group is label noise. Class-conditional flip rates come from each drug's
duplicate groups over the whole cohort, with the majority label taken as truth.
They are applied to each eval set's R/S counts. Below are the flip rates, then
the CV gap to this ceiling per model.
""")

code(r"""
fl = ceil[(ceil.eval_set == "all") & (ceil.task == "single")]
display(fl.pivot_table(index="drug", columns=["locus_set", "input"],
                       values=["flip_R_to_S", "flip_S_to_R"]).reindex(DRUGS).round(3))
nm = (cmp[cmp.eval_set == "cvmean"].groupby(["task", "locus_set", "input", "model"])
      .agg(model_cv=("auc", "mean"), noise_ceiling_cv=("noise_auc", "mean"), strict_ceiling_cv=("oracle_auc", "mean"))
      .reset_index())
nm["gap_to_noise"] = nm["noise_ceiling_cv"] - nm["model_cv"]
nm = nm.set_index(["task", "locus_set", "input", "model"])
nm = nm.reindex(sorted(nm.index, key=lambda k: (k[0] != "single", k[1] != "perdrug", ORDER.index(k[3]))))
display(nm)
""")

md(r"""
## Caveats

- **The strict ceiling is in-sample.** It is the best AUC any function of the
  input could reach on each eval set, with each group's resistant fraction read
  off that same set. It bounds every model, but no trained model can reach it,
  since a model must also generalise to genotypes it has not seen.
- **Unique genotypes push the 19-locus ceiling to ~1 by construction.** For
  19-locus headroom, read the shared-genotype subset or the noise model, not
  the strict ceiling.
- **The noise model is an estimate, not a bound.** Its flip rates come from
  duplicate groups, which over-represent common genotypes, and it treats every
  disagreement as label error. A model can land above it, as some per-drug
  cells do.
- **Inputs.** Genotypes are read as raw strings from `data_snapshot` (the four
  loci with the insertion bug included, exactly as the models saw them) and
  from the TSS-anchored promoter windows. Ns are part of the genotype.
- **LASSO** stores CV as a mean only, so it has no fold-paired test.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                  "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent.parent / "notebooks" / "analysis" / "bayes_ceiling.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}")
