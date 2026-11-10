"""Write notebooks/analysis/locus_sets.ipynb (unexecuted). Every number comes from
collect.py at run time, so the notebook fills in as cells finish; rerun this only
when the notebook's structure changes.

    python results/experiments/locus_sets/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

PROJECT = Path(__file__).resolve().parents[3]
OUT = PROJECT / "notebooks" / "analysis" / "locus_sets.ipynb"

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# Locus sets: WHO tier 1+2 and the 64-locus set

LocusFusion v3, the stacked CNN and LASSO retrained on two larger locus sets, under the
lineage-grouped split (Beijing = CV fold 4), single-drug and multi-drug. Same protocol as
`lineage_split` (300 epochs / patience 30 / min 50 / seed 0, five modality sets).

| locus set | loci |
|---|---|
| Big-TB per-drug | Big-TB's `DRUG_TO_LOCI` (control: `lineage_split/sd_*`) |
| Big-TB 19 loci | the 19 curated loci for every drug (control: `lineage_split/sd19_*`; multi-drug trained here) |
| WHO tier 1+2 | each drug's WHO 2023 Table-21 tier 1+2 genes (6–14 per drug, 54 in the union) |
| 64 loci | every locus in the lab's `genomic_data/aligned` folder, for every drug |

Sweep: `results/experiments/locus_sets/` (`submit.sh`, `collect.py`)."""),
code("""import sys
from pathlib import Path
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "bigtb_ref.py").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "results/experiments/locus_sets")]

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import paper_style as ps
from collect import load, status, macro, significance, LABEL, MODEL_LABEL, LOCUS_SETS, M5, ALL_DRUGS
from datasets.loader import drug_loci, LAB64_LOCI
ps.apply()
pd.set_option("display.precision", 4)
F = load()
COL = {"locusfusion": ps.MODEL["locusfusion"], "mdcnn": ps.MODEL["cnn"], "lasso": ps.MODEL["lasso"]}
MODSET_LABEL = {"dna": "DNA", "dna_protein": "DNA + protein", "dna_biophysical": "DNA + biophysical",
                "dna_regulatory": "DNA + regulatory", "all_modalities": "all modalities"}"""),
md("## Sweep status\n\nCells finished out of expected (modality sets × drugs pooled). Every table below "
   "leaves a cell blank until all 11 drugs are in."),
code("status(F)"),
md("## Loci per drug"),
code("""pd.DataFrame({LABEL["bigtb"]: [len(drug_loci(d)) for d in ALL_DRUGS],
              LABEL["bigtb19"]: 19,
              LABEL["who_t12"]: [len(drug_loci(d, locus_set="who_t12")) for d in ALL_DRUGS],
              LABEL["lab64"]: len(LAB64_LOCI)}, index=ALL_DRUGS)"""),
md("### The two new locus sets\n\nWHO tier 1+2 genes per drug (tier 1 first), then how the WHO union "
   "and the 64-locus set overlap. The 64-locus set is the same list for every drug."),
code("""pd.set_option("display.max_colwidth", None)
pd.Series({d: ", ".join(drug_loci(d, locus_set="who_t12")) for d in ALL_DRUGS},
          name=LABEL["who_t12"]).to_frame()"""),
code("""who_union = set().union(*(drug_loci(d, locus_set="who_t12") for d in ALL_DRUGS))
lab = set(LAB64_LOCI)
pd.DataFrame({"n": [len(who_union & lab), len(who_union - lab), len(lab - who_union)],
              "loci": [", ".join(sorted(s)) for s in (who_union & lab, who_union - lab, lab - who_union)]},
             index=["in both", "WHO tier 1+2 only", "64 loci only"])"""),
md("""**Read-out:** WHO tier 1+2 gives each drug 6–14 genes (54 in the union); the 64-locus set gives
every drug all 64. They share 50 loci. dnaA, PPE35, rpoA and whiB6 are WHO-only; the 14 loci only in
the 64-locus set are bedaquiline/clofazimine (atpE, mmpL5, mmpS5, pepQ, Rv0678, Rv1979c), delamanid/
pretomanid (ddn, fbiA–C, fgd1, Rv2983), linezolid (rplC) and tsnR; none of them is a WHO tier 1+2 gene for any of
the 11 drugs modelled here."""),

md("## Single-drug\n\nMacro CV AUC (mean over 11 drugs) per model, modality set and locus set."),
code("""S = macro(F[F.setting == "single"]).loc["single"].rename(columns=LABEL)
S.rename(index=MODEL_LABEL, level=0)"""),
code("""def ladder(frame, sets, title):
    fig, axes = plt.subplots(1, len(M5), figsize=(3.0 * len(M5), 3.2), sharey=True)
    x = np.arange(len(sets))
    for ax, m in zip(axes, M5):
        for model in ["lasso", "mdcnn", "locusfusion"]:
            if model == "lasso" and m != "dna":
                continue
            try:
                y = frame.loc[(model, m)].reindex(sets).values
            except KeyError:
                continue
            ax.plot(x, y, "-o", color=COL[model], label=MODEL_LABEL[model], ms=5, lw=1.5)
        ax.set_xticks(x, [LABEL[s] for s in sets], rotation=30, ha="right")
        ax.set_title(MODSET_LABEL[m], fontsize=10)
        ax.grid(axis="y", lw=0.6)
    axes[0].set_ylabel("macro CV AUC")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(title, y=1.13)
    plt.show()

ladder(macro(F[F.setting == "single"]).loc["single"], LOCUS_SETS,
       "Single-drug macro CV AUC by locus set and modality set, lineage split")"""),
md("""**Read-out:** LocusFusion's macro CV AUC (all modalities) goes 0.885 (Big-TB per-drug) → 0.913
(Big-TB 19) → 0.904 (WHO tier 1+2) → 0.915 (64 loci); 64 loci vs Big-TB 19 is not significant (+0.001).
The stacked CNN peaks at Big-TB 19 (0.902) and falls to 0.871 at 64 loci (−0.034, significant, 0/11 drugs
up). LASSO follows LocusFusion's ladder 0.014–0.054 below it in CV."""),

md("Per-drug change in CV AUC from Big-TB's per-drug loci, all modalities (LASSO: DNA one-hot)."),
code("""def delta_grid(frame, setting, base, sets, modset="all_modalities"):
    out = {}
    for model in ["locusfusion", "mdcnn", "lasso"]:
        if setting == "multi" and model == "lasso":
            continue
        ms = "dna" if model == "lasso" else modset
        f = frame[(frame.setting == setting) & (frame.model == model) & (frame.modset == ms)]
        p = f.pivot(index="drug", columns="locus_set", values="cv").reindex(index=ALL_DRUGS)
        for s in sets:
            if s in p and base in p:
                out[(MODEL_LABEL[model], LABEL[s])] = p[s] - p[base]
    return pd.DataFrame(out)

def heat(D, title):
    if D.empty:
        print("no cells yet"); return
    v = np.nanmax(np.abs(D.values)) if np.isfinite(D.values).any() else 0.01
    fig, ax = plt.subplots(figsize=(1.1 * D.shape[1] + 2, 0.38 * D.shape[0] + 1.2))
    im = ax.imshow(D.values, cmap=ps.diverging_cmap(), vmin=-v, vmax=v, aspect="auto")
    for (i, j), z in np.ndenumerate(D.values):
        if np.isfinite(z):
            ax.text(j, i, f"{z:+.3f}", ha="center", va="center", fontsize=8, color=ps.NEUTRAL["ink"])
    ax.set_xticks(range(D.shape[1]), [f"{a}\\n{b}" for a, b in D.columns], fontsize=8)
    ax.set_yticks(range(D.shape[0]), D.index, fontsize=8)
    fig.colorbar(im, ax=ax, label="Δ CV AUC", shrink=0.8)
    ax.set_title(title, fontsize=10)
    plt.show()

D1 = delta_grid(F, "single", "bigtb", ["bigtb19", "who_t12", "lab64"])
heat(D1, "Single-drug Δ CV AUC vs Big-TB per-drug loci, all modalities, by drug")"""),
md("""**Read-out:** WHO tier 1+2 over Big-TB per-drug loci is the one locus-set gain that is significant for
every model (LocusFusion +0.025, LASSO +0.031, 10/11 drugs each). Going from 19 to 64 loci adds nothing
for LocusFusion or LASSO and costs the stacked CNN on every drug."""),

md("## Multi-drug\n\nOne model for all 11 drugs, trained on the union of the drugs' loci "
   "(54 for WHO tier 1+2, 64 for the 64-locus set). Per-drug AUCs are read off the joint model; "
   "the multi-drug lineage split is one partition of the joint cohort "
   "(`results/experiments/locus_sets/splits/MULTIDRUG.csv`)."),
code("""Mm = macro(F[F.setting == "multi"])
Mm.loc["multi"].rename(columns=LABEL).rename(index=MODEL_LABEL, level=0) if len(Mm) else "no multi-drug cells yet\""""),
code("""if len(Mm):
    ladder(Mm.loc["multi"], ["bigtb19", "who_t12", "lab64"],
           "Multi-drug macro CV AUC by locus set and modality set, lineage split")"""),
code("""heat(delta_grid(F, "multi", "bigtb19", ["who_t12", "lab64"]),
     "Multi-drug Δ CV AUC vs Big-TB 19 loci, all modalities, by drug")"""),
md("""**Read-out:** The locus set hardly matters for multi-drug LocusFusion (0.895–0.903, no difference
significant). The multi-drug stacked CNN loses 0.019 (WHO tier 1+2) and 0.025 (64 loci) against Big-TB 19,
both significant. Multi-drug is below single-drug for LocusFusion at Big-TB 19 and 64 loci (about −0.019),
neither significant after Holm."""),

md("### Multi-drug vs single-drug\n\nMacro CV AUC, all modalities. The two settings use "
   "different partitions, so this pairs on drugs, not folds."),
code("""A = F[F.modset == "all_modalities"]
T = (A.groupby(["model", "locus_set", "setting"]).cv.mean()
      .where(A.groupby(["model", "locus_set", "setting"]).cv.count() == len(ALL_DRUGS))
      .unstack("setting").reindex(columns=["single", "multi"]))
T["multi − single"] = T["multi"] - T["single"]
T.rename(index=MODEL_LABEL, level=0).rename(index=LABEL, level=1)"""),

md("## Significance\n\nExact two-sided Wilcoxon signed-rank over the 55 fold-paired differences "
   "(11 drugs × 5 folds), Hodges–Lehmann shift with its exact 95% CI, Holm over every row that "
   "could run. Network rows average the five modality sets fold-by-fold; LASSO is DNA one-hot. "
   "Multi − single rows pair on the 11 drugs. Rows whose cells are not all in are listed as pending."),
code("""sig, pending = significance(F)
sig if len(sig) else "no comparison complete yet\""""),
code("pending"),

md("## Held-out test AUC\n\nMacro test AUC (the best CV fold's model on the lineage test split)."),
code("""pd.concat({"single": macro(F[F.setting == "single"], "test").loc["single"] if (F.setting == "single").any() else None,
           "multi": macro(F[F.setting == "multi"], "test").loc["multi"] if (F.setting == "multi").any() else None},
          names=["setting"]).rename(columns=LABEL)"""),

md("""## Baselines against LocusFusion

LocusFusion against LASSO (DNA one-hot) and the stacked CNN, single-drug, on CV and on the held-out lineage
test split. The stacked CNN and LASSO cells on WHO tier 1+2 and 64 loci are new in this sweep; the Big-TB
cells are the `lineage_split` controls. Fold-paired rows use the same split files for all three models
(`lineage_split/splits`)."""),
code("""from results_helpers import WilcoxonComparisons
B = WilcoxonComparisons(F, ALL_DRUGS)
for s in LOCUS_SETS:
    t = f"single:{s}"
    B.folds("LocusFusion − LASSO", f"{LABEL[s]} · LF 5 modality sets", (t, "locusfusion", M5), (t, "lasso", "dna"))
    B.folds("LocusFusion − LASSO", f"{LABEL[s]} · LF all modalities", (t, "locusfusion", "all_modalities"), (t, "lasso", "dna"))
    B.folds("LocusFusion − stacked CNN", f"{LABEL[s]} · 5 modality sets", (t, "locusfusion", M5), (t, "mdcnn", M5))
B.table()"""),
md("Held-out test AUC per drug, all modalities (LASSO: DNA one-hot), with the test-split size and resistant count."),
code("""split_n = {}
for d in ALL_DRUGS:
    sp = pd.read_csv(ROOT / "results/experiments/lineage_split/splits" / f"{d}.csv")
    t = sp[sp.role == "test"]
    split_n[d] = (len(t), int(t.R.sum()))
S1 = F[F.setting == "single"]
def test_col(model, s):
    ms = "dna" if model == "lasso" else "all_modalities"
    return S1[(S1.model == model) & (S1.locus_set == s) & (S1.modset == ms)].set_index("drug").test.reindex(ALL_DRUGS)
TT = pd.DataFrame({(LABEL[s], MODEL_LABEL[m]): test_col(m, s) for s in LOCUS_SETS
                   for m in ["locusfusion", "lasso", "mdcnn"]})
TT.insert(0, ("test split", "n (R)"), [f"{n} ({r})" for n, r in (split_n[d] for d in ALL_DRUGS)])
summ = pd.DataFrame({"macro": TT.iloc[:, 1:].mean(),
                     "macro excl. LEVOFLOXACIN": TT.iloc[:, 1:].drop(index="LEVOFLOXACIN").mean()}).T
pd.concat([TT, summ])"""),
code("""fig, axes = plt.subplots(1, len(LOCUS_SETS), figsize=(3.3 * len(LOCUS_SETS), 4.2), sharey=True, sharex=True)
y = np.arange(len(ALL_DRUGS))
for ax, s in zip(axes, LOCUS_SETS):
    for k, m in enumerate(["locusfusion", "lasso", "mdcnn"]):
        ax.plot(test_col(m, s).values, y + (k - 1) * 0.22, "o", color=COL[m], ms=5, label=MODEL_LABEL[m])
    ax.set_yticks(y, ALL_DRUGS, fontsize=8)
    ax.set_title(LABEL[s], fontsize=10)
    ax.set_xlabel("test AUC")
    ax.grid(axis="x", lw=0.6)
axes[0].invert_yaxis()
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
fig.suptitle("Single-drug held-out test AUC by drug and locus set, all modalities (LASSO: DNA one-hot)", y=1.12)
plt.show()"""),
code("""gap = (S1.assign(gap=S1.test - S1.cv).groupby(["model", "locus_set"]).gap.mean()
       .unstack().reindex(columns=LOCUS_SETS).rename(index=MODEL_LABEL, columns=LABEL))
gap.rename_axis(index="mean test − CV AUC")"""),
md("""**Read-out:**

- **CV:** LocusFusion beats LASSO at every locus set (+0.013 to +0.054, 10–11/11 drugs, all significant)
  and the stacked CNN everywhere except Big-TB per-drug (n.s.).
- **Held-out test, Big-TB 19:** both baselines are above LocusFusion (LASSO 0.904, stacked CNN 0.906 vs
  0.888). Almost all of the gap is levofloxacin (59 test isolates, 12 R: LocusFusion 0.746 vs 0.917 / 0.920)
  and amikacin. Without levofloxacin the three are within 0.003.
- **Held-out test, 64 loci (new LASSO cell):** LASSO 0.905 vs LocusFusion 0.910; LASSO is higher on 5/11
  drugs, and the drug-paired difference is not significant.
- **CV-to-test drop:** LocusFusion loses 0.011–0.030 from CV to test on the larger locus sets; LASSO
  *gains* 0.004–0.017. The CV margin over LASSO does not fully survive the lineage test split."""),
md("""## Caveats

- **Control FASTAs differ from the new arms.** The single-drug Big-TB controls (`lineage_split`) were
  trained on `data_snapshot`'s alignments, where eis, embB, ethA and rrs carry the insertion bug;
  every new arm (and the multi-drug Big-TB 19 control) uses the repaired copies in
  `locus_sets/aligned`. The changed isolates per locus are 17 (eis), 6 (embB), 225 (ethA), 147 (rrs).
- **More loci vs the right loci.** The 64-locus set adds genes for drugs we do not model
  (bedaquiline, clofazimine, delamanid/pretomanid, linezolid), and is not a superset of WHO tier 1+2:
  it lacks dnaA, PPE35, rpoA and whiB6. A gain from it can come from lineage / co-resistance markers
  rather than mechanism.
- **Regulatory inputs grow with the locus set.** A drug loads the WHO promoter windows of its loaded
  loci only (`DRUG_TO_REGULATORY` ∩ loci), so a larger locus set also adds promoters to
  `dna_regulatory` and `all_modalities`. Multi-drug: 16 windows at Big-TB 19, 48 at WHO tier 1+2, 44
  at 64 loci; furA, lpqB, aftA, embB and rrl have no promoter FASTA and are skipped everywhere.
  The windows are the old TSS-anchored set (`regulatory_msa/aligned`), as in every paper run.
- **Levofloxacin's test split is 59 isolates (12 R)**, and its CV folds hold 17–126 isolates. Its test
  AUC moves by ±0.1–0.2 between modality sets of the same model; it dominates any macro test difference.
- **LASSO's CV AUC is the best of its C grid** (selection on the same folds it reports), so its CV is
  optimistic, but its test AUC is not tuned on test. Its constant-column filter sees every isolate, test
  included; the filter is label-free and only drops columns that never vary. No label leakage was found:
  same split files as the networks, C picked on CV folds, test scored once with the best fold's model.
- **No LocusFusion − LASSO row is in the main Holm family** (`significance()`); the baseline section runs
  its own family.
- **Single seed (0).** One training run per cell; fold-paired tests absorb fold variance, not seed variance.
- **Multi-drug partition.** One lineage partition of the joint cohort (17,941 isolates), balanced
  greedily on isolates and per-drug n and R; its per-drug folds are not the single-drug folds, so
  multi vs single is a drug-paired comparison only. Levofloxacin has 8–12 R per CV fold there.
- **The 14 loci only the lab folder had were re-extracted from the VCFs** (gene ± 100 bp, fixed
  extractor; `locus_sets/extract/`): the lab's copies carry the insertion bug, its `pepQ` file is
  Rv3698 and its `mmpS5` window is shifted 100 bp."""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT}")
