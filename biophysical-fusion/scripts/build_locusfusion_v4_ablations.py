"""Builds notebooks/locusfusion_v4_ablations.ipynb -- v4/v5 arms against v3, and
what each ablation is worth. Short companion to locusfusion_v4_results.ipynb.

    python scripts/build_locusfusion_v4_ablations.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/locusfusion_v4_ablations.ipynb

Reads result JSONs only; trains and scores nothing.
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
# LocusFusion v4 / v5 vs v3: ablations

11 drugs, lineage splits, 5 CV folds each. Significance = exact Wilcoxon over the
55 fold-paired CV AUC differences, Hodges-Lehmann shift with 95% CI, Holm across
each table.

| arm | change |
|---|---|
| `noaug` | v4 token (residue one-hot → consequence flag, + distance), no augmentation |
| `aug` | `noaug` + synonymous augmentation |
| `nodist` | `aug` without distance channels |
| `nosyn` | `aug` with synonymous tokens dropped instead |
| `v5` | v3 residue one-hot kept + consequence + distance, synonymous dropped, no aug |
| `v5_nodist` | `v5` without distance |
| `v5_fixdna` | `v5` on insertion-bug-fixed eis/embB/ethA/rrs |
| `v5_atg` | `v5` on start-codon-anchored promoters |
| `v5_fixed` | `v5` on both fixed inputs |
""")

code(r"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display

PROJECT = next(p for p in [Path.cwd().parent, Path.cwd()]
               if (p / "paper_style.py").exists())
sys.path.insert(0, str(PROJECT))
import paper_style as PS
from results_helpers import hodges_lehmann, holm
PS.apply()
pd.set_option("display.width", 200)

V4 = PROJECT / "results" / "experiments" / "locusfusion_v4"
V3 = PROJECT / "results" / "experiments" / "lineage_split" / "sd19_all_modalities_unk__locusfusion_v2"
DRUGS = ["AMIKACIN", "CAPREOMYCIN", "ETHAMBUTOL", "ETHIONAMIDE", "ISONIAZID", "KANAMYCIN",
         "LEVOFLOXACIN", "MOXIFLOXACIN", "PYRAZINAMIDE", "RIFAMPICIN", "STREPTOMYCIN"]
ABBR = {"AMIKACIN": "AMI", "CAPREOMYCIN": "CAP", "ETHAMBUTOL": "EMB", "ETHIONAMIDE": "ETO",
        "ISONIAZID": "INH", "KANAMYCIN": "KAN", "LEVOFLOXACIN": "LEVO", "MOXIFLOXACIN": "MOXI",
        "PYRAZINAMIDE": "PZA", "RIFAMPICIN": "RIF", "STREPTOMYCIN": "STR"}
SD = ["noaug", "aug", "nodist", "nosyn", "v5", "v5_nodist", "v5_fixdna", "v5_atg", "v5_fixed"]
MD = ["md_noaug", "md_aug", "md_nodist", "md_nosyn"]

cv, test, nval = {}, {}, {}      # arm -> DataFrame (drug x fold) / Series (drug)


def put(arm, drug, folds, t):
    cv.setdefault(arm, {})[drug] = [f["per_drug"][drug]["auc"] if "per_drug" in f else f["auc"]
                                    for f in folds]
    nval.setdefault(arm, {})[drug] = [f["per_drug"][drug]["n"] if "per_drug" in f else f["n"]
                                      for f in folds]
    test.setdefault(arm, {})[drug] = t["auc"]


for p in V3.glob("*__*.json"):
    j = json.loads(p.read_text())
    put("v3", j["drug"], j["cv_folds"], j["test"])
for arm in SD:
    for d in DRUGS:
        j = json.loads((V4 / f"sd19_{arm}" / f"{d}.json").read_text())
        put(arm, d, j["cv_folds"], j["per_drug"][d]["test"])
for arm in MD:
    j = json.loads((V4 / arm / "MULTIDRUG.json").read_text())
    for d in DRUGS:
        put(arm, d, j["cv_folds"], j["per_drug"][d]["test"])

cv = {a: pd.DataFrame(v).T.loc[DRUGS] for a, v in cv.items()}
test = {a: pd.Series(v).loc[DRUGS] for a, v in test.items()}
# fold pairing: every single-drug arm must see the same validation-set sizes as v3
for a in SD:
    assert all(nval[a][d] == nval["v3"][d] for d in DRUGS), a
assert all(nval[a] == nval["md_noaug"] for a in MD)
print({a: cv[a].shape for a in cv})
""")

code(r"""
def compare(pairs):
    rows = []
    for a, b in pairs:
        x = (cv[a] - cv[b]).values.ravel()
        hl, lo, hi = hodges_lehmann(x)
        rows.append({"contrast": f"{a} − {b}", "a": a, "b": b,
                     "ΔCV mean": x.mean(), "HL": hl, "lo": lo, "hi": hi,
                     "ΔTEST mean": (test[a] - test[b]).mean(),
                     "drugs better (CV)": int(((cv[a] - cv[b]).mean(axis=1) > 0).sum()),
                     "p": stats.wilcoxon(x, method="exact").pvalue})
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    return t


def forest(t, title, xlabel="Δ CV AUC (Hodges-Lehmann, 95% CI)"):
    t = t.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(t) + 0.9), constrained_layout=True)
    c = PS.MODEL["locusfusion"]
    L = max(0.03, 1.15 * np.abs(t[["lo", "hi"]].values).max())   # drugs beyond ±L sit on the edge
    for i, r in t.iterrows():
        per = (cv[r.a] - cv[r.b]).mean(axis=1).values
        y = np.full(len(per), i) + np.random.default_rng(i).uniform(-.15, .15, len(per))
        out = np.abs(per) > L
        ax.scatter(per[~out], y[~out], s=10, color=PS.NEUTRAL["muted"], alpha=.6, lw=0, zorder=2)
        for v, yy in zip(per[out], y[out]):
            ax.scatter([np.sign(v) * L], [yy], s=22, color=PS.NEUTRAL["muted"], lw=0, zorder=2,
                       marker=">" if v > 0 else "<")
            ax.annotate(f"{v:+.2f}", (np.sign(v) * L, yy), fontsize=6.5, color=PS.NEUTRAL["muted"],
                        ha="left" if v > 0 else "right", va="center",
                        xytext=(3 if v > 0 else -3, 0), textcoords="offset points")
        ax.plot([r.lo, r.hi], [i, i], color=c, lw=2, zorder=3)
        ax.scatter([r.HL], [i], color=c, s=36, zorder=4)
        star = "*" if r.p_holm < 0.05 else ""
        ax.annotate(f"p_Holm={r.p_holm:.2g}{star}", (1.02, i), xycoords=("axes fraction", "data"),
                    va="center", fontsize=8, color=PS.NEUTRAL["ink2"], annotation_clip=False)
    ax.axvline(0, color=PS.NEUTRAL["axis"], lw=1, zorder=1)
    ax.set_xlim(-L * 1.25, L * 1.25)
    ax.set_yticks(range(len(t)), t.contrast)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    plt.show()


cols = ["ΔCV mean", "HL", "lo", "hi", "ΔTEST mean", "drugs better (CV)", "p", "p_holm"]
""")

md(r"""
## Every arm vs v3

Mean CV and TEST AUC over 11 drugs.
""")

code(r"""
summ = pd.DataFrame({"CV AUC": {a: cv[a].values.mean() for a in ["v3"] + SD},
                     "TEST AUC": {a: test[a].mean() for a in ["v3"] + SD}})
display(summ.round(4))

vs3 = compare([(a, "v3") for a in SD])
display(vs3.set_index("contrast")[cols].round(4))
forest(vs3, "Single-drug arms minus v3, CV AUC, 11 drugs × 5 folds")
""")

md(r"""
Copper = HL shift and CI over 55 folds; grey dots = per-drug mean Δ (triangles = off-scale,
value printed). Holm-adjusted p on the right (`*` < 0.05).
""")

md(r"""
## Each ablation in isolation

Each row changes one thing relative to its parent arm.
""")

code(r"""
abl = compare([
    ("aug", "noaug"),          # synonymous augmentation
    ("aug", "nodist"),         # distance channels, both with aug
    ("nosyn", "aug"),          # drop synonymous vs teach invariance
    ("v5", "nosyn"),           # residue one-hot back (and no aug)
    ("v5", "v5_nodist"),       # distance channels, on v5
    ("v5_fixdna", "v5"),       # insertion-bug fix
    ("v5_atg", "v5"),          # ATG-anchored promoters
    ("v5_fixed", "v5"),        # both data fixes
])
display(abl.set_index("contrast")[cols].round(4))
forest(abl, "Single-drug ablation contrasts, CV AUC, 11 drugs × 5 folds")
""")

md(r"""
## Per drug, TEST AUC minus v3
""")

code(r"""
dt = pd.DataFrame({a: test[a] - test["v3"] for a in SD}).rename(index=ABBR).T
lim = np.nanmax(np.abs(dt.values))
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
im = ax.imshow(dt.values, cmap=PS.diverging_cmap(), vmin=-lim, vmax=lim, aspect="auto")
for (i, k), v in np.ndenumerate(dt.values):
    ax.text(k, i, f"{v:+.3f}", ha="center", va="center", fontsize=7, color=PS.NEUTRAL["ink"])
ax.set_xticks(range(dt.shape[1]), dt.columns)
ax.set_yticks(range(dt.shape[0]), dt.index)
fig.colorbar(im, ax=ax, label="Δ TEST AUC vs v3", shrink=.8)
ax.set_title("TEST AUC minus v3, by arm and drug")
plt.show()
""")

md(r"""
## Synonymous changes on an H37Rv background

Each of a locus's synonymous single-base changes is put alone on an otherwise
reference isolate. Shift = mean P(R) over those changes minus P(R) of the bare
H37Rv background (same model). Left: PZA, the 418 synonymous pncA changes, the
same set v3 was scanned on (`results/analysis/pza_substitution_scan`). Right:
every drug × the 17 coding loci, v4/v5 arms only (v3 was scanned on PZA only).
""")

code(r"""
scan = pd.read_csv(PROJECT / "results" / "analysis" / "pza_substitution_scan" / "scan.csv")
chk = json.loads((PROJECT / "results" / "analysis" / "pza_substitution_scan" / "checks.json").read_text())
syn3 = scan.loc[scan.kind == "synonymous", "p_all_modalities"]
v3_base = chk["p_resistant_H37Rv_background"]["all_modalities"]

shift = []   # arm, drug, locus, n, shift
for a in SD:
    for d in DRUGS:
        h = json.loads((V4 / f"sd19_{a}" / f"{d}.json").read_text())["probe"]["h37rv_background"][d]
        for loc, v in h["by_locus"].items():
            shift.append({"arm": a, "drug": d, "locus": loc, "n": v["n"],
                          "shift": v["mean_P_R"] - h["P_R_empty"]})
shift = pd.DataFrame(shift)
SEEN = ["noaug", "aug", "nodist"]                  # trained with synonymous tokens + consequence flag
DROP = [a for a in SD if a not in SEEN]            # synonymous tokens filtered: never seen in training

pza = (shift[(shift.drug == "PYRAZINAMIDE") & (shift.locus == "pncA")]
       .set_index("arm")["shift"].reindex(SD))
assert (shift[(shift.drug == "PYRAZINAMIDE") & (shift.locus == "pncA")].n == len(syn3)).all()
tab = pd.concat([pd.Series({"v3": syn3.mean() - v3_base}), pza]).rename("PZA pncA shift")
display(tab.to_frame().assign(
    **{"median |shift|, all drugs × loci": shift.groupby("arm")["shift"].apply(lambda x: x.abs().median())}).round(4))

c_v3, c_seen, c_drop = PS.ramp(PS.MODEL["locusfusion"], 3, light=0.75)
col = {"v3": c_v3, **{a: c_seen for a in SEEN}, **{a: c_drop for a in DROP}}
order = ["v3"] + SEEN + DROP

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True,
                               gridspec_kw={"width_ratios": [1, 1.25]})
ax1.bar(range(len(order)), tab[order], color=[col[a] for a in order], width=.7)
for i, v in enumerate(tab[order]):
    ax1.text(i, v + .005, f"{v:+.3f}", ha="center", va="bottom", fontsize=7, color=PS.NEUTRAL["ink"])
ax1.set_xticks(range(len(order)), order, rotation=45, ha="right")
ax1.set_ylabel("mean P(R) shift")
ax1.set_title("PZA: one synonymous pncA change, H37Rv background (n=418)")

arms = SEEN + DROP
for i, a in enumerate(arms):
    x = shift.loc[shift.arm == a, "shift"].abs().clip(lower=1e-5).values
    j = np.random.default_rng(i).uniform(-.25, .25, len(x))
    ax2.scatter(i + j, x, s=6, color=col[a], alpha=.6, lw=0)
    ax2.plot([i - .3, i + .3], [np.median(x)] * 2, color=PS.NEUTRAL["ink"], lw=1.5)
ax2.axhline(tab["v3"], color=PS.NEUTRAL["muted"], lw=1, ls="--")
ax2.annotate(f"v3 PZA pncA {tab['v3']:+.2f}", (len(arms) - .5, tab["v3"]), ha="right", va="bottom",
             fontsize=8, color=PS.NEUTRAL["muted"])
ax2.set_yscale("log")
ax2.set_xticks(range(len(arms)), arms, rotation=45, ha="right")
ax2.set_ylabel("|mean P(R) shift|")
ax2.set_title("All 11 drugs × 17 loci, one synonymous change, H37Rv background")
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y")
from matplotlib.patches import Patch
ax1.legend(handles=[Patch(color=c_v3, label="v3"),
                    Patch(color=c_seen, label="synonymous tokens kept in training"),
                    Patch(color=c_drop, label="synonymous tokens dropped")],
           fontsize=8, loc="upper right")
plt.show()
""")

md(r"""
- PZA pncA: v3 +0.38; every arm trained on synonymous tokens with the consequence
  flag ≤ +0.001, augmentation or not.
- All drugs × loci: only the augmented arms (`aug`, `nodist`) are invariant
  everywhere (median |shift| 0.0002). `noaug` is ~50× worse (0.010), on par with
  the dropped arms.
- Dropped arms (`nosyn`, `v5*`) never saw a synonymous token, so an injected one
  is out of distribution (pncA +0.09 to +0.22). In use their tokenizer removes it,
  so their effective shift is 0 by construction.
""")

md(r"""
## Multi-drug arms

One model for all 11 drugs on its own split, so no v3 pairing; contrasts are
between `md_*` arms (55 = 11 drugs × 5 folds, same folds).
""")

code(r"""
display(pd.DataFrame({"CV AUC": {a: cv[a].values.mean() for a in MD},
                      "TEST AUC": {a: test[a].mean() for a in MD}}).round(4))
mdc = compare([("md_aug", "md_noaug"), ("md_aug", "md_nodist"), ("md_nosyn", "md_aug")])
display(mdc.set_index("contrast")[cols].round(4))
forest(mdc, "Multi-drug ablation contrasts, CV AUC, 11 drugs × 5 folds")
""")

md(r"""
## Caveats

- One seed per cell; TEST is one model per drug (best CV fold), so per-drug TEST
  deltas of ~0.02 are within run-to-run noise. The Wilcoxon is on CV only.
- `v5_fixdna` / `v5_atg` / `v5_fixed` change the input data, so they are not
  matched to v3's data.
- `v5 − nosyn` changes two things at once (residue one-hot back, augmentation off).
- Synonymous probe: v3's H37Rv background is the real reference sequence through the v3
  encoder; v4's is an empty token set. Both are "no variants", built by different code.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                                                  "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).resolve().parents[1] / "notebooks" / "locusfusion_v4_ablations.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(cells)} cells)")
