"""Builds notebooks/analysis/modality_flips.ipynb -- when adding a modality changes a
LocusFusion v3 call, and why.

    python scripts/build_modality_flips.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/analysis/modality_flips.ipynb

Reads only raw model outputs: results/analysis/predictions_locusfusion_v3/
(scripts/predict_locusfusion_v3.py) and results/analysis/pza_substitution_scan/
(scripts/pza_substitution_scan.py). The 19-loci section covers whichever drugs
the prediction sweep has finished; re-execute when it completes. Variant calls
come from the aligned FASTAs in data_snapshot/. CPU only, ~10 min (FASTA parsing).
"""
import json
from pathlib import Path

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})

md(r"""
# Modality flips in LocusFusion v3

Test isolates whose resistance call changes when protein, biophysical or regulatory tokens are added to the
DNA-only model, and whether the change can be traced to the isolate's variants. All predictions are raw outputs
of the saved best-fold checkpoints (`results/analysis/predictions_locusfusion_v3/`, rescored test AUCs match the
result JSONs). Calls use each model's operating threshold refit on its own validation fold (the Big-TB rule, max
sensitivity + specificity), not the test-fit threshold in the result JSONs.
""")

code(r"""
import os, sys, bisect
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import roc_auc_score

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "bigtb_ref.py").exists())
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ABR_GENBANK", str(ROOT.parent / "data_snapshot/genbank/genome.gbff"))
import paper_style as ps
ps.apply()
from Bio import SeqIO
from Bio.Seq import Seq
from datasets import cds
from datasets.sequences import reference_row

PRED = ROOT / "results/analysis/predictions_locusfusion_v3"
SCAN_DIR = ROOT / "results/analysis/pza_substitution_scan"
GD = str(ROOT.parent / "data_snapshot/genomic_data/aligned")
OUT = ROOT / "results/figures/modality_flips"
OUT.mkdir(parents=True, exist_ok=True)

DRUG_TO_LOCI = {"ISONIAZID": ["inhA", "katG"], "RIFAMPICIN": ["rpoB", "rpoC"], "ETHAMBUTOL": ["embC", "embA", "embB"],
                "PYRAZINAMIDE": ["pncA"], "STREPTOMYCIN": ["rpsL", "rrs", "gid"], "KANAMYCIN": ["rrs"],
                "AMIKACIN": ["rrs", "eis"], "CAPREOMYCIN": ["rrs", "rrl", "tlyA"], "LEVOFLOXACIN": ["gyrB", "gyrA"],
                "MOXIFLOXACIN": ["gyrB", "gyrA"], "ETHIONAMIDE": ["inhA", "ethA", "ethR"]}     # bigtb_ref tb.DRUG_TO_LOCI
MODSETS = {"dna": "DNA only", "dna_protein": "+ protein", "dna_biophysical": "+ biophysical",
           "dna_regulatory": "+ regulatory", "all_modalities": "all modalities"}
ADDED = [m for m in MODSETS if m != "dna"]
COL = {"dna": ps.MODALITY["dna"], "dna_protein": ps.MODALITY["protein"], "dna_biophysical": ps.MODALITY["biophysical"],
       "dna_regulatory": ps.MODALITY["regulatory"], "all_modalities": ps.MODEL["locusfusion"]}
MARGIN = 0.2      # a "confident" flip: both models at least this far from their own threshold


def cell(arm, ms):
    return PRED / f"{arm}_{ms}_unk__locusfusion_v2"


def predictions(arm, drug):
    # long frame over the five modality sets, or None if the sweep has not written all of them yet
    paths = [cell(arm, ms) / f"{drug}.csv.gz" for ms in MODSETS]
    if not all(p.exists() for p in paths):
        return None
    return pd.concat([pd.read_csv(p).assign(modset=ms) for p, ms in zip(paths, MODSETS)], ignore_index=True)


def youden_threshold(y_res, p_res):
    # tb.get_threshold_val on the resistant scale: cut-offs 0.00..1.00 step 0.01 on P(susceptible),
    # max sensitivity + specificity, ties to the largest; returned as a P(resistant) cut-off
    pS, yS = 1 - p_res, 1 - y_res
    best = None
    for t in np.linspace(0, 1, 101):
        callR = pS < t
        s = (1 - (callR & (yS == 1)).sum() / (yS == 1).sum()) + (callR & (yS == 0)).sum() / (yS == 0).sum()
        if best is None or s >= best[0]:
            best = (s, t)
    return 1 - best[1]


def calls(P):
    # test rows, wide: p_<ms>, th_<ms> (validation-fit), call_<ms>, ok_<ms>
    th = {m: youden_threshold(g.y_resistant.values, g.p_resistant.values) for m, g in P[P.role == "val"].groupby("modset")}
    T = P[P.role == "test"].pivot(index=["isolate_id", "y_resistant"], columns="modset", values="p_resistant").reset_index()
    for m in MODSETS:
        T[f"p_{m}"], T[f"th_{m}"] = T.pop(m), th[m]
        T[f"call_{m}"] = T[f"p_{m}"] > th[m]
        T[f"ok_{m}"] = T[f"call_{m}"] == (T.y_resistant == 1)
    return T


def flips(T, m):
    fixed, broken = ~T.ok_dna & T[f"ok_{m}"], T.ok_dna & ~T[f"ok_{m}"]
    conf = ((T.p_dna - T.th_dna).abs() > MARGIN) & ((T[f"p_{m}"] - T[f"th_{m}"]).abs() > MARGIN)
    return fixed, broken, conf


def changed_isolates(gene):
    # isolates with any called (non-N) difference from H37Rv anywhere in the locus record
    ref = reference_row(GD, gene).upper()
    out = set()
    for r in SeqIO.parse(f"{GD}/{gene}.fasta", "fasta"):
        if "H37Rv" in r.id:
            continue
        s = str(r.seq).upper()
        if any(a != b and a != "N" for a, b in zip(s, ref)):
            out.add(r.id)
    return out


def coding_changes(gene):
    # isolate -> (set of changes on the H37Rv codon grid, {codon: (ref codon, isolate codon)})
    ref = reference_row(GD, gene).upper()
    w0, w1 = cds.cds_columns(GD, gene)
    cols = [c for c in range(w0, w1) if ref[c] != "-"]
    codons = [cols[j:j + 3] for j in range(0, len(cols) - 2, 3)]
    rc = ["".join(ref[c] for c in t) for t in codons]
    ins_cols = [c for c in range(w0, w1) if ref[c] == "-"]
    out = {}
    for r in SeqIO.parse(f"{GD}/{gene}.fasta", "fasta"):
        if "H37Rv" in r.id:
            continue
        s, v = str(r.seq).upper(), set()
        for k, t in enumerate(codons):
            a = "".join(s[c] for c in t)
            if a == rc[k] or not set(a) <= set("ACGT-"):
                continue
            if "-" in a:
                v.add(f"codon {k + 1} deletion")
                continue
            ra, aa = str(Seq(rc[k]).translate()), str(Seq(a).translate())
            v.add(f"{ra}{k + 1}{aa}" + ("(syn)" if ra == aa else ""))
        for c in ins_cols:
            if s[c] not in "-N":
                v.add(f"insertion after codon {(bisect.bisect_left(cols, c) - 1) // 3 + 1}")
        out[r.id] = (v, {k + 1: (rc[k], "".join(s[c] for c in t)) for k, t in enumerate(codons)})
    return out


def kind(v):
    return ("synonymous" if v.endswith("(syn)") else "nonsense" if v.endswith("*")
            else "indel" if ("deletion" in v or v.startswith("insertion")) else "missense")


def tidy(ax, grid="y"):
    for s in ("right", "top"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, lw=0.6)
        ax.set_axisbelow(True)
""")

# --------------------------------------------------------------------------- 19 loci
md(r"""
## 1. 19 loci: how often calls change, and whether the drug's own genes explain it

Every 19-loci modality set against DNA only, on each drug's held-out test set. *Fixed*: DNA only wrong, the
added-modality model right; *broken*: the reverse. Only drugs whose five 19-loci cells the prediction sweep has
written are included.
""")

code(r"""
rows, conf_rows, done = [], [], []
for drug in DRUG_TO_LOCI:
    P = predictions("sd19", drug)
    if P is None:
        continue
    done.append(drug)
    T = calls(P)
    own = set().union(*[changed_isolates(g) for g in DRUG_TO_LOCI[drug]])
    for m in ADDED:
        fixed, broken, conf = flips(T, m)
        rows.append(dict(drug=drug, modset=m, n=len(T), fixed=int(fixed.sum()), broken=int(broken.sum())))
        for iso in T.isolate_id[fixed & conf]:
            conf_rows.append(dict(drug=drug, modset=m, isolate=iso, own_locus_change=iso in own))
print(f"19-loci cells available for {len(done)}/11 drugs: {', '.join(done) or 'none'}")
F19 = pd.DataFrame(rows)
C19 = pd.DataFrame(conf_rows, columns=["drug", "modset", "isolate", "own_locus_change"])
if len(F19):
    display(F19.pivot(index="drug", columns="modset", values=["fixed", "broken"]).rename(columns=MODSETS, level=1))
""")

md(r"""
### Figure 1. Calls fixed and broken relative to DNA only, 19 loci
""")

code(r"""
if len(F19):
    drugs = list(dict.fromkeys(F19.drug))
    fig, axes = plt.subplots(1, len(ADDED), figsize=(3.4 * len(ADDED), 0.42 * len(drugs) + 1.8), sharey=True)
    for ax, m in zip(axes, ADDED):
        g = F19[F19.modset == m].set_index("drug").loc[drugs]
        y = np.arange(len(drugs))
        ax.barh(y + 0.2, g.fixed, height=0.38, color=COL[m], label="fixed")
        ax.barh(y - 0.2, -g.broken, height=0.38, color=ps.NEUTRAL["muted"], label="broken")
        ax.axvline(0, color=ps.NEUTRAL["axis"], lw=0.8)
        ax.set_yticks(y, [d.title() for d in drugs])
        ax.set_title(MODSETS[m], fontsize=11)
        ax.set_xlabel("test isolates (broken ← → fixed)")
        tidy(ax, "x")
    axes[0].invert_yaxis()
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ps.NEUTRAL["ink2"], label="fixed (bar in modality colour)"),
                        plt.Rectangle((0, 0), 1, 1, color=ps.NEUTRAL["muted"], label="broken")],
               loc="lower center", ncol=2, fontsize=9.5)
    fig.suptitle("LocusFusion v3, 19 loci: test calls fixed and broken vs DNA only, by drug and modality set", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(OUT / "fig1_flips_19loci.png", dpi=200)
    plt.show()
    if len(C19):
        share = C19.groupby("modset").own_locus_change.agg(["size", "mean"]).rename(index=MODSETS)
        share.columns = ["confident fixes", "with any change in the drug's own genes"]
        display(share.round(2))
""")

md(r"""
Confident fixes are flips where both models sit at least 0.2 from their own threshold. Where a fix has no change
in the drug's own genes, the call moved on the other loci (co-resistance and lineage background), not on a
variant that bears on that drug.
""")

# --------------------------------------------------------------------------- PZA per-drug
md(r"""
## 2. Per-drug loci, pyrazinamide: *pncA* only

With one locus every token comes from *pncA*, so a changed call can only come from the isolate's *pncA* variants.
""")

code(r"""
P = predictions("sd", "PYRAZINAMIDE")
T = calls(P)
th_tab = pd.DataFrame({
    "threshold (validation fold)": {m: T[f"th_{m}"].iloc[0] for m in MODSETS},
    "threshold (test, reported)": P.groupby("modset").threshold_p_resistant.first(),
    "test AUC": P[P.role == "test"].groupby("modset").apply(
        lambda g: roc_auc_score(g.y_resistant, g.p_resistant)),
    "test sensitivity": {m: (T[f"call_{m}"] & (T.y_resistant == 1)).sum() / (T.y_resistant == 1).sum() for m in MODSETS},
    "test specificity": {m: (~T[f"call_{m}"] & (T.y_resistant == 0)).sum() / (T.y_resistant == 0).sum() for m in MODSETS},
    "fixed vs DNA only": {m: int(flips(T, m)[0].sum()) for m in MODSETS},
    "broken vs DNA only": {m: int(flips(T, m)[1].sum()) for m in MODSETS},
}).loc[list(MODSETS)].rename(index=MODSETS)
display(th_tab.round(3))
""")

md(r"""
+ regulatory is identical to DNA only for this drug (no *pncA* promoter window in the per-drug set). Refitting on
the validation fold moves + protein's cut-off from 0.14 to 0.32; at the test-fit thresholds the flip counts are
dominated by that one outlying cut-off.
""")

code(r"""
CH = coding_changes("pncA")
base = P[P.modset == "dna"].set_index("isolate_id")
train_counts = {}
for iso in base.index[base.role != "test"]:
    for v in CH.get(iso, (set(), {}))[0]:
        train_counts.setdefault(v, [0, 0])[0 if base.y_resistant[iso] else 1] += 1      # [R, S]

rows = []
for _, r in T.iterrows():
    vs = CH.get(r.isolate_id, (set(), {}))[0]
    if len(vs) == 1:
        v = next(iter(vs))
        rows.append(dict(isolate=r.isolate_id, R=r.y_resistant, change=v, kind=kind(v),
                         seen=sum(train_counts.get(v, [0, 0])) > 0, **{m: r[f"p_{m}"] for m in MODSETS}))
ONE = pd.DataFrame(rows)
tab = ONE.groupby(["seen", "kind"]).agg(n=("R", "size"), pct_R=("R", lambda x: 100 * x.mean()),
                                          **{MODSETS[m]: (m, "mean") for m in MODSETS if m != "dna_regulatory"})
display(tab.round(2))
tr = base.index[base.role != "test"]
for k in ["synonymous", "nonsense"]:
    ids = [i for i in tr if {kind(v) for v in CH.get(i, (set(), {}))[0]} == {k}]
    print(f"training isolates whose only pncA change type is {k}: n = {len(ids)}, "
          f"{100 * np.mean([base.y_resistant[i] for i in ids]):.0f}% resistant")
""")

md(r"""
### Figure 2. Mean P(resistant) by *pncA* change type, test isolates with a single coding change

Grouped by whether that exact residue change occurs in the training split. n per group on the axis.
""")

code(r"""
SHOW = [m for m in MODSETS if m != "dna_regulatory"]
groups = [(s, k) for s in (False, True) for k in ["synonymous", "missense", "nonsense", "indel"]
          if ((ONE.seen == s) & (ONE.kind == k)).any()]
fig, ax = plt.subplots(figsize=(11.5, 4.4))
w = 0.8 / len(SHOW)
for j, m in enumerate(SHOW):
    vals = [ONE[(ONE.seen == s) & (ONE.kind == k)][m].mean() for s, k in groups]
    ax.bar(np.arange(len(groups)) + (j - (len(SHOW) - 1) / 2) * w, vals, width=w, color=COL[m], label=MODSETS[m])
ax.set_xticks(range(len(groups)), [f"{k}\n{'in training' if s else 'not in training'}\n"
                                   f"n = {((ONE.seen == s) & (ONE.kind == k)).sum()}, "
                                   f"{100 * ONE[(ONE.seen == s) & (ONE.kind == k)].R.mean():.0f}% R"
                                   for s, k in groups], fontsize=9)
ax.set_ylim(0, 1); ax.set_ylabel("mean P(resistant)")
tidy(ax)
ax.legend(ncol=4, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
ax.set_title("LocusFusion v3, pyrazinamide (pncA only): mean P(resistant) by pncA change type", fontsize=12, pad=30)
fig.tight_layout()
fig.savefig(OUT / "fig2_pza_change_types.png", dpi=200)
plt.show()
""")

md(r"""
### Figure 3. Two single-base *pncA* changes absent from training

A nonsense change in a resistant isolate and a synonymous change in a susceptible one. Bars: P(resistant) and the
call in brackets; ticks: validation-fold thresholds. The subtitle counts training isolates carrying the exact base
change (from the substitution scan, §3).
""")

code(r"""
SCAN = pd.read_csv(SCAN_DIR / "scan.csv")
EXAMPLES = ["SAMN05915262", "SAMEA1118274"]
TH = {m: T[f"th_{m}"].iloc[0] for m in MODSETS}
N_CODONS = 186
fig = plt.figure(figsize=(12.5, 5.8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.6], hspace=0.55, wspace=0.28, left=0.11, right=0.98, top=0.83, bottom=0.12)
for j, iso in enumerate(EXAMPLES):
    r = ONE.set_index("isolate").loc[iso]
    k = int("".join(ch for ch in r.change if ch.isdigit()))
    ref_c, alt_c = CH[iso][1][k]
    sc = SCAN[(SCAN.codon == k) & (SCAN.alt_codon == alt_c)].iloc[0]
    ax = fig.add_subplot(gs[0, j])
    ax.add_patch(plt.Rectangle((1, -0.3), N_CODONS - 1, 0.6, color=ps.NEUTRAL["grid"], lw=0))
    if r.kind == "nonsense":
        ax.add_patch(plt.Rectangle((k, -0.3), N_CODONS - k, 0.6, facecolor="white", edgecolor=ps.NEUTRAL["axis"],
                                   hatch="////", lw=0.6))
    ax.plot([k, k], [-0.55, 0.55], color=ps.NEUTRAL["ink"], lw=2)
    ax.text(k, 0.8, f"codon {k}: {ref_c} → {alt_c}   ({r.change.replace('(syn)', '')}"
                    f"{', stop' if r.kind == 'nonsense' else ', synonymous'})", ha="center", va="bottom", fontsize=10.5)
    ax.set_xlim(-2, N_CODONS + 3); ax.set_ylim(-0.7, 1.6)
    ax.set_yticks([]); ax.set_xticks([1, 50, 100, 150, 186]); ax.set_xlabel("PncA codon", fontsize=9)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"{iso}  ·  phenotype {'resistant' if r.R else 'susceptible'}\n"
                 f"training isolates with this base change: {sc.train_carriers}"
                 + (f" ({sc.train_R} resistant)" if sc.train_carriers else ""), fontsize=11, loc="left")
    ax = fig.add_subplot(gs[1, j])
    ms = SHOW[::-1]
    ax.barh(range(len(ms)), [r[m] for m in ms], color=[COL[m] for m in ms], height=0.62)
    for i, m in enumerate(ms):
        ax.plot([TH[m]] * 2, [i - 0.42, i + 0.42], color=ps.NEUTRAL["ink"], lw=1.6)
        inside = r[m] > 0.8
        ax.text(r[m] - 0.015 if inside else r[m] + 0.015, i, f"{r[m]:.2f}  ({'R' if r[m] > TH[m] else 'S'})",
                va="center", ha="right" if inside else "left", fontsize=10,
                color="white" if inside else ps.NEUTRAL["ink"])
    ax.set_yticks(range(len(ms)), [MODSETS[m] for m in ms] if j == 0 else [""] * len(ms), fontsize=10.5)
    ax.set_xlim(0, 1); ax.set_xlabel("P(resistant)")
    tidy(ax, "x")
fig.legend(handles=[plt.Line2D([], [], color=ps.NEUTRAL["ink"], lw=1.6, label="operating threshold (fit on validation fold)"),
                    plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=ps.NEUTRAL["axis"], hatch="////",
                                  label="codons after the stop")],
           loc="lower center", ncol=2, fontsize=9.5, bbox_to_anchor=(0.55, 0.0))
fig.suptitle("LocusFusion v3, pyrazinamide (pncA only): P(resistant) by modality set for two test isolates", fontsize=13)
fig.savefig(OUT / "fig3_pza_examples.png", dpi=200)
plt.show()
""")

md(r"""
- **S66\*** (TCG→TAG, resistant; no training isolate has this base): DNA only 0.09, called susceptible;
  + protein 0.78 and all modalities 0.82 call it resistant. The residue token is a stop codon, and a premature stop
  is ~91% resistant in training. PncA activates the prodrug, so truncation at codon 66 of 186 is loss of function.
- **A25A** (GCC→GCG, susceptible): DNA only 0.97. One training isolate carries this exact base, a resistant one
  with a highly divergent *pncA*; §3 shows the DNA model's score at codon 25 follows that one carrier.
  + protein 0.17 and all modalities 0.16 call it susceptible; + biophysical 0.68 does not.
""")

# --------------------------------------------------------------------------- scan
md(r"""
## 3. Every single-base *pncA* substitution on an H37Rv background

All 1,674 single-base substitutions in the *pncA* CDS, each alone on an otherwise-H37Rv isolate, scored by the
same per-drug checkpoints (`scripts/pza_substitution_scan.py`). Inputs are built with the loader's encoders; the
script checks them against real isolates and reproduces the two §2 examples' stored predictions.
""")

code(r"""
S = SCAN.assign(seen=SCAN.train_carriers > 0)
pc = {m: f"p_{m}" for m in SHOW}
g = S.groupby(["kind", "seen"])
scan_tab = pd.concat({"mean P(resistant)": g[list(pc.values())].mean().rename(columns={c: MODSETS[m] for m, c in pc.items()}),
                      "called R": g.apply(lambda x: pd.Series({MODSETS[m]: (x[c] > TH[m]).mean() for m, c in pc.items()}))},
                     axis=1).assign(n=g.size())
display(scan_tab.round(2))
""")

md(r"""
### Figure 4. P(resistant) for single-base *pncA* substitutions absent from training

`seen` means the exact base change occurs in a training isolate; shown are the unseen ones. Ticks: validation-fold
thresholds.
""")

code(r"""
KINDS = ["synonymous", "missense", "nonsense"]
U = S[~S.seen]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
rng = np.random.default_rng(0)
for ax, kd in zip(axes, KINDS):
    gk = U[U.kind == kd]
    for i, m in enumerate(SHOW):
        v = gk[pc[m]].values
        ax.boxplot(v, positions=[i], widths=0.55, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="white", edgecolor=COL[m], lw=1.4), medianprops=dict(color=COL[m], lw=2),
                   whiskerprops=dict(color=COL[m]), capprops=dict(color=COL[m]))
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=6 if len(v) > 100 else 16, color=COL[m],
                   alpha=0.35 if len(v) > 100 else 0.8, lw=0, zorder=3)
        ax.plot([i - 0.33, i + 0.33], [TH[m]] * 2, color=ps.NEUTRAL["ink"], lw=1.6, zorder=4)
    ax.set_xticks(range(len(SHOW)), [MODSETS[m] for m in SHOW], rotation=20, ha="right")
    ax.set_title(f"{kd}  ·  n = {len(gk)}", fontsize=11)
    ax.set_ylim(0, 1)
    tidy(ax)
axes[0].set_ylabel("P(resistant)")
fig.legend(handles=[plt.Line2D([], [], color=ps.NEUTRAL["ink"], lw=1.6, label="operating threshold (fit on validation fold)")],
           loc="lower center", fontsize=9.5)
fig.suptitle("LocusFusion v3, pyrazinamide (pncA only): P(resistant) for single-base pncA substitutions absent "
             "from training, H37Rv background", fontsize=12)
fig.tight_layout(rect=(0, 0.06, 1, 0.95))
fig.savefig(OUT / "fig4_pza_scan_by_type.png", dpi=200)
plt.show()
""")

md(r"""
### Figure 5. Synonymous substitutions along *pncA*

Each synonymous substitution's P(resistant) by codon, DNA only against + protein and + biophysical. Filled
markers: the exact base change occurs in training.
""")

code(r"""
syn = S[S.kind == "synonymous"]
fig, axes = plt.subplots(3, 1, figsize=(12.5, 7.2), sharex=True, sharey=True)
for ax, m in zip(axes, ["dna", "dna_protein", "dna_biophysical"]):
    for seen, fc in [(False, "white"), (True, COL[m])]:
        q = syn[syn.seen == seen]
        ax.scatter(q.codon, q[pc[m]], s=22, facecolor=fc, edgecolor=COL[m], lw=0.9, zorder=3,
                   label="in training" if seen else "not in training")
    ax.axhline(TH[m], color=ps.NEUTRAL["ink"], lw=1.2)
    ax.set_ylabel(f"{MODSETS[m]}\nP(resistant)", fontsize=10)
    ax.set_ylim(0, 1)
    tidy(ax)
fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=ps.NEUTRAL["ink2"],
                               label="base change not in training"),
                    plt.Line2D([], [], marker="o", ls="", color=ps.NEUTRAL["ink2"], label="base change in training"),
                    plt.Line2D([], [], color=ps.NEUTRAL["ink"], lw=1.2, label="operating threshold (fit on validation fold)")],
           loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, fontsize=9.5)
axes[-1].set_xlabel("PncA codon")
axes[-1].set_xlim(0, 187)
fig.suptitle("LocusFusion v3, pyrazinamide (pncA only): P(resistant) for each synonymous pncA substitution, "
             "H37Rv background", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT / "fig5_pza_synonymous_along_gene.png", dpi=200)
plt.show()

print({MODSETS[m]: round(syn[[pc[m], "codon"]].corr().iloc[0, 1], 2) for m in SHOW}, "(Pearson r, P(R) vs codon)")
display(S[(S.codon == 25) & (S.kind == "synonymous")]
        [["change", "ref_codon", "alt_codon", "train_carriers", "train_R"] + list(pc.values())]
        .rename(columns={c: MODSETS[m] for m, c in pc.items()}).round(2).reset_index(drop=True))
""")

md(r"""
- **No model learned "synonymous → harmless".** Unseen synonymous substitutions average P(R) 0.61 (DNA only),
  0.54 (+ protein), 0.61 (+ biophysical), 0.52 (all modalities); over half are called resistant by every model.
- **Stop codons generalise through the residue token.** Unseen nonsense substitutions: + protein 0.96, all called
  resistant; DNA only 0.77, 81%.
- **Position matters more than the residue.** Synonymous substitutions at codons ~100–135 score near 1.0 in all
  three models (Figure 5), so the residue token does not override a position the model reads as resistance-prone.
- **Codon 25:** DNA only gives the three synonymous codons 0.56 (GCA), 0.97 (GCG) and 0.10 (GCT); only GCG has a
  training carrier. The per-isolate difference between + protein and + biophysical for SAMEA1118274 is mostly this
  per-base behaviour, not a general difference between the two modalities.

## Caveats

- One checkpoint per modality set (seed 0); seed-to-seed spread is not measured for these cells, and two models
  with the same token information can differ from initialisation alone.
- Small n for the §2 examples: 6 unseen synonymous and 1 unseen nonsense single-variant test isolates.
- Per-isolate effects run both ways; the paper's modality gains are AUC (ranking), not calls at a threshold.
- Thresholds are refit on the validation fold; the result JSONs' thresholds were fit on the test set. Rescored on
  CPU, + biophysical's test-fit threshold is 0.33, not the stored 0.34 (isolates on the 0.01 grid boundary).
- Residue-level "seen in training" (§2) and base-level (§3) differ: a highly divergent sequence can carry a base
  change without translating to the same residue change (A25A).
- The scan uses an H37Rv background, so each scanned isolate has exactly one change.
- The protein and biophysical tokens skip insertions relative to H37Rv (reference-frame translation), so an
  insertion frameshift reaches the model only through the DNA token.
- §1 counts "any change in the drug's own genes" over the whole locus record (CDS and flanks); it does not ask
  whether that change is causal.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "analysis" / "modality_flips.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}")
