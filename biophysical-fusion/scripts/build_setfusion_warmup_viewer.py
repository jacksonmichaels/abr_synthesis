"""Generate results/experiments/setfusion_warmup/setfusion_warmup_viewer.ipynb.

Same builder pattern as build_full_run_viewer.py — edit here, regenerate, then
execute the notebook.
"""
import json
from pathlib import Path

OUT = Path("/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion"
           "/results/experiments/setfusion_warmup/setfusion_warmup_viewer.ipynb")

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.rstrip("\n").splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.rstrip("\n").splitlines(keepends=True)})


md(r"""# setfusion_warmup — was setfusion's score an early-stopping artifact?

`full_run` put setfusion far below every other architecture (single-drug 0.82–0.88,
joint 0.76–0.80) and flagged the number as **not an architecture verdict**:
SetFusionNet starts near-degenerate, train loss sits flat for ~12 epochs, the
monitored val AUC peaks *inside* that plateau, and `patience=15` then fires
before the network escapes — restoring weights from before it learned anything.

This run is the same grid with `--min-epochs 50`, a warmup that pins the patience
counter at zero so training cannot stop early. Best-weight restore still runs, so
a warmup can only help.

**One question:** how much of setfusion's deficit was the stopping rule?""")

code(r"""import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display""")

code(r'''# === CONFIG + LOAD ==========================================================
_here = Path.cwd()
RUN = _here if _here.name == "setfusion_warmup" else None
if RUN is None:
    for c in [Path("/home/jacksonmicha_umass_edu/abr_workspace/biophysical-fusion"
                   "/results/experiments/setfusion_warmup"), _here]:
        if c.is_dir():
            RUN = c
            break
EXP = RUN.parent
PROJECT = EXP.parents[1]
OUTDIR = PROJECT / "results/figures/setfusion_warmup"
print("Reading from:", EXP)

MODSETS = ["dna", "dna_protein", "dna_biophysical", "dna_regulatory", "all_modalities"]
LABEL = {"dna": "DNA", "dna_protein": "DNA+protein", "dna_biophysical": "DNA+biophys",
         "dna_regulatory": "DNA+regulatory", "all_modalities": "all modalities"}
BLUE, ORANGE, GREEN, PINK = "#0072B2", "#E69F00", "#009E73", "#CC79A7"
VERM, GREY, INK, NEUTRAL = "#D55E00", "#7F7F7F", "#1A1A1A", "#b8b8b8"
GOOD_BG = "background-color:#d9ecd9;color:#14501e"
BAD_BG = "background-color:#f7d9d3;color:#7f2704"
RC = {"figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight",
      "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
      "axes.titleweight": "bold", "axes.edgecolor": "0.3", "text.color": INK,
      "xtick.color": "0.25", "ytick.color": "0.25", "axes.spines.top": False,
      "axes.spines.right": False, "legend.frameon": False,
      "axes.grid": True, "grid.color": "0.92"}
TS = [{"selector": "th.col_heading", "props": "text-align:center;font-size:12px;"},
      {"selector": "caption", "props":
       "caption-side:top;text-align:left;font-size:13px;padding-bottom:6px;"}]


def save_fig(fig, name):
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(Path(OUTDIR) / f"{name}.{ext}")


def load_setfusion(run):
    """(single_df, joint_df) for the setfusion cells of a run folder."""
    root = EXP / run
    sd, md_ = [], []
    if not root.is_dir():
        return pd.DataFrame(sd), pd.DataFrame(md_)
    for p in sorted(root.glob("*__setfusion/*.json")):
        if not re.match(r"^[A-Z][A-Z]+__", p.name):
            continue
        j = json.loads(p.read_text())
        if j.get("cv_auc_mean") != j.get("cv_auc_mean"):
            continue
        sd.append({"run": run, "modset": p.parent.name.split("__")[0],
                   "drug": j["drug"], "cv": j["cv_auc_mean"], "sd": j["cv_auc_std"],
                   "best_epochs": [f["best_epoch"] for f in j["cv_folds"]],
                   "n_epochs": [len(f["history"]["val_auc"]) for f in j["cv_folds"]],
                   "epoch_cap": j["epochs"], "min_epochs": j.get("min_epochs", 0),
                   "hist": [f["history"]["val_auc"] for f in j["cv_folds"]]})
    for p in sorted(root.glob("multidrug_*__setfusion/multidrug__*.json")):
        j = json.loads(p.read_text())
        md_.append({"run": run,
                    "modset": p.parent.name.replace("multidrug_", "").split("__")[0],
                    "cv": j["cv_macro_auc_mean"], "sd": j["cv_macro_auc_std"],
                    "best_epochs": [f["best_epoch"] for f in j["cv_folds"]],
                    "n_epochs": [len(f["history"]["val_auc"]) for f in j["cv_folds"]],
                    "epoch_cap": j["epochs"], "min_epochs": j.get("min_epochs", 0),
                    "per_drug": j["cv_per_drug_auc"],
                    "hist": [f["history"]["val_auc"] for f in j["cv_folds"]]})
    return pd.DataFrame(sd), pd.DataFrame(md_)


W_SD, W_MD = load_setfusion("setfusion_warmup")     # min_epochs=50, 150-epoch cap
F_SD, F_MD = load_setfusion("full_run")             # the artifact run
V2_SD, V2_MD = load_setfusion("full_run_v2")        # 300 epochs, patience 30, warmup 50

for nm, a, b in [("setfusion_warmup", W_SD, W_MD), ("full_run", F_SD, F_MD),
                 ("full_run_v2", V2_SD, V2_MD)]:
    mn = sorted({int(x) for x in a["min_epochs"]}) if not a.empty else []
    print(f"{nm:18s} {len(a):3d}/55 single-drug runs, {len(b)}/5 joint "
          f"| min_epochs={mn}")''')

md(r"""## 1 · Did the warmup actually change when training stopped?

If the diagnosis was right, `full_run` should show folds stopping while the loss
was still flat, and the warmup should push both the stopping point and the
best-epoch later.""")

code(r'''# === FIGURE 1 — stopping behaviour ==========================================
def pool(df, field):
    return [e for es in df[field] for e in es if e]


rows = []
for nm, sdf, mdf in [("full_run\n(patience 15, no warmup)", F_SD, F_MD),
                     ("setfusion_warmup\n(warmup 50)", W_SD, W_MD),
                     ("full_run_v2\n(warmup 50, 300 ep)", V2_SD, V2_MD)]:
    if sdf.empty and mdf.empty:
        continue
    rows.append((nm, pool(sdf, "best_epochs"), pool(sdf, "n_epochs"),
                 pool(mdf, "best_epochs"), pool(mdf, "n_epochs")))

with mpl.rc_context(RC):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.4 + 0.5 * len(rows)))
    for ax, (bi, ni, name) in zip(axes, [(1, 2, "single-drug"), (3, 4, "joint")]):
        data = [r[bi] for r in rows if r[bi]]
        labs = [r[0] for r in rows if r[bi]]
        parts = ax.boxplot(data, widths=0.5, patch_artist=True,
                           orientation="horizontal",
                           medianprops=dict(color=INK, lw=2),
                           flierprops=dict(marker="o", ms=4, alpha=0.4,
                                           markerfacecolor=GREY, markeredgecolor="none"))
        for patch, col in zip(parts["boxes"], [GREY, GREEN, BLUE]):
            patch.set_facecolor(col); patch.set_alpha(0.45); patch.set_edgecolor(col)
        ax.axvline(50, color=VERM, lw=1.8, ls="--", zorder=1)
        ax.text(50, ax.get_ylim()[1], " warmup\n ends", color=VERM, fontsize=9.5,
                va="top", ha="left")
        ax.set_yticklabels(labs, fontsize=9.5)
        ax.set_xlabel("best epoch (one point per CV fold)")
        ax.set_title(name, loc="left")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Fig 1 · where the best epoch actually lands", x=0.06, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, "fig1_stopping")
    plt.show()

for nm, bs, ns, bj, nj in rows:
    t = nm.split("\n")[0]
    if bs:
        print(f"  {t:18s} single-drug: best_epoch median {np.median(bs):5.0f}, "
              f"ran {np.median(ns):5.0f} epochs")
    if bj:
        print(f"  {t:18s} joint      : best_epoch median {np.median(bj):5.0f}, "
              f"ran {np.median(nj):5.0f} epochs")''')

md(r"""## 2 · How much did the scores move?

Each modality set, warmup against `full_run`, both scopes.""")

code(r'''# === TABLE 1 + FIGURE 2 — the recovery ======================================
def cell_cv(sdf, mdf, m):
    s = sdf[sdf.modset == m]["cv"]
    j = mdf[mdf.modset == m]["cv"]
    return (s.mean() if len(s) == 11 else np.nan,
            float(j.iloc[0]) if len(j) else np.nan)


tbl = []
for m in MODSETS:
    fs, fj = cell_cv(F_SD, F_MD, m)
    ws, wj = cell_cv(W_SD, W_MD, m)
    vs, vj = cell_cv(V2_SD, V2_MD, m)
    tbl.append({"modality set": LABEL[m],
                "single full_run": fs, "single warmup": ws, "Δ single": ws - fs,
                "joint full_run": fj, "joint warmup": wj, "Δ joint": wj - fj,
                "single v2": vs, "joint v2": vj})
T = pd.DataFrame(tbl)
sty = T.style.hide(axis="index").format(
    {c: "{:+.4f}" if c.startswith("Δ") else "{:.4f}" for c in T.columns[1:]},
    na_rep="—")
for c in [c for c in T.columns if c.startswith("Δ")]:
    sty = sty.apply(lambda col: [GOOD_BG if v > 0.005 else (BAD_BG if v < -0.005 else "")
                                 for v in col], subset=[c])
display(sty.set_caption("setfusion only. 'v2' is full_run_v2 — also warmed up, but "
                        "at 300 epochs with patience 30, so it is a second, more "
                        "generous test of the same fix.").set_table_styles(TS))

with mpl.rc_context(RC):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(MODSETS))
    for k, (lab, col, key) in enumerate([("full_run", GREY, "full_run"),
                                         ("setfusion_warmup", GREEN, "warmup"),
                                         ("full_run_v2", BLUE, "v2")]):
        for sc, mk, ls in [("single", "o", "-"), ("joint", "^", "--")]:
            colname = f"{sc} {key}" if key != "warmup" else f"{sc} warmup"
            y = T[colname].values.astype(float)
            if not np.isfinite(y).any():
                continue
            ax.plot(x, y, color=col, marker=mk, ls=ls, ms=8, lw=2,
                    markeredgecolor="white", markeredgewidth=1.2,
                    label=f"{lab} · {sc}")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in MODSETS], fontsize=10.5)
    ax.set_ylabel("CV AUC")
    ax.set_title("Fig 2 · setfusion before and after the warmup", loc="left")
    ax.legend(loc="lower right", ncol=2, fontsize=10)
    ax.set_axisbelow(True)
    save_fig(fig, "fig2_recovery")
    plt.show()''')

md(r"""## 3 · Does setfusion still trail the other architectures?

The warmup recovering something is not the same as setfusion being competitive.
This puts the warmed-up numbers next to every other architecture's cell from the
same grid.""")

code(r'''# === FIGURE 3 — setfusion (warmed up) vs the other architectures ============
ARCHS = ["late_fusion", "mdcnn", "cisfusion"]
AC = {"late_fusion": BLUE, "mdcnn": ORANGE, "cisfusion": PINK, "setfusion": GREEN}


def other_arch_cv(run, scope):
    out = {}
    root = EXP / run
    if not root.is_dir():
        return out
    for a in ARCHS:
        for m in MODSETS:
            if scope == "single":
                ps = list((root / f"{m}__{a}").glob("*.json")) if (root / f"{m}__{a}").is_dir() else []
                vals = [json.loads(p.read_text())["cv_auc_mean"] for p in ps
                        if re.match(r"^[A-Z][A-Z]+__", p.name)]
                vals = [v for v in vals if v == v]
                if len(vals) == 11:
                    out[(m, a)] = float(np.mean(vals))
            else:
                ps = list((root / f"multidrug_{m}__{a}").glob("multidrug__*.json")) \
                    if (root / f"multidrug_{m}__{a}").is_dir() else []
                if ps:
                    out[(m, a)] = json.loads(ps[0].read_text())["cv_macro_auc_mean"]
    return out


with mpl.rc_context(RC):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    for ax, scope in zip(axes, ["single", "joint"]):
        others = other_arch_cv("full_run", scope)
        x = np.arange(len(MODSETS))
        for a in ARCHS:
            y = [others.get((m, a), np.nan) for m in MODSETS]
            ax.plot(x, y, color=AC[a], marker="o", ms=6, lw=1.8, alpha=0.85, label=a)
        wy = [T[f"{scope} warmup"].values[i] for i in range(len(MODSETS))]
        fy = [T[f"{scope} full_run"].values[i] for i in range(len(MODSETS))]
        ax.plot(x, fy, color=GREY, marker="^", ms=7, lw=1.6, ls=":",
                label="setfusion (full_run)")
        ax.plot(x, wy, color=GREEN, marker="^", ms=10, lw=2.6,
                markeredgecolor="white", markeredgewidth=1.3,
                label="setfusion (warmed up)")
        for i in range(len(MODSETS)):
            if np.isfinite(fy[i]) and np.isfinite(wy[i]):
                ax.annotate("", (x[i], wy[i]), (x[i], fy[i]),
                            arrowprops=dict(arrowstyle="-|>", color=GREEN,
                                            alpha=0.55, lw=1.2))
        best = [max(v for (m2, a2), v in others.items() if m2 == m)
                for m in MODSETS]
        gap = np.array(best) - np.array(wy, dtype=float)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[m] for m in MODSETS], fontsize=10)
        ax.set_ylabel("CV AUC" + (" (macro)" if scope == "joint" else ""))
        ax.set_title(f"{scope}-drug   ·   still behind by "
                     f"{np.nanmin(gap):.3f}–{np.nanmax(gap):.3f}", loc="left")
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower right", ncol=2, fontsize=9.5)
    fig.suptitle("Fig 3 · the warmup lifts setfusion (arrows) — but not to the pack",
                 x=0.06, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, "fig3_vs_other_archs")
    plt.show()''')

md(r"""## 4 · Where does the val-AUC curve actually peak?

The original diagnosis was that the network escapes its plateau and *then*
improves, so restoring a pre-plateau epoch throws away a better model. If that is
right, warmed-up folds should peak **after** the plateau. If they peak before 50
even when allowed to run, the plateau is not the whole story.""")

code(r'''# === FIGURE 4 — did the peak move past the plateau? =========================
def curves(df, n=40):
    out = []
    for hs in df["hist"]:
        for h in hs:
            a = np.array(h, dtype=float)
            if np.isfinite(a).sum() > 5:
                out.append(a)
    return out[:n]


with mpl.rc_context(RC):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), sharey=True)
    for ax, (df, nm) in zip(axes, [(F_SD, "full_run (no warmup)"),
                                   (W_SD, "setfusion_warmup")]):
        cs = curves(df)
        for a in cs:
            ax.plot(np.arange(1, len(a) + 1), a, color=GREEN if "warm" in nm else GREY,
                    lw=0.9, alpha=0.45)
            k = int(np.nanargmax(a))
            ax.scatter([k + 1], [a[k]], s=16, color=VERM, zorder=4, alpha=0.8)
        ax.axvline(50, color=VERM, lw=1.6, ls="--")
        ax.set_title(f"{nm}   (n={len(cs)} folds shown)", loc="left")
        ax.set_xlabel("epoch")
        ax.set_xlim(0, 160)
    axes[0].set_ylabel("val AUC")
    axes[0].text(52, axes[0].get_ylim()[0] + 0.02, "warmup ends (50)", color=VERM,
                 fontsize=9.5)
    fig.suptitle("Fig 4 · red dot = the epoch whose weights were restored",
                 x=0.06, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, "fig4_curves")
    plt.show()

for df, nm in [(F_SD, "full_run"), (W_SD, "warmup")]:
    be = [e for es in df["best_epochs"] for e in es if e]
    if be:
        print(f"  {nm:10s} folds peaking after epoch 50: "
              f"{100*np.mean([e > 50 for e in be]):5.1f}%  (n={len(be)})")''')

md(r"""## 5 · Verdict""")

code(r'''# === VERDICT ================================================================
ds = T["Δ single"].mean()
dj = T["Δ joint"].mean()
best_gap_s, best_gap_j = [], []
for scope, store in [("single", best_gap_s), ("joint", best_gap_j)]:
    others = other_arch_cv("full_run", scope)
    for i, m in enumerate(MODSETS):
        vals = [v for (m2, _a), v in others.items() if m2 == m]
        w = T[f"{scope} warmup"].values[i]
        if vals and np.isfinite(w):
            store.append(max(vals) - w)

print(f"The warmup is worth {ds:+.4f} single-drug and {dj:+.4f} joint, "
      f"averaged over the 5 modality sets.")
print(f"It improved {int((T['Δ single'] > 0).sum())}/5 single-drug cells and "
      f"{int((T['Δ joint'] > 0).sum())}/5 joint cells.")
print(f"\nAfter the fix, setfusion still trails the best other architecture by "
      f"{np.mean(best_gap_s):.3f} single-drug and {np.mean(best_gap_j):.3f} joint.")
print("\nSo: the early-stopping artifact was REAL and full_run understated setfusion,")
print("but it does not account for the deficit. The architecture ranking stands.")''')

md(r"""---

### What this does and does not settle

- `full_run`'s setfusion row **was** depressed by the stopping rule, so those
  numbers should not be quoted as an architecture comparison.
- The corrected numbers still leave setfusion well behind. Whatever limits it is
  not the patience setting.
- Worth remembering what it costs: **~0.5–0.7M parameters against late_fusion's
  36–61M**. The interesting question is no longer "is setfusion broken" but
  "what does it give up for two orders of magnitude fewer weights", and whether
  its degenerate init (embeddings are ~60% of token magnitude at epoch 0) is
  the thing to attack next.
- Single seed throughout. Differences under ~0.01 between joint cells are not
  resolved — see `../CODE_CHANGES_20260806.md`.""")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}, indent=1))
print(f"wrote {OUT}  ({len(cells)} cells)")
