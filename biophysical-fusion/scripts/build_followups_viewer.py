"""Generate results/experiments/followups_viewer.ipynb.

Covers the three runs submitted 2026-08-06: joint_convergence, joint_capacity
and full_run_v2. Same builder pattern as build_full_run_viewer.py — edit here,
regenerate, then execute the notebook.

Palette note: the categorical hues (#0072B2 #E69F00 #009E73 #CC79A7) were
validated for a light surface — lightness band, chroma floor, CVD dE and
normal-vision dE all pass on the all-pairs list EXCEPT green/pink, which land at
CVD dE 7.6 (floor 6, target 8). That pair is therefore always carried with
secondary encoding: distinct markers plus direct end-labels, never colour alone.
Orange and pink fall below 3:1 contrast on white, so every figure is paired with
a table. Diverging deltas use blue -> neutral grey -> vermillion (dE 31.2).
"""
import json
from pathlib import Path

OUT = Path("/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion"
           "/results/experiments/followups_viewer.ipynb")

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.rstrip("\n").splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.rstrip("\n").splitlines(keepends=True)})


# ------------------------------------------------------------------ intro
md(r"""# Follow-up runs — 2026-08-06

Three runs, three questions. Each section states its purpose, then shows the
evidence.

| run | purpose | control |
|---|---|---|
| **joint_convergence** | Were the joint models ever trained to completion? 6 arms, one training change each. | `a0_control` (identical to `full_run`) |
| **joint_capacity** | Is the joint dense head capacity-bound for 11 drugs? 4 arms × 2 archs. | the matching `full_run` cell |
| **full_run_v2** | A replacement baseline whose **weights exist** — `full_run` checkpointed nothing. | `full_run`, and the published SD-CNN / MD-CNN |

Runs are still landing; every cell degrades to a note when its data is absent.

**Read every difference against the noise floor.** Joint fold SD in `full_run`
was 0.003–0.030 on one seed, against a joint-vs-MD-CNN gap of −0.0116. A margin
under ~0.01 between single-seed joint cells is not resolved.""")

# ------------------------------------------------------------------ config
code(r"""import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from IPython.display import display""")

code(r'''# === CONFIG =================================================================
_here = Path.cwd()
EXP = _here if _here.name == "experiments" else None
if EXP is None:
    for c in [Path("/home/jacksonmicha_umass_edu/abr_workspace/biophysical-fusion"
                   "/results/experiments"), _here]:
        if (c / "full_run").is_dir():
            EXP = c
            break
PROJECT = EXP.parents[1]
WEIGHTS = Path("/project/pi_mfiterau_umass_edu/abr_model_weights")
OUTDIR = PROJECT / "results/figures/followups"      # None to display without saving
print("Reading from:", EXP)

ARCH_ORDER = ["late_fusion", "mdcnn", "setfusion", "cisfusion"]
MODSET_ORDER = ["dna", "dna_protein", "dna_biophysical", "dna_regulatory",
                "all_modalities"]
MODSET_LABEL = {"dna": "DNA", "dna_protein": "DNA+protein",
                "dna_biophysical": "DNA+biophys", "dna_regulatory": "DNA+regulatory",
                "all_modalities": "all modalities"}

# --- palette (validated; see the builder docstring) --------------------------
# Categorical: fixed order, never cycled. green/pink are a CVD floor-only pair,
# so archs ALWAYS also carry a distinct marker + a direct end-label.
ARCH_COLOR  = {"late_fusion": "#0072B2", "mdcnn": "#E69F00",
               "setfusion": "#009E73", "cisfusion": "#CC79A7"}
ARCH_MARKER = {"late_fusion": "o", "mdcnn": "s", "setfusion": "^", "cisfusion": "D"}
BLUE, ORANGE, VERM, GREY, INK = "#0072B2", "#E69F00", "#D55E00", "#7F7F7F", "#1A1A1A"
NEUTRAL = "#b8b8b8"                       # diverging midpoint
GOOD_BG = "background-color:#d9ecd9;color:#14501e"
BAD_BG  = "background-color:#f7d9d3;color:#7f2704"

RC = {
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.titleweight": "bold", "axes.edgecolor": "0.3",
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "0.25", "ytick.color": "0.25",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.8,
}
TABLE_STYLES = [
    {"selector": "th.col_heading", "props":
     "text-align:center; white-space:pre-line; font-size:12px;"},
    {"selector": "caption", "props":
     "caption-side:top; text-align:left; font-size:13px; padding-bottom:6px;"},
]


def save_fig(fig, name):
    if OUTDIR is None:
        return
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(OUTDIR) / f"{name}.png")


def colour_delta(col, tol=0.005):
    return [GOOD_BG if v > tol else (BAD_BG if v < -tol else "") for v in col]


def diverging(v, vmax):
    """Blue = better, vermillion = worse, neutral grey at zero."""
    if not np.isfinite(v) or vmax == 0:
        return NEUTRAL
    t = min(abs(v) / vmax, 1.0)
    base = np.array(mpl.colors.to_rgb(BLUE if v >= 0 else VERM))
    return tuple(np.array(mpl.colors.to_rgb(NEUTRAL)) * (1 - t) + base * t)


def note(msg):
    print(f"[not yet] {msg}")''')

# ------------------------------------------------------------------ loaders
code(r'''# === LOAD ===================================================================
# joint_* folders are "{arm}_multidrug_{modset}__{arch}".
# full_run_v2 folders are "{modset}__{arch}" / "multidrug_{modset}__{arch}".
def _read(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


# Empty frames must still carry their columns: a run with nothing finished is
# the normal state while jobs are landing, and `pd.DataFrame([])` has no columns
# at all, so every later `df.best_epochs` / groupby would raise instead of
# yielding nothing.
ARM_COLS = ["arm", "modset", "arch", "cv", "sd", "best_epochs", "epoch_cap",
            "patience", "hours", "params", "per_drug", "result"]
SD_COLS = ["modset", "arch", "drug", "cv", "sd", "best_epochs", "params"]
MD_COLS = ["modset", "arch", "cv", "sd", "best_epochs", "per_drug", "params"]


def load_arms(group):
    """[{arm, modset, arch, cv, sd, best_epochs, hours, params, result}] for a
    joint_* run folder. Missing/unfinished cells simply do not appear."""
    root = EXP / group
    rows = []
    if not root.is_dir():
        return pd.DataFrame(rows, columns=ARM_COLS)
    for d in sorted(root.iterdir()):
        m = re.fullmatch(r"(.+?)_multidrug_(.+?)__(.+)", d.name) if d.is_dir() else None
        if not m:
            continue
        js = sorted(d.glob("multidrug__*.json"))
        if not js:
            continue
        r = _read(js[0])
        if not r:
            continue
        rows.append({
            "arm": m.group(1), "modset": m.group(2), "arch": m.group(3),
            "cv": r["cv_macro_auc_mean"], "sd": r["cv_macro_auc_std"],
            "best_epochs": [f["best_epoch"] for f in r["cv_folds"]],
            "epoch_cap": r["epochs"], "patience": r["patience"],
            "hours": r["seconds"] / 3600, "params": r["n_params"],
            "per_drug": r["cv_per_drug_auc"], "result": r,
        })
    return pd.DataFrame(rows, columns=ARM_COLS)


def load_grid(run):
    """(single_drug_df, joint_df) for a full_run-style folder."""
    root = EXP / run
    sd, md_ = [], []
    if not root.is_dir():
        return pd.DataFrame(sd, columns=SD_COLS), pd.DataFrame(md_, columns=MD_COLS)
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = re.fullmatch(r"(multidrug_)?(.+?)__(.+)", d.name)
        if not m or m.group(2) not in MODSET_ORDER or m.group(3) not in ARCH_ORDER:
            continue
        joint, modset, arch = bool(m.group(1)), m.group(2), m.group(3)
        if joint:
            js = sorted(d.glob("multidrug__*.json"))
            r = _read(js[0]) if js else None
            if r:
                md_.append({"modset": modset, "arch": arch,
                            "cv": r["cv_macro_auc_mean"], "sd": r["cv_macro_auc_std"],
                            "best_epochs": [f["best_epoch"] for f in r["cv_folds"]],
                            "per_drug": r["cv_per_drug_auc"], "params": r["n_params"]})
        else:
            for p in sorted(d.glob("*.json")):
                if not re.match(r"^[A-Z][A-Z]+__", p.name):
                    continue
                r = _read(p)
                if r and r.get("cv_auc_mean") == r.get("cv_auc_mean"):
                    sd.append({"modset": modset, "arch": arch, "drug": r["drug"],
                               "cv": r["cv_auc_mean"], "sd": r["cv_auc_std"],
                               "best_epochs": [f["best_epoch"] for f in r["cv_folds"]],
                               "params": r["n_params"]})
    return (pd.DataFrame(sd, columns=SD_COLS),
            pd.DataFrame(md_, columns=MD_COLS))


CONV = load_arms("joint_convergence")
CAP = load_arms("joint_capacity")
V2_SD, V2_MD = load_grid("full_run_v2")
V1_SD, V1_MD = load_grid("full_run")

# published baselines (carried over unchanged from full_run_viewer)
SD_BASE_CV = pd.Series({
    "AMIKACIN": 0.859, "CAPREOMYCIN": 0.847, "ETHAMBUTOL": 0.926,
    "ETHIONAMIDE": 0.622, "ISONIAZID": 0.912, "KANAMYCIN": 0.867,
    "LEVOFLOXACIN": 0.850, "MOXIFLOXACIN": 0.819, "PYRAZINAMIDE": 0.913,
    "RIFAMPICIN": 0.972, "STREPTOMYCIN": 0.913})
MD_BASE_CV = pd.Series({
    "AMIKACIN": 0.9176, "CAPREOMYCIN": 0.8599, "ETHAMBUTOL": 0.9253,
    "ETHIONAMIDE": 0.9161, "ISONIAZID": 0.9708, "KANAMYCIN": 0.8925,
    "LEVOFLOXACIN": 0.9450, "MOXIFLOXACIN": 0.9298, "PYRAZINAMIDE": 0.9113,
    "RIFAMPICIN": 0.9769, "STREPTOMYCIN": 0.9276})
MDCNN_CV_SUSPECT = {"ETHIONAMIDE"}     # Table-14 row contradicts the authors' own auc.csv
ALL_DRUGS = sorted(SD_BASE_CV.index)

print(f"joint_convergence {len(CONV):>3} cells | joint_capacity {len(CAP):>3} cells")
print(f"full_run_v2       {len(V2_SD):>3} single-drug runs, {len(V2_MD)} joint")
print(f"full_run (v1)     {len(V1_SD):>3} single-drug runs, {len(V1_MD)} joint")''')

# ------------------------------------------------------------------ coverage
md(r"""## 0 · Coverage

What has landed. Everything below reads only finished cells.""")

code(r'''# === COVERAGE ===============================================================
rows = [
    ("joint_convergence", len(CONV), 6),
    ("joint_capacity", len(CAP), 8),
    ("full_run_v2 single-drug", len(V2_SD), 220),
    ("full_run_v2 joint", len(V2_MD), 20),
]
cov = pd.DataFrame(rows, columns=["run", "done", "expected"])
cov["%"] = (100 * cov["done"] / cov["expected"]).round(0).astype(int)
n_w = sum(1 for _ in WEIGHTS.rglob("config.json")) if WEIGHTS.is_dir() else 0
display(cov.style.hide(axis="index").set_caption(
    f"landed so far — {n_w} model checkpoints on disk").set_table_styles(TABLE_STYLES))
print("\nRerun this notebook as jobs land; partial cells are skipped, not averaged.")''')

# ------------------------------------------------------------------ §1
md(r"""---

## 1 · joint_convergence — were the joint models ever trained to completion?

`full_run` reported the joint scope at −0.0116 vs MD-CNN. But 40% of joint
`late_fusion` folds stopped at the 150-epoch cap with `best_epoch` 120–148 and
still improving, so those numbers came from truncated models while the
single-drug numbers came from converged ones.

Six arms on `multidrug dna__late_fusion`, each changing **one** thing against
`a0_control`. `a1` asks whether it is an epoch-budget problem; `a2`/`a3` whether
it is a learning-rate problem wearing an epoch-cap costume (`exp(-9)` ≈ 1.2e-4 is
~10× below a normal Adam setting); `a4` whether patience fired early; `a5` how
much of the stop signal was LEVOFLOXACIN noise.""")

code(r'''# === TABLE 1 — the arms ======================================================
ARM_WHAT = {
    "a0_control": "none (full_run settings)", "a1_ep400": "--epochs 400",
    "a2_lr1e3": "--lr 1e-3", "a3_lr1e3_reg": "--lr 1e-3 +dropout .3 +wd 1e-4",
    "a4_patience30": "--patience 30", "a5_monitor500": "--monitor-min-n 500",
}
if CONV.empty:
    note("joint_convergence: no cell has finished yet.")
else:
    base = CONV[CONV.arm == "a0_control"]
    b_cv = float(base["cv"].iloc[0]) if len(base) else np.nan
    t = CONV.sort_values("arm").copy()
    t["change"] = t["arm"].map(ARM_WHAT)
    t["CV macro-AUC"] = [f"{c:.4f} ± {s:.4f}" for c, s in zip(t.cv, t.sd)]
    t["Δ vs a0"] = t["cv"] - b_cv
    t["best_epoch / fold"] = [", ".join(str(e) for e in es) for es in t.best_epochs]
    t["at cap?"] = [f"{sum(1 for e in es if e and e >= 0.93 * cap)}/{len(es)}"
                    for es, cap in zip(t.best_epochs, t.epoch_cap)]
    t["hours"] = t["hours"].round(1)
    show = t[["arm", "change", "CV macro-AUC", "Δ vs a0", "best_epoch / fold",
              "at cap?", "hours"]]
    display(show.style.hide(axis="index")
            .format({"Δ vs a0": "{:+.4f}"})
            .apply(colour_delta, subset=["Δ vs a0"])
            .set_caption("Arms vs a0_control. 'at cap?' = folds whose best epoch "
                         "is within 7% of the cap, i.e. still improving when "
                         "training stopped.")
            .set_table_styles(TABLE_STYLES))''')

code(r'''# === FIGURE 1a — Δ CV vs the control (diverging) =============================
if CONV.empty or "a0_control" not in set(CONV.arm):
    note("Fig 1a needs a0_control plus at least one other arm.")
else:
    d = CONV.set_index("arm")
    b = d.loc["a0_control", "cv"]
    arms = [a for a in ARM_WHAT if a in d.index and a != "a0_control"]
    vals = [d.loc[a, "cv"] - b for a in arms]
    errs = [d.loc[a, "sd"] for a in arms]
    vmax = max(abs(v) for v in vals) or 1e-9
    with mpl.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(arms) + 2.1))
        y = np.arange(len(arms))
        ax.barh(y, vals, height=0.6, color=[diverging(v, vmax) for v in vals],
                zorder=3)
        ax.errorbar(vals, y, xerr=errs, fmt="none", ecolor="0.35",
                    elinewidth=1.2, capsize=3, zorder=4)
        ax.axvline(0, color="0.2", lw=1.4, zorder=5)
        # the noise floor this run cannot resolve inside
        ax.axvspan(-0.01, 0.01, color="0.85", alpha=0.5, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{a}\n{ARM_WHAT[a]}" for a in arms], fontsize=10)
        for yi, v in zip(y, vals):
            ax.text(v + (0.0012 if v >= 0 else -0.0012), yi, f"{v:+.4f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=10)
        ax.set_xlim(-vmax * 1.55, vmax * 1.55)
        ax.set_xlabel("Δ CV macro-AUC vs a0_control")
        ax.set_title("Fig 1a · which training change moves the joint score",
                     loc="left")
        ax.text(0, 1.015, "grey band = ±0.01, the single-seed noise floor; "
                "bars are mean ± fold SD", transform=ax.transAxes, fontsize=10,
                color="0.35", va="bottom")
        ax.set_axisbelow(True)
        ax.grid(axis="y", visible=False)
        save_fig(fig, "fig1a_convergence_delta")
        plt.show()''')

code(r'''# === FIGURE 1b — did the epoch ceiling actually lift? ========================
if CONV.empty:
    note("Fig 1b needs joint_convergence results.")
else:
    d = CONV.sort_values("arm")
    with mpl.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(d) + 2.0))
        for i, (_, r) in enumerate(d.iterrows()):
            eps = [e for e in r.best_epochs if e]
            cap = r.epoch_cap
            ax.plot([0, cap], [i, i], color="0.88", lw=6, solid_capstyle="round",
                    zorder=1)
            ax.scatter(eps, [i] * len(eps), s=70, color=BLUE, zorder=3,
                       edgecolor="white", linewidth=1.4)
            ax.plot([cap, cap], [i - 0.3, i + 0.3], color=VERM, lw=2.5, zorder=4)
            ax.text(cap + 6, i, f"cap {cap}", va="center", fontsize=9, color=VERM)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(d.arm, fontsize=10)
        ax.set_xlabel("best epoch (one dot per CV fold)")
        ax.set_title("Fig 1b · a dot sitting on the cap means that fold never "
                     "converged", loc="left")
        ax.set_xlim(0, max(d.epoch_cap) * 1.14)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        save_fig(fig, "fig1b_best_epoch")
        plt.show()''')

code(r'''# === FIGURE 1c — validation curves, one panel per arm ========================
# Small multiples rather than six overlaid lines: 6 arms x 5 folds on one axis is
# unreadable, and it would need 6 categorical hues where only 4 validate.
if CONV.empty:
    note("Fig 1c needs joint_convergence results.")
else:
    d = CONV.sort_values("arm").reset_index(drop=True)
    ctrl = d[d.arm == "a0_control"]
    ctrl_curve = None
    if len(ctrl):
        hs = [f["history"]["val_auc"] for f in ctrl.iloc[0].result["cv_folds"]]
        L = max(len(h) for h in hs)
        padded = np.full((len(hs), L), np.nan)
        for i, h in enumerate(hs):
            padded[i, :len(h)] = h
        ctrl_curve = np.nanmean(padded, axis=0)
    ncol = min(3, len(d))                 # don't draw 3 columns for 1 finished arm
    nrow = int(np.ceil(len(d) / ncol))
    with mpl.rc_context(RC):
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.1 * nrow),
                                 sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, (_, r) in zip(axes, d.iterrows()):
            for f in r.result["cv_folds"]:
                h = np.array(f["history"]["val_auc"], dtype=float)
                ax.plot(np.arange(1, len(h) + 1), h, color=BLUE, lw=1.3,
                        alpha=0.75, zorder=3)
            if ctrl_curve is not None and r.arm != "a0_control":
                ax.plot(np.arange(1, len(ctrl_curve) + 1), ctrl_curve,
                        color=GREY, lw=2.0, ls="--", zorder=2)
            ax.set_title(f"{r.arm}\n{ARM_WHAT.get(r.arm, '')}", fontsize=10.5)
            ax.set_xlabel("epoch")
            ax.set_ylim(0.55, 0.95)
        axes[0].set_ylabel("val macro-AUC")
        for ax in axes[len(d):]:
            ax.axis("off")
        handles = [Line2D([], [], color=BLUE, lw=1.6, label="this arm, per fold"),
                   Line2D([], [], color=GREY, lw=2.0, ls="--",
                          label="a0_control (fold mean)")]
        fig.legend(handles=handles, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("Fig 1c · a curve still rising at its right edge did not "
                     "converge", x=0.09, ha="left", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
        save_fig(fig, "fig1c_val_curves")
        plt.show()''')

code(r'''# === ANSWER 1 ================================================================
if CONV.empty or "a0_control" not in set(CONV.arm):
    note("Answer 1 needs a0_control and at least one other arm.")
else:
    d = CONV.set_index("arm")
    b = d.loc["a0_control", "cv"]
    others = [a for a in d.index if a != "a0_control"]
    best = max(others, key=lambda a: d.loc[a, "cv"]) if others else None
    v1 = V1_MD[(V1_MD.modset == "dna") & (V1_MD.arch == "late_fusion")]
    print(f"a0_control reproduces full_run: {b:.4f}"
          + (f" vs {float(v1.cv.iloc[0]):.4f} recorded  "
             f"(Δ {b - float(v1.cv.iloc[0]):+.4f})" if len(v1) else ""))
    if best:
        gain = d.loc[best, "cv"] - b
        verdict = ("resolved" if abs(gain) > 0.01 else
                   "INSIDE the ±0.01 noise floor — not resolved")
        print(f"\nANSWER: best arm is {best} ({ARM_WHAT[best]}) at "
              f"{d.loc[best,'cv']:.4f}, {gain:+.4f} vs control — {verdict}.")
        capped = sum(1 for e in d.loc[best, "best_epochs"]
                     if e and e >= 0.93 * d.loc[best, "epoch_cap"])
        print(f"        {capped}/{len(d.loc[best,'best_epochs'])} of its folds "
              f"still sat at the cap, so "
              + ("the ceiling is still binding — give it more budget."
                 if capped else "the ceiling is gone."))''')

# ------------------------------------------------------------------ §2
md(r"""---

## 2 · joint_capacity — is the joint head capacity-bound?

`MultiDrugNet` reads all 11 drugs off **one 256-d vector via one shared linear
layer**; a single-drug model gets the same 256 units for one task. Adding a
modality widens `fc1`'s *input* without widening that vector, which is the
observed symptom (joint modality gains +0.005 against +0.023–0.031 single-drug).

Four arms × `late_fusion`/`cisfusion` on `dna_protein`, all at 150 epochs so the
matching `full_run` cell is the control and head capacity is the only difference.
`cisfusion` is the cleaner read: its folds converge near epoch 85, so the cap is
not binding for it.""")

code(r'''# === TABLE 2 + FIGURE 2 — capacity arms vs their full_run control ============
ARM_CAP = {"b1_hidden512": "hidden 512", "b2_perdrug64": "per-drug head (k=64)",
           "b3_reg": "dropout .3 + wd 1e-4",
           "b4_all": "all three"}
if CAP.empty:
    note("joint_capacity: no cell has finished yet.")
else:
    ctrl = {(r.modset, r.arch): r.cv for _, r in V1_MD.iterrows()}
    ctrl_p = {(r.modset, r.arch): r.params for _, r in V1_MD.iterrows()}
    t = CAP.copy()
    t["control"] = [ctrl.get((m, a), np.nan) for m, a in zip(t.modset, t.arch)]
    t["Δ vs full_run"] = t["cv"] - t["control"]
    t["params"] = t["params"] / 1e6
    t["Δ params"] = t["params"] - np.array(
        [ctrl_p.get((m, a), np.nan) / 1e6 for m, a in zip(t.modset, t.arch)])
    t["change"] = t["arm"].map(ARM_CAP)
    t["CV macro-AUC"] = [f"{c:.4f} ± {s:.4f}" for c, s in zip(t.cv, t.sd)]
    show = (t.sort_values(["arch", "arm"])
             [["arch", "arm", "change", "CV macro-AUC", "control",
               "Δ vs full_run", "params", "Δ params"]])
    display(show.style.hide(axis="index")
            .format({"control": "{:.4f}", "Δ vs full_run": "{:+.4f}",
                     "params": "{:.1f}M", "Δ params": "{:+.2f}M"})
            .apply(colour_delta, subset=["Δ vs full_run"])
            .set_caption("Capacity arms vs the matching full_run cell "
                         "(dna_protein, 150 epochs, same inputs).")
            .set_table_styles(TABLE_STYLES))

    arms = [a for a in ARM_CAP if a in set(t.arm)]
    archs = [a for a in ARCH_ORDER if a in set(t.arch)]
    with mpl.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9.8, 0.75 * len(arms) + 2.4))
        h = 0.8 / max(len(archs), 1)
        y = np.arange(len(arms))
        for k, arch in enumerate(archs):
            sub = t[t.arch == arch].set_index("arm")
            vals = [sub.loc[a, "Δ vs full_run"] if a in sub.index else np.nan
                    for a in arms]
            off = (k - (len(archs) - 1) / 2) * h
            ax.barh(y + off, vals, height=h * 0.86, color=ARCH_COLOR[arch],
                    label=arch, zorder=3)
            for yi, v in zip(y + off, vals):
                if np.isfinite(v):
                    ax.text(v + (0.0004 if v >= 0 else -0.0004), yi, f"{v:+.4f}",
                            va="center", ha="left" if v >= 0 else "right",
                            fontsize=9)
        ax.axvspan(-0.01, 0.01, color="0.85", alpha=0.5, zorder=1)
        ax.axvline(0, color="0.2", lw=1.4, zorder=5)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{a}\n{ARM_CAP[a]}" for a in arms], fontsize=10)
        ax.set_xlabel("Δ CV macro-AUC vs the full_run control")
        ax.set_title("Fig 2 · does more head capacity help the joint model?",
                     loc="left")
        ax.text(0, 1.02, "grey band = ±0.01 noise floor", transform=ax.transAxes,
                fontsize=10, color="0.35", va="bottom")
        ax.legend(loc="lower right", ncol=len(archs))
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        save_fig(fig, "fig2_capacity_delta")
        plt.show()

    best = t.loc[t["Δ vs full_run"].idxmax()] if t["Δ vs full_run"].notna().any() else None
    if best is not None:
        g = best["Δ vs full_run"]
        print(f"\nANSWER: best is {best.arm} on {best.arch} ({g:+.4f} vs control, "
              f"{best['Δ params']:+.2f}M params) — "
              + ("a real move." if abs(g) > 0.01 else
                 "inside the ±0.01 noise floor, so capacity is NOT demonstrated "
                 "to be the constraint at this seed count."))''')

# ------------------------------------------------------------------ §3
md(r"""---

## 3 · full_run_v2 — the replacement baseline

`full_run` checkpointed nothing, so every model behind its numbers is gone. This
re-runs the same 4×5 grid on the **same inputs**, changing only the training
schedule (300 epochs, patience 30, 50-epoch warmup) and saving weights.

Because three settings moved at once, a v2−v1 difference is *not* attributable to
any one of them — that is what §1 isolates. Expect `setfusion` to move most: its
v1 row was a known early-stopping artifact and this is the first sweep where
every setfusion cell gets the warmup.""")

code(r'''# === FIGURE 3a — v2 minus v1, per cell =======================================
def grid_of(df, joint):
    g = pd.DataFrame(index=MODSET_ORDER, columns=ARCH_ORDER, dtype=float)
    if df.empty:
        return g
    for (m, a), sub in df.groupby(["modset", "arch"]):
        if joint:
            g.loc[m, a] = sub["cv"].mean()
        elif len(sub) == len(ALL_DRUGS):        # complete cells only
            g.loc[m, a] = sub["cv"].mean()
    return g


panels = [("single-drug", grid_of(V2_SD, False), grid_of(V1_SD, False)),
          ("joint", grid_of(V2_MD, True), grid_of(V1_MD, True))]
if all(g2.isna().all().all() for _, g2, _ in panels):
    note("Fig 3a needs at least one complete full_run_v2 cell "
         "(single-drug cells need all 11 drugs).")
else:
    with mpl.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 4.6))
        for ax, (name, g2, g1) in zip(axes, panels):
            d = (g2 - g1).astype(float)
            vmax = np.nanmax(np.abs(d.values)) if np.isfinite(d.values).any() else 0.01
            vmax = max(vmax, 1e-4)
            ax.imshow(np.zeros_like(d.values), cmap="Greys", vmin=0, vmax=1)
            for i in range(d.shape[0]):
                for j in range(d.shape[1]):
                    v = d.values[i, j]
                    missing = not np.isfinite(v)
                    # "not finished" must NOT read as "no change" — the diverging
                    # midpoint is also grey, so unfinished cells get a lighter
                    # hatched fill rather than the same neutral.
                    ax.add_patch(plt.Rectangle(
                        (j - .5, i - .5), 1, 1,
                        facecolor="#f2f2f2" if missing else diverging(v, vmax),
                        hatch="///" if missing else None,
                        edgecolor="#d8d8d8" if missing else "white", linewidth=2))
                    ax.text(j, i, "—" if missing else f"{v:+.3f}",
                            ha="center", va="center", fontsize=11,
                            color="0.55" if missing else
                            (INK if abs(v) < .6 * vmax else "white"),
                            fontweight="normal" if missing else "bold")
            ax.set_xticks(range(len(ARCH_ORDER)))
            ax.set_xticklabels(ARCH_ORDER, fontsize=10)
            ax.set_yticks(range(len(MODSET_ORDER)))
            ax.set_yticklabels([MODSET_LABEL[m] for m in MODSET_ORDER], fontsize=10)
            ax.set_title(f"{name}: full_run_v2 − full_run", loc="left")
            ax.grid(False)
            ax.set_xlim(-.5, len(ARCH_ORDER) - .5)
            ax.set_ylim(len(MODSET_ORDER) - .5, -.5)
        fig.suptitle("Fig 3a · blue = v2 better, vermillion = v2 worse, — = not "
                     "finished", x=0.5, y=1.03, fontsize=12, color="0.35")
        fig.tight_layout()
        save_fig(fig, "fig3a_v2_minus_v1")
        plt.show()''')

code(r'''# === FIGURE 3b — the modality ladder, full_run_v2 ============================
# 4 archs -> 4 categorical hues; green/pink are a CVD floor-only pair, so every
# line also carries its own marker and a direct end-label.
def ladder(ax, grid, baseline, base_label, title):
    xs = [m for m in MODSET_ORDER if m in grid.index]
    x = np.arange(len(xs))
    ax.set_title(title, loc="left")
    if grid.isna().all().all():                     # nothing finished in this scope
        ax.text(0.5, 0.5, "no finished cells yet", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="0.45")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        return

    ends = []
    for arch in ARCH_ORDER:
        if arch not in grid.columns:
            continue
        y = grid[arch].reindex(xs).astype(float).values
        if not np.isfinite(y).any():
            continue
        ax.plot(x, y, color=ARCH_COLOR[arch], marker=ARCH_MARKER[arch],
                ms=8, lw=2.0, zorder=4, label=arch, markeredgecolor="white",
                markeredgewidth=1.2)
        last = np.where(np.isfinite(y))[0][-1]
        ends.append([y[last], last, arch])

    # y-range first: the labels are placed in data units, so the scale must be
    # settled before de-overlapping them.
    vals = [v for v in grid.values.ravel() if np.isfinite(v)] + [baseline]
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.18, 0.004)
    ax.set_ylim(lo - pad, hi + pad)
    span = (hi + pad) - (lo - pad)

    # Direct labels collide whenever two archs finish at the same score (DNA-only
    # late_fusion and cisfusion sit within 0.002). Push them apart vertically.
    gap = span * 0.055
    for group_x in {e[1] for e in ends}:
        col = sorted([e for e in ends if e[1] == group_x], key=lambda e: e[0])
        for i in range(1, len(col)):
            if col[i][0] - col[i - 1][0] < gap:
                col[i][0] = col[i - 1][0] + gap
    # A label can still land on ANOTHER arch's line — while cells are missing a
    # line may end mid-plot, right where a longer one passes. A white halo keeps
    # it readable without moving it off its own data point.
    for yv, xi, arch in ends:
        ax.annotate(arch, (xi, yv), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=10, color=ARCH_COLOR[arch],
                    fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                              alpha=0.82))

    # baseline rule: label ABOVE the line at the left, never sitting on it
    ax.axhline(baseline, color=GREY, lw=1.6, ls="--", zorder=2)
    ax.text(-0.2, baseline + span * 0.012, base_label, va="bottom", ha="left",
            fontsize=10, color=GREY)
    ax.set_xticks(x)
    ax.set_xticklabels([MODSET_LABEL[m] for m in xs], fontsize=10)
    ax.set_ylabel("mean CV AUC")
    ax.set_xlim(-0.25, len(xs) - 0.25 + 0.95)


g_sd, g_md = grid_of(V2_SD, False), grid_of(V2_MD, True)
if g_sd.isna().all().all() and g_md.isna().all().all():
    note("Fig 3b needs finished full_run_v2 cells.")
else:
    with mpl.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4))
        ladder(axes[0], g_sd, SD_BASE_CV.mean(), "SD-CNN baseline",
               "Fig 3b · single-drug (every point = mean over all 11 drugs)")
        ladder(axes[1], g_md, MD_BASE_CV.mean(), "MD-CNN baseline", "joint")
        # legend below the plot so it never competes with the direct labels
        if not g_sd.isna().all().all():
            axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
                           ncol=4, fontsize=10)
        fig.tight_layout()
        save_fig(fig, "fig3b_ladder_v2")
        plt.show()
    print("Cells missing a drug are drawn as a gap, never averaged over fewer drugs.")''')

code(r'''# === TABLE 3 — full_run_v2 leaderboard vs the published baselines ============
rows = []
for (m, a), sub in V2_SD.groupby(["modset", "arch"]):
    drugs = [d for d in sub.drug if d in SD_BASE_CV.index]
    if len(drugs) < len(ALL_DRUGS):
        continue
    s = sub.set_index("drug").loc[drugs, "cv"]
    rows.append({"scope": "single", "modality set": MODSET_LABEL[m], "arch": a,
                 "CV": s.mean(), "Δ vs SD-CNN": (s - SD_BASE_CV[drugs]).mean(),
                 "ahead": f"{int((s > SD_BASE_CV[drugs]).sum())}/{len(drugs)}"})
for _, r in V2_MD.iterrows():
    pdru = pd.Series(r.per_drug)
    drugs = [d for d in pdru.index if d in MD_BASE_CV.index]
    ok = [d for d in drugs if d not in MDCNN_CV_SUSPECT]
    rows.append({"scope": "joint", "modality set": MODSET_LABEL[r.modset],
                 "arch": r.arch, "CV": pdru[drugs].mean(),
                 "Δ vs SD-CNN": np.nan,
                 "Δ vs MD-CNN": (pdru[drugs] - MD_BASE_CV[drugs]).mean(),
                 "Δ excl ⚠": (pdru[ok] - MD_BASE_CV[ok]).mean(),
                 "ahead": f"{int((pdru[drugs] > MD_BASE_CV[drugs]).sum())}/{len(drugs)}"})
if not rows:
    note("Table 3 needs at least one COMPLETE full_run_v2 cell.")
else:
    lb = pd.DataFrame(rows).sort_values(["scope", "CV"], ascending=[True, False])
    cols = [c for c in ["scope", "modality set", "arch", "CV", "Δ vs SD-CNN",
                        "Δ vs MD-CNN", "Δ excl ⚠", "ahead"] if c in lb.columns]
    fmt = {c: "{:+.4f}" for c in cols if c.startswith("Δ")}
    fmt["CV"] = "{:.4f}"
    sty = lb[cols].style.hide(axis="index").format(fmt, na_rep="—")
    for c in cols:
        if c.startswith("Δ"):
            sty = sty.apply(colour_delta, subset=[c])
    display(sty.set_caption("full_run_v2 leaderboard. Single-drug cells appear "
                            "only once all 11 drugs are in. ⚠ = MD-CNN's "
                            "ETHIONAMIDE CV contradicts the authors' own auc.csv.")
            .set_table_styles(TABLE_STYLES))''')

code(r'''# === FIGURE 3c — did the longer schedule change how long training runs? =====
def epoch_pool(df):
    return [e for es in df.best_epochs for e in es if e]


pools = [("full_run\n150 ep, patience 15", epoch_pool(V1_SD), epoch_pool(V1_MD)),
         ("full_run_v2\n300 ep, patience 30,\n50-epoch warmup",
          epoch_pool(V2_SD), epoch_pool(V2_MD))]
if not any(s or j for _, s, j in pools):
    note("Fig 3c needs best_epoch data.")
else:
    with mpl.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2), sharey=False)
        for ax, idx, name in ((axes[0], 1, "single-drug"), (axes[1], 2, "joint")):
            data = [p[idx] for p in pools if p[idx]]
            labs = [p[0] for p in pools if p[idx]]
            if not data:
                ax.axis("off")
                continue
            # `vert` is deprecated in mpl 3.11 and removed in 3.13; `orientation`
            # does not exist before 3.11. Pick by version so the notebook runs
            # clean on either side of that change.
            horiz = ({"orientation": "horizontal"}
                     if tuple(int(x) for x in mpl.__version__.split(".")[:2]) >= (3, 11)
                     else {"vert": False})
            parts = ax.boxplot(data, widths=0.55, patch_artist=True,
                               medianprops=dict(color=INK, lw=2),
                               flierprops=dict(marker="o", ms=4, alpha=0.4,
                                               markerfacecolor=GREY,
                                               markeredgecolor="none"), **horiz)
            for patch, col in zip(parts["boxes"], [GREY, BLUE]):
                patch.set_facecolor(col)
                patch.set_alpha(0.45)
                patch.set_edgecolor(col)
            ax.set_yticklabels(labs, fontsize=9.5)
            ax.set_xlabel("best epoch")
            ax.set_title(name, loc="left")
            ax.grid(axis="y", visible=False)
        fig.suptitle("Fig 3c · training length, v1 vs v2 (one point per CV fold)",
                     x=0.09, ha="left", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save_fig(fig, "fig3c_epochs_v1_v2")
        plt.show()''')

# ------------------------------------------------------------------ §4
md(r"""---

## 4 · Checkpoints

`full_run` saved nothing — the point of `full_run_v2` is that its models exist.
This confirms what is on disk and that a config rebuilds its model.""")

code(r'''# === WEIGHTS INVENTORY =======================================================
if not WEIGHTS.is_dir():
    note(f"weights volume not readable at {WEIGHTS}")
else:
    rows = []
    for cfg in WEIGHTS.rglob("config.json"):
        try:
            c = json.loads(cfg.read_text())
        except Exception:
            continue
        rows.append({"run": c.get("run_name", "?").split("/")[0],
                     "scope": c.get("scope"), "arch": c["model"]["arch"],
                     "params": c["model"].get("n_params") or 0,
                     "folds": len([f for f in c.get("folds", []) if f.get("weights")]),
                     "MB": sum(p.stat().st_size for p in cfg.parent.glob("*.pt")) / 1e6})
    if not rows:
        note("no checkpoints written yet — weights land when a run finishes all folds.")
    else:
        w = pd.DataFrame(rows)
        g = (w.groupby(["run", "scope"])
              .agg(models=("arch", "size"), fold_files=("folds", "sum"),
                   GB=("MB", lambda s: s.sum() / 1000)).reset_index())
        display(g.style.hide(axis="index").format({"GB": "{:.2f}"})
                .set_caption(f"checkpoints under {WEIGHTS}")
                .set_table_styles(TABLE_STYLES))
        print(f"total: {len(w)} models, {w.MB.sum()/1000:.2f} GB")''')

code(r'''# === SPOT-CHECK — a saved config really does rebuild its model ===============
try:
    import sys
    sys.path.insert(0, str(PROJECT))
    from training.checkpoint import load_model
    cfgs = sorted(WEIGHTS.rglob("config.json"))
    if not cfgs:
        note("nothing to spot-check yet.")
    else:
        c = json.loads(cfgs[0].read_text())
        run = "/".join(cfgs[0].parent.parent.parts[-2:])
        model, cfg = load_model(run, cfgs[0].parent.name)
        n = sum(p.numel() for p in model.parameters())
        print(f"rebuilt {run}/{cfgs[0].parent.name}")
        print(f"  arch={cfg['model']['arch']}  params={n:,}  "
              f"recorded={cfg['model']['n_params']:,}  "
              f"match={n == cfg['model']['n_params']}")
        print(f"  drug order: {cfg['model']['drug_names'][:3]}...  "
              f"test isolates recorded: {len(cfg['split']['test_isolate_ids'])}")
except Exception as e:
    print(f"spot-check skipped: {type(e).__name__}: {e}")''')

# ------------------------------------------------------------------ outro
md(r"""---

## What to do next

- If §1 shows folds still on the cap, the joint budget is still short — raise it
  again rather than reading the joint deltas as final.
- If §2's best arm sits inside ±0.01, capacity is not demonstrated; multi-seed
  before pursuing it.
- `full_run_v2` is only a baseline once its cells are complete — partial cells
  are excluded above, not averaged.
- Still outstanding: `--all-regulatory` (ISONIAZID loads 2 of 14 WHO promoter
  regions, KANAMYCIN 1 of 7 and loses `eis`), and multi-seeding the joint cells,
  which is the prerequisite for reporting any joint number.""")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}, indent=1))
print(f"wrote {OUT}  ({len(cells)} cells)")
