"""Generate <run-root>/full_run_viewer.ipynb for a full_run-style sweep folder.

    python scripts/build_full_run_viewer.py                          # default root
    python scripts/build_full_run_viewer.py results/experiments/full_run_v2
    python scripts/build_full_run_viewer.py <root> -o /tmp/preview.ipynb

The run root is the folder holding the ``{modality_set}__{arch}`` result
folders. It is written into the generated notebook's CONFIG cell — so the
notebook still finds its runs when opened from anywhere — and the notebook is
written into it.
"""
import argparse
import json
from pathlib import Path

DEFAULT_RUN_ROOT = Path(
    "/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion"
    "/results/experiments/full_run")

_ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("run_root", nargs="?", type=Path, default=DEFAULT_RUN_ROOT,
                 help="run folder to generate into (default: %(default)s)")
_ap.add_argument("-o", "--output", type=Path, default=None,
                 help="notebook path (default: <run_root>/full_run_viewer.ipynb)")
_args = _ap.parse_args()

RUN_ROOT = _args.run_root.expanduser().resolve()
if not RUN_ROOT.is_dir():
    raise SystemExit(f"run root is not a directory: {RUN_ROOT}")
RUN_NAME = RUN_ROOT.name
OUT = (_args.output.expanduser().resolve() if _args.output
       else RUN_ROOT / "full_run_viewer.ipynb")

cells = []


def _sub(src):
    """Bake the target run root into a generated cell.

    ``@@RUN_NAME@@`` -> the folder name (e.g. ``full_run_v2``),
    ``@@RUN_ROOT@@`` -> its absolute path. Used by the CONFIG cell so the
    generated notebook is self-locating rather than pinned to ``full_run``."""
    return src.replace("@@RUN_NAME@@", RUN_NAME).replace("@@RUN_ROOT@@", str(RUN_ROOT))


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.rstrip("\n").splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.rstrip("\n").splitlines(keepends=True)})


# ---------------------------------------------------------------- intro
md(r"""# full_run — 4 architectures × 5 modality sets, single- and multi-drug

240 jobs submitted 2026-08-04, all at **150 epochs**. This notebook reads the 40
run folders next to it.

## How to read this notebook

The sweep changes two things at once — the **architecture** and the **input
modalities** — so it is read in two stages, each asked separately of the
single-drug and the multi-drug task. Every section ends with a one-line ANSWER.

| § | question | comparison |
|---|---|---|
| **1** | With **DNA alone**, do we beat the single-drug baseline? | ours (DNA only) vs leak-corrected **SD-CNN** |
| **2** | With **DNA alone**, do we beat the multi-drug baseline? | ours (DNA only) vs published **MD-CNN** |
| **3** | What do the **extra modalities** add, single-drug? | each cell vs **its own architecture's DNA-only cell** |
| **4** | What do the **extra modalities** add, multi-drug? | same, joint scope |
| **5** | Takeaways | — |

Stage 1 (§1–§2) isolates the model and training protocol at inputs matched to the
baseline. Stage 2 (§3–§4) isolates the modalities with the architecture held
fixed. Keeping them apart is the only way to say which of the two is responsible
for a margin.

**The short version:** on DNA alone we already clear the single-drug baseline
(+0.017) but not the multi-drug one (−0.012). The modalities then roughly double
the single-drug margin (to +0.041, ahead on 11/11 drugs) and close the multi-drug
deficit to parity. And the modality gain is not spread across drugs — three drugs
carry nearly all of it, for reasons the resistance mechanism predicts:

Best modality gain per drug, mean over `late_fusion` / `mdcnn` / `cisfusion`
(setfusion excluded — see below, its DNA reference is broken):

| drug | what carries resistance | modality that helps | best gain over DNA-only |
|---|---|---|---|
| **PYRAZINAMIDE** | *pncA* loss-of-function — hundreds of distinct inactivating substitutions scattered across the gene, most unseen in training | **protein / biophysical**: amino-acid identity and physicochemical change let the model score "does this substitution break the protein" instead of memorising positions | **+0.099** (up to +0.14 under cisfusion) |
| **ETHIONAMIDE** | the *fabG1–inhA* operon promoter (`c-15t`), invisible to a CDS one-hot | **regulatory** | **+0.094** |
| **ISONIAZID** | *katG*, plus the same *fabG1–inhA* promoter | **regulatory** | **+0.042** |
| LEVOFLOXACIN | *gyrA/gyrB* — but n=269, so this number is noise either way | — | +0.015 |
| CAPREOMYCIN | *rrs*/*rrl* (rRNA) **plus *tlyA***, which is protein-coding — so a small protein gain is expected here and not for AMK/KAN | protein (weakly) | +0.014 |
| STREPTOMYCIN / MOXIFLOXACIN / ETHAMBUTOL | *rpsL* / *gyrA* / *embB* point mutations — positional signal a one-hot already captures | little to add | +0.008 to +0.010 |
| **KANAMYCIN / AMIKACIN** | *rrs* — 16S **rRNA**. No protein product, so protein and biophysical descriptors are undefined on the locus that matters | **none apply** | **+0.002 / +0.001** |
| **RIFAMPICIN** | *rpoB* RRDR, a tight positional signal already at CV 0.976 | **no headroom** | **+0.000** |

Fig 0 in §3 is that claim, unaggregated. Averaging over drugs turns a large,
mechanistically legible effect on three drugs into a bland "+0.03 overall" and
hides that the bottom five drugs got nothing.

**Judge on cross-validated AUC (5-fold mean ± SD).** Held-out test AUC is a
single 20% split of one best-CV-fold model; it is reported because it is what the
paper publishes, but it swings by ±0.05 on the small drugs. `—` is a run that
has not finished, never a zero.

**Baselines (this script is their single source of truth):**

| baseline | what our models compare against | corrected? |
|---|---|---|
| **SD-CNN (OHE)** | our single-drug configs (one model per drug) | **yes** — the published test AUC is inflated by a crossval/assess stratify leak (~80% of its "test" isolates were in training). We use the leak-corrected numbers. |
| **MD-CNN (OHE)** | our joint configs (one model, all 11 drugs) | not needed — its assess script splits on a stored cohort column and never re-splits. |

### Grid cells that are structurally degenerate

These still run and still produce valid numbers, but they cannot show what the
architecture was built for — the notebook greys them out so they are never read
as a fair test of the idea:

| cell | why |
|---|---|
| `cisfusion` on `dna`, `dna_protein`, `dna_biophysical` | no promoter to pair with a CDS → reduces to late fusion plus a constant segment channel. The two cells that DO exercise it are `dna_regulatory__cisfusion` and `all_modalities__cisfusion`. |
| `setfusion` on `dna` | one modality, so nothing to pair across modalities → a weight-sharing ablation, not a locus-pairing test |

### `setfusion` in this run is an early-stopping artifact, not an architecture verdict

**Do not read the setfusion row as evidence about locus-keyed transformer fusion.**
Its training histories show the failure directly: train loss sits flat (~0.2405)
for ~12 epochs — the near-collinear-token degenerate init — the monitored val AUC
peaks *inside* that plateau, and patience=15 then fires around epoch 25 and
restores weights from **before** the network broke out. Across all 25 joint
setfusion folds `best_epoch` is 3–12 while the loss break is at 8–20. The two
folds that happened to survive past the break (`dna_biophysical` folds 3 and 4,
running to epochs 72 and 113) scored 0.811 and **0.8385**, far above the 0.78
cell mean. It does this at **0.5M parameters against late_fusion's 46M**.

Re-run with an early-stopping warmup (`--min-epochs 50`, added to
`training.core.EarlyStopper`) into `results/experiments/setfusion_warmup/`.
Until those land, every setfusion number here is a lower bound.

Two more asymmetries to keep in mind, neither of which the plots can correct for:

- **The scopes use different locus sets.** Single-drug uses BIG-TB's per-drug
  `DRUG_TO_LOCI` (18-locus universe, no *fabG1*, SD-CNN-matched); multi-drug uses
  every curated locus on disk (**19, including *fabG1***), which is MD-CNN's own
  rule. So INH/ETO are not locus-matched *across* scopes, and the joint model sees
  strictly more input per drug than the single-drug model does — "joint wins" in
  Fig E therefore conflates multi-task sharing with a bigger locus universe.
  `results/experiments/alllocus_run/` re-runs the whole single-drug grid on the
  same 19 loci to separate the two.
- **`mdcnn` groups blocks by channel count**, and DNA and regulatory are both
  5-channel — so in `dna_regulatory__mdcnn` and `all_modalities__mdcnn` the
  promoter windows ride as extra channels in the DNA trunk rather than getting
  their own.
""")

# ---------------------------------------------------------------- imports
code(r"""import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from IPython.display import display""")

# ---------------------------------------------------------------- config
code(_sub(r'''# === CONFIG =================================================================
# This notebook lives in <project>/results/experiments/@@RUN_NAME@@/.
_here = Path.cwd()
RUN_DIR = _here if (_here / "README.md").exists() and _here.name == "@@RUN_NAME@@" else None
if RUN_DIR is None:              # fall back to the root this was generated for
    for c in [Path(r"@@RUN_ROOT@@"), _here]:
        if c.is_dir():
            RUN_DIR = c
            break
PROJECT = RUN_DIR.parents[2]
print("Reading runs from:", RUN_DIR)

# --- the grid ---------------------------------------------------------------
ARCH_ORDER = ["late_fusion", "mdcnn", "setfusion", "cisfusion"]
ARCH_LABEL = {          # short enough not to collide as heatmap x-ticks
    "late_fusion": "late fusion\n(per-block)",
    "mdcnn":       "mdcnn\n(BIG-TB)",
    "setfusion":   "setfusion\n(shared+keyed)",
    "cisfusion":   "cisfusion\n(prom⊕CDS)",
}
ARCH_SHORT = {"late_fusion": "late_fusion", "mdcnn": "mdcnn",
              "setfusion": "setfusion", "cisfusion": "cisfusion"}
MODSET_ORDER = ["dna", "dna_protein", "dna_biophysical", "dna_regulatory",
                "all_modalities"]
MODSET_LABEL = {"dna": "DNA", "dna_protein": "DNA+protein",
                "dna_biophysical": "DNA+biophys", "dna_regulatory": "DNA+regulatory",
                "all_modalities": "all modalities"}
REFERENCE_MODSET = "dna"      # modality gains are measured against this

# Cells that cannot demonstrate what the architecture is for (see intro).
# NB: membership must be tested against the modality LIST, not the folder-name
# string — "regulatory" is not a substring of "all_modalities", which previously
# greyed out all_modalities__cisfusion, one of the two cells that DO exercise it.
MODSET_MEMBERS = {
    "dna": {"dna"},
    "dna_protein": {"dna", "protein"},
    "dna_biophysical": {"dna", "biophysical"},
    "dna_regulatory": {"dna", "regulatory"},
    "all_modalities": {"dna", "protein", "biophysical", "regulatory"},
}


def is_degenerate(arch, modset):
    mods = MODSET_MEMBERS.get(modset, set())
    if arch == "cisfusion" and "regulatory" not in mods:
        return "no promoter to pair"
    if arch == "setfusion" and len(mods) < 2:
        return "single modality — nothing to pair"
    return ""

# --- BIG-TB baselines -------------------------------------------------------
# Both tables (leak-corrected SD-CNN, and MD-CNN read from the authors' own run
# output) live in <project>/bigtb_baselines.py, which is their single source of
# truth. Do not paste a second copy into a notebook -- the 2026-08-13 cleanup
# deleted one that had drifted.
import sys
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from bigtb_baselines import (           # noqa: E402
    SDCNN_CLEAN, SDCNN_CV_SD, BIGTB_MDCNN, MDCNN_CV_SUSPECT, ALL_DRUGS,
    SD_BASE_CV, SD_BASE_TEST, MD_BASE_CV, MD_BASE_TEST)

# --- style (colourblind-safe Okabe-Ito palette) -----------------------------
OUTDIR = PROJECT / "results/figures/@@RUN_NAME@@"  # None to display without saving
BLUE, ORANGE, VERM, GREY, INK = "#0072B2", "#E69F00", "#D55E00", "#7F7F7F", "#1A1A1A"
GOOD_BG = "background-color:#d9ecd9;color:#14501e"
BAD_BG  = "background-color:#f7d9d3;color:#7f2704"
DEGEN_BG = "background-color:#ececec;color:#8a8a8a;font-style:italic"
RC = {
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.titleweight": "bold", "axes.edgecolor": "0.3",
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "0.25", "ytick.color": "0.25",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
}
TABLE_STYLES = [
    {"selector": "th.col_heading", "props":
     "text-align:center; white-space:pre-line; font-size:12px; vertical-align:bottom;"},
    {"selector": "th.col_heading.level0", "props":
     "background-color:#eef2f6; border-bottom:2px solid #97a7b5;"},
    {"selector": "caption", "props":
     "caption-side:top; text-align:left; font-size:13px; padding-bottom:6px;"},
]


def save_fig(fig, name):
    if OUTDIR is None:
        return
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(Path(OUTDIR) / f"{name}.{ext}")


def colour_delta(col):
    return [GOOD_BG if v > 0.005 else (BAD_BG if v < -0.005 else "") for v in col]


def pm(mean, sd):
    if pd.isna(mean):
        return "—"
    return f"{mean:.3f} ± {sd:.3f}" if not pd.isna(sd) else f"{mean:.3f}"'''))

# ---------------------------------------------------------------- load
code(r'''# === LOAD ===================================================================
# Folder names are "{modality_set}__{arch}" (single-drug) and
# "multidrug_{modality_set}__{arch}" (joint). Anything that doesn't parse is
# ignored, so stray folders can't corrupt the tables.
def parse_folder(name):
    m = re.fullmatch(r"(multidrug_)?(.+?)__(.+)", name)
    if not m:
        return None
    joint, modset, arch = bool(m.group(1)), m.group(2), m.group(3)
    if modset not in MODSET_ORDER or arch not in ARCH_ORDER:
        return None
    return ("multi" if joint else "single"), modset, arch


sd_rows, md_rows, meta, param_range = [], [], {}, {}
for p in sorted(RUN_DIR.iterdir()):
    if not p.is_dir():
        continue
    parsed = parse_folder(p.name)
    if parsed is None:
        continue
    scope, modset, arch = parsed
    if scope == "single":
        f = p / "summary.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)
        s["drug"] = s["drug"].str.upper()
        s["modset"], s["arch"], s["folder"] = modset, arch, p.name
        sd_rows.append(s)
        # A single-drug cell is 11 DIFFERENT models — each drug loads only its own
        # loci, so n_params ranges ~1.1M-9.3M inside one cell. Reading just the
        # first JSON (alphabetically AMIKACIN) reported one drug's size as the
        # cell's, so keep the whole spread and render it as a range.
        sizes = []
        for jf in sorted(p.glob("*__*.json")):
            raw = json.loads(jf.read_text())
            if p.name not in meta:
                meta[p.name] = raw
            if raw.get("n_params"):
                sizes.append(int(raw["n_params"]))
        if sizes:
            param_range[p.name] = (min(sizes), max(sizes))
    else:
        f = p / "multidrug_summary.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d["drug"] = d["drug"].str.upper()
        # per-drug fold spread isn't in the csv — recover it from the run JSON
        js = sorted(p.glob("multidrug__*.json"))
        sd_map = {}
        if js:
            raw = json.loads(js[0].read_text())
            meta[p.name] = raw
            for drug in raw.get("cv_per_drug_auc", {}):
                vals = [fo["per_drug"][drug]["auc"] for fo in raw.get("cv_folds", [])
                        if drug in fo.get("per_drug", {})
                        and fo["per_drug"][drug]["auc"] == fo["per_drug"][drug]["auc"]]
                sd_map[drug] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        d["cv_auc_std"] = d["drug"].map(sd_map)
        d["modset"], d["arch"], d["folder"] = modset, arch, p.name
        md_rows.append(d)

SD = pd.concat(sd_rows, ignore_index=True) if sd_rows else pd.DataFrame()
MD = pd.concat(md_rows, ignore_index=True) if md_rows else pd.DataFrame()
n_params = {k: v.get("n_params") for k, v in meta.items()}   # joint: one model


def param_str(folder):
    """Model size for a cell. Joint cells are ONE model -> one number. Single-drug
    cells are 11 models of different sizes -> "min - max"."""
    if folder in param_range:
        lo, hi = param_range[folder]
        return f"{lo:,}" if lo == hi else f"{lo/1e6:.2f}–{hi/1e6:.2f}M"
    v = n_params.get(folder)
    return f"{v:,}" if v else "—"

print(f"single-drug: {0 if SD.empty else SD.folder.nunique()}/20 configs, "
      f"{0 if SD.empty else len(SD)}/220 drug-runs")
print(f"joint      : {0 if MD.empty else MD.folder.nunique()}/20 configs")
if SD.empty and MD.empty:
    print("\nNothing has finished yet — rerun this cell when jobs land.")''')

# ---------------------------------------------------------------- coverage
code(r'''# === COVERAGE — what has landed (rerun while jobs are still going) =========
cov = pd.DataFrame(index=[MODSET_LABEL[m] for m in MODSET_ORDER],
                   columns=pd.MultiIndex.from_product(
                       [["single-drug (n drugs / 11)", "joint (done?)"],
                        [ARCH_SHORT[a] for a in ARCH_ORDER]]))
for m in MODSET_ORDER:
    for a in ARCH_ORDER:
        n = 0 if SD.empty else int(((SD.modset == m) & (SD.arch == a)).sum())
        cov.loc[MODSET_LABEL[m], ("single-drug (n drugs / 11)", ARCH_SHORT[a])] = n
        done = (not MD.empty) and (((MD.modset == m) & (MD.arch == a)).any())
        cov.loc[MODSET_LABEL[m], ("joint (done?)", ARCH_SHORT[a])] = "yes" if done else "—"


def _cov_style(v):
    if v == 11 or v == "yes":
        return GOOD_BG
    if v == 0 or v == "—":
        return "color:#999"
    return "background-color:#fff3cd"


display(cov.style.apply(lambda col: [_cov_style(v) for v in col])
        .set_table_styles(TABLE_STYLES)
        .set_caption("<b>Coverage.</b> Green = complete. 220 single-drug jobs "
                     "(11 per cell) + 20 joint jobs. Partial cells are averaged "
                     "over the drugs they have, so treat them as provisional."))''')

# ---------------------------------------------------------------- scorecard
code(r'''# === SCORECARD — the four numbers, computed not typed =======================
def _fmt(v, n=4):
    return "—" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:+.{n}f}"


cards = []

# 1. single-drug vs the leak-corrected SD-CNN, COMPLETE cells only
best_sd = None
if not SD.empty:
    for (m, a), g in SD.groupby(["modset", "arch"]):
        d = g.set_index("drug")
        drugs = [x for x in d.index if x in SD_BASE_CV.index]
        if len(drugs) < len(ALL_DRUGS):
            continue                       # partial cells can't headline
        delta = (d.loc[drugs, "cv_auc_mean"] - SD_BASE_CV[drugs]).mean()
        ahead = int((d.loc[drugs, "cv_auc_mean"] > SD_BASE_CV[drugs]).sum())
        if best_sd is None or delta > best_sd[0]:
            best_sd = (delta, m, a, d.loc[drugs, "cv_auc_mean"].mean(), ahead, len(drugs))
if best_sd:
    delta, m, a, mean_cv, ahead, nd = best_sd
    cards.append({
        "claim": "We beat the corrected SD-CNN",
        "best configuration": f"{MODSET_LABEL[m]} / {ARCH_SHORT[a]}  (single-drug)",
        "headline": f"CV {mean_cv:.4f} vs {SD_BASE_CV.mean():.4f}",
        "Δ": delta, "drugs ahead": f"{ahead}/{nd}",
        "caveat": "CV-vs-CV, both leak-free. Test AUC is best-of-5-folds, so quote CV.",
    })

# 2. joint vs the published MD-CNN
if not MD.empty:
    pdm = MD[MD.drug != "MACRO"]
    best_md = None
    for (m, a), g in pdm.groupby(["modset", "arch"]):
        d = g.set_index("drug")
        drugs = [x for x in d.index if x in MD_BASE_CV.index]
        ok = [x for x in drugs if x not in MDCNN_CV_SUSPECT]
        delta = (d.loc[drugs, "cv_auc"] - MD_BASE_CV[drugs]).mean()
        if best_md is None or delta > best_md[0]:
            best_md = (delta, m, a, d.loc[drugs, "cv_auc"].mean(),
                       (d.loc[ok, "cv_auc"] - MD_BASE_CV[ok]).mean(),
                       int((d.loc[drugs, "cv_auc"] > MD_BASE_CV[drugs]).sum()), len(drugs))
    delta, m, a, macro, d_ok, ahead, nd = best_md
    cards.append({
        "claim": "Joint models reach MD-CNN parity",
        "best configuration": f"{MODSET_LABEL[m]} / {ARCH_SHORT[a]}  (joint)",
        "headline": f"macro CV {macro:.4f} vs {MD_BASE_CV.mean():.4f}",
        "Δ": delta, "drugs ahead": f"{ahead}/{nd}",
        "caveat": f"Δ excluding the flagged ETHIONAMIDE baseline row: {d_ok:+.4f} "
                  "— that single suspect number decides parity-vs-behind.",
    })

# 3. the modality effect, on the drug where it is largest
if not SD.empty:
    core_archs = [a for a in ARCH_ORDER if a != "setfusion"]
    piv0 = SD.pivot_table(index="drug", columns=["arch", "modset"], values="cv_auc_mean")
    gains = {}
    for d in piv0.index:
        vals = []
        for a in core_archs:
            if (a, REFERENCE_MODSET) in piv0.columns and (a, "all_modalities") in piv0.columns:
                vals.append(piv0.loc[d, (a, "all_modalities")] - piv0.loc[d, (a, REFERENCE_MODSET)])
        if vals:
            gains[d] = float(np.nanmean(vals))
    gs = pd.Series(gains).sort_values(ascending=False)
    n_big = int((gs > 0.02).sum())
    cards.append({
        "claim": "Extra modalities help 3 drugs, not all 11",
        "best configuration": f"top: {gs.index[0]} ({gs.iloc[0]:+.3f}), "
                              f"{gs.index[1]} ({gs.iloc[1]:+.3f}), "
                              f"{gs.index[2]} ({gs.iloc[2]:+.3f})",
        "headline": f"{n_big} drugs gain over 0.02; "
                    f"{int((gs.abs() < 0.01).sum())} gain under 0.01",
        "Δ": gs.mean(),
        "drugs ahead": f"{int((gs > 0).sum())}/{len(gs)}",
        "caveat": "all-modalities minus DNA-only, mean over the 3 sound "
                  "architectures. See Fig 0 — the mean is NOT the story.",
    })

# 4. joint vs single on matched inputs
if not MD.empty and not SD.empty:
    pdm = MD[MD.drug != "MACRO"]
    wins, deltas, sound = 0, [], []
    for (m, a), g in pdm.groupby(["modset", "arch"]):
        sv = SD[(SD.modset == m) & (SD.arch == a)].set_index("drug")["cv_auc_mean"]
        j = g.set_index("drug")["cv_auc"]
        common = [d for d in j.index if d in sv.index]
        if not common:
            continue
        dv = (j[common] - sv[common]).mean()
        deltas.append(dv)
        wins += dv > 0
        if a != "setfusion":      # its single-drug reference is the artifact
            sound.append(dv)
    headline_mean = float(np.mean(sound)) if sound else float(np.mean(deltas))
    cards.append({
        "claim": "One joint model beats 11 single-drug models",
        "best configuration": f"{wins}/{len(deltas)} configs favour joint",
        "headline": f"mean Δ {headline_mean:+.4f} over the "
                    f"{len(sound)} sound-architecture configs",
        "Δ": headline_mean, "drugs ahead": "—",
        "caveat": "CONFOUNDED: joint also sees all 19 loci per drug, single-drug "
                  "sees only that drug's — alllocus_run/ separates the two. "
                  "setfusion configs excluded (broken single-drug reference); "
                  f"including them the mean is {np.mean(deltas):+.4f}.",
    })

SC = pd.DataFrame(cards).set_index("claim")
display(SC.style.format({"Δ": "{:+.4f}"})
        .apply(lambda col: [GOOD_BG if v > 0.005 else (BAD_BG if v < -0.005 else "")
                            for v in col], subset=["Δ"])
        .set_properties(subset=["caveat"], **{"font-size": "11px", "color": "#555"})
        .set_table_styles(TABLE_STYLES)
        .set_caption("<b>Scorecard.</b> The four claims of §1–§4 with the caveat "
                     "that limits each. Every number is recomputed from the run "
                     "folders on this notebook's last execution — nothing typed."))''')

# ---------------------------------------------------------------- modality
md(r"""### Fig 0 · which drugs the gain comes from

Each cell is **CV-AUC(modality set) − CV-AUC(DNA-only)** for the same drug and the
same architecture, so every column is a controlled one-modality-at-a-time ablation
against its own DNA baseline. This is the plot the grid means average away.""")

code(r'''# === FIGURE 0 — per-drug modality gain over DNA-only, per architecture ======
# Rows = drugs (ordered by n_valid, smallest at top), columns = (arch, modality
# set) with dna_* subtracted out. Blue = the added modality helped that drug.
if SD.empty:
    print("No single-drug results yet.")
else:
    gain_mods = [m for m in MODSET_ORDER if m != REFERENCE_MODSET]
    piv = SD.pivot_table(index="drug", columns=["arch", "modset"], values="cv_auc_mean")
    n_by_drug = SD.groupby("drug")["n_valid"].max()
    archs = [a for a in ARCH_ORDER if a in piv.columns.get_level_values(0)]
    cols, blocks = [], []
    for a in archs:
        if REFERENCE_MODSET not in piv[a].columns:
            continue
        base = piv[(a, REFERENCE_MODSET)]
        for m in gain_mods:
            if (a, m) in piv.columns:
                cols.append((a, m))
                blocks.append((piv[(a, m)] - base).rename((a, m)))
    if not blocks:
        print("Need the dna reference cell for at least one architecture.")
    else:
        G = pd.concat(blocks, axis=1)
        drugs = [d for d in n_by_drug.sort_values().index if d in G.index]
        G = G.loc[drugs]

        with mpl.rc_context(RC):
            fig, ax = plt.subplots(figsize=(0.62 * G.shape[1] + 5, 0.52 * len(drugs) + 3.4))
            lim = float(np.nanmax(np.abs(G.values))) or 0.01
            im = ax.imshow(G.values, cmap="RdBu",
                           norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                           aspect="auto")
            for i in range(G.shape[0]):
                for j in range(G.shape[1]):
                    v = G.values[i, j]
                    if np.isnan(v):
                        ax.text(j, i, "—", ha="center", va="center", color="0.6",
                                fontsize=8)
                        continue
                    ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8,
                            fontweight="bold" if abs(v) >= 0.02 else "normal",
                            color="white" if abs(v) > 0.62 * lim else "0.15")
            # architecture group separators + labels
            for k in range(1, len(archs)):
                ax.axvline(k * len(gain_mods) - 0.5, color="0.25", lw=2)
            # group labels sit just above the axes (x in data coords, y in axes
            # coords) so they cannot collide with the title however tall it is
            for k, a in enumerate(archs):
                ax.text((k + 0.5) * len(gain_mods) - 0.5, 1.008, ARCH_SHORT[a],
                        transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=11, fontweight="bold", color="0.15")
            ax.set_xticks(range(G.shape[1]),
                          [MODSET_LABEL[m].replace("DNA+", "+") for _, m in cols],
                          fontsize=8.5, rotation=90)
            ax.set_yticks(range(len(drugs)),
                          [f"{d}  (n={n_by_drug[d]:,})" for d in drugs], fontsize=9.5)
            ax.set_xticks(np.arange(-.5, G.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(drugs), 1), minor=True)
            ax.grid(which="minor", color="white", lw=1.5)
            ax.tick_params(which="minor", length=0)
            ax.set_title("Fig 0 · Δ CV-AUC from adding a modality to DNA-only, "
                         "per drug × architecture", loc="left", pad=34)
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.015).set_label("Δ CV-AUC vs DNA-only")
            fig.tight_layout()
        save_fig(fig, "fig0_modality_gain_per_drug")
        plt.show()

        # The same thing as a table, averaged over architectures — but NOT over
        # setfusion. Its DNA-only cell is the early-stopping casualty (CV 0.820
        # against ~0.878 for the others), so every gain measured against that
        # reference is inflated by the broken baseline, not by the modality. It
        # stays in the heatmap above, where it is visibly its own column.
        avg_archs = [a for a in archs if a != "setfusion"] or archs
        Gv = G[[c for c in G.columns if c[0] in avg_archs]]
        tbl = Gv.T.groupby(level=1).mean().T.reindex(columns=gain_mods)
        tbl.columns = [MODSET_LABEL[m] for m in gain_mods]
        tbl.insert(0, "n isolates", [n_by_drug[d] for d in tbl.index])
        excluded = "setfusion" if "setfusion" in archs and avg_archs != archs else None
        display(tbl.style
                .format({c: "{:+.3f}" for c in tbl.columns if c != "n isolates"}, na_rep="—")
                .format({"n isolates": "{:,.0f}"})
                .apply(colour_delta, subset=[c for c in tbl.columns if c != "n isolates"])
                .set_table_styles(TABLE_STYLES)
                .set_caption("<b>Modality gain over DNA-only, averaged over "
                             f"{len(avg_archs)} architectures"
                             + (f" (<b>{excluded} excluded</b> — its DNA-only "
                                "reference is the early-stopping casualty, so gains "
                                "measured against it are inflated by the broken "
                                "baseline)" if excluded else "") +
                             ".</b> Green = the modality added ≥0.005 AUC for that "
                             "drug. Three drugs carry nearly all of it; the rRNA "
                             "drugs (AMIKACIN / KANAMYCIN / CAPREOMYCIN) cannot "
                             "benefit from protein or biophysical features at all, "
                             "because <i>rrs</i> has no protein product."))

        core = tbl.drop(columns=["n isolates"])
        print(f"largest modality gain per drug (mean over {avg_archs}):")
        for d, v in core.max(axis=1).sort_values(ascending=False).items():
            print(f"  {d:<14} {v:+.3f}   ({core.loc[d].idxmax()})")''')

md(r"""---

## Which architecture and which modality set

Now the aggregate view. Two things to hold onto while reading it:

- **`mdcnn` wins single-drug, `late_fusion` wins joint.** The ranking flips
  between scopes (quantified in Fig D2). Mixing every locus at layer 1 is a good
  prior when a drug has 2–4 relevant loci and a bad one when all 19 are present
  for all 11 drugs.
- **`setfusion`'s row is an early-stopping artifact**, not an architecture
  verdict — see the intro. Its numbers here are lower bounds.""")

code(r'''# === TABLE A — single-drug leaderboard: every architecture × modality set ==
# Mean CV-AUC over the drugs a cell has, and the same drugs' SD-CNN baseline, so
# the Δ is always like-for-like even when a cell is only partly finished.
if SD.empty:
    print("No single-drug results yet.")
else:
    rows = []
    for (m, a), g in SD.groupby(["modset", "arch"]):
        d = g.set_index("drug")
        drugs = [x for x in d.index if x in SD_BASE_CV.index]
        base_cv, base_test = SD_BASE_CV[drugs], SD_BASE_TEST[drugs]
        cv, te = d.loc[drugs, "cv_auc_mean"], d.loc[drugs, "test_auc"]
        rows.append({
            "modality set": MODSET_LABEL[m], "architecture": ARCH_SHORT[a],
            "drugs": len(drugs),
            "mean CV": cv.mean(), "Δ CV vs SD-CNN": (cv - base_cv).mean(),
            "drugs ahead (CV)": int((cv > base_cv).sum()),
            "mean test": te.mean(), "Δ test vs SD-CNN": (te - base_test).mean(),
            "params": param_str(f"{m}__{a}"),
            "note": is_degenerate(a, m),
            "_m": m, "_a": a,
        })
    # SORT ON Δ, NOT ON mean CV. `mean CV` averages over whatever drugs a cell
    # finished, so a cell missing a hard drug outranks a complete one for free —
    # dna_biophysical__mdcnn topped the old ordering only because its missing run
    # was ETHIONAMIDE (baseline 0.622, the hardest drug in the set). Δ is matched
    # per cell to the drugs it has, so it is the comparable quantity.
    lb = pd.DataFrame(rows).sort_values("Δ CV vs SD-CNN", ascending=False)
    n_expected = int(lb["drugs"].max()) if len(lb) else 0
    lb["mean CV"] = [f"{v:.4f}" + ("" if n == n_expected else " ⚠")
                     for v, n in zip(lb["mean CV"], lb["drugs"])]
    lb_show = lb.drop(columns=["_m", "_a"]).set_index(["modality set", "architecture"])

    def _row_style(row):
        deg = bool(row["note"])
        return [DEGEN_BG if deg else "" for _ in row]

    display(lb_show.style
            .format({"mean test": "{:.4f}",
                     "Δ CV vs SD-CNN": "{:+.4f}", "Δ test vs SD-CNN": "{:+.4f}"},
                    na_rep="—")
            .apply(_row_style, axis=1)
            .apply(colour_delta, subset=["Δ CV vs SD-CNN"])
            .set_table_styles(TABLE_STYLES)
            .set_caption(
                "<b>Single-drug leaderboard, sorted by Δ CV-AUC vs the baseline.</b> "
                "Δ columns compare against the leak-corrected SD-CNN on the SAME "
                "drugs the cell has finished, so they are comparable across rows; "
                "<b>mean CV is not</b> — a ⚠ marks a cell that is missing a drug, "
                "whose raw mean is inflated or deflated by WHICH drug is missing. "
                "`params` is a range: a single-drug cell is 11 different-sized "
                "models, one per drug's locus set. Greyed rows are structurally "
                "degenerate cells (see intro) — their numbers are valid but they "
                "are not a test of the architecture's idea."))''')

# ---------------------------------------------------------------- fig A
code(r'''# === FIGURE A — the grid: architecture × modality set =======================
# Left: mean CV-AUC. Right: Δ vs the leak-corrected SD-CNN (blue = we beat it).
# Hatched = degenerate cell.
if SD.empty:
    print("No single-drug results yet.")
else:
    grid = pd.DataFrame(index=MODSET_ORDER, columns=ARCH_ORDER, dtype=float)
    dgrid = grid.copy()
    for (m, a), g in SD.groupby(["modset", "arch"]):
        d = g.set_index("drug")
        drugs = [x for x in d.index if x in SD_BASE_CV.index]
        grid.loc[m, a] = d.loc[drugs, "cv_auc_mean"].mean()
        dgrid.loc[m, a] = (d.loc[drugs, "cv_auc_mean"] - SD_BASE_CV[drugs]).mean()

    with mpl.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4))
        for ax, data, title, cmap, norm, fmt in [
            (axes[0], grid, "Mean CV-AUC", "Blues", None, "{:.3f}"),
            (axes[1], dgrid, "Δ CV-AUC vs leak-corrected SD-CNN",
             "RdBu", TwoSlopeNorm(vcenter=0.0), "{:+.3f}")]:
            vals = data.values.astype(float)
            if np.isfinite(vals).any() and norm is not None:
                lim = float(np.nanmax(np.abs(vals)))
                norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
            im = ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")
            for i, m in enumerate(MODSET_ORDER):
                for j, a in enumerate(ARCH_ORDER):
                    v = vals[i, j]
                    if not np.isfinite(v):
                        ax.text(j, i, "—", ha="center", va="center", color="0.6")
                        continue
                    if is_degenerate(a, m):
                        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                                   hatch="///", edgecolor="0.75", lw=0))
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=11,
                            fontweight="bold", color="0.12")
            ax.set_xticks(range(len(ARCH_ORDER)),
                          [ARCH_LABEL[a] for a in ARCH_ORDER], fontsize=9.5)
            ax.set_yticks(range(len(MODSET_ORDER)),
                          [MODSET_LABEL[m] for m in MODSET_ORDER], fontsize=10.5)
            ax.set_xticks(np.arange(-.5, len(ARCH_ORDER), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(MODSET_ORDER), 1), minor=True)
            ax.grid(which="minor", color="white", lw=2)
            ax.tick_params(which="minor", length=0)
            ax.set_title(title + "\n", loc="left")
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        axes[1].text(0, 1.012, "hatched = degenerate cell (see intro)",
                     transform=axes[1].transAxes, fontsize=10, color="0.45", va="bottom")
        axes[0].text(0, 1.012,
                     f"SD-CNN baseline, mean over all 11 drugs: {SD_BASE_CV.mean():.3f}"
                     "  ·  each Δ at right is matched per cell to the drugs it has",
                     transform=axes[0].transAxes, fontsize=10, color="0.45", va="bottom")
        fig.suptitle("Fig A · single-drug: which architecture, which modality set",
                     x=0.01, ha="left", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "figA_grid_single_drug")
    plt.show()

    # marginals — the quick read
    if not SD.empty:
        by_arch = (grid.mean(axis=0).rename("mean CV over modality sets")
                   .to_frame().join(dgrid.mean(axis=0).rename("mean Δ vs SD-CNN")))
        by_mod = (grid.mean(axis=1).rename("mean CV over architectures")
                  .to_frame().join(dgrid.mean(axis=1).rename("mean Δ vs SD-CNN")))
        by_mod.index = [MODSET_LABEL[m] for m in by_mod.index]
        display(by_arch.style.format("{:+.4f}").set_table_styles(TABLE_STYLES)
                .set_caption("<b>Architecture effect</b> — averaged over modality sets."))
        display(by_mod.style.format("{:+.4f}").set_table_styles(TABLE_STYLES)
                .set_caption("<b>Modality effect</b> — averaged over architectures."))''')

# ---------------------------------------------------------------- fig B
md(r"""---

## Beating the SD-CNN, drug by drug

Two views, and the difference between them matters:

- **Fig C1 is the honest one.** It picks the single best *complete* config —
  one configuration, fixed across all 11 drugs, chosen once — and shows it
  against the leak-corrected baseline. Nothing is selected per drug.
- **Fig C is a ceiling.** It picks each drug's best config out of all 20, so it
  is selected on the same CV folds it reports. Read it as "is the grid capable
  of beating the baseline anywhere", never as a deployable model.""")

code(r'''# === FIGURE B — per drug: where do we actually beat the baseline? ==========
# Rows = drugs, columns = the 20 configs. Cell = CV-AUC − leak-corrected SD-CNN
# CV. This is the "did we beat it" view; blue is ahead.
if SD.empty:
    print("No single-drug results yet.")
else:
    piv = SD.assign(cfg=[f"{MODSET_LABEL[m]}\n{ARCH_SHORT[a]}"
                         for m, a in zip(SD.modset, SD.arch)]) \
            .pivot_table(index="drug", columns="cfg", values="cv_auc_mean")
    order_cfg = [f"{MODSET_LABEL[m]}\n{ARCH_SHORT[a]}"
                 for m in MODSET_ORDER for a in ARCH_ORDER
                 if f"{MODSET_LABEL[m]}\n{ARCH_SHORT[a]}" in piv.columns]
    piv = piv[order_cfg]
    drugs = [d for d in SD_BASE_CV.sort_values().index if d in piv.index]
    delta = piv.loc[drugs].sub(SD_BASE_CV[drugs], axis=0)

    with mpl.rc_context(RC):
        fig, ax = plt.subplots(figsize=(0.78 * len(order_cfg) + 4, 0.5 * len(drugs) + 3))
        lim = float(np.nanmax(np.abs(delta.values))) or 0.01
        im = ax.imshow(delta.values, cmap="RdBu",
                       norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), aspect="auto")
        for i in range(delta.shape[0]):
            for j in range(delta.shape[1]):
                v = delta.values[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.62 * lim else "0.15")
        ax.set_xticks(range(len(order_cfg)), order_cfg, fontsize=8, rotation=90)
        ax.set_yticks(range(len(drugs)), drugs, fontsize=10)
        ax.set_xticks(np.arange(-.5, len(order_cfg), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(drugs), 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.5)
        ax.tick_params(which="minor", length=0)
        ax.set_title("Fig B · Δ CV-AUC vs leak-corrected SD-CNN, per drug × config\n",
                     loc="left")
        ax.text(0, 1.004, "blue = we beat the baseline · drugs ordered by baseline "
                "difficulty (hardest at top)", transform=ax.transAxes, fontsize=10,
                color="0.45", va="bottom")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.015).set_label("Δ CV-AUC")
        fig.tight_layout()
    save_fig(fig, "figB_per_drug_vs_baseline")
    plt.show()

    won = (delta > 0).sum().sort_values(ascending=False)
    print("configs beating the baseline on the most drugs:")
    for cfg, n in won.head(5).items():
        print(f"  {cfg.replace(chr(10), ' / '):<42} {n}/{delta[cfg].notna().sum()} drugs")''')

# ---------------------------------------------------------------- fig C1
code(r'''# === FIGURE C1 — ONE fixed config vs the baseline, all 11 drugs =============
# The defensible claim: pick the best COMPLETE cell by mean Δ (one configuration,
# same for every drug, chosen once) and show it drug by drug. No per-drug
# selection anywhere, so this is what we would actually deploy.
if SD.empty:
    print("No single-drug results yet.")
else:
    cand = []
    for (m, a), g in SD.groupby(["modset", "arch"]):
        d = g.set_index("drug")
        drugs = [x for x in d.index if x in SD_BASE_CV.index]
        if len(drugs) < len(ALL_DRUGS):
            continue
        cand.append(((d.loc[drugs, "cv_auc_mean"] - SD_BASE_CV[drugs]).mean(), m, a, d))
    if not cand:
        print("No cell has all 11 drugs yet — see the coverage table.")
    else:
        dlt, m, a, d = max(cand, key=lambda t: t[0])
        drugs = [x for x in SD_BASE_CV.sort_values().index if x in d.index]
        ours, sdv = d.loc[drugs, "cv_auc_mean"], d.loc[drugs, "cv_auc_std"]
        theirs = SD_BASE_CV[drugs]
        with mpl.rc_context(RC):
            fig, ax = plt.subplots(figsize=(12.5, 0.52 * len(drugs) + 3.6))
            y = np.arange(len(drugs))[::-1]
            for yi, dr in zip(y, drugs):
                sd_b = SDCNN_CV_SD.get(dr, np.nan)
                if not pd.isna(sd_b):        # baseline's own fold-to-fold spread
                    ax.plot([theirs[dr] - sd_b, theirs[dr] + sd_b], [yi, yi],
                            color="0.86", lw=7, solid_capstyle="butt", zorder=1)
                if not pd.isna(sdv[dr]):     # ours
                    ax.plot([ours[dr] - sdv[dr], ours[dr] + sdv[dr]], [yi, yi],
                            color=BLUE, alpha=0.30, lw=7, solid_capstyle="butt", zorder=2)
                c = BLUE if ours[dr] >= theirs[dr] else VERM
                ax.annotate("", xy=(ours[dr], yi), xytext=(theirs[dr], yi), zorder=3,
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2,
                                            shrinkA=0, shrinkB=0, mutation_scale=14))
                ax.text(1.004, yi, f"{ours[dr] - theirs[dr]:+.3f}",
                        transform=ax.get_yaxis_transform(), va="center", ha="left",
                        fontsize=9.5, fontweight="bold", color=c)
            ax.scatter(theirs.values, y, s=95, facecolor="white", edgecolor=ORANGE,
                       lw=2.4, zorder=5)
            ax.scatter(ours.values, y, s=95, zorder=6, edgecolor="white", lw=1.2,
                       color=[BLUE if ours[x] >= theirs[x] else VERM for x in drugs])
            ax.set_yticks(y, drugs, fontsize=10)
            ax.set_xlabel("Cross-validated AUC (5-fold mean, ±1 SD band)")
            ax.set_xlim(float(min(ours.min(), theirs.min())) - 0.05, 1.02)
            n_up = int((ours >= theirs).sum())
            ax.set_title(f"Fig C1 · one fixed configuration beats the corrected "
                         f"SD-CNN on {n_up}/{len(drugs)} drugs\n"
                         f"{MODSET_LABEL[m]} / {ARCH_SHORT[a]}, single-drug\n",
                         loc="left", fontsize=13)
            ax.text(0, 1.006, f"mean Δ {dlt:+.4f} · no per-drug selection: the same "
                    "config is used for every drug · grey band = baseline's own CV "
                    "spread, blue band = ours", transform=ax.transAxes, fontsize=9.5,
                    color="0.45", va="bottom")
            ax.grid(axis="x", alpha=0.25)
            ax.set_axisbelow(True)
            fig.legend(handles=[
                Line2D([], [], marker="o", ls="", mfc="white", mec=ORANGE, mew=2.4,
                       ms=10, label="BIG-TB SD-CNN (leak-corrected CV)"),
                Line2D([], [], marker="o", ls="", color=BLUE, ms=10, label="ours — ahead"),
                Line2D([], [], marker="o", ls="", color=VERM, ms=10, label="ours — behind"),
            ], loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3, fontsize=10)
            fig.tight_layout(rect=(0, 0.075, 1, 1))
        save_fig(fig, "figC1_best_complete_config")
        plt.show()''')

# ---------------------------------------------------------------- fig C
code(r'''# === FIGURE C — best config per drug vs the baseline (dumbbell) ============
# Picks, per drug, our best CV config across the whole grid — a selection-biased
# ceiling, so read it as "is the grid capable of beating the baseline anywhere",
# not as a single deployable model.
if SD.empty:
    print("No single-drug results yet.")
else:
    best = (SD.sort_values("cv_auc_mean", ascending=False)
              .drop_duplicates("drug").set_index("drug"))
    drugs = [d for d in SD_BASE_CV.sort_values().index if d in best.index]
    ours = best.loc[drugs, "cv_auc_mean"]
    theirs = SD_BASE_CV[drugs]
    with mpl.rc_context(RC):
        fig, ax = plt.subplots(figsize=(11.5, 0.52 * len(drugs) + 3))
        y = np.arange(len(drugs))[::-1]
        for yi, d in zip(y, drugs):
            sd_b = SDCNN_CV_SD.get(d, np.nan)
            if not pd.isna(sd_b):
                ax.plot([theirs[d] - sd_b, theirs[d] + sd_b], [yi, yi], color="0.82",
                        lw=6, solid_capstyle="butt", zorder=1)
            c = BLUE if ours[d] >= theirs[d] else VERM
            ax.annotate("", xy=(ours[d], yi), xytext=(theirs[d], yi), zorder=3,
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=2.4,
                                        shrinkA=0, shrinkB=0, mutation_scale=15))
            lbl = f"{ours[d] - theirs[d]:+.3f}   {MODSET_LABEL[best.loc[d, 'modset']]} / {ARCH_SHORT[best.loc[d, 'arch']]}"
            ax.text(1.004, yi, lbl, transform=ax.get_yaxis_transform(), va="center",
                    ha="left", fontsize=9, fontweight="bold", color=c)
        ax.scatter(theirs.values, y, s=95, facecolor="white", edgecolor=ORANGE,
                   lw=2.4, zorder=5)
        ax.scatter(ours.values, y, s=95, zorder=6, edgecolor="white", lw=1.2,
                   color=[BLUE if ours[d] >= theirs[d] else VERM for d in drugs])
        ax.set_yticks(y, drugs, fontsize=10)
        ax.set_xlabel("Cross-validated AUC (5-fold mean)")
        ax.set_xlim(float(min(ours.min(), theirs.min())) - 0.04, 1.02)
        n_up = int((ours >= theirs).sum())
        ax.set_title(f"Fig C · our best config beats the corrected SD-CNN on "
                     f"{n_up} of {len(drugs)} drugs\n", loc="left")
        ax.text(0, 1.006, "grey band = ±1 SD across the baseline's own CV folds · "
                "label = Δ and which config won",
                transform=ax.transAxes, fontsize=10, color="0.45", va="bottom")
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        fig.legend(handles=[
            Line2D([], [], marker="o", ls="", mfc="white", mec=ORANGE, mew=2.4, ms=10,
                   label="BIG-TB SD-CNN (leak-corrected CV)"),
            Line2D([], [], marker="o", ls="", color=BLUE, ms=10, label="ours — ahead"),
            Line2D([], [], marker="o", ls="", color=VERM, ms=10, label="ours — behind"),
        ], loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=10)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_fig(fig, "figC_best_vs_sdcnn")
    plt.show()''')

# ---------------------------------------------------------------- md section
md(r"""---

## Joint models vs the published BIG-TB MD-CNN

Every cell above is 11 separate models. Below, each config is **one**
`MultiDrugNet` / `MDCNNNet` / `SetFusionNet` / `CisFusionNet` predicting all 11
drugs at once — the same setting as BIG-TB's **MD-CNN (OHE)**, which is therefore
the baseline here (not SD-CNN).

**CV is the matched comparison** (their crossval and our `run_multidrug_cv` both
use a non-stratified 80/20 seed-42 split + `KFold(5)` on the train portion). Their
test cohort is a predefined set, ours a random 20% hold-out, so test-vs-test is
indicative only. ETHIONAMIDE's MD-CNN CV baseline is flagged ⚠ and excluded from
the "excl. flagged" summaries.""")

code(r'''# === TABLE B + FIGURE D — joint leaderboard and grid ========================
if MD.empty:
    print("No joint results yet.")
else:
    per_drug = MD[MD.drug != "MACRO"]
    rows = []
    for (m, a), g in per_drug.groupby(["modset", "arch"]):
        d = g.set_index("drug")
        drugs = [x for x in d.index if x in MD_BASE_CV.index]
        ok = [x for x in drugs if x not in MDCNN_CV_SUSPECT]
        rows.append({
            "modality set": MODSET_LABEL[m], "architecture": ARCH_SHORT[a],
            "drugs": len(drugs),
            "macro CV": d.loc[drugs, "cv_auc"].mean(),
            "Δ CV vs MD-CNN": (d.loc[drugs, "cv_auc"] - MD_BASE_CV[drugs]).mean(),
            "Δ CV (excl ⚠)": (d.loc[ok, "cv_auc"] - MD_BASE_CV[ok]).mean(),
            "drugs ahead (CV)": int((d.loc[drugs, "cv_auc"] > MD_BASE_CV[drugs]).sum()),
            "macro test": d.loc[drugs, "test_auc"].mean(),
            "Δ test vs MD-CNN": (d.loc[drugs, "test_auc"] - MD_BASE_TEST[drugs]).mean(),
            "params": param_str(f"multidrug_{m}__{a}"),
            "note": is_degenerate(a, m), "_m": m, "_a": a,
        })
    mlb = pd.DataFrame(rows).sort_values("macro CV", ascending=False)
    display(mlb.drop(columns=["_m", "_a"]).set_index(["modality set", "architecture"])
            .style.format({"macro CV": "{:.4f}", "macro test": "{:.4f}",
                           "Δ CV vs MD-CNN": "{:+.4f}", "Δ CV (excl ⚠)": "{:+.4f}",
                           "Δ test vs MD-CNN": "{:+.4f}"}, na_rep="—")
            .apply(lambda r: [DEGEN_BG if r["note"] else "" for _ in r], axis=1)
            .apply(colour_delta, subset=["Δ CV vs MD-CNN"])
            .set_table_styles(TABLE_STYLES)
            .set_caption("<b>Joint-model leaderboard.</b> One model per row, all 11 "
                         "drugs. Δ vs the published MD-CNN CV on the same drugs; "
                         "'excl ⚠' drops ETHIONAMIDE, whose baseline CV value the "
                         "paper contradicts. Greyed = degenerate cell."))

    jg = pd.DataFrame(index=MODSET_ORDER, columns=ARCH_ORDER, dtype=float)
    jd = jg.copy()
    for (m, a), g in per_drug.groupby(["modset", "arch"]):
        d = g.set_index("drug")
        drugs = [x for x in d.index if x in MD_BASE_CV.index]
        jg.loc[m, a] = d.loc[drugs, "cv_auc"].mean()
        jd.loc[m, a] = (d.loc[drugs, "cv_auc"] - MD_BASE_CV[drugs]).mean()

    with mpl.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4))
        for ax, data, title, cmap, fmt, div in [
            (axes[0], jg, "Macro CV-AUC (mean over 11 drugs)", "Blues", "{:.3f}", False),
            (axes[1], jd, "Δ macro CV vs published MD-CNN", "RdBu", "{:+.3f}", True)]:
            vals = data.values.astype(float)
            norm = None
            if div and np.isfinite(vals).any():
                lim = float(np.nanmax(np.abs(vals))) or 0.01
                norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
            im = ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")
            for i, m in enumerate(MODSET_ORDER):
                for j, a in enumerate(ARCH_ORDER):
                    v = vals[i, j]
                    ax.text(j, i, "—" if not np.isfinite(v) else fmt.format(v),
                            ha="center", va="center", fontsize=11, fontweight="bold",
                            color="0.12" if np.isfinite(v) else "0.6")
                    if is_degenerate(a, m) and np.isfinite(v):
                        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                                   hatch="///", edgecolor="0.75", lw=0))
            ax.set_xticks(range(len(ARCH_ORDER)),
                          [ARCH_LABEL[a] for a in ARCH_ORDER], fontsize=9.5)
            ax.set_yticks(range(len(MODSET_ORDER)),
                          [MODSET_LABEL[m] for m in MODSET_ORDER], fontsize=10.5)
            ax.set_xticks(np.arange(-.5, len(ARCH_ORDER), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(MODSET_ORDER), 1), minor=True)
            ax.grid(which="minor", color="white", lw=2)
            ax.tick_params(which="minor", length=0)
            ax.set_title(title + "\n", loc="left")
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        axes[0].text(0, 1.012, f"MD-CNN macro CV baseline: {MD_BASE_CV.mean():.3f}",
                     transform=axes[0].transAxes, fontsize=10, color="0.45", va="bottom")
        fig.suptitle("Fig D · joint models: architecture × modality set",
                     x=0.01, ha="left", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "figD_grid_joint")
    plt.show()''')

# ---------------------------------------------------------------- fig D2
code(r'''# === FIGURE D2 — the architecture ranking FLIPS between scopes ==============
# Mean CV over modality sets, single-drug vs joint, same architectures. mdcnn is
# the best single-drug topology and among the worst joint ones; late_fusion the
# reverse. Slope = how much a topology gains from being trained jointly.
if SD.empty or MD.empty:
    print("Need both scopes.")
else:
    pdm = MD[MD.drug != "MACRO"]
    sing, joint = {}, {}
    for a in ARCH_ORDER:
        sv = [g.set_index("drug").loc[[x for x in g.drug if x in SD_BASE_CV.index],
                                      "cv_auc_mean"].mean()
              for (m, aa), g in SD.groupby(["modset", "arch"]) if aa == a]
        jv = [g.set_index("drug").loc[[x for x in g.drug if x in MD_BASE_CV.index],
                                      "cv_auc"].mean()
              for (m, aa), g in pdm.groupby(["modset", "arch"]) if aa == a]
        if sv:
            sing[a] = float(np.nanmean(sv))
        if jv:
            joint[a] = float(np.nanmean(jv))
    archs = [a for a in ARCH_ORDER if a in sing and a in joint]

    def _spread(vals, gap):
        """Nudge label positions apart so near-equal series don't overprint
        (late_fusion and cisfusion land within 0.001 on the single-drug side)."""
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = list(vals)
        for k in range(1, len(order)):
            lo, hi = order[k - 1], order[k]
            if out[hi] - out[lo] < gap:
                out[hi] = out[lo] + gap
        return out

    with mpl.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9.5, 6))
        palette = {"late_fusion": BLUE, "mdcnn": ORANGE,
                   "setfusion": GREY, "cisfusion": VERM}
        span = max(list(sing.values()) + list(joint.values())) - \
            min(list(sing.values()) + list(joint.values()))
        gap = 0.045 * max(span, 1e-6)
        ls = _spread([sing[a] for a in archs], gap)
        lj = _spread([joint[a] for a in archs], gap)
        for a, ys, yj in zip(archs, ls, lj):
            c = palette.get(a, INK)
            ax.plot([0, 1], [sing[a], joint[a]], "-o", color=c, lw=2.6, ms=10,
                    zorder=3, markeredgecolor="white", markeredgewidth=1.2)
            ax.annotate(f"{ARCH_SHORT[a]}  {sing[a]:.3f}", xy=(0, sing[a]),
                        xytext=(-0.045, ys), ha="right", va="center", fontsize=10.5,
                        fontweight="bold", color=c,
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.8, alpha=0.45,
                                        shrinkA=0, shrinkB=6))
            ax.annotate(f"{joint[a]:.3f}  {ARCH_SHORT[a]}", xy=(1, joint[a]),
                        xytext=(1.045, yj), ha="left", va="center", fontsize=10.5,
                        fontweight="bold", color=c,
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.8, alpha=0.45,
                                        shrinkA=0, shrinkB=6))
        ax.axhline(SD_BASE_CV.mean(), ls="--", lw=1.3, color="0.55", zorder=1)
        ax.text(0.02, SD_BASE_CV.mean(), " SD-CNN baseline", fontsize=9,
                color="0.45", va="bottom")
        ax.axhline(MD_BASE_CV.mean(), ls=":", lw=1.3, color="0.55", zorder=1)
        ax.text(0.98, MD_BASE_CV.mean(), "MD-CNN baseline ", fontsize=9,
                color="0.45", va="bottom", ha="right")
        ax.set_xticks([0, 1], ["single-drug\n(11 models)", "joint\n(1 model)"],
                      fontsize=11)
        ax.set_xlim(-0.70, 1.70)
        ax.set_ylabel("Mean CV-AUC over the 5 modality sets")
        ax.set_title("Fig D2 · the best architecture is not the same in both scopes\n",
                     loc="left")
        ax.text(0, 1.006, "mdcnn leads single-drug and trails joint; late_fusion the "
                "reverse. setfusion is depressed by the early-stopping artifact.",
                transform=ax.transAxes, fontsize=9.5, color="0.45", va="bottom")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines["bottom"].set_visible(False)
        fig.tight_layout()
    save_fig(fig, "figD2_arch_rank_flip")
    plt.show()''')

# ---------------------------------------------------------------- fig E
code(r'''# === FIGURE E — joint vs single-drug on IDENTICAL inputs ====================
# For every (modality set, architecture) that has finished in both scopes: does
# one model predicting 11 drugs beat 11 separate models? Positive = joint wins.
if MD.empty or SD.empty:
    print("Need both scopes to compare.")
else:
    pd_md = MD[MD.drug != "MACRO"]
    pairs = []
    for (m, a), g in pd_md.groupby(["modset", "arch"]):
        s = SD[(SD.modset == m) & (SD.arch == a)]
        if s.empty:
            continue
        j = g.set_index("drug")["cv_auc"]
        k = s.set_index("drug")["cv_auc_mean"]
        common = [d for d in j.index if d in k.index]
        if not common:
            continue
        pairs.append({"cfg": f"{MODSET_LABEL[m]} / {ARCH_SHORT[a]}",
                      "joint": j[common].mean(), "single": k[common].mean(),
                      "Δ joint − single": (j[common] - k[common]).mean(),
                      "drugs": len(common),
                      "drugs joint wins": int((j[common] > k[common]).sum()),
                      "note": is_degenerate(a, m)})
    if not pairs:
        print("No matched pairs finished yet.")
    else:
        P = pd.DataFrame(pairs).sort_values("Δ joint − single")
        with mpl.rc_context(RC):
            fig, ax = plt.subplots(figsize=(10.5, 0.46 * len(P) + 2.8))
            y = np.arange(len(P))
            v = P["Δ joint − single"].values
            ax.barh(y, v, height=0.66, zorder=2,
                    color=[BLUE if x >= 0 else VERM for x in v])
            ax.axvline(0, color="0.25", lw=1.2, zorder=3)
            m_ = float(np.nanmax(np.abs(v))) or 0.01
            for yi, (_, r) in zip(y, P.iterrows()):
                ax.text(r["Δ joint − single"] + 0.03 * m_ * (1 if r["Δ joint − single"] >= 0 else -1),
                        yi, f"{r['Δ joint − single']:+.3f}  ({r['drugs joint wins']}/{r['drugs']} drugs)",
                        va="center", ha="left" if r["Δ joint − single"] >= 0 else "right",
                        fontsize=9, color="0.2")
            ax.set_yticks(y, P["cfg"], fontsize=9.5)
            ax.set_xlim(-2.1 * m_, 2.1 * m_)
            ax.set_xlabel("Δ mean CV-AUC  (joint − single-drug)")
            ax.set_title("Fig E · one joint model vs 11 single-drug models, same inputs\n",
                         loc="left")
            ax.text(0, 1.01, "blue = joint training wins on average for that config",
                    transform=ax.transAxes, fontsize=10, color="0.45", va="bottom")
            ax.grid(axis="x", alpha=0.25)
            ax.set_axisbelow(True)
            fig.tight_layout()
        save_fig(fig, "figE_joint_vs_single")
        plt.show()''')

# ---------------------------------------------------------------- fig E2
code(r'''# === FIGURE E2 — who gains from joint training, against sample size ========
# Per-drug (joint - single) against the drug's sample size. Multi-task sharing
# should pay most where a single-drug model has least data to fit — and it does.
if SD.empty or MD.empty:
    print("Need both scopes.")
else:
    pdm = MD[MD.drug != "MACRO"]
    n_by_drug = SD.groupby("drug")["n_valid"].max()
    recs = []
    for (m, a), g in pdm.groupby(["modset", "arch"]):
        s = SD[(SD.modset == m) & (SD.arch == a)].set_index("drug")["cv_auc_mean"]
        j = g.set_index("drug")["cv_auc"]
        for dr in [x for x in j.index if x in s.index]:
            recs.append({"drug": dr, "arch": a, "d": j[dr] - s[dr]})
    R = pd.DataFrame(recs)
    # setfusion's single-drug reference is the artifact — exclude from the trend
    Rc = R[R.arch != "setfusion"] if (R.arch != "setfusion").any() else R
    agg = Rc.groupby("drug")["d"].agg(["mean", "std", "count"])
    agg["n"] = n_by_drug.reindex(agg.index)
    agg = agg.sort_values("n")
    TOO_SMALL = 500          # below this a 5-fold CV AUC is not a measurement
    with mpl.rc_context(RC):
        fig, ax = plt.subplots(figsize=(11.5, 6.4))
        ax.axhline(0, color="0.25", lw=1.2, zorder=2)
        # stagger labels so the 3-4k cluster (AMK/KAN/CAP) doesn't overprint
        placed = []
        for dr, r in agg.iterrows():
            small = r["n"] < TOO_SMALL
            c = GREY if small else (BLUE if r["mean"] >= 0 else VERM)
            ax.errorbar(r["n"], r["mean"], yerr=(r["std"] if r["count"] > 1 else 0),
                        fmt="o", ms=11, color=c, ecolor=c, elinewidth=1.6, capsize=4,
                        alpha=0.45 if small else 1.0, markeredgecolor="white",
                        markeredgewidth=1.2, zorder=4)
            dy = 15
            while any(abs(np.log10(r["n"]) - lx) < 0.10 and abs(dy - ly) < 15
                      for lx, ly in placed):
                dy += 16
            placed.append((np.log10(r["n"]), dy))
            ax.annotate(dr + (" (too small)" if small else ""), (r["n"], r["mean"]),
                        textcoords="offset points", xytext=(0, dy), ha="center",
                        fontsize=8.5, color="0.45" if small else "0.25")
        ax.axvspan(agg["n"].min() * 0.7, TOO_SMALL, color="0.92", zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel("Isolates with a phenotype for that drug (log scale)")
        ax.set_ylabel("Δ CV-AUC   (joint − single-drug)")
        ax.set_title("Fig E2 · who gains from joint training\n", loc="left")
        ax.text(0, 1.006, "point = mean over the sound architectures (setfusion "
                "excluded), bar = spread across them · above 0 = the joint model wins",
                transform=ax.transAxes, fontsize=9.5, color="0.45", va="bottom")
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        fig.tight_layout()
    save_fig(fig, "figE2_joint_gain_vs_n")
    plt.show()

    print("Among drugs with enough data to measure, the joint gain shrinks as n "
          "grows: the shared\ntrunk substitutes for data the single-drug model does "
          "not have. LEVOFLOXACIN (grey band,\nn=269, ~15 resistant per fold) is too "
          "small for either model to fit and is not evidence\nagainst that — it is "
          "why the band is drawn.\n")
    meas = agg[agg["n"] >= TOO_SMALL]
    if len(meas) > 2:
        rho = float(np.corrcoef(np.log10(meas["n"]), meas["mean"])[0, 1])
        print(f"Among the {len(meas)} measurable drugs, corr(log n, joint gain) = "
              f"{rho:+.2f}  ({int((meas['mean'] > 0).sum())}/{len(meas)} gain from "
              "joint training).")
    print(f"Excluded: {', '.join(agg[agg['n'] < TOO_SMALL].index)} "
          f"(n < {TOO_SMALL}).")''')

# ---------------------------------------------------------------- follow-ups
md(r"""---

## Follow-up runs

Two things this sweep could not answer are being re-run. The cell below picks
them up automatically as jobs land; it prints a note and nothing else until then.

| run folder | question it settles |
|---|---|
| `setfusion_warmup/` | Is setfusion's 0.78 real, or the early-stopping artifact? Same grid, `--min-epochs 50`. |
| `alllocus_run/` | Is "joint wins" multi-task sharing, or just the bigger 19-locus input? Same single-drug grid on the joint locus set. |
""")

code(r'''# === FOLLOW-UPS — setfusion_warmup / alllocus_run, if they have landed ======
EXP = PROJECT / "results/experiments"


def _load_single(root):
    """{(modset, arch): DataFrame} for a run root laid out like full_run."""
    out = {}
    if not root.is_dir():
        return out
    for p in sorted(root.iterdir()):
        if not p.is_dir() or not (p / "summary.csv").exists():
            continue
        parsed = parse_folder(p.name)
        if parsed is None or parsed[0] != "single":
            continue
        s = pd.read_csv(p / "summary.csv")
        s["drug"] = s["drug"].str.upper()
        out[(parsed[1], parsed[2])] = s.set_index("drug")
    return out


# --- setfusion warmup -------------------------------------------------------
warm = _load_single(EXP / "setfusion_warmup")
if not warm:
    print("setfusion_warmup/: nothing yet.")
else:
    rows = []
    for (m, a), d in sorted(warm.items()):
        old = SD[(SD.modset == m) & (SD.arch == a)].set_index("drug")["cv_auc_mean"]
        common = [x for x in d.index if x in old.index]
        if not common:
            continue
        new = d.loc[common, "cv_auc_mean"]
        rows.append({"modality set": MODSET_LABEL[m], "arch": ARCH_SHORT[a],
                     "drugs done": len(common),
                     "orig CV": old[common].mean(), "warmup CV": new.mean(),
                     "Δ": (new - old[common]).mean(),
                     "improved": f"{int((new > old[common]).sum())}/{len(common)}"})
    if rows:
        W = pd.DataFrame(rows).set_index(["modality set", "arch"])
        display(W.style.format({"orig CV": "{:.4f}", "warmup CV": "{:.4f}",
                                "Δ": "{:+.4f}"})
                .apply(colour_delta, subset=["Δ"])
                .set_table_styles(TABLE_STYLES)
                .set_caption("<b>setfusion with an early-stopping warmup "
                             "(--min-epochs 50).</b> Matched per cell to the drugs "
                             "that have finished. A positive Δ means the original "
                             "full_run number was an early-stopping artifact, not "
                             "the architecture's ceiling."))

# --- all-locus single-drug --------------------------------------------------
allloc = _load_single(EXP / "alllocus_run")
if not allloc:
    print("alllocus_run/: nothing yet.")
elif MD.empty:
    print("alllocus_run/: need joint results to decompose against.")
else:
    pdm = MD[MD.drug != "MACRO"]
    rows = []
    for (m, a), d in sorted(allloc.items()):
        old = SD[(SD.modset == m) & (SD.arch == a)].set_index("drug")["cv_auc_mean"]
        jg = pdm[(pdm.modset == m) & (pdm.arch == a)].set_index("drug")["cv_auc"]
        common = [x for x in d.index if x in old.index and x in jg.index]
        if not common:
            continue
        new = d.loc[common, "cv_auc_mean"]
        loci_effect = (new - old[common]).mean()            # more loci, still single
        task_effect = (jg[common] - new).mean()             # same loci, now joint
        rows.append({"modality set": MODSET_LABEL[m], "arch": ARCH_SHORT[a],
                     "drugs done": f"{len(common)}/{len(ALL_DRUGS)}"
                                   + ("" if len(common) == len(ALL_DRUGS) else " ⚠"),
                     "total Δ (joint − 18-locus single)": (jg[common] - old[common]).mean(),
                     "…from more loci": loci_effect,
                     "…from joint training": task_effect})
    if rows:
        A = pd.DataFrame(rows).set_index(["modality set", "arch"])
        display(A.style.format({c: "{:+.4f}" for c in A.columns if c != "drugs done"})
                .apply(colour_delta, subset=["…from more loci", "…from joint training"])
                .set_table_styles(TABLE_STYLES)
                .set_caption("<b>Decomposing the joint-vs-single gain.</b> The total "
                             "splits into the part explained by giving the "
                             "single-drug model the same 19 loci, and the part left "
                             "over for multi-task training itself. Fig E's headline "
                             "conflates the two. ⚠ = the cell is still running, so "
                             "its split is averaged over only the drugs that have "
                             "landed and will move."))''')

# ---------------------------------------------------------------- headline
code(r'''# === HEADLINE ===============================================================
print("SINGLE-DRUG  vs  BIG-TB SD-CNN (leak-corrected CV)")
print("=" * 78)
if SD.empty:
    print("  nothing finished yet")
else:
    best_cfg = None
    for (m, a), g in SD.groupby(["modset", "arch"]):
        d = g.set_index("drug")
        drugs = [x for x in d.index if x in SD_BASE_CV.index]
        if len(drugs) < 11:                     # complete cells only for the headline
            continue
        delta = (d.loc[drugs, "cv_auc_mean"] - SD_BASE_CV[drugs]).mean()
        if best_cfg is None or delta > best_cfg[0]:
            best_cfg = (delta, m, a, d.loc[drugs, "cv_auc_mean"].mean(),
                        int((d.loc[drugs, "cv_auc_mean"] > SD_BASE_CV[drugs]).sum()))
    if best_cfg is None:
        print("  no cell has all 11 drugs yet — see the coverage table")
    else:
        delta, m, a, mean_cv, ahead = best_cfg
        print(f"  best complete cell: {MODSET_LABEL[m]} / {ARCH_SHORT[a]}")
        print(f"    mean CV {mean_cv:.4f}  vs baseline {SD_BASE_CV.mean():.4f}"
              f"   Δ {delta:+.4f}   {ahead}/11 drugs ahead")
        if is_degenerate(a, m):
            print(f"    NOTE: degenerate cell ({is_degenerate(a, m)})")

print("\nJOINT  vs  BIG-TB MD-CNN (published CV)")
print("=" * 78)
if MD.empty:
    print("  nothing finished yet")
else:
    pdm = MD[MD.drug != "MACRO"]
    for (m, a), g in sorted(pdm.groupby(["modset", "arch"]),
                            key=lambda kv: -kv[1].set_index("drug")["cv_auc"].mean()):
        d = g.set_index("drug")
        drugs = [x for x in d.index if x in MD_BASE_CV.index]
        ok = [x for x in drugs if x not in MDCNN_CV_SUSPECT]
        print(f"  {MODSET_LABEL[m]:<16} {ARCH_SHORT[a]:<12} "
              f"macro CV {d.loc[drugs, 'cv_auc'].mean():.4f}"
              f"   Δ {(d.loc[drugs, 'cv_auc'] - MD_BASE_CV[drugs]).mean():+.4f}"
              f"   (excl ⚠ {(d.loc[ok, 'cv_auc'] - MD_BASE_CV[ok]).mean():+.4f})"
              f"   {int((d.loc[drugs, 'cv_auc'] > MD_BASE_CV[drugs]).sum())}/{len(drugs)} ahead"
              + ("   [degenerate]" if is_degenerate(a, m) else ""))

print("\nCaveats: CV is the matched, leak-free comparison on both sides. "
      "Single-drug uses\nthe 18-locus per-drug set (no fabG1); joint uses all 19 "
      "on disk — INH/ETO are not\nlocus-matched across scopes. LEVOFLOXACIN has "
      "269 phenotyped isolates, so every\nLEVO number on either side is noise.")''')

# ============================================================================
# THE NARRATIVE CELLS (§1–§5). Authored here, moved into reading order by the
# READING_ORDER permutation at the bottom of this file.
# ============================================================================

# ---------------------------------------------------------------- §1
md(r"""---

# 1 · With DNA alone, do we beat the single-drug baseline?

**The controlled question first.** Before any of our modalities enter, give our
architectures exactly what BIG-TB's SD-CNN gets — one-hot DNA over the drug's own
loci — and ask whether the model and training protocol alone are worth anything.

This is the fairest comparison in the whole sweep, and one cell makes it airtight:
**`dna / mdcnn` is BIG-TB's own SD-CNN topology re-implemented**, so that row is
matched on inputs *and* on architecture. Whatever it gains is protocol, not
modelling.""")

code(r'''# === FIGURE 1 — single-drug, DNA ONLY, vs the leak-corrected SD-CNN ========
def _baseline_bar(ax, deltas, ahead, n_drugs, title, sub, note_map):
    """Horizontal Δ-vs-baseline bars for one modality set, one bar per arch."""
    order = sorted(deltas, key=lambda a: deltas[a])
    y = np.arange(len(order))
    vals = [deltas[a] for a in order]
    cols = [GREY if note_map.get(a) else (BLUE if v >= 0 else VERM)
            for a, v in zip(order, vals)]
    ax.barh(y, vals, height=0.62, color=cols, zorder=3)
    ax.axvline(0, color="0.2", lw=1.4, zorder=4)
    m = max(abs(v) for v in vals) or 0.01
    for yi, a, v in zip(y, order, vals):
        lbl = f"{v:+.4f}   {ahead[a]}/{n_drugs} drugs ahead"
        if note_map.get(a):
            lbl += f"   — {note_map[a]}"
        ax.text(v + 0.04 * m * (1 if v >= 0 else -1), yi, lbl, va="center",
                ha="left" if v >= 0 else "right", fontsize=9.5,
                color="0.45" if note_map.get(a) else "0.15",
                fontweight="normal" if note_map.get(a) else "bold")
    ax.set_yticks(y, [ARCH_SHORT[a] for a in order], fontsize=11)
    ax.set_xlim(-2.3 * m, 2.3 * m)
    ax.set_title(title + "\n", loc="left")
    ax.text(0, 1.006, sub, transform=ax.transAxes, fontsize=9.5, color="0.45",
            va="bottom")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)


if SD.empty:
    print("No single-drug results yet.")
else:
    dna_sd = SD[SD.modset == REFERENCE_MODSET]
    deltas, ahead, notes, table = {}, {}, {}, []
    for a, g in dna_sd.groupby("arch"):
        d = g.set_index("drug")
        drugs = [x for x in d.index if x in SD_BASE_CV.index]
        cv = d.loc[drugs, "cv_auc_mean"]
        deltas[a] = (cv - SD_BASE_CV[drugs]).mean()
        ahead[a] = int((cv > SD_BASE_CV[drugs]).sum())
        if a == "setfusion":
            notes[a] = "early-stopping artifact, see §5"
        table.append({"architecture": ARCH_SHORT[a], "CV (DNA only)": cv.mean(),
                      "SD-CNN": SD_BASE_CV[drugs].mean(), "Δ": deltas[a],
                      "drugs ahead": f"{ahead[a]}/{len(drugs)}",
                      "note": notes.get(a, "")})
    n_drugs = int(dna_sd.groupby("arch").size().max())
    with mpl.rc_context(RC):
        fig, ax = plt.subplots(figsize=(11, 4.4))
        _baseline_bar(ax, deltas, ahead, n_drugs,
                      "Fig 1 · single-drug, DNA only — Δ CV-AUC vs the "
                      "leak-corrected SD-CNN",
                      f"same inputs as the baseline (each drug's own loci, one-hot "
                      f"DNA) · baseline mean CV {SD_BASE_CV.mean():.4f} · "
                      "blue = we are ahead", notes)
        ax.set_xlabel("Δ CV-AUC   (ours − SD-CNN, matched per drug)")
        fig.tight_layout()
    save_fig(fig, "fig1_step1_dna_single_vs_sdcnn")
    plt.show()

    T = pd.DataFrame(table).sort_values("Δ", ascending=False).set_index("architecture")
    display(T.style.format({"CV (DNA only)": "{:.4f}", "SD-CNN": "{:.4f}",
                            "Δ": "{:+.4f}"})
            .apply(colour_delta, subset=["Δ"])
            .set_table_styles(TABLE_STYLES)
            .set_caption("<b>Step 1.</b> DNA-only, single-drug. <code>mdcnn</code> "
                         "is BIG-TB's own SD-CNN topology, so its row is matched on "
                         "architecture as well as inputs — read that Δ as the value "
                         "of the training protocol alone."))
    best = max(deltas, key=lambda a: deltas[a])
    extra = ""
    if best != "mdcnn" and "mdcnn" in deltas:
        extra = (f"\nOn the architecture-matched cell (mdcnn, BIG-TB's own topology) "
                 f"the gap is {deltas['mdcnn']:+.4f}.")
    elif best == "mdcnn":
        extra = ("\nAnd that best cell IS BIG-TB's own topology, so the gap is "
                 "attributable to the training\nprotocol alone — not to anything "
                 "we changed about the model.")
    print(f"\nANSWER: yes — {ARCH_SHORT[best]} is {deltas[best]:+.4f} ahead of the "
          f"corrected SD-CNN on DNA alone\n({ahead[best]}/{n_drugs} drugs), before a "
          f"single extra modality is added.{extra}")''')

# ---------------------------------------------------------------- §2
md(r"""---

# 2 · With DNA alone, do we beat the multi-drug baseline?

Same question, joint scope: one model predicting all 11 drugs, DNA only, against
the published **MD-CNN**. Note the baseline is much stronger here — MD-CNN's macro
CV is 0.9248 against SD-CNN's 0.8636, because joint training helps the baseline
too.

The ⚠ caveat matters more in this section than anywhere else: MD-CNN's published
ETHIONAMIDE CV (0.9161) contradicts the authors' own `auc.csv` (0.622) and their
Table 4 (0.709). Both Δs are shown.""")

code(r'''# === FIGURE 2 — joint, DNA ONLY, vs the published MD-CNN ===================
if MD.empty:
    print("No joint results yet.")
else:
    dna_md = MD[(MD.modset == REFERENCE_MODSET) & (MD.drug != "MACRO")]
    deltas, d_ok, ahead, notes, table = {}, {}, {}, {}, []
    for a, g in dna_md.groupby("arch"):
        d = g.set_index("drug")
        drugs = [x for x in d.index if x in MD_BASE_CV.index]
        ok = [x for x in drugs if x not in MDCNN_CV_SUSPECT]
        cv = d.loc[drugs, "cv_auc"]
        deltas[a] = (cv - MD_BASE_CV[drugs]).mean()
        d_ok[a] = (d.loc[ok, "cv_auc"] - MD_BASE_CV[ok]).mean()
        ahead[a] = int((cv > MD_BASE_CV[drugs]).sum())
        if a == "setfusion":
            notes[a] = "early-stopping artifact, see §5"
        table.append({"architecture": ARCH_SHORT[a], "macro CV (DNA only)": cv.mean(),
                      "MD-CNN": MD_BASE_CV[drugs].mean(), "Δ": deltas[a],
                      "Δ excl ⚠ ETO": d_ok[a],
                      "drugs ahead": f"{ahead[a]}/{len(drugs)}",
                      "note": notes.get(a, "")})
    n_drugs = int(dna_md.groupby("arch").size().max())
    with mpl.rc_context(RC):
        fig, ax = plt.subplots(figsize=(11, 4.4))
        _baseline_bar(ax, deltas, ahead, n_drugs,
                      "Fig 2 · joint, DNA only — Δ macro CV-AUC vs the published "
                      "MD-CNN",
                      f"one model, all 11 drugs, DNA only · baseline macro CV "
                      f"{MD_BASE_CV.mean():.4f} · blue = we are ahead", notes)
        for yi, a in enumerate(sorted(deltas, key=lambda x: deltas[x])):
            ax.plot([d_ok[a]], [yi], marker="D", ms=7, color=INK, zorder=6,
                    markeredgecolor="white", markeredgewidth=1.0)
        ax.set_xlabel("Δ macro CV-AUC   (ours − MD-CNN, matched per drug)")
        ax.legend(handles=[Line2D([], [], marker="D", ls="", color=INK, ms=7,
                                  label="Δ excluding the flagged ETHIONAMIDE row")],
                  loc="lower right", fontsize=9.5)
        fig.tight_layout()
    save_fig(fig, "fig2_step2_dna_joint_vs_mdcnn")
    plt.show()

    T = pd.DataFrame(table).sort_values("Δ", ascending=False).set_index("architecture")
    display(T.style.format({"macro CV (DNA only)": "{:.4f}", "MD-CNN": "{:.4f}",
                            "Δ": "{:+.4f}", "Δ excl ⚠ ETO": "{:+.4f}"})
            .apply(colour_delta, subset=["Δ", "Δ excl ⚠ ETO"])
            .set_table_styles(TABLE_STYLES)
            .set_caption("<b>Step 2.</b> DNA-only, joint. Unlike step 1 we do NOT "
                         "clear this baseline on DNA alone — the best cell is a hair "
                         "behind, and only reaches parity once the flagged "
                         "ETHIONAMIDE baseline value is dropped."))
    best = max(deltas, key=lambda a: deltas[a])
    print(f"\nANSWER: no — on DNA alone the best joint cell ({ARCH_SHORT[best]}) is "
          f"{deltas[best]:+.4f} vs MD-CNN\n({d_ok[best]:+.4f} excluding the flagged "
          f"ETHIONAMIDE row), i.e. parity at best. Step 4 is where this changes.")''')

# ---------------------------------------------------------------- §3
md(r"""---

# 3 · What do the extra modalities add, single-drug?

Now add protein, biophysical and regulatory features on top of the same DNA and
measure the increment **against each architecture's own DNA-only cell** — so the
architecture is held fixed and only the inputs change.

Two readings, and you need both:

- **Fig 3 (the ladder)** — how far each architecture climbs as modalities are
  added, and where it crosses the baseline.
- **Fig 0 (per drug)** — *which drugs* the climb comes from. It is not spread
  evenly, and that is the most interesting result in the sweep.""")

code(r'''# === FIGURE 3 — the modality ladder, single-drug ===========================
def modality_ladder(ax, tbl, baseline, base_label, ylabel, title, sub):
    """tbl: index=modality set, columns=arch, values=CV. One line per arch, the
    baseline as a rule, and the region above it shaded — so 'where do we cross'
    is answerable at a glance. Start (DNA-only) is labelled on the left and the
    end point on the right, both vertically de-overlapped."""
    xs = [m for m in MODSET_ORDER if m in tbl.index]
    x = np.arange(len(xs))
    palette = {"late_fusion": BLUE, "mdcnn": ORANGE, "setfusion": GREY,
               "cisfusion": VERM}
    archs = [c for c in ARCH_ORDER if c in tbl.columns]
    vals = {a: [tbl.loc[m, a] for m in xs] for a in archs}
    finite = np.array([v for a in archs for v in vals[a] if np.isfinite(v)])
    lo = float(finite.min()) if finite.size else 0.8
    hi = float(finite.max()) if finite.size else 1.0
    lo, hi = min(lo, baseline) - 0.012, max(hi, baseline) + 0.014

    def _spread(ys, gap):
        order = sorted(range(len(ys)), key=lambda i: ys[i])
        out = list(ys)
        for k in range(1, len(order)):
            a_, b_ = order[k - 1], order[k]
            if out[b_] - out[a_] < gap:
                out[b_] = out[a_] + gap
        return out

    gap = 0.052 * (hi - lo)
    starts = _spread([vals[a][0] for a in archs], gap)
    ends = _spread([vals[a][-1] for a in archs], gap)

    ax.axhspan(baseline, hi, color="#e8f2e8", zorder=0)
    ax.axhline(baseline, ls="--", lw=1.6, color="0.35", zorder=2)
    ax.text(len(xs) - 1, baseline, f"{base_label} {baseline:.4f} ", fontsize=9.5,
            color="0.35", va="bottom", ha="right")
    for a, ys, ye in zip(archs, starts, ends):
        c = palette.get(a, INK)
        dim = a == "setfusion"
        ax.plot(x, vals[a], "-o", color=c, lw=2.4, ms=8, zorder=3,
                alpha=0.40 if dim else 1.0, markeredgecolor="white",
                markeredgewidth=1.1,
                label=ARCH_SHORT[a] + (" (artifact)" if dim else ""))
        for xi, yv, ytxt, ha, dx in ((0, vals[a][0], ys, "right", -0.11),
                                     (len(xs) - 1, vals[a][-1], ye, "left", 0.11)):
            if not np.isfinite(yv):
                continue
            ax.annotate(f"{yv:.3f}", xy=(xi, yv), xytext=(xi + dx, ytxt), ha=ha,
                        va="center", fontsize=9, color=c, fontweight="bold",
                        alpha=0.5 if dim else 1.0,
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.8,
                                        alpha=0.35, shrinkA=0, shrinkB=5))
        best_i = int(np.nanargmax(vals[a]))
        if best_i not in (0, len(xs) - 1) and not dim:   # star an interior winner
            ax.plot([best_i], [vals[a][best_i]], marker="*", ms=15, color=c,
                    zorder=5, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xticks(x, [MODSET_LABEL[m] for m in xs], fontsize=10)
    ax.set_xlim(-0.85, len(xs) - 0.15)
    ax.set_ylim(lo, hi)
    ax.set_ylabel(ylabel)
    ax.set_title(title + "\n", loc="left")
    ax.text(0, 1.008, sub, transform=ax.transAxes, fontsize=9, color="0.45",
            va="bottom", linespacing=1.45)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9.5, ncol=2)


def gain_table(tbl, baseline, value_name):
    """Per architecture: where DNA-only sat vs the baseline, which modality set is
    best, and how much of the final margin the modalities contributed."""
    rows = []
    for a in [c for c in ARCH_ORDER if c in tbl.columns]:
        if REFERENCE_MODSET not in tbl.index or not np.isfinite(tbl.loc[REFERENCE_MODSET, a]):
            continue
        base = tbl.loc[REFERENCE_MODSET, a]
        cand = tbl[a].dropna()
        bm = cand.idxmax()
        rows.append({
            "architecture": ARCH_SHORT[a],
            f"{value_name} DNA only": base,
            "Δ vs baseline (DNA only)": base - baseline,
            "best modality set": MODSET_LABEL[bm],
            f"{value_name} best": cand[bm],
            "modality gain": cand[bm] - base,
            "Δ vs baseline (best)": cand[bm] - baseline,
        })
    return pd.DataFrame(rows).set_index("architecture")


if SD.empty:
    print("No single-drug results yet.")
else:
    # Every point is a mean over ALL 11 drugs. A cell that has not finished all 11
    # is left as a GAP in its line rather than plotted over a smaller drug set —
    # dna_biophysical__mdcnn is missing ETHIONAMIDE (baseline 0.622, the hardest
    # drug) and read ~+0.03 too high when it was averaged over the 10 it has.
    # Dropping the drug everywhere instead would hide the largest modality effect
    # in the sweep, so we drop the point, not the drug.
    sgrid = pd.DataFrame(index=MODSET_ORDER, columns=ARCH_ORDER, dtype=float)
    incomplete = []
    for (m, a), g in SD.groupby(["modset", "arch"]):
        d = g.set_index("drug")
        drugs = [x for x in d.index if x in SD_BASE_CV.index]
        if len(drugs) < len(ALL_DRUGS):
            incomplete.append(f"{MODSET_LABEL[m]}/{ARCH_SHORT[a]} "
                              f"({len(drugs)}/{len(ALL_DRUGS)} drugs)")
            continue                      # leave NaN -> the line breaks here
        sgrid.loc[m, a] = d.loc[drugs, "cv_auc_mean"].mean()
    base_sd = SD_BASE_CV.mean()
    if incomplete:
        print("Fig 3: every point is a mean over all "
              f"{len(ALL_DRUGS)} drugs. Cells still running are shown as a GAP "
              f"in the line rather\nthan averaged over fewer drugs: "
              f"{'; '.join(incomplete)}.\n")
    with mpl.rc_context(RC):
        fig, ax = plt.subplots(figsize=(11.5, 6.2))
        modality_ladder(ax, sgrid, base_sd, "SD-CNN",
                        f"Mean CV-AUC over all {len(ALL_DRUGS)} drugs",
                        "Fig 3 · single-drug: what each modality set adds\n",
                        "green band = ahead of the leak-corrected SD-CNN · every "
                        f"point is a mean over all {len(ALL_DRUGS)} drugs\n"
                        "labels = DNA-only start and end point · ★ = an interior "
                        "best · a gap = that cell is still running")
        fig.tight_layout()
    save_fig(fig, "fig3_step3_modality_ladder_single")
    plt.show()

    G = gain_table(sgrid, base_sd, "CV")
    display(G.style.format({c: "{:+.4f}" if "Δ" in c or "gain" in c else "{:.4f}"
                            for c in G.columns if c != "best modality set"})
            .apply(colour_delta, subset=["Δ vs baseline (DNA only)", "modality gain",
                                         "Δ vs baseline (best)"])
            .set_table_styles(TABLE_STYLES)
            .set_caption("<b>Step 3.</b> Modalities are measured against the SAME "
                         "architecture's DNA-only cell, so the architecture is held "
                         "fixed and only the inputs change. The last column is the "
                         "first column plus the modality gain — that decomposition "
                         "is the point of this table."))
    core = [a for a in ARCH_ORDER if a != "setfusion" and a in sgrid.columns]
    g0 = float(np.nanmean([sgrid.loc[REFERENCE_MODSET, a] for a in core]))
    g1 = float(np.nanmean([sgrid[a].max() for a in core]))
    print(f"\nANSWER: the modalities are worth about {g1 - g0:+.4f} CV-AUC on top of "
          f"DNA (mean over the\nsound architectures), which takes the best cell from "
          f"{max(sgrid.loc[REFERENCE_MODSET, a] for a in core) - base_sd:+.4f} "
          f"to {max(sgrid[a].max() for a in core) - base_sd:+.4f} against "
          f"the SD-CNN —\nroughly doubling a margin we already had. Fig 0 shows the "
          "gain is NOT spread across drugs.")''')

# ---------------------------------------------------------------- §4
md(r"""---

# 4 · What do the extra modalities add, multi-drug?

The same ladder in the joint scope. This is where step 2's deficit gets settled —
and where the architectures stop behaving alike: the modality gain is small for
`late_fusion`, which already sees all 19 loci for every drug, and large for
`cisfusion`, which pairs promoter with CDS per locus.""")

code(r'''# === FIGURE 4 — the modality ladder, joint =================================
if MD.empty:
    print("No joint results yet.")
else:
    per = MD[MD.drug != "MACRO"]
    jgrid = pd.DataFrame(index=MODSET_ORDER, columns=ARCH_ORDER, dtype=float)
    for (m, a), g in per.groupby(["modset", "arch"]):
        d = g.set_index("drug")
        drugs = [x for x in d.index if x in MD_BASE_CV.index]
        jgrid.loc[m, a] = d.loc[drugs, "cv_auc"].mean()
    with mpl.rc_context(RC):
        fig, ax = plt.subplots(figsize=(11.5, 6.2))
        modality_ladder(ax, jgrid, MD_BASE_CV.mean(), "MD-CNN",
                        "Macro CV-AUC over the 11 drugs",
                        "Fig 4 · joint: what each modality set adds\n",
                        "green band = ahead of the published MD-CNN · all 11 drugs\n"
                        "labels = DNA-only start and end point · ★ = an interior best")
        fig.tight_layout()
    save_fig(fig, "fig4_step4_modality_ladder_joint")
    plt.show()

    G = gain_table(jgrid, MD_BASE_CV.mean(), "macro CV")
    display(G.style.format({c: "{:+.4f}" if "Δ" in c or "gain" in c else "{:.4f}"
                            for c in G.columns if c != "best modality set"})
            .apply(colour_delta, subset=["Δ vs baseline (DNA only)", "modality gain",
                                         "Δ vs baseline (best)"])
            .set_table_styles(TABLE_STYLES)
            .set_caption("<b>Step 4.</b> Same decomposition as step 3, joint scope, "
                         "against the published MD-CNN. Compare the 'modality gain' "
                         "column with step 3's: it is roughly a third the size for "
                         "late_fusion and larger for cisfusion."))
    core = [a for a in ARCH_ORDER if a != "setfusion" and a in jgrid.columns]
    ok_drugs = [d for d in MD_BASE_CV.index if d not in MDCNN_CV_SUSPECT]
    bm, ba = max(((jgrid.loc[m, a], m, a) for m in MODSET_ORDER for a in core
                  if np.isfinite(jgrid.loc[m, a])))[1:]
    d_all = per[(per.modset == bm) & (per.arch == ba)].set_index("drug")["cv_auc"]
    print(f"\nANSWER: modalities are worth {jgrid.loc[bm, ba] - jgrid.loc[REFERENCE_MODSET, ba]:+.4f} "
          f"on {ARCH_SHORT[ba]}, taking it from "
          f"{jgrid.loc[REFERENCE_MODSET, ba] - MD_BASE_CV.mean():+.4f} to "
          f"{jgrid.loc[bm, ba] - MD_BASE_CV.mean():+.4f}\nagainst MD-CNN "
          f"({(d_all[ok_drugs] - MD_BASE_CV[ok_drugs]).mean():+.4f} excluding the "
          f"flagged ETHIONAMIDE row).\nSo the modalities close step 2's deficit and "
          "reach parity — they do not produce the decisive\nwin they produce in the "
          "single-drug scope.")''')

# ---------------------------------------------------------------- §5
md(r"""---

# 5 · Takeaways

### On the models

1. **Our training protocol is worth ~+0.017 CV-AUC on its own.** `dna / mdcnn` is
   BIG-TB's SD-CNN topology on BIG-TB's inputs and still beats the corrected
   published numbers on 10/11 drugs. Nothing architectural is required to clear
   that baseline — the early-stopping and class-weighting choices do it.
2. **No architecture wins both scopes.** `mdcnn` is the best single-drug topology
   and among the worst joint ones; `late_fusion` is the reverse (Fig D2). Mixing
   every locus at layer 1 is a good prior when a drug has 2–4 relevant loci and a
   bad one when all 19 are present for all 11 drugs.
3. **`cisfusion` only earns its keep jointly.** Single-drug it tracks
   `late_fusion` to within 0.002; jointly it is the best cell in the sweep. Its
   promoter⊕CDS pairing needs the full locus set to have something to pair.
4. **`setfusion` was mis-evaluated, and is still last once fixed.** The
   early-stopping artifact was real — flat loss for ~12 epochs from a degenerate
   init, val AUC peaking inside that plateau, patience firing before the network
   escapes — and the `--min-epochs 50` rerun recovers **+0.005 to +0.021**
   single-drug and **+0.000 to +0.039** joint. But it does not change the verdict:
   its best single-drug cell reaches 0.8846 against 0.9105 for the best
   non-setfusion cell, and jointly it is still **0.78–0.83 against 0.91–0.92**,
   i.e. 0.09–0.15 *behind* MD-CNN. The warmup recovered roughly a fifth of the
   joint gap; the rest is the architecture. Locus-keyed transformer fusion is not
   competitive here as built — though note it does this at **0.5M parameters
   against late_fusion's 46M**, so the interesting question is efficiency, not
   accuracy.

### On the modalities

5. **The modality choice matters more single-drug than joint.** +0.023 to +0.031
   single-drug (roughly doubling our margin over SD-CNN) but only +0.005 for joint
   `late_fusion`. A joint model already sees every locus for every drug, so it
   recovers by sharing what the single-drug model has to be told.
6. **The gain is concentrated in three drugs, and the mechanism predicts which.**
   PYRAZINAMIDE +0.099 from protein/biophysical, ETHIONAMIDE +0.094 and ISONIAZID
   +0.042 from regulatory. AMIKACIN/KANAMYCIN/RIFAMPICIN gain nothing. This is the
   result worth writing up: it is not "multi-modal helps", it is "**amino-acid
   features rescue loss-of-function genes, promoter windows rescue
   promoter-mediated resistance, and neither can help an rRNA target**".
7. **`all_modalities` is the safe default but rarely the best single choice.** It
   wins 3 of 4 architectures single-drug, but `dna_protein` wins every
   architecture jointly. Adding regulatory windows to a drug that has no
   promoter-mediated resistance is mild noise (PZA −0.02 to −0.04 under
   mdcnn/late_fusion).

### Resolved by the follow-up runs

8. **"Joint beats single" survives the confound, but only about two thirds of it
   is multi-task learning.** Giving the single-drug models the same 19 loci
   (`alllocus_run/`) splits the +0.0189 mean gain into **+0.0069 from the larger
   locus set** and **+0.0120 from joint training** across the 12 complete
   non-setfusion cells. It is strongly architecture-dependent: for `late_fusion`
   almost all of it is multi-task (+0.031 to +0.053, with the locus term often
   *negative*), while for `mdcnn` it is almost all the bigger input (+0.010 to
   +0.018 from loci, ~0.000 from joint training). Only 6 of 12 cells have the
   multi-task term larger. So "one joint model beats 11 single-drug models" is
   true on average and for late_fusion specifically — but for mdcnn it is mostly
   just "more input".

### What is still open

9. **ETHIONAMIDE decides the joint verdict.** Parity-vs-behind against MD-CNN
   turns on one published baseline number that contradicts its own paper.
10. **`alllocus_run/` is a log reconstruction, not original output.** The run
   folder was lost and rebuilt from archived SLURM logs on 2026-08-18 (210 of 220
   results, no per-epoch histories, no weights). Its `summary.csv` numbers are the
   run's own printed values and are exact for `cv_auc_mean`/`cv_auc_std`; see that
   folder's README before quoting anything else from it.""")

# ---------------------------------------------------------------- appendix
md(r"""---

# Appendix — the full grid

Everything above is the narrative. Below is the complete evidence: the 20-cell
leaderboards, the architecture × modality grids, per-drug deltas against both
baselines, joint-vs-single, and the follow-up runs. Nothing here contradicts
§1–§5; it is the detail those sections summarise.""")

md(r"""---

# Appendix Z — joint performance, drug by drug

Every other joint view averages over the 11 drugs, which is what makes a macro Δ
of −0.001 look like "parity" when the per-drug picture is nothing of the sort.
This section keeps each drug on its own row so the question *where are we better
or worse than BIG-TB* is answered directly.

Read the two Δ columns as different claims:

* **Δ DNA-only** — our best joint cell on **DNA alone**, i.e. the same input the
  MD-CNN gets. This is the architecture/protocol comparison, matched.
* **Δ best** — our best joint cell over all five modality sets. This is what the
  project can actually deliver, and it is *not* input-matched to the baseline.

Baseline is the MD-CNN's own 5-fold `auc.csv`, loaded in the CONFIG cell — not
the paper's Table 14. `setfusion` is excluded from both picks because its
numbers are an early-stopping artifact, but it gets its own column so nothing is
hidden.""")

code(r'''# === TABLE G — joint per-drug scoreboard vs the MD-CNN baseline =============
if MD.empty:
    print("No joint results yet.")
else:
    J = MD[(MD.drug != "MACRO") & (MD.drug.isin(MD_BASE_CV.index))]
    real = J[J.arch != "setfusion"]

    rows = {}
    for drug, g in real.groupby("drug"):
        base_test, base_cv, base_sd = BIGTB_MDCNN[drug]
        dna = g[g.modset == REFERENCE_MODSET]
        best = g.loc[g.cv_auc.idxmax()]
        sf = J[(J.drug == drug) & (J.arch == "setfusion")]
        dna_cv = dna.cv_auc.max() if len(dna) else np.nan
        rows[drug] = {
            "n R": int(g.n_R.max()), "n S": int(g.n_S.max()),
            "MD-CNN CV": base_cv, "± SD": base_sd,
            "DNA-only CV": dna_cv, "Δ DNA-only": dna_cv - base_cv,
            "best CV": best.cv_auc, "Δ best": best.cv_auc - base_cv,
            "best cell": f"{MODSET_LABEL[best.modset]} / {ARCH_SHORT[best.arch]}",
            "setfusion": sf.cv_auc.max() if len(sf) else np.nan,
            "MD-CNN test": base_test, "our test": best.test_auc,
            "Δ test": best.test_auc - base_test,
        }
    G = (pd.DataFrame.from_dict(rows, orient="index")
         .sort_values("Δ best", ascending=False))
    G.index.name = "drug"

    # MACRO row: means over drugs, except the counts (sums) and the cell label,
    # which carries the ahead/behind tally instead.
    ahead = int((G["Δ best"] > 0).sum())
    macro = G.select_dtypes("number").mean().to_dict()
    macro.update({"n R": G["n R"].sum(), "n S": G["n S"].sum(),
                  "best cell": f"{ahead}/{len(G)} drugs ahead"})
    G.loc["MACRO (mean)"] = pd.Series(macro)

    fmt = {c: "{:.4f}" for c in ["MD-CNN CV", "± SD", "DNA-only CV", "best CV",
                                 "setfusion", "MD-CNN test", "our test"]}
    fmt.update({c: "{:+.4f}" for c in ["Δ DNA-only", "Δ best", "Δ test"]})
    fmt.update({"n R": "{:,.0f}", "n S": "{:,.0f}"})
    display(G.style.format(fmt, na_rep="—")
            .apply(colour_delta, subset=["Δ DNA-only", "Δ best", "Δ test"])
            .set_table_styles(TABLE_STYLES)
            .set_caption(
                "<b>Joint models, per drug, vs the BIG-TB MD-CNN.</b> CV-AUC, "
                "5-fold, the matched metric. 'DNA-only' is input-matched to the "
                "baseline; 'best' is our best cell over all modality sets and is "
                "not. Green = ahead of the MD-CNN by more than 0.005, red = "
                "behind by more than 0.005. Test columns are indicative only — "
                "their assess cohort differs from our hold-out."))

    print(f"CV vs MD-CNN: ahead on {ahead}/{len(G) - 1} drugs at our best cell, "
          f"{int((G['Δ DNA-only'].iloc[:-1] > 0).sum())}/{len(G) - 1} on DNA alone.")

    # --- the full grid: every drug against every joint cell -----------------
    combos = [(m, a) for m in MODSET_ORDER for a in ARCH_ORDER]
    grid = (J.pivot_table(index="drug", columns=["modset", "arch"], values="cv_auc")
            .reindex(columns=pd.MultiIndex.from_tuples(combos)))
    dgrid = grid.sub(MD_BASE_CV, axis=0).reindex(
        index=[d for d in G.index if d in grid.index])
    dgrid.columns = pd.MultiIndex.from_tuples(
        [(MODSET_LABEL[m], ARCH_SHORT[a]) for m, a in combos])
    vals = dgrid.to_numpy(dtype=float)
    lim = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 0.01
    display(dgrid.style.format("{:+.3f}", na_rep="—")
            .background_gradient(cmap="RdBu", vmin=-lim, vmax=lim, axis=None)
            .set_table_styles(TABLE_STYLES)
            .set_caption(
                "<b>Δ CV vs the MD-CNN, every drug × every joint cell.</b> Blue "
                "ahead, red behind. The setfusion columns are the early-stopping "
                "artifact and should be read as a floor, not a result."))''')

# ============================================================================
# READING ORDER. Cells are authored above in whatever order was convenient;
# this permutes them into the order the notebook should be READ in. Each entry
# is a substring unique to one cell. It is a hard error for an entry to match
# nothing or for a cell to go unplaced, so this cannot silently drop content.
# ============================================================================
READING_ORDER = [
    "# full_run — 4 architectures",          # title + framing
    "import json",                           # imports
    "# === CONFIG",
    "# === LOAD",
    "# === COVERAGE",
    # --- the narrative -----------------------------------------------------
    "# 1 · With DNA alone, do we beat the single-drug baseline?",
    "# === FIGURE 1 —",
    "# 2 · With DNA alone, do we beat the multi-drug baseline?",
    "# === FIGURE 2 —",
    "# 3 · What do the extra modalities add, single-drug?",
    "# === FIGURE 3 —",
    "### Fig 0",                             # per-drug modality detail, inside §3
    "# === FIGURE 0 —",
    "# 4 · What do the extra modalities add, multi-drug?",
    "# === FIGURE 4 —",
    "# 5 · Takeaways",
    "# === SCORECARD",
    # --- appendix ----------------------------------------------------------
    "# Appendix — the full grid",
    "## Which architecture and which modality set",
    "# === TABLE A",
    "# === FIGURE A —",
    "## Beating the SD-CNN, drug by drug",
    "# === FIGURE B —",
    "# === FIGURE C1 —",
    "# === FIGURE C —",
    "## Joint models vs the published BIG-TB MD-CNN",
    "# === TABLE B",
    "# === FIGURE D2 —",
    "# === FIGURE E —",
    "# === FIGURE E2 —",
    "## Follow-up runs",
    "# === FOLLOW-UPS",
    "# === HEADLINE",
    "# Appendix Z — joint performance, drug by drug",
    "# === TABLE G",
]

_placed, _used = [], set()
for _pat in READING_ORDER:
    for _i, _c in enumerate(cells):
        if _i in _used:
            continue
        if _pat in "".join(_c["source"]):
            _placed.append(_i)
            _used.add(_i)
            break
    else:
        raise SystemExit(f"READING_ORDER: nothing matches {_pat!r}")
_left = [i for i in range(len(cells)) if i not in _used]
if _left:
    raise SystemExit("cells not placed by READING_ORDER: "
                     + repr(["".join(cells[i]["source"])[:70] for i in _left]))
cells = [cells[i] for i in _placed]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", OUT, f"({len(cells)} cells)")
