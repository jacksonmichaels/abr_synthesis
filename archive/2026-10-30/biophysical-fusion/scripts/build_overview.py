"""Generate notebooks/overview.ipynb — project status at a glance.

    python scripts/build_overview.py
    python scripts/build_overview.py -o /tmp/preview.ipynb

The project is a cube: **task x modality set x model**. This notebook reads
every run folder under results/experiments/, works out which cell of that cube
each one fills, and prints one grid per task.

Nothing about the grid is hardcoded. Tasks, modality sets and models are all
derived from the result JSONs (`arch`, `encoder_types`, `modalities`, and the
locus list), so a new architecture, a new modality set or a whole new run shows
up as a new row/column with no edit to this file. See ADDING A RUN in the
notebook's own load cell.

Per-run write-ups live in each run folder's README.md; `build_full_run_viewer.py`
reads ONE sweep in depth. This is the cross-run status view.
"""
import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

_ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("-o", "--output", type=Path, default=None,
                 help="notebook path (default: <project>/notebooks/overview.ipynb)")
_args = _ap.parse_args()
OUT = (_args.output.expanduser().resolve() if _args.output
       else PROJECT / "notebooks" / "overview.ipynb")

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.rstrip("\n").splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.rstrip("\n").splitlines(keepends=True)})


# ================================================================== intro
md(r"""# Project status

Three **tasks**, crossed with **modality sets**, crossed with **models**:

| task | what it is | locus set |
|---|---|---|
| `single-drug` | one model per drug | that drug's own genes (2–3) |
| `single-drug, all loci` | one model per drug | all 19 curated loci |
| `multi-drug` | one model, 11 outputs | all 19 curated loci |

Modality sets are any subset of **DNA / protein / biophysical / regulatory**.
Models are `architecture` + which branch encoder it used (`+tf` = transformer).

`status()` prints one modality × model grid per task. That is the whole point of
the notebook; everything else is a lookup helper.

| function | answers |
|---|---|
| `status()` | the three grids — macro CV AUC per (task, modality, model) |
| `coverage()` | what has and has not been run, and what is incomplete |
| `drug("RIFAMPICIN")` | every result for one drug, ranked |
| `loci_effect("RIFAMPICIN")` | what all-19-loci did — per drug, or per cell |
| `params()` | parameter cost of each modality set |
| `frame` | the tidy DataFrame behind all of it |

**Judge on CV AUC** (5-fold mean, macro over the 11 drugs shared with the
baselines). `test` is one 20% split of one best-fold model and swings ±0.05 on
the small drugs. Single seed throughout — **differences under ~0.01 are
unresolved**.

> Generated — edit `scripts/build_overview.py`, not the `.ipynb`.
> `python scripts/build_overview.py && jupyter nbconvert --to notebook --execute --inplace notebooks/overview.ipynb`""")

# ================================================================== load
code(r'''import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display

_here = Path.cwd()
PROJECT = next(p for p in [_here, *_here.parents] if (p / "bigtb_ref.py").exists())
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
EXP = PROJECT / "results" / "experiments"

# Baselines -- single source of truth, never copied here.
from bigtb_baselines import SD_BASE_CV, SD_BASE_TEST, MD_BASE_CV, MD_BASE_TEST, ALL_DRUGS
DRUGS = list(ALL_DRUGS)

# --- ADDING A RUN -----------------------------------------------------------
# Nothing below needs editing. Drop a run folder under results/experiments/ and
# it is picked up: task, modality set and model are all read off the result
# JSONs. A new architecture becomes a new column; a new modality set a new row.
# The only manual lists are these two.
DROP_ARCH = {"setfusion"}                     # degenerate tokens; see token_signal
DROP_RUNS = {
    "full_run", "alllocus_run",           # superseded 150/15/0 schedule
    # Hyperparameter sweeps, not grid cells: their folders are named for the ARM
    # (`a2_lr1e3_multidrug_dna__late_fusion`), so each arm would invent its own
    # modality-set row. They have their own write-ups; keep them out.
    "joint_capacity", "joint_convergence", "setfusion_scaling", "setfusion_warmup",
    "token_signal",
    "newmodels_pilot",                    # 2 drugs, superseded by newmodels_full
}

# Some runs put the task in the CELL name rather than in the run name --
# newmodels_full submits `sd_`, `sd19_` and `md_multidrug_` cells into ONE
# folder. Strip those so `sd_all_modalities__locusfusion` lands on the
# `all_modalities` row beside every other model instead of inventing a row
# nobody else can be compared against. The task itself is still derived from the
# summary file and the locus count, never from the prefix.
TASK_PREFIXES = ("md_multidrug_", "multidrug_", "sd19_", "sd_", "md_")

# A cell folder is normally `<modset>__<arch>`. A few are not -- the sparse
# baselines in variant_aggregators_20260825 are named for the METHOD alone,
# because they are not an architecture at all (L1-logistic / gradient boosting on
# the variant matrix). They still belong in the grid, as the answer to "does a
# linear model on variants already do this", so fall back to reading the modality
# set off the result's own tag and use the folder name as the model.
TAG_TO_MODSET = {
    "dna": "dna", "dna+protein": "dna_protein",
    "dna+biophysical": "dna_biophysical", "dna+regulatory": "dna_regulatory",
    "dna+protein+biophysical+regulatory": "all_modalities",
}


def _strip_task_prefix(name):
    for pre in TASK_PREFIXES:
        if name.startswith(pre):
            return name[len(pre):]
    return name
ALL_LOCI = 19                                 # what counts as "all loci"

TASKS = ["single-drug", "single-drug, all loci", "multi-drug"]''')

code(r'''# === LOAD ===================================================================
# One pass over every result JSON collects the shape of each run (arch,
# encoders, locus count, params); summary.csv supplies the metrics. Task,
# modality set and model are all DERIVED, so a new run needs no code change.
def _read(p):
    try:
        j = json.loads(p.read_text())
    except Exception:
        return None
    loci = j.get("loci") or j.get("genes") or []
    enc = set(j.get("encoder_types") or []) | set((j.get("encoders") or {}).values())
    tf = "transformer" in enc
    return dict(arch=j.get("arch"), tf=tf,
                model=f"{j.get('arch')}{'+tf' if tf else ''}",
                n_loci=len(loci), n_params=j.get("n_params"), drug=j.get("drug"))


shape = {}                       # (run, celldir, drug|None) -> shape dict
for run_dir in sorted(p for p in EXP.iterdir() if p.is_dir()):
    if run_dir.name in DROP_RUNS:
        continue
    for d in (p for p in run_dir.iterdir() if p.is_dir()):
        for p in d.glob("*.json"):
            if p.name == "weights_location.json":
                continue
            s = _read(p)
            if s:
                shape[(run_dir.name, d.name, s.pop("drug"))] = s

rows, skipped = [], []
for run_dir in sorted(p for p in EXP.iterdir() if p.is_dir()):
    if run_dir.name in DROP_RUNS:
        continue
    for d in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        # Which summary file exists IS the task, and is authoritative. Reading it
        # off the folder name was fine while every joint cell was called
        # `multidrug_*`, but newmodels_full names its joint cells
        # `md_multidrug_<modset>__<arch>` and they were silently dropped.
        joint = (d / "multidrug_summary.csv").is_file()
        f = d / ("multidrug_summary.csv" if joint else "summary.csv")
        if not f.is_file():
            continue
        stem = _strip_task_prefix(d.name)
        here = {k: v for k, v in shape.items() if k[0] == run_dir.name and k[1] == d.name}
        if not here:                       # no JSON -> cannot be placed
            skipped.append(f"{run_dir.name}/{d.name}")
            continue
        any_shape = next(iter(here.values()))
        if any_shape["arch"] in DROP_ARCH:
            continue
        if "__" in stem:
            modset, model = stem.rsplit("__", 1)[0], any_shape["model"]
        else:
            # method-named cell (the sparse baselines): modality set off the tag,
            # folder name as the model, so two vocabulary policies stay distinct
            tag = str(pd.read_csv(f)["modalities"].iloc[0])
            modset, model = TAG_TO_MODSET.get(tag), stem
            if modset is None:
                skipped.append(f"{run_dir.name}/{d.name} (unmapped tag {tag!r})")
                continue
        # A run is "all loci" if its cells load the full curated set. Taken from
        # the max over the cell's drugs so a 1-locus drug cannot mislabel it.
        cell_loci = max(v["n_loci"] for v in here.values())
        task = ("multi-drug" if joint else
                "single-drug, all loci" if cell_loci >= ALL_LOCI else "single-drug")
        df = pd.read_csv(f)
        df = (df[df["drug"] != "MACRO"].rename(columns={"cv_auc": "cv"}) if joint
              else df.rename(columns={"cv_auc_mean": "cv", "cv_auc_std": "cv_sd"}))
        for _, r in df.iterrows():
            if r["drug"] not in DRUGS:
                continue
            s = here.get((run_dir.name, d.name, r["drug"]), any_shape)
            rows.append(dict(task=task, modset=modset, model=model,
                             arch=any_shape["arch"], run=run_dir.name, cell=stem,
                             drug=r["drug"], n_loci=s["n_loci"], n_params=s["n_params"],
                             cv=r.get("cv"), cv_sd=r.get("cv_sd"), test=r.get("test_auc")))

frame = pd.DataFrame(rows)
for c in ("cv", "cv_sd", "test"):
    frame[c] = pd.to_numeric(frame[c], errors="coerce")

# Modality sets: the standard ladder first, anything new appended.
_STD = ["dna", "dna_protein", "dna_biophysical", "dna_regulatory", "all_modalities"]
MODSETS = [m for m in _STD if m in set(frame.modset)] + \
          sorted(set(frame.modset) - set(_STD))
MODELS = sorted(set(frame.model))

dupes = (frame.groupby(["task", "modset", "model", "run"]).size()
         .reset_index().groupby(["task", "modset", "model"]).size())
print(f"{len(frame):,} drug-runs | {frame.groupby(['task','modset','model']).ngroups} "
      f"(task x modality x model) cells | models: {', '.join(MODELS)}")
if skipped:
    print(f"skipped (no result JSON, cannot be placed): {', '.join(skipped)}")
if (dupes > 1).any():
    print("NOTE two runs fill the same cell:",
          ", ".join(f"{a}/{b}/{c}" for a, b, c in dupes[dupes > 1].index))''')

# ================================================================== status
code(r'''# === THE GRIDS ==============================================================
GOOD, BAD = "background-color:#d9ecd9", "background-color:#f7d9d3"


def _grid(task, value="cv"):
    s = frame[frame.task == task]
    g = s.pivot_table(index="modset", columns="model", values=value, aggfunc="mean")
    return g.reindex(index=[m for m in MODSETS if m in g.index],
                     columns=[m for m in MODELS if m in g.columns])


def status(value="cv", baseline=True):
    """One modality x model grid per task. The project in three tables."""
    for task in [t for t in TASKS if t in set(frame.task)]:
        g = _grid(task, value)
        base = SD_BASE_CV.mean() if task == "single-drug" else MD_BASE_CV.mean()
        name = "SD-CNN" if task == "single-drug" else "MD-CNN"
        n = frame[frame.task == task].groupby("modset").drug.nunique()
        # 0 is a real value here (regulatory_only loads no CODING loci), but it
        # describes that one row, not the task -- so it is left out of the range.
        loci = sorted({x for x in frame[frame.task == task].n_loci if x})
        rng = f"{loci[0]}" if len(loci) == 1 else f"{min(loci)}-{max(loci)}"
        print(f"\n=== {task}   ({len(g.columns)} models x {len(g.index)} modality sets, "
              f"{rng} loci)")
        sty = (g.style.format("{:.4f}", na_rep="—")
               .set_caption(f"macro CV AUC over 11 drugs &nbsp;|&nbsp; "
                            f"{name} baseline = {base:.4f}"))
        if baseline:
            sty = sty.map(lambda v: GOOD if v > base else (BAD if v < base - 0.02 else ""))
        display(sty)
        short = n[n < 11]
        if len(short):
            print("   incomplete rows:", dict(short))


def coverage():
    """What exists, what is missing, what is short of 11 drugs."""
    out = []
    for task in [t for t in TASKS if t in set(frame.task)]:
        s = frame[frame.task == task]
        have = set(map(tuple, s[["modset", "model"]].drop_duplicates().values))
        models = sorted(set(s.model))
        for m in MODSETS:
            for mo in models:
                n = s[(s.modset == m) & (s.model == mo)].drug.nunique()
                out.append(dict(task=task, modset=m, model=mo, drugs=n,
                                state="—" if (m, mo) not in have
                                else ("ok" if n == 11 else f"{n}/11")))
    t = pd.DataFrame(out)
    piv = t.pivot_table(index=["task", "modset"], columns="model", values="state",
                        aggfunc="first").fillna("—")
    return piv.style.map(lambda v: BAD if v == "—" else
                         ("" if v == "ok" else "background-color:#fdf0d0"))''')

md(r"""## Status""")
code(r'''status()''')

md(r"""Green beats that task's baseline. Read **down** a column for what the
modalities buy a fixed model, and **across** a row for what the model choice
buys a fixed input. `—` is a cell nobody has run — `coverage()` shows those
explicitly, along with any row short of 11 drugs.""")

code(r'''coverage()''')

# ================================================================== helpers
code(r'''# === LOOKUPS ================================================================
def _sty(df, delta_cols=(), fmt=None):
    f = {c: "{:.4f}" for c in df.columns if df[c].dtype.kind == "f"}
    f.update({c: "{:+.4f}" for c in delta_cols})
    f.update(fmt or {})
    s = df.style.format(f, na_rep="—")
    for c in delta_cols:
        s = s.map(lambda v: GOOD if v > 0.005 else (BAD if v < -0.005 else ""), subset=[c])
    return s


def drug(d, n=12):
    """One drug: its best cell in each (task, model), ranked."""
    s = frame[frame.drug == d]
    t = (s.sort_values("cv", ascending=False)
         .groupby(["task", "model"], as_index=False).first()
         [["task", "model", "modset", "run", "n_loci", "cv", "cv_sd", "test", "n_params"]]
         .sort_values("cv", ascending=False).head(n))
    t["vs_SD-CNN"] = t.cv - SD_BASE_CV[d]
    t["vs_MD-CNN"] = t.cv - MD_BASE_CV[d]
    print(f"{d}  |  SD-CNN {SD_BASE_CV[d]:.3f}   MD-CNN {MD_BASE_CV[d]:.4f}   "
          f"headroom {1 - SD_BASE_CV[d]:.3f}")
    return _sty(t, ["vs_SD-CNN", "vs_MD-CNN"],
                {"n_params": "{:,.0f}", "n_loci": "{:.0f}"}).hide(axis="index")


def loci_effect(d=None):
    """'single-drug, all loci' minus 'single-drug' -- same models, 19 loci vs 2-3.

    d=None -> one row per drug; d="RIFAMPICIN" -> one row per (modality, model).

    Paired on (drug, modality set, model): a cell that exists on only one side
    is dropped, not averaged in. Without that the leave-one-out modality sets,
    which only ever ran at per-drug loci, drag that side down and every delta
    comes out inflated.
    """
    a = frame[frame.task == "single-drug"]
    b = frame[frame.task == "single-drug, all loci"]
    m = a.merge(b, on=["drug", "modset", "model"], suffixes=("_a", "_b"))
    if d:
        m = m[m.drug == d]
        if m.empty:
            raise KeyError(f"{d}: no paired cells")
    key = ["modset", "model"] if d else ["drug"]
    gp = m.groupby(key)
    t = pd.DataFrame({"per-drug loci": gp.cv_a.mean(), "all 19 loci": gp.cv_b.mean()})
    t["delta"] = t["all 19 loci"] - t["per-drug loci"]
    t["loci"] = [f"{int(x)} -> {int(y)}"
                 for x, y in zip(gp.n_loci_a.max(), gp.n_loci_b.max())]
    t["params"] = [f"{x/1e6:.1f}M -> {y/1e6:.1f}M"
                   for x, y in zip(gp.n_params_a.mean(), gp.n_params_b.mean())]
    if not d:
        t["headroom"] = 1 - t.index.map(SD_BASE_CV)
        t["% of headroom"] = t.delta / t.headroom
    n_cells = m.groupby(key).size().iloc[0] if len(t) else 0
    print(f"all 19 loci minus per-drug loci  |  paired on (modality, model), "
          f"{n_cells} cells per row")
    return _sty(t.sort_values("delta", ascending=False), ["delta"],
                {"% of headroom": "{:+.0%}"})


def params(task="single-drug", relative=False):
    """Parameter cost of each modality set, by model."""
    s = frame[frame.task == task]
    g = s.pivot_table(index="model", columns="modset", values="n_params", aggfunc="mean")
    g = g.reindex(columns=[m for m in MODSETS if m in g.columns])
    if relative and "dna" in g.columns:
        return g.div(g["dna"], axis=0).style.format("{:.2f}×", na_rep="—")
    out = (g / 1e6).round(2)
    if "dna" in g.columns and "all_modalities" in g.columns:
        out["all/dna"] = (g["all_modalities"] / g["dna"]).round(2)
    print(f"{task}: millions of parameters (mean over drugs)")
    return out


print("ready:  status()  coverage()  drug()  loci_effect()  params()  frame")''')

# ================================================================== examples
md(r"""## Examples

**"How did adding the 19 loci help RIFAMPICIN?"**""")
code(r'''loci_effect("RIFAMPICIN")''')

md(r"""Barely — **+0.002 to +0.005**, against a fold SD of ~0.002 on a single
seed. RIFAMPICIN starts at CV 0.976 because the *rpoB* RRDR is a tight
positional signal a one-hot already captures, so there is only ~0.024 of
headroom and 19 loci claims almost none of it.

Across all drugs the gain tracks **headroom**, not biology — except KANAMYCIN,
whose per-drug map is a *single* gene (`rrs`) and which therefore also loses the
*eis* promoter, its best-known mechanism. At 19 loci that comes back:""")
code(r'''loci_effect()''')

md(r"""KANAMYCIN gains most — **46% of all the headroom it had**. Its per-drug map
is a *single* gene (`rrs`), so it also loses the *eis* promoter, its best-known
mechanism; at 19 loci that comes back. LEVOFLOXACIN's **−0.075** is the other
outlier and is not a finding: 76 resistant isolates, fold SD ~0.037.

**"How many parameters did we gain when adding modalities?"**""")
code(r'''params()''')

md(r"""~1.5–1.7× from DNA-only to all four. One row is worth knowing: **`mdcnn`
at `dna_regulatory` costs nothing** — `MDCNNNet` groups blocks by channel count
and DNA and regulatory are both 5-channel, so promoter windows stack as extra
*channels* on the existing trunk instead of getting a branch. Giving regulatory
its own branch under `mdcnn` needs `--mdcnn-trunk-per-modality`, which no
recorded run uses.""")
code(r'''drug("KANAMYCIN")''')

# ================================================================== caveats
md(r"""## Caveats

- **Single seed.** Fold SD is 0.003–0.037 by drug (`cv_sd`). Under ~0.01 is
  unresolved. Joint rows have **no** `cv_sd` — `multidrug_summary.csv` records
  one CV number per drug, not a spread.
- **`single-drug, all loci` cannot be quoted against SD-CNN.** That baseline
  sees the per-drug map, so it is a different input. It *is* input-matched to
  MD-CNN, which uses all 19 by its own rule.
- **`test` is favourably selected** — best of 5 folds by val AUC, where BIG-TB
  uses fold 4 unconditionally. Judge on CV.
- **Transformer cells are not capacity-matched** to their CNN counterparts
  (parameter ratio 0.70–1.34) — check `params()` before reading a difference as
  an architecture effect.
- **`noisyor` is missing from the grid on purpose, and is rerunning.** Its first
  run scored macro CV **0.4956** — below chance — because the model is monotone
  in its evidence and was pointed at P(resistant), while this project encodes
  **R=0/S=1**; a variant therefore pushed every isolate toward the wrong class.
  It never escaped its init either (train loss 0.3152 -> 0.3023 over 99 epochs,
  against `additive`'s 0.2246 -> 0.0875 on the identical cell). Fixed
  2026-08-26 in `models/experimental_models.py` (the product is P(susceptible);
  the saturating `-4.0` init is now `-2.0`) and resubmitted at the identical
  protocol. The broken results are in
  `results/archive/noisyor_polarity_bug_20260825/` with a write-up, so they are
  out of this notebook's scan and cannot be quoted by accident. Rebuild this
  notebook once the rerun lands.
- **`setfusion` and the 150-epoch runs are excluded** (`DROP_ARCH` / `DROP_RUNS`
  in the load cell). setfusion's tokens are near-collinear at init so its
  numbers are a lower bound, not an architecture verdict; `full_run` and
  `alllocus_run` ran at a schedule where the epoch cap was binding.
- **A max over many cells is optimistic.** `drug()` selects on the same CV it
  reports — an upper bound, not a model.""")

# ================================================================== write
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(cells)} cells)")
