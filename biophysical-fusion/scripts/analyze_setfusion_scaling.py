#!/usr/bin/env python3
"""
Read whatever setfusion_scaling has finished and say what it shows.

    python scripts/analyze_setfusion_scaling.py                 # report to stdout
    python scripts/analyze_setfusion_scaling.py --write         # + ANALYSIS.md, .csv, .png
    python scripts/analyze_setfusion_scaling.py --scope single  # single-drug stages

Safe to run at any time: unfinished cells are reported as missing rather than
silently dropped, so a partial read never looks like a complete one.

What it does that a bare table would not
----------------------------------------
* Every arm is differenced against ITS OWN control — the matching
  ``full_run_v2/{,multidrug_}{mods}__setfusion`` cell, same modality set, same
  scope, same protocol.
* Every delta is judged against a NOISE BAND rather than reported bare. The band
  is max(0.01, the control's own fold SD): one seed, five folds, and full_run's
  joint fold SD ran 0.003-0.030 against the gap being chased. An arm inside the
  band is reported as UNRESOLVED, not as a small win — that distinction is the
  whole point of the run.
* Cap-hit detection: an arm whose folds stopped at the 300-epoch ceiling was
  still climbing, so its capacity was measured under undertraining and its delta
  is a lower bound. Flagged per arm.
* The R arms (training-only) must have the control's exact parameter count, and
  the D arms must not. Both are asserted, because a mismatch means the flags did
  not reach the model and the cell is measuring nothing.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from sweep_setfusion_scaling import (ARMS, CONTROL_RUN, MODALITY_SETS,  # noqa: E402
                                     RUN_PREFIX)

EXP = PROJECT / "results" / "experiments"
SWEEP = EXP / RUN_PREFIX
NOISE = 0.01                 # the floor full_run's own READMEs read against
CAP_FRACTION = 0.9           # best_epoch >= 0.9 * cap counts as "still climbing"
ARM_BY_NAME = {a.name: a for a in ARMS}
AXIS_TITLE = {
    "A": "A - token width (d_model)",
    "B": "B - post-encoder capacity (dim_ff / layers / hidden)",
    "C": "C - encoder block (channels / depth / bins)",
    "D": "D - per-drug read-out head",
    "R": "R - training regime, capacity unchanged",
    "X": "X - crosses (bundles, read against A-R)",
}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _tag(mods):
    return "+".join(MODALITY_SETS[mods])


def _fold_sd(j):
    """Fold SD of the CV metric, joint or single."""
    if "cv_macro_auc_std" in j:
        return j["cv_macro_auc_std"]
    return j.get("cv_auc_std", float("nan"))


def _cv(j):
    return j["cv_macro_auc_mean"] if "cv_macro_auc_mean" in j else j["cv_auc_mean"]


def _test(j):
    return j["test_macro_auc"] if "test_macro_auc" in j else j["test"]["auc"]


def load_control(mods, scope):
    """The full_run_v2 cell this sweep's arms are differenced against."""
    if scope == "joint":
        p = EXP / CONTROL_RUN / f"multidrug_{mods}__setfusion" / f"multidrug__{_tag(mods)}.json"
        if not p.exists():
            return None
        j = json.loads(p.read_text())
        return {"cv": _cv(j), "test": _test(j), "sd": _fold_sd(j),
                "params": j["n_params"], "per_drug": j.get("cv_per_drug_auc", {}),
                "epochs": j["epochs"]}
    d = EXP / CONTROL_RUN / f"{mods}__setfusion"
    per = {}
    for p in sorted(d.glob(f"*__{_tag(mods)}.json")):
        j = json.loads(p.read_text())
        per[j["drug"]] = {"cv": _cv(j), "test": _test(j), "sd": _fold_sd(j),
                          "params": j["n_params"], "epochs": j["epochs"]}
    return per or None


def load_cells(scope):
    """{(arm, mods): record} for every finished cell of this scope."""
    out = {}
    if not SWEEP.exists():
        return out
    for d in sorted(SWEEP.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        for mods in MODALITY_SETS:
            infix = f"_multidrug_{mods}__setfusion" if scope == "joint" else f"_{mods}__setfusion"
            if not name.endswith(infix):
                continue
            arm = name[: -len(infix)]
            if arm not in ARM_BY_NAME:
                continue
            if scope == "joint":
                p = d / f"multidrug__{_tag(mods)}.json"
                if not p.exists():
                    continue
                j = json.loads(p.read_text())
                out[(arm, mods)] = {
                    "cv": _cv(j), "test": _test(j), "sd": _fold_sd(j),
                    "params": j["n_params"], "epochs": j["epochs"],
                    "best_epochs": [f["best_epoch"] for f in j["cv_folds"]],
                    "hours": j["seconds"] / 3600, "per_drug": j.get("cv_per_drug_auc", {}),
                    "setfusion": j.get("setfusion"), "n_drugs": len(j["drugs"]),
                }
            else:
                drugs = {}
                for p in sorted(d.glob(f"*__{_tag(mods)}.json")):
                    j = json.loads(p.read_text())
                    drugs[j["drug"]] = {
                        "cv": _cv(j), "test": _test(j), "sd": _fold_sd(j),
                        "params": j["n_params"], "epochs": j["epochs"],
                        "best_epochs": [f["best_epoch"] for f in j["cv_folds"]],
                        "hours": j["seconds"] / 3600}
                if drugs:
                    out[(arm, mods)] = {"drugs": drugs}
    return out


# ---------------------------------------------------------------------------
# judging
# ---------------------------------------------------------------------------

def verdict(delta, band):
    if delta > band:
        return "GAIN"
    if delta < -band:
        return "LOSS"
    return "unresolved"


def cap_hit(best_epochs, cap):
    return any(e is not None and e >= CAP_FRACTION * cap for e in best_epochs)


def joint_rows(cells):
    rows = []
    for mods in MODALITY_SETS:
        ctl = load_control(mods, "joint")
        if ctl is None:
            continue
        for arm in ARM_BY_NAME:
            rec = cells.get((arm, mods))
            if rec is None:
                continue
            band = max(NOISE, ctl["sd"])
            delta = rec["cv"] - ctl["cv"]
            a = ARM_BY_NAME[arm]
            rows.append({
                "axis": a.axis, "arm": arm, "mods": mods, "note": a.note,
                "params": rec["params"], "ratio": rec["params"] / ctl["params"],
                "cv": rec["cv"], "cv_sd": rec["sd"], "ctl_cv": ctl["cv"],
                "delta": delta, "band": band, "verdict": verdict(delta, band),
                "test": rec["test"], "test_delta": rec["test"] - ctl["test"],
                "cap_hit": cap_hit(rec["best_epochs"], rec["epochs"]),
                "best_epochs": rec["best_epochs"], "hours": rec["hours"],
                "ctl_params": ctl["params"],
            })
    rows.sort(key=lambda r: (r["axis"], r["arm"], r["mods"]))
    return rows


def single_rows(cells):
    """One row per (arm, mods): the mean over the drugs BOTH the arm and the
    control scored, so a partial arm is never compared against 11 control drugs."""
    rows = []
    for mods in MODALITY_SETS:
        ctl = load_control(mods, "single")
        if not ctl:
            continue
        for arm in ARM_BY_NAME:
            rec = cells.get((arm, mods))
            if rec is None:
                continue
            shared = sorted(set(rec["drugs"]) & set(ctl))
            if not shared:
                continue
            cv = sum(rec["drugs"][d]["cv"] for d in shared) / len(shared)
            cvc = sum(ctl[d]["cv"] for d in shared) / len(shared)
            band = max(NOISE, sum(ctl[d]["sd"] for d in shared) / len(shared))
            a = ARM_BY_NAME[arm]
            any_cap = any(cap_hit(rec["drugs"][d]["best_epochs"],
                                  rec["drugs"][d]["epochs"]) for d in shared)
            rows.append({
                "axis": a.axis, "arm": arm, "mods": mods, "note": a.note,
                "params": rec["drugs"][shared[0]]["params"],
                "ratio": rec["drugs"][shared[0]]["params"] / ctl[shared[0]]["params"],
                "cv": cv, "cv_sd": float("nan"), "ctl_cv": cvc,
                "delta": cv - cvc, "band": band, "verdict": verdict(cv - cvc, band),
                "test": sum(rec["drugs"][d]["test"] for d in shared) / len(shared),
                "test_delta": (sum(rec["drugs"][d]["test"] for d in shared)
                               - sum(ctl[d]["test"] for d in shared)) / len(shared),
                "cap_hit": any_cap, "best_epochs": [],
                "hours": sum(rec["drugs"][d]["hours"] for d in shared),
                "n_drugs": len(shared), "ctl_params": ctl[shared[0]]["params"],
            })
    rows.sort(key=lambda r: (r["axis"], r["arm"], r["mods"]))
    return rows


def integrity_checks(rows):
    """Cheap assertions that the flags actually reached the model. An R arm with
    a different parameter count, or a D arm with the same one, means the cell is
    measuring something other than what its name says."""
    out = []
    for r in rows:
        if r["axis"] == "R" and r["params"] != r["ctl_params"]:
            out.append(f"FAIL {r['arm']}/{r['mods']}: training-only arm has "
                       f"{r['params']:,} params, control has {r['ctl_params']:,}")
        if r["axis"] in ("A", "B", "C", "D") and r["params"] == r["ctl_params"]:
            out.append(f"FAIL {r['arm']}/{r['mods']}: capacity arm is identical "
                       f"to the control ({r['params']:,} params) — flags did not land")
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def coverage(scope, cells):
    """(finished, expected, missing-arm-names) for this scope."""
    expected = [(a.name, m) for a in ARMS for m in MODALITY_SETS
                if not (scope == "single" and a.scope == "joint")]
    have = set(cells)
    return len(have), len(expected), [f"{a}/{m}" for a, m in expected if (a, m) not in have]


def render(rows, scope, cells, fh=sys.stdout):
    n, total, missing = coverage(scope, cells)
    w = fh.write
    w(f"# setfusion_scaling — {scope}-drug analysis\n\n")
    w(f"{n}/{total} cells finished")
    if missing:
        w(f"; **{len(missing)} not in yet**: {', '.join(missing[:8])}"
          + (" ..." if len(missing) > 8 else ""))
    w(".\n\n")
    if not rows:
        w("No finished cells to analyse yet.\n")
        return

    problems = integrity_checks(rows)
    if problems:
        w("## Integrity\n\n")
        for p in problems:
            w(f"- {p}\n")
        w("\n")

    gains = [r for r in rows if r["verdict"] == "GAIN"]
    w("## Headline\n\n")
    band = rows[0]["band"]
    w(f"Noise band: **±{band:.4f}** (max of 0.01 and the control's fold SD). "
      "One seed, five folds — an arm inside the band has not been shown to do "
      "anything.\n\n")
    if gains:
        best = max(gains, key=lambda r: r["delta"])
        w(f"**{len(gains)} of {len(rows)} arms clear the band.** Best: "
          f"`{best['arm']}` on {best['mods']} at {best['delta']:+.4f} "
          f"(CV {best['cv']:.4f} vs control {best['ctl_cv']:.4f}), "
          f"{best['params']/1e6:.2f}M params = {best['ratio']:.1f}x the control.\n\n")
        by_axis = {}
        for r in gains:
            by_axis.setdefault(r["axis"], []).append(r)
        w("Which axes moved:\n\n")
        for ax in sorted(by_axis):
            arms = ", ".join(sorted({f"{r['arm']} ({r['delta']:+.4f}, {r['mods']})"
                                     for r in by_axis[ax]}))
            w(f"- **{AXIS_TITLE[ax]}** — {arms}\n")
        w("\n")
    else:
        w(f"**No arm clears the ±{band:.4f} band.** On this evidence the gap is "
          "not a capacity problem at the scales tested — the ceiling arm is "
          f"{max(r['ratio'] for r in rows):.0f}x the control and still inside the "
          "noise. Read the R arms before concluding: if they are also flat, the "
          "training regime is not the limiter either, which points at the shape "
          "of the model (see the README's two structural notes).\n\n")

    capped = [r for r in rows if r["cap_hit"]]
    if capped:
        w(f"**{len(capped)} arms had folds at the 300-epoch ceiling** — measured "
          "under undertraining, so their deltas are lower bounds: "
          + ", ".join(sorted({f"{r['arm']}/{r['mods']}" for r in capped})[:10]) + "\n\n")

    w("## Every finished arm\n\n")
    cur = None
    for r in rows:
        if r["axis"] != cur:
            cur = r["axis"]
            w(f"\n### {AXIS_TITLE[cur]}\n\n")
            w("| arm | mods | params | xctl | CV | vs control | verdict | TEST | "
              "cap-hit | note |\n|---|---|---|---|---|---|---|---|---|---|\n")
        w(f"| `{r['arm']}` | {r['mods']} | {r['params']:,} | {r['ratio']:.2f} | "
          f"{r['cv']:.4f} | {r['delta']:+.4f} | {r['verdict']} | "
          f"{r['test']:.4f} ({r['test_delta']:+.4f}) | "
          f"{'yes' if r['cap_hit'] else '-'} | {r['note']} |\n")

    w("\n## Caveats\n\n")
    w("- One seed. This is a screen that produces a shortlist to multi-seed, not "
      "a result. Re-run the shortlist at 3+ seeds before quoting any number.\n")
    w("- TEST is the best-of-5-folds model scored once, so it is favourably "
      "selected; CV is the honest comparison and TEST is shown only as a "
      "consistency check.\n")
    if scope == "single":
        w("- Single-drug `dna` cells have 1-2 blocks per drug, so axes A and B "
          "are near-vacuous there by construction; weight the `dna_protein` "
          "column when reading them.\n")
        w("- Each row is a mean over the drugs the arm and the control BOTH "
          "scored, so partial arms are not compared against 11 control drugs.\n")


# ---------------------------------------------------------------------------
# figure: does CV rise with capacity, per axis?
# ---------------------------------------------------------------------------
# Form: magnitude against an ordered capacity ladder -> dots joined in ladder
# order, one panel per axis (small multiples), the control as a reference line
# with its noise band drawn rather than implied. Colour carries the modality set
# only (2 series), and every series also has its own marker plus a direct label,
# so identity never rests on hue alone. Hues are the pair scripts/param_scaling.py
# already uses (Okabe-Ito): no node on this cluster, so the palette validator
# could not be run here — reusing an already-validated pair rather than inventing.

MODS_COLOR = {"dna": "#0072B2", "dna_protein": "#E69F00"}
MODS_MARKER = {"dna": "o", "dna_protein": "s"}
INK = "#1A1A1A"
RC = {
    "figure.dpi": 130, "savefig.dpi": 220, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "font.size": 11.5, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.titleweight": "bold", "axes.edgecolor": "0.35",
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "0.25", "ytick.color": "0.25",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.grid": True, "grid.color": "0.91", "grid.linewidth": 0.8,
}


def figure(rows, scope, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes_present = [a for a in ("A", "B", "C", "D", "R", "X")
                    if any(r["axis"] == a for r in rows)]
    if not axes_present:
        return None
    ncol = min(3, len(axes_present))
    nrow = (len(axes_present) + ncol - 1) // ncol
    with matplotlib.rc_context(RC):
        fig, axs = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 4.4 * nrow),
                                squeeze=False)
        for ax in axs.flat[len(axes_present):]:
            ax.set_visible(False)
        for ax, axis in zip(axs.flat, axes_present):
            sub = [r for r in rows if r["axis"] == axis]
            # the R arms change training, not size — within one modality set
            # every point sits at that control's exact parameter count, so a
            # parameter axis stacks them on a single tick. Test per series, not
            # across them: the two modality sets have different counts, which
            # would otherwise make the panel look like it had a real x-spread.
            categorical = all(
                len({r["params"] for r in sub if r["mods"] == m}) < 2
                for m in MODALITY_SETS if any(r["mods"] == m for r in sub))
            arm_order = sorted({r["arm"] for r in sub})
            for mods in MODALITY_SETS:
                pts = sorted((r for r in sub if r["mods"] == mods),
                             key=lambda r: (arm_order.index(r["arm"]) if categorical
                                            else r["params"]))
                if not pts:
                    continue
                ctl_cv, band = pts[0]["ctl_cv"], pts[0]["band"]
                ax.axhline(ctl_cv, color=MODS_COLOR[mods], lw=1.1, ls="--", alpha=0.55,
                           zorder=2)
                ax.axhspan(ctl_cv - band, ctl_cv + band, color=MODS_COLOR[mods],
                           alpha=0.07, zorder=1)
                x = ([arm_order.index(r["arm"]) for r in pts] if categorical
                     else [r["params"] / 1e6 for r in pts])
                y = [r["cv"] for r in pts]
                ax.plot(x, y, color=MODS_COLOR[mods], lw=1.6, alpha=0.75, zorder=3)
                ax.scatter(x, y, s=74, color=MODS_COLOR[mods], marker=MODS_MARKER[mods],
                           edgecolor="white", linewidth=1.2, zorder=5,
                           label=mods.replace("_", "+"))
                if not categorical:      # label the best point; ticks name the rest
                    best = max(pts, key=lambda r: r["cv"])
                    ax.annotate(best["arm"], (best["params"] / 1e6, best["cv"]),
                                xytext=(0, 11), textcoords="offset points",
                                fontsize=8.6, ha="center",
                                color=MODS_COLOR[mods], fontweight="bold", zorder=6,
                                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                          ec="none", alpha=0.82))
            if categorical:
                ax.set_xticks(range(len(arm_order)))
                ax.set_xticklabels(arm_order, fontsize=9)
                ax.set_xlim(-0.45, len(arm_order) - 0.55)
                ax.set_xlabel("arm (parameter count unchanged)")
            else:
                ax.set_xscale("log")
                ax.minorticks_off()
                # a log axis defaults to 10^0 labels, which is noise on a slide
                lo = min(r["params"] for r in sub) / 1e6
                hi = max(r["params"] for r in sub) / 1e6
                ticks = [t for t in (0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 15, 20)
                         if lo * 0.8 <= t <= hi * 1.25]
                if ticks:
                    ax.set_xticks(ticks)
                    ax.set_xticklabels([f"{t:g}" for t in ticks])
                ax.set_xlim(lo * 0.7, hi * 1.45)     # room for the direct label
                ax.set_xlabel("parameters (millions, log)")
            # headroom so an annotation never lands outside the axes
            ys = [r["cv"] for r in sub] + [r["ctl_cv"] + r["band"] for r in sub] \
                 + [r["ctl_cv"] - r["band"] for r in sub]
            span = max(max(ys) - min(ys), 1e-3)
            ax.set_ylim(min(ys) - 0.08 * span, max(ys) + 0.16 * span)
            ax.set_title(AXIS_TITLE[axis], loc="left", fontsize=11)
            ax.set_ylabel("mean CV AUC" + (" (macro)" if scope == "joint" else ""))
            ax.set_axisbelow(True)
        handles = {}
        for ax in axs.flat:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(l, h)
        fig.legend(list(handles.values()), list(handles), loc="lower center",
                   ncol=2, bbox_to_anchor=(0.5, -0.055))
        fig.suptitle(f"setfusion_scaling — {scope}-drug: does capacity buy CV AUC?"
                     "\ndashed line = full_run_v2 control · shaded = noise band "
                     "(one seed, unresolved)", x=0.5, y=1.02, fontsize=13.5,
                     fontweight="bold")
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png)
        plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", default="joint", choices=["joint", "single"])
    ap.add_argument("--write", action="store_true",
                    help="write ANALYSIS_{scope}.md, .csv and .png into the run folder")
    args = ap.parse_args()

    cells = load_cells(args.scope)
    rows = joint_rows(cells) if args.scope == "joint" else single_rows(cells)
    render(rows, args.scope, cells)

    if args.write:
        SWEEP.mkdir(parents=True, exist_ok=True)
        md = SWEEP / f"ANALYSIS_{args.scope}.md"
        with open(md, "w") as fh:
            render(rows, args.scope, cells, fh=fh)
        print(f"\nwrote {md}")
        if rows:
            csv_path = SWEEP / f"analysis_{args.scope}.csv"
            keys = [k for k in rows[0] if k != "best_epochs"]
            with open(csv_path, "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
                wr.writeheader()
                wr.writerows(rows)
            print(f"wrote {csv_path}")
            png = figure(rows, args.scope, SWEEP / f"capacity_vs_cv_{args.scope}.png")
            if png:
                print(f"wrote {png}")


if __name__ == "__main__":
    main()
