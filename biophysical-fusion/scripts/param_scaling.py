#!/usr/bin/env python3
"""
What do the extra modalities cost, and is anyone paying for performance?

    python scripts/param_scaling.py                    # full_run_v2
    python scripts/param_scaling.py --run full_run
    python scripts/param_scaling.py --scope single     # one scope only

Two slide-ready figures, both read from the recorded runs:

    params_vs_modalities_{run}.png|pdf   parameter count as modalities are added
    params_vs_performance_{run}.png|pdf  parameter count against CV-AUC, with the
                                         Pareto frontier marked

Single-drug cells are a mean over the 11 drugs (params differ per drug because
the locus set does); a cell missing any drug is skipped rather than averaged over
fewer. Joint cells are one model each, so their numbers are exact.

Reading rule carried over from the run READMEs: joint fold SD in full_run was
0.003-0.030 on one seed, so vertical gaps under ~0.01 between joint cells are
not resolved. The figures draw that band rather than leaving it implied.
"""
import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

EXP = PROJECT / "results" / "experiments"
OUTDIR = PROJECT / "results" / "figures" / "param_scaling"
ARCH_ORDER = ["late_fusion", "mdcnn", "setfusion", "cisfusion"]
MODSET_ORDER = ["dna", "dna_protein", "dna_biophysical", "dna_regulatory",
                "all_modalities"]
MODSET_LABEL = {"dna": "DNA", "dna_protein": "DNA\n+protein",
                "dna_biophysical": "DNA\n+biophys",
                "dna_regulatory": "DNA\n+regulatory",
                "all_modalities": "all\nmodalities"}
# validated categorical hues; green/pink are a CVD floor-only pair, so every
# series also carries its own marker and a direct label — never colour alone
ARCH_COLOR = {"late_fusion": "#0072B2", "mdcnn": "#E69F00",
              "setfusion": "#009E73", "cisfusion": "#CC79A7"}
ARCH_MARKER = {"late_fusion": "o", "mdcnn": "s", "setfusion": "^", "cisfusion": "D"}
INK, GREY = "#1A1A1A", "#7F7F7F"
NOISE = 0.01                       # single-seed joint noise floor (see docstring)

RC = {
    "figure.dpi": 130, "savefig.dpi": 220, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "font.size": 12.5, "axes.titlesize": 13.5, "axes.labelsize": 12.5,
    "axes.titleweight": "bold", "axes.edgecolor": "0.35",
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "0.25", "ytick.color": "0.25",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.grid": True, "grid.color": "0.91", "grid.linewidth": 0.8,
}
N_DRUGS = 11


def load(run):
    """{scope: {(modset, arch): (params, cv, n_drugs_seen)}} from the run folder."""
    sd, md = {}, {}
    acc = {}
    for p in glob.glob(str(EXP / run / "*" / "[A-Z]*.json")):
        cell = Path(p).parent.name
        m = re.fullmatch(r"(.+?)__(.+)", cell)
        if not m or m.group(1) not in MODSET_ORDER or m.group(2) not in ARCH_ORDER:
            continue
        j = json.loads(Path(p).read_text())
        if j.get("cv_auc_mean") != j.get("cv_auc_mean"):     # NaN
            continue
        acc.setdefault((m.group(1), m.group(2)), []).append(
            (j["n_params"], j["cv_auc_mean"]))
    for k, v in acc.items():
        # partial cells are dropped, not averaged over fewer drugs
        if len(v) == N_DRUGS:
            sd[k] = (float(np.mean([x[0] for x in v])),
                     float(np.mean([x[1] for x in v])), len(v))
    for p in glob.glob(str(EXP / run / "multidrug_*" / "multidrug__*.json")):
        cell = Path(p).parent.name
        m = re.fullmatch(r"multidrug_(.+?)__(.+)", cell)
        if not m or m.group(1) not in MODSET_ORDER or m.group(2) not in ARCH_ORDER:
            continue
        j = json.loads(Path(p).read_text())
        md[(m.group(1), m.group(2))] = (float(j["n_params"]),
                                        float(j["cv_macro_auc_mean"]), N_DRUGS)
    return {"single": sd, "joint": md}


def _save(fig, name):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {OUTDIR}/{name}.png / .pdf")


def fig_params_vs_modalities(data, run, scopes):
    with matplotlib.rc_context(RC):
        fig, axes = plt.subplots(1, len(scopes), figsize=(8.2 * len(scopes), 6.0),
                                 squeeze=False)
        for ax, scope in zip(axes[0], scopes):
            d = data[scope]
            if not d:
                ax.text(0.5, 0.5, "no finished cells", transform=ax.transAxes,
                        ha="center", va="center", color="0.45")
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                ax.set_title(scope, loc="left")
                continue
            xs = [m for m in MODSET_ORDER if any(k[0] == m for k in d)]
            x = np.arange(len(xs))
            ends = []
            for arch in ARCH_ORDER:
                y = [d[(m, arch)][0] / 1e6 if (m, arch) in d else np.nan for m in xs]
                if not np.isfinite(y).any():
                    continue
                ax.plot(x, y, color=ARCH_COLOR[arch], marker=ARCH_MARKER[arch],
                        ms=8, lw=2.0, markeredgecolor="white", markeredgewidth=1.2,
                        label=arch, zorder=4)
                last = int(np.where(np.isfinite(y))[0][-1])
                ends.append([y[last], last, arch])
            ax.set_yscale("log")
            # plain tick labels: a log axis defaults to 6x10^0 style, which is
            # noise on a slide
            ys_all = [v[0] / 1e6 for v in d.values()]
            yt = [t for t in (0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100)
                  if min(ys_all) * 0.8 <= t <= max(ys_all) * 1.25]
            if yt:
                ax.set_yticks(yt)
                ax.set_yticklabels([f"{t:g}" for t in yt])
            ax.minorticks_off()
            ax.set_xticks(x)
            ax.set_xticklabels([MODSET_LABEL[m] for m in xs], fontsize=10.5)
            ax.set_ylabel("trainable parameters (millions, log scale)")
            ax.set_title(f"{scope}-drug", loc="left")
            ax.set_xlim(-0.3, len(xs) - 0.3 + 0.95)
            # direct labels, de-overlapped in LOG space
            ends.sort(key=lambda e: e[0])
            for i in range(1, len(ends)):
                if np.log10(ends[i][0] / ends[i - 1][0]) < 0.075:
                    ends[i][0] = ends[i - 1][0] * 10 ** 0.075
            for yv, xi, arch in ends:
                ax.annotate(arch, (xi, yv), xytext=(8, 0),
                            textcoords="offset points", va="center", fontsize=10,
                            color=ARCH_COLOR[arch], fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                      ec="none", alpha=0.8))
            ax.set_axisbelow(True)
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, fontsize=10.5,
                   bbox_to_anchor=(0.5, -0.045))
        fig.suptitle("What the extra modalities cost in parameters"
                     f"\n{run} · every point is a finished cell",
                     x=0.5, y=1.03, fontsize=15, fontweight="bold")
        fig.tight_layout()
        _save(fig, f"params_vs_modalities_{run}")


def _pareto(points):
    """Indices that nothing else beats on BOTH fewer params and higher CV."""
    keep = []
    for i, (p, c, *_r) in enumerate(points):
        if not any(pp <= p and cc >= c and (pp, cc) != (p, c)
                   for pp, cc, *_ in points):
            keep.append(i)
    return keep


def fig_params_vs_performance(data, run, scopes):
    with matplotlib.rc_context(RC):
        fig, axes = plt.subplots(1, len(scopes), figsize=(8.6 * len(scopes), 6.4),
                                 squeeze=False)
        for ax, scope in zip(axes[0], scopes):
            d = data[scope]
            if not d:
                ax.text(0.5, 0.5, "no finished cells", transform=ax.transAxes,
                        ha="center", va="center", color="0.45")
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                ax.set_title(scope, loc="left")
                continue
            best = max(d.values(), key=lambda v: v[1])[1]
            labels = []
            # the band nothing in this run can resolve
            ax.axhspan(best - NOISE, best, color="0.88", alpha=0.55, zorder=1)
            for arch in ARCH_ORDER:
                pts = [(d[(m, arch)][0] / 1e6, d[(m, arch)][1], m)
                       for m in MODSET_ORDER if (m, arch) in d]
                if not pts:
                    continue
                # joined in modality order: the path IS "what adding data does"
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=ARCH_COLOR[arch], lw=1.4, alpha=0.55, zorder=3)
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=95,
                           color=ARCH_COLOR[arch], marker=ARCH_MARKER[arch],
                           edgecolor="white", linewidth=1.3, label=arch, zorder=5)
                xm, ym, _ = max(pts, key=lambda p: p[1])
                labels.append([ym, xm, arch])
            lo = min(v[1] for v in d.values())
            hi = max(v[1] for v in d.values())
            span = hi - lo
            # de-overlap pushes labels upward, so make room first — otherwise the
            # topmost arch label lands outside the axes and silently disappears
            ax.set_ylim(lo - span * 0.06, hi + span * 0.13)
            labels.sort()
            for i in range(1, len(labels)):
                if labels[i][0] - labels[i - 1][0] < span * 0.055:
                    labels[i][0] = labels[i - 1][0] + span * 0.055
            for ym, xm, arch in labels:
                ax.annotate(arch, (xm, ym), xytext=(9, 6),
                            textcoords="offset points", fontsize=10,
                            color=ARCH_COLOR[arch], fontweight="bold", zorder=6,
                            bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                      ec="none", alpha=0.82))
            flat = [(v[0] / 1e6, v[1], k) for k, v in d.items()]
            front = sorted((flat[i] for i in _pareto(flat)), key=lambda t: t[0])
            ax.plot([f[0] for f in front], [f[1] for f in front], color=INK,
                    lw=1.3, ls="--", alpha=0.65, zorder=4)
            for j, (x0, y0, k) in enumerate(front):
                ax.annotate(MODSET_LABEL[k[0]].replace("\n", " "), (x0, y0),
                            xytext=(0, -17 if j % 2 == 0 else 11),
                            textcoords="offset points", ha="center", fontsize=8.4,
                            color="0.35", zorder=6,
                            bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                      ec="none", alpha=0.75))
            ax.set_xscale("log")
            xs_all = [v[0] / 1e6 for v in d.values()]
            ticks = [t for t in (0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100)
                     if min(xs_all) * 0.75 <= t <= max(xs_all) * 1.3]
            if ticks:
                ax.set_xticks(ticks)
                ax.set_xticklabels([f"{t:g}" for t in ticks])
            ax.minorticks_off()
            ax.set_xlabel("trainable parameters (millions, log scale)")
            ax.set_ylabel("mean CV AUC" + ("  (macro)" if scope == "joint" else ""))
            ax.set_title(f"{scope}-drug", loc="left", pad=26)
            ax.text(0, 1.015, "dashed = best CV per parameter · grey band = "
                    f"top {NOISE:.2f} AUC (unresolved at one seed)",
                    transform=ax.transAxes, fontsize=9.2, color="0.4", va="bottom")
            ax.set_axisbelow(True)
        axes[0][0].legend(loc="lower right", ncol=2, fontsize=10.5)
        fig.suptitle("Do the extra parameters buy anything?"
                     f"\n{run} · each path joins one architecture across modality sets",
                     x=0.5, y=1.04, fontsize=15, fontweight="bold")
        fig.tight_layout()
        _save(fig, f"params_vs_performance_{run}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="full_run_v2")
    ap.add_argument("--scope", default="both", choices=["both", "single", "joint"])
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    global OUTDIR
    if args.outdir:
        OUTDIR = Path(args.outdir)

    data = load(args.run)
    if not data["single"] and not data["joint"]:
        sys.exit(f"No finished cells under results/experiments/{args.run}.")
    scopes = ["single", "joint"] if args.scope == "both" else [args.scope]

    print(f"{args.run}: {len(data['single'])} complete single-drug cells, "
          f"{len(data['joint'])} joint cells")
    for scope in scopes:
        d = data[scope]
        if not d:
            continue
        cheap = min(d.items(), key=lambda kv: kv[1][0])
        best = max(d.items(), key=lambda kv: kv[1][1])
        print(f"\n  {scope}-drug")
        print(f"    best CV      : {best[0][1]:<12s} {best[0][0]:<16s} "
              f"{best[1][1]:.4f} at {best[1][0]/1e6:,.1f}M params")
        print(f"    smallest     : {cheap[0][1]:<12s} {cheap[0][0]:<16s} "
              f"{cheap[1][1]:.4f} at {cheap[1][0]/1e6:,.1f}M params "
              f"({best[1][0]/cheap[1][0]:,.0f}x fewer, "
              f"{cheap[1][1]-best[1][1]:+.4f} CV)")
        flat = [(v[0] / 1e6, v[1], k) for k, v in d.items()]
        front = sorted((flat[i] for i in _pareto(flat)), key=lambda t: t[0])
        print("    Pareto front : " + " → ".join(
            f"{k[1]}/{k[0]} ({p:,.1f}M, {c:.4f})" for p, c, k in front))

    print()
    fig_params_vs_modalities(data, args.run, scopes)
    fig_params_vs_performance(data, args.run, scopes)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTDIR / f"params_perf_{args.run}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "modality_set", "arch", "n_params", "cv_auc", "n_drugs"])
        for scope in scopes:
            for (m, a), (p, c, n) in sorted(data[scope].items()):
                w.writerow([scope, m, a, int(p), round(c, 4), n])
    print(f"  wrote {csv_path}")


if __name__ == "__main__":
    main()
