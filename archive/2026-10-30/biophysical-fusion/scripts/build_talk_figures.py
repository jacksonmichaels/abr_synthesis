"""Talk figures that no notebook draws, in the manuscript style.

Writes to results/figures/talk/new/:
  uncalled_channel_per_drug   CV AUC with vs without the uncalled-base channel, per drug:
                              (a) LocusFusion v2 draft (6-ch DNA) vs cleaned (5-ch) tokenizer,
                                  per-drug loci, DNA;  (b) v3 vs its ablation, 19 loci, mean of 5 modality sets
  uncalled_hotspots           resistant share among isolates uncalled at a position vs called there,
                              top positions from results/analysis/missingness_v3/C_top.csv
  lineage_vs_random           macro CV AUC per model, random vs lineage-grouped split, 19 loci
  structure_tokens            how the AlphaFold structure channels entered LocusFusion v2 tokens, on the
                              fig_tokenization isolate (katG R119R, S315T, uncalled 3' flank base)
  site_distance_by_drug       key rrs / RpsL variants: distance to each ribosome drug vs % resistant carriers,
                              one panel per variant (data built by binding_site_visuals.ipynb's own cells)
  recap_effects               wrap-up: every headline comparison as one bar of macro CV AUC difference
  lasso_explainer             (a) the model as a weight per one-hot column, (b) rifampicin's regularization
                              path (nonzero weights vs CV AUC), (c) its largest weights, labelled by position

Reads result JSONs and CSVs only.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import paper_style as ps
from paper_style import MODEL, MODALITY, NEUTRAL as N, model_color
from bigtb_baselines import ALL_DRUGS

ps.apply()
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})
EXP = ROOT / "results" / "experiments"
OUT = ROOT / "results" / "figures" / "talk" / "new"
SHORT = dict(zip(ALL_DRUGS, "AMK CAP EMB ETO INH KAN LFX MXF PZA RIF STR".split()))
M5 = ["dna", "dna_protein", "dna_biophysical", "dna_regulatory", "all_modalities"]


def cv_by_drug(d):
    """drug -> 5-fold CV AUC mean, from one run folder's per-drug JSONs."""
    out = {}
    for p in sorted(Path(d).glob("*.json")):
        j = json.loads(p.read_text())
        if "cv_auc_mean" in j and "drug" in j:
            out[j["drug"]] = j["cv_auc_mean"]
    return pd.Series(out).reindex(ALL_DRUGS)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def dumbbell(ax, lo, hi, order, c_on, c_off, lab_on, lab_off):
    y = np.arange(len(order))[::-1]
    for yi, d in zip(y, order):
        ax.plot([lo[d], hi[d]], [yi, yi], color=N["grid"], lw=5, solid_capstyle="round", zorder=1)
    ax.scatter(lo[order], y, s=60, facecolor="white", edgecolor=c_off, lw=2, zorder=3, label=lab_off)
    ax.scatter(hi[order], y, s=60, color=c_on, zorder=3, label=lab_on)
    ax.set_yticks(y, [SHORT[d] for d in order])
    ax.grid(axis="x")
    ax.set_xlabel("CV AUC")


def uncalled_channel():
    v2 = EXP / "locusfusion_v2"
    draft, clean = cv_by_drug(v2 / "sd_dna__locusfusion_v2"), cv_by_drug(v2 / "sd41_dna__locusfusion_v2")
    for d, w in [("sd_dna__locusfusion_v2", 6), ("sd41_dna__locusfusion_v2", 5)]:
        j = json.loads(next((v2 / d).glob("*__dna.json")).read_text())
        assert j["branch_specs"][0][0] == w, (d, j["branch_specs"])   # 6 = with uncalled-base channel
    v3 = EXP / "locusfusion_v3"
    full = pd.concat([cv_by_drug(v3 / f"sd19_{m}_unk__locusfusion_v2") for m in M5], axis=1).mean(axis=1)
    abl = pd.concat([cv_by_drug(v3 / f"ablation_sd19_{m}__locusfusion_v2") for m in M5], axis=1).mean(axis=1)
    assert not draft.isna().any() and not clean.isna().any() and not full.isna().any() and not abl.isna().any()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=False)
    on, off = MODEL["locusfusion"], MODALITY["uncalled"]
    for ax, hi, lo, title in [
        (axes[0], draft, clean, "(a) LocusFusion v2, per-drug loci, DNA"),
        (axes[1], full, abl, "(b) LocusFusion v3, 19 loci, mean of 5 modality sets"),
    ]:
        order = list((hi - lo).sort_values(ascending=False).index)
        dumbbell(ax, lo, hi, order, on, off, "with uncalled-base channel", "without")
        ax.set_title(title, loc="left")
        d = hi - lo
        ax.text(0.02, 0.02, f"macro Δ = {d.mean():+.4f}   {int((d > 0).sum())}/11 drugs higher",
                transform=ax.transAxes, color=N["ink"], fontsize=12)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 0.08))
    fig.suptitle("CV AUC with vs without the uncalled-base channel, per drug", x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    save(fig, "uncalled_channel_per_drug")
    print(f"  v2 draft−clean {(draft - clean).mean():+.4f}; v3 full−ablation {(full - abl).mean():+.4f}")


def site_label(locus, column, h37rv):
    """'rpoB codon 450 (c.1349)' for CDS columns, 'rrs 1401' for rRNA columns."""
    if isinstance(h37rv, str) and h37rv.startswith("("):
        nt, codon = eval(h37rv)
        return f"{locus} codon {codon} (c.{nt})" if isinstance(nt, int) else f"{locus} {nt} {codon} nt"
    from datasets.cds import cds_columns, _reference_row
    from bigtb_ref import REAL_GENOTYPE_DIR
    d = str(REAL_GENOTYPE_DIR)
    win = cds_columns(d, locus, feature="rRNA")
    ref = _reference_row(f"{d}/{locus}.fasta")
    return f"{locus} {len(ref[:column].replace('-', '')) - len(ref[:win[0]].replace('-', '')) + 1}"


def uncalled_hotspots(p_max=1e-3):
    t = pd.read_csv(ROOT / "results/analysis/missingness_v3/C_top.csv")
    t = t[t["fisher p"] < p_max].copy()
    t["site"] = [site_label(r.locus, r.column, r.h37rv) for r in t.itertuples()]
    t["label"] = [f"{s}  ·  {SHORT[d]}  (n = {n})" for s, d, n in zip(t.site, t.drug, t["N isolates"])]
    t = t.sort_values("R share | N here")
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(t) + 1.4))
    for yi, (a, b) in enumerate(zip(t["R share | called"], t["R share | N here"])):
        ax.plot([a * 100, b * 100], [yi, yi], color=N["grid"], lw=5, solid_capstyle="round", zorder=1)
    ax.scatter(t["R share | called"] * 100, y, s=60, facecolor="white", edgecolor=N["ink2"], lw=2, zorder=3,
               label="isolates with a called base here")
    ax.scatter(t["R share | N here"] * 100, y, s=70, color=MODALITY["uncalled"], edgecolor=N["ink"], lw=0.8,
               zorder=3, label="isolates with an uncalled base (N) here")
    ax.set_yticks(y, t.label)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% resistant")
    ax.grid(axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.set_title("% resistant by base-call status at a position, per-drug loci (Fisher p < 0.001)", loc="left")
    save(fig, "uncalled_hotspots")
    print(t[["site", "drug", "N isolates", "R share | N here", "R share | called"]].to_string(index=False))


def lineage_vs_random():
    lin = EXP / "lineage_split"
    cells = {
        "LocusFusion": ("locusfusion", lin / "sd19_all_modalities_unk__locusfusion_v2",
                        EXP / "locusfusion_v3" / "sd19_all_modalities_unk__locusfusion_v2"),
        "stacked CNN": ("mdcnn", lin / "sd19_all_modalities__mdcnn", EXP / "alllocus_run_v2" / "all_modalities__mdcnn"),
        "cis-fusion": ("cisfusion", lin / "sd19_all_modalities__cisfusion",
                       EXP / "alllocus_run_v2" / "all_modalities__cisfusion"),
        "late-fusion": ("late_fusion", lin / "sd19_all_modalities__late_fusion",
                        EXP / "alllocus_run_v2" / "all_modalities__late_fusion"),
        "LASSO": ("lasso_onehot", lin / "lasso_onehot_all", EXP / "lasso_wholeseq_20260901" / "onehot_all"),
    }
    rows = []
    for name, (mid, dl, dr) in cells.items():
        l, r = cv_by_drug(dl), cv_by_drug(dr)
        assert not l.isna().any() and not r.isna().any(), name
        rows.append(dict(model=name, mid=mid, random=r.mean(), lineage=l.mean()))
    t = pd.DataFrame(rows).sort_values("random")
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for yi, r in zip(y, t.itertuples()):
        c = model_color(r.mid)
        ax.plot([r.lineage, r.random], [yi, yi], color=c, alpha=0.35, lw=4, solid_capstyle="round", zorder=1)
        ax.scatter(r.random, yi, s=90, facecolor="white", edgecolor=c, lw=2.2, zorder=3)
        ax.scatter(r.lineage, yi, s=90, color=c, zorder=3)
        ax.text(r.random + 0.0015, yi, f"{r.random:.3f}", va="center", color=N["ink2"], fontsize=12)
        ax.text(r.lineage - 0.0015, yi, f"{r.lineage:.3f}", va="center", ha="right", color=N["ink"], fontsize=12)
    ax.set_yticks(y, t.model)
    ax.set_xlabel("macro CV AUC")
    ax.grid(axis="x")
    ax.set_xlim(t.lineage.min() - 0.012, t.random.max() + 0.012)
    h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=N["ink2"], mew=2, ms=9, label="random split"),
         plt.Line2D([], [], marker="o", ls="", color=N["ink2"], ms=9, label="lineage-grouped split")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.set_title("Macro CV AUC by split, 19 loci, all modalities (LASSO: DNA)", loc="left")
    save(fig, "lineage_vs_random")
    print(t.assign(delta=t.lineage - t.random).round(4).to_string(index=False))


def column_label(locus, col):
    """(label, H37Rv base) for a 0-based alignment column: 'rpoB c.1349 (codon 450)', 'rrl 433', 'katG c.-12'."""
    from datasets.cds import cds_columns, _reference_row
    from bigtb_ref import REAL_GENOTYPE_DIR
    d = str(REAL_GENOTYPE_DIR)
    ref = _reference_row(f"{d}/{locus}.fasta")
    rna = locus in ("rrs", "rrl")
    win = cds_columns(d, locus, feature="rRNA" if rna else "CDS")
    off = len(ref[:col].replace("-", "")) - len(ref[:win[0]].replace("-", ""))
    span = len(ref[win[0]:win[1]].replace("-", ""))
    if ref[col] == "-":
        where = f"{locus} insertion after {off}"
    elif 0 <= off < span:
        where = f"{locus} {off + 1}" if rna else f"{locus} c.{off + 1} (codon {off // 3 + 1})"
    else:
        where = f"{locus} c.{off - span + 1 if off >= span else off}" if not rna else f"{locus} flank {off + 1}"
        if off >= span:
            where = f"{locus} c.*{off - span + 1}"
    return where, ref[col].upper()


def lasso_explainer(drug="RIFAMPICIN", n_top=8):
    j = json.loads((EXP / "lasso_wholeseq_20260901" / "onehot_all" / f"{drug}__dna-onehot-all.json").read_text())
    teal, R_COL, S_COL = MODEL["lasso"], "#c0392b", "#2a78d6"
    fig = plt.figure(figsize=(16, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 0.9], wspace=0.75)

    # (a) one weight per (position, base) column of the one-hot alignment
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 15)
    ax.set_ylim(-4.6, 6.4)
    ax.axis("off")
    rng = np.random.default_rng(1)
    bases = "ACTG–"
    calls = rng.integers(0, 4, 13)
    calls[[3, 9]] = [1, 3]
    w = {(3, 1): -1.0, (9, 3): 0.8}                 # (column, base row) -> weight sign for the drawing
    for r, b in enumerate(bases):
        y = 4.6 - r * 0.85
        ax.text(0.9, y + 0.35, b, ha="right", va="center", fontsize=12, color=N["ink2"])
        for c in range(13):
            on = calls[c] == r
            ax.add_patch(plt.Rectangle((1.1 + c * 1.0, y), 0.9, 0.75, fc=N["ink2"] if on else N["grid"], ec="none"))
    ax.text(0.9, 0.55, "w", ha="right", va="center", fontsize=13, color=teal, style="italic")
    for c in range(13):
        nz = [(k, v) for k, v in w.items() if k[0] == c]
        x = 1.1 + c * 1.0 + 0.45
        if nz:
            v = nz[0][1]
            ax.add_patch(plt.Rectangle((x - 0.3, 0.55 if v > 0 else 0.55 + v), 0.6, abs(v), fc=teal, ec="none"))
        else:
            ax.text(x, 0.55, "0", ha="center", va="center", fontsize=11, color=N["muted"])
    ax.plot([1.1, 14.1], [0.55, 0.55], color=N["axis"], lw=0.8, zorder=0)
    ax.text(7.6, -1.6, r"$P(\mathrm{resistant}) = \sigma\left(b + \sum_j w_j\, x_j\right)$",
            ha="center", fontsize=15, color=N["ink"])
    ax.text(7.6, -3.3, r"fit: $\min_w\ \mathrm{log\ loss} + \lambda \sum_j |w_j|$", ha="center", fontsize=15,
            color=N["ink"])
    ax.text(7.6, 6.0, "one isolate's one-hot columns  $x_j$", ha="center", fontsize=13, color=N["ink2"])
    ax.set_title("(a) model", loc="left")

    # (b) regularization path: weaker penalty -> more nonzero weights
    ax = fig.add_subplot(gs[1])
    path = pd.DataFrame(j["c_path"])
    best = path.C == j["best_C"]
    ax.plot(path.support_mean, path.cv_auc_mean, color=teal, lw=1.5, zorder=2)
    ax.scatter(path.support_mean[~best], path.cv_auc_mean[~best], s=50, facecolor="white", edgecolor=teal, lw=2,
               zorder=3, label="penalty strengths tried")
    ax.scatter(path.support_mean[best], path.cv_auc_mean[best], s=80, color=teal, zorder=4,
               label="selected by CV")
    for r in path.itertuples():
        if r.C in (path.C.min(), j["best_C"]):
            sel = r.C == j["best_C"]
            ax.annotate(f"{r.support_mean:.0f} weights\nAUC {r.cv_auc_mean:.3f}", (r.support_mean, r.cv_auc_mean),
                        xytext=(0, -14) if sel else (12, 0), textcoords="offset points", fontsize=12,
                        ha="center" if sel else "left", va="top" if sel else "center", color=N["ink"])
    ax.set_xscale("log")
    ax.set_xlabel(f"nonzero weights (of {j['n_features']:,} columns)")
    ax.set_ylabel("CV AUC")
    ax.grid(True)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.12))
    ax.set_title(f"(b) {drug.lower()}, 19 loci: penalty path", loc="left")

    # (c) the largest weights, signed toward resistance (labels: 1 = susceptible, so flip)
    ax = fig.add_subplot(gs[2])
    top = pd.DataFrame(j["top_features"]).head(n_top)
    rows = []
    for f in top.itertuples():
        locus, col, base = f.feature.split(":")
        where, refb = column_label(locus, int(col))
        if "(codon" in where:                                  # 'rpoB c.1349 (codon 450)' -> 'rpoB 450'
            where = f"{locus} {where.split('codon ')[1].rstrip(')')}"
        kind = "H37Rv" if base == refb else ("gap" if base == "-" else "alt")
        rows.append((f"{where} {base} ({kind})", -f.coef))
    lab, val = zip(*rows[::-1])
    y = np.arange(len(val))
    ax.barh(y, val, color=[R_COL if v > 0 else S_COL for v in val], height=0.65)
    ax.set_yticks(y, lab, fontsize=12)
    ax.axvline(0, color=N["ink2"], lw=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("weight (+ resistant, − susceptible)")
    ax.grid(axis="x")
    ax.set_title(f"(c) {drug.lower()}: {n_top} largest weights", loc="left")
    save(fig, "lasso_explainer")
    for r in rows:
        print(f"  {r[0]:45s} {r[1]:+.2f}")
    print(path[["C", "support_mean", "cv_auc_mean"]].round(4).to_string(index=False))


def structure_tokens(length=10.0, n_slots=16):
    """katG example: C-alpha trace, the slot-aligned closeness matrix, and its rows appended to the tokens."""
    z = np.load(ROOT.parent / "structures" / "alphafold" / "katG.npz")
    ca = z["ca"]
    toks = [("v1  R119R", 119), ("v2  S315T", 315), ("v3  3\u2032 flank N", None)]
    M = np.zeros((n_slots, n_slots))
    for i, (_, ci) in enumerate(toks):
        for k, (_, ck) in enumerate(toks):
            if ci and ck:
                M[i, k] = np.exp(-np.linalg.norm(ca[ci - 1] - ca[ck - 1]) / length)
    d = np.linalg.norm(ca[118] - ca[314])
    lf = MODEL["locusfusion"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("close", ["#ffffff", lf])

    fig = plt.figure(figsize=(17, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.3], wspace=0.5)

    # (a) the fold, projected onto its two main axes
    ax = fig.add_subplot(gs[0])
    ok = ~np.isnan(ca).any(axis=1)                  # codons with no coordinate (e.g. the stop) stay NaN
    xyz = ca - ca[ok].mean(0)
    e1 = (ca[314] - ca[118]) / np.linalg.norm(ca[314] - ca[118])      # first axis: R119 -> S315, true length
    v = np.linalg.svd(xyz[ok], full_matrices=False)[2]
    e2 = next(u - (u @ e1) * e1 for u in v if np.linalg.norm(u - (u @ e1) * e1) > 0.5)
    e2 /= np.linalg.norm(e2)
    xy = xyz @ np.stack([e1, e2], axis=1)
    ax.plot(xy[:, 0], xy[:, 1], color=N["axis"], lw=0.8, zorder=1)
    (x1, y1), (x2, y2) = xy[118], xy[314]
    ax.plot([x1, x2], [y1, y2], ls="--", color=N["ink"], lw=1.5, zorder=2)
    ax.scatter([x1, x2], [y1, y2], s=110, color=lf, edgecolor=N["ink"], zorder=3)
    for (x, y), t, ha in (((x1, y1), "R119", "right"), ((x2, y2), "S315", "left")):
        ax.annotate(t, (x, y), xytext=(-12 if ha == "right" else 12, -16), textcoords="offset points", ha=ha,
                    fontsize=14, fontweight="bold", color=N["ink"],
                    bbox=dict(fc="white", ec="none", pad=1, alpha=0.85))
    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 2.5, f"{d:.1f} \u00c5", fontsize=14, ha="center", va="bottom",
            color=N["ink"], bbox=dict(fc="white", ec="none", pad=1, alpha=0.85))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) katG AlphaFold model, C\u03b1 trace", loc="left")

    # (b) slot-aligned closeness matrix over this isolate's token slots
    ax = fig.add_subplot(gs[1])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
    for (i, k), t, xyt in [((0, 0), "1.00  same residue", (5.0, -0.2)),
                           ((0, 1), f"{M[0, 1]:.2f}  R119\u2013S315 ({d:.1f} \u00c5)", (5.0, 1.5)),
                           ((2, 2), "0  no residue (flank)", (5.0, 3.6)),
                           ((9, 9), "0  empty slots", (5.0, 9.0))]:
        ax.annotate(t, (k, i), xytext=xyt, textcoords="data", fontsize=12, color=N["ink"], va="center",
                    arrowprops=dict(arrowstyle="-", color=N["ink2"], lw=1),
                    bbox=dict(fc="white", ec="none", pad=1))
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 3, 3, fill=False, ec=lf, lw=2))
    ax.set_xticks(range(n_slots), [f"v{k + 1}" for k in range(n_slots)], fontsize=9, rotation=90)
    ax.set_yticks(range(n_slots), [f"v{k + 1}" for k in range(n_slots)], fontsize=9)
    ax.set_xticks(np.arange(-0.5, n_slots), minor=True)
    ax.set_yticks(np.arange(-0.5, n_slots), minor=True)
    ax.grid(which="minor", color=N["grid"], lw=0.6)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.set_title(f"(b) closeness exp(\u2212d / {length:.0f} \u00c5), katG token slots", loc="left")

    # (c) row i of the matrix appended to token i
    ax = fig.add_subplot(gs[2])
    ax.set_xlim(-12, 60)
    ax.set_ylim(-1.2, 4.6)
    ax.axis("off")
    base_on = {0: [2, 6 + 15, 29, 30, 31, 44], 1: [1, 6 + 16, 29, 30, 31, 38, 40, 43], 2: [5]}
    for i, (lab, _) in enumerate(toks):
        y = 3.0 - i * 1.25
        ax.text(-0.6, y + 0.4, lab, ha="right", va="center", fontsize=12, color=N["ink"])
        for c in range(41):
            on = c in base_on[i]
            ax.add_patch(plt.Rectangle((c, y), 0.85, 0.8, fc=N["ink2"] if on else N["grid"], ec="none"))
        for k in range(n_slots):
            v = M[i, k]
            ax.add_patch(plt.Rectangle((42 + k, y), 0.85, 0.8, fc=cmap(v), ec=lf if k < 3 else N["grid"], lw=0.6))
    ax.text(20.5, 4.1, "41 token channels", ha="center", fontsize=12, color=N["ink2"])
    ax.text(50, 4.1, f"+ {n_slots} structure", ha="center", fontsize=12, color=lf)
    ax.text(29, -0.9, f"token width 41 + {n_slots} = {41 + n_slots}", ha="center", fontsize=12, color=N["ink2"])
    ax.set_title("(c) row i of (b) appended to token i", loc="left")
    save(fig, "structure_tokens")
    print(f"  d(R119, S315) = {d:.2f} A, closeness {np.exp(-d / length):.3f}")


def site_distance_by_drug():
    """Same variant, four drugs: distance to the drug vs % resistant carriers, one panel per variant."""
    import os
    import nbformat
    nb = nbformat.read(ROOT / "notebooks" / "binding_site_visuals.ipynb", as_version=4)
    cwd = os.getcwd()
    os.chdir(ROOT / "notebooks")
    ns = {"__name__": "__nb__"}
    try:
        for i in (1, 2):
            exec(nb.cells[i].source, ns)
        src = nb.cells[8].source
        exec(src[:src.index("fig, (a1")], ns)                 # data only: RV, DIST_KEY, cnt
    finally:
        os.chdir(cwd)
        ps.apply()
        plt.rcParams.update({"font.size": 13})
    DIST_KEY, cnt, DRUGS, DC = ns["DIST_KEY"], ns["cnt"], ns["DRUGS"], ns["DRUG_COLOR"]
    abbr = {"STREPTOMYCIN": "STR", "KANAMYCIN": "KAN", "AMIKACIN": "AMK", "CAPREOMYCIN": "CAP"}
    panels = ["rrs 517T", "RpsL K43R", "rrs 1401G", "rrs 514C", "rrs 492T"]

    size = lambda n: 30 + 1.6 * n
    fig, axes = plt.subplots(1, len(panels), figsize=(17, 4.6), sharey=True)
    for ax, v in zip(axes, panels):
        ax.axvspan(0, 4, color="#f3f3f3", zorder=0)
        for d in DRUGS:
            if (v, d) not in cnt.index:
                continue
            x, r = DIST_KEY.loc[v, d], cnt.loc[(v, d)]
            ax.scatter(x, r.pctR, s=size(r.n), color=DC[d], edgecolor="white", lw=1.2, zorder=3, alpha=0.95)
        ax.set_xlim(0, 42)
        ax.set_ylim(-5, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlabel("distance to drug (\u00c5)")
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(v, fontsize=15, fontweight="bold")
    axes[0].set_ylabel("% resistant carriers")
    h = [plt.Line2D([], [], marker="o", ls="", color=DC[d], ms=11, label=d.lower()) for d in DRUGS]
    h += [plt.Rectangle((0, 0), 1, 1, fc="#f3f3f3", ec="none", label="\u2264 4 \u00c5")]
    h += [plt.scatter([], [], s=size(n), color=N["muted"], label=f"n = {n}") for n in (25, 100, 200)]
    fig.legend(handles=h, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.13), frameon=False, fontsize=13,
               handletextpad=0.4, columnspacing=1.6)
    fig.suptitle("Distance to each ribosome drug vs % resistant carriers, per variant, test isolates",
                 x=0.01, ha="left", y=1.03, fontsize=15)
    fig.tight_layout()
    save(fig, "site_distance_by_drug")
    print(pd.concat({"dist_A": DIST_KEY.loc[panels].round(1),
                     "pctR": cnt.pctR.unstack("drug").reindex(panels).round(0),
                     "n": cnt.n.unstack("drug").reindex(panels)}, axis=1).to_string())


def recap_effects():
    """One bar per headline comparison (paper family + lineage split), coloured by model family."""
    fam = pd.read_csv(ROOT / "results/figures/manuscript/sig_paper_family.csv")
    lin = pd.read_csv(ROOT / "results/analysis/lineage_split_significance/significance.csv")
    ext = pd.read_csv(EXP / "locusfusion_v3" / "extras.csv").assign(p_holm=lambda d: d.p)   # unadjusted
    pick = [  # (section, label, source frame, comparison string, colour)
        ("locus set: all 19 loci \u2212 own loci", "LASSO", fam, "lasso_onehot: 19 − per-drug loci", MODEL["lasso"]),
        (None, "LocusFusion", fam, "locusfusion: 19 − per-drug loci · 5 modsets", MODEL["locusfusion"]),
        (None, "established CNNs", fam, "established: 19 − per-drug loci · 5 modsets", MODEL["cnn"]),
        ("modalities: all \u2212 DNA, own loci", "established CNNs", fam,
         "established: all_modalities − dna · per-drug loci", MODEL["cnn"]),
        (None, "LocusFusion\u2020", ext, "v3: all_modalities − dna, per-drug loci", MODEL["locusfusion"]),
        ("model size, 19 loci", "stacked CNN 5.1M \u2212 late-fusion 45.9M", fam,
         "mdcnn (5.1M) − late_fusion (45.9M) · 19 loci · 5 modsets", MODEL["cnn"]),
        (None, "stacked CNN 5.1M \u2212 cis-fusion 47.9M", fam,
         "mdcnn (5.1M) − cisfusion (47.9M) · 19 loci · 5 modsets", MODEL["cnn"]),
        ("LocusFusion, 19 loci", "\u2212 LASSO", fam, "locusfusion − lasso_onehot · 19 loci", MODEL["locusfusion"]),
        (None, "\u2212 stacked CNN", fam, "locusfusion − mdcnn · 19 loci · 5 modsets", MODEL["locusfusion"]),
        (None, "\u2212 no uncalled-base channel", fam,
         "locusfusion: fixed − old tokenizer · 19 loci · 5 modsets", MODEL["locusfusion"]),
        ("lineage split \u2212 random split, 19 loci", "LocusFusion", lin,
         "locusfusion · 19 loci · all_modalities", MODEL["locusfusion"]),
        (None, "stacked CNN", lin, "mdcnn · 19 loci · all_modalities", MODEL["cnn"]),
        (None, "LASSO", lin, "lasso_onehot · 19 loci", MODEL["lasso"]),
    ]
    rows, y, ylab, heads = [], 0.0, [], []
    for sec, lab, df, comp, c in pick:
        if sec:
            y -= 0.6 if rows else 0
            heads.append((y, sec))
            y -= 0.9
        r = df[df.comparison == comp]
        assert len(r) == 1, comp
        r = r.iloc[0]
        rows.append(dict(y=y, label=lab, delta=r.delta, wins=r.wins, sig=r.p_holm < 0.05, c=c))
        y -= 1.0
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 7.6))
    for r in t.itertuples():
        ax.barh(r.y, r.delta, height=0.72, color=r.c if r.sig else "white", edgecolor=r.c, lw=2,
                hatch=None if r.sig else "///")
        ax.text(max(r.delta, 0) + 0.0008, r.y, f"{r.delta:+.3f}  ({r.wins})",
                va="center", ha="left", fontsize=12, color=N["ink"])
    for yh, sec in heads:
        ax.text(-0.0335, yh, sec, fontsize=13, fontweight="bold", color=N["ink"], va="center", zorder=5,
                bbox=dict(fc="white", ec="none", pad=1.5))
    ax.set_yticks(t.y, t.label, fontsize=12)
    ax.axvline(0, color=N["ink2"], lw=1)
    ax.set_xlim(-0.034, 0.06)
    ax.set_ylim(t.y.min() - 0.7, 0.5)
    ax.set_xlabel("\u0394 macro CV AUC  (drugs where the first is higher)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    h = [plt.Rectangle((0, 0), 1, 1, fc=N["ink2"], ec=N["ink2"], label="significant (Holm p < 0.05)"),
         plt.Rectangle((0, 0), 1, 1, fc="white", ec=N["ink2"], hatch="///", label="not significant"),
         plt.Line2D([], [], ls="", label="\u2020 unadjusted p, outside the Holm family")]
    ax.legend(handles=h, loc="lower right", frameon=False)
    ax.set_title("Paired comparisons, macro CV AUC difference", loc="left")
    save(fig, "recap_effects")
    print(t[["label", "delta", "wins", "sig"]].round(4).to_string(index=False))


if __name__ == "__main__":
    lasso_explainer()
    uncalled_channel()
    uncalled_hotspots()
    lineage_vs_random()
