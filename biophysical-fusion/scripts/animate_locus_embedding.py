#!/usr/bin/env python3
"""
What the learned locus embedding actually does, animated.

`SetFusionNet` shares ONE `SharedBlockEncoder` per modality across every locus of
that modality, so `dna:gyrB` and `dna:gyrA` hit identical weights and are told
apart only by what the sequence contains. The learned `locus_emb` is the thing
that is supposed to make them distinguishable downstream. This script shows that
happening, by animating the one step that separates the two:

    enc  = SharedBlockEncoder[modality](x)                    # no identity yet
    pre  = enc + modality_emb[modality] + locus_emb[locus]    # identity added

Measured on full_run_v2/all_modalities__setfusion, MOXIFLOXACIN, 2,868 isolates:

    cross-LOCUS centroid distance (same modality)   0.0416 -> 0.3236   (7.8x)
    cross-MODALITY centroid distance                0.9381 -> 1.0253   (1.09x)
    within-group spread                             0.00090 -> 0.00090 (1.00x)

So the honest claim is narrow and worth stating on the figure: the embeddings
separate LOCI, not modalities (the encoders had already separated those), and
they do it by rigid translation — every token of a group moves by the same
constant vector, so no cloud changes shape.

Three things this script does deliberately, because the obvious alternatives are
each wrong here:

1. **The tokens are NOT centered.** `notebooks/token_pca.ipynb` centers every
   `(modality, locus)` group before its PCA, for good reasons that do not apply
   here: the shift added between the two stages IS a per-group constant, so
   subtracting the per-group mean removes it EXACTLY and the two stages become
   bit-identical. A centered version of this animation is provably frozen.
2. **The 2-D basis is constructed, not fitted** (`split_basis`). One basis per
   panel, held fixed across the morph so motion is real movement and not a change
   of basis. A PCA over the pooled stages was the first attempt and it is
   measurably wrong here: its variance is dominated by the embedding shift and the
   final split, so the *initial* split projects 0.0253 -> 0.0017 on the DNA panel,
   a 15x understatement that makes the encoder look more locus-blind than it is
   and inflates the effect. Spanning the two separation vectors instead reproduces
   BOTH endpoint distances to within float32 round-off (1e-5 to 1e-4 relative,
   printed per panel and written to the CSV), so the on-screen ratio IS the
   128-d ratio.
3. **One panel per modality, and the panels share an axis SCALE.** In a single
   shared view the split is real but only ~5% of the frame, because the much
   larger modality separation — which the embeddings did *not* cause — sets the
   axis scale. Faceting refuses to let an unrelated larger effect hide this one.
   Each panel is centred on its own data but every panel spans the same distance,
   so bar lengths are comparable across modalities — which is how the punchline
   shows up: all four end at the SAME separation, because that separation just is
   ||locus_emb[gyrB] - locus_emb[gyrA]|| = 0.3184, identical for every modality.

The plot is **2D, equal-aspect**, not 3D. Pooling one modality's two loci over
both stages leaves four centroids, and a PCA of them puts ~0.0% in PC3 — the
structure is rank 2. A 3D axes renders that as a pancake and, without an aspect
lock, stretches the empty third axis to the same visual length as the informative
first. Equal aspect in 2D is the version where a distance on screen is the
distance.

What is deliberately NOT shown: the within-group cloud. Its spread (0.0001-0.0027)
is 100-800x smaller than the separation, so at any scale where the split is
visible every group is a single point. It is reported per panel in the CSV rather
than drawn as a smear that would suggest the clouds change shape. They do not —
the shift is a per-group constant, and the measured spread is identical before
and after to five decimals.

Outputs (results/figures/token_pca/):
    locus_embedding_split_{DRUG}.gif        the animation
    locus_embedding_split_{DRUG}.png        static before/after, for slides
    locus_embedding_split_{DRUG}.csv        per-panel distances behind the labels
    locus_embedding_tokens_{DRUG}.npz       cached enc/pre (skips the 50s reload)

    python scripts/animate_locus_embedding.py
    python scripts/animate_locus_embedding.py --drug MOXIFLOXACIN --fps 12
    python scripts/animate_locus_embedding.py --no-cache      # recompute tokens
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from sklearn.decomposition import PCA                    # noqa: E402

from bigtb_ref import (REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV,  # noqa: E402
                       REAL_REGULATORY_DIR)
from datasets import load_dataset                        # noqa: E402
from training.checkpoint import load_model               # noqa: E402

# --- house style (matches notebooks/token_pca.ipynb) ------------------------
SURFACE, INK, MUTED, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
MODALITY_COLOR = {"dna": "#2a78d6", "protein": "#e87ba4",
                  "biophysical": "#008300", "regulatory": "#4a3aa7"}
# Locus identity is consistent across every panel and never carried by colour
# alone — blue/orange is the standard CVD-safe pair, plus a distinct marker.
LOCUS_STYLE = [("#2a78d6", "o"), ("#e07b39", "^"), ("#008300", "s"),
               ("#4a3aa7", "D")]
SMIN, SMAX = 30, 300          # marker area in pt^2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="full_run_v2/all_modalities__setfusion")
    p.add_argument("--drug", default="MOXIFLOXACIN")
    p.add_argument("--stem", default=None,
                   help="checkpoint stem (default: {DRUG}__{modalities from config})")
    p.add_argument("--frames", type=int, default=72)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--elev", type=float, default=18.0)
    p.add_argument("--azim", type=float, default=-60.0)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--no-cache", action="store_true",
                   help="recompute tokens even if the .npz exists")
    p.add_argument("--out-dir", default=str(PROJECT / "results" / "figures" / "token_pca"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# 1. tokens
# ---------------------------------------------------------------------------

def token_stages(model, arrays, rows, batch=256, device="cpu"):
    """The two stages either side of the embedding add.

    Returns two (len(rows), n_blocks, d_model) arrays. This is
    ``SetFusionNet.forward`` up to the fusion transformer, split one line
    earlier than ``notebooks/token_pca.ipynb`` splits it — ``enc`` is the bare
    encoder output, which that notebook never keeps.
    """
    ids = model._default_ids
    names = [model.modalities[i] for i in ids[:, 0].tolist()]
    enc, pre = [], []
    for start in range(0, len(rows), batch):
        idx = rows[start:start + batch]
        xs = [torch.from_numpy(a[idx]).float().to(device) for a in arrays]
        e = torch.stack([model.encoders[m](x) for m, x in zip(names, xs)], dim=1)
        enc.append(e.cpu().numpy())
        pre.append((e + model.modality_emb(ids[:, 0])
                    + model.locus_emb(ids[:, 1])).cpu().numpy())
    return np.concatenate(enc), np.concatenate(pre)


def load_tokens(args, out_dir):
    """enc/pre plus the block keys, from cache when possible."""
    cache = out_dir / f"locus_embedding_tokens_{args.drug}.npz"
    if cache.exists() and not args.no_cache:
        z = np.load(cache, allow_pickle=True)
        print(f"tokens from cache: {cache.name}  {z['enc'].shape}")
        return z["enc"], z["pre"], [tuple(k) for k in z["keys"]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stem = args.stem
    model, cfg = load_model(args.run, stem, map_location=device) if stem else (None, None)
    if model is None:                     # derive the stem from the run's config
        for guess in _stem_guesses(args):
            try:
                model, cfg = load_model(args.run, guess, map_location=device)
                stem = guess
                break
            except Exception:
                continue
        if model is None:
            raise SystemExit(f"could not resolve a checkpoint stem for {args.drug} "
                             f"in {args.run}; pass --stem")
    model.to(device).eval()
    print(f"{cfg['model']['arch']}  {cfg['drug']}  fold {cfg['best_fold']}  "
          f"{cfg['model']['n_params']:,} params  device={device}")

    t0 = time.time()
    data = load_dataset(args.drug, cfg["data"]["modalities_used"],
                        REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV,
                        regulatory_dir=REAL_REGULATORY_DIR,
                        per_modality_branch=False, verbose=False)
    rows = np.nonzero(data.y != -1)[0]
    print(f"loaded {data.n:,} isolates in {time.time() - t0:.0f}s; "
          f"{len(rows):,} phenotyped")

    with torch.no_grad():
        enc, pre = token_stages(model, data.arrays(), rows, args.batch, device)
    keys = [tuple(k) for k in model.block_keys]
    np.savez_compressed(cache, enc=enc, pre=pre, keys=np.array(keys, dtype=object))
    print(f"tokens cached -> {cache.name}")
    return enc, pre, keys


def _stem_guesses(args):
    return [f"{args.drug}__dna+protein+biophysical+regulatory",
            f"{args.drug}__dna", f"{args.drug}__dna+protein",
            f"{args.drug}__dna+biophysical", f"{args.drug}__dna+regulatory"]


# ---------------------------------------------------------------------------
# 2. per-panel projection
# ---------------------------------------------------------------------------

def centroid_distance(tok, i, j):
    """Full-space distance between two blocks' centroids."""
    return float(np.linalg.norm(tok[:, i].mean(0) - tok[:, j].mean(0)))


def split_basis(enc, pre, i, j):
    """A 2-D basis that contains BOTH separation vectors exactly.

    A PCA over the pooled stages is the obvious choice and it is the wrong one
    here. Its variance is dominated by the embedding shift and the final split,
    so the *initial* split — the small quantity the whole figure is a comparison
    against — lands almost entirely outside the retained plane: on the DNA panel
    it projects 0.0253 down to 0.0017, a 15x understatement that would make the
    encoder look far more locus-blind than it is and inflate the effect.

    So the plane is constructed rather than fitted: `u` along the post-embedding
    split, `v` along whatever part of the pre-embedding split is orthogonal to
    it. Both difference vectors lie in span(u, v) by construction, so BOTH
    endpoint distances are reproduced exactly (0% distortion) and the ratio on
    screen is the ratio in 128-d. Everything else — within-group spread, the
    shift's other components — is what gets projected away, and those are
    reported separately rather than drawn.
    """
    d1 = pre[:, i].mean(0) - pre[:, j].mean(0)          # after the embedding
    d0 = enc[:, i].mean(0) - enc[:, j].mean(0)          # before it
    n1 = np.linalg.norm(d1)
    if n1 < 1e-12:                                       # degenerate: no split
        return PCA(n_components=2, random_state=0).fit(
            np.concatenate([enc[:, [i, j]].reshape(-1, enc.shape[-1]),
                            pre[:, [i, j]].reshape(-1, pre.shape[-1])])).components_
    u = d1 / n1
    w = d0 - (d0 @ u) * u
    nw = np.linalg.norm(w)
    if nw < 1e-9 * max(np.linalg.norm(d0), 1e-12):
        # d0 is parallel to d1 — the second axis carries no separation, so spend
        # it on the largest remaining variance instead of an arbitrary direction
        pool = np.concatenate([enc[:, [i, j]].reshape(-1, enc.shape[-1]),
                               pre[:, [i, j]].reshape(-1, pre.shape[-1])])
        resid = pool - np.outer(pool @ u, u)
        v = PCA(n_components=1, random_state=0).fit(resid).components_[0]
    else:
        v = w / nw
    return np.stack([u, v])                              # (2, d_model)


def build_panels(enc, pre, keys):
    """One panel per modality: its own PCA over that modality's enc+pre pooled."""
    modalities = list(dict.fromkeys(m for m, _ in keys))
    panels = []
    for modality in modalities:
        idx = [i for i, (m, _l) in enumerate(keys) if m == modality]
        if len(idx) < 2:
            print(f"  skip {modality}: only {len(idx)} locus block(s), nothing to split")
            continue
        loci = [keys[i][1] for i in idx]
        basis = split_basis(enc, pre, idx[0], idx[1])    # (2, d_model)
        A = enc[:, idx] @ basis.T                        # (n_iso, n_loci, 2)
        B = pre[:, idx] @ basis.T

        # full-space vs projected separation, averaged over the locus pairs
        pairs = [(p, q) for p in range(len(idx)) for q in range(p + 1, len(idx))]
        full0 = np.mean([centroid_distance(enc, idx[p], idx[q]) for p, q in pairs])
        full1 = np.mean([centroid_distance(pre, idx[p], idx[q]) for p, q in pairs])
        proj0 = np.mean([np.linalg.norm(A[:, p].mean(0) - A[:, q].mean(0)) for p, q in pairs])
        proj1 = np.mean([np.linalg.norm(B[:, p].mean(0) - B[:, q].mean(0)) for p, q in pairs])
        spread = float(np.linalg.norm(enc[:, idx] - enc[:, idx].mean(0), axis=-1).mean())

        panels.append({
            "modality": modality, "loci": loci, "A": A, "B": B, "basis": basis,
            "full0": full0, "full1": full1, "proj0": proj0, "proj1": proj1,
            "spread": spread,
            "points": dedupe_paths(A, B),
        })
    return panels


def dedupe_paths(A, B, decimals=6):
    """Collapse isolates that share a token to one marker, per locus.

    Thousands of isolates carry a byte-identical alignment at these loci and land
    on exactly the same token, so drawing every isolate stacks thousands of dots
    in one place. Dedupe on the (start, end) PAIR so a marker is the same object
    in every frame of the morph, and carry the pile-up as the count.
    """
    out = []
    d = A.shape[-1]
    for k in range(A.shape[1]):
        key = np.round(np.hstack([A[:, k], B[:, k]]), decimals)
        uniq, count = np.unique(key, axis=0, return_counts=True)
        out.append((uniq[:, :d], uniq[:, d:], count))
    return out


def marker_size(n, n_max):
    """Area proportional to sqrt(count): a 1,000x pile-up must not make the
    singletons invisible, which straight area-proportional sizing would do."""
    return SMIN + (SMAX - SMIN) * np.sqrt(n / max(n_max, 1))


# ---------------------------------------------------------------------------
# 3. figure
# ---------------------------------------------------------------------------

def shared_extent(panels, xpad=1.22, ymin_frac=0.34):
    """One (half-width, half-height) for EVERY panel.

    Panels are individually centred but identically scaled, so a separation drawn
    in the `protein` panel is the same number of pixels as the same separation in
    `dna`. That comparability is the point: all four modalities end at ~0.32
    because that distance IS the locus-embedding difference, which does not
    depend on the modality.

    The two axes are sized independently — the data is a horizontal band, and an
    equal x/y range would spend most of every panel on empty space. Aspect stays
    locked to 'equal' regardless, so one unit is one unit on both axes and a
    length on screen is still a length; only the amount of blank margin changes.
    """
    wx = wy = 0.0
    for p in panels:
        stack = np.vstack([np.vstack([a, b]) for a, b, _n in p["points"]])
        centre = 0.5 * (stack.max(0) + stack.min(0))
        half = np.abs(stack - centre).max(axis=0)
        wx, wy = max(wx, float(half[0])), max(wy, float(half[1]))
    half_w = wx * xpad
    return half_w, max(wy * 1.6, half_w * ymin_frac)


def draw_panel(ax, panel, extent, t=0.0):
    """Scatter one modality at morph position t; returns the updatable artists."""
    half_w, half_h = extent
    stack = np.vstack([np.vstack([a, b]) for a, b, _n in panel["points"]])
    cx, cy = 0.5 * (stack.max(0) + stack.min(0))
    n_max = max(n.max() for _a, _b, n in panel["points"])

    # the measured separation, drawn as a segment between the two centroids —
    # the quantity the animation is about, shown rather than only asserted
    (a0, b0, _), (a1, b1, _) = panel["points"][0], panel["points"][1]
    c0 = (panel["A"][:, 0].mean(0), panel["B"][:, 0].mean(0))
    c1 = (panel["A"][:, 1].mean(0), panel["B"][:, 1].mean(0))
    bar, = ax.plot([], [], color=MUTED, lw=1.1, ls=(0, (3, 2)), zorder=1)
    # the live distance sits in a fixed corner rather than on the bar: at t=0 the
    # bar is a few pixels long and a label pinned to its midpoint lands under the
    # markers, which is exactly the frame that has to stay legible
    label = ax.text(0.02, 0.90, "", transform=ax.transAxes, color=INK, fontsize=11,
                    ha="left", va="center", family="monospace", zorder=6)

    arts = []
    for k, ((a, b, n), locus) in enumerate(zip(panel["points"], panel["loci"])):
        color, marker = LOCUS_STYLE[k % len(LOCUS_STYLE)]
        p = (1 - t) * a + t * b
        sc = ax.scatter(p[:, 0], p[:, 1], s=marker_size(n, n_max), alpha=0.9,
                        color=color, marker=marker, linewidths=0.7,
                        edgecolors=SURFACE, label=locus, zorder=4)
        arts.append((sc, a, b))

    ax.set_xlim(cx - half_w, cx + half_w)
    ax.set_ylim(cy - half_h, cy + half_h)
    # 'equal' with adjustable='datalim' would silently widen the limits to fit the
    # box and break the shared scale; 'box' reshapes the box instead and keeps the
    # limits — and therefore the cross-panel comparability — exactly as set.
    ax.set_aspect("equal", adjustable="box")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.set_xticks([]); ax.set_yticks([])

    err = max(abs(panel["proj0"] / max(panel["full0"], 1e-12) - 1),
              abs(panel["proj1"] / max(panel["full1"], 1e-12) - 1))
    ax.set_xlabel(f"split axis · both distances exact ({err:.0e} error)",
                  fontsize=8, color=MUTED, labelpad=5)
    fold = panel["full1"] / max(panel["full0"], 1e-12)
    ax.set_title(f"{panel['modality']}", color=MODALITY_COLOR.get(panel["modality"], INK),
                 fontsize=12.5, loc="left", pad=24)
    ax.text(0.0, 1.03, f"{panel['full0']:.3f} → {panel['full1']:.3f}   ({fold:.1f}×)",
            transform=ax.transAxes, color=MUTED, fontsize=9.5, ha="left", va="bottom")
    return {"markers": arts, "bar": bar, "label": label, "c0": c0, "c1": c1}


def build_figure(panels, drug, n_iso, dpi):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": MUTED, "grid.color": RULE,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
        "legend.frameon": False,
    })
    ncols = min(len(panels), 2 if len(panels) <= 4 else 4)
    nrows = int(np.ceil(len(panels) / ncols))
    extent = shared_extent(panels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 2.7 * nrows + 3.0),
                             dpi=dpi)
    axes = np.atleast_1d(axes).ravel()
    handles = [draw_panel(axes[i], p, extent) for i, p in enumerate(panels)]
    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(f"{drug} · setfusion · what the learned locus embedding does",
                 color=INK, fontsize=16, x=0.03, ha="left", y=0.985)
    # Lines are wrapped by hand to the CANVAS width, not to the tight bounding box:
    # the static PNGs are saved with bbox_inches='tight' and would absorb an
    # overhang, but FuncAnimation renders the raw canvas, so anything past the edge
    # is simply cut off in the GIF.
    for y, line in ((0.945, "One shared encoder per modality maps gyrB and gyrA to "
                            "nearly the same point — the learned locus embedding is "
                            "what pulls them apart."),
                    (0.917, "It adds one constant vector per block, so the clouds "
                            "translate without changing shape."),
                    (0.889, "All four modalities land at the same ~0.32, because that "
                            "distance is ‖locus_emb[gyrB] − locus_emb[gyrA]‖ = 0.3184.")):
        fig.text(0.03, y, line, color=MUTED, fontsize=11, ha="left", va="top")

    legend = [plt.Line2D([], [], ls="", marker=m, ms=9, color=c, label=locus)
              for (c, m), locus in zip(LOCUS_STYLE, panels[0]["loci"])]
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.975, 1.0),
               ncols=len(legend), fontsize=11, labelcolor=INK, frameon=False,
               handletextpad=0.3, columnspacing=1.4)

    caption = fig.text(0.03, 0.105, "", color=INK, fontsize=11.5, ha="left",
                       va="center", family="monospace")
    fig.text(0.03, 0.045,
             f"{n_iso:,} isolates · one marker per distinct token, area ∝ √isolates · "
             f"2D equal-aspect, in a basis built to contain both separation vectors so "
             f"both endpoint distances are exact rather than projected\n"
             f"tokens uncentred (per-group centring would remove this effect exactly) · "
             f"every panel shares one scale, each centred on its own data\n"
             f"within-group spread (0.0001–0.0027) is 100–800× smaller than the split, "
             f"and is unchanged by the embedding — reported in the CSV, not drawn",
             color=MUTED, fontsize=8.5, ha="left", va="center", linespacing=1.6)
    fig.subplots_adjust(left=0.035, right=0.975, top=0.795, bottom=0.185,
                        wspace=0.14, hspace=0.62)
    return fig, handles, caption


def set_t(handles, caption, t):
    for h in handles:
        for sc, a, b in h["markers"]:
            sc.set_offsets((1 - t) * a + t * b)
        p0 = (1 - t) * h["c0"][0] + t * h["c0"][1]
        p1 = (1 - t) * h["c1"][0] + t * h["c1"][1]
        h["bar"].set_data([p0[0], p1[0]], [p0[1], p1[1]])
        h["label"].set_text(f"{np.linalg.norm(p1 - p0):.3f}")
    filled = int(round(t * 30))
    caption.set_text(f"encoder output  |{'█' * filled}{'·' * (30 - filled)}|  "
                     f"+ modality & locus embeddings   {t:5.0%}")


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    enc, pre, keys = load_tokens(args, out_dir)
    n_iso = len(enc)

    print(f"\nblocks: {[f'{m}:{l}' for m, l in keys]}")
    panels = build_panels(enc, pre, keys)
    if not panels:
        raise SystemExit("no modality has 2+ locus blocks — nothing to animate. "
                         "This needs a per-locus run (per_modality_branch=False).")

    rows = []
    print("\nper-panel locus separation (centroid distance):")
    for p in panels:
        print(f"  {p['modality']:12s} {'/'.join(p['loci']):12s} "
              f"full {p['full0']:.4f} -> {p['full1']:.4f} ({p['full1']/p['full0']:.1f}x)   "
              f"projected {p['proj0']:.4f} -> {p['proj1']:.4f} "
              f"({p['proj1']/p['full1']:.0%} of full retained)   "
              f"within-group spread {p['spread']:.5f}")
        rows.append({"modality": p["modality"], "loci": "/".join(p["loci"]),
                     "full_enc": p["full0"], "full_pre": p["full1"],
                     "fold_change": p["full1"] / p["full0"],
                     "proj_enc": p["proj0"], "proj_pre": p["proj1"],
                     "proj_retained": p["proj1"] / p["full1"],
                     "within_spread": p["spread"],
                     "proj_error": max(abs(p["proj0"] / max(p["full0"], 1e-12) - 1), abs(p["proj1"] / max(p["full1"], 1e-12) - 1))})
    csv = out_dir / f"locus_embedding_split_{args.drug}.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"\n{csv}")

    fig, handles, caption = build_figure(panels, args.drug, n_iso, args.dpi)

    # static before/after for slides
    for t, label in ((0.0, "encoder"), (1.0, "embedded")):
        set_t(handles, caption, t)
        png = out_dir / f"locus_embedding_split_{args.drug}_{label}.png"
        fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
        print(f"{png}")

    frames = args.frames

    def frame(i):
        t = 0.5 - 0.5 * np.cos(2 * np.pi * i / frames)     # 0 -> 1 -> 0, eased
        set_t(handles, caption, t)
        return ()

    anim = FuncAnimation(fig, frame, frames=frames,
                         interval=1000 / args.fps, blit=False)
    gif = out_dir / f"locus_embedding_split_{args.drug}.gif"
    anim.save(gif, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"{gif}  ({gif.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
