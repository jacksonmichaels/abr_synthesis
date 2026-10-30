"""Plot style, table style and paired significance tests for notebooks/results.ipynb.

Kept out of the notebook so its cells show analysis, not plumbing.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy import stats

# ------------------------------------------------------------------ palette
# Colour encodes identity only: blue = established architectures, orange =
# variant-token aggregators, rust = locusfusion, aqua = non-network baselines.
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, RUST, AQUA = "#2a78d6", "#eb6834", "#a8420f", "#1baf7a"
RUST_LIGHT = "#d9895c"                            # locusfusion_v2, beside v1's rust
PALE, PALE_BLUE = "#f0efec", "#cde2fb"
TINTS = ["#f8d3c1", "#ef9264", ORANGE]            # one model at three settings
AQUA_RAMP = ["#0d5c43", "#1e9370", "#63bb9d"]     # validated light->dark on SURFACE
FAM_TINT = {"established": PALE_BLUE, "experimental": "#fbdcce", "baseline": "#cdf0e2"}

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "figure.dpi": 120,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlepad": 10,
})


def tidy(ax, xgrid=True, ygrid=False):
    ax.set_axisbelow(True)
    ax.grid(axis="x" if xgrid else "y", visible=xgrid or ygrid)
    if ygrid:
        ax.grid(axis="y", visible=True)
    return ax


_OFFSETS = [(0, 17), (0, -29), (0, 35), (0, -47), (58, 5), (-58, 5),
            (58, -13), (-58, -13), (0, 53), (0, -65), (72, 22), (-72, 22)]


def label_points(ax, xs, ys, texts, fontsize=7.2, pad=3.0):
    """Direct labels with leaders, greedily placed so none overlaps another.

    Call after the axis scale and limits are final (placement is in pixels).
    """
    xs, ys, texts = list(xs), list(ys), list(texts)
    trans = ax.transData
    placed = []
    for x, y in zip(xs, ys):
        px, py = trans.transform((x, y))
        placed.append((px - 7, py - 7, px + 7, py + 7))

    def overlaps(box):
        l, b, r, t = box
        return any(not (r < L or l > R or t < B or b > T) for L, B, R, T in placed)

    for i in np.argsort(np.asarray(xs, dtype=float)):
        lines = texts[i].split("\n")
        w = max(len(ln) for ln in lines) * fontsize * 0.60
        h = len(lines) * fontsize * 1.45
        px, py = trans.transform((xs[i], ys[i]))
        for dx, dy in _OFFSETS:
            cy = py + dy + (h / 2 if dy >= 0 else -h / 2)
            cx = px + dx + (0 if dx == 0 else (w / 2 if dx > 0 else -w / 2))
            box = (cx - w / 2 - pad, cy - h / 2 - pad, cx + w / 2 + pad, cy + h / 2 + pad)
            if not overlaps(box):
                break
        placed.append(box)
        ax.annotate(texts[i], (xs[i], ys[i]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=fontsize, color=INK2,
                    ha="center" if dx == 0 else ("right" if dx < 0 else "left"),
                    va="bottom" if dy >= 0 else "top", linespacing=1.4, zorder=5,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.6,
                                    shrinkA=0, shrinkB=5))


# ------------------------------------------------------------------ tables
def tint(bg):
    """CSS for a tinted cell: the text colour is set with the background, so the
    cell reads the same in a light or a dark editor theme."""
    return f"background-color:{bg}; color:{INK}"


def style_table(styler, caption=None, hide_index=True):
    """Theme-neutral table style: no text colour or background on plain cells,
    so they inherit the editor's theme. Only cells given `tint()` carry colour."""
    s = styler.hide(axis="index") if hide_index else styler
    if caption:
        s = s.set_caption(caption)
    return s.set_table_styles([
        {"selector": "caption",
         "props": "caption-side:top; font-size:11px; text-align:left; "
                  "padding-bottom:6px; opacity:0.8"},
        {"selector": "th", "props": "font-weight:600; font-size:11px; text-align:left"},
        {"selector": "td", "props": "font-size:11px; padding:3px 8px"},
    ], overwrite=False)


# ------------------------------------------------------------------ tests
K, RHO, MARGIN = 5, 1 / 4, 0.01      # folds, n_val/n_train, equivalence margin


def _listify(x):
    return [x] if isinstance(x, str) else list(x)


def fold_matrix(frame, drugs, task, models, modsets):
    """drug x fold CV AUC, averaged fold-by-fold over the (model, modset) cells."""
    models, modsets = _listify(models), _listify(modsets)
    s = frame[(frame.task == task) & frame.model.isin(models) & frame.modset.isin(modsets)]
    out = {}
    for d in drugs:
        v = [np.asarray(t) for t in s[s.drug == d].fold_aucs if len(t) == K]
        if len(v) != len(models) * len(modsets):
            raise ValueError(f"{task}/{models}/{modsets}/{d}: {len(v)} cells with folds")
        out[d] = np.mean(v, axis=0)
    return pd.DataFrame(out).T


def _t_result(M, se, df):
    h = stats.t.ppf(0.975, df) * se
    return dict(delta=M, lo=M - h, hi=M + h, df=df,
                p=2 * stats.t.sf(abs(M / se), df),
                p_eq=max(stats.t.sf((M + MARGIN) / se, df),
                         stats.t.cdf((M - MARGIN) / se, df)))


def corrected_t(diff):
    """Nadeau-Bengio corrected t on the macro of a drug x fold difference matrix."""
    D = len(diff)
    v = (1 / K + RHO) * diff.var(axis=1, ddof=1)
    df = v.sum() ** 2 / ((v ** 2).sum() / (K - 1))
    return _t_result(diff.mean(axis=1).mean(), np.sqrt(v.sum()) / D, df)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]).clip(max=1)
    out = np.empty(len(p)); out[o] = adj
    return out


def _signed_rank_cdf(n):
    """P(W+ <= w) for w = 0..n(n+1)/2 under the exact signed-rank null."""
    c = np.zeros(n * (n + 1) // 2 + 1); c[0] = 1.0
    for i in range(1, n + 1):
        c[i:] = c[i:] + c[:-i].copy()
    return np.cumsum(c) / c.sum()


def hodges_lehmann(x, alpha=0.05):
    """Hodges-Lehmann shift and the exact CI that inverts the signed-rank test:
    the median of the Walsh averages, bounded by their k-th smallest and largest."""
    x = np.asarray(x, float)
    n = len(x)
    walsh = np.sort((x[:, None] + x[None, :])[np.triu_indices(n)] / 2)
    k = int(np.searchsorted(_signed_rank_cdf(n), alpha / 2, side="right"))  # P(W<=k-1)<=a/2
    # n <= 5 cannot reach alpha (smallest two-sided p is 2/2^n): the range is the
    # widest interval there is, at 1 - 2/2^n coverage (93.75% for one drug's 5 folds)
    k = max(k, 1)
    return np.median(walsh), walsh[k - 1], walsh[len(walsh) - k]


class WilcoxonComparisons:
    """The paper's test: two-sided exact Wilcoxon signed-rank over the 55
    fold-paired differences (11 drugs x 5 folds), BIG-TB's method, with Holm
    across the family. Interval = Hodges-Lehmann shift and its exact 95% CI over
    the same 55 differences. A non-significant row is "not significant", never a
    claim of equivalence."""

    def __init__(self, frame, drugs):
        self.frame, self.drugs, self.rows = frame, list(drugs), []

    def folds(self, group, label, a, b, drugs=None):
        """a, b = (task, model(s), modset(s)); fold-paired test of a − b."""
        dr = drugs or self.drugs
        diff = fold_matrix(self.frame, dr, *a) - fold_matrix(self.frame, dr, *b)
        x = diff.values.ravel()
        hl, lo, hi = hodges_lehmann(x)
        per = diff.mean(axis=1)
        self.rows.append(dict(
            group=group, comparison=label, n=len(x), delta=per.mean(), hl=hl, lo=lo, hi=hi,
            wins=f"{int((per > 0).sum())}/{len(per)}",
            p=stats.wilcoxon(x, method="exact" if len(x) <= 60 else "auto").pvalue))

    def drugs_only(self, group, label, a, b):
        """Wilcoxon over the per-drug Δ, for cells on different CV partitions
        (joint vs single-drug), where folds do not pair."""
        per = (fold_matrix(self.frame, self.drugs, *a).mean(axis=1)
               - fold_matrix(self.frame, self.drugs, *b).mean(axis=1))
        hl, lo, hi = hodges_lehmann(per.values)
        self.rows.append(dict(
            group=group, comparison=label + " (paired on drugs)", n=len(per),
            delta=per.mean(), hl=hl, lo=lo, hi=hi,
            wins=f"{int((per > 0).sum())}/{len(per)}",
            p=stats.wilcoxon(per.values, method="exact").pvalue))

    def table(self):
        t = pd.DataFrame(self.rows)
        t["p_holm"] = holm(t.p)
        t["verdict"] = np.where(t.p_holm < 0.05, "significant", "not significant")
        return t


class Comparisons:
    """Collects paired comparisons, then Holm-adjusts and assigns verdicts."""

    def __init__(self, frame, drugs):
        self.frame, self.drugs, self.rows = frame, list(drugs), []

    def _add(self, sec, label, paired, per, r, p_bigtb):
        r["p_drug"] = stats.wilcoxon(per, method="exact").pvalue if len(per) > 1 else np.nan
        r["wins"] = f"{int((per > 0).sum())}/{len(per)}"
        self.rows.append(dict(sec=sec, comparison=label, paired=paired, p_bigtb=p_bigtb, **r))

    def folds(self, sec, label, a, b, drugs=None):
        """a, b = (task, model(s), modset(s)); fold-paired test of a − b."""
        dr = drugs or self.drugs
        diff = fold_matrix(self.frame, dr, *a) - fold_matrix(self.frame, dr, *b)
        self._add(sec, label, "folds", diff.mean(axis=1), corrected_t(diff),
                  stats.wilcoxon(diff.values.ravel(), method="exact").pvalue)

    def drugs_only(self, sec, label, a, b):
        """Drug-paired t, for cells on different CV partitions (joint vs single)."""
        per = (fold_matrix(self.frame, self.drugs, *a).mean(axis=1)
               - fold_matrix(self.frame, self.drugs, *b).mean(axis=1))
        n = len(per)
        self._add(sec, label, "drugs", per,
                  _t_result(per.mean(), per.std(ddof=1) / np.sqrt(n), n - 1), np.nan)

    def table(self):
        t = pd.DataFrame(self.rows)
        t["p_holm"] = holm(t.p)
        t["p_eq_holm"] = holm(t.p_eq)
        m = t.p_drug.notna()
        t.loc[m, "p_drug_holm"] = holm(t.loc[m, "p_drug"])
        diff, eq = t.p_holm < 0.05, t.p_eq_holm < 0.05
        t["verdict"] = np.select([diff & eq, diff, eq],
                                 ["different, within ±0.01", "different", "equivalent"],
                                 "unresolved")
        return t


# Studentised range / sqrt(2), alpha = 0.05, k = 2..12 (Demsar 2006, Table 5)
NEMENYI_Q05 = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
               9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}


# ------------------------------------------------------------------ the paper's family
# One definition, read by results.ipynb §9 and display_figures.ipynb. Changing a
# row changes every Holm p -- record why in SIGNIFICANCE_TESTS.md.
_PD, _L19, _WG = "single-drug", "single-drug, all loci", "single-drug, whole genome"
_A, _EST3 = "all_modalities", ["mdcnn", "cisfusion", "late_fusion"]
PAPER_GROUPS = ("LASSO vs. CNN", "LocusFusion", "Locus set", "Modalities", "Model size")
# tested and tabled, not drawn in the paper figure
FIG_DROP = ["locusfusion: fixed − old tokenizer · 19 loci · 5 modsets",
            "lasso: onehot − delta · 19 loci", "lasso_delta: whole genome − 19 loci"]


def paper_family(frame, drugs, modsets):
    """The paper's 14 comparisons as a WilcoxonComparisons table. `frame` must
    carry the old-tokenizer locusfusion rows as model "locusfusion_old"."""
    G1, G2, G3, G4, G5 = PAPER_GROUPS
    L19, PD, WG, A, EST3, M5 = _L19, _PD, _WG, _A, _EST3, modsets
    c = WilcoxonComparisons(frame, drugs)
    c.folds(G1, "lasso_onehot − mdcnn · 19 loci", (L19, "lasso_onehot", "dna"), (L19, "mdcnn", A))
    c.folds(G1, "lasso_onehot − mdcnn · per-drug loci", (PD, "lasso_onehot", "dna"), (PD, "mdcnn", A))
    c.folds(G1, "lasso: onehot − delta · 19 loci",
            (L19, "lasso_onehot", "dna"), (L19, "lasso_delta", "dna"))
    c.folds(G2, "locusfusion − lasso_onehot · 19 loci",
            (L19, "locusfusion", A), (L19, "lasso_onehot", "dna"))
    c.folds(G2, "locusfusion − mdcnn · 19 loci · 5 modsets",
            (L19, "locusfusion", M5), (L19, "mdcnn", M5))
    c.folds(G2, "locusfusion: fixed − old tokenizer · 19 loci · 5 modsets",
            (L19, "locusfusion", M5), (L19, "locusfusion_old", M5))
    c.folds(G3, "established: 19 − per-drug loci · 5 modsets", (L19, EST3, M5), (PD, EST3, M5))
    c.folds(G3, "locusfusion: 19 − per-drug loci · 5 modsets",
            (L19, "locusfusion", M5), (PD, "locusfusion", M5))
    c.folds(G3, "lasso_onehot: 19 − per-drug loci",
            (L19, "lasso_onehot", "dna"), (PD, "lasso_onehot", "dna"))
    c.folds(G3, "lasso_delta: whole genome − 19 loci",
            (WG, "lasso_genome", "dna"), (L19, "lasso_delta", "dna"))
    c.folds(G4, "established: all_modalities − dna · per-drug loci", (PD, EST3, A), (PD, EST3, "dna"))
    # model size: every pair of established architectures, smaller − larger
    npar = (frame[(frame.task == L19) & frame.modset.isin(M5)]
            .groupby("model").n_params.median())
    order = sorted(EST3, key=npar.get)
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            c.folds(G5, f"{a} ({npar[a] / 1e6:.1f}M) − {b} ({npar[b] / 1e6:.1f}M) · 19 loci · 5 modsets",
                    (L19, a, M5), (L19, b, M5))
    return c.table()


def _pshort(v):
    return "p < 0.0001" if v < 1e-4 else f"p = {v:.4f}"


def plot_sig_by_claim(sig, drop=FIG_DROP, labels=None):
    """Forest plot of a paper_family() table, one panel per claim. Filled marker =
    significant after Holm. `labels` maps a comparison to its display name.
    Returns the figure; the caller saves and shows it."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    s = sig[~sig.comparison.isin(drop)]
    if labels:
        s = s.assign(comparison=s.comparison.map(lambda c: labels.get(c, c)))
    groups = list(dict.fromkeys(s.group))
    fig, axes = plt.subplots(
        len(groups), 1, sharex=True,
        figsize=(10.5, 0.42 * len(s) + 0.5 * len(groups) + 0.9),
        gridspec_kw=dict(height_ratios=[(s.group == g).sum() for g in groups]))
    for ax, g in zip(np.atleast_1d(axes), groups):
        f = s[s.group == g].iloc[::-1].reset_index(drop=True)
        y = np.arange(len(f))
        ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
        for yi, r in zip(y, f.itertuples()):
            ax.plot([r.lo, r.hi], [yi, yi], color=INK2, lw=1.6, zorder=2,
                    solid_capstyle="round")
            ax.scatter([r.hl], [yi], s=46, zorder=3, linewidth=1.6, edgecolor=BLUE,
                       color=BLUE if r.verdict == "significant" else SURFACE)
            ax.annotate(f"{r.wins}   {_pshort(r.p_holm)}", (1.0, yi),
                        xycoords=("axes fraction", "data"), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=7.6, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels(f.comparison, fontsize=8)
        ax.set_ylim(-0.6, len(f) - 0.4)
        ax.set_title(g, loc="left", fontsize=9.5, pad=4)
        tidy(ax)
    axes = np.atleast_1d(axes)
    axes[-1].set_xlabel("Hodges–Lehmann Δ CV AUC, a − b, with 95% CI")
    axes[0].annotate("drugs a > b   Holm p", (1.0, 1.0), xycoords="axes fraction",
                     xytext=(6, 4), textcoords="offset points", fontsize=7.6,
                     color=MUTED, va="bottom")
    fig.suptitle(f"Paired Δ CV AUC by claim, {len(s)} of {len(sig)} paper comparisons",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.legend([Line2D([0], [0], marker="o", lw=0, markerfacecolor=fc, markeredgecolor=BLUE,
                       markeredgewidth=1.6, markersize=7) for fc in (BLUE, SURFACE)],
               ["significant (Holm p < 0.05)", "not significant"],
               loc="lower left", ncol=2, bbox_to_anchor=(0.01, -0.01))
    fig.tight_layout(rect=(0, 0.03, 0.86, 0.97))
    return fig
