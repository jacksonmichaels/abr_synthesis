"""One palette for every manuscript figure. Change a colour here and every figure follows.

Each colour has exactly one meaning across the paper; never reuse a hex for
something else. Backgrounds are pure white so figures sit cleanly on the page.

    from paper_style import MODEL, MODALITY, NEUTRAL, ramp, diverging_cmap, apply

Validated (dataviz six checks, OKLab ΔE×100, Machado 2009 CVD, on #ffffff):
  MODEL     all pairs: CVD ≥ 10.9, normal vision ≥ 19.1 (scatter-safe)
  MODALITY  all pairs: CVD ≥ 14.5, normal vision ≥ 18.1
  MODALITY vs MODEL: normal vision ≥ 11 (closest: DNA plum vs CNN blue)
Mustard (protein) is below 3:1 contrast on white; every figure that uses it has
a legend, and the numbers are printed beside it in the notebook.
"""
import math

# ---- neutrals: pure greys on pure white -------------------------------------
NEUTRAL = {
    "surface": "#ffffff",
    "ink": "#1a1a1a",        # titles, values
    "ink2": "#4d4d4d",       # tick labels, secondary text
    "muted": "#8c8c8c",      # reference lines (published baselines), quiet labels
    "grid": "#e8e8e8",
    "axis": "#bdbdbd",
}

# ---- models: identity of the thing being compared ---------------------------
MODEL = {
    "locusfusion": "#b26235",   # copper
    "cnn": "#2f6094",           # steel blue: MD-CNN, cis-fusion, late-fusion
    "lasso": "#309d8b",         # teal: every LASSO variant
}
_MODEL_OF = {"locusfusion": "locusfusion", "mdcnn": "cnn", "cisfusion": "cnn",
             "late_fusion": "cnn", "cnn": "cnn", "established": "cnn"}


def model_color(name):
    """Colour for a model id as it appears in the results frame."""
    if name.startswith("lasso") or name.startswith("sparse_baseline"):
        return MODEL["lasso"]
    return MODEL[_MODEL_OF.get(name, "cnn")]


# ---- modalities: what the input carries --------------------------------------
MODALITY = {
    "dna": "#784d96",           # plum
    "protein": "#c0a241",       # mustard
    "biophysical": "#c57fa6",   # dusty pink
    "regulatory": "#4f7638",    # moss
    "uncalled": "#a39e95",      # warm grey: uncalled-base (N / X) channels
}

# ---- ramps and the diverging map ---------------------------------------------
def _hex_to_oklch(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
    a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
    bb = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
    return L, math.hypot(a, bb), math.degrees(math.atan2(bb, a))


def _oklch_to_hex(L, C, H):
    a, b = C * math.cos(math.radians(H)), C * math.sin(math.radians(H))
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
           -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
           -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s]
    f = lambda v: 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055
    return "#" + "".join(f"{round(max(0, min(1, f(max(0, v)))) * 255):02x}" for v in rgb)


def ramp(color, n, light=0.88):
    """n steps of one hue, dark -> light, ending at the given colour's darkest
    useful step. Use for ordered categories of one model (e.g. own WHO genes ->
    other drugs' genes -> outside the catalogue)."""
    L, C, H = _hex_to_oklch(color)
    lo = min(L, 0.45)
    out = []
    for i in range(n):
        t = i / max(n - 1, 1)
        Li = lo + t * (light - lo)
        out.append(_oklch_to_hex(Li, C * (1 - 0.65 * t), H))
    return out


def diverging_cmap(pos=MODEL["locusfusion"], neg=NEUTRAL["muted"]):
    """Diverging map for 'A minus B' heatmaps: A ahead takes A's colour, behind is
    grey, zero is white."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("paper_div", [neg, NEUTRAL["surface"], pos])


def apply(results_helpers=None):
    """Matplotlib defaults for manuscript figures, and point the shared plot
    helpers (results_helpers.tidy / label_points) at the same neutrals."""
    import matplotlib as mpl
    N = NEUTRAL
    mpl.rcParams.update({
        "figure.facecolor": N["surface"], "axes.facecolor": N["surface"],
        "savefig.facecolor": N["surface"], "savefig.transparent": False,
        "text.color": N["ink"], "axes.labelcolor": N["ink2"], "axes.edgecolor": N["axis"],
        "xtick.color": N["ink2"], "ytick.color": N["ink2"], "grid.color": N["grid"],
        "legend.frameon": False,
    })
    if results_helpers is not None:
        RH = results_helpers
        RH.SURFACE, RH.INK, RH.INK2, RH.MUTED = N["surface"], N["ink"], N["ink2"], N["muted"]
        RH.GRID, RH.AXIS = N["grid"], N["axis"]
