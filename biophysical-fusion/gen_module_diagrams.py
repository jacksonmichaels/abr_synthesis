"""
Concept diagrams for the four "module choices" from experimental_plan.pdf
(these are training / interpretability schemes, not tensor forward-passes, so
they're hand-laid-out rather than traced):

  1. adversarial_lineage_decoupling  — gradient-reversal head -> lineage-agnostic latent
  2. mic_abr_multitask               — shared latent, dual heads (binary R/S + ordinal MIC)
  3. causal_probing                  — inject canonical-variant latents into a frozen model
  4. staged_frozen_training          — pretrain encoders -> freeze -> fuse + train heads

Writes diagrams/<name>.svg (+ .png preview if cairosvg is available).
Run:  conda activate abr_env && python gen_module_diagrams.py
"""
from pathlib import Path

OUT_DIR = Path(__file__).parent / "diagrams"

FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "'SF Mono', 'Menlo', 'Consolas', monospace"
INK, SUBINK, EDGE = "#1f2933", "#52606d", "#7b8794"
GRAD = "#c0453b"   # adversarial / gradient color

PALETTE = {
    "input":  "#5c6b7a",
    "enc":    "#2f8f4e",
    "latent": "#7b5ea7",
    "head":   "#324a5f",
    "abr":    "#0f9d8f",
    "mic":    "#c77d0a",
    "lineage": "#3f7cac",
    "grl":    "#c0453b",
    "probe":  "#7b5ea7",
    "frozen": "#5b6670",
    "out":    "#0f9d8f",
    "note":   "#eef3f8",
}


def _esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SVG:
    def __init__(self, w, h, title, subtitle):
        self.w, self.h = w, h
        self.p = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{FONT}">',
            '<defs>'
            '<filter id="sh" x="-20%" y="-20%" width="140%" height="140%">'
            '<feDropShadow dx="0" dy="2" stdDeviation="2.5" flood-color="#0b1420" flood-opacity="0.20"/>'
            '</filter>'
            f'<marker id="arw" markerWidth="12" markerHeight="12" refX="7.5" refY="4" orient="auto" markerUnits="userSpaceOnUse">'
            f'<path d="M0,0 L9,4 L0,8 Z" fill="{EDGE}"/></marker>'
            f'<marker id="arwR" markerWidth="12" markerHeight="12" refX="7.5" refY="4" orient="auto" markerUnits="userSpaceOnUse">'
            f'<path d="M0,0 L9,4 L0,8 Z" fill="{GRAD}"/></marker>'
            f'<marker id="arwP" markerWidth="12" markerHeight="12" refX="7.5" refY="4" orient="auto" markerUnits="userSpaceOnUse">'
            f'<path d="M0,0 L9,4 L0,8 Z" fill="{PALETTE["probe"]}"/></marker>'
            '</defs>',
            f'<rect width="{w}" height="{h}" fill="#fbfcfd"/>',
            f'<text x="34" y="46" font-size="23" font-weight="700" fill="{INK}">{_esc(title)}</text>',
            f'<text x="34" y="71" font-size="13.5" fill="{SUBINK}">{_esc(subtitle)}</text>',
        ]

    # ---- containers ----
    def panel(self, x, y, w, h, title, accent):
        self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" '
                      f'fill="#ffffff" stroke="{accent}" stroke-width="1.4" stroke-opacity="0.5"/>')
        self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="34" rx="14" fill="{accent}" fill-opacity="0.14"/>')
        self.p.append(f'<rect x="{x}" y="{y+20}" width="{w}" height="14" fill="#ffffff"/>')
        self.p.append(f'<text x="{x+w/2}" y="{y+22}" font-size="14" font-weight="700" '
                      f'fill="{accent}" text-anchor="middle">{_esc(title)}</text>')

    # ---- boxes ----
    def module(self, cx, top, w, h, header, stage, lines, frozen=False, tag=None):
        x, hh = cx - w / 2, 30
        col = PALETTE["frozen"] if frozen else PALETTE[stage]
        self.p.append(f'<rect x="{x}" y="{top}" width="{w}" height="{h}" rx="11" '
                      f'fill="#ffffff" stroke="#d7dee6" stroke-width="1.2" filter="url(#sh)"/>')
        r = 11
        self.p.append(f'<path d="M{x},{top+hh} L{x},{top+r} Q{x},{top} {x+r},{top} '
                      f'L{x+w-r},{top} Q{x+w},{top} {x+w},{top+r} L{x+w},{top+hh} Z" fill="{col}"/>')
        tx = cx
        if frozen:
            self._lock(x + 15, top + 15, "#ffffff"); tx = cx + 9
        self.p.append(f'<text x="{tx}" y="{top+20}" font-size="13.5" font-weight="700" '
                      f'fill="#ffffff" text-anchor="middle">{_esc(header)}</text>')
        ly = top + hh + 19
        for ln in lines:
            self.p.append(f'<text x="{x+15}" y="{ly}" font-size="11.5" fill="{SUBINK}">{_esc(ln)}</text>')
            ly += 16
        if tag:
            self._tag(x + w, top, *tag)
        return (cx, top, w, h)

    def solid(self, cx, top, w, h, stage, lines, sub=None, tag=None, frozen=False):
        x = cx - w / 2
        col = PALETTE["frozen"] if frozen else PALETTE[stage]
        self.p.append(f'<rect x="{x}" y="{top}" width="{w}" height="{h}" rx="11" fill="{col}" filter="url(#sh)"/>')
        lines = [lines] if isinstance(lines, str) else lines
        n = len(lines) + (1 if sub else 0)
        cy = top + h / 2 - (n - 1) * 9 + 5
        for ln in lines:
            self.p.append(f'<text x="{cx}" y="{cy}" font-size="14" font-weight="700" '
                          f'fill="#ffffff" text-anchor="middle">{_esc(ln)}</text>')
            cy += 18
        if sub:
            self.p.append(f'<text x="{cx}" y="{cy}" font-size="11" fill="#eef2f6" text-anchor="middle">{_esc(sub)}</text>')
        if tag:
            self._tag(x + w, top, *tag)
        return (cx, top, w, h)

    # ---- glyphs ----
    def _lock(self, cx, cy, color):
        self.p.append(f'<rect x="{cx-6}" y="{cy-1}" width="12" height="9" rx="1.6" fill="{color}"/>')
        self.p.append(f'<path d="M{cx-3.5},{cy-1} v-2.5 a3.5,3.5 0 0 1 7,0 v2.5" '
                      f'fill="none" stroke="{color}" stroke-width="1.6"/>')

    def _tag(self, xr, yt, text, color):
        w = 16 + len(text) * 6.6
        x = xr - w + 6
        self.p.append(f'<rect x="{x}" y="{yt-11}" width="{w}" height="21" rx="10.5" '
                      f'fill="{color}" stroke="#ffffff" stroke-width="1.4"/>')
        self.p.append(f'<text x="{x+w/2}" y="{yt+3.5}" font-size="10.5" font-weight="700" '
                      f'fill="#ffffff" text-anchor="middle">{_esc(text)}</text>')

    def inject(self, cx, cy, color):
        self.p.append(f'<circle cx="{cx}" cy="{cy}" r="11" fill="#ffffff" stroke="{color}" stroke-width="2"/>')
        self.p.append(f'<path d="M{cx-5},{cy} h10 M{cx},{cy-5} v10" stroke="{color}" stroke-width="2"/>')

    def chip(self, cx, cy, text):
        w = 20 + len(text) * 7.3
        self.p.append(f'<rect x="{cx-w/2}" y="{cy-13}" width="{w}" height="26" rx="13" '
                      f'fill="#eef3f8" stroke="#cdd8e3"/>')
        self.p.append(f'<text x="{cx}" y="{cy+4.5}" font-size="12" font-family="{MONO}" '
                      f'fill="{INK}" text-anchor="middle">{_esc(text)}</text>')

    def caption(self, x, y, w, lines, accent):
        h = 20 + len(lines) * 17
        self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" '
                      f'fill="{accent}" fill-opacity="0.08" stroke="{accent}" stroke-opacity="0.35"/>')
        ly = y + 22
        for ln in lines:
            self.p.append(f'<text x="{x+14}" y="{ly}" font-size="12" fill="{INK}">{_esc(ln)}</text>')
            ly += 17

    # ---- edges ----
    def _poly(self, pts, color, dashed, mid, width=2):
        d = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        das = ' stroke-dasharray="6 5"' if dashed else ''
        self.p.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{width}"{das} '
                      f'marker-end="url(#{mid})"/>')

    def down(self, src, dst, shape=None, color=EDGE, mid="arw", dashed=False):
        sx, sb, dx, dt = src[0], src[1] + src[3], dst[0], dst[1]
        self._poly([(sx, sb), (dx, dt - 1)], color, dashed, mid)
        if shape:
            self.chip(sx, (sb + dt) / 2, shape)

    def elbow(self, src, dst, busY, shape=None, color=EDGE, mid="arw", dashed=False):
        sx, sb, dx, dt = src[0], src[1] + src[3], dst[0], dst[1]
        self._poly([(sx, sb), (sx, busY), (dx, busY), (dx, dt - 1)], color, dashed, mid)
        if shape:
            self.chip(sx, (sb + busY) / 2, shape)

    def line(self, pts, color=EDGE, mid="arw", dashed=False, label=None, lx=None, ly=None):
        self._poly(pts, color, dashed, mid)
        if label:
            self.p.append(f'<text x="{lx}" y="{ly}" font-size="11.5" font-weight="600" '
                          f'fill="{color}">{_esc(label)}</text>')

    def legend(self, x, y, items):
        self.p.append(f'<rect x="{x}" y="{y}" width="270" height="{18+len(items)*20}" rx="8" '
                      f'fill="#ffffff" stroke="#dde3ea"/>')
        cy = y + 22
        for color, dashed, txt in items:
            das = ' stroke-dasharray="6 5"' if dashed else ''
            self.p.append(f'<line x1="{x+14}" y1="{cy-4}" x2="{x+44}" y2="{cy-4}" '
                          f'stroke="{color}" stroke-width="2.2"{das}/>')
            self.p.append(f'<text x="{x+52}" y="{cy}" font-size="11.5" fill="{INK}">{_esc(txt)}</text>')
            cy += 20

    def save(self, name):
        self.p.append('</svg>')
        (OUT_DIR / f"{name}.svg").write_text("\n".join(self.p), encoding="utf-8")
        print(f"wrote {OUT_DIR / f'{name}.svg'}")


# ==========================================================================
# 1. Adversarial lineage decoupling
# ==========================================================================
def d_adversarial():
    s = SVG(1180, 760, "Adversarial Lineage Decoupling",
            "A gradient-reversal head penalizes lineage information in the shared latent -> lineage-agnostic ABR features")
    inp = s.solid(560, 92, 300, 56, "input", "Genotype features", "DNA / protein / … encoders")
    enc = s.module(560, 186, 300, 84, "Shared encoder / trunk", "enc",
                   ["conv branches -> shared MLP", "produces latent z"])
    z = s.solid(560, 314, 210, 50, "latent", "Shared latent  z")

    abr_h = s.solid(392, 424, 250, 56, "head", "ABR head", "Linear -> sigmoid")
    abr_o = s.solid(392, 536, 250, 52, "abr", "Resistant / Susceptible")

    grl = s.solid(772, 410, 290, 66, "grl",
                  ["Gradient Reversal Layer"], sub="forward: identity   ·   backward: x (-lambda)")
    lin_h = s.solid(772, 524, 250, 52, "lineage", "Lineage head")
    lin_o = s.solid(772, 620, 250, 52, "lineage", "Lineage  (L1 … L7)")

    s.down(inp, enc)
    s.down(enc, z, "z in R^d")
    s.elbow(z, abr_h, 398)
    s.elbow(z, grl, 398)
    s.down(abr_h, abr_o)
    s.down(grl, lin_h)
    s.down(lin_h, lin_o)
    # adversarial gradient path (dashed red, routed up the right margin into the encoder)
    s.line([(930, 424), (1120, 424), (1120, 228), (712, 228)], color=GRAD, mid="arwR", dashed=True)
    s.p.append(f'<text x="1112" y="330" font-size="11.5" font-weight="600" fill="{GRAD}" '
               f'text-anchor="end">back-prop: sign flipped</text>')
    s.p.append(f'<text x="1112" y="347" font-size="11.5" font-weight="600" fill="{GRAD}" '
               f'text-anchor="end">-> removes lineage signal from z</text>')
    s.legend(40, 640, [(EDGE, False, "forward / data flow"),
                       (GRAD, True, "adversarial gradient (reversed)")])
    s.save("adversarial_lineage_decoupling")


# ==========================================================================
# 2. Predict both MIC and ABR (multi-task)
# ==========================================================================
def d_multitask():
    s = SVG(1060, 640, "Multi-Task: Predict MIC and ABR jointly",
            "Two heads off one shared latent -> features organize around mutation severity, not just a binary label")
    inp = s.solid(530, 92, 300, 56, "input", "Genotype features")
    enc = s.module(530, 186, 300, 84, "Shared encoder", "enc",
                   ["conv branches -> shared MLP", "produces latent z"])
    z = s.solid(530, 314, 300, 52, "latent", "Shared latent  z", sub="organizes around biological severity")

    abr_h = s.solid(360, 430, 250, 56, "head", "ABR head", "Linear -> sigmoid")
    abr_o = s.solid(360, 542, 250, 52, "abr", "R / S   (binary)")
    mic_h = s.solid(700, 430, 250, 56, "head", "MIC head", "Linear -> ordinal bins")
    mic_o = s.solid(700, 542, 250, 52, "mic", "MIC bin  (qualitative)", tag=("labels sparse -> masked loss", "#c77d0a"))

    s.down(inp, enc)
    s.down(enc, z)
    s.elbow(z, abr_h, 400)
    s.elbow(z, mic_h, 400)
    s.down(abr_h, abr_o)
    s.down(mic_h, mic_o)
    s.save("mic_abr_multitask")


# ==========================================================================
# 3. Causal interpretability via probing
# ==========================================================================
def d_probing():
    s = SVG(1220, 700, "Causal Interpretability via Latent Probing",
            "Inject a canonical-variant latent into a FROZEN model and read the change in prediction — a causal probe, not correlational SHAP")
    inp = s.solid(430, 100, 280, 54, "input", "Genotype features")
    enc = s.module(430, 188, 280, 78, "Frozen encoder", "enc",
                   ["weights fixed (already trained)"], frozen=True)
    z = s.solid(430, 310, 250, 56, "latent", "Latent  z")
    s.inject(430 + 125, 310 + 28, PALETTE["probe"])   # ⊕ on right edge of z
    head = s.module(430, 420, 280, 70, "Frozen prediction head", "head", ["weights fixed"], frozen=True)
    pred = s.solid(430, 540, 280, 54, "out", "Prediction  y-hat")

    bank = s.solid(910, 196, 300, 74,
                   "probe", ["Canonical variant bank"], sub="sparse ref-alt latent per variant")

    s.down(inp, enc)
    s.down(enc, z)
    s.down(z, head)
    s.down(head, pred)
    # injection: bank -> ⊕ on latent z
    s.line([(910, 270), (910, 338), (568, 338)], color=PALETTE["probe"], mid="arwP", dashed=True)
    s.p.append(f'<text x="900" y="300" font-size="11.5" font-weight="600" '
               f'fill="{PALETTE["probe"]}" text-anchor="end">inject variant latent (intervention)</text>')
    s.caption(760, 400, 420, [
        "Compare y-hat to the un-injected baseline:",
        "   Δy-hat  =  causal effect of that variant.",
        "Read-only — no weights updated.",
        "Contrast SHAP, which is correlational and",
        "confounded by co-resistance to other drugs.",
    ], PALETTE["probe"])
    s.save("causal_probing")


# ==========================================================================
# 4. Staged / frozen training
# ==========================================================================
def d_staged():
    s = SVG(1340, 620, "Staged Training with Frozen Encoders",
            "Pretrain sequence encoders -> freeze them -> add the lineage branch and train only the fusion head")
    A = PALETTE["enc"]
    # Stage 1
    s.panel(40, 108, 380, 440, "Stage 1 — pretrain encoders", A)
    s.solid(230, 200, 250, 74, "enc", ["DNA encoder"], sub="trained on sequences",
            tag=("trainable", PALETTE["enc"]))
    s.solid(230, 330, 250, 74, "enc", ["Protein encoder"], sub="trained on sequences",
            tag=("trainable", PALETTE["enc"]))
    s.p.append(f'<text x="230" y="470" font-size="11.5" fill="{SUBINK}" text-anchor="middle">'
               f'each learns its own sequence representation</text>')
    # Stage 2
    s.panel(460, 108, 380, 440, "Stage 2 — freeze", PALETTE["frozen"])
    s.solid(650, 200, 250, 74, "enc", ["DNA encoder"], sub="weights locked", frozen=True,
            tag=("frozen", PALETTE["frozen"]))
    s.solid(650, 330, 250, 74, "enc", ["Protein encoder"], sub="weights locked", frozen=True,
            tag=("frozen", PALETTE["frozen"]))
    s.p.append(f'<text x="650" y="470" font-size="11.5" fill="{SUBINK}" text-anchor="middle">'
               f'latents reused, no longer updated</text>')
    # Stage 3
    s.panel(880, 108, 420, 440, "Stage 3 — fuse and train heads", PALETTE["latent"])
    dna = s.solid(955, 180, 120, 64, "enc", ["DNA enc"], frozen=True)
    prot = s.solid(1090, 180, 120, 64, "enc", ["Protein enc"], frozen=True)
    lin = s.solid(1225, 180, 120, 64, "lineage", ["Lineage"], tag=("train", PALETTE["lineage"]))
    cat = s.solid(1090, 300, 300, 46, "latent", "Concatenate latents")
    dense = s.solid(1090, 380, 300, 60, "head", "Dense head", sub="only this is trained now")
    out = s.solid(1090, 480, 220, 46, "out", "ABR prediction")
    s.elbow(dna, cat, 278)
    s.elbow(prot, cat, 278)
    s.elbow(lin, cat, 278)
    s.down(cat, dense)
    s.down(dense, out)
    # stage-to-stage arrows
    s.line([(424, 328), (456, 328)], color=EDGE, mid="arw")
    s.line([(844, 328), (876, 328)], color=EDGE, mid="arw")
    s.p.append(f'<text x="440" y="318" font-size="11" font-weight="600" fill="{SUBINK}" text-anchor="middle">freeze</text>')
    s.p.append(f'<text x="860" y="318" font-size="11" font-weight="600" fill="{SUBINK}" text-anchor="middle">reuse</text>')
    s.save("staged_frozen_training")


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    d_adversarial()
    d_multitask()
    d_probing()
    d_staged()
    # optional PNG previews
    try:
        import cairosvg
        for n in ["adversarial_lineage_decoupling", "mic_abr_multitask",
                  "causal_probing", "staged_frozen_training"]:
            cairosvg.svg2png(url=str(OUT_DIR / f"{n}.svg"),
                             write_to=str(OUT_DIR / f"{n}.png"), scale=1.5, background_color="white")
        print("rendered PNG previews")
    except ImportError:
        print("cairosvg not installed — SVGs written, skipping PNG previews")
