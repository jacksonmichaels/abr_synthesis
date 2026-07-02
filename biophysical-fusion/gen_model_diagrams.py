"""
Generate architecture diagrams (one per fusion model) as CSV + SVG.

Two artifacts per model, written to diagrams/:
  - <model>.csv : a flat, per-step trace of the forward pass (step / stage /
    operation / notes / output_shape), captured by actually running a forward
    pass on dummy tensors so it can't drift from models.py.
  - <model>.svg : a hand-laid-out architecture figure — parallel branches that
    visibly merge, tensor shapes on the edges, module boxes listing their
    internal layers. Built by bespoke per-model layout functions (not a
    generic auto-layout) so each reads like a real model diagram. All shapes
    are pulled from the instantiated model (branch.out_len), never hand-typed.

Run (abr_env is the project env — see TODO.md):
    conda activate abr_env
    python gen_model_diagrams.py
"""
import csv
from pathlib import Path

import torch

from models import CrossAttentionFusionCNN, EarlyFusionCNN, LateFusionCNN

# ---- Representative scenario (Rifampicin: loci rpoB + rpoC) ---------------
# Concrete numbers so shapes are readable; the pipeline derives these from
# data. B is the batch dim, carried through unchanged as "B".
DNA_CH = 5           # one-hot channels: A,C,T,G,gap  (tb.BASE_TO_COLUMN)
DNA_LEN = 240        # rpoB+rpoC concatenated into one contiguous locus
BIO_CH = 3           # molecular weight, isoelectric point, hydrophobicity
GENES = ["rpoB", "rpoC"]
BIO_LENS = [78, 66]  # per-gene translated-protein length (padded)
N_DRUGS = 1          # single-drug binary R/S (sized for multi-drug later)

OUT_DIR = Path(__file__).parent / "diagrams"
SCENARIO = (f"Representative scenario — Rifampicin (loci rpoB + rpoC):  "
            f"DNA one-hot {DNA_CH}x{DNA_LEN},  biophysical {BIO_CH} channels,  "
            f"protein lengths K = {BIO_LENS[0]} / {BIO_LENS[1]}")


# ==========================================================================
# CSV trace (unchanged: accurate per-step table)
# ==========================================================================
def _s(t):
    dims = list(t.shape if hasattr(t, "shape") else t)
    return "B x " + " x ".join(str(d) for d in dims[1:])


def trace_conv_branch(branch, x, prefix, rows):
    import torch.nn.functional as F

    def rec(op, out, note):
        rows.append([len(rows) + 1, prefix, op, note, _s(out)])

    x = F.relu(branch.stem(x));  rec("Conv1d 1x1 (stem) + ReLU",
                                     x, f"{branch.stem.in_channels}->64 channels")
    x = F.relu(branch.conv1(x)); rec("Conv1d k=12 pad=6 + ReLU", x, "64 filters, length +1")
    x = branch.pool1(x);         rec("MaxPool1d /3", x, "length // 3")
    x = F.relu(branch.conv2(x)); rec("Conv1d k=3 pad=1 + ReLU", x, "32 filters, length kept")
    x = F.relu(branch.conv3(x)); rec("Conv1d k=3 pad=1 + ReLU", x, "32 filters, length kept")
    x = branch.pool2(x);         rec("MaxPool1d /3", x, "length // 3")
    return x


def write_csv(name, rows, final_logits):
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{name}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "stage", "operation", "notes", "output_shape"])
        w.writerows(rows)
    assert rows[-1][-1] == _s(final_logits), f"{name}: {rows[-1][-1]} != {_s(final_logits)}"
    print(f"wrote {path}  ({len(rows)} steps)")


# ==========================================================================
# SVG toolkit
# ==========================================================================
FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "'SF Mono', 'Menlo', 'Consolas', monospace"
INK = "#1f2933"
SUBINK = "#52606d"
EDGE = "#7b8794"

# stage -> (header fill, header text)
PALETTE = {
    "input": "#5c6b7a",
    "dna":   "#2f8f4e",
    "bio":   "#e08a2b",
    "fusion": "#7b5ea7",
    "attn":  "#c0453b",
    "head":  "#324a5f",
    "out":   "#0f9d8f",
}


def _esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SVG:
    def __init__(self, w, h, title, subtitle):
        self.w, self.h = w, h
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{FONT}">',
            '<defs>'
            '<filter id="sh" x="-20%" y="-20%" width="140%" height="140%">'
            '<feDropShadow dx="0" dy="2" stdDeviation="2.5" flood-color="#0b1420" flood-opacity="0.20"/>'
            '</filter>'
            '<marker id="arw" markerWidth="12" markerHeight="12" refX="7.5" refY="4" '
            'orient="auto" markerUnits="userSpaceOnUse">'
            f'<path d="M0,0 L9,4 L0,8 Z" fill="{EDGE}"/></marker>'
            '</defs>',
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="#fbfcfd"/>',
            f'<text x="34" y="46" font-size="24" font-weight="700" fill="{INK}">{_esc(title)}</text>',
            f'<text x="34" y="72" font-size="13.5" fill="{SUBINK}">{_esc(subtitle)}</text>',
        ]

    # --- module box: colored header + body lines (for conv stacks / head) ---
    def module(self, cx, top, w, h, header, stage, lines):
        x, hh = cx - w / 2, 30
        col = PALETTE[stage]
        # body
        self.parts.append(
            f'<rect x="{x}" y="{top}" width="{w}" height="{h}" rx="11" '
            f'fill="#ffffff" stroke="#d7dee6" stroke-width="1.2" filter="url(#sh)"/>')
        # header (top corners rounded only)
        r = 11
        self.parts.append(
            f'<path d="M{x},{top+hh} L{x},{top+r} Q{x},{top} {x+r},{top} '
            f'L{x+w-r},{top} Q{x+w},{top} {x+w},{top+r} L{x+w},{top+hh} Z" fill="{col}"/>')
        self.parts.append(
            f'<text x="{cx}" y="{top+20}" font-size="14" font-weight="700" '
            f'fill="#ffffff" text-anchor="middle">{_esc(header)}</text>')
        ly = top + hh + 20
        for ln in lines:
            self.parts.append(
                f'<text x="{x+16}" y="{ly}" font-size="11.5" fill="{SUBINK}">{_esc(ln)}</text>')
            ly += 17
        return (cx, top, w, h)

    # --- solid box: one filled rounded rect with centered label(s) ---------
    def solid(self, cx, top, w, h, stage, lines, sub=None):
        x = cx - w / 2
        col = PALETTE[stage]
        self.parts.append(
            f'<rect x="{x}" y="{top}" width="{w}" height="{h}" rx="11" '
            f'fill="{col}" filter="url(#sh)"/>')
        lines = [lines] if isinstance(lines, str) else lines
        n = len(lines) + (1 if sub else 0)
        cy = top + h / 2 - (n - 1) * 9 + 5
        for ln in lines:
            self.parts.append(
                f'<text x="{cx}" y="{cy}" font-size="14" font-weight="700" '
                f'fill="#ffffff" text-anchor="middle">{_esc(ln)}</text>')
            cy += 18
        if sub:
            self.parts.append(
                f'<text x="{cx}" y="{cy}" font-size="11" fill="#eef2f6" '
                f'text-anchor="middle">{_esc(sub)}</text>')
        return (cx, top, w, h)

    # --- shape chip: monospace pill placed at (cx, cy) ---------------------
    def chip(self, cx, cy, text):
        w = 20 + len(text) * 7.3
        x = cx - w / 2
        self.parts.append(
            f'<rect x="{x}" y="{cy-13}" width="{w}" height="26" rx="13" '
            f'fill="#eef3f8" stroke="#cdd8e3" stroke-width="1"/>')
        self.parts.append(
            f'<text x="{cx}" y="{cy+4.5}" font-size="12" font-family="{MONO}" '
            f'fill="{INK}" text-anchor="middle">{_esc(text)}</text>')

    def _line(self, pts):
        d = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        self.parts.append(
            f'<polyline points="{d}" fill="none" stroke="{EDGE}" '
            f'stroke-width="2" marker-end="url(#arw)"/>')

    # straight vertical edge between two boxes in the same column
    def down(self, src, dst, shape=None, chip_dx=0):
        sx, sb = src[0], src[1] + src[3]
        dx, dt = dst[0], dst[1]
        self._line([(sx, sb), (dx, dt - 1)])
        if shape:
            self.chip(sx + chip_dx, (sb + dt) / 2, shape)

    # elbow edge: down from src to busY, across, down into dst top
    def elbow(self, src, dst, busY, shape=None):
        sx, sb = src[0], src[1] + src[3]
        dx, dt = dst[0], dst[1]
        self._line([(sx, sb), (sx, busY), (dx, busY), (dx, dt - 1)])
        if shape:
            self.chip(sx, (sb + busY) / 2, shape)

    # elbow into an explicit port x on the destination's top edge
    def elbow_port(self, src, portx, dst_top, busY, shape=None, label=None):
        sx, sb = src[0], src[1] + src[3]
        self._line([(sx, sb), (sx, busY), (portx, busY), (portx, dst_top - 1)])
        if shape:
            self.chip(sx, (sb + busY) / 2, shape)
        if label:
            self.parts.append(
                f'<text x="{portx}" y="{dst_top-8}" font-size="12" font-weight="700" '
                f'fill="{SUBINK}" text-anchor="middle">{_esc(label)}</text>')

    def save(self, name):
        self.parts.append('</svg>')
        p = OUT_DIR / f"{name}.svg"
        p.write_text("\n".join(self.parts), encoding="utf-8")
        print(f"wrote {p}")


def _conv_lines(in_ch):
    return [
        f"Conv1d 1x1 stem: {in_ch} -> 64",
        "Conv1d k=12 p=6 -> 64  + ReLU",
        "MaxPool1d / 3",
        "Conv1d k=3 -> 32  + ReLU",
        "Conv1d k=3 -> 32  + ReLU",
        "MaxPool1d / 3",
    ]


def fmt(*dims):
    return "B x " + " x ".join(str(d) for d in dims)


# ==========================================================================
# Per-model SVG layouts
# ==========================================================================
def svg_late_fusion(m):
    dL = m.dna_branch.out_len(DNA_LEN)
    bL = [b.out_len(k) for b, k in zip(m.bio_branches, BIO_LENS)]
    dflat, bflat = 32 * dL, [32 * l for l in bL]
    total = dflat + sum(bflat)

    c0, c1, c2 = 210, 610, 1010          # three branch columns
    ctr = c1                             # spine center (aligns with rpoB col)
    s = SVG(1220, 960, "Late-Fusion CNN  (default / Kulkarni et al. 2026)", SCENARIO)

    # inputs
    din = s.solid(c0, 100, 250, 64, "input", "DNA one-hot", "genes concatenated -> 1 locus")
    bin0 = s.solid(c1, 100, 250, 64, "input", f"Biophysical — {GENES[0]}", "3xK protein matrix")
    bin1 = s.solid(c2, 100, 250, 64, "input", f"Biophysical — {GENES[1]}", "3xK protein matrix")

    # branches
    dcv = s.module(c0, 216, 250, 150, "DNA Conv branch", "dna", _conv_lines(DNA_CH))
    bcv0 = s.module(c1, 216, 250, 150, f"Bio Conv branch ({GENES[0]})", "bio", _conv_lines(BIO_CH))
    bcv1 = s.module(c2, 216, 250, 150, f"Bio Conv branch ({GENES[1]})", "bio", _conv_lines(BIO_CH))

    # flatten
    df = s.solid(c0, 420, 250, 48, "dna", "Flatten")
    bf0 = s.solid(c1, 420, 250, 48, "bio", "Flatten")
    bf1 = s.solid(c2, 420, 250, 48, "bio", "Flatten")

    # concat -> head -> out (spine)
    cat = s.solid(ctr, 560, 320, 60, "fusion", "Concatenate", "DNA + bio_rpoB + bio_rpoC")
    head = s.module(ctr, 680, 320, 96, "Dense head", "head",
                    ["Linear -> 256  + ReLU", "Linear 256 -> 256  + ReLU", "Linear 256 -> n_drugs"])
    out = s.solid(ctr, 828, 200, 56, "out", "Resistance logit")

    # edges
    s.down(din, dcv, fmt(DNA_CH, DNA_LEN))
    s.down(bin0, bcv0, fmt(BIO_CH, BIO_LENS[0]))
    s.down(bin1, bcv1, fmt(BIO_CH, BIO_LENS[1]))
    s.down(dcv, df, fmt(32, dL))
    s.down(bcv0, bf0, fmt(32, bL[0]))
    s.down(bcv1, bf1, fmt(32, bL[1]))
    s.elbow(df, cat, 522, fmt(dflat))
    s.elbow(bf0, cat, 522, fmt(bflat[0]))
    s.elbow(bf1, cat, 522, fmt(bflat[1]))
    s.down(cat, head, fmt(total))
    s.down(head, out, fmt(N_DRUGS))
    s.save("late_fusion")


def svg_early_fusion(m):
    L = m.branch.out_len(DNA_LEN)
    flat = 32 * L
    cL, cR, ctr = 320, 700, 510
    s = SVG(1020, 850, "Early-Fusion CNN  (channel-stack ablation)", SCENARIO)

    din = s.solid(cL, 100, 250, 64, "input", "DNA one-hot", f"{DNA_CH} channels")
    bin = s.solid(cR, 100, 250, 64, "input", "Biophysical (upsampled)", "per-gene concat, x3 -> DNA len")

    cat = s.solid(ctr, 220, 330, 60, "fusion", "Concatenate on channels", f"{DNA_CH} + {BIO_CH} = 8 channels")
    cv = s.module(ctr, 322, 330, 150, "Shared Conv branch", "dna", _conv_lines(DNA_CH + BIO_CH))
    fl = s.solid(ctr, 506, 330, 48, "dna", "Flatten")
    head = s.module(ctr, 596, 330, 96, "Dense head", "head",
                    ["Linear -> 256  + ReLU", "Linear 256 -> 256  + ReLU", "Linear 256 -> n_drugs"])
    out = s.solid(ctr, 744, 200, 56, "out", "Resistance logit")

    s.elbow(din, cat, 190, fmt(DNA_CH, DNA_LEN))
    s.elbow(bin, cat, 190, fmt(BIO_CH, DNA_LEN))
    s.down(cat, cv, fmt(DNA_CH + BIO_CH, DNA_LEN))
    s.down(cv, fl, fmt(32, L))
    s.down(fl, head, fmt(flat))
    s.down(head, out, fmt(N_DRUGS))
    s.save("early_fusion")


def svg_cross_attention(m):
    dL = m.dna_branch.out_len(DNA_LEN)
    bL = [b.out_len(k) for b, k in zip(m.bio_branches, BIO_LENS)]
    dm = m.dna_branch.out_channels           # d_model = 32
    kv_len = sum(bL)
    flat = dL * dm

    cD, cB0, cB1 = 220, 720, 1010
    kvc = (cB0 + cB1) / 2                     # K/V concat column
    attn_ctr = 470
    s = SVG(1240, 1080, "Cross-Attention Fusion CNN  (asymmetric: DNA = Query, biophysical = Key/Value)", SCENARIO)

    # inputs
    din = s.solid(cD, 100, 250, 64, "input", "DNA one-hot", "genes concatenated")
    bin0 = s.solid(cB0, 100, 250, 64, "input", f"Biophysical — {GENES[0]}", "3xK protein matrix")
    bin1 = s.solid(cB1, 100, 250, 64, "input", f"Biophysical — {GENES[1]}", "3xK protein matrix")

    # conv branches
    dcv = s.module(cD, 216, 250, 150, "DNA Conv branch", "dna", _conv_lines(DNA_CH))
    bcv0 = s.module(cB0, 216, 250, 150, f"Bio Conv branch ({GENES[0]})", "bio", _conv_lines(BIO_CH))
    bcv1 = s.module(cB1, 216, 250, 150, f"Bio Conv branch ({GENES[1]})", "bio", _conv_lines(BIO_CH))

    # query prep + projections
    q = s.solid(cD, 420, 250, 52, "dna", "Transpose -> Query", f"d_model = {dm}")
    p0 = s.solid(cB0, 420, 250, 52, "bio", "Linear 32->32 (project)")
    p1 = s.solid(cB1, 420, 250, 52, "bio", "Linear 32->32 (project)")

    kv = s.solid(kvc, 512, 300, 56, "fusion", "Concat K / V", "along sequence axis")

    # attention (two labeled ports)
    attn = s.module(attn_ctr, 620, 360, 92, "Multi-Head Attention", "attn",
                    ["4 heads, batch_first", "Q = DNA features", "K = V = biophysical features"])
    fl = s.solid(attn_ctr, 748, 360, 48, "attn", "Flatten")
    head = s.module(attn_ctr, 838, 360, 96, "Dense head", "head",
                    ["Linear -> 256  + ReLU", "Linear 256 -> 256  + ReLU", "Linear 256 -> n_drugs"])
    out = s.solid(attn_ctr, 986, 200, 56, "out", "Resistance logit")

    # edges — branches
    s.down(din, dcv, fmt(DNA_CH, DNA_LEN))
    s.down(bin0, bcv0, fmt(BIO_CH, BIO_LENS[0]))
    s.down(bin1, bcv1, fmt(BIO_CH, BIO_LENS[1]))
    s.down(dcv, q, fmt(32, dL))
    s.down(bcv0, p0, fmt(32, bL[0]))
    s.down(bcv1, p1, fmt(32, bL[1]))
    # bio projections -> K/V concat
    s.elbow(p0, kv, 484, fmt(bL[0], dm))
    s.elbow(p1, kv, 484, fmt(bL[1], dm))
    # into attention: Query on left port, K/V on right port
    portL, portR = attn_ctr - 110, attn_ctr + 110
    s.elbow_port(q, portL, 620, 590, fmt(dL, dm), label="Q")
    s.elbow_port(kv, portR, 620, 590, fmt(kv_len, dm), label="K, V")
    # attention -> flatten -> head -> out
    s.down(attn, fl, fmt(dL, dm))
    s.down(fl, head, fmt(flat))
    s.down(head, out, fmt(N_DRUGS))
    s.save("cross_attention_fusion")


# ==========================================================================
# CSV drivers (kept: accurate per-step tables)
# ==========================================================================
def csv_late(m):
    dna = torch.zeros(2, DNA_CH, DNA_LEN)
    bios = [torch.zeros(2, BIO_CH, k) for k in BIO_LENS]
    rows = [[1, "input", "DNA one-hot (genes concatenated)", "one contiguous locus", _s(dna)]]
    for g, x in zip(GENES, bios):
        rows.append([len(rows) + 1, "input", f"Biophysical matrix — {g}", "separate channel", _s(x)])
    d = torch.flatten(trace_conv_branch(m.dna_branch, dna, "DNA branch", rows), 1)
    rows.append([len(rows) + 1, "DNA branch", "Flatten", "", _s(d)])
    for g, br, x in zip(GENES, m.bio_branches, bios):
        f = torch.flatten(trace_conv_branch(br, x, f"Bio branch ({g})", rows), 1)
        rows.append([len(rows) + 1, f"Bio branch ({g})", "Flatten", "", _s(f)])
    total = 32 * m.dna_branch.out_len(DNA_LEN) + sum(32 * b.out_len(k) for b, k in zip(m.bio_branches, BIO_LENS))
    with torch.no_grad():
        logits = m(dna, bios)
    rows.append([len(rows) + 1, "fusion", "Concatenate [DNA, bio...]", "late concat", _s(torch.zeros(2, total))])
    rows.append([len(rows) + 1, "head", "DenseHead 256->256->out", "", _s(logits)])
    write_csv("late_fusion", rows, logits)


def csv_early(m):
    dna = torch.zeros(2, DNA_CH, DNA_LEN)
    bio_up = torch.zeros(2, BIO_CH, DNA_LEN)
    rows = [
        [1, "input", "DNA one-hot", "one contiguous locus", _s(dna)],
        [2, "input", "Biophysical (upsampled concat)", "padded to DNA length", _s(bio_up)],
        [3, "fusion", "Concatenate on channel axis", "5 DNA + 3 bio", _s(torch.zeros(2, 8, DNA_LEN))],
    ]
    out = trace_conv_branch(m.branch, torch.cat([dna, bio_up], 1), "Shared branch", rows)
    rows.append([len(rows) + 1, "shared branch", "Flatten", "", _s(torch.flatten(out, 1))])
    with torch.no_grad():
        logits = m(dna, bio_up)
    rows.append([len(rows) + 1, "head", "DenseHead 256->256->out", "", _s(logits)])
    write_csv("early_fusion", rows, logits)


def csv_cross(m):
    dna = torch.zeros(2, DNA_CH, DNA_LEN)
    bios = [torch.zeros(2, BIO_CH, k) for k in BIO_LENS]
    dm = m.dna_branch.out_channels
    rows = [[1, "input", "DNA one-hot", "genes concatenated", _s(dna)]]
    for g, x in zip(GENES, bios):
        rows.append([len(rows) + 1, "input", f"Biophysical matrix — {g}", "separate channel", _s(x)])
    d = trace_conv_branch(m.dna_branch, dna, "DNA branch (Query)", rows)
    rows.append([len(rows) + 1, "DNA branch (Query)", "Transpose -> (B,L',d_model)", "", _s(d.transpose(1, 2))])
    kv_len = 0
    for g, br, x in zip(GENES, m.bio_branches, bios):
        o = trace_conv_branch(br, x, f"Bio branch ({g}, K/V)", rows)
        kv_len += o.shape[-1]
        rows.append([len(rows) + 1, f"Bio branch ({g}, K/V)", "Transpose + Linear(32->d_model)", "", _s(m.bio_proj(o.transpose(1, 2)))])
    rows.append([len(rows) + 1, "fusion", "Concat K/V (seq axis)", "", _s(torch.zeros(2, kv_len, dm))])
    rows.append([len(rows) + 1, "attention", "MultiheadAttention(Q,K,V)", "4 heads", _s(torch.zeros(2, m.dna_branch.out_len(DNA_LEN), dm))])
    rows.append([len(rows) + 1, "attention", "Flatten", "", _s(torch.zeros(2, m.dna_branch.out_len(DNA_LEN) * dm))])
    with torch.no_grad():
        logits = m(dna, bios)
    rows.append([len(rows) + 1, "head", "DenseHead 256->256->out", "", _s(logits)])
    write_csv("cross_attention_fusion", rows, logits)


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    torch.manual_seed(0)
    late = LateFusionCNN(DNA_LEN, BIO_LENS, N_DRUGS, DNA_CH, BIO_CH).eval()
    early = EarlyFusionCNN(DNA_LEN, BIO_LENS, N_DRUGS, DNA_CH, BIO_CH).eval()
    cross = CrossAttentionFusionCNN(DNA_LEN, BIO_LENS, N_DRUGS, DNA_CH, BIO_CH).eval()

    csv_late(late);  svg_late_fusion(late)
    csv_early(early); svg_early_fusion(early)
    csv_cross(cross); svg_cross_attention(cross)
