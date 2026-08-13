#!/usr/bin/env python3
"""
One figure per model: the real data going in, and its shape at every step out.

    python scripts/data_shapes.py                          # ISONIAZID, all modalities
    python scripts/data_shapes.py --scope joint            # the 19-locus joint input
    python scripts/data_shapes.py --arch mdcnn             # just one model
    python scripts/data_shapes.py --modalities dna regulatory --drug KANAMYCIN
    python scripts/data_shapes.py --from-data              # load the real FASTAs instead

Each figure is one architecture. The left panel is the data as loaded — one bar
per input block, to scale, with any zero padding shown — and the right panel
follows a tensor through that model, stage by stage, to the output logits.

Nothing is idealised or hand-derived. Block shapes come from the `branch_specs`
of runs that actually happened (results/experiments/…/*.json). Every downstream
shape is captured by running a real forward pass with forward hooks on each leaf
module, so the numbers are what the layers genuinely produced.

Writes to results/figures/data_shapes/:
    model_{arch}_{tag}.png|pdf     one per architecture — the slide
    shapes_{tag}.csv               every stage of every model, as numbers
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch,  # noqa: E402
                                Patch, PathPatch, Rectangle)
from matplotlib.path import Path as MPath                     # noqa: E402

from models import CisFusionNet, MDCNNNet, MultiDrugNet, MultiModalNet, SetFusionNet  # noqa: E402
from models.net import parse_block_key               # noqa: E402

EXP = PROJECT / "results" / "experiments"
OUTDIR = PROJECT / "results" / "figures" / "data_shapes"
ARCHS = ["late_fusion", "mdcnn", "setfusion", "cisfusion"]
ONE_LINER = {
    "late_fusion": "one encoder per block; the blocks only meet after being flattened",
    "mdcnn": "loci become CHANNELS on one zero-padded axis — layer 1 mixes every locus at once",
    "setfusion": "one encoder shared per modality; every block becomes one token, fused by attention",
    "cisfusion": "each promoter is glued onto its own gene before encoding, in transcription order",
}
MOD_COLOR = {"dna": "#0072B2", "protein": "#E69F00",
             "biophysical": "#009E73", "regulatory": "#CC79A7"}
PAD_COLOR = "#e2e2e2"
STAGE_FC = "#eef3f8"
STAGE_EC = "#9fb6c9"
HEAD_FC = "#f4ece2"
HEAD_EC = "#d0b48d"
INK = "#1A1A1A"
BATCH = 2

RC = {
    "figure.dpi": 130, "savefig.dpi": 220, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "font.size": 12.5, "axes.titlesize": 13.5, "axes.labelsize": 12.5,
    "axes.titleweight": "bold", "axes.edgecolor": "0.35",
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "0.25", "ytick.color": "0.25",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
}


# --------------------------------------------------------------------- specs
def specs_from_runs(drug, modalities, scope, runs=("full_run_v2", "full_run")):
    """{arch: (block_names, specs)} plus {arch: n_params}, from real runs.

    late_fusion and the per-locus archs see DIFFERENT block lists for the same
    data (late_fusion merges each modality's loci into one branch), so this is
    read per architecture rather than assumed shared."""
    tag = "+".join(m for m in ["dna", "protein", "biophysical", "regulatory"]
                   if m in modalities)
    cell = "_".join(modalities) if len(modalities) > 1 else modalities[0]
    cell = {"dna+protein+biophysical+regulatory": "all_modalities"}.get(tag, cell)
    out, params = {}, {}
    for arch in ARCHS:
        for run in runs:
            folder = (EXP / run / (f"multidrug_{cell}__{arch}" if scope == "joint"
                                   else f"{cell}__{arch}"))
            pat = "multidrug__*.json" if scope == "joint" else f"{drug}__*.json"
            hits = sorted(glob.glob(str(folder / pat)))
            if not hits:
                continue
            j = json.loads(Path(hits[0]).read_text())
            out[arch] = (j["blocks"], [tuple(s) for s in j["branch_specs"]])
            params[arch] = j.get("n_params")
            break
    return out, params


def specs_from_data(drug, modalities, scope):
    """Re-derive the specs by loading the real FASTAs (slow; needs no prior run)."""
    from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR
    from datasets import load_dataset, load_multidrug_dataset, loci_on_disk
    out = {}
    for arch in ARCHS:
        per_locus = arch in ("mdcnn", "setfusion", "cisfusion")
        kw = dict(regulatory_dir=REAL_REGULATORY_DIR,
                  per_modality_branch=not per_locus, verbose=False)
        if scope == "joint":
            d = load_multidrug_dataset(None, modalities, REAL_GENOTYPE_DIR,
                                       REAL_PHENOTYPE_CSV,
                                       loci=loci_on_disk(REAL_GENOTYPE_DIR), **kw)
        else:
            d = load_dataset(drug, modalities, REAL_GENOTYPE_DIR,
                             REAL_PHENOTYPE_CSV, **kw)
        out[arch] = ([b.name for b in d.blocks], [b.spec() for b in d.blocks])
    return out, {}


# ------------------------------------------------------------------- tracing
def build(arch, names, specs, n_drugs):
    keys = [parse_block_key(n) for n in names]
    drugs = [f"d{i}" for i in range(n_drugs)] if n_drugs > 1 else None
    if arch == "mdcnn":
        return MDCNNNet(specs, n_drugs=n_drugs, drug_names=drugs)
    if arch == "setfusion":
        return SetFusionNet(keys, specs, n_drugs=n_drugs, drug_names=drugs)
    if arch == "cisfusion":
        return CisFusionNet(keys, specs, n_drugs=n_drugs, drug_names=drugs)
    if n_drugs > 1:
        return MultiDrugNet(specs, drugs)
    return MultiModalNet(specs, n_drugs=1)


def record_shapes(model, specs):
    """{module_name: output_shape} from a real forward pass, via hooks."""
    rec, handles = {}, []

    def hook(name):
        def fn(mod, inp, out):
            # nn.MultiheadAttention returns (output, weights), so a tensor-only
            # check silently loses setfusion's cross-attention shape
            if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
                out = out[0]
            if torch.is_tensor(out):
                rec.setdefault(name, tuple(out.shape))
        return fn

    # every named module, not just leaves: nn.MultiheadAttention owns an
    # out_proj child, so a leaf-only filter skips setfusion's cross-attention
    for name, mod in model.named_modules():
        if name:
            handles.append(mod.register_forward_hook(hook(name)))
    model.eval()
    with torch.no_grad():
        model([torch.zeros(BATCH, c, l) for c, l in specs])
    for h in handles:
        h.remove()
    return rec


def shp(t, batch_label="B"):
    if t is None:
        return "?"
    return " × ".join([batch_label] + [f"{d:,}" for d in t[1:]])


def input_rows(arch, names, specs, model):
    """[(label, channels, real_len, padded_len, modality, block_idx)] as loaded.

    `block_idx` is the position in the model's own block list, which is what the
    branch wiring refers to. mdcnn is re-ordered so each trunk's blocks are
    contiguous — otherwise the fan-in lines cross for no reason."""
    keys = [parse_block_key(n) for n in names]
    if arch == "mdcnn":
        rows = []
        for idxs, L in zip(model.group_idx, model.group_len):
            for i in idxs:
                rows.append((names[i], specs[i][0], specs[i][1], L, keys[i][0], i))
        return rows
    return [(n, c, L, L, k[0], i)
            for i, (n, (c, L), k) in enumerate(zip(names, specs, keys))]


def branch_structure(arch, names, specs, model, rec, rows, n_drugs):
    """(branches, merge, head) — the graph the right panel draws.

    branches : [{label, srcs (row positions), in_shape, out_shape, note}]
    merge    : {label, shape, note} — where the branches finally meet
    head     : [(op, shape, note)] after the merge
    """
    pos = {r[5]: k for k, r in enumerate(rows)}          # block_idx -> row position
    br = []

    def g(k):
        return rec.get(k)

    if arch == "mdcnn":
        for t, (idxs, L) in enumerate(zip(model.group_idx, model.group_len)):
            c = specs[idxs[0]][0]
            out = g(f"trunks.{t}.pool2")
            br.append({
                "label": f"trunk {t + 1}  ({c}-channel loci)",
                "srcs": [pos[i] for i in idxs],
                "in_shape": f"B × {len(idxs)} × {c} × {L:,}",
                "out_shape": f"{shp(out)}  →  flatten {model.trunks[t].out_features:,}",
                "note": f"Conv2d({len(idxs)}→64, {c}×12)",
            })
        total = sum(t.out_features for t in model.trunks)
        merge = {"label": "concatenate the trunks", "shape": f"B × {total:,}",
                 "note": "the only place the channel groups meet"}
    elif arch == "setfusion":
        for m in model.modalities:
            srcs = [pos[i] for i, (mm, _l) in enumerate(model.block_keys) if mm == m]
            out = g(f"encoders.{m}.proj")
            br.append({
                "label": f"shared {m} encoder",
                "srcs": srcs,
                "in_shape": f"{len(srcs)} block(s), any length",
                "out_shape": f"{len(srcs)} × ({shp(out)})",
                "note": "ONE set of weights for every locus of this modality",
            })
        d = model.encoders[model.modalities[0]].out_features
        merge = {"label": "tokens + modality & locus embeddings → Transformer ×2",
                 "shape": f"B × {len(rows)} × {d}",
                 "note": "attention, not concatenation — order does not matter"}
    elif arch == "cisfusion":
        for u, (name, spec) in enumerate(zip(model.unit_names, model.unit_specs)):
            reg_i, dna_i, pass_i = model.units[u]
            srcs = [pos[i] for i in (reg_i, dna_i, pass_i) if i is not None]
            out = g(f"encoders.{u}.branch.pool2")
            paired = reg_i is not None and dna_i is not None
            br.append({
                "label": name,
                "kind": ("cis-unit (promoter ⊕ gene)" if paired
                         else f"{model.unit_kinds[u]} branch"),
                "srcs": srcs,
                "in_shape": f"B × {spec[0]} × {spec[1]:,}",
                "out_shape": (f"{shp(out)}  →  flatten "
                              f"{model.encoders[u].out_features:,}"),
                "note": ("promoter ⊕ gene, +1 segment channel" if paired
                         else "passes through unchanged"),
                "flat": model.encoders[u].out_features,
                "len": spec[1], "ch": spec[0],
            })
        total = sum(e.out_features for e in model.encoders)
        merge = {"label": "flatten every branch, concatenate",
                 "shape": f"B × {total:,}",
                 "note": "the only place the branches meet"}
    else:                                                # late_fusion
        for i, (c, L) in enumerate(specs):
            out = g(f"encoders.{i}.branch.pool2")
            br.append({
                "label": names[i],
                "srcs": [pos[i]],
                "in_shape": f"B × {c} × {L:,}",
                "out_shape": (f"{shp(out)}  →  flatten "
                              f"{model.encoders[i].out_features:,}"),
                "note": "its own encoder weights",
            })
        total = sum(e.out_features for e in model.encoders)
        merge = {"label": "flatten every branch, concatenate",
                 "shape": f"B × {total:,}",
                 "note": "the only place the branches meet"}

    # A joint cisfusion model has 57 branches — one lane each would be a 68-inch
    # figure. Collapse same-kind branches into one box carrying the count and the
    # span of shapes; the wiring stays truthful because the connectors still come
    # from every source block in the group.
    MAX_LANES = 8
    if len(br) > MAX_LANES and all("kind" in b for b in br):
        grouped, order = {}, []
        for b in br:
            k = b["kind"]
            if k not in grouped:
                grouped[k] = []
                order.append(k)
            grouped[k].append(b)
        collapsed = []
        for k in order:
            gs = grouped[k]
            lens = [b["len"] for b in gs]
            collapsed.append({
                "label": f"{k}  ×{len(gs)}",
                "srcs": sorted({s for b in gs for s in b["srcs"]}),
                "in_shape": (f"B × {gs[0]['ch']} × {min(lens):,}–{max(lens):,}"
                             if min(lens) != max(lens)
                             else f"B × {gs[0]['ch']} × {lens[0]:,}"),
                "out_shape": f"flatten {sum(b['flat'] for b in gs):,} in total",
                "note": gs[0]["note"] + f" · {len(gs)} separate encoders",
            })
        br = collapsed

    if arch == "setfusion":
        head = [(f"{n_drugs} drug quer{'y' if n_drugs == 1 else 'ies'} cross-attend",
                 shp(g("pool_attn")), "each drug reads the loci it needs"),
                ("Linear → ReLU", shp(g("fc1")), None),
                ("Linear → logits", f"B × {n_drugs}", "one per drug")]
    else:
        head = [("Linear → 256 → ReLU", shp(g("head.fc1")), None),
                ("Linear → 256 → ReLU", shp(g("head.fc2")), None),
                ("Linear → logits", f"B × {n_drugs}", "one per drug")]
    return br, merge, head


def stages_for(arch, names, specs, model, rec, n_drugs):
    """[(op, shape_str, note)] — the flow, in execution order."""
    n = len(specs)
    S = []

    def g(key):
        return rec.get(key)

    if arch in ("late_fusion", "cisfusion"):
        if arch == "cisfusion":
            units = list(zip(model.unit_names, model.unit_specs))
            npair = sum(1 for k in model.unit_kinds if k == "cis")
            widest = max(model.unit_specs, key=lambda s: s[1])
            S.append(("regroup into cis-units — promoter ⊕ its own gene",
                      f"{len(units)} branches, widest B × {widest[0]} × {widest[1]:,}",
                      f"{npair} paired; +1 segment channel"))
        # representative = the widest branch, since every branch has its own encoder
        i = int(np.argmax([s[1] for s in (model.unit_specs if arch == "cisfusion"
                                          else specs)]))
        p = f"encoders.{i}.branch"
        S += [
            ("Conv1d 1×1 stem  (→64)", shp(g(f"{p}.stem")), "per branch"),
            ("Conv1d k=12  (→64)", shp(g(f"{p}.conv1")), None),
            ("MaxPool 3", shp(g(f"{p}.pool1")), None),
            ("Conv1d k=3 (→32) ×2", shp(g(f"{p}.conv3")), None),
            ("MaxPool 3", shp(g(f"{p}.pool2")), "widest branch shown"),
        ]
        total = sum(e.out_features for e in model.encoders)
        S.append(("flatten each branch, concatenate all",
                  f"B × {total:,}",
                  f"{len(model.encoders)} branches — this is the ONLY place they meet"))
    elif arch == "mdcnn":
        gl = [(len(i), specs[i[0]][0], L) for i, L in zip(model.group_idx, model.group_len)]
        i = int(np.argmax([L for _, _, L in gl]))
        # Show the widest group only — the chain below follows that trunk, and
        # all three shapes on one line overruns the box.
        others = f" (widest of {len(gl)}: {', '.join(f'{c}ch' for _, c, _ in gl)})" \
            if len(gl) > 1 else ""
        S.append(("zero-pad to the group max, stack loci as channels",
                  f"B × {gl[i][0]} × {gl[i][1]} × {gl[i][2]:,}",
                  f"one group per channel count{others}"))
        p = f"trunks.{i}"
        S += [
            (f"Conv2d({gl[i][0]}→64, kernel {gl[i][1]}×12)", shp(g(f"{p}.conv_in")),
             "every locus mixed here, at layer 1"),
            ("squeeze + Conv1d k=12 (→64)", shp(g(f"{p}.conv1")), None),
            ("MaxPool 3", shp(g(f"{p}.pool1")), None),
            ("Conv1d k=3 (→32) ×2", shp(g(f"{p}.conv3")), None),
            ("MaxPool 3", shp(g(f"{p}.pool2")), "widest trunk shown"),
        ]
        total = sum(t.out_features for t in model.trunks)
        S.append(("flatten each trunk, concatenate", f"B × {total:,}", None))
    elif arch == "setfusion":
        mods = model.modalities
        p = f"encoders.{mods[0]}"
        S += [
            ("Conv1d 1×1 stem  (→64)", shp(g(f"{p}.stem")),
             f"ONE encoder per modality ({len(mods)}), shared by all its loci"),
            ("Conv1d k=12 + MaxPool 3", shp(g(f"{p}.pool1")), None),
            ("Conv1d k=3 (→32) ×2 + MaxPool 3", shp(g(f"{p}.pool2")), None),
            ("adaptive avg+max pool to 4 bins → Linear", shp(g(f"{p}.proj")),
             "length-agnostic: any locus → one fixed token"),
            ("stack tokens + modality & locus embeddings",
             f"B × {n} × {model.encoders[mods[0]].out_features}",
             "identity is carried, not implied by position"),
            ("Transformer encoder ×2", f"B × {n} × {model.encoders[mods[0]].out_features}",
             None),
            (f"{n_drugs} drug quer{'y' if n_drugs == 1 else 'ies'} cross-attend",
             shp(g("pool_attn")) if g("pool_attn") else
             f"B × {n_drugs} × {model.encoders[mods[0]].out_features}",
             "each drug reads the loci it needs"),
        ]

    # shared dense head
    if arch == "setfusion":
        S += [("Linear → ReLU", shp(g("fc1")), None),
              ("Linear → logits", f"B × {n_drugs}", "one per drug")]
    else:
        S += [("Linear → 256 → ReLU", shp(g("head.fc1")), None),
              ("Linear → 256 → ReLU", shp(g("head.fc2")), None),
              ("Linear → logits", f"B × {n_drugs}", "one per drug")]
    return S


# ------------------------------------------------------------------- drawing
def _nice_step(xmax, target=6):
    raw = max(xmax, 1) / target
    mag = 10 ** int(np.floor(np.log10(raw)))
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def draw_inputs(ax, rows, title):
    """The data as loaded — one bar per block, to scale, padding shown."""
    xmax = max(r[3] for r in rows)
    dense = len(rows) > 16
    bh = 0.68 if not dense else 0.82
    for i, (label, c, real, padded, mod, _bi) in enumerate(rows):
        y = len(rows) - 1 - i
        if padded > real:
            ax.add_patch(Rectangle((real, y - bh / 2), padded - real, bh,
                                   facecolor=PAD_COLOR, edgecolor="white",
                                   linewidth=1.2 if not dense else 0.4,
                                   hatch=None if dense else "///", zorder=2))
        ax.add_patch(Rectangle((0, y - bh / 2), real, bh,
                               facecolor=MOD_COLOR.get(mod, "#888"),
                               edgecolor="white",
                               linewidth=1.2 if not dense else 0.4, zorder=3))
        if not dense:
            ax.text(-xmax * 0.015, y, label, ha="right", va="center", fontsize=10.5)
            ax.text(padded + xmax * 0.01, y, f"{c}×{real:,}", ha="left",
                    va="center", fontsize=9.5, color="0.35")
    if dense:                          # label each modality run once instead
        runs_ = []
        for i, r in enumerate(rows):
            if not runs_ or runs_[-1][0] != r[4]:
                runs_.append([r[4], i, i])
            else:
                runs_[-1][2] = i
        for mod, i0, i1 in runs_:
            ymid = len(rows) - 1 - (i0 + i1) / 2
            ax.text(-xmax * 0.015, ymid, f"{mod} ×{i1 - i0 + 1}", ha="right",
                    va="center", fontsize=11, color=MOD_COLOR.get(mod, "#888"),
                    fontweight="bold")
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlim(-xmax * 0.34, xmax * 1.16)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, xmax * 1.10, _nice_step(xmax)))
    ax.ticklabel_format(axis="x", style="plain")
    ax.set_xlabel("positions (bp for DNA/regulatory, residues for protein/biophysical)")
    ax.set_title(title, loc="left", pad=10, fontsize=12.5)
    ax.grid(axis="x", color="0.92", zorder=0)
    ax.set_axisbelow(True)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)


def _connector(ax, x0, y0, x1, y1, color, lw=1.1, alpha=0.75):
    """A smooth left-to-right link from an input row into its branch box."""
    verts = [(x0, y0), (x0 + (x1 - x0) * 0.45, y0),
             (x0 + (x1 - x0) * 0.55, y1), (x1, y1)]
    ax.add_patch(PathPatch(MPath(verts, [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4,
                                         MPath.CURVE4]),
                           fill=False, edgecolor=color, linewidth=lw,
                           alpha=alpha, zorder=2))


def draw_graph(ax, rows, branches, merge, head, n_drugs, ax_h_in):
    """Branches as separate blocks, wired to the inputs they actually consume.

    Shares its y-scale with the input panel, so a branch box sits level with the
    bars feeding it and the connectors read as one continuous picture."""
    n = len(rows)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, n - 0.2)                 # matches draw_inputs
    ax.axis("off")
    BX0, BX1 = 0.13, 0.55                      # branch boxes
    MX = 0.635                                 # merge bar
    HX0, HX1 = 0.70, 0.995                     # head chain
    dense = n > 16
    total_links = sum(len(b["srcs"]) for b in branches)

    # ---- branch boxes in their own evenly spaced lane -----------------------
    # Deliberately NOT sized to the y-extent of their sources: a cisfusion unit
    # pairs a promoter with a gene that sit at opposite ends of the input list,
    # so an extent-sized box would span the whole panel and bury its neighbours.
    # Uniform lanes + connectors show the same wiring without the collisions.
    # Box heights are in DATA units but the text inside them is in points, so a
    # fixed data height shrinks under its own text as the panel gets taller
    # (73 input rows vs 8). Convert a target height in INCHES into data units.
    upi = (n + 0.6) / max(ax_h_in, 1.0)        # data units per inch
    nb = len(branches)
    top_y, bot_y = n - 1.0, 0.0
    lane = (top_y - bot_y) / max(nb, 1)
    box_h = min(0.80 * upi, lane * 0.86)
    centres = [top_y - lane * (i + 0.5) for i in range(nb)]
    if nb == 1:
        centres = [(top_y + bot_y) / 2]
    for b, yc in zip(branches, centres):
        ax.add_patch(FancyBboxPatch((BX0, yc - box_h / 2), BX1 - BX0, box_h,
                                    boxstyle="round,pad=0.004,rounding_size=0.015",
                                    facecolor=STAGE_FC, edgecolor=STAGE_EC,
                                    linewidth=1.4, zorder=3, mutation_aspect=0.5))
        three = box_h > 0.55 * upi
        ax.text(BX0 + 0.018, yc + (box_h * 0.30 if three else box_h * 0.24),
                b["label"], fontsize=10.5, fontweight="bold", va="center", zorder=4)
        ax.text(BX0 + 0.018, yc + (0 if three else -box_h * 0.26), b["in_shape"],
                fontsize=9.6, va="center", family="DejaVu Sans Mono",
                color="#20486b", zorder=4)
        if three:
            ax.text(BX0 + 0.018, yc - box_h * 0.30, b["out_shape"], fontsize=9.6,
                    va="center", family="DejaVu Sans Mono", color="#20486b",
                    zorder=4)
            # the note shares the label's line; combined width is what matters,
            # so fall back to the shape line and then to dropping it
            if b["note"]:
                if len(b["label"]) + len(b["note"]) <= 42:
                    ax.text(BX1 - 0.015, yc + box_h * 0.30, b["note"],
                            fontsize=8.6, va="center", ha="right", color="0.45",
                            style="italic", zorder=4)
                elif len(b["in_shape"]) + len(b["note"]) <= 40:
                    ax.text(BX1 - 0.015, yc, b["note"], fontsize=8.6,
                            va="center", ha="right", color="0.45",
                            style="italic", zorder=4)
        else:
            ax.text(BX1 - 0.015, yc + box_h * 0.24, b["out_shape"], fontsize=9.2,
                    va="center", ha="right", family="DejaVu Sans Mono",
                    color="#20486b", zorder=4)
        # fan-in: one line per source block, or a shaded wedge when too many
        if total_links <= 30:
            for s in b["srcs"]:
                _connector(ax, 0.0, n - 1 - s, BX0, yc,
                           MOD_COLOR.get(rows[s][4], "#999"))
        else:
            ys = [n - 1 - s for s in b["srcs"]]
            ax.add_patch(PathPatch(MPath(
                [(0.0, max(ys) + 0.5), (BX0, yc + box_h / 2),
                 (BX0, yc - box_h / 2), (0.0, min(ys) - 0.5)],
                [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO]),
                facecolor=MOD_COLOR.get(rows[b["srcs"][0]][4], "#999"),
                edgecolor="none", alpha=0.16, zorder=1))

    # ---- merge -------------------------------------------------------------
    ytop, ybot = max(centres), min(centres)
    for yc in centres:
        _connector(ax, BX1, yc, MX, (ytop + ybot) / 2, "0.55", lw=1.3, alpha=0.9)
    ax.plot([MX, MX], [ybot, ytop], color="#54728c",
            lw=3.2, solid_capstyle="round", zorder=4)
    ymid = (ytop + ybot) / 2
    # merge caption goes ABOVE the bar, clear of both the branch column and the
    # head column that flank it
    ax.text(MX, ytop + 0.62 * upi, merge["label"], fontsize=10.5, fontweight="bold",
            ha="center", va="bottom", zorder=5)
    ax.text(MX, ytop + 0.30 * upi, merge["shape"], fontsize=10,
            family="DejaVu Sans Mono", color="#20486b", ha="center",
            va="bottom", zorder=5)
    if merge["note"]:
        ax.text(MX, ybot - 0.34 * upi, merge["note"], fontsize=9, ha="center",
                va="top", color="0.45", style="italic", zorder=5)

    # ---- head chain, stacked downward from the merge ------------------------
    k = len(head)
    box_h = 0.62 * upi                          # same inch-based sizing
    gap = 0.26 * upi
    y = ymid + ((k - 1) / 2) * (box_h + gap)
    _connector(ax, MX, ymid, HX0, y, "0.55", lw=1.3, alpha=0.9)
    for i, (op, shape, note) in enumerate(head):
        ax.add_patch(FancyBboxPatch((HX0, y - box_h / 2), HX1 - HX0, box_h,
                                    boxstyle="round,pad=0.004,rounding_size=0.015",
                                    facecolor=HEAD_FC, edgecolor=HEAD_EC,
                                    linewidth=1.4, zorder=3, mutation_aspect=0.5))
        ax.text(HX0 + 0.018, y + box_h * 0.22, op, fontsize=10.5,
                fontweight="bold", va="center", zorder=4)
        ax.text(HX0 + 0.018, y - box_h * 0.24, shape, fontsize=10,
                family="DejaVu Sans Mono", color="#20486b", va="center", zorder=4)
        if note:
            ax.text(HX1 - 0.015, y - box_h * 0.24, note, fontsize=8.8,
                    va="center", ha="right", color="0.45", style="italic", zorder=4)
        if i < k - 1:
            y_next = y - (box_h + gap)
            ax.add_patch(FancyArrowPatch(((HX0 + HX1) / 2, y - box_h / 2),
                                         ((HX0 + HX1) / 2, y_next + box_h / 2),
                                         arrowstyle="-|>", mutation_scale=12,
                                         color="0.55", linewidth=1.3, zorder=2))
            y = y_next
    ax.text((HX0 + HX1) / 2, y - box_h / 2 - 0.12,
            f"↓  {n_drugs} logit(s) per isolate", ha="center", va="top",
            fontsize=10.5, color="0.35")


# tints for the locus embedding: near-neutral on purpose, so they never compete
# with the modality hues. Same locus -> same fill is the whole point of the panel.
LOCUS_TINT = ["#dfe3ea", "#ece0e6", "#e2eae0", "#efe7dc", "#e6e2ee"]


def fig_setfusion_detail(names, specs, model, rec, tag, subtitle, n_drugs):
    """Why setfusion needs a locus embedding, drawn one token at a time.

    The encoders are shared per MODALITY, so `dna:katG` and `regulatory:katG`
    come out of different encoders with nothing tying them together. The locus
    embedding is what re-attaches that identity: both tokens have the same
    locus_id, so the SAME learned vector is added to both, and attention can
    pair them. Position cannot do this job — the block count and order change
    from drug to drug."""
    keys = [parse_block_key(n) for n in names]
    d = model.encoders[model.modalities[0]].out_features
    # show the loci that actually demonstrate pairing (present in >1 modality)
    by_locus = {}
    for i, (m, l) in enumerate(keys):
        by_locus.setdefault(l or "<none>", []).append(i)
    interesting = sorted(by_locus, key=lambda l: -len(by_locus[l]))
    shown_loci = [l for l in interesting if len(by_locus[l]) > 1][:3] or interesting[:2]
    shown = [i for l in shown_loci for i in by_locus[l]]
    hidden = len(names) - len(shown)
    tint = {l: LOCUS_TINT[k % len(LOCUS_TINT)] for k, l in enumerate(shown_loci)}

    nrow = len(shown)
    h = 0.62 * nrow + 5.4
    with matplotlib.rc_context(RC):
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(16.5, h), gridspec_kw={"height_ratios": [nrow * 0.62, 2.5]})
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.9, nrow - 0.1)
        ax.axis("off")
        BW, BH = 0.085, 0.62                      # token rect size
        XE, XT, XM, XL, XK = 0.145, 0.325, 0.455, 0.585, 0.735

        def rect(x, y, w, fc, ec, label, sub=None, bold=True):
            ax.add_patch(FancyBboxPatch((x, y - BH / 2), w, BH,
                                        boxstyle="round,pad=0.003,rounding_size=0.012",
                                        facecolor=fc, edgecolor=ec, linewidth=1.3,
                                        zorder=3, mutation_aspect=0.45))
            ax.text(x + w / 2, y + (0.10 if sub else 0), label, ha="center",
                    va="center", fontsize=10.5,
                    fontweight="bold" if bold else "normal", zorder=4)
            if sub:
                ax.text(x + w / 2, y - 0.16, sub, ha="center", va="center",
                        fontsize=9, family="DejaVu Sans Mono", color="#20486b",
                        zorder=4)

        # shared encoders: one merged box per modality, spanning its rows
        runs = []
        for k, i in enumerate(shown):
            m = keys[i][0]
            if runs and runs[-1][0] == m:
                runs[-1][2] = k
            else:
                runs.append([m, k, k])
        for m, k0, k1 in runs:
            y0, y1 = nrow - 1 - k1, nrow - 1 - k0
            ax.add_patch(FancyBboxPatch((XE, y0 - BH / 2), 0.135,
                                        (y1 - y0) + BH,
                                        boxstyle="round,pad=0.003,rounding_size=0.012",
                                        facecolor=MOD_COLOR[m], alpha=0.16,
                                        edgecolor=MOD_COLOR[m], linewidth=1.4,
                                        zorder=3, mutation_aspect=0.45))
            ax.text(XE + 0.0675, (y0 + y1) / 2, f"shared\n{m} encoder",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color=MOD_COLOR[m], zorder=4)

        for k, i in enumerate(shown):
            y = nrow - 1 - k
            m, l = keys[i]
            l = l or "<none>"
            ax.text(0.135, y, names[i], ha="right", va="center", fontsize=10.5)
            ax.annotate("", (XE, y), (0.142, y), arrowprops=dict(
                arrowstyle="-|>", color=MOD_COLOR[m], lw=1.3))
            ax.annotate("", (XT, y), (XE + 0.135, y), arrowprops=dict(
                arrowstyle="-|>", color="0.55", lw=1.2))
            rect(XT, y, BW, MOD_COLOR[m], "none", "token", f"{d}-d")
            ax.text(XT + BW + 0.011, y, "+", ha="center", va="center",
                    fontsize=15, color="0.35", zorder=4)
            rect(XM, y, BW, MOD_COLOR[m], MOD_COLOR[m], f"m[{m[:4]}]", f"{d}-d")
            ax.text(XM + BW + 0.011, y, "+", ha="center", va="center",
                    fontsize=15, color="0.35", zorder=4)
            rect(XL, y, BW, tint[l], "#8d93a0", f"ℓ[{l}]", f"{d}-d")
            ax.text(XL + BW + 0.011, y, "=", ha="center", va="center",
                    fontsize=15, color="0.35", zorder=4)
            rect(XK, y, 0.105, "#f7f7f5", "#8d93a0", "keyed token", f"B × {d}")

        # bracket the rows that share a locus — the actual message
        for l in shown_loci:
            ks = [k for k, i in enumerate(shown) if (keys[i][1] or "<none>") == l]
            if len(ks) < 2:
                continue
            y0, y1 = nrow - 1 - max(ks), nrow - 1 - min(ks)
            xb = XK + 0.115
            ax.plot([xb, xb + 0.012, xb + 0.012, xb],
                    [y0 - 0.3, y0 - 0.3, y1 + 0.3, y1 + 0.3],
                    color="#8d93a0", lw=1.6, clip_on=False, zorder=4)
            ax.text(xb + 0.02, (y0 + y1) / 2,
                    f"same ℓ[{l}] vector added to all {len(ks)}\n"
                    f"→ attention can pair them", fontsize=9.8, va="center",
                    color="#4a4f59", zorder=4)

        ax.text(XM + BW / 2, nrow - 0.35, "learned per MODALITY", ha="center",
                fontsize=9.5, color="0.45", style="italic")
        ax.text(XL + BW / 2, nrow - 0.35, "learned per LOCUS", ha="center",
                fontsize=9.5, color="0.45", style="italic")
        ax.set_title(
            "1 · every block becomes one token, then gets its identity added back"
            + (f"    ({len(shown)} of {len(names)} tokens shown; "
               f"{hidden} more follow the same rule)" if hidden else ""),
            loc="left", pad=12, fontsize=12.5)

        # ---- downstream chain -------------------------------------------
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis("off")
        n_tok = len(names)
        chain = [
            ("all tokens as a SET", f"B × {n_tok} × {d}",
             "order carries no meaning —\nidentity is in the embeddings"),
            ("Transformer encoder ×2", f"B × {n_tok} × {d}",
             "tokens attend to each other;\nmatched loci can now interact"),
            (f"{n_drugs} drug quer{'y' if n_drugs == 1 else 'ies'} cross-attend",
             shp(rec.get("pool_attn")) or f"B × {n_drugs} × {d}",
             "each drug pools the loci\nIT needs, and only those"),
            ("Linear → ReLU → Linear", f"B × {n_drugs}",
             f"{n_drugs} logit(s) per isolate"),
        ]
        w = 0.205
        for j, (op, shape, note) in enumerate(chain):
            x = 0.012 + j * 0.2475
            ax2.add_patch(FancyBboxPatch((x, 0.30), w, 0.42,
                                         boxstyle="round,pad=0.006,rounding_size=0.02",
                                         facecolor=HEAD_FC if j == len(chain) - 1
                                         else STAGE_FC,
                                         edgecolor=HEAD_EC if j == len(chain) - 1
                                         else STAGE_EC,
                                         linewidth=1.4, zorder=3))
            ax2.text(x + w / 2, 0.615, op, ha="center", va="center",
                     fontsize=10.8, fontweight="bold", zorder=4)
            ax2.text(x + w / 2, 0.475, shape, ha="center", va="center",
                     fontsize=10, family="DejaVu Sans Mono", color="#20486b",
                     zorder=4)
            ax2.text(x + w / 2, 0.20, note, ha="center", va="top", fontsize=9.2,
                     color="0.45", style="italic", zorder=4)
            if j < len(chain) - 1:
                ax2.add_patch(FancyArrowPatch((x + w, 0.51), (x + 0.2475, 0.51),
                                              arrowstyle="-|>", mutation_scale=13,
                                              color="0.55", linewidth=1.3, zorder=2))
        ax2.set_title("2 · the set is fused, then each drug reads what it needs",
                      loc="left", pad=10, fontsize=12.5)

        fig.suptitle(f"setfusion — how a token keeps its identity\n{subtitle}",
                     x=0.5, y=1.0 + 0.62 / h, fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        OUTDIR.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(OUTDIR / f"model_setfusion_keying_{tag}.{ext}")
        plt.close(fig)
        print(f"  wrote {OUTDIR}/model_setfusion_keying_{tag}.png / .pdf")


def fig_model(arch, rows, branches, merge, head, tag, subtitle, n_params, n_drugs):
    # tall enough for the input bars AND for every branch box to hold its text
    h = max(0.40 * len(rows) if len(rows) <= 16 else 0.135 * len(rows),
            1.15 * len(branches), 1.0 * len(head)) + 2.2
    with matplotlib.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(19.5, h),
                                 gridspec_kw={"width_ratios": [1.0, 1.45],
                                              "wspace": 0.02})
        draw_inputs(axes[0], rows, "the data going in")
        draw_graph(axes[1], rows, branches, merge, head, n_drugs,
                   ax_h_in=max(h - 2.2, 1.5))
        axes[1].set_title(f"{len(branches)} branch(es) → merge → head",
                          loc="left", pad=10, fontsize=12.5)
        handles = [Patch(facecolor=MOD_COLOR[m], label=m) for m in MOD_COLOR
                   if any(r[4] == m for r in rows)]
        if any(r[3] > r[2] for r in rows):
            handles.append(Patch(facecolor=PAD_COLOR, hatch="///",
                                 label="zero padding"))
        # Both live above the axes, so they are placed in FIGURE fractions with
        # an explicit gap in inches — anchoring the legend to an axes made it
        # ride up into the title whenever the figure was short.
        fig.legend(handles=handles, loc="upper left",
                   bbox_to_anchor=(0.012, 1.0 + 0.26 / h), ncol=len(handles),
                   fontsize=10.5)
        pstr = f"{n_params/1e6:,.1f}M parameters" if n_params else ""
        fig.suptitle(f"{arch} — {ONE_LINER[arch]}\n{subtitle}"
                     + (f"   ·   {pstr}" if pstr else ""),
                     x=0.5, y=1.0 + 0.95 / h, fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        OUTDIR.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(OUTDIR / f"model_{arch}_{tag}.{ext}")
        plt.close(fig)
        print(f"  wrote {OUTDIR}/model_{arch}_{tag}.png / .pdf")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drug", default="ISONIAZID")
    ap.add_argument("--modalities", nargs="+",
                    default=["dna", "protein", "biophysical", "regulatory"])
    ap.add_argument("--scope", default="single", choices=["single", "joint"])
    ap.add_argument("--arch", nargs="+", default=ARCHS, choices=ARCHS)
    ap.add_argument("--from-data", action="store_true",
                    help="load the real FASTAs instead of reading recorded specs")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    global OUTDIR
    if args.outdir:
        OUTDIR = Path(args.outdir)
    drug = args.drug.upper()
    n_drugs = 11 if args.scope == "joint" else 1

    if args.from_data:
        print("loading real data (reads every FASTA — slow)…")
        blocks, params = specs_from_data(drug, args.modalities, args.scope)
    else:
        blocks, params = specs_from_runs(drug, args.modalities, args.scope)
    if not blocks:
        sys.exit("No recorded run matched that drug/modality/scope. "
                 "Try --from-data, or check results/experiments/.")

    subject = ("joint model, all 11 drugs, 19 loci" if args.scope == "joint"
               else f"{drug}, single-drug")
    subtitle = f"{subject} · {', '.join(args.modalities)}"
    tag = ("joint" if args.scope == "joint" else drug.lower()) + "_" + \
          ("all" if len(args.modalities) == 4 else "_".join(args.modalities))

    table = []
    for arch in args.arch:
        if arch not in blocks:
            print(f"  (skipping {arch}: no recorded run)")
            continue
        names, specs = blocks[arch]
        print(f"\n{arch}: tracing a real forward pass ({len(specs)} blocks)…")
        model = build(arch, names, specs, n_drugs)
        rec = record_shapes(model, specs)
        rows = input_rows(arch, names, specs, model)
        branches, merge, head = branch_structure(arch, names, specs, model, rec,
                                                 rows, n_drugs)
        if not params.get(arch):
            params[arch] = sum(p.numel() for p in model.parameters())
        for label, c, real, padded, mod, _i in rows:
            table.append({"arch": arch, "stage": "input", "what": label,
                          "shape": f"{c}x{real}", "padded_to": padded,
                          "modality": mod})
        for b in branches:
            table.append({"arch": arch, "stage": "branch", "what": b["label"],
                          "shape": f'{b["in_shape"]} -> {b["out_shape"]}',
                          "padded_to": "", "modality": ""})
        table.append({"arch": arch, "stage": "merge", "what": merge["label"],
                      "shape": merge["shape"], "padded_to": "", "modality": ""})
        for k, (op, shape, note) in enumerate(head, 1):
            table.append({"arch": arch, "stage": f"head{k}", "what": op,
                          "shape": shape, "padded_to": "", "modality": ""})
        print(f"  {len(rows)} block(s) → {len(branches)} branch(es) → "
              f"{merge['shape'].replace(' ', '')} → B×{n_drugs}")
        fig_model(arch, rows, branches, merge, head, tag, subtitle,
                  params.get(arch), n_drugs)
        if arch == "setfusion":
            # the locus keying is the whole idea of this model and does not fit
            # in one merge box — it gets its own figure
            fig_setfusion_detail(names, specs, model, rec, tag, subtitle, n_drugs)
        del model

    OUTDIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTDIR / f"shapes_{tag}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"\n  wrote {csv_path}")


if __name__ == "__main__":
    main()
