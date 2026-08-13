"""
Push ONE real isolate through every network we train and draw what happens to
its tensors — every layer, every shape, every parameter count.

Covers the whole live matrix of models (``--traces`` picks a subset):

  sd_dna              single-drug, DNA only            late_fusion (MultiModalNet)
  sd_multimodal       single-drug, all modalities      late_fusion, one branch per MODALITY
  sd_multimodal_loci  single-drug, all modalities      late_fusion, one branch per LOCUS
  sd_dna_mdcnn        single-drug, DNA only            mdcnn (MDCNNNet = BIG-TB SD-CNN)
  md_dna              multi-drug,  DNA only            late_fusion (MultiDrugNet)
  md_multimodal       multi-drug,  all modalities      late_fusion, one branch per MODALITY
  md_dna_mdcnn        multi-drug,  DNA only            mdcnn (MDCNNNet = BIG-TB MD-CNN)
  sd_setfusion        single-drug, all modalities      setfusion (SetFusionNet), per LOCUS
  md_setfusion        multi-drug,  all modalities      setfusion (SetFusionNet), per LOCUS
  sd_cisfusion        single-drug, all modalities      cisfusion (CisFusionNet), per LOCUS
  md_cisfusion        multi-drug,  all modalities      cisfusion (CisFusionNet), per LOCUS

For each one it writes two files to --outdir:

  {trace}.png   flow diagram — input blocks -> per-branch stacks -> fusion ->
                dense head -> logits, with the tensor shape, op signature,
                parameter count and output statistics at every step
  {trace}.txt   the same trace as text, complete (every branch, not just the
                ones the diagram shows) and greppable

The weights are FRESHLY INITIALISED — we never checkpoint, so the logits are
noise. Shapes, parameter counts and dataflow are the point; the sample is real
so the input side (block lengths, padding, channel layout) is exactly what the
training jobs see.

Loading real data costs minutes and GBs: one load per scope (single-drug,
multi-drug) is shared across that scope's traces, and --synthetic swaps in
fixtures for a fast wiring check.

Examples (run from the project root):
    python scripts/trace_models.py --synthetic              # fast, tiny fixtures
    python scripts/trace_models.py                          # all traces, real data
    python scripts/trace_models.py --traces sd_dna sd_dna_mdcnn
    python scripts/trace_models.py --drug RIFAMPICIN --md-drugs ISONIAZID RIFAMPICIN
"""
import argparse
import contextlib
import sys
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from bigtb_ref import (REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV,  # noqa: E402
                       REAL_REGULATORY_DIR)
from datasets import (ALL_DRUGS, MODALITIES, load_dataset,  # noqa: E402
                      load_multidrug_dataset, union_loci, union_regulatory)
from datasets.base import merge_modality_blocks  # noqa: E402
from datasets.fixtures import build_fixture_dataset  # noqa: E402
from models import (CisFusionNet, MDCNNNet, MultiDrugNet,  # noqa: E402
                    MultiModalNet, SetFusionNet)

# --- palette: three validated categorical hues by OP ROLE + neutral ink ------
# (dataviz reference palette slots 1-3, the set that clears the all-pairs
# floors; text stays ink, the hue rides the border/tint of each box.)
INK, INK_DIM, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
ROLE_COLOR = {
    "conv":   "#2a78d6",   # slot 1 blue    — learned convolutions
    "pool":   "#eb6834",   # slot 2 orange  — pooling (no parameters)
    "dense":  "#1baf7a",   # slot 3 aqua    — fully connected / head
    "shape":  "#8a8a85",   # neutral        — reshape/pad/stack/concat (no params)
    "input":  "#52514e",   # neutral        — the data itself
}

TRACES = {
    #                scope  modalities   layout        arch
    "sd_dna":            ("sd", ["dna"], "modality", "late_fusion"),
    "sd_multimodal":     ("sd", "all",   "modality", "late_fusion"),
    "sd_multimodal_loci": ("sd", "all",  "locus",    "late_fusion"),
    "sd_dna_mdcnn":      ("sd", ["dna"], "locus",    "mdcnn"),
    "md_dna":            ("md", ["dna"], "modality", "late_fusion"),
    "md_multimodal":     ("md", "all",   "modality", "late_fusion"),
    "md_dna_mdcnn":      ("md", ["dna"], "locus",    "mdcnn"),
    # locus-keyed set fusion: shared per-modality encoders, so the interesting
    # case is the per-locus multi-modal layout (that is the block set whose
    # count/order it exists to stop caring about).
    "sd_setfusion":      ("sd", "all",   "locus",    "setfusion"),
    "md_setfusion":      ("md", "all",   "locus",    "setfusion"),
    # cis fusion: same per-locus input, but the promoter is spliced onto its own
    # CDS before encoding — so the block list and the BRANCH list differ, and the
    # diagram's first column is a cis-unit rather than a block.
    "sd_cisfusion":      ("sd", "all",   "locus",    "cisfusion"),
    "md_cisfusion":      ("md", "all",   "locus",    "cisfusion"),
}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def _views(blocks):
    """(per-locus blocks, per-modality blocks) from one per-locus load. The
    per-modality view is what the loaders build with per_modality_branch=True —
    each modality's blocks concatenated along length — so both layouts come
    from a single (expensive) load."""
    by_mod = {}
    for b in blocks:
        by_mod.setdefault(b.modality, []).append(b)
    merged = [merge_modality_blocks(v, m) if len(v) > 1 else v[0]
              for m, v in by_mod.items()]
    return list(blocks), merged


def _pick(blocks, modalities):
    return [b for b in blocks if b.modality in modalities]


def _branch_inputs(model, blocks):
    """What feeds branch i, as ``(label, shape_without_batch, subtitle, kind)``.

    For every positional model — and for SetFusionNet, whose grouper already
    resolves a shared encoder call back to the block in flight — branch i IS
    block i. CisFusionNet is the exception: it regroups the blocks into
    cis-units first, so branch i is a UNIT that may own two blocks, and the
    diagram has to name both halves and report the FUSED width."""
    if not isinstance(model, CisFusionNet):
        return [(b.name, tuple(b.array.shape[1:]), f"{b.modality}  {b.array.dtype}",
                 b.modality) for b in blocks]
    out = []
    for name, unit, spec, kind in zip(model.unit_names, model.units,
                                      model.unit_specs, model.unit_kinds):
        reg_i, dna_i, pass_i = unit
        if pass_i is not None:
            b = blocks[pass_i]
            out.append((b.name, spec, f"{b.array.dtype}  (not a nucleotide block)", kind))
        else:
            parts = [blocks[j].name for j in (reg_i, dna_i) if j is not None]
            sub = (" ⊕ ".join(parts) + "  +segment ch") if len(parts) > 1 else \
                  (parts[0] + "  +segment ch (unpaired)")
            out.append((name, spec, sub, kind))
    return out


def _sample_index(labels):
    """Index of the isolate with the most non-missing phenotypes — a sample
    whose label(s) actually mean something."""
    lab = labels.reshape(len(labels), -1)
    valid = (lab != -1).sum(axis=1)
    return int(np.argmax(valid))


# ---------------------------------------------------------------------------
# tracing
# ---------------------------------------------------------------------------

def _signature(mod):
    """One-line description of what a layer does, with its hyperparameters."""
    t = type(mod).__name__
    if t == "Conv1d":
        return (f"Conv1d  {mod.in_channels}->{mod.out_channels}",
                f"k={mod.kernel_size[0]} s={mod.stride[0]} p={mod.padding[0]}", "conv")
    if t == "Conv2d":
        k = mod.kernel_size
        return (f"Conv2d  {mod.in_channels}->{mod.out_channels}",
                f"k={k[0]}x{k[1]} s={mod.stride[0]} p={mod.padding[0]}", "conv")
    if t == "MaxPool1d":
        return ("MaxPool1d", f"k={mod.kernel_size} s={mod.stride}", "pool")
    if t == "Linear":
        return (f"Linear  {mod.in_features}->{mod.out_features}", "bias=True", "dense")
    if t == "LayerNorm":
        return ("LayerNorm", f"{tuple(mod.normalized_shape)}", "shape")
    if t in ("AdaptiveAvgPool1d", "AdaptiveMaxPool1d"):
        return (t, f"-> {mod.output_size} bins", "pool")
    if t == "Embedding":
        return (f"Embedding {mod.num_embeddings}x{mod.embedding_dim}",
                "identity lookup", "dense")
    if t == "TransformerEncoder":
        d = mod.layers[0].linear1.in_features
        return ("TransformerEncoder", f"{len(mod.layers)} layers, d={d}", "dense")
    if t == "MultiheadAttention":
        return ("MultiheadAttention", f"{mod.num_heads} heads, d={mod.embed_dim}", "dense")
    return (t, "", "conv" if any(p.requires_grad for p in mod.parameters()) else "shape")


# Containers traced as ONE step: their internals (per-layer attention projections,
# feed-forwards, norms) would bury the dataflow in dozens of rows that say nothing
# about how a sample moves through the model.
OPAQUE = (torch.nn.TransformerEncoder, torch.nn.TransformerEncoderLayer,
          torch.nn.MultiheadAttention)


def _stats(t):
    a = t.detach().float()
    return {"mean": float(a.mean()), "std": float(a.std(unbiased=False)),
            "min": float(a.min()), "max": float(a.max()),
            "nonzero": float((a != 0).float().mean())}


def _group_of(path):
    """Which column of the diagram a module belongs to: one group per branch /
    trunk, plus the shared head."""
    parts = path.split(".")
    if parts[0] in ("encoders", "trunks"):
        return f"{parts[0][:-1]} {parts[1]}"
    return "head"


def _grouper(model, blocks):
    """Path -> diagram group, as a stateful callable.

    Positional models map a path to a branch directly (``encoders.3.conv1`` is
    branch 3). SetFusionNet cannot: its encoder is SHARED per modality, so
    ``encoders.dna.conv1`` fires once per DNA locus and the path alone says
    nothing about which block is in flight. We count invocations instead — the
    k-th call of a given leaf under modality m is that modality's k-th block —
    which puts each BLOCK on its own row, the thing worth seeing."""
    if type(model).__name__ != "SetFusionNet":
        return _group_of

    order = {}
    for i, b in enumerate(blocks):
        order.setdefault(str(b.name).partition(":")[0], []).append(i)
    seen = {}

    def fn(path):
        if not path.startswith("encoders."):
            return "head"
        m = path.split(".")[1]
        k = seen.get((m, path), 0)
        seen[(m, path)] = k + 1
        idx = order.get(m, [])
        return f"encoder {idx[k]}" if k < len(idx) else f"encoder-{m} call {k}"
    return fn


def _hook_targets(model):
    """Modules to hook: every leaf, except that an OPAQUE container is hooked
    whole and its subtree skipped."""
    targets, skip = [], []
    for path, mod in model.named_modules():
        if not path or any(path.startswith(p + ".") for p in skip):
            continue
        if isinstance(mod, OPAQUE):
            targets.append((path, mod))
            skip.append(path)
        elif not list(mod.children()):
            targets.append((path, mod))
    return targets


def trace(model, xs, blocks):
    """Run `xs` through `model` with hooks on every traced module. Returns
    (records, output). Each record is one step: what it is, what went in, what
    came out, how many parameters it holds, and the output's statistics.
    Pseudo-records (no parameters) are spliced in for the steps the forward does
    functionally — zero-padding, locus stacking, pooling concat, fusion
    concat — so the diagram shows those too."""
    records = []
    handles = []
    group_of = _grouper(model, blocks)

    def hook(path, mod):
        def fn(m, inp, out):
            if isinstance(out, (tuple, list)):        # e.g. MultiheadAttention
                out = next((o for o in out if isinstance(o, torch.Tensor)), None)
            if not isinstance(out, torch.Tensor):
                return
            name, cfg, role = _signature(m)
            records.append({
                "group": group_of(path), "path": path, "op": name, "cfg": cfg,
                "role": role,
                "in": tuple(inp[0].shape) if isinstance(inp[0], torch.Tensor) else None,
                "out": tuple(out.shape),
                "params": sum(p.numel() for p in m.parameters()),
                "stats": _stats(out),
            })
        return mod.register_forward_hook(fn)

    for path, mod in _hook_targets(model):
        handles.append(hook(path, mod))
    try:
        model.eval()
        with torch.no_grad():
            out = model(xs)
    finally:
        for h in handles:
            h.remove()

    records = _splice_structure(model, records, xs, blocks)
    return records, out


def _rec(group, op, cfg, in_shape, out_shape, note=""):
    return {"group": group, "path": "(functional)", "op": op, "cfg": cfg,
            "role": "shape", "in": in_shape, "out": out_shape, "params": 0,
            "stats": None, "note": note}


def _splice_structure(model, records, xs, blocks):
    """Insert the parameter-free steps the forward performs inline."""
    out = []
    if type(model).__name__ == "SetFusionNet":
        return _splice_setfusion(model, records, xs)
    if isinstance(model, MDCNNNet):
        # per group: pad every locus to the group's longest, stack as channels
        for gi, (idxs, length) in enumerate(zip(model.group_idx, model.group_len)):
            grp = f"trunk {gi}"
            first = next(r for r in records if r["group"] == grp)
            lens = [tuple(xs[i].shape) for i in idxs]
            out.append(_rec(grp, "zero-pad", f"each locus -> L={length}",
                            f"{len(idxs)} x {lens[0]}..{lens[-1]}",
                            f"{len(idxs)} x (1, {xs[idxs[0]].shape[1]}, {length})",
                            note="right-pad with zeros (distinct from the gap channel)"))
            out.append(_rec(grp, "stack loci", f"-> channel axis ({len(idxs)} loci)",
                            f"{len(idxs)} x (1, {xs[idxs[0]].shape[1]}, {length})",
                            first["in"], note="this is what makes layer 1 see every locus"))
            out += [r for r in records if r["group"] == grp]
        head = [r for r in records if r["group"] == "head"]
        if head:
            out.append(_rec("head", "concat trunks", f"{len(model.trunks)} trunk(s)",
                            f"{len(model.trunks)} x (1, F_t)", head[0]["in"]))
            out += head
        return out

    pre = _cis_pre_records(model, xs, blocks) if isinstance(model, CisFusionNet) else {}
    groups = [g for g in dict.fromkeys(r["group"] for r in records) if g != "head"]
    for g in groups:
        out += pre.get(g, [])
        out += [r for r in records if r["group"] == g]
    head = [r for r in records if r["group"] == "head"]
    if head:
        out.append(_rec("head", "concat branches", f"{len(groups)} branch(es)",
                        f"{len(groups)} x (1, F_b)", head[0]["in"],
                        note="the ONLY place branches meet in late fusion"))
        out += head
    return out


def _cis_pre_records(model, xs, blocks):
    """The steps CisFusionNet performs in ``cis_inputs`` before any encoder sees
    a tensor — appending the promoter/CDS marker channel and splicing the two
    segments into one axis. They hold no parameters, so no hook fires on them,
    but they are the entire point of the model and belong in the trace."""
    pre = {}
    for i, (unit, spec, kind) in enumerate(zip(model.units, model.unit_specs,
                                               model.unit_kinds)):
        reg_i, dna_i, pass_i = unit
        if pass_i is not None:
            continue                                  # protein/biophysical: untouched
        g, recs = f"encoder {i}", []
        for j, seg, flag in ((reg_i, "promoter", 0), (dna_i, "CDS", 1)):
            if j is None:
                continue
            recs.append(_rec(
                g, "+ segment ch", f"{seg} -> flag {flag}", tuple(xs[j].shape),
                (1, spec[0], xs[j].shape[-1]),
                note=f"6th channel marks {seg} columns; {blocks[j].name}"))
        if kind == "cis":
            widths = [xs[j].shape[-1] for j in (reg_i, dna_i)]
            if model.spacer:
                recs.append(_rec(g, "spacer", f"{model.spacer} zero columns",
                                 f"(1, {spec[0]}, {widths[0]:,})",
                                 f"(1, {spec[0]}, {model.spacer})",
                                 note="all-zero incl. the flag: 'gap here', not a base"))
            recs.append(_rec(
                g, "concat promoter⊕CDS", "transcription order",
                f"(1, {spec[0]}, {widths[0]:,}) + (1, {spec[0]}, {widths[1]:,})",
                (1, spec[0], spec[1]),
                note="one kernel at the junction now spans BOTH"))
        pre[g] = recs
    return pre


def _splice_setfusion(model, records, xs):
    """SetFusionNet's shape story lives in the functional glue: the per-block
    pooled concat, the token stack, the identity embeddings added onto every
    token, and the drug-query attention. Spell each of them out."""
    out = []
    groups = [g for g in dict.fromkeys(r["group"] for r in records) if g != "head"]
    for g in groups:
        recs = [r for r in records if r["group"] == g]
        pooled = [r for r in recs if "Pool1d" in r["op"] and "Adaptive" in r["op"]]
        for r in recs:
            if r["path"].endswith(".proj") and pooled:
                out.append(_rec(g, "concat avg|max", f"{len(pooled)} pooled maps",
                                f"{len(pooled)} x {_shape_str(pooled[-1]['out'])}",
                                r["in"],
                                note="mean AND max per bin: 'how much' + 'did it occur'"))
            out.append(r)

    head = [r for r in records if r["group"] == "head"]
    if not head:
        return out
    fusion = next((r for r in head if r["op"] == "TransformerEncoder"), None)
    n_tok = fusion["in"][1] if fusion else len(xs)
    d = fusion["in"][2] if fusion else 0
    spliced = [_rec("head", "stack tokens", f"{n_tok} blocks -> a SET",
                    f"{n_tok} x (1, {d})", fusion["in"] if fusion else None,
                    note="one token per block; order stops mattering from here")]
    for r in head:
        if fusion is not None and r is fusion:
            spliced.append(_rec("head", "+ identity emb", "modality + locus vector",
                                fusion["in"], fusion["in"],
                                note="this is what pairs dna:katG with regulatory:katG"))
        spliced.append(r)
    return out + spliced


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _shape_str(s):
    if s is None:
        return "-"
    if isinstance(s, str):
        return s
    return "(" + ", ".join(f"{d:,}" for d in s) + ")"


def _box(ax, x, y, w, h, lines, role, bold_first=True):
    c = ROLE_COLOR[role]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
                                linewidth=1.6, edgecolor=c, facecolor=c + "1f", zorder=2))
    n = len(lines)
    for i, (txt, size, col) in enumerate(lines):
        ax.text(x + w / 2, y + h - h * (i + 0.75) / n, txt, ha="center", va="center",
                fontsize=size, color=col, zorder=3,
                fontweight="bold" if (i == 0 and bold_first) else "normal",
                family="monospace" if i else None)


def _arrow(ax, x0, y0, x1, y1, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9,
                                 linewidth=0.9, color="#9a9a94", zorder=1,
                                 connectionstyle="arc3,rad=0.0"))


def _representatives(groups, max_show, inputs):
    """Which branch rows to draw. Every MODALITY (or cis-unit kind) gets a row
    first — its channel count and length differ, so it is genuinely a different
    stack; leftover slots spread evenly over the rest, which are identical apart
    from length. Returns (rows, n_elided)."""
    if len(groups) <= max_show:
        return groups, 0

    def modality(g):
        i = int(g.split()[-1])
        return inputs[i][3] if g.startswith("encoder") and i < len(inputs) else g

    keep, seen = [], set()
    for g in groups:                      # one per modality, in input order
        m = modality(g)
        if m not in seen:
            seen.add(m)
            keep.append(g)
    rest = [g for g in groups if g not in keep]
    slots = max_show - len(keep)
    if slots > 0 and rest:
        step = max(1, len(rest) // slots)
        keep += rest[::step][:slots]
    keep = sorted(dict.fromkeys(keep[:max_show]), key=lambda g: int(g.split()[-1]))
    return keep, len(groups) - len(keep)


def render(name, cfg_line, sample_line, blocks, records, out_tensor, out_labels,
           path, max_branches=6, table_rows=46, inputs=None):
    inputs = inputs if inputs is not None else _branch_inputs(None, blocks)
    branch_groups = [g for g in dict.fromkeys(r["group"] for r in records) if g != "head"]
    shown, elided = _representatives(branch_groups, max_branches, inputs)
    head = [r for r in records if r["group"] == "head"]
    per_branch = {g: [r for r in records if r["group"] == g] for g in shown}
    n_stage = max((len(v) for v in per_branch.values()), default=0)

    n_rows = len(shown) + (1 if elided else 0)
    n_cols = 1 + n_stage + len(head)
    col_w, row_h = 2.35, 1.28
    flow_h = max(2.0, n_rows * row_h + 0.5)

    table_lines = _table_lines(records, blocks, limit=table_rows)
    table_h = 0.19 * (len(table_lines) + 2)
    gap = 0.55                                  # legend strip between flow and table
    fig_w = max(13.0, 1.2 + n_cols * col_w)
    fig_h = 1.55 + flow_h + gap + table_h

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=SURFACE)
    fig.text(0.008, 1 - 0.28 / fig_h, name, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.008, 1 - 0.62 / fig_h, cfg_line, fontsize=9, color=INK_DIM, va="top")
    fig.text(0.008, 1 - 0.90 / fig_h, sample_line, fontsize=9, color=INK_DIM, va="top",
             family="monospace")

    ax = fig.add_axes([0.006, (table_h + gap) / fig_h, 0.988, flow_h / fig_h])
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.axis("off")

    bw, bh = 0.90, 0.82
    row_y = {g: n_rows - 1 - i for i, g in enumerate(shown)}

    for g in shown:
        y = row_y[g] + (1 - bh) / 2
        recs = per_branch[g]
        bi = int(g.split()[-1])
        src = inputs[bi] if g.startswith("encoder") and bi < len(inputs) else None
        if src is not None:
            label, shape, subtitle, _kind = src
            lines = [(label, 8, INK), (_shape_str((1,) + tuple(shape)), 7.5, INK),
                     (subtitle, 7, INK_DIM)]
        else:
            lines = [(f"{g}: loci group", 8, INK),
                     (_shape_str(recs[0]["in"]), 7.5, INK), ("input blocks", 7, INK_DIM)]
        _box(ax, 0.05, y, bw, bh, lines, "input")
        prev_x = 0.05 + bw
        for si, r in enumerate(recs):
            x = 1 + si + 0.05
            _box(ax, x, y, bw, bh, _box_lines(r), r["role"])
            _arrow(ax, prev_x, y + bh / 2, x, y + bh / 2)
            prev_x = x + bw
        for _ in range(n_stage - len(recs)):
            pass

    if elided:
        y = 0 + (1 - bh) / 2
        lens = [i[1][-1] for i in inputs]
        for cx in range(n_cols):
            ax.add_patch(FancyBboxPatch((cx + 0.05, y), bw, bh,
                                        boxstyle="round,pad=0.012,rounding_size=0.03",
                                        linewidth=1.0, linestyle=(0, (4, 3)),
                                        edgecolor="#b9b9b3", facecolor="none", zorder=2))
        ax.text(n_cols / 2, y + bh / 2,
                f"+ {elided} more branches, identical stack shapes  "
                f"(branch lengths L = {min(lens):,} .. {max(lens):,}; "
                f"every one is listed in the .txt trace)",
                ha="center", va="center", fontsize=9, color=INK_DIM, style="italic")

    # head chain, centred vertically
    hy = (n_rows - 1) / 2 + (1 - bh) / 2 if n_rows > 1 else (1 - bh) / 2
    hx0 = 1 + n_stage
    for hi, r in enumerate(head):
        x = hx0 + hi + 0.05
        _box(ax, x, hy, bw, bh, _box_lines(r), r["role"])
        if hi == 0:
            for g in shown:
                _arrow(ax, 1 + len(per_branch[g]) - 1 + 0.05 + bw, row_y[g] + 0.5,
                       x, hy + bh / 2)
        else:
            _arrow(ax, x - 0.05, hy + bh / 2, x, hy + bh / 2)

    if head:
        logits = out_tensor.reshape(-1)
        probs = torch.sigmoid(logits)
        txt = "\n".join(f"{lbl}: logit {float(l): .3f} -> p(S) {float(p):.3f}"
                        for lbl, l, p in list(zip(out_labels, logits, probs))[:12])
        ax.text(n_cols - 0.05, hy - 0.10, txt, ha="right", va="top", fontsize=7.5,
                family="monospace", color=INK_DIM)

    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=8,
                          markerfacecolor=ROLE_COLOR[k] + "1f", markeredgecolor=ROLE_COLOR[k],
                          label=v)
               for k, v in [("input", "input block"), ("conv", "convolution (learned)"),
                            ("pool", "pooling"), ("shape", "reshape / pad / concat"),
                            ("dense", "fully connected / attention")]]
    fig.legend(handles=handles, loc="lower left", ncol=5, frameon=False, fontsize=8.5,
               handletextpad=0.4, columnspacing=1.8,
               bbox_to_anchor=(0.008, (table_h + 0.10) / fig_h))

    fig.text(0.006, table_h / fig_h, "\n".join(table_lines), fontsize=7.2,
             family="monospace", color=INK, va="top", ha="left", linespacing=1.45)

    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _box_lines(r):
    s = r["stats"]
    third = (f"mu{s['mean']: .2f} sd{s['std']:.2f} nz{s['nonzero']*100:.0f}%"
             if s else (r.get("note", "")[:38] or "no parameters"))
    return [(r["op"], 8, INK),
            (_shape_str(r["out"]), 7.5, INK),
            (f"{r['cfg']}  {r['params']:,}p" if r["params"] else third, 6.8, INK_DIM)]


def _table_lines(records, blocks, limit):
    w = (6, 10, 38, 31, 29, 11)
    head = ("#", "group", "operation", "input", "output", "params")
    lines = ["".join(f"{h:<{n}}" for h, n in zip(head, w))
             + "output stats  (mean / std / %nonzero)  or  note",
             "-" * (sum(w) + 44)]
    for i, r in enumerate(records[:limit], 1):
        s = r["stats"]
        stat = (f"{s['mean']: .3f} / {s['std']:.3f} / {s['nonzero']*100:5.1f}%" if s
                else r.get("note", ""))
        cells = [str(i), r["group"], f"{r['op']} {r['cfg']}".strip(),
                 _shape_str(r["in"]), _shape_str(r["out"]),
                 f"{r['params']:,}" if r["params"] else "-"]
        lines.append("".join(f"{c[:n-1]:<{n}}" for c, n in zip(cells, w)) + stat)
    if len(records) > limit:
        lines.append(f"... {len(records) - limit} more steps — see the .txt trace")
    lines.append("")
    lines.append(f"input blocks ({len(blocks)}): " + ", ".join(
        f"{b.name} {tuple(b.array.shape[1:])}" for b in blocks[:8])
        + (f", ... +{len(blocks) - 8} more" if len(blocks) > 8 else ""))
    lines.append("ReLU is applied (functionally) after every convolution and after "
                 "fc1/fc2 — not shown as its own step. Weights are randomly initialised.")
    return lines


def write_text(path, name, cfg_line, sample_line, blocks, records, out_tensor, out_labels,
               inputs=None, n_params=None):
    L = [f"{name}", "=" * len(name), cfg_line, sample_line, "",
         f"input blocks ({len(blocks)}):"]
    for b in blocks:
        L.append(f"  {b.name:<28} {str(tuple(b.array.shape[1:])):<16} "
                 f"{b.modality:<12} {b.array.dtype}  {b.note}")
    if inputs is not None and len(inputs) != len(blocks):
        # the model regrouped the blocks (CisFusionNet): show what each branch
        # actually receives, since it no longer matches the block list above
        L += ["", f"model branches ({len(inputs)}) — blocks regrouped before encoding:"]
        for label, shape, subtitle, kind in inputs:
            L.append(f"  {label:<28} {str(tuple(shape)):<16} {kind:<12} {subtitle}")
    L += ["", "forward trace (one isolate, batch of 1):",
          f"{'#':<6}{'group':<11}{'operation':<34}{'input':<30}{'output':<26}"
          f"{'params':<12}output stats / note",
          "-" * 150]
    for i, r in enumerate(records, 1):
        s = r["stats"]
        stat = (f"mean {s['mean']: .4f}  std {s['std']:.4f}  min {s['min']: .3f}  "
                f"max {s['max']: .3f}  nonzero {s['nonzero']*100:.1f}%" if s
                else r.get("note", ""))
        cells = [str(i), r["group"], (r["op"] + " " + r["cfg"]).strip(),
                 _shape_str(r["in"]), _shape_str(r["out"]),
                 f"{r['params']:,}" if r["params"] else "-"]
        widths = (6, 11, 34, 30, 26, 12)
        L.append("".join(f"{c[:n - 1]:<{n}}" for c, n in zip(cells, widths)) + stat)
    # Sum over UNIQUE modules: a weight-shared encoder (SetFusionNet runs one
    # per modality over every locus) fires once per block, so summing per step
    # counts it once per invocation — that reported 9.7M for a 741k model.
    unique = {}
    for r in records:
        if r["params"] and r["path"] != "(functional)":
            unique[r["path"]] = r["params"]
    traced = sum(unique.values())
    line = (f"parameters: {traced:,} across {len(unique)} distinct modules "
            f"({len(records)} steps traced; a weight-shared module is counted once "
            "here and once per invocation above)")
    if n_params is not None and n_params != traced:
        # bare nn.Parameters (SetFusionNet.drug_queries, positional embeddings)
        # never fire a forward hook, so they cannot appear as a traced step.
        line += (f"\n            + {n_params - traced:,} in bare nn.Parameters with no "
                 f"forward hook  =  {n_params:,} total in the model")
    L += ["", line, "", "outputs (untrained weights):"]
    for lbl, lo in zip(out_labels, out_tensor.reshape(-1)):
        L.append(f"  {lbl:<16} logit {float(lo): .4f}   p(susceptible) "
                 f"{float(torch.sigmoid(lo)):.4f}")
    Path(path).write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", nargs="+", default=list(TRACES), metavar="NAME",
                    help=f"which traces to run (default: all). Options: {list(TRACES)}")
    ap.add_argument("--drug", default="ISONIAZID", help="drug for the single-drug traces")
    ap.add_argument("--md-drugs", nargs="+", default=None,
                    help="drugs for the multi-drug traces (default: all 11)")
    real = ap.add_mutually_exclusive_group()
    real.add_argument("--real", dest="real", action="store_true", default=True)
    real.add_argument("--synthetic", dest="real", action="store_false",
                      help="tiny fixtures instead of the real load (wiring check)")
    ap.add_argument("--isolate", type=int, default=None,
                    help="isolate row to trace (default: the one with the most "
                         "non-missing phenotypes)")
    ap.add_argument("--max-branches", type=int, default=6,
                    help="branch rows drawn per figure (default 3; the .txt has all)")
    ap.add_argument("--outdir", default=str(PROJECT_DIR / "diagrams" / "model_traces"))
    args = ap.parse_args()

    unknown = [t for t in args.traces if t not in TRACES]
    if unknown:
        ap.error(f"unknown trace(s) {unknown}; choose from {list(TRACES)}")
    drug = args.drug.upper()
    md_drugs = [d.upper() for d in (args.md_drugs or ALL_DRUGS)]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    need_sd = any(TRACES[t][0] == "sd" for t in args.traces)
    need_md = any(TRACES[t][0] == "md" for t in args.traces)

    with contextlib.ExitStack() as stack:
        if args.real:
            geno, pheno, reg = REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR
        else:
            tmp = stack.enter_context(tempfile.TemporaryDirectory())
            genes = sorted(union_loci(md_drugs if need_md else [drug]))
            regions = [r for r in union_regulatory(md_drugs if need_md else [drug])
                       if r not in set(genes)]
            geno, pheno = build_fixture_dataset(tmp, genes=genes, drugs=md_drugs + [drug],
                                                n_isolates=60, n_codons=200, seed=0,
                                                regulatory_regions=regions)
            reg = geno

        data = {}
        if need_sd:
            print(f"[trace] loading single-drug {drug} (all modalities, per-locus) ...",
                  flush=True)
            d = load_dataset(drug, list(MODALITIES), geno, pheno, regulatory_dir=reg,
                             per_modality_branch=False)
            data["sd"] = (d, *_views(d.blocks), d.y, [drug], d.isolate_ids)
        if need_md:
            print(f"[trace] loading multi-drug ({len(md_drugs)} drugs, all modalities, "
                  "per-locus) ...", flush=True)
            d = load_multidrug_dataset(md_drugs, list(MODALITIES), geno, pheno,
                                       regulatory_dir=reg, per_modality_branch=False)
            data["md"] = (d, *_views(d.blocks), d.Y, d.drugs, d.isolate_ids)

    for name in args.traces:
        scope, mods, layout, arch = TRACES[name]
        bundle, per_locus, per_mod, labels, drug_names, ids = data[scope]
        mods = list(MODALITIES) if mods == "all" else mods
        blocks = _pick(per_locus if layout == "locus" else per_mod, mods)
        if not blocks:
            print(f"[trace] {name}: no blocks for modalities {mods} — skipped", flush=True)
            continue

        i = args.isolate if args.isolate is not None else _sample_index(labels)
        xs = [torch.from_numpy(b.array[i:i + 1]).float() for b in blocks]
        specs = [b.spec() for b in blocks]
        n_drugs = len(drug_names)
        try:
            if arch == "mdcnn":
                model = MDCNNNet(specs, n_drugs=n_drugs)
            elif arch == "setfusion":
                model = SetFusionNet.from_blocks(blocks, drug_names=drug_names)
            elif arch == "cisfusion":
                model = CisFusionNet.from_blocks(blocks, drug_names=drug_names)
            elif scope == "md":
                model = MultiDrugNet(specs, drug_names)
            else:
                model = MultiModalNet(specs, n_drugs=1)
            records, out = trace(model, xs, blocks)
        except (RuntimeError, ValueError) as e:
            # a branch too short for the conv/pool stack — raised either by
            # MDCNNTrunk up front or by CNNEncoder's probe forward. Real loci are
            # hundreds of bp, so this is a synthetic-fixture artifact (random
            # sequence hits a stop codon early -> a 1-residue protein block);
            # report which block and carry on rather than kill the whole run.
            short = min((b.spec()[1], b.name) for b in blocks)
            print(f"[trace] {name}: skipped — {type(e).__name__}: {e} "
                  f"Shortest block is {short[1]} (L={short[0]}); the conv stack "
                  f"needs a longer position axis (real data is fine — this bites "
                  f"on --synthetic).", flush=True)
            continue
        n_params = sum(p.numel() for p in model.parameters())
        inputs = _branch_inputs(model, blocks)

        lab = labels.reshape(len(labels), -1)[i]
        code = {0: "RESISTANT", 1: "susceptible", -1: "missing"}
        shown_lab = ", ".join(f"{d}={code[int(v)]}" for d, v in zip(drug_names, lab))[:150]
        # a regrouping model turns N blocks into M branches — say both, and how
        # many loci actually got their promoter back
        branch_line = f"{len(blocks)} input block(s)"
        if len(inputs) != len(blocks):
            paired = sum(1 for _l, _s, _t, k in inputs if k == "cis")
            branch_line += (f" -> {len(inputs)} branch(es), {paired} cis-paired "
                            f"(promoter⊕CDS)")
        cfg_line = (f"{type(model).__name__}  |  arch={arch}  |  "
                    f"{'one branch per ' + ('LOCUS' if layout == 'locus' else 'MODALITY')}  |  "
                    f"modalities={'+'.join(mods)}  |  {branch_line}  |  "
                    f"{n_params:,} parameters  |  outputs={n_drugs}")
        sample_line = (f"sample: {ids[i]} (row {i} of {len(ids):,}, "
                       f"{'REAL' if args.real else 'synthetic'} data)   {shown_lab}")

        png, txt = outdir / f"{name}.png", outdir / f"{name}.txt"
        render(name, cfg_line, sample_line, blocks, records, out, drug_names, png,
               max_branches=args.max_branches, inputs=inputs)
        write_text(txt, name, cfg_line, sample_line, blocks, records, out, drug_names,
                   inputs=inputs, n_params=n_params)
        print(f"[trace] {name}: {len(records)} steps, {n_params:,} params -> "
              f"{png.name} + {txt.name}", flush=True)

    print(f"\nWrote traces to {outdir}")


if __name__ == "__main__":
    main()
