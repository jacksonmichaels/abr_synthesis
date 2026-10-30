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
  sd_locusfusion      single-drug, all modalities      locusfusion (LocusFusionNet), per LOCUS
  md_locusfusion      multi-drug,  all modalities      locusfusion (LocusFusionNet), per LOCUS
  sd_catalogue        single-drug, all modalities      catalogue  (CatalogueNet), per LOCUS
  md_catalogue        multi-drug,  all modalities      catalogue  (CatalogueNet), per LOCUS
  sd_additive         single-drug, all modalities      additive   (AdditiveVariantNet)
  sd_noisyor          single-drug, all modalities      noisyor    (NoisyOrVariantNet)
  sd_gatedpool        single-drug, all modalities      gatedpool  (GatedPoolNet)
  sd_deepsets         single-drug, all modalities      deepsets   (DeepSetsVariantNet)
  sd_fm               single-drug, all modalities      fm         (FactorizedInteractionNet)

The last seven are the VARIANT-TOKEN family (``models.DELTA_ARCHS``): they read
reference-difference input, so they get their own ``--delta`` load, and their
first columns are the tokenizer rather than a conv stack — one row per block,
collapsing a 2.5 kb one-hot to the handful of columns where this isolate differs
from H37Rv. The five aggregators that only differ in how the token set is
combined are traced single-drug only; multi-drug changes nothing but the output
width (``catalogue`` and ``locusfusion`` are traced both ways to show that).

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

Loading real data costs minutes and GBs: one load per (scope, encoding) —
single-drug/multi-drug x plain/delta — is shared across every trace that wants
it, and --synthetic swaps in fixtures for a fast wiring check.

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
from models import (DELTA_ARCHS, EXPERIMENTAL_DEFAULTS,  # noqa: E402
                    EXPERIMENTAL_MODELS, LOCUSFUSION_DEFAULTS, CisFusionNet,
                    LocusFusionNet, MDCNNNet, MultiDrugNet, MultiModalNet,
                    SetFusionNet, make_experimental, parse_block_key)
from models.locusfusion import C_TOK  # noqa: E402

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
    # the variant-token family. Every one of these is in models.DELTA_ARCHS, so
    # the loader hands it reference-difference blocks and the first columns of
    # the diagram are the tokenizer: occupancy -> cap -> feature slots.
    "sd_locusfusion":    ("sd", "all",   "locus",    "locusfusion"),
    "md_locusfusion":    ("md", "all",   "locus",    "locusfusion"),
    # `catalogue` is the family's control -- one learned scalar per exact
    # variant id, i.e. logistic regression on the variant matrix -- so it is the
    # baseline the other five aggregators are read against, and it gets both
    # scopes. They differ from it ONLY in the aggregate step, which is why one
    # scope is enough for them.
    "sd_catalogue":      ("sd", "all",   "locus",    "catalogue"),
    "md_catalogue":      ("md", "all",   "locus",    "catalogue"),
    "sd_additive":       ("sd", "all",   "locus",    "additive"),
    "sd_noisyor":        ("sd", "all",   "locus",    "noisyor"),
    "sd_gatedpool":      ("sd", "all",   "locus",    "gatedpool"),
    "sd_deepsets":       ("sd", "all",   "locus",    "deepsets"),
    "sd_fm":             ("sd", "all",   "locus",    "fm"),
}

# One line per aggregator, spliced in as the step where the token axis
# disappears -- the only thing that differs across the six.
AGGREGATORS = {
    "CatalogueNet":  ("sum w[variant id]",
                      "a learned scalar per EXACT variant id"),
    "AdditiveVariantNet": ("sum w(features)",
                           "generalises to substitutions never seen"),
    "NoisyOrVariantNet": ("1 - prod(1 - p_v)",
                          "susceptible unless something confers R"),
    "GatedPoolNet":  ("sum sigmoid(gate) * value",
                      "absolute gate: no dilution by neutrals"),
    "DeepSetsVariantNet": ("sum + max + count",
                           "plain additivity, no attention at all"),
    "FactorizedInteractionNet": ("first order + rank-k pairs",
                                 "all pairs, priced O(T*k) not O(T^2)"),
}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def _views(blocks, merge=True):
    """(per-locus blocks, per-modality blocks) from one per-locus load. The
    per-modality view is what the loaders build with per_modality_branch=True —
    each modality's blocks concatenated along length — so both layouts come
    from a single (expensive) load. ``merge=False`` skips building it (it
    copies every array a second time): the delta load feeds only per-locus
    architectures, which is what DELTA_ARCHS means."""
    if not merge:
        return list(blocks), []
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
    if t == "KeyedTokenNorm":
        return ("KeyedTokenNorm", f"{mod.running_mean.shape[0]} keys", "shape")
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


def _is_variant_model(model):
    """Is this one of the variant-token architectures (locusfusion + the six
    aggregators)? They share a tokenizer and a diagram shape, and nothing else
    in the project looks like them."""
    return isinstance(model, LocusFusionNet) or \
        type(model) in set(EXPERIMENTAL_MODELS.values())


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
    if _is_variant_model(model):
        # the variant-token family has no per-block branch at all: one shared
        # tokenizer, then one shared pipeline. Every hooked module belongs to
        # the trunk, and the per-block rows are spliced in functionally below.
        return lambda path: "head"
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
        elif isinstance(mod, torch.nn.Identity):
            continue          # a disabled dropout: a row that says nothing
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
    if _is_variant_model(model):
        rows = _variant_token_rows(model, xs, blocks)
        return rows + (_splice_locusfusion(model, records)
                       if isinstance(model, LocusFusionNet)
                       else _splice_variant_head(model, records))
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


def _token_streams(model, blocks):
    """``[(block indices sharing one token set, length, label)]``.

    LocusFusionNet unions the occupancy over the blocks of one (locus,
    coordinate stream): protein and biophysical are co-indexed by construction,
    so a changed residue is ONE token carrying both views of it. The flat
    family tokenizes each block on its own."""
    plan = getattr(model, "_plan", None)
    if plan is None:
        return [([i], b.spec()[1], b.name) for i, b in enumerate(blocks)]
    return [([i for i, _m in members], length, f"{model.loci[li]} · {stream}")
            for li, streams in enumerate(plan)
            for stream, members, length in streams]


def _variant_token_rows(model, xs, blocks):
    """One diagram row per input block: the tokenizer, which is where these
    architectures do their real work.

    No parameters live here, so no hook fires — but this is the step the whole
    family exists for, and the counts are REAL for the traced isolate: a delta
    block is non-zero only at the columns where it differs from H37Rv, so
    ``occupancy`` is literally this isolate's variant count for that block."""
    cap = int(getattr(model, "max_variants", 0) or model.tok.max_variants)
    rows = {}
    for members, length, label in _token_streams(model, blocks):
        lead, k = members[0], max(1, min(cap, length))
        occ = torch.zeros(length, dtype=torch.bool)
        counts = []
        for j in members:
            o = xs[j][0].abs().sum(0) > 0
            counts.append(int(o.sum()))
            occ |= o
        n = int(occ.sum())
        for j, c in zip(members, counts):
            rows.setdefault(j, []).append(_rec(
                f"tokens {j}", "occupancy", "|x| > 0 per column", tuple(xs[j].shape),
                (1, length),
                note=f"{c} of {length:,} columns differ from H37Rv"))
        if len(members) > 1:
            rows[lead].append(_rec(
                f"tokens {lead}", "union occupancy", f"{len(members)} co-indexed blocks",
                f"{len(members)} x (1, {length:,})", (1, length),
                note=f"{label}: {n} changed column(s)"))
        rows[lead].append(_rec(
            f"tokens {lead}", "select variants", f"cap {cap}", (1, length), (1, k),
            note=f"{min(n, k)} real, {k - min(n, k)} padded and masked"))
        slots = "+".join(blocks[j].modality for j in members)
        rows[lead].append(_rec(
            f"tokens {lead}", "gather -> features", slots, tuple(xs[lead].shape),
            (1, k, C_TOK),
            note=f"{C_TOK}-dim layout; absent slots stay 0"))
        for j in members[1:]:
            rows[j].append(_rec(
                f"tokens {j}", "-> shared tokens", f"co-indexed with {blocks[lead].name}",
                (1, length), (1, k, C_TOK),
                note="same columns, its own feature slot"))
    return [r for j in sorted(rows) for r in rows[j]]


def _has_token_axis(shape, T):
    """Does this output still carry the token axis? That is what tells us where
    the aggregator sits: the step after which T is gone."""
    return shape is not None and len(shape) >= 3 and T in tuple(shape)[1:-1]


def _splice_variant_head(model, records):
    """The flat aggregator family: one set of tokens, one aggregate step.

    The six differ ONLY in that step, so it is named explicitly rather than
    left implicit in a shape change."""
    head = [r for r in records if r["group"] == "head"]
    T, n_blocks, k = model.tok.tokens, model.tok.n_blocks, model.tok.max_variants
    out = [_rec("head", "concat blocks", f"{n_blocks} block(s) x {k} slots",
                f"{n_blocks} x (1, {k}, {C_TOK})", (1, T, C_TOK),
                note="ONE flat set per isolate, no hierarchy")]
    last_tok = max((i for i, r in enumerate(head) if _has_token_axis(r["out"], T)),
                   default=-1)
    for i, r in enumerate(head):
        if r["path"].endswith("pos_proj"):
            out.append(_rec("head", "sinusoid(coord)", f"-> {model.emb.pos_dims} dims",
                            (1, T), r["in"],
                            note="continuous: the 1/3 fraction is the phase"))
        if r["path"].endswith("emb.norm"):
            out.append(_rec("head", "sum token parts", "features + position + locus + modality",
                            r["in"], r["in"],
                            note="exact position survives, nothing pooled"))
        out.append(r)
        if i == last_tok:
            op, why = AGGREGATORS.get(type(model).__name__,
                                      ("aggregate", "over the variant set"))
            nxt = head[i + 1]["in"] if i + 1 < len(head) else (1, model.n_drugs)
            out.append(_rec("head", op, f"over {T} slots", r["out"], nxt, note=why))
    out.append(_rec("head", "+ uncovered", f"{n_blocks} block flag(s)",
                    (1, n_blocks), (1, model.n_drugs),
                    note="a missing gene is not a wild-type gene"))
    return out


def _splice_locusfusion(model, records):
    """LocusFusionNet: the two-stage story lives between the hooks — the [WT]
    sentinel, the batched per-locus encoder call, and the fact that stage 2
    reads the sentinel's row and nothing else."""
    head = [r for r in records if r["group"] == "head"]
    n_loci, T, d = len(model.loci), model.tokens_per_locus, model.d_model
    out = [_rec("head", "stack loci", f"{n_loci} locus token set(s)",
                f"{n_loci} x (1, {T}, {C_TOK})", (1, n_loci, T, C_TOK),
                note="[WT] sentinel at slot 0 of each locus")]
    film = "" if model.film_scale is None else " + per-locus FiLM"
    # `locus_encoder="per_locus"` builds one encoder per locus instead of one
    # shared call, so the reshape belongs before the FIRST and the [WT] readout
    # after the LAST — with the default ("adapter") they are the same record.
    enc = [i for i, r in enumerate(head) if r["path"].startswith("encoders.")]
    shared = model.locus_encoder != "per_locus"
    for i, r in enumerate(head):
        path = r["path"]
        if path.endswith("pos_proj"):
            out.append(_rec("head", "sinusoid(coord)", f"-> {model.pos_dims} dims",
                            (1, n_loci, T), r["in"],
                            note="continuous: the 1/3 fraction is the phase"))
        if path.endswith("tok_norm"):
            out.append(_rec("head", "sum token parts",
                            f"features + position + locus{film}", r["in"], r["in"],
                            note="[WT] also gets its count / coverage"))
        if enc and i == enc[0]:
            out.append(_rec("head", "loci -> batch", f"{n_loci} loci x {T} tokens",
                            (1, n_loci, T, d), r["in"],
                            note="every locus in ONE encoder call" if shared
                            else f"{n_loci} encoders, one per locus"))
        if path.endswith("pool_attn"):
            out.append(_rec("head", "drug queries", f"{model.n_drugs} learned query(ies)",
                            (model.n_drugs, d), r["in"],
                            note="its attention map IS the attribution"))
        out.append(r)
        if enc and i == enc[-1]:
            out.append(_rec("head", "take [WT] row", "slot 0 of each locus",
                            r["out"], (1, n_loci, d),
                            note="the sentinel row IS the locus summary"))
    return out


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


def _arrow(ax, x0, y0, x1, y1, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9,
                                 linewidth=0.9, color="#9a9a94", zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def _input_index(g):
    """Which input block a branch row shows, or None. ``encoder i`` and
    ``tokens i`` are both keyed to block i; ``trunk i`` is NOT (an mdcnn trunk
    owns a whole group of loci) and neither is the head."""
    kind, _, num = g.rpartition(" ")
    return int(num) if kind in ("encoder", "tokens") and num.isdigit() else None


def _representatives(groups, max_show, inputs):
    """Which branch rows to draw. Every MODALITY (or cis-unit kind) gets a row
    first — its channel count and length differ, so it is genuinely a different
    stack; leftover slots spread evenly over the rest, which are identical apart
    from length. Returns (rows, n_elided)."""
    if len(groups) <= max_show:
        return groups, 0

    def modality(g):
        i = _input_index(g)
        return inputs[i][3] if i is not None and i < len(inputs) else g

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
    keep = sorted(dict.fromkeys(keep[:max_show]), key=lambda g: int(g.rpartition(" ")[2]))
    return keep, len(groups) - len(keep)


def render(name, cfg_line, sample_line, blocks, records, out_tensor, out_labels,
           path, max_branches=6, table_rows=46, inputs=None, max_head_cols=9):
    inputs = inputs if inputs is not None else _branch_inputs(None, blocks)
    branch_groups = [g for g in dict.fromkeys(r["group"] for r in records) if g != "head"]
    shown, elided = _representatives(branch_groups, max_branches, inputs)
    head = [r for r in records if r["group"] == "head"]
    per_branch = {g: [r for r in records if r["group"] == g] for g in shown}
    n_stage = max((len(v) for v in per_branch.values()), default=0)

    # A long trunk (the variant-token nets run 15-20 steps after the tokenizer)
    # would otherwise make the figure wider than it is tall by a factor of five,
    # so past `max_head_cols` the head wraps into its own band underneath.
    n_branch_rows = len(shown) + (1 if elided else 0)
    head_cols = min(len(head), max_head_cols)
    head_rows = -(-len(head) // head_cols) if head else 0
    wrap = head_rows > 1
    spare = 1 if wrap else 0                       # a band for the logits readout
    n_rows = n_branch_rows + (head_rows + spare if wrap else 0)
    n_cols = max(1 + n_stage + (0 if wrap else len(head)), head_cols)
    col_w, row_h = 2.35, 1.28
    flow_h = max(2.0, n_rows * row_h + 0.5)

    # the table mirrors what the figure draws: with 60 blocks the tokenizer rows
    # alone would fill it and the trunk — the part that differs between models —
    # would never appear.
    drawn = [r for r in records if r["group"] == "head" or r["group"] in set(shown)]
    table_lines = _table_lines(drawn, blocks, limit=table_rows,
                               undrawn=len(records) - len(drawn))
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
        bi = _input_index(g)
        src = inputs[bi] if bi is not None and bi < len(inputs) else None
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
        y = n_rows - 1 - len(shown) + (1 - bh) / 2
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

    # head chain: one row to the right of the branches, or a wrapped band below
    if wrap:
        # boustrophedon: the next row runs back the other way, so continuing the
        # chain is one step DOWN rather than an arc across the whole figure.
        top = n_rows - n_branch_rows - 1
        pos = []
        for hi in range(len(head)):
            r, c = divmod(hi, head_cols)
            pos.append((0.05 + (c if r % 2 == 0 else head_cols - 1 - c),
                        top - r + (1 - bh) / 2))
    else:
        hy = (n_rows - 1) / 2 + (1 - bh) / 2 if n_rows > 1 else (1 - bh) / 2
        pos = [(1 + n_stage + hi + 0.05, hy) for hi in range(len(head))]
    for hi, r in enumerate(head):
        x, y = pos[hi]
        _box(ax, x, y, bw, bh, _box_lines(r), r["role"])
        if hi == 0:
            for g in shown:
                _arrow(ax, 1 + len(per_branch[g]) - 1 + 0.05 + bw, row_y[g] + 0.5,
                       x + (bw / 2 if wrap else 0), y + (bh if wrap else bh / 2))
        elif y == pos[hi - 1][1]:
            back = x < pos[hi - 1][0]
            _arrow(ax, x + (bw + 0.05 if back else -0.05), y + bh / 2,
                   x + (bw if back else 0), y + bh / 2)
        else:                       # end of a wrapped row -> straight down
            _arrow(ax, x + bw / 2, pos[hi - 1][1], x + bw / 2, y + bh)

    if head:
        logits = out_tensor.reshape(-1)
        probs = torch.sigmoid(logits)
        txt = "\n".join(f"{lbl}: logit {float(l): .3f} -> p(S) {float(p):.3f}"
                        for lbl, l, p in list(zip(out_labels, logits, probs))[:12])
        lx, ly = pos[-1]
        if wrap:
            ax.text(lx, ly - 0.10, txt, ha="left", va="top", fontsize=7.5,
                    family="monospace", color=INK_DIM)
        else:
            ax.text(n_cols - 0.05, ly - 0.10, txt, ha="right", va="top", fontsize=7.5,
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


def _table_lines(records, blocks, limit, undrawn=0):
    w = (6, 10, 42, 31, 29, 11)
    head = ("#", "group", "operation", "input", "output", "params")
    lines = ["".join(f"{h:<{n}}" for h, n in zip(head, w))
             + "output stats  (mean / std / %nonzero)  or  note",
             "-" * (sum(w) + 48)]
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
    if undrawn:
        lines.append(f"... plus {undrawn} steps in the branches this figure does not "
                     "draw (identical stacks) — all of them are in the .txt trace")
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
          f"{'#':<6}{'group':<11}{'operation':<40}{'input':<30}{'output':<26}"
          f"{'params':<12}output stats / note",
          "-" * 156]
    for i, r in enumerate(records, 1):
        s = r["stats"]
        stat = (f"mean {s['mean']: .4f}  std {s['std']:.4f}  min {s['min']: .3f}  "
                f"max {s['max']: .3f}  nonzero {s['nonzero']*100:.1f}%" if s
                else r.get("note", ""))
        cells = [str(i), r["group"], (r["op"] + " " + r["cfg"]).strip(),
                 _shape_str(r["in"]), _shape_str(r["out"]),
                 f"{r['params']:,}" if r["params"] else "-"]
        widths = (6, 11, 40, 30, 26, 12)
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
                    help="branch rows drawn per figure (default 6; the .txt has all)")
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

    # one load per (scope, encoding). The variant-token archs need
    # reference-difference input; everything else needs the plain one-hot, and
    # neither can be derived from the other (delta needs the H37Rv row).
    need = {(TRACES[t][0], TRACES[t][3] in DELTA_ARCHS) for t in args.traces}
    need_md = any(scope == "md" for scope, _d in need)
    # a delta load feeds per-locus architectures only, so skip the per-modality
    # view for it: merging copies every array a second time.
    need_merge = {(TRACES[t][0], TRACES[t][3] in DELTA_ARCHS)
                  for t in args.traces if TRACES[t][2] == "modality"}

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

        def dataset(key):
            """Load one (scope, encoding) on demand, keeping only ONE alive.

            The multi-drug all-modality load is tens of GB, and adding the
            variant-token archs means up to four of them in a run; holding them
            all at once is what would push this job past its memory ask. The
            trace loop below is ordered by key so nothing is loaded twice."""
            if key not in data:
                data.clear()
                scope, delta = key
                enc = "reference-difference (--delta)" if delta else "one-hot"
                merge = key in need_merge
                if scope == "sd":
                    print(f"[trace] loading single-drug {drug} (all modalities, "
                          f"per-locus, {enc}) ...", flush=True)
                    d = load_dataset(drug, list(MODALITIES), geno, pheno,
                                     regulatory_dir=reg, per_modality_branch=False,
                                     delta=delta)
                    data[key] = (d, *_views(d.blocks, merge=merge), d.y, [drug],
                                 d.isolate_ids)
                else:
                    print(f"[trace] loading multi-drug ({len(md_drugs)} drugs, all "
                          f"modalities, per-locus, {enc}) ...", flush=True)
                    d = load_multidrug_dataset(md_drugs, list(MODALITIES), geno, pheno,
                                               regulatory_dir=reg,
                                               per_modality_branch=False, delta=delta)
                    data[key] = (d, *_views(d.blocks, merge=merge), d.Y, d.drugs,
                                 d.isolate_ids)
            return data[key]

        # grouped by (scope, encoding) so each expensive load happens once
        for name in sorted(args.traces,
                           key=lambda t: (TRACES[t][0],
                                          TRACES[t][3] in DELTA_ARCHS)):
            scope, mods, layout, arch = TRACES[name]
            delta = arch in DELTA_ARCHS
            bundle, per_locus, per_mod, labels, drug_names, ids = dataset((scope, delta))
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
                elif arch == "locusfusion":
                    model = LocusFusionNet.from_blocks(blocks, drug_names=drug_names,
                                                      **LOCUSFUSION_DEFAULTS)
                elif arch in EXPERIMENTAL_MODELS:
                    # same defaults the runners pass, so the traced shapes and
                    # parameter counts are the ones a real arm trains with
                    model = make_experimental(
                        arch, [parse_block_key(b.name) for b in blocks], specs,
                        drug_names=drug_names, **EXPERIMENTAL_DEFAULTS)
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
            # the variant-token nets have no per-block branch: one tokenizer row per
            # block, then a single shared trunk. Saying "one branch per LOCUS" of
            # them would describe a structure they do not have.
            shape_line = ("one TOKEN SET per locus" if arch == "locusfusion" else
                          "one flat token set per isolate" if arch in EXPERIMENTAL_MODELS
                          else "one branch per " + ("LOCUS" if layout == "locus" else "MODALITY"))
            cfg_line = (f"{type(model).__name__}  |  arch={arch}  |  {shape_line}  |  "
                        f"modalities={'+'.join(mods)}  |  {branch_line}  |  "
                        + ("input=delta vs H37Rv  |  " if delta else "")
                        + f"{n_params:,} parameters  |  outputs={n_drugs}")
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
