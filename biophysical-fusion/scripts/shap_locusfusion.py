"""
SHAP attribution for the saved ``locusfusion`` checkpoints — the variant-token
arm of scripts/shap_attribution.py, behind notebooks/shap_locusfusion.ipynb.

Why this is not just ``--arch locusfusion`` on the other script
--------------------------------------------------------------
The conv nets read a dense one-hot, so "column 314 of protein:katG" is a
position in a sequence every isolate carries and the per-column mean over the
explained isolates is the natural summary. locusfusion reads
REFERENCE-DIFFERENCE input: a column is zero in every isolate that matches
H37Rv, and a token exists only where an isolate deviates. Three consequences,
and they are what this script is:

1. **Attribution is sparse and it is a variant.** A nonzero column IS a mutation
   somebody carries, so the unit of the report is not "column 314" but
   "``dna:katG`` column 944 = G, carried by 41 of 300 explained isolates". The
   mean over all isolates that the conv script reports would divide a real
   effect by the 259 wild-type isolates it never applied to, so everything here
   is reported PER CARRIER as well as per isolate.
2. **The config lies about the input.** ``run_experiment.py`` loads with
   ``args.delta or arch in DELTA_ARCHS`` but records ``args.delta``, so every
   locusfusion checkpoint says ``"delta": false`` and was trained on delta
   input. ``shap_attribution.rebuild_inputs`` resolves that (this script reuses
   it); the AUC gate below is what proves the resolution was right.
3. **The wild type is a real baseline.** Expected gradients needs a reference
   distribution, and for this architecture the model's own null — the all-zero
   isolate, which presents 19 [WT] sentinels and no variant tokens — is one.
   ``--background wt`` uses it, and the attribution then reads as "what this
   isolate's deviations from H37Rv buy over a pan-wild-type genome", which is
   the quantity the architecture was built around. ``--background train`` is
   the conventional empirical reference. Both arms run by default; every table
   carries a ``background`` column saying which produced the row.

Written per (cell, drug), all under ``{out}/{run}/{arch}/{DRUG}/``::

    {cell}__{bg}__variants.csv   THE table: one row per (block, column, allele)
                                 an explained isolate carries, with its
                                 attribution and its phenotype cross-tab
    {cell}__{bg}__loci.csv       per-locus attribution share, beside the
                                 read-out attention over the same loci
    {cell}__{bg}__blocks.csv     per-block / per-modality share
    {cell}__{bg}__isolates.csv   per isolate: label, prediction, token count,
                                 total attribution, top variants
    {cell}__{bg}__shap.npz       the sparse per-column arrays, for anything else
    {cell}__{bg}__meta.json      the gate: reported vs re-scored test AUC

Two attribution columns, and they answer different questions:

    signed_attr   sum of the SHAP values over the channel axis. Expected
                  gradients is already grad x (x - background), so this sums
                  over every column to f(x) - E[f(background)] — the
                  completeness property. Use it to say how much of the score
                  a locus is responsible for.
    carried_attr  the same, times the input. On a one-hot that keeps only the
                  channel the isolate actually has, so it names the ALLELE
                  rather than the position. It does not sum to the logit gap.

Orientation, as everywhere in this project: POSITIVE = pushes toward RESISTANT.
The nets emit log-odds of SUSCEPTIBLE, so the SHAP values are negated on the way
out.

Examples (from the project root):
    python scripts/shap_locusfusion.py --drugs ISONIAZID
    python scripts/shap_locusfusion.py --drugs all --cells sd19_dna_protein
    python scripts/shap_locusfusion.py --aggregate
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling scripts

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

import shap  # noqa: E402

from bigtb_ref import tb  # noqa: E402
from training.checkpoint import load_model, run_weights_dir  # noqa: E402

# the rebuild + the AUC gate + the model wrapper are shared with the conv sweep
# ON PURPOSE: if those two disagree about how a checkpoint is reloaded, one of
# the two sets of attributions is wrong and nothing says which.
from shap_attribution import Wrap, predict, rebuild_inputs, resolve_stem  # noqa: E402

RESULTS_DIR = PROJECT_DIR / "results" / "experiments"
DEFAULT_OUT = PROJECT_DIR / "results" / "analysis" / "shap_locusfusion"
ALL_DRUGS = list(tb.DRUG_TO_LOCI)

# blocks whose channel axis is a one-hot alphabet -> a column names an allele.
# biophysical is 3 continuous properties (MW / pI / hydrophobicity), so an
# argmax over its channels names nothing.
ONEHOT_MODALITIES = {"dna", "protein", "regulatory"}
BACKGROUNDS = ("wt", "train")


def background_arrays(data, kind):
    """The reference distribution expected gradients integrates from.

    'wt'    one all-zero isolate — the model's own null. On delta input that is
            a genome identical to H37Rv at every loaded locus, which presents
            nothing but [WT] sentinels, so an attribution against it reads as
            "what this isolate's mutations are worth".
    'train' `n_background` isolates drawn from the training split — the
            conventional empirical reference, against which a common variant
            attributes less because the reference often carries it too.

    The wild-type arm has a property the empirical one does not: expected
    gradients integrates along ``t * x + (1 - t) * background``, and with an
    all-zero background that ray only SCALES the isolate's one-hot, so the same
    columns are occupied at every t and the tokenizer builds the same token set
    the whole way. Against a training isolate the occupancy is the union of the
    two genotypes and the token set changes along the path — still a valid
    attribution, but of a model whose input discretization moved under it.
    """
    if kind == "wt":
        return [np.zeros((1,) + a.shape[1:], dtype=a.dtype) for a in data["test"]]
    return data["bg"]


def compute_shap(model, data, pick, bg, nsamples, device, batch):
    """Per-block SHAP for the isolates in `pick`, oriented toward RESISTANT.

    `batch` is isolates per explainer CALL, which bounds the float64 output
    buffer shap allocates (batch x every block); the peak device allocation is
    set by `nsamples` instead, since expected gradients holds that many
    interpolated copies of every block at once.
    """
    explainer = shap.GradientExplainer(
        Wrap(model), [torch.from_numpy(a).to(device).float() for a in bg])
    chunks = []
    for s in range(0, len(pick), batch):
        sel = pick[s:s + batch]
        raw = explainer.shap_values(
            [torch.from_numpy(a[sel]).to(device).float() for a in data["test"]],
            nsamples=nsamples)
        # a single-block model gets back a bare array, not a one-element list —
        # iterating that walks the isolate axis and produces garbage shapes
        if not isinstance(raw, list):
            raw = [raw]
        # (n, C, L, 1) -> (n, C, L); flip sign so + == toward RESISTANT
        chunks.append([-np.asarray(r).squeeze(-1) for r in raw])
    return [np.concatenate([c[b] for c in chunks]) for b in range(len(chunks[0]))]


@torch.no_grad()
def readout_attention(model, arrays, device, batch=64):
    """(n, n_loci) read-out attention — what the drug query pooled over.

    Free and exact: it is the model's own attention, not an estimate of it, and
    it is the independent check on the SHAP shares. Only defined when stage 2
    holds one token per locus (`carry_variants=0`, the default); a wider stage-2
    set returns None rather than a mislabelled array.
    """
    n_loci = len(model.loci)
    out = []
    for s in range(0, len(arrays[0]), batch):
        xs = [torch.from_numpy(a[s:s + batch]).to(device).float() for a in arrays]
        _logits, attn = model(xs, return_attn=True)
        if attn is None or attn.shape[-1] != n_loci:
            return None
        out.append(attn[:, 0].float().cpu().numpy())      # single-drug: query 0
    return np.concatenate(out)


@torch.no_grad()
def token_census(model, arrays, device, batch=64):
    """Per (isolate, locus): variant tokens the tokenizer built, and truncation.

    `n_variants` is the count BEFORE the `max_variants` cap, so `truncated`
    marks the isolate-loci where the model could not see every deviation — the
    one place this architecture drops input, and worth knowing before reading a
    locus's attribution as complete."""
    n_loci = len(model.loci)
    counts, trunc, unc = [], [], []
    for s in range(0, len(arrays[0]), batch):
        xs = [torch.from_numpy(a[s:s + batch]).to(device).float() for a in arrays]
        n = torch.zeros(xs[0].shape[0], n_loci, device=xs[0].device)
        t = torch.zeros_like(n, dtype=torch.bool)
        u = torch.zeros_like(t)
        # variant_report is flat over (locus, stream); walk it with the plan
        rep = model.variant_report(xs)
        k = 0
        for li, streams in enumerate(model._plan):
            for _stream in streams:
                r = rep[k]
                n[:, li] += r["n_variants"].to(n.dtype)
                t[:, li] |= r["n_variants"] > model.max_variants
                u[:, li] |= r["uncovered"]
                k += 1
        counts.append(n.cpu().numpy())
        trunc.append(t.cpu().numpy())
        unc.append(u.cpu().numpy())
    return (np.concatenate(counts), np.concatenate(trunc), np.concatenate(unc))


def cross_tab(array, col, symbol_idx, y):
    """Phenotype counts for the isolates carrying `symbol_idx` at `col`.

    `symbol_idx=None` counts every isolate that deviates at all, which is what
    a block with no alphabet (biophysical) can be asked.

    Over the WHOLE test split, not the explained subset: it is a
    genotype-phenotype count that does not depend on which isolates SHAP
    sampled, and a variant carried by 30 isolates all but vanishes from a
    300-isolate draw."""
    v = np.abs(array[:, :, col])                   # see `occupancy` on the abs
    m = v.max(1) > 0
    if symbol_idx is not None:
        m = m & (v.argmax(1) == symbol_idx)
    n_r, n_s = int((y[m] == 0).sum()), int((y[m] == 1).sum())
    return n_r, n_s


def occupancy(x):
    """(n, L) mask + (n, L) channel index of the value each isolate carries.

    Taken on |x|, which is the model's own test (`LocusFusionNet._stream_tokens`
    builds its occupancy from ``x.abs().sum(1) > 0``). It matters for the
    biophysical blocks and only for those: their three channels are z-scored MW
    / pI / hydrophobicity, so a real variant column is all-negative whenever the
    substituted residue is below average on all three, and a `max > 0` test
    would drop it from the report while the model saw it.
    """
    a = np.abs(x)
    return a.max(axis=1) > 0, a.argmax(axis=1)


def variant_table(cfg, phi, data, pick, min_carriers):
    """One row per (block, column, allele) an explained isolate actually carries.

    This is the report. A row says: this many of the explained isolates carry
    this symbol at this column, the model moved their score this far because of
    it, and across the whole held-out split that allele is this resistant."""
    y_all = data["y"]
    y_pick = y_all[pick]
    total_abs = sum(float(np.abs(p).sum()) for p in phi)
    rows = []
    for b, p, arr in zip(cfg["model"]["blocks"], phi, data["test"]):
        x = arr[pick]                                  # (n, C, L) one-hot delta
        signed = p.sum(axis=1)                         # (n, L) completeness-safe
        carried = (p * x).sum(axis=1)                  # (n, L) the allele's own
        occ, sym = occupancy(x)                        # who deviates where, with what
        onehot = b["modality"] in ONEHOT_MODALITIES
        names = b["channel_names"] or []
        for col in np.nonzero(occ.any(axis=0))[0]:
            carriers = np.nonzero(occ[:, col])[0]
            if len(carriers) < min_carriers:
                continue
            # A one-hot block splits its carriers by the allele they carry — that
            # is what names the mutation. The biophysical block has no alphabet
            # (three z-scored properties of the same residue), so splitting it by
            # argmax would fragment the carriers of ONE substitution across three
            # rows by which property happens to move most; it gets a single row
            # per column instead, to be read beside the co-indexed protein row.
            groups = ([(int(k), carriers[sym[carriers, col] == k])
                       for k in np.unique(sym[carriers, col])] if onehot
                      else [(None, carriers)])
            for k, who in groups:
                if len(who) < min_carriers:
                    continue
                n_r, n_s = cross_tab(arr, int(col), k, y_all)
                rows.append({
                    "block": b["name"], "modality": b["modality"],
                    "locus": b["locus"], "column": int(col),
                    "position_1based": int(col) + 1,
                    "symbol": (names[k] if onehot and k < len(names)
                               else "(continuous)"),
                    "n_carriers": int(len(who)),
                    "carrier_frac": float(len(who) / len(pick)),
                    "mean_carried_attr": float(carried[who, col].mean()),
                    "mean_signed_attr": float(signed[who, col].mean()),
                    "sum_abs_attr": float(np.abs(p[who][:, :, col]).sum()),
                    "share_of_total": (float(np.abs(p[who][:, :, col]).sum()
                                             / total_abs) if total_abs > 0 else 0.0),
                    "carriers_R": int((y_pick[who] == 0).sum()),
                    "carriers_S": int((y_pick[who] == 1).sum()),
                    "test_n_R": n_r, "test_n_S": n_s,
                    "test_pct_R": (100.0 * n_r / (n_r + n_s)) if n_r + n_s else np.nan,
                })
    df = pd.DataFrame(rows)
    return (df.sort_values("mean_carried_attr", ascending=False)
            if len(df) else df)


def block_table(cfg, phi):
    """Per-block attribution total and its share of the model's whole budget."""
    t = pd.DataFrame([
        {"block": b["name"], "modality": b["modality"], "locus": b["locus"],
         "length": b["length"],
         "mean_abs_attr": float(np.abs(p).sum(axis=(1, 2)).mean()),
         "mean_signed_attr": float(p.sum(axis=(1, 2)).mean())}
        for b, p in zip(cfg["model"]["blocks"], phi)])
    total = t["mean_abs_attr"].sum()
    t["share"] = t["mean_abs_attr"] / total if total > 0 else 0.0
    return t


def locus_table(blocks, model, attn, counts, trunc, unc, y_pick):
    """Per-locus SHAP share beside the model's own read-out attention.

    The two are independent readings of "which gene did this decision come
    from" — one from the input gradients, one from the attention weights — so
    they are put side by side rather than reported separately."""
    g = (blocks.groupby("locus")[["mean_abs_attr", "mean_signed_attr", "share"]]
         .sum().reset_index())
    order = {l: i for i, l in enumerate(model.loci)}
    g["locus_index"] = g["locus"].map(order)
    g = g.dropna(subset=["locus_index"]).sort_values("locus_index")
    li = g["locus_index"].astype(int).to_numpy()
    if attn is not None:
        g["attn_mean"] = attn[:, li].mean(axis=0)
        g["attn_mean_R"] = attn[y_pick == 0][:, li].mean(axis=0)
        g["attn_mean_S"] = attn[y_pick == 1][:, li].mean(axis=0)
    g["mean_variant_tokens"] = counts[:, li].mean(axis=0)
    g["frac_truncated"] = trunc[:, li].mean(axis=0)
    g["frac_uncovered"] = unc[:, li].mean(axis=0)
    return g.drop(columns="locus_index")


def isolate_table(cfg, phi, data, pick, p_resistant, counts, top_k=3):
    """Per explained isolate: what it is, what the model said, what drove it."""
    names, cols, carried = [], [], []
    for b, p, arr in zip(cfg["model"]["blocks"], phi, data["test"]):
        x = arr[pick]
        names.append(b["name"])
        cols.append((p * x).sum(axis=1))               # (n, L)
        carried.append(np.abs(p).sum(axis=(1, 2)))     # (n,)
    rows = []
    for i, gi in enumerate(pick):
        best = []
        for bi, c in enumerate(cols):
            for col in np.argsort(np.abs(c[i]))[::-1][:top_k]:
                if c[i, col] != 0:
                    best.append((abs(c[i, col]), f"{names[bi]}:{col + 1}"
                                 f"({c[i, col]:+.3f})"))
        best.sort(reverse=True)
        rows.append({
            "isolate_id": data["ids"][gi], "y_resistant": int(data["y"][gi] == 0),
            "p_resistant": float(p_resistant[gi]),
            "n_variant_tokens": float(counts[i].sum()),
            "total_signed_attr": float(sum(p.sum(axis=(1, 2))[i] for p in phi)),
            "total_abs_attr": float(sum(c[i] for c in carried)),
            "top_variants": "; ".join(s for _, s in best[:top_k]),
        })
    return pd.DataFrame(rows)


def save_sparse(path, cfg, phi, data, pick):
    """Per-column attribution, sparse over the columns anybody deviates at.

    The dense arrays are (300, 20, 4066) per block x 38 blocks and ~99.9% zero
    by construction; this keeps the columns that carry anything, which is what
    the notebook re-reads."""
    out = {"pick": pick, "y": data["y"][pick], "ids": np.array(data["ids"][pick])}
    for i, (b, p, arr) in enumerate(zip(cfg["model"]["blocks"], phi, data["test"])):
        x = arr[pick]
        occ, sym = occupancy(x)
        keep = np.nonzero((np.abs(p).sum(axis=(0, 1)) > 0) | occ.any(0))[0]
        out[f"b{i}_name"] = b["name"]
        out[f"b{i}_cols"] = keep.astype(np.int32)
        out[f"b{i}_signed"] = p[:, :, keep].sum(axis=1).astype(np.float32)
        out[f"b{i}_carried"] = (p[:, :, keep] * x[:, :, keep]).sum(axis=1).astype(np.float32)
        out[f"b{i}_symbol"] = np.where(occ[:, keep], sym[:, keep], -1).astype(np.int8)
    np.savez_compressed(path, **out)


def run_cell(cell, drug, args, rng):
    """One (cell, drug): rebuild, gate on AUC, attribute under each background."""
    run_dir = RESULTS_DIR / args.run / f"{cell}__{args.arch}"
    stem = resolve_stem(run_dir, drug)
    weights = run_weights_dir(f"{args.run}/{cell}__{args.arch}", stem)
    model, cfg = load_model(f"{args.run}/{cell}__{args.arch}", stem,
                            fold=args.fold, map_location=args.device)
    model = model.to(args.device).eval()
    if cfg["model"]["arch"] != "locusfusion":
        raise RuntimeError(f"{cell} is arch {cfg['model']['arch']}, not locusfusion "
                           "— use scripts/shap_attribution.py")

    data = rebuild_inputs(cfg, weights, args.n_background, rng)

    # --- the gate: an attribution from a wrongly-rebuilt model is worthless.
    # For this arch it is also the delta check — dense input scores far away.
    p = predict(model, data["test"], args.device)
    reported = float(pd.read_csv(run_dir / "summary.csv")
                     .set_index("drug").loc[drug, "test_auc"])
    got = float(roc_auc_score(data["y"], p))
    if abs(got - reported) > args.auc_tol:
        raise RuntimeError(
            f"{cell}/{drug}: rebuilt model scores {got:.4f} but summary.csv says "
            f"{reported:.4f} (tol {args.auc_tol}) — rebuild is wrong, not the SHAP")

    n = min(args.n_explain, len(data["y"]))
    pick = np.sort(rng.choice(len(data["y"]), size=n, replace=False))
    p_resistant = 1.0 - p

    xs_pick = [a[pick] for a in data["test"]]
    attn = readout_attention(model, xs_pick, args.device)
    counts, trunc, unc = token_census(model, xs_pick, args.device)

    out_dir = Path(args.out) / args.run / args.arch / drug
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    for bg_kind in args.background:
        t0 = time.time()
        phi = compute_shap(model, data, pick, background_arrays(data, bg_kind),
                           args.nsamples, args.device, args.shap_batch)
        secs = time.time() - t0
        pre = f"{cell}__{bg_kind}"

        blocks = block_table(cfg, phi)
        loci = locus_table(blocks, model, attn, counts, trunc, unc, data["y"][pick])
        variants = variant_table(cfg, phi, data, pick, args.min_carriers)
        isolates = isolate_table(cfg, phi, data, pick, p_resistant, counts)
        for t in (blocks, loci, variants, isolates):
            t.insert(0, "background", bg_kind)
            t.insert(0, "cell", cell)
            t.insert(0, "drug", drug)
        blocks.to_csv(out_dir / f"{pre}__blocks.csv", index=False)
        loci.to_csv(out_dir / f"{pre}__loci.csv", index=False)
        variants.to_csv(out_dir / f"{pre}__variants.csv", index=False)
        isolates.to_csv(out_dir / f"{pre}__isolates.csv", index=False)
        if not args.no_npz:
            save_sparse(out_dir / f"{pre}__shap.npz", cfg, phi, data, pick)

        # completeness: expected gradients sums to f(x) - E[f(background)], so a
        # large residual means the attribution does not account for the score
        # and no share computed from it is trustworthy.
        with torch.no_grad():
            bg_logit = float(torch.cat([
                model([torch.from_numpy(a).to(args.device).float()
                       for a in background_arrays(data, bg_kind)])]).mean())
        pr = np.clip(p_resistant[pick], 1e-7, 1 - 1e-7)
        logits = np.log(pr / (1 - pr))
        attr_sum = sum(p_.sum(axis=(1, 2)) for p_ in phi)
        gap = logits - (-bg_logit)
        meta = {"drug": drug, "cell": cell, "run": args.run, "arch": args.arch,
                "background": bg_kind,
                "fold": cfg["best_fold"] if args.fold is None else args.fold,
                "n_test": int(len(data["y"])), "n_explained": int(n),
                "n_background": (1 if bg_kind == "wt" else int(len(data["bg"][0]))),
                "nsamples": args.nsamples, "seed": args.seed,
                "test_auc_reported": reported, "test_auc_rescored": got,
                "test_auc_pr": float(average_precision_score(1 - data["y"], p_resistant)),
                "n_variant_rows": int(len(variants)),
                "mean_variant_tokens": float(counts.sum(axis=1).mean()),
                "frac_isolates_truncated": float((trunc.any(axis=1)).mean()),
                "completeness_r": (float(np.corrcoef(attr_sum, gap)[0, 1])
                                   if np.std(gap) > 0 else float("nan")),
                "completeness_mae": float(np.abs(attr_sum - gap).mean()),
                "shap_seconds": round(secs, 1)}
        (out_dir / f"{pre}__meta.json").write_text(json.dumps(meta, indent=2))
        metas.append(meta)

        by_mod = blocks.groupby("modality")["share"].sum()
        top = loci.nlargest(3, "share")["locus"].tolist()
        print(f"  [{drug}/{cell}/{bg_kind}] AUC {got:.4f} (reported {reported:.4f}) | "
              f"{n} isolates in {secs:.0f}s | {len(variants)} variant rows | "
              f"top loci {top} | shares: "
              + ", ".join(f"{m}={v:.1%}" for m, v in by_mod.items()), flush=True)
        del phi
        gc.collect()

    del model, data
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return metas


def aggregate(out, run, arch):
    """Combine the per-drug files into one table each. Run after the sweep."""
    root = Path(out) / run / arch
    for name, pattern in (("blocks_all.csv", "*/*__blocks.csv"),
                          ("loci_all.csv", "*/*__loci.csv"),
                          ("variants_all.csv", "*/*__variants.csv"),
                          ("isolates_all.csv", "*/*__isolates.csv")):
        paths = sorted(root.glob(pattern))
        if paths:
            df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
            df.to_csv(root / name, index=False)
            print(f"  {name}: {len(df)} rows from {len(paths)} file(s)")
    metas = [json.loads(p.read_text()) for p in sorted(root.glob("*/*__meta.json"))]
    if metas:
        pd.DataFrame(metas).sort_values(["drug", "cell", "background"]).to_csv(
            root / "shap_index.csv", index=False)
        print(f"  shap_index.csv: {len(metas)} (cell, background) result(s)")
    stale = sorted(root.glob("shap_failures_*.json"))
    if stale:
        print(f"  NOTE: {len(stale)} failure file(s) still present: "
              + ", ".join(p.name for p in stale))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="newmodels_full", help="run folder")
    ap.add_argument("--arch", default="locusfusion")
    ap.add_argument("--cells", nargs="+",
                    default=["sd19_dna_protein", "sd19_all_modalities"],
                    metavar="CELL", help="modality cells to attribute")
    ap.add_argument("--drugs", nargs="+", default=["ISONIAZID"], metavar="DRUG",
                    help="drug(s), or 'all'")
    ap.add_argument("--background", nargs="+", default=list(BACKGROUNDS),
                    choices=BACKGROUNDS,
                    help="reference distribution(s): 'wt' = the all-zero "
                         "wild-type isolate, 'train' = a sample of the training "
                         "split (default: both)")
    ap.add_argument("--fold", type=int, default=None,
                    help="CV fold to explain (default: the config's best_fold, "
                         "the model TEST was scored on)")
    ap.add_argument("--n-background", type=int, default=100,
                    help="training isolates in the 'train' reference")
    ap.add_argument("--n-explain", type=int, default=300)
    ap.add_argument("--nsamples", type=int, default=64,
                    help="expected-gradient samples per isolate")
    ap.add_argument("--shap-batch", type=int, default=25,
                    help="isolates per explainer call")
    ap.add_argument("--min-carriers", type=int, default=2,
                    help="drop a (column, allele) carried by fewer explained "
                         "isolates than this")
    ap.add_argument("--auc-tol", type=float, default=5e-3,
                    help="max |rescored - reported| test AUC before refusing. "
                         "The gate catches a WRONG rebuild — wrong block order, "
                         "wrong split, dense input where the run used --delta — "
                         "which lands far away, not float noise.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-npz", action="store_true",
                    help="skip the raw sparse arrays, write only the tables")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--aggregate", action="store_true",
                    help="only combine existing per-drug files into the *_all.csv "
                         "tables, then exit. Run once after the sweep drains.")
    args = ap.parse_args()

    if args.aggregate:
        print(f"aggregating {args.out}/{args.run}/{args.arch}")
        aggregate(args.out, args.run, args.arch)
        return

    drugs = (ALL_DRUGS if any(d.lower() == "all" for d in args.drugs)
             else [d.upper() for d in args.drugs])
    unknown = [d for d in drugs if d not in ALL_DRUGS]
    if unknown:
        ap.error(f"unknown drug(s) {unknown}; choose from {ALL_DRUGS} or 'all'")

    print(f"SHAP attribution (locusfusion) | run={args.run} cells={args.cells} "
          f"drugs={len(drugs)} backgrounds={args.background} device={args.device}")
    print(f"  n_background={args.n_background} n_explain={args.n_explain} "
          f"nsamples={args.nsamples} seed={args.seed}\n", flush=True)

    metas, failures = [], []
    for drug in drugs:
        for cell in args.cells:
            done = (Path(args.out) / args.run / args.arch / drug /
                    f"{cell}__{args.background[-1]}__variants.csv")
            if args.skip_existing and done.exists():
                print(f"  [{drug}/{cell}] skip (exists)", flush=True)
                continue
            # a rebuild that fails for one drug must not lose the other ten
            try:
                metas += run_cell(cell, drug, args, np.random.default_rng(args.seed))
            except Exception as e:                      # noqa: BLE001
                print(f"  [{drug}/{cell}] FAILED: {type(e).__name__}: {e}", flush=True)
                failures.append({"drug": drug, "cell": cell,
                                 "error": f"{type(e).__name__}: {e}"})

    root = Path(args.out) / args.run / args.arch
    root.mkdir(parents=True, exist_ok=True)
    print(f"\nwrote {len(metas)} result(s) -> {root}")

    # Per-drug files only: the sweep runs one job PER DRUG into this same
    # directory, so a job that also wrote the combined CSVs would race every
    # sibling. `--aggregate` builds those once, after the sweep drains.
    if failures:
        tag = "_".join(sorted({f["drug"] for f in failures}))
        (root / f"shap_failures_{tag}.json").write_text(json.dumps(failures, indent=2))
        print(f"\n{len(failures)} FAILED:")
        for f in failures:
            print(f"  {f['drug']}/{f['cell']}: {f['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
