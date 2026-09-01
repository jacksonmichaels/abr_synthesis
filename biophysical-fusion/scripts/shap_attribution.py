"""
SHAP attribution for saved single-drug models — the cross-drug sweep behind
notebooks/shap_notebook.ipynb.

For each (cell, drug) it rebuilds the checkpoint, re-scores it on its own stored
held-out split, runs expected-gradient attribution, and writes three things:

    {out}/{DRUG}/{cell}__shap.npz     raw per-block SHAP arrays (n, C, L)
    {out}/{DRUG}/{cell}__blocks.csv   per-block + per-modality attribution share
    {out}/{DRUG}/{cell}__columns.csv  top columns, WITH the allele table below

The columns file is the point. A bare "column 314 has high attribution" is not a
finding; `protein:katG` column 314 is residue 315, where S is 10.8% resistant and
T is 98.4% (katG S315T). Every one-hot block gets that cross-tab computed
automatically, so a peak arrives already named and already checked against the
phenotype rather than being eyeballed later.

Attribution is oriented so POSITIVE = pushes toward RESISTANT. The nets emit
log-odds of SUSCEPTIBLE (labels are 0=R, 1=S and sigmoid is applied outside the
model), so everything is multiplied by -1 on the way out.

Refuses to write a result whose re-scored test AUC does not match the run's
summary.csv: an attribution from a wrongly-rebuilt model looks perfectly
plausible and is worthless.

Examples (run from the project root):
    python scripts/shap_attribution.py --drugs ISONIAZID
    python scripts/shap_attribution.py --drugs all --device cuda
    python scripts/shap_attribution.py --drugs all --cells dna all_modalities \
        --n-explain 400 --nsamples 128
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np  # noqa: E402  (import after the sys.path bootstrap, by design)
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

import shap  # noqa: E402

from bigtb_ref import tb  # noqa: E402
from datasets import load_dataset  # noqa: E402
from models import DELTA_ARCHS  # noqa: E402
from training.checkpoint import load_model, run_weights_dir  # noqa: E402

RESULTS_DIR = PROJECT_DIR / "results" / "experiments"
DEFAULT_OUT = PROJECT_DIR / "results" / "analysis" / "shap"
ALL_DRUGS = list(tb.DRUG_TO_LOCI)

# blocks whose channel axis is a one-hot alphabet -> a column names an allele.
# biophysical is 3 continuous properties, so argmax over its channels is
# meaningless and it gets no allele table.
ONEHOT_MODALITIES = {"dna", "protein", "regulatory"}


class Wrap(torch.nn.Module):
    """GradientExplainer passes inputs positionally; the nets take one list."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *xs):
        return self.model(list(xs))


def resolve_stem(run_dir, drug):
    """The {DRUG}__{tag} weights stem for this drug, from the run's pointer file."""
    pointer = json.loads((run_dir / "weights_location.json").read_text())
    hits = [k for k in pointer if k.startswith(f"{drug}__")]
    if len(hits) != 1:
        raise KeyError(f"{drug} in {run_dir.name}: expected 1 stem, got {hits}")
    return hits[0]


def rebuild_inputs(cfg, weights_dir, n_background, rng):
    """Reload this checkpoint's inputs, sliced to its own stored held-out split.

    Mirrors scripts/run_experiment.py -> load_dataset, then
    training.multimodal.run_modal_cv's `keep = y != -1` filter.

    `delta` is RESOLVED, not read: the config records `args.delta`, the raw
    flag, while run_experiment.py loads with `args.delta or arch in
    DELTA_ARCHS`. Every variant-token run therefore has `"delta": false` in its
    config and was trained on delta input anyway, so trusting the key rebuilds
    a locusfusion checkpoint on dense one-hot — a different input, a wrong
    model, and attributions that still look plausible."""
    d = cfg["data"]
    delta = bool(d.get("delta")) or cfg["model"]["arch"] in DELTA_ARCHS
    data = load_dataset(
        cfg["drug"], d["modalities_requested"],
        d["genotype_dir"], d["phenotype_csv"], regulatory_dir=d["regulatory_dir"],
        loci=d["loci_override"], regulatory_loci=d["regulatory_loci_override"],
        per_modality_branch=d["per_modality_branch"],
        extra_loci=d["extra_loci"], all_regulatory=d["all_regulatory"],
        delta=delta, verbose=False)

    keep = np.nonzero(data.y != -1)[0]
    y = data.y[keep]
    ids = np.asarray(data.isolate_ids, dtype=object)[keep]
    arrays = [a[keep] for a in data.arrays()]

    saved = (Path(weights_dir) / "isolates.txt").read_text().split()
    if list(ids) != saved:
        raise RuntimeError("isolate order drifted from the checkpoint's isolates.txt")

    pos = {s: i for i, s in enumerate(ids)}
    test_idx = np.array([pos[s] for s in cfg["split"]["test_isolate_ids"]])
    train_idx = np.setdiff1d(np.arange(len(ids)), test_idx)
    bg_pick = rng.choice(train_idx, size=min(n_background, len(train_idx)), replace=False)

    out = {"test": [np.ascontiguousarray(a[test_idx]) for a in arrays],
           "bg": [np.ascontiguousarray(a[bg_pick]) for a in arrays],
           "y": y[test_idx], "ids": ids[test_idx]}
    del data, arrays
    gc.collect()
    return out


@torch.no_grad()
def predict(model, arrays, device, batch=128):
    """P(susceptible) per row — the same path as training.multimodal._predict."""
    out = []
    for s in range(0, len(arrays[0]), batch):
        xs = [torch.from_numpy(a[s:s + batch]).to(device).float() for a in arrays]
        out.append(torch.sigmoid(model(xs)).cpu().numpy().reshape(-1))
    return np.concatenate(out)


def compute_shap(model, data, pick, nsamples, device, batch):
    """Per-block SHAP, oriented toward RESISTANT. Explained rows in `pick` order.

    Batched: expected gradients holds (batch x nsamples) copies of every block at
    once, which for an all-modalities drug is the peak allocation of the run."""
    explainer = shap.GradientExplainer(
        Wrap(model), [torch.from_numpy(a).to(device).float() for a in data["bg"]])
    chunks = []
    for s in range(0, len(pick), batch):
        sel = pick[s:s + batch]
        raw = explainer.shap_values(
            [torch.from_numpy(a[sel]).to(device).float() for a in data["test"]],
            nsamples=nsamples)
        # a SINGLE-block model (KANAMYCIN dna is rrs and nothing else) gets back a
        # bare array, not a one-element list — iterating that walks the isolate
        # axis and silently produces garbage shapes
        if not isinstance(raw, list):
            raw = [raw]
        # (n, C, L, 1) -> (n, C, L); flip sign so + == toward RESISTANT
        chunks.append([-np.asarray(r).squeeze(-1) for r in raw])
    return [np.concatenate([c[b] for c in chunks]) for b in range(len(chunks[0]))]


def allele_table(array, col, channel_names, y, min_n=5):
    """Cross-tab of the symbol at `col` against phenotype, resistant-first.

    `array` is (n, C, L) one-hot; the argmax channel is the symbol the isolate
    actually carries, and an all-zero column is padding / an unknown residue."""
    v = array[:, :, col]
    idx, on = v.argmax(1), v.max(1) > 0
    rows = []
    for k in np.unique(idx[on]):
        m = on & (idx == k)
        n_r = int((y[m] == 0).sum())
        n_s = int((y[m] == 1).sum())
        if n_r + n_s < min_n:
            continue
        rows.append({"symbol": channel_names[k], "n_R": n_r, "n_S": n_s,
                     "pct_R": 100.0 * n_r / (n_r + n_s)})
    n_pad_r = int((y[~on] == 0).sum())
    n_pad_s = int((y[~on] == 1).sum())
    if n_pad_r + n_pad_s >= min_n:
        rows.append({"symbol": "-/pad", "n_R": n_pad_r, "n_S": n_pad_s,
                     "pct_R": 100.0 * n_pad_r / (n_pad_r + n_pad_s)})
    return sorted(rows, key=lambda r: -r["pct_R"])


def block_table(cfg, contrib):
    """Per-block attribution total and its share of the model's whole budget."""
    t = pd.DataFrame([
        {"block": b["name"], "modality": b["modality"], "locus": b["locus"],
         "length": b["length"],
         "mean_abs_attr": float(np.abs(c).sum(axis=1).mean()),
         "mean_signed_attr": float(c.sum(axis=1).mean())}
        for b, c in zip(cfg["model"]["blocks"], contrib)])
    total = t["mean_abs_attr"].sum()
    t["share"] = t["mean_abs_attr"] / total if total > 0 else 0.0
    return t


def column_table(cfg, contrib, data, pick, top_k, min_n):
    """Top-attribution columns per block, each with its allele cross-tab.

    `position_1based` is the residue number for protein blocks — which is what
    makes `protein:katG` 314 legible as katG 315 — and the alignment column + 1
    for nucleotide blocks, where it is NOT a genome coordinate.

    The attribution comes from the explained subset, but the allele cross-tab is
    computed over the WHOLE test split: it is a genotype-phenotype count that
    does not depend on which isolates SHAP happened to sample, and the rare
    alleles that name a mutation vanish from a 200-isolate subset."""
    y = data["y"]
    rows = []
    for b, c, arr in zip(cfg["model"]["blocks"], contrib, data["test"]):
        prof = np.abs(c).mean(axis=0)
        tot = prof.sum()
        onehot = b["modality"] in ONEHOT_MODALITIES
        for h in np.argsort(prof)[::-1][:top_k]:
            rec = {"block": b["name"], "modality": b["modality"], "locus": b["locus"],
                   "column": int(h), "position_1based": int(h) + 1,
                   "mean_abs_attr": float(prof[h]),
                   "share_of_block": float(prof[h] / tot) if tot > 0 else 0.0,
                   "mean_signed_attr": float(c[:, h].mean()),
                   "alleles": ""}
            if onehot:
                tab = allele_table(arr, int(h), b["channel_names"], y, min_n)
                rec["alleles"] = "; ".join(
                    f"{r['symbol']}:{r['pct_R']:.1f}%R(n={r['n_R'] + r['n_S']})"
                    for r in tab)
            rows.append(rec)
    return pd.DataFrame(rows).sort_values("mean_abs_attr", ascending=False)


def run_cell(cell, drug, args, rng):
    """One (cell, drug): rebuild, verify, attribute, write. Returns summary rows."""
    run_dir = RESULTS_DIR / args.run / f"{cell}__{args.arch}"
    stem = resolve_stem(run_dir, drug)
    weights = run_weights_dir(f"{args.run}/{cell}__{args.arch}", stem)
    model, cfg = load_model(f"{args.run}/{cell}__{args.arch}", stem,
                            fold=args.fold, map_location=args.device)
    model = model.to(args.device).eval()

    data = rebuild_inputs(cfg, weights, args.n_background, rng)

    # --- the gate: an attribution from a wrongly-rebuilt model is worthless ---
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

    t0 = time.time()
    sv = compute_shap(model, data, pick, args.nsamples, args.device, args.shap_batch)
    # (n, C, L) SHAP x input -> (n, L) contribution of the value the isolate has
    contrib = [(s * a[pick]).sum(axis=1) for s, a in zip(sv, data["test"])]
    secs = time.time() - t0

    out_dir = Path(args.out) / args.run / args.arch / drug
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{cell}__shap.npz", pick=pick,
                        **{f"b{i}": s for i, s in enumerate(sv)})

    blocks = block_table(cfg, contrib)
    blocks.insert(0, "cell", cell)
    blocks.insert(0, "drug", drug)
    blocks.to_csv(out_dir / f"{cell}__blocks.csv", index=False)

    cols = column_table(cfg, contrib, data, pick, args.top_k, args.min_allele_n)
    cols.insert(0, "cell", cell)
    cols.insert(0, "drug", drug)
    cols.to_csv(out_dir / f"{cell}__columns.csv", index=False)

    meta = {"drug": drug, "cell": cell, "run": args.run, "arch": args.arch,
            "fold": cfg["best_fold"] if args.fold is None else args.fold,
            "n_test": int(len(data["y"])), "n_explained": int(n),
            "n_background": int(len(data["bg"][0])), "nsamples": args.nsamples,
            "seed": args.seed, "test_auc_reported": reported, "test_auc_rescored": got,
            "test_auc_pr": float(average_precision_score(1 - data["y"], 1 - p)),
            "shap_seconds": round(secs, 1)}
    (out_dir / f"{cell}__meta.json").write_text(json.dumps(meta, indent=2))

    by_mod = blocks.groupby("modality")["share"].sum()
    print(f"  [{drug}/{cell}] AUC {got:.4f} (reported {reported:.4f}) | "
          f"{n} isolates in {secs:.0f}s | shares: "
          + ", ".join(f"{m}={v:.1%}" for m, v in by_mod.items()), flush=True)

    del model, data, sv, contrib
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="full_run_v2", help="run folder (default: full_run_v2)")
    ap.add_argument("--arch", default="mdcnn", help="architecture cell (default: mdcnn)")
    ap.add_argument("--cells", nargs="+", default=["dna", "all_modalities"],
                    metavar="CELL", help="modality cells to attribute")
    ap.add_argument("--drugs", nargs="+", default=["ISONIAZID"], metavar="DRUG",
                    help="drug(s), or 'all'")
    ap.add_argument("--fold", type=int, default=None,
                    help="CV fold to explain (default: the config's best_fold, "
                         "i.e. the model TEST was scored on)")
    ap.add_argument("--n-background", type=int, default=100)
    ap.add_argument("--n-explain", type=int, default=200)
    ap.add_argument("--nsamples", type=int, default=64,
                    help="expected-gradient samples per isolate (default: 64)")
    ap.add_argument("--shap-batch", type=int, default=25,
                    help="isolates per explainer call; peak memory is "
                         "batch x nsamples copies of every block (default: 25)")
    ap.add_argument("--top-k", type=int, default=25, help="columns kept per block")
    ap.add_argument("--min-allele-n", type=int, default=5,
                    help="drop alleles rarer than this from the cross-tab")
    ap.add_argument("--auc-tol", type=float, default=5e-3,
                    help="max |rescored - reported| test AUC before refusing "
                         "(default: 5e-3). This gate exists to catch a WRONG "
                         "rebuild — wrong block order, wrong split — which lands "
                         "far away, not to chase float noise. Rebuilding on GPU "
                         "when the run trained on GPU still moves AUC by ~1e-3 on "
                         "the small drugs (AMIKACIN test n=741), so a 1e-3 gate "
                         "rejects correct rebuilds.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a (cell, drug) whose blocks.csv already exists")
    ap.add_argument("--aggregate", action="store_true",
                    help="only combine existing per-drug files into "
                         "blocks_all.csv / columns_all.csv / shap_index.csv, "
                         "then exit. Run once after the sweep drains.")
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

    print(f"SHAP attribution | run={args.run} arch={args.arch} "
          f"cells={args.cells} drugs={len(drugs)} device={args.device}")
    print(f"  n_background={args.n_background} n_explain={args.n_explain} "
          f"nsamples={args.nsamples} seed={args.seed}\n", flush=True)

    metas, failures = [], []
    for drug in drugs:
        for cell in args.cells:
            done = (Path(args.out) / args.run / args.arch / drug /
                    f"{cell}__blocks.csv")
            if args.skip_existing and done.exists():
                print(f"  [{drug}/{cell}] skip (exists)", flush=True)
                continue
            # a rebuild that fails for one drug must not lose the other ten
            try:
                metas.append(run_cell(cell, drug, args,
                                      np.random.default_rng(args.seed)))
            except Exception as e:                      # noqa: BLE001
                print(f"  [{drug}/{cell}] FAILED: {type(e).__name__}: {e}", flush=True)
                failures.append({"drug": drug, "cell": cell,
                                 "error": f"{type(e).__name__}: {e}"})

    root = Path(args.out) / args.run / args.arch
    root.mkdir(parents=True, exist_ok=True)
    print(f"\nwrote {len(metas)} cell(s) -> {root}")

    # Per-drug files only. The sweep runs one job PER DRUG, all writing this same
    # directory at once, so a job that also wrote the combined CSVs would race
    # every sibling and the last finisher would win with a partial view. The
    # notebook globs the per-drug files; `--aggregate` rebuilds the combined ones
    # once, after the sweep has drained.
    if failures:
        tag = "_".join(sorted({f["drug"] for f in failures}))
        (root / f"shap_failures_{tag}.json").write_text(json.dumps(failures, indent=2))
        print(f"\n{len(failures)} FAILED:")
        for f in failures:
            print(f"  {f['drug']}/{f['cell']}: {f['error']}")
        sys.exit(1)


def aggregate(out, run, arch):
    """Combine the per-drug files into one table each. Run after the sweep."""
    root = Path(out) / run / arch
    for name, pattern in (("blocks_all.csv", "*/*__blocks.csv"),
                          ("columns_all.csv", "*/*__columns.csv")):
        paths = sorted(root.glob(pattern))
        if paths:
            df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
            df.to_csv(root / name, index=False)
            print(f"  {name}: {len(df)} rows from {len(paths)} file(s)")
    metas = [json.loads(p.read_text()) for p in sorted(root.glob("*/*__meta.json"))]
    if metas:
        pd.DataFrame(metas).sort_values(["drug", "cell"]).to_csv(
            root / "shap_index.csv", index=False)
        print(f"  shap_index.csv: {len(metas)} cell(s)")
    stale = sorted(root.glob("shap_failures_*.json"))
    if stale:
        print(f"  NOTE: {len(stale)} failure file(s) still present: "
              + ", ".join(p.name for p in stale))


if __name__ == "__main__":
    main()
