"""
SHAP attribution for the JOINT (multi-drug) models — does MD-CNN read the right
locus for the right drug?

The joint model predicts all 11 drugs from one pass over the union of every
drug's loci (19 coding genes; 73 blocks once protein/biophysical/regulatory are
added). Nothing in the architecture ties output column j to any particular
block: the trunk sees every locus and the head is free to key ISONIAZID off
`rpoB` if that happens to correlate. Whether it does is an empirical question,
and this script answers it by attributing EACH output column separately and
scoring the result against `DRUG_TO_LOCI`.

    {out}/{cell}__drug_locus.csv   attribution of every (drug, block) pair
    {out}/{cell}__ontarget.csv     per drug: on-target share vs. a length null
    {out}/{cell}__profiles.npz     per-block (L, n_drugs) position profiles
    {out}/{cell}__columns.csv      top columns per drug, with allele tables

Memory: the raw SHAP tensor for the all-modalities joint model is (n, C, L, 11)
per block over 66,502 positions — ~20 MB PER EXPLAINED ISOLATE. It is never
materialised whole; each batch is reduced to the (block, drug) totals and the
(L, drug) profiles, and then dropped.

Attribution is oriented so POSITIVE = pushes toward RESISTANT (the nets emit
log-odds of SUSCEPTIBLE; labels are 0=R, 1=S).

Examples (run from the project root):
    python scripts/shap_multidrug.py --cells dna
    python scripts/shap_multidrug.py --cells dna all_modalities --device cuda
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling scripts/

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import shap  # noqa: E402

from bigtb_ref import tb  # noqa: E402
from datasets import load_multidrug_dataset  # noqa: E402
from datasets.loader import EXTRA_LOCI  # noqa: E402
from training.checkpoint import load_model, run_weights_dir  # noqa: E402

from shap_attribution import ONEHOT_MODALITIES, Wrap, allele_table  # noqa: E402

RESULTS_DIR = PROJECT_DIR / "results" / "experiments"
DEFAULT_OUT = PROJECT_DIR / "results" / "analysis" / "shap_multidrug"


def rebuild_inputs(cfg, weights_dir, n_background, rng):
    """Reload the joint checkpoint's inputs, sliced to its stored test split.

    Mirrors scripts/run_multidrug.py -> load_multidrug_dataset, then
    training.multidrug.run_multidrug_cv's `keep = (Y != -1).any(axis=1)` filter
    (an isolate survives if it has a phenotype for ANY drug, not all).

    The test slice AND the background sample are both taken here so the
    full-cohort arrays can be released before returning. The joint all-modalities
    cohort is ~575 KB/isolate x 17,941 = ~10 GB; holding it alongside a copy is
    what OOM-kills the job."""
    d = cfg["data"]
    data = load_multidrug_dataset(
        d["drugs"], d["modalities_requested"],
        d["genotype_dir"], d["phenotype_csv"], regulatory_dir=d["regulatory_dir"],
        loci=d["loci"], regulatory_loci=d.get("regulatory_loci"),
        per_modality_branch=d["per_modality_branch"],
        extra_loci=d["extra_loci"], all_regulatory=d["all_regulatory"], verbose=False)

    keep = np.nonzero((data.Y != -1).any(axis=1))[0]
    Y = data.Y[keep]
    ids = np.asarray(data.isolate_ids, dtype=object)[keep]
    arrays = [a[keep] for a in data.arrays()]

    saved = (Path(weights_dir) / "isolates.txt").read_text().split()
    if list(ids) != saved:
        raise RuntimeError("isolate order drifted from the checkpoint's isolates.txt")

    pos = {s: i for i, s in enumerate(ids)}
    test_idx = np.array([pos[s] for s in cfg["split"]["test_isolate_ids"]])
    train_idx = np.setdiff1d(np.arange(len(ids)), test_idx)
    bg_pick = rng.choice(train_idx, size=min(n_background, len(train_idx)),
                         replace=False)

    out = {"test": [np.ascontiguousarray(a[test_idx]) for a in arrays],
           "bg": [np.ascontiguousarray(a[bg_pick]) for a in arrays],
           "Y": Y[test_idx], "ids": ids[test_idx]}
    del data, arrays
    gc.collect()
    return out


@torch.no_grad()
def predict(model, arrays, device, batch=64):
    """(n, n_drugs) P(susceptible)."""
    out = []
    for s in range(0, len(arrays[0]), batch):
        xs = [torch.from_numpy(a[s:s + batch]).to(device).float() for a in arrays]
        out.append(torch.sigmoid(model(xs)).cpu().numpy())
    return np.concatenate(out)


def attribute(model, data, bg, pick, nsamples, device, batch, n_drugs):
    """Streamed attribution. Returns (totals, profiles).

    totals[b]    (n_drugs,)      sum over isolates+positions of |contribution|
    profiles[b]  (L, n_drugs)    sum over isolates of |contribution| per position

    The full (n, C, L, n_drugs) tensor is never held: each batch is reduced to
    these two and released, which is what makes the 73-block model tractable."""
    explainer = shap.GradientExplainer(
        Wrap(model), [torch.from_numpy(a).to(device).float() for a in bg])
    n_blocks = len(data["test"])
    totals = [np.zeros(n_drugs) for _ in range(n_blocks)]
    profiles = [np.zeros((a.shape[-1], n_drugs)) for a in data["test"]]
    signed = [np.zeros(n_drugs) for _ in range(n_blocks)]

    for s in range(0, len(pick), batch):
        sel = pick[s:s + batch]
        xs_np = [a[sel] for a in data["test"]]
        raw = explainer.shap_values(
            [torch.from_numpy(a).to(device).float() for a in xs_np], nsamples=nsamples)
        if not isinstance(raw, list):
            raw = [raw]
        for b, (r, x) in enumerate(zip(raw, xs_np)):
            r = np.asarray(r)
            if r.ndim == 3:                       # single-output safety
                r = r[..., None]
            # (n, C, L, D) * (n, C, L, 1) -> sum over channels -> (n, L, D):
            # the contribution of the value this isolate actually carries
            contrib = -(r * x[..., None]).sum(axis=1)
            totals[b] += np.abs(contrib).sum(axis=(0, 1))
            signed[b] += contrib.sum(axis=(0, 1))
            profiles[b] += np.abs(contrib).sum(axis=0)
        del raw
        gc.collect()

    n = len(pick)
    return ([t / n for t in totals], [p / n for p in profiles],
            [g / n for g in signed])


def on_target_table(cfg, totals, drugs, extended=True):
    """Per drug: how much attribution lands on loci that drug is actually about.

    The null is the share a model would show if it spread attribution evenly per
    POSITION — a drug whose genes are long would otherwise look on-target for
    free. `enrichment` is the ratio; 1.0 is chance."""
    blocks = cfg["model"]["blocks"]
    lengths = np.array([b["length"] for b in blocks], dtype=float)
    mat = np.vstack(totals)                      # (n_blocks, n_drugs)
    rows = []
    for j, drug in enumerate(drugs):
        want = set(tb.DRUG_TO_LOCI.get(drug, []))
        if extended:
            want |= set(EXTRA_LOCI.get(drug, []))
        on = np.array([b["locus"] in want for b in blocks])
        tot = mat[:, j].sum()
        share = mat[on, j].sum() / tot if tot > 0 else np.nan
        null = lengths[on].sum() / lengths.sum()
        order = np.argsort(mat[:, j])[::-1]
        rows.append({
            "drug": drug,
            "on_target_share": share,
            "null_share_by_length": null,
            "enrichment": share / null if null > 0 else np.nan,
            "target_loci": "+".join(sorted(want)),
            "top_blocks": ", ".join(
                f"{blocks[i]['name']}({mat[i, j] / tot:.0%})" for i in order[:4]
                if tot > 0),
            "top_locus_is_on_target": bool(on[order[0]]) if tot > 0 else None,
        })
    return pd.DataFrame(rows)


def drug_locus_table(cfg, totals, signed, drugs):
    """Long-form (drug, block) attribution — the matrix behind the heatmap."""
    blocks = cfg["model"]["blocks"]
    mat, sg = np.vstack(totals), np.vstack(signed)
    rows = []
    for j, drug in enumerate(drugs):
        tot = mat[:, j].sum()
        want = set(tb.DRUG_TO_LOCI.get(drug, [])) | set(EXTRA_LOCI.get(drug, []))
        for i, b in enumerate(blocks):
            rows.append({
                "drug": drug, "block": b["name"], "modality": b["modality"],
                "locus": b["locus"], "length": b["length"],
                "mean_abs_attr": mat[i, j],
                "mean_signed_attr": sg[i, j],
                "share_of_drug": mat[i, j] / tot if tot > 0 else 0.0,
                "on_target": b["locus"] in want,
            })
    return pd.DataFrame(rows)


def column_table(cfg, profiles, data, drugs, top_k, min_n):
    """Top columns per (drug, block), with the allele cross-tab where meaningful.

    The cross-tab uses the whole test split and the drug's own label column,
    skipping isolates with no phenotype for that drug."""
    blocks = cfg["model"]["blocks"]
    rows = []
    for j, drug in enumerate(drugs):
        y = data["Y"][:, j]
        valid = y != -1
        best = np.array([p[:, j].sum() for p in profiles])
        for i in np.argsort(best)[::-1][:top_k]:
            b, prof = blocks[i], profiles[i][:, j]
            tot = prof.sum()
            h = int(np.argmax(prof))
            rec = {"drug": drug, "block": b["name"], "modality": b["modality"],
                   "locus": b["locus"], "column": h, "position_1based": h + 1,
                   "mean_abs_attr": float(prof[h]),
                   "share_of_drug_total": float(best[i] / sum(best)) if sum(best) else 0.0,
                   "share_of_block": float(prof[h] / tot) if tot > 0 else 0.0,
                   "alleles": ""}
            if b["modality"] in ONEHOT_MODALITIES and valid.sum() >= min_n:
                tab = allele_table(data["test"][i][valid], h, b["channel_names"],
                                   y[valid], min_n)
                rec["alleles"] = "; ".join(
                    f"{r['symbol']}:{r['pct_R']:.1f}%R(n={r['n_R'] + r['n_S']})"
                    for r in tab)
            rows.append(rec)
    return pd.DataFrame(rows)


def run_cell(cell, args, rng):
    run_name = f"{args.run}/multidrug_{cell}__{args.arch}"
    run_dir = RESULTS_DIR / args.run / f"multidrug_{cell}__{args.arch}"
    pointer = json.loads((run_dir / "weights_location.json").read_text())
    stem = next(iter(pointer))
    weights = run_weights_dir(run_name, stem)

    model, cfg = load_model(run_name, stem, fold=args.fold, map_location=args.device)
    model = model.to(args.device).eval()
    drugs = cfg["model"]["drug_names"]

    data = rebuild_inputs(cfg, weights, args.n_background, rng)
    print(f"  [{cell}] {len(cfg['model']['blocks'])} blocks, {len(drugs)} drugs, "
          f"test n={len(data['Y'])}", flush=True)

    # --- gate: per-drug test AUC must reproduce multidrug_summary.csv --------
    P = predict(model, data["test"], args.device)
    reported = pd.read_csv(run_dir / "multidrug_summary.csv").set_index("drug")["test_auc"]
    checks = []
    for j, drug in enumerate(drugs):
        v = data["Y"][:, j] != -1
        if v.sum() == 0 or len(np.unique(data["Y"][v, j])) < 2:
            continue
        got = float(roc_auc_score(data["Y"][v, j], P[v, j]))
        checks.append({"drug": drug, "reported": float(reported.get(drug, np.nan)),
                       "rescored": got, "delta": got - float(reported.get(drug, np.nan))})
    chk = pd.DataFrame(checks)
    worst = chk["delta"].abs().max()
    if worst > args.auc_tol:
        raise RuntimeError(
            f"{cell}: worst per-drug test AUC delta {worst:.4f} > {args.auc_tol} "
            f"— rebuild is wrong, not the SHAP\n{chk.to_string(index=False)}")
    print(f"  [{cell}] AUC reproduces (worst delta {worst:+.5f} over "
          f"{len(chk)} drugs)", flush=True)

    bg = data["bg"]                  # already sampled from the TRAIN side
    n = min(args.n_explain, len(data["Y"]))
    pick = np.sort(rng.choice(len(data["Y"]), size=n, replace=False))

    t0 = time.time()
    totals, profiles, signed = attribute(model, data, bg, pick, args.nsamples,
                                         args.device, args.shap_batch, len(drugs))
    secs = time.time() - t0

    out_dir = Path(args.out) / args.run / args.arch
    out_dir.mkdir(parents=True, exist_ok=True)

    dl = drug_locus_table(cfg, totals, signed, drugs)
    dl.insert(0, "cell", cell)
    dl.to_csv(out_dir / f"{cell}__drug_locus.csv", index=False)

    ot = on_target_table(cfg, totals, drugs)
    ot.insert(0, "cell", cell)
    ot.to_csv(out_dir / f"{cell}__ontarget.csv", index=False)

    cols = column_table(cfg, profiles, data, drugs, args.top_k, args.min_allele_n)
    cols.insert(0, "cell", cell)
    cols.to_csv(out_dir / f"{cell}__columns.csv", index=False)

    np.savez_compressed(out_dir / f"{cell}__profiles.npz",
                        drugs=np.array(drugs),
                        blocks=np.array([b["name"] for b in cfg["model"]["blocks"]]),
                        **{f"b{i}": p for i, p in enumerate(profiles)})

    meta = {"cell": cell, "run": args.run, "arch": args.arch, "stem": stem,
            "fold": cfg["best_fold"] if args.fold is None else args.fold,
            "n_blocks": len(cfg["model"]["blocks"]), "drugs": drugs,
            "n_test": int(len(data["Y"])), "n_explained": int(n),
            "n_background": int(len(bg[0])), "nsamples": args.nsamples,
            "seed": args.seed, "worst_auc_delta": float(worst),
            "shap_seconds": round(secs, 1)}
    (out_dir / f"{cell}__meta.json").write_text(json.dumps(meta, indent=2))

    print(f"  [{cell}] {n} isolates in {secs:.0f}s | median on-target share "
          f"{ot['on_target_share'].median():.1%} (null "
          f"{ot['null_share_by_length'].median():.1%}), "
          f"{int(ot['top_locus_is_on_target'].sum())}/{len(ot)} drugs' top locus "
          f"is on-target", flush=True)

    del model, data, totals, profiles, bg
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="full_run_v2")
    ap.add_argument("--arch", default="mdcnn")
    ap.add_argument("--cells", nargs="+", default=["dna", "all_modalities"])
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n-background", type=int, default=50)
    ap.add_argument("--n-explain", type=int, default=150)
    ap.add_argument("--nsamples", type=int, default=64)
    ap.add_argument("--shap-batch", type=int, default=5,
                    help="isolates per explainer call. Much smaller than the "
                         "single-drug default: the joint tensor carries an "
                         "11-wide output axis over 73 blocks (default: 5)")
    ap.add_argument("--top-k", type=int, default=6,
                    help="blocks kept per drug in the columns table")
    ap.add_argument("--min-allele-n", type=int, default=5)
    ap.add_argument("--auc-tol", type=float, default=5e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    print(f"Joint SHAP | run={args.run} arch={args.arch} cells={args.cells} "
          f"device={args.device}")
    print(f"  n_background={args.n_background} n_explain={args.n_explain} "
          f"nsamples={args.nsamples} shap_batch={args.shap_batch}\n", flush=True)

    metas, failures = [], []
    for cell in args.cells:
        try:
            metas.append(run_cell(cell, args, np.random.default_rng(args.seed)))
        except Exception as e:                      # noqa: BLE001
            print(f"  [{cell}] FAILED: {type(e).__name__}: {e}", flush=True)
            failures.append({"cell": cell, "error": f"{type(e).__name__}: {e}"})

    root = Path(args.out) / args.run / args.arch
    if metas:
        root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{k: v for k, v in m.items() if k != "drugs"} for m in metas]
                     ).to_csv(root / "index.csv", index=False)
        print(f"\nwrote {len(metas)} cell(s) -> {root}")
    if failures:
        (root / "failures.json").write_text(json.dumps(failures, indent=2))
        for f in failures:
            print(f"  FAILED {f['cell']}: {f['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
