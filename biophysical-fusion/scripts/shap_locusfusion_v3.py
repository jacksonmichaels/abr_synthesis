"""
SHAP attribution for the LocusFusion v3 checkpoints (``locusfusion_v2`` built
with ``unknown_symbols=True``) — the v3 arm of scripts/shap_locusfusion.py.

Same method, same settings, same output layout as the v1 study
(results/experiments/SHAP_LOCUSFUSION_20260901.md): expected gradients over the
raw per-block delta input, ``--n-explain 300 --nsamples 64 --shap-batch 25``,
seed 0, the config's ``best_fold``, POSITIVE = toward RESISTANT, and the AUC
reproduction gate (refuse if |rescored - reported test AUC| > 5e-3).

What differs from v1, and why
-----------------------------
* **Code comes from the frozen v3 snapshot.** Models, datasets, checkpoint
  loader and the v1 helpers are imported from
  ``/scratch3/.../v3_snapshot/biophysical-fusion`` (BASE_COMMIT +
  working_tree.patch), i.e. exactly the code the checkpoints were trained with.
  Only results are written into this repo.
* **Data is loaded with unknown_symbols=True**, delta input, reference protein
  frame, per-locus blocks — the run_experiment.py path for locusfusion_v2.
* **One load per drug, shared by every cell.** The 19-locus cells differ only
  in which modality blocks they carry, so a task loads the union once and hands
  each cell its own blocks by name. The gate still runs per cell; if it fails on
  the shared load the cell is retried on a per-cell load before giving up.
* **v2 has one plan entry per locus** (a locus's regulatory window is merged
  into its axis by the model itself), so the read-out attention and the token
  census are per locus, and ``regulatory:X`` is credited to locus ``X``.
* **Per-channel table** (``__channels.csv``): |attr| per block x channel, so
  the share landing on the uncalled-base channels (dna N, protein '-' / 'X',
  regulatory N) can be read off directly.
* **Variant rows carry H37Rv coordinates**: ``genome_pos`` (1-based H37Rv
  position of the column's base; for a protein/biophysical codon, its first
  base) and ``codon`` (1-based, H37Rv frame, from the model's own codon map).

Written per (drug, cell, background) under ``{out}/{run}/{arch}/{DRUG}/``,
mirroring v1: ``__blocks / __channels / __loci / __variants / __isolates .csv``,
``__shap.npz`` and ``__meta.json``. ``--aggregate`` then writes, under
``{out}``: modality_shares.csv, locus_shares.csv, top_features.csv, gate.csv,
locus_mapping.csv (+ the v1-style *_all.csv under ``{out}/{run}/{arch}``).

``--split lineage`` (or ABR_SHAP_SPLIT=lineage) switches snapshot, weights root,
run and default --out to the lineage-grouped retrain (see SPLITS below); the
checkpoint configs carry the lineage test split, and the best fold is the
config's best_fold (= the run JSON's test_model_fold, the best CV-AUC fold).

Examples (from the project root, on a GPU node — never on the login node):
    python scripts/shap_locusfusion_v3.py --drugs LEVOFLOXACIN --cells sd19_dna_unk
    python scripts/shap_locusfusion_v3.py --aggregate
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent          # the repo: outputs only
WS = Path("/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis")
DATA = WS / "data_snapshot"

# Which trained grid to attribute. Picked BEFORE any snapshot import (the
# snapshot goes on sys.path at import time), from `--split X` / `--split=X` on
# the command line or else $ABR_SHAP_SPLIT; scripts/shapley_locusfusion_v3.py
# imports this module and so inherits the same choice.
#   random   LocusFusion v3 sweep, random stratified split (the original study)
#   lineage  the same cells retrained on lineage-grouped splits
#            (results/experiments/lineage_split/submit.sh; test split + CV folds
#            from results/experiments/lineage_split/splits/{DRUG}.csv, recorded
#            in each checkpoint's config["split"]["test_isolate_ids"])
SPLITS = {
    "random": dict(snapshot=WS / "v3_snapshot" / "biophysical-fusion",
                   weights=WS / "model_weights_v3", run="locusfusion_v3",
                   out=PROJECT_DIR / "results" / "analysis" / "shap_locusfusion_v3"),
    "lineage": dict(snapshot=WS / "lineage_snapshot" / "biophysical-fusion",
                    weights=WS / "model_weights_lineage", run="lineage_split",
                    out=PROJECT_DIR / "results" / "analysis"
                    / "shap_locusfusion_lineage_alltest"),
}


def _split_from_argv(argv):
    for i, a in enumerate(argv):
        if a == "--split" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--split="):
            return a.split("=", 1)[1]
    return os.environ.get("ABR_SHAP_SPLIT", "random")


SPLIT = _split_from_argv(sys.argv[1:])
if SPLIT not in SPLITS:
    sys.exit(f"--split / ABR_SHAP_SPLIT must be one of {sorted(SPLITS)}, got {SPLIT!r}")
SNAPSHOT = SPLITS[SPLIT]["snapshot"]                          # frozen training code
WEIGHTS_ROOT = SPLITS[SPLIT]["weights"]
DEFAULT_RUN = SPLITS[SPLIT]["run"]

# The snapshot is read-only; don't try to drop __pycache__ into it.
sys.dont_write_bytecode = True
sys.path.insert(0, str(SNAPSHOT / "scripts"))
sys.path.insert(0, str(SNAPSHOT))
# env overrides the v3 jobs ran with (bigtb_ref / datasets.cds read them at import)
os.environ.setdefault("ABR_GENOTYPE_DIR", str(DATA / "genomic_data" / "aligned"))
os.environ.setdefault("ABR_PHENOTYPE_CSV",
                      str(DATA / "phenotype_data" / "master_resistance_table.csv"))
os.environ.setdefault("ABR_GENBANK", str(DATA / "genbank" / "genome.gbff"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

import bigtb_ref  # noqa: E402
from bigtb_ref import tb  # noqa: E402
from datasets import load_dataset  # noqa: E402
from models import DELTA_ARCHS, REFERENCE_FRAME_ARCHS  # noqa: E402
from training.checkpoint import load_model  # noqa: E402
from shap_attribution import predict, resolve_stem  # noqa: E402
from shap_locusfusion import (block_table, compute_shap, isolate_table,  # noqa: E402
                              occupancy, save_sparse, variant_table)

import shap_locusfusion as _v1  # noqa: E402
for _m in (bigtb_ref, _v1):
    assert Path(_m.__file__).resolve().is_relative_to(SNAPSHOT), _m.__file__

RESULTS_DIR = PROJECT_DIR / "results" / "experiments"
DEFAULT_OUT = SPLITS[SPLIT]["out"]
ALL_DRUGS = sorted(tb.DRUG_TO_LOCI)
BACKGROUNDS = ("wt", "train")
DEFAULT_CELLS = ["sd19_all_modalities_unk", "sd19_dna_regulatory_unk",
                 "sd19_dna_unk", "sd19_dna_protein_unk", "sd19_dna_biophysical_unk"]
MODALITIES = ("dna", "protein", "biophysical", "regulatory")
# the uncalled-base channels v3 added: (modality, channel name)
UNCALLED = {("dna", "N"), ("regulatory", "N"), ("protein", "-"), ("protein", "X")}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_union(cfg, modalities):
    """load_dataset exactly as scripts/run_experiment.py does for a v3 cell,
    over `modalities` (the union of every cell this task attributes). Returns
    the raw bundle: name -> full-length array, plus y / ids."""
    d = cfg["data"]
    arch = cfg["model"]["arch"]
    delta = bool(d.get("delta")) or arch in DELTA_ARCHS
    protein_frame = d.get("protein_frame") or (
        "reference" if arch in REFERENCE_FRAME_ARCHS else "isolate")
    if not d.get("unknown_symbols"):
        raise RuntimeError(f"{cfg['run_name']}: config is not unknown_symbols=True — "
                           "not a v3 checkpoint")
    data = load_dataset(
        cfg["drug"], list(modalities), d["genotype_dir"], d["phenotype_csv"],
        regulatory_dir=d["regulatory_dir"],
        loci=d["loci_override"], regulatory_loci=d["regulatory_loci_override"],
        per_modality_branch=d["per_modality_branch"],
        extra_loci=d["extra_loci"], all_regulatory=d["all_regulatory"],
        delta=delta, protein_frame=protein_frame,
        locus_set=d.get("locus_set", "bigtb"),
        unknown_symbols=True, verbose=False)
    keep = np.nonzero(data.y != -1)[0]
    bundle = {"keep": keep, "y": data.y[keep],
              "ids": np.asarray(data.isolate_ids, dtype=object)[keep],
              "blocks": {b.name: b.array for b in data.blocks},
              "modalities": tuple(modalities)}
    del data
    gc.collect()
    return bundle


def cell_inputs(cfg, weights_dir, bundle, n_background, rng):
    """This cell's blocks, in its config's order, sliced to its own held-out
    split (+ a training-split background). Slices straight from the full arrays
    so the `y != -1` filter never doubles the resident set."""
    names = [b["name"] for b in cfg["model"]["blocks"]]
    missing = [n for n in names if n not in bundle["blocks"]]
    if missing:
        raise RuntimeError(f"blocks {missing} not in the shared load")
    specs = [tuple(s) for s in cfg["model"]["branch_specs"]]
    got = [tuple(bundle["blocks"][n].shape[1:]) for n in names]
    if got != specs:
        bad = [(n, g, s) for n, g, s in zip(names, got, specs) if g != s]
        raise RuntimeError(f"block shapes differ from branch_specs: {bad[:3]}")

    ids = bundle["ids"]
    saved = (Path(weights_dir) / "isolates.txt").read_text().split()
    if list(ids) != saved:
        raise RuntimeError("isolate order drifted from the checkpoint's isolates.txt")
    pos = {s: i for i, s in enumerate(ids)}
    test_ids = cfg["split"]["test_isolate_ids"]
    split_file = cfg["split"].get("split_file")
    if split_file:                      # fixed split (lineage): config must match the CSV
        sp = pd.read_csv(PROJECT_DIR / split_file, dtype={"New_ID": str})
        if set(map(str, test_ids)) != set(sp.loc[sp["role"] == "test", "New_ID"]):
            raise RuntimeError(f"test ids in config differ from {split_file}")
    test_idx = np.array([pos[s] for s in test_ids])
    train_idx = np.setdiff1d(np.arange(len(ids)), test_idx)
    bg_pick = rng.choice(train_idx, size=min(n_background, len(train_idx)),
                         replace=False)
    k = bundle["keep"]
    return {"test": [np.ascontiguousarray(bundle["blocks"][n][k[test_idx]]) for n in names],
            "bg": [np.ascontiguousarray(bundle["blocks"][n][k[bg_pick]]) for n in names],
            "y": bundle["y"][test_idx], "ids": ids[test_idx]}


def background_arrays(data, kind):
    if kind == "wt":
        return [np.zeros((1,) + a.shape[1:], dtype=a.dtype) for a in data["test"]]
    return data["bg"]


# ---------------------------------------------------------------------------
# model-side readings (v2: one plan entry per locus)
# ---------------------------------------------------------------------------

@torch.no_grad()
def readout_attention(model, arrays, device, batch=64):
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
    """(n, n_loci) variant tokens before the cap, and whether the cap bit."""
    counts, trunc = [], []
    for s in range(0, len(arrays[0]), batch):
        xs = [torch.from_numpy(a[s:s + batch]).to(device).float() for a in arrays]
        rep = model.variant_report(xs)
        n = torch.stack([model_r["n_variants"].float() for model_r in rep], 1)
        counts.append(n.cpu().numpy())
        trunc.append((n > model.max_variants).cpu().numpy())
    return np.concatenate(counts), np.concatenate(trunc)


def locus_table(blocks, model, attn, counts, trunc, y_pick):
    """Per-locus share (a locus's regulatory window credited to the locus —
    every block's `locus` field already names its gene), beside the read-out
    attention over the same loci."""
    g = (blocks.groupby("locus")[["mean_abs_attr", "mean_signed_attr", "share"]]
         .sum().reset_index())
    reg = blocks[blocks["modality"] == "regulatory"].groupby("locus")["share"].sum()
    g["regulatory_share"] = g["locus"].map(reg).fillna(0.0)
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
    return g.drop(columns="locus_index")


def channel_table(cfg, phi):
    """|attr| per (block, channel): the uncalled-base share comes from here."""
    rows = []
    for b, p in zip(cfg["model"]["blocks"], phi):
        names = b["channel_names"] or [f"ch{i}" for i in range(p.shape[1])]
        per = np.abs(p).sum(axis=2).mean(axis=0)              # (C,) mean over isolates
        sgn = p.sum(axis=2).mean(axis=0)
        for c, (nm, v, s) in enumerate(zip(names, per, sgn)):
            rows.append({"block": b["name"], "modality": b["modality"],
                         "locus": b["locus"], "channel": nm, "channel_index": c,
                         "uncalled": (b["modality"], nm) in UNCALLED,
                         "mean_abs_attr": float(v), "mean_signed_attr": float(s)})
    t = pd.DataFrame(rows)
    tot = t["mean_abs_attr"].sum()
    t["share"] = t["mean_abs_attr"] / tot if tot > 0 else 0.0
    return t


# ---------------------------------------------------------------------------
# H37Rv coordinates for a variant row
# ---------------------------------------------------------------------------

def _genome_pos(meta, col):
    """1-based H37Rv position of alignment column `col` of a dna/regulatory
    block. A reference-gap column (an insertion) gets the base before it."""
    if not meta or "genome_start" not in meta:
        return None
    gaps = np.asarray(meta.get("ref_gap_cols") or [], dtype=int)
    off = int(col) - int((gaps < col).sum())
    if col in set(gaps.tolist()):
        off -= 1
    if meta.get("strand", "+") == "+":
        return int(meta["genome_start"]) + off + 1
    return int(meta["genome_end"]) - off


def annotate_coords(df, cfg, model):
    if not len(df):
        return df
    blocks = {b["name"]: b for b in cfg["model"]["blocks"]}
    plans = {p["locus"]: (li, p) for li, p in enumerate(model._plan)}
    gpos, codons = [], []
    for name, mod, locus, col in zip(df["block"], df["modality"], df["locus"], df["column"]):
        b = blocks[name]
        li, p = plans[locus]
        cmap = getattr(model, f"codon_map_{li}").cpu().numpy()
        pmap = getattr(model, f"phase_map_{li}").cpu().numpy()
        gp, cd = None, None
        if mod in ("dna", "regulatory"):
            gp = _genome_pos(b["meta"], int(col))
            nt = next((x for x in p["nt"] if x["modality"] == mod), None)
            if nt is not None and int(col) < nt["length"]:
                c = int(cmap[nt["offset"] + int(col)])
                cd = c + 1 if c >= 0 else None
        else:                                       # protein / biophysical: col = codon
            cd = int(col) + 1
            dna = next((x for x in p["nt"] if x["modality"] == "dna"), None)
            hit = np.nonzero((cmap == int(col)) & (pmap == 0))[0]
            if dna is not None and len(hit):
                dcol = int(hit[0]) - dna["offset"]
                if 0 <= dcol < dna["length"]:
                    gp = _genome_pos(blocks[f"dna:{locus}"]["meta"], dcol)
        gpos.append(gp)
        codons.append(cd)
    df = df.copy()
    df.insert(df.columns.get_loc("column") + 1, "genome_pos", pd.array(gpos, dtype="Int64"))
    df.insert(df.columns.get_loc("genome_pos") + 1, "codon", pd.array(codons, dtype="Int64"))
    return df


# ---------------------------------------------------------------------------
# one (drug, cell)
# ---------------------------------------------------------------------------

def gate(model, data, reported, device, tol, label):
    p = predict(model, data["test"], device)
    got = float(roc_auc_score(data["y"], p))
    if abs(got - reported) > tol:
        raise RuntimeError(f"{label}: rebuilt model scores {got:.4f} but the result "
                           f"JSON says {reported:.4f} (tol {tol})")
    return p, got


def run_cell(cell, drug, args, bundle_cache):
    run_name = f"{args.run}/{cell}__{args.arch}"
    run_dir = RESULTS_DIR / run_name
    stem = resolve_stem(run_dir, drug)
    weights = WEIGHTS_ROOT / run_name / stem
    model, cfg = load_model(run_name, stem, fold=args.fold, weights_dir=WEIGHTS_ROOT,
                            map_location=args.device)
    model = model.to(args.device).eval()
    if cfg["model"]["arch"] != "locusfusion_v2" or not model.unknown_symbols:
        raise RuntimeError(f"{cell}: not a v3 (locusfusion_v2 + unknown_symbols) model")
    reported = float(json.loads((run_dir / f"{stem}.json").read_text())["test"]["auc"])

    rng = np.random.default_rng(args.seed)
    union = bundle_cache.get("bundle")
    data, shared = None, union is not None
    if shared:
        try:
            data = cell_inputs(cfg, weights, union, args.n_background, rng)
            p, got = gate(model, data, reported, args.device, args.auc_tol,
                          f"{cell}/{drug} (shared load)")
        except RuntimeError as e:
            print(f"  [{drug}/{cell}] shared load rejected ({e}); per-cell reload",
                  flush=True)
            data, shared = None, False
    if data is None:
        own = load_union(cfg, cfg["data"]["modalities_requested"])
        rng = np.random.default_rng(args.seed)
        data = cell_inputs(cfg, weights, own, args.n_background, rng)
        del own
        gc.collect()
        p, got = gate(model, data, reported, args.device, args.auc_tol, f"{cell}/{drug}")

    n = min(args.n_explain, len(data["y"]))
    pick = np.sort(rng.choice(len(data["y"]), size=n, replace=False))
    p_resistant = 1.0 - p
    xs_pick = [a[pick] for a in data["test"]]
    attn = readout_attention(model, xs_pick, args.device)
    counts, trunc = token_census(model, xs_pick, args.device)

    out_dir = Path(args.out) / args.run / args.arch / drug
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    for bg_kind in args.background:
        pre = f"{cell}__{bg_kind}"
        if args.skip_existing and (out_dir / f"{pre}__meta.json").exists():
            print(f"  [{drug}/{cell}/{bg_kind}] skip (exists)", flush=True)
            continue
        t0 = time.time()
        bg = background_arrays(data, bg_kind)
        phi = compute_shap(model, data, pick, bg, args.nsamples, args.device,
                           args.shap_batch)
        secs = time.time() - t0

        blocks = block_table(cfg, phi)
        chans = channel_table(cfg, phi)
        loci = locus_table(blocks, model, attn, counts, trunc, data["y"][pick])
        variants = annotate_coords(variant_table(cfg, phi, data, pick, args.min_carriers),
                                   cfg, model)
        isolates = isolate_table(cfg, phi, data, pick, p_resistant, counts)
        for t in (blocks, chans, loci, variants, isolates):
            t.insert(0, "background", bg_kind)
            t.insert(0, "cell", cell)
            t.insert(0, "drug", drug)
        blocks.to_csv(out_dir / f"{pre}__blocks.csv", index=False)
        chans.to_csv(out_dir / f"{pre}__channels.csv", index=False)
        loci.to_csv(out_dir / f"{pre}__loci.csv", index=False)
        variants.to_csv(out_dir / f"{pre}__variants.csv", index=False)
        isolates.to_csv(out_dir / f"{pre}__isolates.csv", index=False)
        if not args.no_npz:
            save_sparse(out_dir / f"{pre}__shap.npz", cfg, phi, data, pick)

        with torch.no_grad():
            bg_logit = float(torch.cat([
                model([torch.from_numpy(a).to(args.device).float() for a in bg])
            ]).mean())
        pr = np.clip(p_resistant[pick], 1e-7, 1 - 1e-7)
        logits = np.log(pr / (1 - pr))
        attr_sum = sum(p_.sum(axis=(1, 2)) for p_ in phi)
        gap = logits - (-bg_logit)
        by_mod = blocks.groupby("modality")["share"].sum()
        meta = {"drug": drug, "cell": cell, "split": SPLIT, "run": args.run,
                "arch": args.arch, "background": bg_kind, "stem": stem, "weights": str(weights),
                "fold": cfg["best_fold"] if args.fold is None else args.fold,
                "shared_load": shared,
                "n_test": int(len(data["y"])), "n_explained": int(n),
                "n_background": (1 if bg_kind == "wt" else int(len(data["bg"][0]))),
                "nsamples": args.nsamples, "seed": args.seed,
                "test_auc_reported": reported, "test_auc_rescored": got,
                "auc_abs_diff": abs(got - reported), "gate_tol": args.auc_tol,
                "gate_pass": True,
                "test_auc_pr": float(average_precision_score(1 - data["y"], p_resistant)),
                "n_variant_rows": int(len(variants)),
                "mean_variant_tokens": float(counts.sum(axis=1).mean()),
                "frac_isolates_truncated": float(trunc.any(axis=1).mean()),
                "completeness_r": (float(np.corrcoef(attr_sum, gap)[0, 1])
                                   if np.std(gap) > 0 else float("nan")),
                "completeness_mae": float(np.abs(attr_sum - gap).mean()),
                "uncalled_share": float(chans.loc[chans["uncalled"], "share"].sum()),
                **{f"share_{m}": float(by_mod.get(m, 0.0)) for m in MODALITIES},
                "shap_seconds": round(secs, 1)}
        (out_dir / f"{pre}__meta.json").write_text(json.dumps(meta, indent=2))
        metas.append(meta)

        top = loci.nlargest(3, "share")["locus"].tolist()
        print(f"  [{drug}/{cell}/{bg_kind}] AUC {got:.4f} (reported {reported:.4f}) | "
              f"{n} isolates in {secs:.0f}s | {len(variants)} variant rows | "
              f"top loci {top} | uncalled {meta['uncalled_share']:.1%} | "
              f"compl_r {meta['completeness_r']:.3f} | "
              + ", ".join(f"{m}={v:.1%}" for m, v in by_mod.items()), flush=True)
        del phi
        gc.collect()

    del model, data
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return metas


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def aggregate(out, run, arch, top_n):
    out = Path(out)
    root = out / run / arch
    cat = {}
    for name in ("blocks", "channels", "loci", "variants", "isolates"):
        paths = sorted(root.glob(f"*/*__{name}.csv"))
        if paths:
            cat[name] = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
            cat[name].to_csv(root / f"{name}_all.csv", index=False)
            print(f"  {name}_all.csv: {len(cat[name])} rows from {len(paths)} file(s)")
    metas = [json.loads(p.read_text()) for p in sorted(root.glob("*/*__meta.json"))]
    if not metas:
        print("  nothing to aggregate")
        return
    idx = pd.DataFrame(metas).sort_values(["drug", "cell", "background"])
    idx.to_csv(root / "shap_index.csv", index=False)

    keys = ["drug", "cell", "background"]
    # gate
    idx[keys + ["stem", "fold", "shared_load", "n_test", "n_explained", "n_background",
                "test_auc_reported", "test_auc_rescored", "auc_abs_diff", "gate_tol",
                "gate_pass", "completeness_r", "completeness_mae",
                "mean_variant_tokens", "frac_isolates_truncated",
                "shap_seconds"]].to_csv(out / "gate.csv", index=False)

    # modality + uncalled-channel shares
    ch = cat["channels"]
    mod = ch.pivot_table(index=keys, columns="modality", values="share",
                         aggfunc="sum").reindex(columns=list(MODALITIES)).fillna(0.0)
    mod.columns = [f"share_{c}" for c in mod.columns]
    ch = ch.assign(tag=ch["modality"] + "_" + ch["channel"].map(
        {"N": "N", "-": "gap", "X": "X"}).fillna(""))
    unc = ch[ch["uncalled"]].pivot_table(index=keys, columns="tag", values="share",
                                         aggfunc="sum")
    unc = unc.reindex(columns=["dna_N", "protein_gap", "protein_X", "regulatory_N"])
    unc.columns = [f"share_{c}" for c in unc.columns]   # NaN = modality absent
    dna_gap = (ch[(ch["modality"] == "dna") & (ch["channel"] == "-")]
               .groupby(keys)["share"].sum().rename("share_dna_gap"))
    ms = mod.join(unc).join(dna_gap)
    ms["share_uncalled"] = ms[[c for c in unc.columns]].sum(axis=1, min_count=0)
    ms = ms.reset_index().merge(idx[keys + ["test_auc_rescored", "completeness_r"]],
                                on=keys)
    ms.to_csv(out / "modality_shares.csv", index=False)

    # locus shares (regulatory window merged into its gene), renormalised per row
    lo = cat["loci"]
    lo = lo.assign(share=lo["share"] / lo.groupby(keys)["share"].transform("sum"))
    lo["rank"] = lo.groupby(keys)["share"].rank(ascending=False, method="first").astype(int)
    cols = keys + ["locus", "share", "rank", "regulatory_share", "mean_abs_attr",
                   "mean_signed_attr"] + [c for c in ("attn_mean", "attn_mean_R",
                                                      "attn_mean_S") if c in lo] \
        + ["mean_variant_tokens", "frac_truncated"]
    lo[cols].sort_values(keys + ["rank"]).to_csv(out / "locus_shares.csv", index=False)

    # top features
    v = cat["variants"].copy()
    for c in ("genome_pos", "codon"):
        v[c] = v[c].round().astype("Int64")
    v["mean_abs_attr_per_carrier"] = v["sum_abs_attr"] / v["n_carriers"]
    v["channel"] = v["symbol"]
    v["uncalled"] = [(m, s) in UNCALLED for m, s in zip(v["modality"], v["symbol"])]
    v["rank"] = v.groupby(keys)["share_of_total"].rank(ascending=False,
                                                       method="first").astype(int)
    tv = v[v["rank"] <= top_n].sort_values(keys + ["rank"])
    tv[keys + ["rank", "locus", "modality", "block", "column", "position_1based",
               "genome_pos", "codon", "channel", "uncalled", "n_carriers",
               "carrier_frac", "mean_abs_attr_per_carrier", "mean_carried_attr",
               "mean_signed_attr", "share_of_total", "carriers_R", "carriers_S",
               "test_n_R", "test_n_S", "test_pct_R"]].to_csv(out / "top_features.csv",
                                                             index=False)

    # locus mapping actually used
    b = cat["blocks"][["block", "modality", "locus"]].drop_duplicates()
    b.sort_values(["locus", "modality"]).to_csv(out / "locus_mapping.csv", index=False)

    fails = sorted(root.glob("shap_failures_*.json"))
    print(f"  gate.csv {len(idx)} rows | modality_shares {len(ms)} | "
          f"locus_shares {len(lo)} | top_features {len(tv)}")
    if fails:
        print(f"  NOTE: {len(fails)} failure file(s): " + ", ".join(p.name for p in fails))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=sorted(SPLITS), default=SPLIT,
                    help="trained grid: snapshot, weights root, run and default --out "
                         "(read before import; also $ABR_SHAP_SPLIT)")
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--arch", default="locusfusion_v2")
    ap.add_argument("--cells", nargs="+", default=DEFAULT_CELLS, metavar="CELL")
    ap.add_argument("--drugs", nargs="+", default=["LEVOFLOXACIN"], metavar="DRUG")
    ap.add_argument("--background", nargs="+", default=["train"], choices=BACKGROUNDS)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n-background", type=int, default=100)
    ap.add_argument("--n-explain", type=int, default=300)
    ap.add_argument("--nsamples", type=int, default=64)
    ap.add_argument("--shap-batch", type=int, default=25)
    ap.add_argument("--min-carriers", type=int, default=2)
    ap.add_argument("--auc-tol", type=float, default=5e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-npz", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    if args.aggregate:
        print(f"aggregating {args.out}")
        aggregate(args.out, args.run, args.arch, args.top_n)
        return

    drugs = (ALL_DRUGS if any(d.lower() == "all" for d in args.drugs)
             else [d.upper() for d in args.drugs])
    bad = [d for d in drugs if d not in ALL_DRUGS]
    if bad:
        ap.error(f"unknown drug(s) {bad}")

    print(f"SHAP v3 | split={SPLIT} snapshot={SNAPSHOT} weights={WEIGHTS_ROOT} | "
          f"run={args.run} cells={args.cells} "
          f"drugs={drugs} backgrounds={args.background} device={args.device}")
    print(f"  n_background={args.n_background} n_explain={args.n_explain} "
          f"nsamples={args.nsamples} seed={args.seed}", flush=True)

    root = Path(args.out) / args.run / args.arch
    root.mkdir(parents=True, exist_ok=True)
    metas, failures = [], []
    for drug in drugs:
        todo = [c for c in args.cells if not (args.skip_existing and all(
            (root / drug / f"{c}__{bg}__meta.json").exists() for bg in args.background))]
        if not todo:
            print(f"  [{drug}] all cells exist, skip", flush=True)
            continue
        # one shared load: the union of the cells' modalities, built from the
        # config of the widest cell
        cache = {}
        try:
            cfgs = {}
            for c in todo:
                rd = RESULTS_DIR / args.run / f"{c}__{args.arch}"
                stem = resolve_stem(rd, drug)
                cfgs[c] = json.loads((WEIGHTS_ROOT / args.run / f"{c}__{args.arch}"
                                      / stem / "config.json").read_text())
            widest = max(cfgs, key=lambda c: len(cfgs[c]["data"]["modalities_requested"]))
            mods = [m for m in MODALITIES
                    if any(m in cfgs[c]["data"]["modalities_requested"] for c in todo)]
            t0 = time.time()
            cache["bundle"] = load_union(cfgs[widest], mods)
            print(f"  [{drug}] shared load {mods} in {time.time() - t0:.0f}s "
                  f"({len(cache['bundle']['y'])} labelled isolates)", flush=True)
        except Exception as e:                          # noqa: BLE001
            print(f"  [{drug}] shared load failed ({type(e).__name__}: {e}); "
                  "per-cell loads", flush=True)
        for cell in todo:
            try:
                metas += run_cell(cell, drug, args, cache)
            except Exception as e:                      # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"  [{drug}/{cell}] FAILED: {type(e).__name__}: {e}", flush=True)
                failures.append({"drug": drug, "cell": cell,
                                 "error": f"{type(e).__name__}: {e}"})
        cache.clear()
        gc.collect()

    print(f"\nwrote {len(metas)} result(s) -> {root}")
    if failures:
        tag = "_".join(sorted({f["drug"] for f in failures}))
        (root / f"shap_failures_{tag}.json").write_text(json.dumps(failures, indent=2))
        print(f"\n{len(failures)} FAILED:")
        for f in failures:
            print(f"  {f['drug']}/{f['cell']}: {f['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
