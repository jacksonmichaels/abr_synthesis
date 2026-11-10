"""
Raw per-isolate outputs of every LocusFusion v3 checkpoint (no training, no
attribution): each cell's saved best-fold weights scored on every labelled
isolate of its own data, with the isolate's role in that cell's split.

Covers all 28 cells under results/experiments/locusfusion_v3 — sd / sd19 /
ablation_sd19 / seeds s1000-4000 / partitions cv101-104 (single-drug, one
checkpoint per drug) and md_multidrug_* (one checkpoint, 11 drug heads).

Code path. Single-drug cells were trained from the frozen v3 snapshot
(results/experiments/locusfusion_v3/README.md), so they are rebuilt with it
(--code snapshot). The md_* cells were trained from the repo at the same base
commit (their config argv carries --lf-unknown-symbols, which the snapshot's
run_multidrug.py does not plumb), so they use --code repo. Either way a cell is
refused if its rescored test AUC differs from the result JSON by > 5e-3.

Roles. ``test`` = the cell's stored held-out isolates; ``val`` = the CV fold the
checkpoint early-stopped on (out of sample for this model); ``train`` = the
other four folds. Folds are re-derived exactly as training did
(StratifiedKFold(kfold_seed) on the stratified seed-42 split; KFold(42) on the
unstratified split for multi-drug) and the test half is checked against the
config's test_isolate_ids.

Output, one file per (cell, drug):
  results/analysis/predictions_locusfusion_v3/{cell}/{DRUG}.csv.gz
    isolate_id, y_resistant (1 = R), p_resistant (= 1 - sigmoid(logit)),
    role, best_fold, threshold_p_resistant (1 - the JSON's P(S) operating
    threshold; NaN for multi-drug, which stores none)
  .../{cell}/{DRUG}.meta.json   rescored vs reported test AUC, code path, counts

    python scripts/predict_locusfusion_v3.py --drugs PYRAZINAMIDE --cells 'sd_*'
    python scripts/predict_locusfusion_v3.py --multidrug --code repo
"""
import argparse
import fnmatch
import gc
import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
WS = Path("/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis")
SNAPSHOT = WS / "v3_snapshot" / "biophysical-fusion"
WEIGHTS_ROOT = WS / "model_weights_v3"
DATA = WS / "data_snapshot"
RESULTS = PROJECT_DIR / "results" / "experiments" / "locusfusion_v3"
OUT = PROJECT_DIR / "results" / "analysis" / "predictions_locusfusion_v3"
MODALITIES = ["dna", "protein", "biophysical", "regulatory"]
AUC_TOL = 5e-3


def _code_path():
    code = "repo" if "--code" in sys.argv and sys.argv[sys.argv.index("--code") + 1] == "repo" else "snapshot"
    root = PROJECT_DIR if code == "repo" else SNAPSHOT
    sys.dont_write_bytecode = True                 # the snapshot is read-only
    sys.path.insert(0, str(root))
    os.environ.setdefault("ABR_GENOTYPE_DIR", str(DATA / "genomic_data" / "aligned"))
    os.environ.setdefault("ABR_PHENOTYPE_CSV", str(DATA / "phenotype_data" / "master_resistance_table.csv"))
    os.environ.setdefault("ABR_GENBANK", str(DATA / "genbank" / "genome.gbff"))
    return code, root


CODE, CODE_ROOT = _code_path()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split  # noqa: E402

import bigtb_ref  # noqa: E402
from bigtb_ref import tb  # noqa: E402
from datasets import load_dataset  # noqa: E402
from training.checkpoint import load_model  # noqa: E402
assert Path(bigtb_ref.__file__).resolve().is_relative_to(CODE_ROOT), bigtb_ref.__file__


@torch.no_grad()
def predict(model, arrays, batch=256):
    """sigmoid(logits): P(susceptible), (n, n_heads)."""
    out = []
    for s in range(0, len(arrays[0]), batch):
        xs = [torch.from_numpy(np.ascontiguousarray(a[s:s + batch])).float() for a in arrays]
        out.append(torch.sigmoid(model(xs)).numpy().reshape(len(xs[0]), -1))
    return np.concatenate(out)


def stem_of(cell, drug):
    pointer = json.loads((RESULTS / cell / "weights_location.json").read_text())
    hits = [k for k in pointer if k.startswith(f"{drug}__")]
    if len(hits) != 1:
        raise KeyError(f"{cell}: expected one {drug} stem, got {hits}")
    return hits[0]


def roles(n, y, cfg, stratified):
    """'test' / 'val' / 'train' per row, re-derived as training did."""
    sp = cfg["split"]
    tr, te = train_test_split(np.arange(n), test_size=sp["test_size"], random_state=sp["split_seed"],
                              stratify=y if stratified else None)
    kf = (StratifiedKFold(sp["n_splits"], shuffle=True, random_state=sp["kfold_seed"]).split(tr, y[tr])
          if stratified else KFold(sp["n_splits"], shuffle=True, random_state=sp["kfold_seed"]).split(tr))
    va = dict(enumerate(kf))[cfg["best_fold"]][1]
    role = np.full(n, "train", dtype=object)
    role[te] = "test"
    role[tr[va]] = "val"
    return role


def write(cell, drug, ids, y_res, p_res, role, cfg, threshold_S, reported):
    test = role == "test"
    auc = float(roc_auc_score(y_res[test], p_res[test]))
    if abs(auc - reported) > AUC_TOL:
        raise RuntimeError(f"{cell}/{drug}: rescored test AUC {auc:.4f}, reported {reported:.4f}")
    d = OUT / cell
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"isolate_id": ids, "y_resistant": y_res.astype(int), "p_resistant": p_res,
                  "role": role, "best_fold": cfg["best_fold"],
                  "threshold_p_resistant": 1 - threshold_S}).to_csv(d / f"{drug}.csv.gz", index=False)
    meta = {"cell": cell, "drug": drug, "code": CODE, "test_auc_rescored": auc, "test_auc_reported": reported,
            "best_fold": cfg["best_fold"], "n": int(len(ids)),
            "n_by_role": {r: int((role == r).sum()) for r in ("train", "val", "test")}}
    (d / f"{drug}.meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  {cell}/{drug}: test AUC {auc:.4f} (reported {reported:.4f}) n={len(ids)}", flush=True)


# --- single-drug cells ---------------------------------------------------------

def data_key(cfg):
    d = cfg["data"]
    return json.dumps({k: d.get(k) for k in ("loci_override", "regulatory_loci_override", "per_modality_branch",
                                             "extra_loci", "all_regulatory", "locus_set", "unknown_symbols",
                                             "regulatory_dir", "loci")}, sort_keys=True)


def load_single(cfg):
    """All four modalities, as run_experiment.py loads a v3 cell; y == -1 dropped."""
    d = cfg["data"]
    data = load_dataset(cfg["drug"], MODALITIES, d["genotype_dir"], d["phenotype_csv"],
                        regulatory_dir=d["regulatory_dir"], loci=d["loci_override"],
                        regulatory_loci=d["regulatory_loci_override"],
                        per_modality_branch=d["per_modality_branch"], extra_loci=d["extra_loci"],
                        all_regulatory=d["all_regulatory"], delta=True, protein_frame="reference",
                        locus_set=d.get("locus_set", "bigtb"), unknown_symbols=bool(d["unknown_symbols"]),
                        verbose=False)
    keep = np.nonzero(data.y != -1)[0]
    out = {"y": data.y[keep], "ids": np.asarray(data.isolate_ids, dtype=object)[keep],
           "blocks": {b.name: b.array[keep] for b in data.blocks}}
    del data
    gc.collect()
    return out


def run_drug(drug, cells, skip_existing):
    jobs = []
    for cell in cells:
        if skip_existing and (OUT / cell / f"{drug}.csv.gz").exists():
            continue
        stem = stem_of(cell, drug)
        cfg = json.loads((WEIGHTS_ROOT / "locusfusion_v3" / cell / stem / "config.json").read_text())
        jobs.append((data_key(cfg), cell, stem))
    bundle, key = None, None
    for k, cell, stem in sorted(jobs):
        model, cfg = load_model(f"locusfusion_v3/{cell}", stem, weights_dir=WEIGHTS_ROOT)
        assert cfg["model"]["arch"] == "locusfusion_v2", cell
        assert model.unknown_symbols == bool(cfg["data"]["unknown_symbols"]), cell
        if k != key:
            bundle = None
            gc.collect()
            bundle, key = load_single(cfg), k
            print(f"[{drug}] loaded {len(bundle['ids'])} isolates for {cell}", flush=True)
        saved = (WEIGHTS_ROOT / "locusfusion_v3" / cell / stem / "isolates.txt").read_text().split()
        assert list(bundle["ids"]) == saved, f"{cell}/{drug}: isolate order drifted from the checkpoint"
        names = [b["name"] for b in cfg["model"]["blocks"]]
        got = [tuple(bundle["blocks"][n].shape[1:]) for n in names]
        assert got == [tuple(s) for s in cfg["model"]["branch_specs"]], f"{cell}/{drug}: block shapes differ"
        y = bundle["y"]
        role = roles(len(y), y, cfg, stratified=True)
        assert set(bundle["ids"][role == "test"]) == set(cfg["split"]["test_isolate_ids"]), f"{cell}/{drug}: test split"
        p_S = predict(model, [bundle["blocks"][n] for n in names])[:, 0]
        res = json.loads((RESULTS / cell / f"{stem}.json").read_text())["test"]
        write(cell, drug, bundle["ids"], (y == 0), 1 - p_S, role, cfg, res["threshold"], res["auc"])
        del model


# --- multi-drug cells ----------------------------------------------------------

def run_multidrug(cells, skip_existing):
    from datasets import load_multidrug_dataset
    for cell in cells:
        stem = stem_of(cell, "multidrug")
        cfg = json.loads((WEIGHTS_ROOT / "locusfusion_v3" / cell / stem / "config.json").read_text())
        drugs = cfg["data"]["drugs"]
        if skip_existing and all((OUT / cell / f"{d}.csv.gz").exists() for d in drugs):
            continue
        model, cfg = load_model(f"locusfusion_v3/{cell}", stem, weights_dir=WEIGHTS_ROOT)
        d = cfg["data"]
        data = load_multidrug_dataset(drugs, d["modalities_requested"], d["genotype_dir"], d["phenotype_csv"],
                                      regulatory_dir=d["regulatory_dir"], loci=d["loci"],
                                      per_modality_branch=d["per_modality_branch"],
                                      all_regulatory=d["all_regulatory"], delta=True,
                                      protein_frame=d["protein_frame"],
                                      unknown_symbols=bool(d["unknown_symbols"]), verbose=False)
        keep = np.nonzero((data.Y != -1).any(1))[0]
        Y, ids = data.Y[keep], np.asarray(data.isolate_ids, dtype=object)[keep]
        saved = (WEIGHTS_ROOT / "locusfusion_v3" / cell / stem / "isolates.txt").read_text().split()
        assert list(ids) == saved, f"{cell}: isolate order drifted from the checkpoint"
        blocks = {b.name: b.array for b in data.blocks}
        names = [b["name"] for b in cfg["model"]["blocks"]]
        P_S = predict(model, [blocks[n][keep] for n in names])
        role = roles(len(ids), None, cfg, stratified=False)
        assert set(ids[role == "test"]) == set(cfg["split"]["test_isolate_ids"]), f"{cell}: test split"
        rep = json.loads((RESULTS / cell / f"{stem}.json").read_text())["test_per_drug"]
        for j, drug in enumerate(drugs):
            m = Y[:, j] != -1
            write(cell, drug, ids[m], (Y[m, j] == 0), 1 - P_S[m, j], role[m], cfg, np.nan, rep[drug]["auc"])
        del model, data, blocks
        gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", choices=["snapshot", "repo"], default="snapshot")
    ap.add_argument("--drugs", nargs="+", default=sorted(tb.DRUG_TO_LOCI))
    ap.add_argument("--cells", nargs="+", default=["*"], help="glob(s) over cell folder names")
    ap.add_argument("--multidrug", action="store_true", help="score the md_* cells instead")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    all_cells = sorted(p.name for p in RESULTS.iterdir() if p.is_dir() and p.name.endswith("__locusfusion_v2"))
    cells = [c for c in all_cells if any(fnmatch.fnmatch(c, g) for g in args.cells)]
    md = [c for c in cells if c.startswith("md_")]
    single = [c for c in cells if not c.startswith("md_")]
    print(f"code={CODE} ({CODE_ROOT})", flush=True)
    if args.multidrug:
        run_multidrug(md, args.skip_existing)
        return
    for drug in args.drugs:
        run_drug(drug, single, args.skip_existing)


if __name__ == "__main__":
    main()
