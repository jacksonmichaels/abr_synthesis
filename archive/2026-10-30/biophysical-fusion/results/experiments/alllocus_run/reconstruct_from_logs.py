#!/usr/bin/env python
"""Rebuild this run's per-cell tables from its archived SLURM logs.

`alllocus_run` (221 jobs, submitted 2026-08-06) trained the whole single-drug
grid on the joint 19-locus input, to separate "joint models win" from "joint
models simply see more loci". Its `results/experiments/alllocus_run/` folder was
lost — the cleanup of 2026-08-13 does not record deleting it, and it is not in
git. What survives is `slurm_logs/archive/20260806/` plus the job manifests, and
those logs print every number a `summary.csv` carries.

This script re-derives the tables from them. Run from the project root:

    python results/experiments/alllocus_run/reconstruct_from_logs.py
    python results/experiments/alllocus_run/reconstruct_from_logs.py --run full_run_v2 --check

`--check` rebuilds a run whose real `summary.csv` files still exist and diffs
against them; that is how this parser was validated before being trusted here.

Per cell it writes `summary.csv` (same columns as a genuine run, `seconds` left
empty — see below), `cv_folds.csv` (per-fold metrics and `best_epoch`) and
`provenance.csv` (drug -> job id, log file, n_params, SLURM state). It writes
`missing.csv` at the run root for the jobs that produced no result.

**`seconds` is empty on purpose.** A real run measures training time inside the
process; the logs do not print it. SLURM's `Elapsed` covers the whole job
including a multi-minute data load, so it is a different quantity and putting it
in a column named `seconds` would be a silent substitution. It is recorded as
`job_elapsed` in `provenance.csv` instead.
"""
import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
MANIFESTS = PROJECT / "slurm_logs" / "manifests"
LOG_GLOBS = ("slurm_logs/archive/*/*.out", "slurm_logs/*.out")

SUMMARY_COLUMNS = ["drug", "modalities", "genes", "n_valid", "n_R", "n_S",
                   "cv_auc_mean", "cv_auc_std", "cv_auc_pr_mean",
                   "test_auc", "test_auc_pr", "test_sens", "test_spec", "seconds"]
FOLD_COLUMNS = ["drug", "modalities", "fold", "auc", "auc_pr", "sens", "spec",
                "n_val", "best_epoch"]
PROV_COLUMNS = ["drug", "modalities", "job_id", "slurm_state", "job_elapsed",
                "max_rss_kb", "arch", "n_params", "n_blocks", "n_test",
                "test_model_fold", "log"]

RE_FOLD = re.compile(
    r"^\[(?P<drug>[^/\]]+)/(?P<mods>[^\]]+)\] CV fold (?P<fold>\d+): "
    r"AUC=(?P<auc>[\d.]+) AUC_PR=(?P<auc_pr>[\d.]+) "
    r"sens=(?P<sens>[\d.]+) spec=(?P<spec>[\d.]+) "
    r"\(n_val=(?P<n_val>\d+), best_epoch=(?P<best_epoch>\d+)\)", re.M)
RE_CV = re.compile(r"\] CV  AUC = (?P<mean>[\d.]+) \+/- (?P<std>[\d.]+)", re.M)
RE_TEST = re.compile(
    r"\] TEST AUC = (?P<auc>[\d.]+) AUC_PR=(?P<auc_pr>[\d.]+) "
    r"sens=(?P<sens>[\d.]+) spec=(?P<spec>[\d.]+) "
    r"\(best CV fold (?P<fold>\d+), n_test=(?P<n_test>\d+)\)", re.M)
RE_HEAD = re.compile(
    r"^\[(?P<drug>[^/\]]+)/(?P<mods>[^\]]+)\] arch=(?P<arch>\S+) "
    r"blocks=\[(?P<blocks>.*?)\] specs=", re.M | re.S)
RE_COUNTS = re.compile(r"n_valid=(?P<n_valid>\d+) R=(?P<n_R>\d+) S=(?P<n_S>\d+)")
RE_PARAMS = re.compile(r"\] (?P<arch>\S+): (?P<n>[\d,]+) parameters")


RE_LOCI = re.compile(r"--loci ((?:[A-Za-z0-9_]+ ?)+?)(?= --|$)")


def manifest_jobs(run):
    """(job_id, cell, drug, loci) for every job submitted under `run`.

    `loci` is the explicit `--loci` list from the recorded command, or [] when the
    job took the per-drug default. It is the only surviving source of gene names
    for a `late_fusion` cell, whose blocks are per-MODALITY -- every locus is
    concatenated into one `dna` block, so the log never names them."""
    out = []
    for p in sorted(MANIFESTS.glob("*.json")):
        for j in json.loads(p.read_text()).get("submitted_jobs", []):
            name = j.get("run_name", "")
            if name.split("/")[0] == run and "/" in name:
                m = RE_LOCI.search(j.get("command", ""))
                out.append((j["job_id"], name.split("/", 1)[1], j["drug"],
                            m.group(1).split() if m else []))
    return out


def log_index():
    """job_id -> log path, from the archived and loose slurm logs."""
    idx = {}
    for g in LOG_GLOBS:
        for p in PROJECT.glob(g):
            m = re.search(r"-(\d+)\.out$", p.name)
            if m:
                idx[m.group(1)] = p
    return idx


def sacct(job_ids):
    """job_id -> (state, elapsed, max_rss_kb). Empty if sacct is unavailable or
    the records have aged out -- this is provenance, not a result."""
    if not job_ids:
        return {}
    try:
        raw = subprocess.run(
            ["sacct", "-j", ",".join(sorted(job_ids)), "-n", "-P",
             "--format=JobID,State,Elapsed,MaxRSS"],
            capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    out = {}
    for line in raw.splitlines():
        f = line.split("|")
        if len(f) < 4:
            continue
        jid, state, elapsed, rss = f[0], f[1], f[2], f[3]
        base = jid.split(".")[0]
        rec = out.setdefault(base, {"state": "", "elapsed": "", "rss": ""})
        if "." not in jid:
            rec["state"], rec["elapsed"] = state, elapsed
        elif rss:
            rec["rss"] = rss.rstrip("K")
    return out


def parse(path, loci=()):
    """One log -> a result dict, or None if the job produced no CV result."""
    text = path.read_text(errors="replace")
    cv, test = RE_CV.search(text), RE_TEST.search(text)
    if not cv or not test:
        return None
    head = RE_HEAD.search(text)
    if not head:
        return None
    folds = [m.groupdict() for m in RE_FOLD.finditer(text)]
    if not folds:
        return None
    blocks = re.findall(r"'([^']+)'", head.group("blocks"))
    # `genes` in a real summary.csv is DrugData.gene_order -- the loci found on
    # disk, in load order. The per-locus DNA blocks are built over exactly that
    # sequence, so their order reproduces it (and it is NOT alphabetical: e.g.
    # CAPREOMYCIN is rrs+rrl+tlyA).
    genes = []
    for b in blocks:
        if ":" in b:
            g = b.split(":", 1)[1]
            if g not in genes:
                genes.append(g)
    if not genes:
        # A `late_fusion` cell: blocks are per-modality ('dna', 'protein', ...),
        # so the log carries no gene names. Fall back to the `--loci` list the
        # manifest recorded for this job; if the job took the per-drug default,
        # leave `genes` empty rather than guess it.
        genes = list(loci)
    counts = RE_COUNTS.search(text)
    params = RE_PARAMS.search(text)
    aucs = [float(f["auc"]) for f in folds]
    prs = [float(f["auc_pr"]) for f in folds]
    return {
        "drug": head.group("drug"), "modalities": head.group("mods"),
        "genes": "+".join(genes), "arch": head.group("arch"),
        "n_valid": counts.group("n_valid") if counts else "",
        "n_R": counts.group("n_R") if counts else "",
        "n_S": counts.group("n_S") if counts else "",
        # Straight from the run's own "CV  AUC = m +/- s" line. Recomputing these
        # from the per-fold lines is worse on both counts: the per-fold AUCs are
        # already rounded to 4 dp so their mean can be off in the last digit, and
        # the run reports POPULATION std (ddof=0) while statistics.stdev is
        # ddof=1 -- a systematic factor of sqrt(5/4) = 1.118. Both were caught by
        # `--check` against full_run_v2.
        "cv_auc_mean": float(cv.group("mean")),
        "cv_auc_std": float(cv.group("std")),
        # No printed aggregate exists for AUC-PR, so this one IS a mean of 4-dp
        # per-fold values and can differ from a real run's in the 4th decimal.
        "cv_auc_pr_mean": statistics.mean(prs),
        "test_auc": float(test.group("auc")), "test_auc_pr": float(test.group("auc_pr")),
        "test_sens": float(test.group("sens")), "test_spec": float(test.group("spec")),
        "test_model_fold": test.group("fold"), "n_test": test.group("n_test"),
        "n_params": params.group("n").replace(",", "") if params else "",
        "n_blocks": len(blocks), "folds": folds,
    }


def r4(x):
    """Match `pd.DataFrame.round(4)`, which a real summary.csv goes through."""
    return "" if x == "" or x is None else f"{round(float(x), 4):g}"


def collect(run):
    jobs = manifest_jobs(run)
    logs, cells, missing = log_index(), defaultdict(dict), []
    meta = sacct({j[0] for j in jobs})
    for job_id, cell, drug, loci in jobs:
        path = logs.get(job_id)
        rec = parse(path, loci) if path else None
        if rec is None:
            state = meta.get(job_id, {}).get("state", "")
            missing.append({"cell": cell, "drug": drug, "job_id": job_id,
                            "slurm_state": state or "UNKNOWN",
                            "log": str(path.relative_to(PROJECT)) if path else ""})
            continue
        rec["job_id"], rec["log"] = job_id, str(path.relative_to(PROJECT))
        rec.update({"slurm_state": meta.get(job_id, {}).get("state", ""),
                    "job_elapsed": meta.get(job_id, {}).get("elapsed", ""),
                    "max_rss_kb": meta.get(job_id, {}).get("rss", "")})
        prev = cells[cell].get(drug)
        # A (cell, drug) pair can appear twice when a failed job was resubmitted
        # (KANAMYCIN all_modalities__late_fusion died on a CUDA OOM and was rerun).
        # Keep the later job id; a parsed result always beats an unparsed one.
        if prev is None or int(job_id) > int(prev["job_id"]):
            cells[cell][drug] = rec
    # Drop the superseded attempts from `missing`.
    missing = [m for m in missing if m["drug"] not in cells.get(m["cell"], {})]
    return cells, missing


def write(run_root, cells, missing, dry_run=False):
    for cell, byDrug in sorted(cells.items()):
        out = run_root / cell
        rows = [byDrug[d] for d in sorted(byDrug)]
        files = {
            "summary.csv": (SUMMARY_COLUMNS, [
                {**{k: r[k] for k in ("drug", "modalities", "genes",
                                      "n_valid", "n_R", "n_S")},
                 "cv_auc_mean": r4(r["cv_auc_mean"]), "cv_auc_std": r4(r["cv_auc_std"]),
                 "cv_auc_pr_mean": r4(r["cv_auc_pr_mean"]),
                 "test_auc": r4(r["test_auc"]), "test_auc_pr": r4(r["test_auc_pr"]),
                 "test_sens": r4(r["test_sens"]), "test_spec": r4(r["test_spec"]),
                 "seconds": ""} for r in rows]),
            "cv_folds.csv": (FOLD_COLUMNS, [
                {"drug": r["drug"], "modalities": r["modalities"], "fold": f["fold"],
                 "auc": f["auc"], "auc_pr": f["auc_pr"], "sens": f["sens"],
                 "spec": f["spec"], "n_val": f["n_val"], "best_epoch": f["best_epoch"]}
                for r in rows for f in r["folds"]]),
            "provenance.csv": (PROV_COLUMNS, [
                {k: r.get(k, "") for k in PROV_COLUMNS} for r in rows]),
        }
        if dry_run:
            continue
        out.mkdir(parents=True, exist_ok=True)
        for name, (cols, data) in files.items():
            with (out / name).open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                w.writerows(data)
    if not dry_run:
        with (run_root / "missing.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["cell", "drug", "job_id",
                                               "slurm_state", "log"])
            w.writeheader()
            w.writerows(sorted(missing, key=lambda m: (m["cell"], m["drug"])))


def check(run, cells):
    """Diff the reconstruction against real summary.csv files, where they exist."""
    root = PROJECT / "results/experiments" / run
    # Tolerance per field = the precision the LOG carries, not the precision the
    # real summary.csv carries. The run prints sens/spec to 3 dp, and
    # cv_auc_pr_mean has to be averaged from 4-dp per-fold values. Anything
    # outside these bounds is a parser bug, not a rounding artifact.
    TOL = {"test_sens": 5e-4, "test_spec": 5e-4, "cv_auc_pr_mean": 1e-4}
    fields = ["modalities", "genes", "n_valid", "n_R", "n_S", "cv_auc_mean",
              "cv_auc_std", "cv_auc_pr_mean", "test_auc", "test_auc_pr",
              "test_sens", "test_spec"]
    n_cells = n_rows = n_bad = n_soft = 0
    for cell, byDrug in sorted(cells.items()):
        real_path = root / cell / "summary.csv"
        if not real_path.is_file():
            continue
        n_cells += 1
        with real_path.open(newline="") as fh:
            real = {r["drug"]: r for r in csv.DictReader(fh)}
        for drug, rec in sorted(byDrug.items()):
            if drug not in real:
                print(f"  MISSING in real: {cell}/{drug}")
                n_bad += 1
                continue
            n_rows += 1
            for f in fields:
                want_raw = real[drug][f]
                if not f.startswith(("cv_", "test_")):
                    if f == "genes" and not rec[f]:
                        n_soft += 1          # not in the log, not in the manifest
                        continue
                    if str(rec[f]) != want_raw:
                        print(f"  DIFF {cell}/{drug}.{f}: "
                              f"reconstructed={rec[f]!r} real={want_raw!r}")
                        n_bad += 1
                    continue
                got, want = float(r4(rec[f])), float(want_raw)
                delta = abs(got - want)
                tol = TOL.get(f, 0.0)
                if delta > tol + 1e-12:
                    print(f"  DIFF {cell}/{drug}.{f}: reconstructed={got} "
                          f"real={want} (delta {delta:.5f} > tol {tol})")
                    n_bad += 1
                elif delta:
                    n_soft += 1
    print(f"\ncheck: {n_rows} rows across {n_cells} cells, "
          f"{n_bad} mismatches beyond log precision, "
          f"{n_soft} within it (sens/spec 3 dp, auc_pr mean of 4-dp folds)")
    return n_bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="alllocus_run")
    ap.add_argument("--check", action="store_true",
                    help="diff against the run's real summary.csv files, write nothing")
    a = ap.parse_args()

    cells, missing = collect(a.run)
    n = sum(len(v) for v in cells.values())
    print(f"{a.run}: recovered {n} results across {len(cells)} cells; "
          f"{len(missing)} jobs produced none")
    for cell in sorted(cells):
        print(f"  {cell:34s} {len(cells[cell]):2d}/11")

    if a.check:
        sys.exit(0 if check(a.run, cells) else 1)

    root = PROJECT / "results/experiments" / a.run
    write(root, cells, missing)
    print(f"\nwrote {len(cells)} cell folders + missing.csv under {root}")
    for m in sorted(missing, key=lambda m: (m["cell"], m["drug"])):
        print(f"  no result: {m['cell']:34s} {m['drug']:14s} {m['slurm_state']}")


if __name__ == "__main__":
    main()
