"""
Submit GenomeFusion runs to SLURM, from a frozen copy of the code.

    cd biophysical-fusion
    python -m genomefusion.sbatch --arms canary --dry-run     # print, submit nothing
    python -m genomefusion.sbatch --arms canary               # PZA, whole genome, fold 0
    python -m genomefusion.sbatch --arms gf19 gf_genome gf_minus19

Arms (every arm is 11 jobs, one per drug, on the lineage split):
    canary       PYRAZINAMIDE, whole genome, fold 0 only -- measures an epoch
                 before anything bigger is queued
    gf19         the LocusFusion v3/v5 loci. A PARITY arm, not a result: it says
                 whether the sparse code path reproduces v5 on the same loci
    gf_genome    every locus the partition keeps
    gf_minus19   every locus except the 19 -- is there signal outside them?

Not here yet: the multi-drug model, the `flat` ablation (no locus hierarchy),
and the `pop` arm (homoplasy needs a phylogeny of the cohort, which we do not
have). See docs/genomefusion/.

Code runs from <workspace>/gf_snapshot/<stamp>/, with BASE_COMMIT and
working_tree.patch alongside, so editing the repo cannot reach a queued job.
Results go to results/experiments/genomefusion/<arm>/, logs to
results/experiments/genomefusion/logs/, and every submission writes a manifest.
"""
import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
DATA = WORKSPACE / "data_snapshot"
RUN_ROOT = PROJECT / "results" / "experiments" / "genomefusion"
LINEAGE_SPLITS = PROJECT / "results" / "experiments" / "lineage_split" / "splits"
TOKENS = WORKSPACE / "genomefusion_tokens" / "tokens.npz"
CONDA_ENV = "abr_env"
DRUGS = ["AMIKACIN", "CAPREOMYCIN", "ETHAMBUTOL", "ETHIONAMIDE", "ISONIAZID", "KANAMYCIN",
         "LEVOFLOXACIN", "MOXIFLOXACIN", "PYRAZINAMIDE", "RIFAMPICIN", "STREPTOMYCIN"]
# A job runs from a frozen snapshot, whose parent holds none of the shared data,
# so every shared path is passed explicitly and points at the live workspace.
ENV = {"ABR_GENOTYPE_DIR": DATA / "genomic_data" / "aligned",
       "ABR_PHENOTYPE_CSV": DATA / "phenotype_data" / "master_resistance_table.csv",
       "ABR_GENBANK": DATA / "genbank" / "genome.gbff",
       "ABR_WORKSPACE": WORKSPACE,
       "ABR_GF_REFS": WORKSPACE / "genomefusion_refs",
       "ABR_GF_CACHE": WORKSPACE / "genomefusion_cache"}

RESOURCES = {   # kind -> (partition, cpus, mem, time, gpus, constraint)
    "gpu": ("gpu", 4, "64G", "48:00:00", 1, "vram23"),
    "canary": ("gpu", 4, "48G", "4:00:00", 1, "vram23"),
}

ARMS = {"gf19": "19", "gf_genome": "genome", "gf_minus19": "minus19"}


def arm_jobs(arm, batch_size):
    bs = ["--batch-size", str(batch_size)]
    if arm == "canary":
        return [("gf_canary_PYRAZINAMIDE", "canary",
                 ["--drugs", "PYRAZINAMIDE", "--locus-set", "genome", "--folds", "0", *bs],
                 "canary")]
    if arm in ARMS:
        return [(f"gf_{arm}_{d}", arm,
                 ["--drugs", d, "--locus-set", ARMS[arm], *bs], "gpu") for d in DRUGS]
    raise SystemExit(f"unknown arm {arm!r}; pick from canary, {', '.join(ARMS)}")


def make_snapshot(stamp, dry):
    snap = WORKSPACE / "gf_snapshot" / stamp
    cmds = [
        ["mkdir", "-p", str(snap / "Big-TB-benchmark" / "dna-tasks" / "SD-CNN" / "model_training")],
        ["rsync", "-a", "--exclude", "results", "--exclude", "notebooks", "--exclude",
         "slurm_logs", "--exclude", "__pycache__", "--exclude", "diagrams",
         f"{PROJECT}/", f"{snap / 'biophysical-fusion'}/"],
        ["rsync", "-a",
         f"{WORKSPACE}/Big-TB-benchmark/dna-tasks/SD-CNN/model_training/parameters",
         f"{snap / 'Big-TB-benchmark' / 'dna-tasks' / 'SD-CNN' / 'model_training'}/"],
    ]
    if dry:
        for c in cmds:
            print("  [snapshot] " + " ".join(shlex.quote(x) for x in c))
        return snap
    for c in cmds:
        subprocess.run(c, check=True)
    for name, args in (("BASE_COMMIT", ["rev-parse", "HEAD"]),
                       ("working_tree.patch", ["diff", "HEAD", "--", "biophysical-fusion"]),
                       ("status.txt", ["status", "--short", "biophysical-fusion"])):
        r = subprocess.run(["git", "-C", str(WORKSPACE), *args],
                           capture_output=True, text=True, check=True)
        (snap / name).write_text(r.stdout)
    subprocess.run(["chmod", "-R", "a-w", str(snap / "biophysical-fusion")], check=True)
    return snap


def job_script(name, sub, targs, kind, snap, tokens, weights):
    part, cpus, mem, tlim, gpus, constraint = RESOURCES[kind]
    logs, out = RUN_ROOT / "logs", RUN_ROOT / sub
    lines = ["#!/bin/bash",
             f"#SBATCH --job-name={name}",
             f"#SBATCH --output={logs}/%x-%j.out",
             f"#SBATCH --error={logs}/%x-%j.err",
             f"#SBATCH --partition={part}",
             f"#SBATCH --time={tlim}",
             f"#SBATCH --cpus-per-task={cpus}",
             f"#SBATCH --mem={mem}"]
    if gpus:
        lines.append(f"#SBATCH --gpus={gpus}")
    if constraint:
        lines.append(f"#SBATCH --constraint={constraint}")
    cmd = ["python", "-u", "-m", "genomefusion.train", *targs,
           "--out", str(out), "--split-dir", str(LINEAGE_SPLITS), "--tokens", str(tokens)]
    if weights:
        cmd += ["--weights-dir", str(weights)]
    lines += ["set -euo pipefail",
              'echo "host $(hostname)  start $(date)"',
              "module list 2>&1 || true",
              "source ~/.bashrc",
              f"conda activate {CONDA_ENV}",
              "which python && python --version",
              "nvidia-smi || true",
              'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
              'export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
              *[f"export {k}={shlex.quote(str(v))}" for k, v in ENV.items()],
              f"cd {shlex.quote(str(snap / 'biophysical-fusion'))}",
              " ".join(shlex.quote(c) for c in cmd),
              'echo "done $(date)"']
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--dependency", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tokens", default=str(TOKENS))
    ap.add_argument("--weights-dir", default=str(WORKSPACE / "model_weights_gf"))
    args = ap.parse_args()

    jobs = [j for arm in args.arms for j in arm_jobs(arm, args.batch_size)]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    snap = Path(args.snapshot) if args.snapshot else make_snapshot(stamp, args.dry_run)
    if args.snapshot and not (snap / "biophysical-fusion" / "genomefusion").exists():
        raise SystemExit(f"{snap} is not a genomefusion snapshot")
    scripts = RUN_ROOT / "logs" / "scripts" / stamp
    if not args.dry_run:
        scripts.mkdir(parents=True, exist_ok=True)
    manifest = {"stamp": stamp, "snapshot": str(snap), "arms": args.arms,
                "dependency": args.dependency, "batch_size": args.batch_size,
                "tokens": args.tokens, "jobs": []}
    for name, sub, targs, kind in jobs:
        text = job_script(name, sub, targs, kind, snap, args.tokens, args.weights_dir)
        if args.dry_run:
            print(f"### {name}  [{kind}: {RESOURCES[kind]}]")
            print("    " + text.splitlines()[-2])
            continue
        path = scripts / f"{name}.sh"
        path.write_text(text)
        dep = [f"--dependency={args.dependency}"] if args.dependency else []
        res = subprocess.run(["sbatch", *dep, str(path)], capture_output=True, text=True)
        jid = res.stdout.strip().split()[-1] if res.returncode == 0 else None
        print(f"{name}: {'job ' + jid if jid else 'FAILED ' + res.stderr.strip()}")
        manifest["jobs"].append({"name": name, "results": str(RUN_ROOT / sub), "job_id": jid,
                                 "script": str(path), "kind": kind})
    if args.dry_run:
        print(f"\n{len(jobs)} jobs (dry run; nothing written, nothing submitted)")
        return
    (RUN_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "logs" / f"manifest_{stamp}.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
