#!/usr/bin/env python3
"""
Generate and submit one sbatch job per (experiment x drug) combination.

Each job runs run_experiment.py for a SINGLE drug under one modality/encoder
configuration, so every run gets its own SLURM allocation (own --mem, own GPU)
and a single-drug memory footprint. That is what keeps the runs from crashing
each other: the all-drugs-in-one-process host-RAM growth that OOM-killed the
interactive `--drugs all` run can't happen when each drug is its own job.

Modeled on multimodal-modality-conflict/scripts/sbatch_all_models.py — the
model x dataset grid there becomes experiment x drug here.

Examples:
    # dry-run: print the scripts that would be submitted, submit nothing
    python scripts/sbatch_all_runs.py --dry_run

    # submit the DNA-only baseline for every drug
    python scripts/sbatch_all_runs.py --experiments dna --drugs all

    # submit two experiments for three drugs, more memory, 6h limit
    python scripts/sbatch_all_runs.py --experiments dna dna_biophysical \
        --drugs ISONIAZID RIFAMPICIN PYRAZINAMIDE --mem 48G --time 06:00:00
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime

# Project checkout that holds run_experiment.py (resolved from this file's
# location: scripts/ -> project root), so jobs cd to whatever checkout you
# launched the generator from. Override with --project-dir.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDA_ENV = "abr_env"

# All drugs (the DRUG_TO_LOCI keys). 'all' on the CLI expands to this.
ALL_DRUGS = [
    "ISONIAZID", "RIFAMPICIN", "ETHAMBUTOL", "PYRAZINAMIDE", "STREPTOMYCIN",
    "KANAMYCIN", "AMIKACIN", "CAPREOMYCIN", "LEVOFLOXACIN", "MOXIFLOXACIN",
    "ETHIONAMIDE",
]

# Each experiment is one modality/encoder configuration = one run-name folder
# under results/experiments/. Its per-drug jobs all write there (each writes its
# own {DRUG}__{tag}.json, so they don't collide). Edit / add freely.
#   modalities : subset of dna/protein/biophysical/regulatory (or ["all"])
#   encoders   : optional {modality: cnn|transformer}
#   loci       : optional explicit gene loci override
EXPERIMENTS = {
    # "dna":                 {"modalities": ["dna"]},
    "dna_biophysical":     {"modalities": ["dna", "biophysical"]},
    "dna_protein":         {"modalities": ["dna", "protein"]},
    "all_modalities":      {"modalities": ["dna", "regulatory"]},
    "protein_transformer": {"modalities": ["dna", "protein", "biophysical", "regulatory"]},
}


def run_experiment_command(exp_name, cfg, drug, args):
    """The `python run_experiment.py ...` line for one (experiment, drug) job."""
    parts = [
        "python", "run_experiment.py", "--real", "--device", "cuda",
        "--modalities", *cfg["modalities"],
        "--drugs", drug,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--n-splits", str(args.n_splits),
        "--seed", str(args.seed),
        "--run-name", exp_name,          # per-experiment folder, shared across its drugs
    ]
    if cfg.get("encoders"):
        parts += ["--encoders", *[f"{m}={e}" for m, e in cfg["encoders"].items()]]
    if cfg.get("loci"):
        parts += ["--loci", *cfg["loci"]]
    if args.tb:
        parts += ["--tb"]
    return " ".join(shlex.quote(p) for p in parts)


def create_sbatch_script(exp_name, cfg, drug, args, job_id):
    """Build the sbatch script text + the log path for one job."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"abr_{exp_name}_{drug}"
    log_path = os.path.join(args.logs_dir, f"slurm-{exp_name}_{drug}_{timestamp}_job{job_id}-%j.out")
    command = run_experiment_command(exp_name, cfg, drug, args)

    constraint = f"#SBATCH --constraint={args.constraint}\n" if args.constraint else ""
    script = f"""#!/bin/bash
#SBATCH -c {args.cpus}                # cores
#SBATCH --mem={args.mem}             # host memory (the knob that avoids the OOM kill)
#SBATCH -p {args.partition}          # partition
{constraint}#SBATCH --gpus={args.gpus}            # GPUs
#SBATCH -t {args.time}               # time limit (H:MM:SS)
#SBATCH -o {log_path}
#SBATCH --job-name={job_name}

# --- {exp_name} on {drug} ---
# modalities: {cfg['modalities']}  encoders: {cfg.get('encoders', {})}
echo "Job: {exp_name} on {drug}  (host $(hostname))"

source ~/.bashrc
conda activate {CONDA_ENV}
cd {shlex.quote(args.project_dir)}
mkdir -p {shlex.quote(args.logs_dir)}

{command}

echo "Done: {exp_name} on {drug}"
"""
    return script, log_path


def submit_job(script_text, job_id, keep_scripts_dir=None):
    """Write the script and submit with sbatch; return the SLURM job id."""
    fname = os.path.join(keep_scripts_dir or ".", f"_sbatch_{job_id}.sh")
    with open(fname, "w") as fh:
        fh.write(script_text)
    try:
        out = subprocess.run(["sbatch", fname], capture_output=True, text=True, check=True)
        return out.stdout.strip().split()[-1]  # "Submitted batch job 12345"
    except subprocess.CalledProcessError as e:
        print(f"  ERROR submitting job {job_id}: {e.stderr.strip()}")
        return None
    finally:
        if not keep_scripts_dir:
            os.remove(fname)


def _resolve(requested, universe, kind):
    if not requested or any(x.lower() == "all" for x in requested):
        return list(universe)
    norm = [x.upper() if kind == "drug" else x for x in requested]
    unknown = [x for x in norm if x not in universe]
    if unknown:
        sys.exit(f"Unknown {kind}(s) {unknown}; choose from {list(universe)} or 'all'")
    return norm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", nargs="+", default=None,
                   help=f"which experiments (default: all). Options: {list(EXPERIMENTS)} or 'all'")
    p.add_argument("--drugs", nargs="+", default=None,
                   help="which drugs (default: all). Drug names or 'all'")
    # run_experiment.py knobs
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tb", action="store_true", help="pass --tb to run_experiment.py")
    # SLURM resources
    p.add_argument("--mem", default="48G", help="host memory per job (default: 48G)")
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--gpus", default="1", help="e.g. '1' or 'a100:1' (default: 1)")
    p.add_argument("--partition", default="gpu")
    p.add_argument("--constraint", default=None, help="e.g. vram48 (default: none)")
    p.add_argument("--time", default="12:00:00", help="SLURM time limit H:MM:SS (default: 12:00:00)")
    # bookkeeping
    p.add_argument("--project-dir", default=PROJECT_DIR, help=f"checkout to cd into (default: {PROJECT_DIR})")
    p.add_argument("--logs-dir", default=None, help="sbatch log dir (default: <project>/slurm_logs)")
    p.add_argument("--delay", type=float, default=1.0, help="seconds between submissions")
    p.add_argument("--dry_run", action="store_true", help="print scripts, submit nothing")
    return p.parse_args()


def main():
    args = parse_args()
    args.logs_dir = args.logs_dir or os.path.join(args.project_dir, "slurm_logs")
    experiments = _resolve(args.experiments, EXPERIMENTS, "experiment")
    drugs = _resolve(args.drugs, ALL_DRUGS, "drug")
    os.makedirs(args.logs_dir, exist_ok=True)

    total = len(experiments) * len(drugs)
    print(f"Experiments: {experiments}")
    print(f"Drugs: {drugs}")
    print(f"Total jobs: {total}  | mem={args.mem} gpus={args.gpus} time={args.time} "
          f"epochs={args.epochs} bs={args.batch_size}")
    print(f"Project: {args.project_dir}\nLogs: {args.logs_dir}")
    if args.dry_run:
        print("DRY RUN — nothing will be submitted.\n")

    submitted, job_id = [], 0
    for exp_name in experiments:
        cfg = EXPERIMENTS[exp_name]
        for drug in drugs:
            job_id += 1
            script, log_path = create_sbatch_script(exp_name, cfg, drug, args, job_id)
            header = f"[{job_id}/{total}] {exp_name} on {drug}"
            if args.dry_run:
                print(f"===== {header} =====\n{script}")
                continue
            sid = submit_job(script, job_id)
            if sid:
                submitted.append({"job_id": sid, "experiment": exp_name, "drug": drug,
                                  "run_name": exp_name, "log": log_path,
                                  "command": run_experiment_command(exp_name, cfg, drug, args)})
                print(f"  {header}: submitted {sid}")
            else:
                print(f"  {header}: FAILED to submit")
            if job_id < total:
                time.sleep(args.delay)

    if not args.dry_run:
        print(f"\nSubmitted {len(submitted)}/{total} jobs.")
        if submitted:
            track = os.path.join(args.logs_dir,
                                 "submitted_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
            with open(track, "w") as fh:
                json.dump({"invocation": " ".join(map(shlex.quote, sys.argv)),
                           "submitted_jobs": submitted}, fh, indent=2)
            print(f"Tracking: {track}")
            print("Monitor with:  squeue -u $USER   |   cancel with:  scancel <job_id>")


if __name__ == "__main__":
    main()
