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
# The grid: every architecture x every modality set. Names are
# "{modalities}__{arch}", e.g. "dna_regulatory__cisfusion", and each becomes one
# run folder. MODALITY_SETS is DNA alone, DNA paired with each other modality,
# and everything at once — the ablation ladder for "does modality X add
# anything on top of DNA".
MODALITY_SETS = {
    "dna":              ["dna"],
    "dna_protein":      ["dna", "protein"],
    "dna_biophysical":  ["dna", "biophysical"],
    "dna_regulatory":   ["dna", "regulatory"],
    "all_modalities":   ["dna", "protein", "biophysical", "regulatory"],
}
GRID_ARCHS = ("late_fusion", "mdcnn", "setfusion", "cisfusion")


def _build_grid():
    return {f"{mods}__{arch}": {"modalities": list(names), "arch": arch}
            for mods, names in MODALITY_SETS.items() for arch in GRID_ARCHS}


EXPERIMENTS = _build_grid()


def _training_knobs(args):
    """Flags shared by both entry points. Only emitted when they differ from the
    script default, so an unchanged grid produces byte-identical commands to the
    ones already in the submission manifests — and every default here is the
    value that produced results/experiments/full_run."""
    parts = []
    if args.monitor != "auc":
        parts += ["--monitor", args.monitor]
    if args.patience != 15:
        parts += ["--patience", str(args.patience)]
    if args.min_epochs:
        parts += ["--min-epochs", str(args.min_epochs)]
    if args.min_delta != 1e-4:
        parts += ["--min-delta", str(args.min_delta)]
    if args.all_regulatory:
        parts += ["--all-regulatory"]
    if args.lr is not None:
        parts += ["--lr", str(args.lr)]
    if args.weight_decay:
        parts += ["--weight-decay", str(args.weight_decay)]
    if args.dropout:
        parts += ["--dropout", str(args.dropout)]
    if args.hidden != 256:
        parts += ["--hidden", str(args.hidden)]
    if args.per_drug_hidden:
        parts += ["--per-drug-hidden", str(args.per_drug_hidden)]
    if args.mdcnn_trunk_per_modality:
        parts += ["--mdcnn-trunk-per-modality"]
    if args.delta:
        parts += ["--delta"]
    if args.lr_schedule != "none":
        parts += ["--lr-schedule", args.lr_schedule]
    if args.warmup_epochs:
        parts += ["--warmup-epochs", str(args.warmup_epochs)]
    # setfusion capacity: emitted only when set, so an unchanged grid still
    # produces the exact command strings recorded in the full_run manifests
    for flag, val in (("--d-model", args.d_model), ("--nhead", args.nhead),
                      ("--fusion-layers", args.fusion_layers),
                      ("--dim-ff", args.dim_ff),
                      ("--fusion-dropout", args.fusion_dropout),
                      ("--enc-width", args.enc_width),
                      ("--enc-out-channels", args.enc_out_channels),
                      ("--enc-depth", args.enc_depth),
                      ("--token-norm", args.token_norm),
                      ("--enc-bins", args.enc_bins)):
        if val is not None:
            parts += [flag, str(val)]
    if args.save_weights != "best":
        parts += ["--save-weights", args.save_weights]
    if args.weights_dir:
        parts += ["--weights-dir", args.weights_dir]
    return parts


def run_experiment_command(exp_name, cfg, drug, args):
    """The `python scripts/run_experiment.py ...` line for one (experiment, drug) job."""
    parts = [
        "python", "scripts/run_experiment.py", "--real", "--device", "cuda",
        "--modalities", *cfg["modalities"],
        "--drugs", drug,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--n-splits", str(args.n_splits),
        "--seed", str(args.seed),
        "--run-name", args.run_prefix + exp_name,   # per-experiment folder (opt. prefix)
    ]
    arch = cfg.get("arch", args.arch)
    if arch != "late_fusion":
        parts += ["--arch", arch]
    if cfg.get("extra_loci", args.extra_loci):
        parts += ["--extra-loci"]
    if args.per_locus_branches:
        parts += ["--per-locus-branches"]
    if cfg.get("encoders"):
        parts += ["--encoders", *[f"{m}={e}" for m, e in cfg["encoders"].items()]]
    loci = cfg.get("loci") or args.loci
    if loci:
        parts += ["--loci", *loci]
    parts += _training_knobs(args)
    if args.tb:
        parts += ["--tb"]
    return " ".join(shlex.quote(p) for p in parts)


def run_multidrug_command(exp_name, cfg, args):
    """The `python scripts/run_multidrug.py ...` line for one experiment — one
    job total, not one per drug (the model predicts every drug at once)."""
    parts = [
        "python", "-u", "scripts/run_multidrug.py", "--real", "--device", "cuda",
        "--modalities", *cfg["modalities"],
        "--drugs", "all",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--n-splits", str(args.n_splits),
        "--seed", str(args.seed),
        "--run-name", args.run_prefix + "multidrug_" + exp_name,
    ]
    arch = cfg.get("arch", args.arch)
    if arch != "late_fusion":
        parts += ["--arch", arch]
    loci = cfg.get("loci") or args.loci
    if loci:
        parts += ["--loci", *loci]
    parts += _training_knobs(args)
    if args.monitor_min_n:                 # joint-only: no macro to trim single-drug
        parts += ["--monitor-min-n", str(args.monitor_min_n)]
    if args.tb:
        parts += ["--tb"]
    return " ".join(shlex.quote(p) for p in parts)


def create_sbatch_script(exp_name, cfg, drug, args, job_id):
    """Build the sbatch script text + the log path for one job."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    multidrug = drug is None
    tag = "multidrug" if multidrug else drug
    job_name = f"abr_{exp_name}_{tag}"
    log_path = os.path.join(args.logs_dir, f"slurm-{exp_name}_{tag}_{timestamp}_job{job_id}-%j.out")
    command = (run_multidrug_command(exp_name, cfg, args) if multidrug
               else run_experiment_command(exp_name, cfg, drug, args))

    constraint = f"#SBATCH --constraint={args.constraint}\n" if args.constraint else ""
    script = f"""#!/bin/bash
#SBATCH -c {args.cpus}                # cores
#SBATCH --mem={args.mem}             # host memory (the knob that avoids the OOM kill)
#SBATCH -p {args.partition}          # partition
{constraint}#SBATCH --gpus={args.gpus}            # GPUs
#SBATCH -t {args.time}               # time limit (H:MM:SS)
#SBATCH -o {log_path}
#SBATCH --job-name={job_name}

# --- {exp_name} on {tag} ---
# modalities: {cfg['modalities']}  encoders: {cfg.get('encoders', {})}  arch: {cfg.get('arch', args.arch)}
echo "Job: {exp_name} on {tag}  (host $(hostname))"

source ~/.bashrc
conda activate {CONDA_ENV}
cd {shlex.quote(args.project_dir)}
mkdir -p {shlex.quote(args.logs_dir)}

# Caching allocator fragmentation is what OOM-killed the wide configs on 11GB
# cards: one job died with 4.5 GiB "reserved but unallocated" while asking for
# 1.21 GiB. The branch-per-locus nets allocate many differently-sized activation
# blocks per step, which is exactly the pattern expandable segments fix.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

{command}

echo "Done: {exp_name} on {tag}"
"""
    return script, log_path


def command_of(exp_name, cfg, drug, args):
    return (run_multidrug_command(exp_name, cfg, args) if drug is None
            else run_experiment_command(exp_name, cfg, drug, args))


def run_name_of(exp_name, drug, args):
    """The --run-name the job actually receives, i.e. its results folder.

    Multi-drug jobs get a 'multidrug_' infix (see run_multidrug_command). The
    manifest used to record the single-drug form for BOTH, so every joint job's
    tracking entry pointed at a folder it never wrote to — which silently broke
    any tooling that joined manifests to results by run_name."""
    return args.run_prefix + ("" if drug is not None else "multidrug_") + exp_name


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
    p.add_argument("--multidrug", action="store_true",
                   help="submit ONE run_multidrug.py job per experiment (all drugs "
                        "in one model) instead of one run_experiment.py job per "
                        "(experiment x drug)")
    # run_experiment.py knobs
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-prefix", default="", help="prefix for run-name folders, "
                   "e.g. 'permod_' -> results/experiments/permod_dna (default: none)")
    p.add_argument("--arch", default="late_fusion",
                   help="network topology passed to run_experiment.py: "
                        "'late_fusion' (default) or 'mdcnn' (BIG-TB's own; implies "
                        "per-locus branches). An experiment's own 'arch' key wins.")
    p.add_argument("--extra-loci", action="store_true",
                   help="pass --extra-loci to run_experiment.py (adds fabG1 for "
                        "INH/ETO). An experiment's own 'extra_loci' key wins.")
    p.add_argument("--per-locus-branches", action="store_true",
                   help="pass --per-locus-branches to run_experiment.py (default: "
                        "one branch per modality)")
    p.add_argument("--loci", nargs="+", default=None,
                   help="explicit gene loci for EVERY job, e.g. the 19 curated "
                        "loci the multi-drug runs use, so single-drug results "
                        "become locus-matched to them. An experiment's own 'loci' "
                        "key wins. Off by default (each drug uses DRUG_TO_LOCI).")
    p.add_argument("--monitor", default="auc", choices=["auc", "loss"],
                   help="early-stopping metric (default: auc)")
    p.add_argument("--patience", type=int, default=15,
                   help="early-stopping patience (default: 15)")
    p.add_argument("--min-epochs", type=int, default=0,
                   help="early-stopping warmup passed to the run scripts "
                        "(default: 0 = off; setfusion needs ~40)")
    p.add_argument("--min-delta", type=float, default=1e-4,
                   help="early-stopping min improvement (default: 1e-4)")
    p.add_argument("--monitor-min-n", type=int, default=0,
                   help="MULTI-DRUG ONLY: drop drugs with fewer than this many "
                        "labelled train isolates from the early-stopping macro "
                        "(default: 0 = keep all). 500 excludes LEVOFLOXACIN, "
                        "whose ~15 resistant val isolates dominate the monitor's "
                        "variance. They are still trained on and reported.")
    # --- data / optimizer / capacity. Every default below is the full_run
    # value, so an unchanged invocation reproduces that grid exactly.
    p.add_argument("--all-regulatory", action="store_true",
                   help="keep the FULL WHO promoter set instead of intersecting "
                        "it with the coding loci. full_run had this OFF, which "
                        "cut ISONIAZID 14 regions -> 2, PYRAZINAMIDE 8 -> 1 and "
                        "KANAMYCIN 7 -> 1 (losing the eis promoter), while all 48 "
                        "windows sit on disk with full isolate coverage.")
    p.add_argument("--lr", type=float, default=None,
                   help="Adam learning rate (default: exp(-9) ~ 1.2e-4, BIG-TB's)")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="weight decay; > 0 switches Adam -> AdamW (default: 0)")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="dense-head dropout (default: 0 — full_run had none)")
    p.add_argument("--hidden", type=int, default=256,
                   help="dense-head width (default: 256)")
    p.add_argument("--per-drug-hidden", type=int, default=0,
                   help="MULTI-DRUG: per-drug 'hidden -> k -> 1' branch off the "
                        "shared trunk (default: 0 = one shared output linear)")
    # --- setfusion capacity, forwarded verbatim to the runners (which reject
    # them unless --arch setfusion). None = leave the runner default alone.
    # These are the axes results/experiments/setfusion_scaling sweeps.
    p.add_argument("--d-model", type=int, default=None,
                   help="setfusion token width (default: 128)")
    p.add_argument("--nhead", type=int, default=None,
                   help="setfusion attention heads (default: 4)")
    p.add_argument("--fusion-layers", type=int, default=None,
                   help="setfusion transformer layers (default: 2)")
    p.add_argument("--dim-ff", type=int, default=None,
                   help="setfusion transformer feed-forward width (default: 256)")
    p.add_argument("--fusion-dropout", type=float, default=None,
                   help="setfusion transformer/attention dropout (default: 0.1)")
    p.add_argument("--enc-width", type=int, default=None,
                   help="setfusion encoder stem/conv1 channels (default: 64)")
    p.add_argument("--enc-out-channels", type=int, default=None,
                   help="setfusion encoder conv2/conv3 channels (default: 32)")
    p.add_argument("--enc-depth", type=int, default=None,
                   help="setfusion encoder conv stages after conv1 (default: 1)")
    p.add_argument("--token-norm", default=None,
                   help="pass --token-norm to run_experiment.py (setfusion only): "
                        "'keyed' standardises each token across the batch with "
                        "per-(modality, locus) statistics.")
    p.add_argument("--delta", action="store_true",
                   help="pass --delta to run_experiment.py: reference-difference "
                        "input encoding (zero every column matching H37Rv).")
    p.add_argument("--enc-bins", type=int, default=None,
                   help="setfusion pooled segments per block (default: 4)")
    p.add_argument("--lr-schedule", default="none", choices=["none", "cosine"],
                   help="per-epoch LR schedule (default: none = flat, as in "
                        "full_run). 'cosine' pairs with --warmup-epochs.")
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="linear LR warmup for --lr-schedule cosine (default: 0)")
    p.add_argument("--mdcnn-trunk-per-modality", action="store_true",
                   help="--arch mdcnn: one trunk per (modality, channels) rather "
                        "than per channel count, so promoter windows are not "
                        "padded to the longest CDS inside the DNA trunk")
    p.add_argument("--save-weights", default="best",
                   choices=["best", "all", "none"],
                   help="which CV fold weights each job persists, alongside a "
                        "config that fully rebuilds the model (default: best = "
                        "the fold the TEST numbers come from)")
    p.add_argument("--weights-dir", default=None,
                   help="override the shared weights root (default: "
                        "bigtb_ref.MODEL_WEIGHTS_DIR)")
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
    # multi-drug: one job per experiment, so the drug loop runs once with None
    drugs = [None] if args.multidrug else _resolve(args.drugs, ALL_DRUGS, "drug")
    os.makedirs(args.logs_dir, exist_ok=True)

    total = len(experiments) * len(drugs)
    print(f"Experiments: {experiments}")
    print("Mode: multi-drug (one job per experiment)" if args.multidrug
          else f"Drugs: {drugs}")
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
            header = f"[{job_id}/{total}] {exp_name} on {drug or 'multidrug'}"
            if args.dry_run:
                print(f"===== {header} =====\n{script}")
                continue
            sid = submit_job(script, job_id)
            if sid:
                submitted.append({"job_id": sid, "experiment": exp_name,
                                  "drug": drug or "ALL(multidrug)",
                                  "run_name": run_name_of(exp_name, drug, args),
                                  "log": log_path, "command": command_of(exp_name, cfg, drug, args)})
                print(f"  {header}: submitted {sid}")
            else:
                print(f"  {header}: FAILED to submit")
            if job_id < total:
                time.sleep(args.delay)

    if not args.dry_run:
        print(f"\nSubmitted {len(submitted)}/{total} jobs.")
        if submitted:
            # The manifest is the record of the exact command behind every job —
            # full_run/README.md cites it as such — so it must never be
            # overwritten. A second-resolution name silently clobbered four of
            # six manifests when joint_convergence/submit.sh called this script
            # in a loop (2026-08-06): the jobs ran fine, but their provenance was
            # gone. Microseconds plus a collision guard, so back-to-back
            # invocations each keep their own file.
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            # Manifests live in their own folder (2026-08-13): they are the only
            # job_id -> run mapping there is, and keeping them out of the .out
            # stream means log housekeeping can never sweep them up by accident.
            manifest_dir = os.path.join(args.logs_dir, "manifests")
            os.makedirs(manifest_dir, exist_ok=True)
            track = os.path.join(manifest_dir, f"submitted_{stamp}.json")
            n = 0
            while os.path.exists(track):
                n += 1
                track = os.path.join(manifest_dir, f"submitted_{stamp}_{n}.json")
            with open(track, "w") as fh:
                json.dump({"invocation": " ".join(map(shlex.quote, sys.argv)),
                           "submitted_jobs": submitted}, fh, indent=2)
            print(f"Tracking: {track}")
            print("Monitor with:  squeue -u $USER   |   cancel with:  scancel <job_id>")


if __name__ == "__main__":
    main()
