#!/usr/bin/env python3
"""
Run the experiment sweep.

Everything that decides WHAT runs is a list at the top of this file — comment a
line out to drop it. The cross product of DRUGS x MODALITY_SETS x ARCHITECTURES
x SCOPES is what gets submitted.

    python scripts/sweep.py --dry-run     # print the plan, submit nothing
    python scripts/sweep.py               # cancel, wipe, submit

Results land in results/experiments/{RUN_PREFIX}{modality_set}__{arch}/ for
single-drug runs and .../multidrug_{modality_set}__{arch}/ for joint ones.
"""

# =============================================================================
# WHAT TO RUN — comment out any line to leave it out of the sweep
# =============================================================================

DRUGS = [
    "ISONIAZID",
    "RIFAMPICIN",
    "ETHAMBUTOL",
    "PYRAZINAMIDE",
    "STREPTOMYCIN",
    "KANAMYCIN",
    "AMIKACIN",
    "CAPREOMYCIN",
    "LEVOFLOXACIN",
    "MOXIFLOXACIN",
    "ETHIONAMIDE",
]

MODALITY_SETS = {
    "dna":              ["dna"],
    "dna_protein":      ["dna", "protein"],
    "dna_biophysical":  ["dna", "biophysical"],
    "dna_regulatory":   ["dna", "regulatory"],
    "all_modalities":   ["dna", "protein", "biophysical", "regulatory"],
}

ARCHITECTURES = [
    "late_fusion",      # our per-block encoder net
    "mdcnn",            # BIG-TB's own locus-as-channel topology
    "setfusion",        # shared per-modality encoders, locus-keyed set fusion
    "cisfusion",        # promoter (+) CDS cis-units
]

SCOPES = [
    "single",           # one model per drug  (len(DRUGS) jobs per cell)
    "multi",            # one model, all drugs (1 job per cell)
]

# =============================================================================
# HOW TO RUN
# =============================================================================

EPOCHS = 150
BATCH_SIZE = 128
N_SPLITS = 5
SEED = 0
RUN_PREFIX = "full_run/"        # parent folder under results/experiments/

CANCEL_RUNNING = True           # scancel our own abr_* jobs first
WIPE_EXISTING = True            # delete the run folders this sweep will rewrite

SINGLE_SLURM = dict(mem="48G", time="08:00:00", cpus=4, gpus="1", partition="gpu")
MULTI_SLURM = dict(mem="64G", time="16:00:00", cpus=6, gpus="1", partition="gpu")

# =============================================================================

import argparse
import os
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import sbatch_all_runs as S  # noqa: E402  (the submitter; reused wholesale)

RESULTS = os.path.join(PROJECT, "results", "experiments")


def cells():
    """(experiment_name, cfg) for every selected modality set x architecture."""
    return [(f"{name}__{arch}", {"modalities": list(mods), "arch": arch})
            for name, mods in MODALITY_SETS.items() for arch in ARCHITECTURES]


def cancel_running(dry):
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h",
                          "-o", "%i %j"], capture_output=True, text=True).stdout
    ids = [ln.split()[0] for ln in out.splitlines() if " abr_" in f" {ln}"]
    if not ids:
        print("  nothing of ours is queued")
        return
    print(f"  cancelling {len(ids)} job(s)")
    if not dry:
        subprocess.run(["scancel", *ids])
        time.sleep(3)


def wipe(names, dry):
    gone = 0
    for exp, _ in names:
        for folder in (RUN_PREFIX + exp, RUN_PREFIX + "multidrug_" + exp):
            path = os.path.join(RESULTS, folder)
            if os.path.isdir(path):
                gone += 1
                if not dry:
                    shutil.rmtree(path)
    print(f"  {'would remove' if dry else 'removed'} {gone} run folder(s)")


def submit(scope, name_cfgs, args, dry):
    """One job per (cell x drug) for single-drug, one per cell for multi-drug."""
    drugs = DRUGS if scope == "single" else [None]
    total, sent, jid = len(name_cfgs) * len(drugs), 0, 0
    for exp, cfg in name_cfgs:
        for drug in drugs:
            jid += 1
            script, _log = S.create_sbatch_script(exp, cfg, drug, args, jid)
            if dry:
                if jid <= 2:
                    print("    e.g. " + S.command_of(exp, cfg, drug, args))
                continue
            if S.submit_job(script, jid):
                sent += 1
            time.sleep(0.4)
    print(f"  {scope}-drug: {total if dry else sent}/{total} job(s)"
          f"{' (dry run)' if dry else ' submitted'}")
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and one example command per scope")
    opt = ap.parse_args()
    dry = opt.dry_run

    name_cfgs = cells()
    n_single = len(name_cfgs) * len(DRUGS) if "single" in SCOPES else 0
    n_multi = len(name_cfgs) if "multi" in SCOPES else 0

    print(f"sweep: {len(MODALITY_SETS)} modality set(s) x {len(ARCHITECTURES)} "
          f"architecture(s) = {len(name_cfgs)} cells")
    print(f"       scopes {SCOPES}, {len(DRUGS)} drug(s), {EPOCHS} epochs")
    print(f"       -> {n_single} single-drug + {n_multi} joint = "
          f"{n_single + n_multi} jobs into results/experiments/{RUN_PREFIX}\n")

    if CANCEL_RUNNING:
        cancel_running(dry)
    if WIPE_EXISTING:
        wipe(name_cfgs, dry)
    print()

    logs = os.path.join(PROJECT, "slurm_logs")
    os.makedirs(logs, exist_ok=True)
    common = dict(epochs=EPOCHS, batch_size=BATCH_SIZE, n_splits=N_SPLITS, seed=SEED,
                  run_prefix=RUN_PREFIX, arch="late_fusion", extra_loci=False,
                  per_locus_branches=False, tb=False, constraint=None,
                  project_dir=PROJECT, logs_dir=logs, dry_run=dry)
    if "single" in SCOPES:
        submit("single", name_cfgs, SimpleNamespace(**common, **SINGLE_SLURM), dry)
    if "multi" in SCOPES:
        submit("multi", name_cfgs, SimpleNamespace(**common, **MULTI_SLURM), dry)

    if not dry:
        print("\nwatch:  squeue -u $USER   |   results: "
              f"results/experiments/{RUN_PREFIX}")


if __name__ == "__main__":
    main()
