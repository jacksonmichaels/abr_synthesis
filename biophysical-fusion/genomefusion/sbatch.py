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
    gf19p        the same loci PLUS the intergenic loci covering their 100 bp 5'
                 flank. v3's padded FASTAs carry those bases and `gf19` drops
                 them, so this is the input-matched parity arm; it tests whether
                 the gf19 shortfall is the missing promoters
    gf_genome    every locus the partition keeps
    gf_minus19p  every locus except the 19 AND their promoter loci -- is there
                 signal outside the known mechanisms?
    gf_minus19   the complement of the 19 alone, so it still sees their
                 promoters (c-15t included). Superseded by gf_minus19p; kept
                 only to reproduce older runs

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
#: every ABR job bills Anna Green's allocation. Set here, not via the user default,
#: because this account's default (pi_mfiterau_umass_edu) is for its other project.
SLURM_ACCOUNT = "pi_annagreen_umass_edu"
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

# `gpu` is not preemptible but is heavily contended by other groups;
# `gpu-preempt-highvram` has far more idle vram23 capacity (L40S / A100 / H100)
# and PreemptMode=REQUEUE with a 900 s grace period. Listing both lets SLURM
# take whichever frees first, and --requeue means a preempted job comes back
# rather than dying. A requeued job restarts from scratch -- there is no
# checkpointing -- so this trades a chance of lost work for a much shorter wait.
PARTITIONS = "gpu,gpu-preempt-highvram"
#: the newest CUDA arch this env's torch has kernels for. torch 2.6.0+cu124
#: stops at sm_90, so a Blackwell card (sm_120) fails at the first kernel with
#: "no kernel image is available for execution on the device" -- which is how
#: the first canary died on gpu058. Nodes advertising anything newer are
#: excluded at submit time rather than hard-coded, so a card added later cannot
#: quietly start eating jobs.
MAX_SM = 90


def unsupported_nodes(max_sm=MAX_SM, partitions=PARTITIONS):
    """Nodes whose GPU arch is newer than this torch build supports."""
    try:
        out = subprocess.run(["sinfo", "-h", "-o", "%n %f", "-p", partitions],
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    bad = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        for f in parts[1].split(","):
            if f.startswith("sm_") and f[3:].isdigit() and int(f[3:]) > max_sm:
                bad.add(parts[0])
    return sorted(bad)
ALL_GPU_PARTITIONS = "gpu,gpu-preempt,gpu-preempt-highvram"
RESOURCES = {   # kind -> (partition, cpus, mem, time, gpus, constraint)
    "gpu": (PARTITIONS, 4, "64G", "48:00:00", 1, "vram23"),
    # The canary is a measurement, not a result, so it takes whatever is free.
    # Memory here is dominated by the across-loci attention, which --locus-budget
    # caps, not by the 1.5M parameters, so an 11 GB card is enough at a smaller
    # budget.
    "canary": (ALL_GPU_PARTITIONS, 4, "32G", "4:00:00", 1, "vram11"),
    # gf19 reads 19 loci, of which an isolate changes ~5, so its batches are
    # tiny next to the genome arms' ~707 active loci. It takes any free card
    # rather than queueing for a big one it would not use.
    "gpu_small": (ALL_GPU_PARTITIONS, 4, "32G", "24:00:00", 1, "vram11"),
}

ARMS = {"gf19": "19", "gf19r": "19r", "gf19p": "19p", "gf_genome": "genome",
        "gf_minus19p": "minus19p", "gf_minus19": "minus19",
        "gf_who_t12": "who_t12", "gf_who_t12r": "who_t12r",
        "gf_lab64": "lab64", "gf_lab64r": "lab64r"}
#: arms whose isolates activate a handful of loci, so they fit any free card
SMALL_ARMS = ("gf19", "gf19r", "gf19p", "gf_who_t12", "gf_who_t12r", "gf_lab64", "gf_lab64r")
#: The locus-size ladder, up to the whole genome. Nested by construction (one
#: --ladder-seed for every rung), so a change along it is the cost of carrying
#: more loci, not a change of content. Rungs at or below LADDER_SMALL activate
#: few enough loci per isolate to take any free card.
LADDER = (64, 128, 256, 512, 1024, 2048, 4096, 6776)
LADDER_SMALL = 512
LADDER_SEED = 0


def arm_jobs(arm, batch_size, locus_budget=None, drugs=None):
    bs = ["--batch-size", str(batch_size)]
    if locus_budget:
        bs += ["--locus-budget", str(locus_budget)]
    if arm == "canary":
        return [("gf_canary_PYRAZINAMIDE", "canary",
                 ["--drugs", "PYRAZINAMIDE", "--locus-set", "genome", "--folds", "0", *bs],
                 "canary")]
    if arm.startswith("ladder"):
        # "ladder" = every rung; "ladder:512" = that rung only
        want = ([int(arm.split(":", 1)[1])] if ":" in arm else list(LADDER))
        bad = [n for n in want if n not in LADDER]
        if bad:
            raise SystemExit(f"rung(s) {bad} not in the ladder {LADDER}")
        jobs = []
        for n in want:
            small = n <= LADDER_SMALL
            for d in (drugs or DRUGS):
                jobs.append((f"gf_size{n}_{d}", f"size{n}",
                             ["--drugs", d, "--locus-set", f"size:{n}",
                              "--ladder-seed", str(LADDER_SEED),
                              "--max-hours", "20" if small else "44", *bs],
                             "gpu_small" if small else "gpu"))
        return jobs
    if arm == "gf19r_nofeat":
        # gf19r without the 29 copied target features, to measure what they contribute.
        # The flag leaves the reg_gate edge in place, so this is not a no-edge control.
        return [(f"gf_{arm}_{d}", arm,
                 ["--drugs", d, "--locus-set", "19r", "--no-regulated-features", "--max-hours", "20", *bs],
                 "gpu_small")
                for d in (drugs or DRUGS)]
    if arm in ARMS:
        # stop 4 h short of the 48 h SLURM limit, so a run that outlasts its
        # budget writes the folds it finished instead of being killed
        small = arm in SMALL_ARMS
        hours = "20" if small else "44"
        return [(f"gf_{arm}_{d}", arm,
                 ["--drugs", d, "--locus-set", ARMS[arm], "--max-hours", hours, *bs],
                 "gpu_small" if small else "gpu")
                for d in (drugs or DRUGS)]
    raise SystemExit(f"unknown arm {arm!r}; pick from canary, ladder, ladder:N, "
                     f"gf19r_nofeat, {', '.join(ARMS)}")


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


def job_script(name, sub, targs, kind, snap, tokens, weights, constraint=None, cache=None):
    part, cpus, mem, tlim, gpus, default_constraint = RESOURCES[kind]
    constraint = constraint or default_constraint
    logs, out = RUN_ROOT / "logs", RUN_ROOT / sub
    lines = ["#!/bin/bash",
             f"#SBATCH --account={SLURM_ACCOUNT}",
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
    # a preempted job is requeued rather than lost; it restarts from scratch
    lines.append("#SBATCH --requeue")
    bad = unsupported_nodes()
    if bad:
        lines.append(f"#SBATCH --exclude={','.join(bad)}")
    cmd = ["python", "-u", "-m", "genomefusion.train", *targs,
           "--out", str(out), "--split-dir", str(LINEAGE_SPLITS), "--tokens", str(tokens)]
    if weights:
        cmd += ["--weights-dir", str(weights)]
    # Passed explicitly, so a new arm can use a NEW partition cache while jobs
    # already queued keep reading the one they were submitted against. The
    # cache is the only mutable input shared by every job: rebuilding it in
    # place is what would break them.
    if cache:
        cmd += ["--partition-cache", str(cache)]
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
    ap.add_argument("--drugs", nargs="+", default=None,
                    help="submit only these drugs of each arm (default: all 11). Use it to "
                         "smoke-test one drug before committing the arm's GPU hours.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--dependency", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--locus-budget", type=int, default=None,
                    help="passed to train.py; cap on batch x widest isolate. Smaller fits a "
                         "smaller card. Default: train.py's own (32768), or 16384 for a canary")
    ap.add_argument("--constraint", default=None, help="override the arm's GPU constraint")
    ap.add_argument("--tokens", default=str(TOKENS))
    ap.add_argument("--partition-cache", default=str(WORKSPACE / "genomefusion_cache_v3"),
                   help="directory holding partition.npz for these jobs. Point a new arm at a "
                        "NEW directory rather than rebuilding the shared one, which every "
                        "queued job reads at startup and would crash on.")
    ap.add_argument("--weights-dir", default=str(WORKSPACE / "model_weights_gf"))
    args = ap.parse_args()

    if args.drugs:
        unknown = [d for d in args.drugs if d.upper() not in DRUGS]
        if unknown:
            raise SystemExit(f"unknown drug(s) {unknown}; pick from {DRUGS}")
        args.drugs = [d.upper() for d in args.drugs]
    jobs = [j for arm in args.arms
            for j in arm_jobs(arm, args.batch_size,
                              args.locus_budget or (16384 if arm == "canary" else None),
                              args.drugs)]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    snap = Path(args.snapshot) if args.snapshot else make_snapshot(stamp, args.dry_run)
    if args.snapshot and not (snap / "biophysical-fusion" / "genomefusion").exists():
        raise SystemExit(f"{snap} is not a genomefusion snapshot")
    scripts = RUN_ROOT / "logs" / "scripts" / stamp
    if not args.dry_run:
        scripts.mkdir(parents=True, exist_ok=True)
    manifest = {"stamp": stamp, "account": SLURM_ACCOUNT, "snapshot": str(snap), "arms": args.arms,
                "dependency": args.dependency, "batch_size": args.batch_size,
                "locus_budget": args.locus_budget, "constraint": args.constraint,
                "tokens": args.tokens, "partition_cache": args.partition_cache,
                "ladder": list(LADDER), "ladder_seed": LADDER_SEED, "jobs": []}
    for name, sub, targs, kind in jobs:
        text = job_script(name, sub, targs, kind, snap, args.tokens, args.weights_dir,
                          args.constraint, args.partition_cache)
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
