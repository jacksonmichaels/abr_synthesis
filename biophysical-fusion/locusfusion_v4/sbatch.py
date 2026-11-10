"""
Submit LocusFusion v4 runs to SLURM, from a frozen copy of the code.

    cd biophysical-fusion
    python -m locusfusion_v4.sbatch --arms canary --dry-run      # print, submit nothing
    python -m locusfusion_v4.sbatch --arms tokens                # CPU: build the token caches
    python -m locusfusion_v4.sbatch --arms canary --dependency afterok:<tokens job>
    python -m locusfusion_v4.sbatch --arms tune                  # aug frac x lambda, PZA
    python -m locusfusion_v4.sbatch --arms noaug aug nodist nosyn md_noaug md_aug md_nodist md_nosyn

Arms (every single-drug arm = 11 jobs, one per drug, 19 loci, lineage split):
    tokens     CPU: build every token cache once, so GPU jobs never race to build them
    canary     PYRAZINAMIDE, augmentation on (frac 0.5, lambda 0.5)
    tune       PYRAZINAMIDE, aug frac {0.25, 0.5, 1.0} x lambda {0, 0.5, 2}
    noaug      v4 tokens, no augmentation                    (the tokenizer arm)
    aug        v4 + synonymous augmentation                  (the main model)
    nodist     v4 + augmentation, no site-proximity channel  (distance ablation)
    nosyn      v4, synonymous changes dropped from the tokens (exact-invariance arm)
    v5         v3's token + consequence + distance, synonymous dropped, no augmentation
    v5_nodist  v5 without the distance channels (what the distance is worth in v5)
    v5_fixdna  v5 on the re-extracted eis/embB/ethA/rrs (insertion bug repaired)
    v5_atg     v5 on the start-codon-anchored promoter windows
    v5_fixed   v5 on both repaired inputs
    md_*       the same four on the multi-drug model (one job each)
Set the augmentation for aug / nodist / md_aug / md_nodist with --aug-frac / --aug-lambda
once `tune` has picked them (defaults 0.5 / 0.5).

Code runs from <workspace>/v4_snapshot/<stamp>/ (biophysical-fusion minus results,
notebooks and logs, plus the BIG-TB parameters folder bigtb_ref imports), with
BASE_COMMIT and working_tree.patch alongside, so editing the repo cannot reach a
queued job. Data comes from data_snapshot via ABR_*. Results go to
biophysical-fusion/results/experiments/locusfusion_v4/<arm>/, weights to
<workspace>/model_weights_v4/, job logs to results/experiments/locusfusion_v4/logs/,
and every submission writes a manifest there.
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
RUN_ROOT = PROJECT / "results" / "experiments" / "locusfusion_v4"
LINEAGE_SPLITS = PROJECT / "results" / "experiments" / "lineage_split" / "splits"
MD_SPLIT = RUN_ROOT / "splits" / "MULTIDRUG.csv"
CONDA_ENV = "abr_env"
DRUGS = ["AMIKACIN", "CAPREOMYCIN", "ETHAMBUTOL", "ETHIONAMIDE", "ISONIAZID", "KANAMYCIN",
         "LEVOFLOXACIN", "MOXIFLOXACIN", "PYRAZINAMIDE", "RIFAMPICIN", "STREPTOMYCIN"]
ENV = {"ABR_GENOTYPE_DIR": DATA / "genomic_data" / "aligned",
       "ABR_PHENOTYPE_CSV": DATA / "phenotype_data" / "master_resistance_table.csv",
       "ABR_GENBANK": DATA / "genbank" / "genome.gbff"}


def arm_jobs(arm, frac, lam):
    """[(job name, results subfolder, train.py args, kind)] for one arm."""
    aug = ["--aug-frac", str(frac), "--aug-lambda", str(lam)]
    sd = lambda sub, extra: [(f"v4_{sub}_{d}", sub, ["--drugs", d, "--split-dir", str(LINEAGE_SPLITS),  # noqa: E731
                                                       *extra], "gpu") for d in DRUGS]
    md = lambda sub, extra: [(f"v4_{sub}", sub, ["--drugs", "all", "--split-file", str(MD_SPLIT),  # noqa: E731
                                                  *extra], "gpu_md")]
    if arm == "tokens":
        return [("v4_tokens", "tokens", [], "cpu")]
    if arm == "canary":
        return [("v4_canary_PYRAZINAMIDE", "canary",
                 ["--drugs", "PYRAZINAMIDE", "--split-dir", str(LINEAGE_SPLITS),
                  "--aug-frac", "0.5", "--aug-lambda", "0.5"], "gpu")]
    if arm == "tune":
        return [(f"v4_tune_f{f}_l{l}_PYRAZINAMIDE", f"tune/f{f}_l{l}",
                 ["--drugs", "PYRAZINAMIDE", "--split-dir", str(LINEAGE_SPLITS),
                  "--aug-frac", str(f), "--aug-lambda", str(l)], "gpu")
                for f in (0.25, 0.5, 1.0) for l in (0.0, 0.5, 2.0)]
    table = {"noaug": [], "aug": aug, "nodist": aug + ["--no-distance"],
             "nosyn": ["--drop-synonymous"],
             # v5 (2026-10-28): v3's token kept whole -- residue one-hot AND
             # consequence -- plus distance, synonymous tokens dropped, no
             # augmentation. The sweep's three CV-positive ingredients, and
             # none of the one that cost CV.
             "v5": ["--keep-residue", "--drop-synonymous"],
             "v5_nodist": ["--keep-residue", "--drop-synonymous", "--no-distance"],
             # the data fixes (2026-10-28), each against v5 as the control:
             #   fixdna  eis/embB/ethA/rrs re-extracted without the insertion bug
             #   atg     promoter windows anchored on the start codon, not the TSS
             #   fixed   both
             "v5_fixdna": ["--keep-residue", "--drop-synonymous",
                           "--genotype-dir", str(FIXED_GENO)],
             "v5_atg": ["--keep-residue", "--drop-synonymous",
                        "--regulatory-dir", str(ATG_REG)],
             "v5_fixed": ["--keep-residue", "--drop-synonymous",
                          "--genotype-dir", str(FIXED_GENO), "--regulatory-dir", str(ATG_REG)]}
    if arm in table:
        return sd(f"sd19_{arm}", table[arm])
    if arm.startswith("md_") and arm[3:] in table:
        return md(arm, table[arm[3:]])
    raise SystemExit(f"unknown arm {arm!r}")


RESOURCES = {   # kind -> (partition, cpus, mem, time, gpus, constraint)
    "cpu": ("cpu", 2, "12G", "1:30:00", 0, None),     # ~50 s + 4.2 GB per cache, 24 caches
    "gpu": ("gpu", 4, "32G", "12:00:00", 1, "vram23"),
    "gpu_md": ("gpu", 4, "48G", "24:00:00", 1, "vram23"),
}

# the repaired inputs (see curated_fix/aligned19/README.md and
# regulatory_msa/aligned_atg); both are the same cohort, 17,942 isolates
FIXED_GENO = WORKSPACE / "curated_fix" / "aligned19"
ATG_REG = WORKSPACE / "regulatory_msa" / "aligned_atg"

TOKENS_PY = r"""
from bigtb_ref import REAL_GENOTYPE_DIR as G, REAL_PHENOTYPE_CSV as P, REAL_REGULATORY_DIR as R
from locusfusion_v4.tokenizer import ALL_DRUGS, load_or_build
import sys
cache = sys.argv[1]
GEO_FIX = sys.argv[2] if len(sys.argv) > 2 else None
REG_ATG = sys.argv[3] if len(sys.argv) > 3 else None
for drop, res in ((False, False), (True, False), (True, True)):
    for drugs in [[d] for d in ALL_DRUGS] + [ALL_DRUGS]:
        load_or_build(cache, drugs, G, P, R, drop_synonymous=drop, keep_residue=res)
# the data-fix arms: keep_residue + drop_synonymous on the repaired inputs
for geno, reg in ((GEO_FIX, R), (G, REG_ATG), (GEO_FIX, REG_ATG)):
    if not geno or not reg:
        continue
    for drugs in [[d] for d in ALL_DRUGS]:
        load_or_build(cache, drugs, geno, P, reg, drop_synonymous=True, keep_residue=True)
"""


def make_snapshot(stamp, dry):
    snap = WORKSPACE / "v4_snapshot" / stamp
    cmds = [
        ["mkdir", "-p", str(snap / "Big-TB-benchmark" / "dna-tasks" / "SD-CNN" / "model_training")],
        ["rsync", "-a", "--exclude", "results", "--exclude", "notebooks", "--exclude", "slurm_logs",
         "--exclude", "__pycache__", "--exclude", "diagrams", f"{PROJECT}/", f"{snap / 'biophysical-fusion'}/"],
        ["rsync", "-a", f"{WORKSPACE}/Big-TB-benchmark/dna-tasks/SD-CNN/model_training/parameters",
         f"{snap / 'Big-TB-benchmark' / 'dna-tasks' / 'SD-CNN' / 'model_training'}/"],
    ]
    if dry:
        for c in cmds:
            print("  [snapshot] " + " ".join(shlex.quote(x) for x in c))
        return snap
    for c in cmds:
        subprocess.run(c, check=True)
    commit = subprocess.run(["git", "-C", str(WORKSPACE), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout
    (snap / "BASE_COMMIT").write_text(commit)
    patch = subprocess.run(["git", "-C", str(WORKSPACE), "diff", "HEAD", "--", "biophysical-fusion"],
                           capture_output=True, text=True, check=True).stdout
    (snap / "working_tree.patch").write_text(patch)
    status = subprocess.run(["git", "-C", str(WORKSPACE), "status", "--short", "biophysical-fusion"],
                            capture_output=True, text=True, check=True).stdout
    (snap / "status.txt").write_text(status)
    subprocess.run(["chmod", "-R", "a-w", str(snap / "biophysical-fusion")], check=True)
    return snap


def job_script(name, sub, targs, kind, snap, cache, weights):
    part, cpus, mem, tlim, gpus, constraint = RESOURCES[kind]
    logs = RUN_ROOT / "logs"
    out = RUN_ROOT / sub
    lines = ["#!/bin/bash",
             "#SBATCH --account=pi_annagreen_umass_edu",  # ABR jobs bill Anna Green, not the user default
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
    lines += ["set -euo pipefail",
              'echo "host $(hostname)  start $(date)"',
              "module list 2>&1 || true",
              "source ~/.bashrc",
              f"conda activate {CONDA_ENV}",
              "which python && python --version",
              'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
              'export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
              *[f"export {k}={shlex.quote(str(v))}" for k, v in ENV.items()],
              f"cd {shlex.quote(str(snap / 'biophysical-fusion'))}"]
    if kind == "cpu":
        lines.append(f"python -u - {shlex.quote(str(cache))} {shlex.quote(str(FIXED_GENO))} "
                     f"{shlex.quote(str(ATG_REG))} <<'EOF'\n{TOKENS_PY.strip()}\nEOF")
    else:
        cmd = ["python", "-u", "-m", "locusfusion_v4.train", *targs,
               "--out", str(out), "--token-cache", str(cache), "--weights-dir", str(weights)]
        lines.append(" ".join(shlex.quote(c) for c in cmd))
    lines.append('echo "done $(date)"')
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--aug-frac", type=float, default=0.5)
    ap.add_argument("--aug-lambda", type=float, default=0.5)
    ap.add_argument("--snapshot", default=None,
                    help="reuse an existing v4_snapshot/<stamp> instead of making a new one")
    ap.add_argument("--dependency", default=None,
                    help="e.g. afterok:12345 -- hold every job until the tokens job is done")
    ap.add_argument("--token-cache", default=str(WORKSPACE / "token_cache_v4"))
    ap.add_argument("--weights-dir", default=str(WORKSPACE / "model_weights_v4"))
    args = ap.parse_args()

    jobs = [j for arm in args.arms for j in arm_jobs(arm, args.aug_frac, args.aug_lambda)]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    snap = Path(args.snapshot) if args.snapshot else make_snapshot(stamp, args.dry_run)
    if args.snapshot and not (snap / "biophysical-fusion" / "locusfusion_v4").exists():
        raise SystemExit(f"{snap} is not a v4 snapshot")
    scripts = RUN_ROOT / "logs" / "scripts" / stamp
    if not args.dry_run:
        scripts.mkdir(parents=True, exist_ok=True)
    manifest = {"stamp": stamp, "snapshot": str(snap), "arms": args.arms,
                "dependency": args.dependency,
                "aug_frac": args.aug_frac, "aug_lambda": args.aug_lambda, "jobs": []}
    for name, sub, targs, kind in jobs:
        text = job_script(name, sub, targs, kind, snap, args.token_cache, args.weights_dir)
        if args.dry_run:
            print(f"### {name}  [{kind}: {RESOURCES[kind]}]")
            print(text.splitlines()[-2])
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
