#!/usr/bin/env python3
"""
Submit one CPU SLURM job per WHO Table-21 tier 1+2 GENE BODY (gene_windows.json), each running the
(write-safe) make_aligned_msa.py extractor over all BIG-TB VCFs to build that
region's per-isolate aligned FASTA in this workspace.

Reads gene_windows.json (built by build_gene_windows.py), or --windows. Everything
written stays in this workspace; the pi_annagreen inputs are read-only
(VCF path list + reference GenBank).

    python submit_gene_msa.py --dry_run        # print scripts, submit nothing
    python submit_gene_msa.py                  # submit all missing regions
    python submit_gene_msa.py --genes ahpC furA # just these
    python submit_gene_msa.py --force          # re-extract even if FASTA exists
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = "make_msa"
# read-only inputs under pi_annagreen (never written to)
MSA_SRC = "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/make_msa"
VCF_PATHS = os.path.join(MSA_SRC, "path_list/big_tb_file_paths.txt")
GENBANK = os.path.join(MSA_SRC, "genbank_reference_data/genome.gbff")

EXTRACTOR = os.path.join(HERE, "..", "regulatory_msa", "make_aligned_msa.py")   # write-safe copy
WINDOWS_JSON = os.path.join(HERE, "gene_windows.json")
# aligned_fixed/ = re-extraction with the 2026-10-22 insertion fix in
# make_aligned_msa.py; aligned/ (Sep 2026, insertion bug) is kept for the who_t12 results built on it
OUT_DIR = os.path.join(HERE, "aligned_fixed")
LOGS_DIR = os.path.join(HERE, "slurm_logs")


def extract_command(gene, w, out_dir):
    parts = [
        "python", EXTRACTOR,
        "-f", VCF_PATHS,
        "-start", str(w["start"]), "-end", str(w["end"]), "-sense", w["sense"],
        "-o", os.path.join(out_dir, f"{gene}.fasta"),
        "-g", GENBANK,
        "--gene", gene,
        "--resistance-groups", "who_t12",
    ]
    return " ".join(shlex.quote(p) for p in parts)


def sbatch_script(gene, w, args):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"gene_{gene}_{ts}-%j.out")
    command = extract_command(gene, w, args.out_dir)
    return f"""#!/bin/bash
#SBATCH -c {args.cpus}
#SBATCH --mem={args.mem}
#SBATCH -p {args.partition}
#SBATCH -t {args.time}
#SBATCH -o {log_path}
#SBATCH --job-name=genemsa_{gene}

# {gene}: gene body +/- {w['pad']} bp [{w['start']}, {w['end']}] {w['sense']}  (WHO T21 tier 1+2; {', '.join(w['drugs'])})
echo "Extracting {gene} gene body  (host $(hostname))  $(date)"

source ~/.bashrc
conda activate {ENV}
set -euo pipefail
which python && python --version
cd {shlex.quote(HERE)}
mkdir -p {shlex.quote(args.out_dir)} {shlex.quote(LOGS_DIR)}

{command}

echo "Done: {gene}  $(date)"
""", log_path


def submit(script_text, gene, keep_dir):
    fname = os.path.join(keep_dir, f"_sbatch_{gene}.sh")
    with open(fname, "w") as fh:
        fh.write(script_text)
    try:
        out = subprocess.run(["sbatch", fname], capture_output=True, text=True, check=True)
        return out.stdout.strip().split()[-1]
    except subprocess.CalledProcessError as e:
        print(f"  ERROR submitting {gene}: {e.stderr.strip()}")
        return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--genes", nargs="+", default=None, help="subset (default: all in json)")
    p.add_argument("--force", action="store_true", help="re-extract even if FASTA exists")
    p.add_argument("--windows", default=WINDOWS_JSON, help="window JSON (default: %(default)s)")
    p.add_argument("--out-dir", default=OUT_DIR, help="output FASTA dir (default: %(default)s)")
    p.add_argument("--mem", default="8G")
    p.add_argument("--cpus", type=int, default=2)
    p.add_argument("--partition", default="cpu")
    p.add_argument("--time", default="02:00:00", help="H:MM:SS (default 2h; ~18min/gene observed)")
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir = os.path.abspath(args.out_dir)
    with open(args.windows) as f:
        windows = json.load(f)["windows"]
    genes = args.genes or list(windows)
    unknown = [g for g in genes if g not in windows]
    if unknown:
        sys.exit(f"Unknown gene(s) {unknown}; choose from {list(windows)}")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    todo = []
    for g in genes:
        fasta = os.path.join(args.out_dir, f"{g}.fasta")
        if os.path.isfile(fasta) and not args.force:
            print(f"  skip {g}: {fasta} exists (use --force to redo)")
            continue
        todo.append(g)

    print(f"\n{len(todo)} region(s) to extract: {todo}")
    print(f"env={ENV} partition={args.partition} cpus={args.cpus} mem={args.mem} time={args.time}")
    print(f"windows={args.windows}\nout={args.out_dir}")
    if args.dry_run:
        print("DRY RUN — nothing submitted.\n")

    submitted = []
    for i, g in enumerate(todo):
        script, log_path = sbatch_script(g, windows[g], args)
        if args.dry_run:
            print(f"===== {g} =====\n{script}")
            continue
        sid = submit(script, g, LOGS_DIR)
        if sid:
            submitted.append({"job_id": sid, "gene": g, "log": log_path,
                              "window": [windows[g]["start"], windows[g]["end"], windows[g]["sense"]]})
            print(f"  [{i+1}/{len(todo)}] {g}: submitted {sid}")
        if i < len(todo) - 1:
            time.sleep(args.delay)

    if not args.dry_run and submitted:
        track = os.path.join(LOGS_DIR, "submitted_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
        with open(track, "w") as fh:
            json.dump({"invocation": " ".join(map(shlex.quote, sys.argv)),
                       "submitted": submitted}, fh, indent=2)
        print(f"\nSubmitted {len(submitted)} jobs. Tracking: {track}")
        print("Monitor:  squeue -u $USER   |   cancel:  scancel <job_id>")


if __name__ == "__main__":
    main()
