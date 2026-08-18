#!/usr/bin/env python3
"""
Submit one CPU SLURM job per WHO Table-22 promoter region, each running the
(write-safe) make_aligned_msa.py extractor over all BIG-TB VCFs to build that
region's per-isolate aligned FASTA in this workspace.

Reads promoter_windows.json (built by build_promoter_coords.py). Everything
written stays under regulatory_msa/ ; the pi_annagreen inputs are read-only
(VCF path list + reference GenBank).

    python submit_promoter_msa.py --dry_run        # print scripts, submit nothing
    python submit_promoter_msa.py                  # submit all missing regions
    python submit_promoter_msa.py --genes embA eis # just these
    python submit_promoter_msa.py --force          # re-extract even if FASTA exists
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

EXTRACTOR = os.path.join(HERE, "make_aligned_msa.py")     # write-safe copy
WINDOWS_JSON = os.path.join(HERE, "promoter_windows.json")
OUT_DIR = os.path.join(HERE, "aligned")
LOGS_DIR = os.path.join(HERE, "slurm_logs")


def extract_command(gene, w):
    parts = [
        "python", EXTRACTOR,
        "-f", VCF_PATHS,
        "-start", str(w["start"]), "-end", str(w["end"]), "-sense", w["sense"],
        "-o", os.path.join(OUT_DIR, f"{gene}.fasta"),
        "-g", GENBANK,
        "--gene", gene,
        "--resistance-groups", "regulatory",
    ]
    return " ".join(shlex.quote(p) for p in parts)


def sbatch_script(gene, w, args):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"promoter_{gene}_{ts}-%j.out")
    command = extract_command(gene, w)
    return f"""#!/bin/bash
#SBATCH -c {args.cpus}
#SBATCH --mem={args.mem}
#SBATCH -p {args.partition}
#SBATCH -t {args.time}
#SBATCH -o {log_path}
#SBATCH --job-name=prom_{gene}

# {gene}: WHO Table-22 promoter window [{w['start']}, {w['end']}] {w['sense']}
#   core [{w['core_start']}, {w['core_end']}]  tss={w['tss']}{' (CDS-start proxy)' if w['tss_is_proxy'] else ''}
echo "Extracting {gene} promoter  (host $(hostname))  $(date)"

source ~/.bashrc
conda activate {ENV}
cd {shlex.quote(HERE)}
mkdir -p {shlex.quote(OUT_DIR)} {shlex.quote(LOGS_DIR)}

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
    p.add_argument("--mem", default="8G")
    p.add_argument("--cpus", type=int, default=2)
    p.add_argument("--partition", default="cpu")
    p.add_argument("--time", default="02:00:00", help="H:MM:SS (default 2h; ~18min/gene observed)")
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    with open(WINDOWS_JSON) as f:
        windows = json.load(f)["windows"]
    genes = args.genes or list(windows)
    unknown = [g for g in genes if g not in windows]
    if unknown:
        sys.exit(f"Unknown gene(s) {unknown}; choose from {list(windows)}")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    todo = []
    for g in genes:
        fasta = os.path.join(OUT_DIR, f"{g}.fasta")
        if os.path.isfile(fasta) and not args.force:
            print(f"  skip {g}: {fasta} exists (use --force to redo)")
            continue
        todo.append(g)

    print(f"\n{len(todo)} region(s) to extract: {todo}")
    print(f"env={ENV} partition={args.partition} cpus={args.cpus} mem={args.mem} time={args.time}")
    print(f"out={OUT_DIR}")
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
