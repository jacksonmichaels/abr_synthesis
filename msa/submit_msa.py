#!/usr/bin/env python3
"""
Build per-isolate aligned FASTAs from the BIG-TB VCFs: one CPU SLURM job per
window, each running make_aligned_msa.py over every VCF.

Our promoter (regulatory) windows are the only MSAs built here; the gene MSAs
come from Sai's folder (paths.REAL_GENOTYPE_DIR). The windows file is
msa/windows/promoter_windows_atg.json (from msa/build_promoter_coords.py):
{"windows": {name: {"start", "end", "sense", ...}}}, start/end 1-based inclusive.

    python msa/submit_msa.py --windows msa/windows/promoter_windows_atg.json \
        --out-dir regulatory_msa/aligned_atg --dry_run

Inputs are read-only under pi_annagreen (VCF path list + GenBank). The
extractor needs the `make_msa` conda env (PyVCF). Each job reads ~18k VCFs from
/project, ~18 min per window; use --max-concurrent to throttle.
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
MSA_SRC = "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/make_msa"
VCF_PATHS = os.path.join(MSA_SRC, "path_list/big_tb_file_paths.txt")
GENBANK = os.path.join(MSA_SRC, "genbank_reference_data/genome.gbff")
EXTRACTOR = os.path.join(HERE, "make_aligned_msa.py")
ACCOUNT = "pi_annagreen_umass_edu"


def extract_command(gene, w, out_dir):
    parts = ["python", EXTRACTOR, "-f", VCF_PATHS,
             "-start", str(w["start"]), "-end", str(w["end"]), "-sense", w["sense"],
             "-o", os.path.join(out_dir, f"{gene}.fasta"), "-g", GENBANK, "--gene", gene]
    return " ".join(shlex.quote(p) for p in parts)


def sbatch_script(gene, w, args, logs_dir):
    log_path = os.path.join(logs_dir, f"msa_{gene}-%j.out")
    return f"""#!/bin/bash
#SBATCH --account={ACCOUNT}
#SBATCH -c {args.cpus}
#SBATCH --mem={args.mem}
#SBATCH -p cpu
#SBATCH -t {args.time}
#SBATCH -o {log_path}
#SBATCH --job-name=msa_{gene}
# {gene}: [{w['start']}, {w['end']}] {w['sense']}  from {os.path.basename(args.windows)}
set -euo pipefail
echo "Extracting {gene}  (host $(hostname))  $(date)"
source ~/.bashrc
conda activate {ENV}
which python && python --version
mkdir -p {shlex.quote(args.out_dir)}
{extract_command(gene, w, args.out_dir)}
echo "Done: {gene}  $(date)"
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows", required=True, help="windows JSON")
    p.add_argument("--out-dir", required=True, help="output FASTA dir")
    p.add_argument("--genes", nargs="+", default=None, help="subset (default: all in the JSON)")
    p.add_argument("--force", action="store_true", help="re-extract even if the FASTA exists")
    p.add_argument("--mem", default="8G")
    p.add_argument("--cpus", type=int, default=2)
    p.add_argument("--time", default="02:00:00")
    p.add_argument("--max-concurrent", type=int, default=0,
                   help="hold each job on the one N submissions earlier (0 = no throttle)")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    args.out_dir = os.path.abspath(args.out_dir)
    logs_dir = os.path.join(args.out_dir, "slurm_logs")

    with open(args.windows) as f:
        windows = json.load(f)["windows"]
    genes = args.genes or list(windows)
    unknown = [g for g in genes if g not in windows]
    if unknown:
        sys.exit(f"Unknown window(s) {unknown}; choose from {list(windows)}")
    todo = [g for g in genes
            if args.force or not os.path.isfile(os.path.join(args.out_dir, f"{g}.fasta"))]
    print(f"{len(todo)} window(s) to extract into {args.out_dir}: {todo}")
    if args.dry_run:
        for g in todo:
            print(f"===== {g} =====\n{sbatch_script(g, windows[g], args, logs_dir)}")
        print("DRY RUN: nothing submitted.")
        return

    os.makedirs(logs_dir, exist_ok=True)
    job_ids = []
    for g in todo:
        fname = os.path.join(logs_dir, f"_sbatch_{g}.sh")
        with open(fname, "w") as fh:
            fh.write(sbatch_script(g, windows[g], args, logs_dir))
        cmd = ["sbatch", "--parsable"]
        if args.max_concurrent and len(job_ids) >= args.max_concurrent:
            cmd.append(f"--dependency=afterany:{job_ids[-args.max_concurrent]}")
        out = subprocess.run(cmd + [fname], capture_output=True, text=True, check=True)
        job_ids.append(out.stdout.strip().split(";")[0])
        print(f"  {g}: submitted {job_ids[-1]}")
        time.sleep(0.5)
    with open(os.path.join(logs_dir, f"submitted_{datetime.now():%Y%m%d_%H%M%S}.json"), "w") as fh:
        json.dump({"invocation": " ".join(map(shlex.quote, sys.argv)),
                   "jobs": dict(zip(todo, job_ids))}, fh, indent=2)


if __name__ == "__main__":
    main()
