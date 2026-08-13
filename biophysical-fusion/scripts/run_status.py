#!/usr/bin/env python3
"""
One-screen status for the submitted sweeps: what finished, what is still going.

    python scripts/run_status.py              # active + recent runs
    python scripts/run_status.py -v           # break each run into its cells
    python scripts/run_status.py -a           # every run ever submitted
    python scripts/run_status.py full_run_v2  # only runs matching a substring

Two things this has to work around:

* squeue job names are `abr_{cell}_{drug}` with no run prefix, so
  `dna__late_fusion_ISONIAZID` is ambiguous across full_run_v2 / alllocus_run /
  full_run. job_id -> run mapping only exists in the slurm_logs manifests.
* manifests written before 2026-08-06 recorded `run_name` WITHOUT the
  `multidrug_` infix for joint jobs, so that field points at a folder those jobs
  never wrote to. We therefore parse `--run-name` out of the recorded `command`,
  which is authoritative: it is the string that actually ran.

"done" counts result json on disk, not exit codes — a job can exit 0 having
written nothing. Submitted, no longer queued, and no result = MISSING, which is
where to look first when a number seems low.
"""
import glob
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
EXPERIMENTS = PROJECT / "results" / "experiments"
WEIGHTS = Path("/project/pi_mfiterau_umass_edu/abr_model_weights")
RECENT_DAYS = 3


def squeue_states():
    """{job_id: 'R'|'PD'|...}; empty dict if squeue is unavailable."""
    try:
        out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h",
                              "-o", "%i|%t"], capture_output=True, text=True,
                             timeout=30)
    except Exception:
        return {}
    states = {}
    for line in out.stdout.splitlines():
        jid, _, st = line.partition("|")
        jid = jid.split("_")[0].strip()
        if jid.isdigit():
            states[jid] = st.strip()
    return states


def _run_name(entry):
    """--run-name from the recorded command; falls back to the run_name field."""
    cmd = entry.get("command") or ""
    try:
        toks = shlex.split(cmd)
        if "--run-name" in toks:
            return toks[toks.index("--run-name") + 1]
    except ValueError:
        pass
    return entry.get("run_name")


def manifest_jobs():
    """{run_name: {target: {"jobs": [job_id], "age": days}}}.

    A *target* is one (run_name, drug) pair — i.e. one output json. Counting
    targets rather than job submissions is what keeps resubmissions honest:
    full_run's 240 targets were each submitted twice, so a job-count denominator
    reported 480 submitted / 241 "missing" for a run that is in fact complete.
    """
    runs = defaultdict(lambda: defaultdict(lambda: {"jobs": [], "age": 1e9}))
    now = time.time()
    for f in sorted(glob.glob(str(PROJECT / "slurm_logs" / "submitted_*.json"))):
        try:
            data = json.loads(Path(f).read_text())
        except Exception:
            continue
        age = (now - os.path.getmtime(f)) / 86400.0
        for j in data.get("submitted_jobs", []):
            name = _run_name(j)
            if not name:
                continue
            t = runs[name][j.get("drug") or "ALL"]
            t["jobs"].append(str(j["job_id"]))
            t["age"] = min(t["age"], age)
    return runs


def results_on_disk(run_name):
    d = EXPERIMENTS / run_name
    if not d.is_dir():
        return 0
    joint = list(d.glob("multidrug__*.json"))
    single = [p for p in d.glob("*.json") if re.match(r"^[A-Z][A-Z]+__", p.name)]
    return len(joint) + len(single)


def weights_on_disk(run_name):
    d = WEIGHTS / run_name
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if (p / "config.json").exists())


def main():
    argv = sys.argv[1:]
    verbose = any(a in ("-v", "--verbose") for a in argv)
    show_all = any(a in ("-a", "--all") for a in argv)
    patterns = [a for a in argv if not a.startswith("-")]

    states = squeue_states()
    runs = manifest_jobs()
    if not runs:
        print("No submission manifests in slurm_logs/.")
        return

    rows = defaultdict(lambda: dict(sub=0, run=0, pend=0, done=0, miss=0, wts=0))
    detail = defaultdict(list)
    for run_name, targets in runs.items():
        group = run_name.split("/")[0]
        # a target is live if ANY of its (possibly resubmitted) jobs is queued
        live = {k: [j for j in t["jobs"] if j in states] for k, t in targets.items()}
        active = any(live.values())
        recent = min(t["age"] for t in targets.values()) <= RECENT_DAYS
        if patterns:
            if not any(p in run_name for p in patterns):
                continue
        elif not show_all and not (active or recent):
            continue

        done = results_on_disk(run_name)
        r = sum(1 for js in live.values() if any(states[j] == "R" for j in js))
        p = sum(1 for js in live.values()
                if js and not any(states[j] == "R" for j in js))
        miss = max(0, len(targets) - r - p - done)
        c = rows[group]
        c["sub"] += len(targets); c["run"] += r; c["pend"] += p
        c["done"] += done; c["miss"] += miss; c["wts"] += weights_on_disk(run_name)
        detail[group].append((run_name, len(targets), done, r, p, miss))

    if not rows:
        print(f"Nothing matched {patterns or 'active/recent runs'}. "
              f"Try -a for every run.")
        return

    w = max(len(g) for g in rows) + 2
    head = f"{'run':<{w}}{'done':>6}{'R':>5}{'PD':>6}{'miss':>6}{'total':>7}{'wts':>6}"
    print(head)
    print("-" * len(head))
    tot = defaultdict(int)
    for g in sorted(rows):
        c = rows[g]
        for k, v in c.items():
            tot[k] += v
        bar = "" if c["sub"] == 0 else f"  {100 * c['done'] // c['sub']:>3}%"
        print(f"{g:<{w}}{c['done']:>6}{c['run']:>5}{c['pend']:>6}"
              f"{c['miss']:>6}{c['sub']:>7}{c['wts']:>6}{bar}")
        if verbose:
            for name, n, d, r, p, m in sorted(detail[g]):
                cell = name.split("/", 1)[-1] if "/" in name else name
                print(f"    {cell:<{w + 4}}{d:>6}{r:>5}{p:>6}{m:>6}{n:>7}")
    print("-" * len(head))
    print(f"{'TOTAL':<{w}}{tot['done']:>6}{tot['run']:>5}{tot['pend']:>6}"
          f"{tot['miss']:>6}{tot['sub']:>7}{tot['wts']:>6}")
    if tot["miss"]:
        print("\nmiss = submitted, not queued, no result json. Check with:")
        print("  grep -l 'Traceback\\|CANCELLED\\|OOM\\|TIME LIMIT' slurm_logs/*.out | tail")
    if not show_all and not patterns:
        print(f"\n(active runs + those submitted in the last {RECENT_DAYS}d; "
              f"-a for all, -v for per-cell)")
    if not states:
        print("(squeue unavailable — run/pend shown as 0)")


if __name__ == "__main__":
    main()
