#!/usr/bin/env python3
"""
Per-JOB glance at a live sweep: what finished, what is still going, how far.

    python scripts/run_monitors/sweep_status.py                    # whatever is live
    python scripts/run_monitors/sweep_status.py setfusion_scaling  # one sweep
    python scripts/run_monitors/sweep_status.py -d                 # done cells too
    watch -n 120 python scripts/run_monitors/sweep_status.py       # live dashboard

Complements ``scripts/run_status.py`` rather than repeating it: that one rolls
whole runs up into one line each ("full_run_v2: 218 done, 2 running"). This one
goes the other way — one line per JOB, with the progress of the ones still
running, which is what you want while a sweep is in flight.

Progress comes from each job's own SLURM log. The manifests record the log path
with SLURM's ``%j`` placeholder, so substituting the job id gives the exact file;
counting its ``CV fold k:`` lines gives folds-done out of 5 without parsing
anything fragile.

States, and the one that matters:

    done   result json on disk (the metric is read from it)
    R/PD   still queued, per squeue
    FAIL   submitted, no longer queued, NO result json — a job can exit 0 having
           written nothing, so this is inferred from disk, never from an exit
           code. This is the state to look at first.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT = HERE.parent.parent.parent
sys.path.insert(0, str(HERE.parent.parent))          # scripts/ — for run_status

from run_status import _run_name  # noqa: E402  (the --run-name-from-command rule)

EXPERIMENTS = PROJECT / "results" / "experiments"
RECENT_DAYS = 5


def squeue_jobs():
    """{job_id: (state, elapsed, time_left, node_or_reason)}."""
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i|%t|%M|%L|%R"],
            capture_output=True, text=True, timeout=30)
    except Exception:
        return {}
    jobs = {}
    for line in out.stdout.splitlines():
        parts = line.split("|")
        if len(parts) == 5 and parts[0].split("_")[0].strip().isdigit():
            jobs[parts[0].split("_")[0].strip()] = tuple(p.strip() for p in parts[1:])
    return jobs


def manifest_rows(patterns, days):
    """One row per submitted job, newest submission wins on resubmission."""
    rows, now = {}, time.time()
    for f in sorted(glob.glob(str(PROJECT / "slurm_logs" / "submitted_*.json")),
                    key=os.path.getmtime):
        age = (now - os.path.getmtime(f)) / 86400.0
        if age > days:
            continue
        try:
            data = json.loads(Path(f).read_text())
        except Exception:
            continue
        for j in data.get("submitted_jobs", []):
            run = _run_name(j)
            if not run or (patterns and not any(p in run for p in patterns)):
                continue
            drug = j.get("drug") or "ALL"
            rows[(run, drug)] = {"job": str(j["job_id"]), "run": run, "drug": drug,
                                 "log": j.get("log", ""), "age": age}
    return list(rows.values())


def result_json(run, drug):
    d = EXPERIMENTS / run
    if not d.is_dir():
        return None
    hits = (sorted(d.glob("multidrug__*.json")) if drug.startswith("ALL")
            else sorted(d.glob(f"{drug}__*.json")))
    return hits[0] if hits else None


def metric(path):
    try:
        j = json.loads(path.read_text())
    except Exception:
        return None
    cv = j.get("cv_macro_auc_mean", j.get("cv_auc_mean"))
    return (cv, j.get("n_params"), j.get("seconds", 0) / 3600) if cv is not None else None


def progress(log_glob, job):
    """(folds_done, last_line_hint) for a running job, from its own log."""
    if not log_glob:
        return None
    path = Path(log_glob.replace("%j", job))
    if not path.exists():                       # %j substitution is the normal case
        cand = sorted(glob.glob(log_glob.replace("%j", "*")))
        if not cand:
            return None
        path = Path(cand[-1])
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return None
    folds = len(re.findall(r"CV fold \d+:", text))
    params = re.search(r"([\d,]+) parameters", text)
    return folds, (params.group(1) if params else "")


def scope(drug):
    """The manifests write 'ALL(multidrug)' for joint jobs; 12 columns of that
    on every row is noise when the run name already says multidrug."""
    return "joint" if drug.startswith("ALL") else drug


ARCHS = {"__setfusion": "sf", "__late_fusion": "lf", "__cisfusion": "cis",
         "__mdcnn": "md"}
DROP_ARCH = True          # set per view in main(); see short()


def arch_of(run):
    return next((s for s in ARCHS if run.endswith(s)), "")


def short(run):
    """'setfusion_scaling/a4_d512_multidrug_dna__setfusion' -> 'a4_d512_multidrug_dna'.

    The arch suffix is dropped only when the whole view is ONE architecture.
    Dropping it unconditionally silently merged joint_capacity's cisfusion and
    late_fusion arms under one label — two different models printed under the
    same name, which is worse than a longer line.
    """
    cell = run.split("/", 1)[-1]
    suffix = arch_of(run)
    if not suffix:
        return cell
    return cell[: -len(suffix)] + ("" if DROP_ARCH else f" [{ARCHS[suffix]}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patterns", nargs="*", help="substring(s) of the run name")
    ap.add_argument("-d", "--done", action="store_true", help="list finished cells too")
    ap.add_argument("--days", type=float, default=RECENT_DAYS,
                    help=f"only manifests this recent (default: {RECENT_DAYS})")
    args = ap.parse_args()

    live = squeue_jobs()
    rows = manifest_rows(args.patterns, args.days)
    if not rows:
        print(f"No submissions in the last {args.days:g} days"
              + (f" matching {args.patterns}" if args.patterns else "") + ".")
        return
    # with no pattern, narrow to the sweeps that actually have jobs queued
    if not args.patterns:
        active = {r["run"].split("/")[0] for r in rows if r["job"] in live}
        if active:
            rows = [r for r in rows if r["run"].split("/")[0] in active]

    global DROP_ARCH
    DROP_ARCH = len({arch_of(r["run"]) for r in rows}) <= 1

    done, running, pending, failed = [], [], [], []
    for r in rows:
        res = result_json(r["run"], r["drug"])
        if res is not None:
            r["metric"] = metric(res)
            done.append(r)
        elif r["job"] in live:
            r["state"] = live[r["job"]]
            (running if r["state"][0] == "R" else pending).append(r)
        else:
            failed.append(r)

    groups = sorted({r["run"].split("/")[0] for r in rows})
    total = len(rows)
    print(f"{', '.join(groups)} — {len(done)} done · {len(running)} running · "
          f"{len(pending)} pending · {len(failed)} FAIL   ({total} jobs)")
    bar = int(30 * len(done) / total) if total else 0
    print("[" + "#" * bar + "." * (30 - bar) + f"] {100*len(done)//max(total,1)}%\n")

    if running:
        print("RUNNING")
        for r in sorted(running, key=lambda r: short(r["run"])):
            _, elapsed, left, node = r["state"]
            p = progress(r["log"], r["job"])
            note = f"fold {p[0]}/5" if p else "starting"
            if p and p[1]:
                note += f", {p[1]} params"
            print(f"  {short(r['run']):<42} {scope(r['drug'])[:12]:<13} {elapsed:>9} elapsed "
                  f"({left:>9} left)  {note}")
    if pending:
        print(f"\nPENDING ({len(pending)})")
        for r in sorted(pending, key=lambda r: short(r["run"])):
            print(f"  {short(r['run']):<42} {scope(r['drug'])[:12]:<13} {r['state'][3]}")
    if failed:
        print(f"\nFAIL — submitted, not queued, no result json ({len(failed)})")
        for r in sorted(failed, key=lambda r: short(r["run"])):
            log = r["log"].replace("%j", r["job"])
            hit = sorted(glob.glob(log)) or sorted(glob.glob(r["log"].replace("%j", "*")))
            print(f"  {short(r['run']):<42} {scope(r['drug'])[:12]:<13} "
                  f"{Path(hit[-1]).name if hit else '(no log)'}")
    if done:
        print(f"\nDONE ({len(done)})" + ("" if args.done else " — -d to list"))
        if args.done:
            for r in sorted(done, key=lambda r: (-(r["metric"] or (0,))[0],
                                                 short(r["run"]))):
                if not r["metric"]:
                    print(f"  {short(r['run']):<42} {scope(r['drug'])[:12]:<13} (unreadable json)")
                    continue
                cv, n, hrs = r["metric"]
                print(f"  {short(r['run']):<42} {scope(r['drug'])[:12]:<13} "
                      f"CV={cv:.4f}  {(n or 0)/1e6:5.2f}M  {hrs:5.1f}h")
    if not live:
        print("\n(squeue unavailable — running/pending shown as FAIL; ignore this "
              "column until it is back)")


if __name__ == "__main__":
    main()
