"""Supervisor for the locusfusion_v3 sweep. Runs as one long CPU SLURM job
(babysit.sh) and, every 10 minutes:

  1. grid: submits every cell that has no result and no live job, in priority
     order (19 loci, per-drug, ablation, seeds, partitions). Not gated.
  2. canary: LEVOFLOXACIN at 19 loci, all modalities, is watched. If it fails
     with anything but a missing-file (mount) error, ALERT.md is written and no
     further submissions or retries happen -- the queued jobs likely share the
     bug, so cancel them (scancel), fix the snapshot, and restart.
  3. retries: a job that left the queue without writing its result JSON is
     resubmitted, up to MAX_TRIES per cell. run_experiment.py exits 0 when a
     drug FAILS (SLURM says COMPLETED), so the result JSON is the only proof.
  4. writes STATUS.md, and reruns compare.py -> SUMMARY.md every hour and at
     the end.

Jobs run from the frozen code snapshot (v3_snapshot/) against the local data
copy (data_snapshot/), so neither repo edits nor the /project mount can touch
them. State lives in state.json; restarting this script resumes.
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = Path("/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis")
SNAP = W / "v3_snapshot" / "biophysical-fusion"
# multi-drug jobs need run_multidrug.py --lf-unknown-symbols (added 2026-10-01
# after SNAP was frozen): same code plus that one change, see multidrug.patch
SNAP_MD = W / "v3md_snapshot" / "biophysical-fusion"
MD = "MULTIDRUG"                                   # the one "drug" of a joint cell
DATA = W / "data_snapshot"
WEIGHTS = W / "model_weights_v3"
EXP = W / "biophysical-fusion" / "results" / "experiments"
PY = Path.home() / ".conda/envs/abr_env/bin/python"
STATE = HERE / "state.json"

DRUGS = ["ISONIAZID", "RIFAMPICIN", "ETHAMBUTOL", "PYRAZINAMIDE", "STREPTOMYCIN",
         "KANAMYCIN", "AMIKACIN", "CAPREOMYCIN", "LEVOFLOXACIN", "MOXIFLOXACIN",
         "ETHIONAMIDE"]
MODS5 = ["dna", "dna_protein", "dna_biophysical", "dna_regulatory", "all_modalities"]
LOCI19 = ["--loci", "eis", "embA", "embB", "embC", "ethA", "ethR", "fabG1", "gid",
          "gyrA", "gyrB", "inhA", "katG", "pncA", "rpoB", "rpoC", "rpsL", "rrl",
          "rrs", "tlyA"]
COMMON = ["--epochs", "300", "--patience", "30", "--min-epochs", "50",
          "--save-weights", "best", "--weights-dir", str(WEIGHTS),
          "--cpus", "4", "--gpus", "1", "--constraint", "vram23",
          "--project-dir", str(SNAP),
          "--env", f"ABR_GENOTYPE_DIR={DATA}/genomic_data/aligned",
          "--env", f"ABR_PHENOTYPE_CSV={DATA}/phenotype_data/master_resistance_table.csv",
          "--env", f"ABR_GENBANK={DATA}/genbank/genome.gbff"]
BIG = ["--mem", "128G", "--time", "36:00:00"]      # 19 loci
SMALL = ["--mem", "64G", "--time", "24:00:00"]     # per-drug loci
MAX_TRIES = 3
CANARY = ("sd19", "all_modalities_unk__locusfusion_v2", "LEVOFLOXACIN")

# (arm, run prefix, experiments, extra flags), in submission priority order
ARMS = [("sd19", "locusfusion_v3/sd19_", [f"{m}_unk__locusfusion_v2" for m in MODS5],
         LOCI19 + BIG),
        ("sd", "locusfusion_v3/sd_", [f"{m}_unk__locusfusion_v2" for m in MODS5], SMALL),
        ("ablation", "locusfusion_v3/ablation_sd19_", [f"{m}__locusfusion_v2" for m in MODS5],
         LOCI19 + BIG)]
ARMS += [(f"seed{s}", f"locusfusion_v3/s{s}_", ["all_modalities_unk__locusfusion_v2"],
          LOCI19 + BIG + ["--seed", str(s)]) for s in (1000, 2000, 3000, 4000)]
ARMS += [(f"cv{c}", f"locusfusion_v3/cv{c}_", ["all_modalities_unk__locusfusion_v2"],
          LOCI19 + BIG + ["--cv-seed", str(c)]) for c in (101, 102, 103, 104)]
# multi-drug, 19 loci, 5 modsets: one job per modset, all 11 drugs in one model
ARMS += [("md", "locusfusion_v3/md_", [f"{m}_unk__locusfusion_v2" for m in MODS5],
          LOCI19 + ["--mem", "128G", "--time", "36:00:00", "--cpus", "6",
                    "--multidrug", "--project-dir", str(SNAP_MD)])]
ARM = {a[0]: a for a in ARMS}


def drugs_of(arm):
    return [MD] if arm == "md" else DRUGS


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] {msg}", flush=True)


def cells():
    for arm, prefix, exps, _ in ARMS:
        for exp in exps:
            for drug in drugs_of(arm):
                yield (arm, exp, drug)


def key(c):
    return "|".join(c)


def done(c):
    arm, exp, drug = c
    if drug == MD:
        folder, pat, field = EXP / f"{ARM[arm][1]}multidrug_{exp}", "multidrug__*.json", "cv_macro_auc_mean"
    else:
        folder, pat, field = EXP / f"{ARM[arm][1]}{exp}", f"{drug}__*.json", "cv_auc_mean"
    for p in folder.glob(pat):
        try:
            if field in json.loads(p.read_text()):
                return True
        except Exception:
            pass
    return False


def live_jobs():
    out = subprocess.run(["squeue", "-h", "-u", os.environ["USER"], "-o", "%i"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return None                       # squeue hiccup: do nothing this round
    return set(out.stdout.split())


def failure_reason(log_path, job_id):
    p = Path(log_path.replace("%j", str(job_id)))
    if not p.is_file():
        return "no log (never started, or cancelled while pending)"
    text = p.read_text(errors="replace")
    m = re.search(r"FAILED: (.*)", text)
    if m:
        return m.group(1)[:300]
    for pat in ("CANCELLED", "TIME LIMIT", "OUT_OF_MEMORY", "oom-kill", "Killed",
                "Traceback"):
        if pat in text:
            return pat
    return "exited without a result JSON: " + text.strip().splitlines()[-1][:200] if text.strip() else "empty log"


def submit(arm, exp, drugs):
    """-> {drug: (job_id, log)} for what was submitted."""
    _, prefix, _, flags = ARM[arm]
    pick = [] if drugs == [MD] else ["--drugs", *drugs]
    cmd = [str(PY), "scripts/sbatch_all_runs.py", "--experiments", exp, *pick,
           "--run-prefix", prefix, *COMMON, *flags, "--delay", "0.5"]
    out = subprocess.run(cmd, cwd=SNAP, capture_output=True, text=True)
    got = {}
    for m in re.finditer(r"\] (\S+) on (\S+): submitted (\d+)", out.stdout):
        got[MD if m.group(2) == "multidrug" else m.group(2)] = m.group(3)
    manifest = re.search(r"Tracking: (\S+)", out.stdout)
    logs = {}
    if manifest:
        for j in json.loads(Path(manifest.group(1)).read_text())["submitted_jobs"]:
            logs[MD if j["drug"].startswith("ALL") else j["drug"]] = j["log"]
    if out.returncode != 0 or len(got) < len(drugs):
        log(f"submit {arm}/{exp}: {len(got)}/{len(drugs)} submitted; "
            f"stderr: {out.stderr.strip()[-300:]}")
    return {d: (jid, logs.get(d, "")) for d, jid in got.items()}


def write_status(state, live, halted):
    rows = []
    for arm, prefix, exps, _ in ARMS:
        n_done = n_live = n_fail = 0
        for exp in exps:
            for drug in drugs_of(arm):
                c = (arm, exp, drug)
                s = state["cells"].get(key(c), {})
                if done(c):
                    n_done += 1
                elif s.get("job") in live:
                    n_live += 1
                elif s.get("gave_up"):
                    n_fail += 1
        rows.append(f"| {arm} | {prefix}* | {n_done}/{len(exps) * len(drugs_of(arm))} | {n_live} | {n_fail} |")
    gave_up = [f"- {k}: {v['reasons'][-1]}" for k, v in state["cells"].items() if v.get("gave_up")]
    retried = [f"- {k}: {len(v['reasons'])} failure(s), last: {v['reasons'][-1]}"
               for k, v in state["cells"].items() if v.get("reasons") and not v.get("gave_up")]
    txt = [f"# locusfusion_v3 sweep status ({datetime.now():%Y-%m-%d %H:%M})", "",
           "HALTED -- see ALERT.md" if halted else "running", "",
           "| arm | folders | done | in queue | gave up |", "|---|---|---|---|---|", *rows, ""]
    if gave_up:
        txt += ["## Gave up (MAX_TRIES failures)", *gave_up, ""]
    if retried:
        txt += ["## Failed and resubmitted", *retried, ""]
    (HERE / "STATUS.md").write_text("\n".join(txt) + "\n")


def run_compare():
    out = subprocess.run([str(PY), str(HERE / "compare.py")], capture_output=True, text=True,
                         cwd=W / "biophysical-fusion")
    log("compare.py: " + (out.stdout.strip()[-200:] or out.stderr.strip()[-300:]))


def main(once=False):
    state = json.loads(STATE.read_text()) if STATE.exists() else {"cells": {}, "halted": False}
    last_compare = 0.0
    while True:
        live = live_jobs()
        if live is None:
            time.sleep(120)
            continue
        canary_ok = done(CANARY)
        to_submit = {}                     # (arm, exp) -> [drugs]
        for c in cells():
            if done(c):
                continue
            s = state["cells"].setdefault(key(c), {"tries": 0, "reasons": []})
            if s.get("gave_up"):
                continue
            if s.get("job") and s["job"] in live:
                continue
            if s.get("job"):              # left the queue without a result
                reason = failure_reason(s.get("log", ""), s["job"])
                s["reasons"].append(f"job {s['job']}: {reason}")
                s["job"] = None
                log(f"FAILED {key(c)}: {reason}")
                mount = "FileNotFoundError" in reason or "No such file" in reason
                if c == CANARY and not mount:
                    state["halted"] = True
                    (HERE / "ALERT.md").write_text(
                        f"# Canary failed ({datetime.now():%Y-%m-%d %H:%M})\n\n"
                        f"{key(c)}: {reason}\n\nNothing else was submitted. Fix, delete "
                        f"this file and state.json's 'halted', then restart babysit.sh.\n")
                if s["tries"] >= MAX_TRIES:
                    s["gave_up"] = True
                    continue
            if state["halted"]:
                continue
            to_submit.setdefault((c[0], c[1]), []).append(c[2])
        for (arm, exp), drugs in to_submit.items():
            for drug, (jid, lg) in submit(arm, exp, drugs).items():
                s = state["cells"][key((arm, exp, drug))]
                s.update(job=jid, log=lg, tries=s["tries"] + 1)
                log(f"submitted {arm}/{exp}/{drug} -> {jid} (try {s['tries']})")
        STATE.write_text(json.dumps(state, indent=1))
        live = live_jobs() or set()
        write_status(state, live, state["halted"])

        finished = all(done(c) or state["cells"].get(key(c), {}).get("gave_up")
                       for c in cells())
        if canary_ok and (time.time() - last_compare > 3600 or finished):
            run_compare()
            last_compare = time.time()
        if once:
            return
        if finished:
            log("all cells done or given up; exiting")
            (HERE / "DONE").write_text(f"{datetime.now():%Y-%m-%d %H:%M}\n")
            return
        if state["halted"]:
            log("halted on canary failure; exiting")
            return
        time.sleep(600)


if __name__ == "__main__":
    sys.exit(main(once="--once" in sys.argv))
