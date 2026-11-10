"""Supervisor for the locus_sets sweep, adapted from locusfusion_v3/babysit.py.
Runs as one long CPU SLURM job (babysit.sh); every 10 minutes it:

  1. data: once the 14-locus extraction array (EXTRACT_JOB) has left the queue,
     runs locus_sets/extract/validate_extracted.py. Failed loci are re-extracted
     once; on a pass the 14 symlinks in locus_sets/aligned are repointed from
     rebuilt/ (insertion bug, wrong pepQ/mmpS5 windows) to extracted/. Nothing
     lab64 is submitted before that. A second failure -> ALERT.md, lab64 blocked,
     everything else carries on.
  2. canaries gate their groups. A group is submitted only after its canary cell
     wrote a result; its memory/time come from the canary's sacct MaxRSS/Elapsed
     (the canary is the group's biggest cell: all_modalities, largest drug or
     the joint model). A canary that fails for a non-resource reason halts its
     groups (ALERT.md); the rest of the sweep carries on.
  3. retries: a cell that left the queue without its result JSON is resubmitted
     up to MAX_TRIES times; OUT_OF_MEMORY -> x1.5 memory, TIMEOUT -> x2 time.
     run_experiment.py exits 0 when a drug FAILS, so the JSON is the only proof.
  4. writes STATUS.md; when every cell is done or given up, executes
     notebooks/analysis/locus_sets.ipynb and writes DONE.

State lives in state.json; restarting resumes. To stop: scancel the babysit job
(queued cells keep running; restart later to pick up where it left off).
"""
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = Path("/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis")
SNAP = W / "locus_sets_snapshot" / "biophysical-fusion"
DATA = W / "data_snapshot"
GENO = W / "locus_sets" / "aligned"
EXTRACT = W / "locus_sets" / "extract"
EXTRACTED = W / "locus_sets" / "extracted"
EXP = W / "biophysical-fusion" / "results" / "experiments"
NOTEBOOK = W / "biophysical-fusion" / "notebooks" / "analysis" / "locus_sets.ipynb"
PY = Path.home() / ".conda/envs/abr_env/bin/python"
JUPYTER = Path.home() / ".conda/envs/abr_env/bin/jupyter"
STATE = HERE / "state.json"
EXTRACT_JOB = "66380513"
LAB14 = ["atpE", "ddn", "fbiA", "fbiB", "fbiC", "fgd1", "mmpL5", "mmpS5", "pepQ", "rplC",
         "Rv0678", "Rv1979c", "Rv2983", "tsnR"]          # = extract_14.sbatch GENES order

DRUGS = ["ISONIAZID", "RIFAMPICIN", "ETHAMBUTOL", "PYRAZINAMIDE", "STREPTOMYCIN",
         "KANAMYCIN", "AMIKACIN", "CAPREOMYCIN", "LEVOFLOXACIN", "MOXIFLOXACIN",
         "ETHIONAMIDE"]
MD = "MULTIDRUG"
MODS5 = ["dna", "dna_protein", "dna_biophysical", "dna_regulatory", "all_modalities"]
LF = [f"{m}_unk__locusfusion_v2" for m in MODS5]
CNN = [f"{m}__mdcnn" for m in MODS5]
LOCI19 = ["--loci", "eis", "embA", "embB", "embC", "ethA", "ethR", "fabG1", "gid", "gyrA",
          "gyrB", "inhA", "katG", "pncA", "rpoB", "rpoC", "rpsL", "rrl", "rrs", "tlyA"]
COMMON = ["--project-dir", str(SNAP), "--weights-dir", str(W / "model_weights_locus_sets"),
          "--env", f"ABR_GENOTYPE_DIR={DATA}/genomic_data/aligned",
          "--env", f"ABR_PHENOTYPE_CSV={DATA}/phenotype_data/master_resistance_table.csv",
          "--env", f"ABR_GENBANK={DATA}/genbank/genome.gbff",
          "--genotype-dir", str(GENO),
          "--epochs", "300", "--patience", "30", "--min-epochs", "50", "--save-weights", "best",
          "--cpus", "4", "--gpus", "1", "--constraint", "vram23", "--delay", "0.5"]
SD_SPLIT = ["--split-dir", "results/experiments/lineage_split/splits"]
MD_SPLIT = ["--split-dir", "results/experiments/locus_sets/splits"]
MAX_TRIES = 3
MEM_CAP_GB, TIME_CAP_H = 370, 119            # biggest vram23 nodes ~376 GB; gpu max 5 days

# Canaries: (name) -> (cell, ...). Cells are (arm, exp, drug).
# who canaries were submitted by hand (submit.sh canary_who, jobs 66380594-97);
# the lab64 canaries are the grid's own biggest cells.
CANARIES = {
    "md":      ("canary_md", "all_modalities_unk__locusfusion_v2", MD),
    "md_cnn":  ("canary_md", "all_modalities__mdcnn", MD),
    "who_lf":  ("canary_who_sd", "all_modalities_unk__locusfusion_v2", "ISONIAZID"),
    "who_cnn": ("canary_who_sd", "all_modalities__mdcnn", "ISONIAZID"),
    "lab_lf":  ("lab_sd_lf", "all_modalities_unk__locusfusion_v2", "RIFAMPICIN"),
    "lab_cnn": ("lab_sd_cnn", "all_modalities__mdcnn", "RIFAMPICIN"),
}
PRESUBMITTED = {  # cell key -> job id (submitted before the babysitter existed)
    "canary_md|all_modalities_unk__locusfusion_v2|MULTIDRUG": "66380594",
    "canary_md|all_modalities__mdcnn|MULTIDRUG": "66380595",
    "canary_who_sd|all_modalities_unk__locusfusion_v2|ISONIAZID": "66380596",
    "canary_who_sd|all_modalities__mdcnn|ISONIAZID": "66380597",
}

# arm -> dict(prefix, exps, drugs, flags, kind, needs_data, gate=[canary names],
#             size=(canary name, mem factor) or fixed (mem_gb, hours))
ARMS = {
    "canary_md":     dict(prefix="locus_sets/canary_who_t12_", exps=[LF[-1], CNN[-1]], drugs=[MD],
                          flags=["--multidrug", "--locus-set", "who_t12"] + MD_SPLIT,
                          fixed=(240, 72), gate=[], data=False),
    "canary_who_sd": dict(prefix="locus_sets/canary_who_t12_sd_", exps=[LF[-1], CNN[-1]],
                          drugs=["ISONIAZID"], flags=["--locus-set", "who_t12"] + SD_SPLIT,
                          fixed=(128, 36), gate=[], data=False),
    "who_sd_lf":  dict(prefix="locus_sets/who_t12_sd_", exps=LF, drugs=DRUGS,
                       flags=["--locus-set", "who_t12"] + SD_SPLIT, size=("who_lf", 1.0),
                       gate=["who_lf"], data=False),
    "who_sd_cnn": dict(prefix="locus_sets/who_t12_sd_", exps=CNN, drugs=DRUGS,
                       flags=["--locus-set", "who_t12"] + SD_SPLIT, size=("who_cnn", 1.0),
                       gate=["who_cnn"], data=False),
    "md_bigtb19": dict(prefix="locus_sets/bigtb19_", exps=LF + CNN, drugs=[MD],
                       flags=["--multidrug"] + LOCI19 + MD_SPLIT, fixed=(128, 48),
                       gate=["md", "md_cnn"], data=False),
    "md_who_lf":  dict(prefix="locus_sets/who_t12_", exps=LF, drugs=[MD],
                       flags=["--multidrug", "--locus-set", "who_t12"] + MD_SPLIT,
                       size=("md", 1.0), gate=["md"], data=False),
    "md_who_cnn": dict(prefix="locus_sets/who_t12_", exps=CNN, drugs=[MD],
                       flags=["--multidrug", "--locus-set", "who_t12"] + MD_SPLIT,
                       size=("md_cnn", 1.0), gate=["md_cnn"], data=False),
    # lab64: the RIFAMPICIN all_modalities cell is the canary; the rest wait on it
    "lab_sd_lf":  dict(prefix="locus_sets/lab64_sd_", exps=LF, drugs=DRUGS,
                       flags=["--locus-set", "lab64"] + SD_SPLIT, size=("lab_lf", 1.0),
                       canary_fixed=(240, 72), gate=["lab_lf"], data=True),
    "lab_sd_cnn": dict(prefix="locus_sets/lab64_sd_", exps=CNN, drugs=DRUGS,
                       flags=["--locus-set", "lab64"] + SD_SPLIT, size=("lab_cnn", 1.0),
                       canary_fixed=(240, 72), gate=["lab_cnn"], data=True),
    "md_lab_lf":  dict(prefix="locus_sets/lab64_", exps=LF, drugs=[MD],
                       flags=["--multidrug", "--locus-set", "lab64"] + MD_SPLIT,
                       size=("md", 1.15), gate=["md"], data=True),
    "md_lab_cnn": dict(prefix="locus_sets/lab64_", exps=CNN, drugs=[MD],
                       flags=["--multidrug", "--locus-set", "lab64"] + MD_SPLIT,
                       size=("md_cnn", 1.15), gate=["md_cnn"], data=True),
    "lasso_who":  dict(kind="lasso", set="who_t12", drugs=DRUGS, exps=["lasso"], gate=[], data=False),
    "lasso_lab":  dict(kind="lasso", set="lab64", drugs=DRUGS, exps=["lasso"], gate=[], data=True),
}
LASSO_MEM_GB = 64


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] {msg}", flush=True)


def key(c):
    return "|".join(c)


def cells():
    for arm, a in ARMS.items():
        for exp in a["exps"]:
            for drug in a["drugs"]:
                yield (arm, exp, drug)


def done(c):
    arm, exp, drug = c
    a = ARMS[arm]
    if a.get("kind") == "lasso":
        p = EXP / "locus_sets" / f"lasso_onehot_{a['set']}" / f"{drug}__dna-onehot-{a['set']}.json"
        files, field = [p], "cv_auc_mean"
    elif drug == MD:
        files = (EXP / f"{a['prefix']}multidrug_{exp}").glob("multidrug__*.json")
        field = "cv_macro_auc_mean"
    else:
        files, field = (EXP / f"{a['prefix']}{exp}").glob(f"{drug}__*.json"), "cv_auc_mean"
    for p in files:
        try:
            if p.is_file() and field in json.loads(p.read_text()):
                return True
        except Exception:
            pass
    return False


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def live_jobs():
    out = sh(["squeue", "-h", "-u", os.environ["USER"], "-o", "%i"])
    if out.returncode != 0:
        return None
    return set(out.stdout.split())


def sacct(job):
    """-> (state, max_rss_gb, elapsed_h) for a finished job, from sacct."""
    out = sh(["sacct", "-n", "-P", "-j", str(job), "-o", "JobID,State,MaxRSS,ElapsedRaw"])
    state, rss, el = "UNKNOWN", 0.0, 0.0
    for line in out.stdout.splitlines():
        jid, st, mr, er = (line.split("|") + ["", "", "", ""])[:4]
        if jid == str(job):
            state, el = st.split()[0] if st else "UNKNOWN", float(er or 0) / 3600
        if mr:
            m = re.match(r"([\d.]+)([KMGT]?)", mr)
            if m:
                f = {"K": 1 / 2**20, "M": 1 / 2**10, "G": 1, "T": 2**10, "": 1 / 2**30}[m.group(2)]
                rss = max(rss, float(m.group(1)) * f)
    return state, rss, el


def log_reason(log_path, job):
    p = Path(str(log_path).replace("%j", str(job))) if log_path else None
    if not p or not p.is_file():
        return ""
    text = p.read_text(errors="replace")
    m = re.search(r"FAILED: (.*)", text)
    if m:
        return m.group(1)[:300]
    for pat in ("CUDA out of memory", "oom-kill", "Killed", "Traceback"):
        if pat in text:
            return pat
    return text.strip().splitlines()[-1][:200] if text.strip() else "empty log"


def fmt_time(h):
    h = min(max(h, 1), TIME_CAP_H)
    return f"{int(h // 24)}-{int(h % 24):02d}:00:00"


def resources(arm, c, state):
    """(mem_gb, hours) for a cell's next submission, before retry bumps."""
    a = ARMS[arm]
    if "fixed" in a:
        return a["fixed"]
    if a.get("kind") == "lasso":
        return (LASSO_MEM_GB, 12)
    name, fac = a["size"]
    if CANARIES[name] == c:                       # the cell IS its group's canary
        return a["canary_fixed"]
    cs = state["canary"].get(name)
    mem = math.ceil(max(48, 1.35 * fac * cs["rss"]) / 8) * 8
    # The canary is its group's biggest cell, so what it succeeded with is enough
    # for the group: never ask for more (above ~250 GB only 4 vram23 nodes exist).
    if fac <= 1.0:
        mem = min(mem, cs["req"])
    return (min(mem, MEM_CAP_GB), max(12, 2.5 * fac * cs["hours"]))


def submit_net(arm, exp, drugs, mem, hours):
    a = ARMS[arm]
    pick = [] if drugs == [MD] else ["--drugs", *drugs]
    cmd = [str(PY), "scripts/sbatch_all_runs.py", "--experiments", exp, *pick,
           "--run-prefix", a["prefix"], *COMMON, *a["flags"],
           "--mem", f"{mem}G", "--time", fmt_time(hours)]
    out = sh(cmd, cwd=SNAP)
    got = {}
    for m in re.finditer(r"\] (\S+) on (\S+): submitted (\d+)", out.stdout):
        got[MD if m.group(2) == "multidrug" else m.group(2)] = m.group(3)
    logs = {}
    man = re.search(r"Tracking: (\S+)", out.stdout)
    if man:
        try:
            for j in json.loads(Path(man.group(1)).read_text())["submitted_jobs"]:
                logs[MD if j["drug"].startswith("ALL") else j["drug"]] = j["log"]
        except Exception:
            pass
    if out.returncode != 0 or len(got) < len(drugs):
        log(f"submit {arm}/{exp}: {len(got)}/{len(drugs)}; stderr: {out.stderr.strip()[-300:]}")
    return {d: (j, logs.get(d, "")) for d, j in got.items()}


def submit_lasso(arm, drug, mem):
    s = ARMS[arm]["set"]
    logp = str(W / "locus_sets" / "logs" / f"lasso-{s}-{drug}-%j.out")
    out = sh(["sbatch", f"--chdir={SNAP}", f"--job-name=lasso-locusset-{s}-{drug}",
              f"--mem={mem}G", f"--output={logp}",
              str(SNAP / "results/experiments/locus_sets/sbatch_lasso_locus_sets.sh"), drug, s])
    m = re.search(r"Submitted batch job (\d+)", out.stdout)
    if not m:
        log(f"submit lasso {s}/{drug} failed: {out.stderr.strip()[-300:]}")
        return None
    return m.group(1), logp


# ------------------------------------------------------------------ data gate
def data_step(state, live):
    d = state["data"]
    if d["status"] in ("ready", "blocked") or DRY:
        return
    if any(j.startswith(EXTRACT_JOB) for j in live) or (d.get("retry_job") and
                                                         any(j.startswith(d["retry_job"]) for j in live)):
        return
    v = sh([str(PY), str(EXTRACT / "validate_extracted.py"), "--dir", str(EXTRACTED)])
    (EXTRACT / "validation.log").write_text(v.stdout + v.stderr)
    if v.returncode == 0:
        for g in LAB14:
            link = GENO / f"{g}.fasta"
            tmp = GENO / f".{g}.fasta.tmp"
            if tmp.exists() or tmp.is_symlink():
                tmp.unlink()
            tmp.symlink_to(EXTRACTED / f"{g}.fasta")
            os.replace(tmp, link)                 # atomic swap of the symlink
        d["status"] = "ready"
        log("data: 14 extracted loci validated; locus_sets/aligned repointed to extracted/")
        return
    failed = [g for g in LAB14 if re.search(rf"^FAIL {re.escape(g)}:", v.stdout, re.M)]
    if not d.get("retried") and failed:
        idx = ",".join(str(LAB14.index(g)) for g in failed)
        out = sh(["sbatch", f"--array={idx}%3", str(EXTRACT / "extract_14.sbatch")])
        m = re.search(r"Submitted batch job (\d+)", out.stdout)
        d.update(retried=True, retry_job=m.group(1) if m else None)
        log(f"data: validation failed for {failed}; re-extracting (job {d['retry_job']})")
        return
    d["status"] = "blocked"
    alert(state, f"lab64 data validation failed: {failed or 'see validation.log'}. "
                 f"lab64 cells are blocked; who_t12/bigtb19 continue. "
                 f"Fix, set state.json data.status='pending', restart.")


def alert(state, msg):
    state.setdefault("alerts", []).append(f"{datetime.now():%Y-%m-%d %H:%M} {msg}")
    (HERE / "ALERT.md").write_text("# locus_sets sweep alerts\n\n" +
                                   "\n".join(f"- {a}" for a in state["alerts"]) + "\n")
    log("ALERT: " + msg)


# ------------------------------------------------------------------ status
def write_status(state, live):
    rows = []
    for arm, a in ARMS.items():
        cs = [(arm, e, d) for e in a["exps"] for d in a["drugs"]]
        n_done = sum(done(c) for c in cs)
        n_live = sum(state["cells"].get(key(c), {}).get("job") in live for c in cs if not done(c))
        n_fail = sum(bool(state["cells"].get(key(c), {}).get("gave_up")) for c in cs)
        gate = ",".join(a["gate"]) or "-"
        rows.append(f"| {arm} | {a.get('prefix', 'lasso_onehot_' + a.get('set', ''))} | "
                    f"{n_done}/{len(cs)} | {n_live} | {n_fail} | {gate} |")
    can = [f"- {n}: {v}" for n, v in state["canary"].items()]
    txt = [f"# locus_sets sweep status ({datetime.now():%Y-%m-%d %H:%M})", "",
           f"data (14 re-extracted loci): **{state['data']['status']}**", "",
           "| arm | folder | done | in queue | gave up | gated on |",
           "|---|---|---|---|---|---|", *rows, "", "## Canaries", *(can or ["- none finished yet"]), ""]
    retried = [f"- {k}: {'; '.join(v['reasons'])}" for k, v in state["cells"].items() if v.get("reasons")]
    if retried:
        txt += ["## Failures (resubmitted or given up)", *retried, ""]
    if state.get("alerts"):
        txt += ["## Alerts (see ALERT.md)", *[f"- {a}" for a in state["alerts"]], ""]
    (HERE / "STATUS.md").write_text("\n".join(txt) + "\n")


def run_notebook():
    out = sh([str(JUPYTER), "nbconvert", "--to", "notebook", "--execute", "--inplace",
              "--ExecutePreprocessor.timeout=1800", str(NOTEBOOK)], cwd=NOTEBOOK.parent)
    log("notebook: " + ("executed" if out.returncode == 0 else out.stderr.strip()[-400:]))


# ------------------------------------------------------------------ main loop
def tick(state):
    live = live_jobs()
    if live is None:
        return False
    data_step(state, live)
    # canary bookkeeping
    for name, c in CANARIES.items():
        if name in state["canary"] or name in state["halted"]:
            continue
        s = state["cells"].get(key(c), {})
        if done(c) and s.get("job"):
            st, rss, hrs = sacct(s["job"])
            a = ARMS[c[0]]
            req = math.ceil((a.get("canary_fixed") or a["fixed"])[0] * s.get("bump_mem", 1.0) / 8) * 8
            state["canary"][name] = {"job": s["job"], "rss": round(rss, 1), "hours": round(hrs, 2),
                                     "req": req}
            log(f"canary {name} done: MaxRSS {rss:.1f} GB, {hrs:.1f} h")
    to_submit = {}
    for c in cells():
        arm, exp, drug = c
        a = ARMS[arm]
        if done(c):
            continue
        s = state["cells"].setdefault(key(c), {"tries": 0, "reasons": [], "bump_mem": 1.0,
                                               "bump_time": 1.0})
        if s.get("gave_up") or (s.get("job") and s["job"] in live):
            continue
        if s.get("job"):                          # left the queue without a result
            st, rss, hrs = sacct(s["job"])
            reason = f"job {s['job']}: {st}" + (f" ({log_reason(s.get('log'), s['job'])})"
                                                  if st not in ("OUT_OF_MEMORY", "TIMEOUT") else "")
            s["reasons"].append(reason)
            s["job"] = None
            log(f"FAILED {key(c)}: {reason}")
            resource = st in ("OUT_OF_MEMORY", "TIMEOUT") or "oom-kill" in reason or "Killed" in reason
            if st == "OUT_OF_MEMORY" or "oom-kill" in reason or "Killed" in reason:
                s["bump_mem"] *= 1.5
            if st == "TIMEOUT":
                s["bump_time"] *= 2
            for name, cc in CANARIES.items():
                if cc == c and not resource:
                    state["halted"].append(name)
                    alert(state, f"canary {name} ({key(c)}) failed: {reason}. Its groups are "
                                 f"halted; fix the snapshot, clear 'halted' in state.json, restart.")
            if s["tries"] >= MAX_TRIES:
                s["gave_up"] = True
                continue
        # gates
        if a["data"] and state["data"]["status"] != "ready":
            continue
        is_canary = c in CANARIES.values()
        names = [n for n, cc in CANARIES.items() if cc == c]
        if is_canary and any(n in state["halted"] for n in names):
            continue
        if not is_canary and any(g in state["halted"] or g not in state["canary"] for g in a["gate"]):
            continue
        if not is_canary and "size" in a and a["size"][0] not in state["canary"]:
            continue
        mem, hrs = resources(arm, c, state)
        mem = min(MEM_CAP_GB, math.ceil(mem * s["bump_mem"] / 8) * 8)
        hrs = min(TIME_CAP_H, hrs * s["bump_time"])
        to_submit.setdefault((arm, exp, mem, round(hrs)), []).append(drug)
    for (arm, exp, mem, hrs), drugs in to_submit.items():
        if DRY:
            log(f"DRY would submit {arm}/{exp} x{len(drugs)} {drugs[:3]}... ({mem}G, {fmt_time(hrs)})")
            continue
        if ARMS[arm].get("kind") == "lasso":
            got = {}
            for d in drugs:
                r = submit_lasso(arm, d, mem)
                if r:
                    got[d] = r
        else:
            got = submit_net(arm, exp, drugs, mem, hrs)
        for d, (jid, lg) in got.items():
            s = state["cells"][key((arm, exp, d))]
            s.update(job=jid, log=lg, tries=s["tries"] + 1)
            log(f"submitted {arm}/{exp}/{d} -> {jid} ({mem}G, {fmt_time(hrs)}, try {s['tries']})")
    if DRY:
        return True
    STATE.write_text(json.dumps(state, indent=1))
    write_status(state, live_jobs() or set())
    return True


def finished(state):
    blocked = lambda c: ((ARMS[c[0]]["data"] and state["data"]["status"] == "blocked")
                         or any(g in state["halted"] for g in ARMS[c[0]]["gate"])
                         or any(n in state["halted"] for n, cc in CANARIES.items() if cc == c))
    return all(done(c) or state["cells"].get(key(c), {}).get("gave_up") or blocked(c)
               for c in cells())


def main(once=False):
    state = (json.loads(STATE.read_text()) if STATE.exists() else
             {"cells": {}, "canary": {}, "halted": [], "data": {"status": "pending"}})
    for k, j in PRESUBMITTED.items():
        state["cells"].setdefault(k, {"tries": 1, "reasons": [], "bump_mem": 1.0,
                                      "bump_time": 1.0, "job": j, "log": ""})
    last_nb = time.time()
    while True:
        try:
            ok = tick(state)
        except Exception:
            log("tick error:\n" + traceback.format_exc())
            ok = False
        if once:
            return
        if ok and finished(state):
            run_notebook()
            (HERE / "DONE").write_text(f"{datetime.now():%Y-%m-%d %H:%M}\n")
            log("all cells done, given up, or blocked; exiting")
            return
        if time.time() - last_nb > 6 * 3600:
            run_notebook()
            last_nb = time.time()
        time.sleep(600)


DRY = "--dry" in sys.argv      # with --once: print decisions, submit/write nothing

if __name__ == "__main__":
    sys.exit(main(once="--once" in sys.argv or DRY))
