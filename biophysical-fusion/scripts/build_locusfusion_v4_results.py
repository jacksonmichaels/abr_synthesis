"""Builds notebooks/locusfusion_v4/locusfusion_v4_results.ipynb -- the v4 sweep's results, with
a coverage table of what has run and what has not.

    python scripts/build_locusfusion_v4_results.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/locusfusion_v4/locusfusion_v4_results.ipynb

Reads only what is on disk: the result JSONs under
results/experiments/locusfusion_v4/, the sbatch manifests, and `sacct` for the
state of jobs that have not written a result yet. It trains nothing, scores
nothing and needs no data mount, so it is safe to re-run at any point in the
sweep -- every section degrades to "nothing yet" while its arm is still queued.

The expected cell list comes from `locusfusion_v4.sbatch.arm_jobs`, so coverage
cannot drift from what the submitter actually submits.
"""
import json
from pathlib import Path

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": t.strip("\n").splitlines(keepends=True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})


md(r"""
# LocusFusion v4 sweep

What has finished, and what it says against v3 on the same lineage splits.

Everything here is read off disk -- result JSONs, the sbatch manifests, and
`sacct` for jobs still in the queue. Re-run the notebook to refresh it; sections
whose arm has not finished say so rather than guessing.

**The arms.** Each isolates one change, 11 drugs on the per-drug lineage splits,
protocol identical to the v3 cells (300 epochs, patience 30, min 50, Adam at
exp(-9), early stop on validation AUC, best fold scored once on TEST).

| arm | isolates |
|---|---|
| `sd19_noaug` | the token change alone, against the v3 lineage cell |
| `sd19_aug` | the augmentation, against `noaug` |
| `sd19_nodist` | the proximity channels, removed from `aug` |
| `sd19_nosyn` | whether the consequence flag is needed: synonymous tokens dropped before the cap |
| `sd19_v5` | v3's token kept whole (residue one-hot **and** consequence) + distance, synonymous dropped, no augmentation |
| `sd19_v5_nodist` | `v5` without the distance channels |
| `sd19_v5_fixdna` | `v5` on eis/embB/ethA/rrs re-extracted without the insertion bug |
| `sd19_v5_atg` | `v5` on start-codon-anchored promoter windows |
| `sd19_v5_fixed` | `v5` on both repaired inputs |
| `md_*` | the same four as one multi-drug model (11 queries, 11 proximity channels, its own split) |
| `tune` | PZA only, augmentation fraction x lambda |
| `canary` | PZA plumbing check at frac 0.5 / lambda 0.5 -- not a result |
""")

code(r"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

PROJECT = next(p for p in [Path.cwd(), *Path.cwd().parents,
                           Path("/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion")]
               if (p / "training" / "checkpoint.py").exists())
sys.path.insert(0, str(PROJECT))

from locusfusion_v4.sbatch import DRUGS, arm_jobs

pd.set_option("display.width", 240, "display.max_columns", 80, "display.max_rows", 200)

V4 = PROJECT / "results" / "experiments" / "locusfusion_v4"
V3_CELL = (PROJECT / "results" / "experiments" / "lineage_split"
           / "sd19_all_modalities_unk__locusfusion_v2")
ARMS = ["sd19_noaug", "sd19_aug", "sd19_nodist", "sd19_nosyn",
        "sd19_v5", "sd19_v5_nodist", "sd19_v5_fixdna", "sd19_v5_atg", "sd19_v5_fixed",
        "md_noaug", "md_aug", "md_nodist", "md_nosyn"]
SHORT = {a: a.replace("sd19_", "") for a in ARMS}

# what SHOULD exist, straight from the submitter
expected = {}      # job name -> (arm, drug, json path)
for arm in ["canary", "tune", "noaug", "aug", "nodist", "nosyn", "v5", "v5_nodist",
            "v5_fixdna", "v5_atg", "v5_fixed",
            "md_noaug", "md_aug", "md_nodist", "md_nosyn"]:
    for name, sub, args, kind in arm_jobs(arm, 0.5, 0.5):
        drug = args[args.index("--drugs") + 1] if "--drugs" in args else "-"
        tag = "MULTIDRUG" if drug == "all" else drug
        expected[name] = (sub, tag, V4 / sub / f"{tag}.json")

# what the scheduler says about the jobs that were submitted
submitted = {}     # job name -> job id
for mp in sorted((V4 / "logs").glob("manifest_*.json")):
    for j in json.loads(mp.read_text()).get("jobs", []):
        submitted[j["name"]] = j.get("job_id")

state = {}
if submitted:
    ids = ",".join(i for i in submitted.values() if i)
    try:
        q = subprocess.run(["sacct", "-j", ids, "-X", "-n", "-P",
                            "--format=JobID,JobName,State"],
                           capture_output=True, text=True, timeout=60)
        for line in q.stdout.strip().splitlines():
            jid, jname, st = (line.split("|") + ["", ""])[:3]
            state[jname] = st.split()[0]
    except Exception as e:
        print(f"sacct unavailable ({e}); falling back to result files only")

print(f"{len(expected)} cells expected, "
      f"{sum(n in submitted for n in expected)} submitted, "
      f"{sum(p.exists() for _, _, p in expected.values())} results on disk")

# the token-cache job writes caches, not a result JSON
for n in submitted:
    if n not in expected:
        print(f"{n}: {state.get(n, 'unknown')} (job {submitted[n]})")
cache = Path("/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/token_cache_v4")
if cache.exists():
    npz = sorted(cache.glob("tokens_*.npz"))
    print(f"token caches: {len(npz)} files, "
          f"{sum(f.stat().st_size for f in npz) / 1e9:.1f} GB")
""")

md(r"""
## Coverage

`done` = a result JSON is on disk. Anything else is the scheduler's word for the
job: `RUNNING`, `PENDING`, `FAILED`, `TIMEOUT`, `CANCELLED`. `not submitted`
means no manifest mentions it.

A job that says `COMPLETED` with no JSON is a silent failure -- the /project
mount lets those exit 0 -- and shows as `COMPLETED (no json)`.
""")

code(r"""
def status(name):
    sub, tag, p = expected[name]
    if p.exists():
        return "done"
    if name not in submitted:
        return "not submitted"
    st = state.get(name, "unknown")
    return "COMPLETED (no json)" if st == "COMPLETED" else st


rows = []
for name, (sub, tag, p) in expected.items():
    arm = "tune" if sub.startswith("tune/") else sub
    cell = sub.split("/", 1)[1] if sub.startswith("tune/") else tag
    rows.append({"arm": arm, "cell": cell, "job": name, "job id": submitted.get(name),
                 "status": status(name)})
cov = pd.DataFrame(rows)

sd = cov[cov.arm.str.startswith("sd19_")].pivot(index="arm", columns="cell", values="status")
print("single-drug arms, 11 drugs each")
display(sd.reindex(index=[a for a in ARMS if a in sd.index],
                   columns=[d for d in DRUGS if d in sd.columns]).fillna("-"))

other = cov[~cov.arm.str.startswith("sd19_")]
print("multi-drug, tune and canary")
display(other.set_index(["arm", "cell"])[["job id", "status"]])

print("\n" + cov.status.value_counts().to_string())
todo = int((cov.status != "done").sum())
print(f"\n{len(cov) - todo} of {len(cov)} cells done, {todo} still to run")
""")

md(r"""
## Load every result

One row per (arm, drug): the CV mean over folds and the single TEST score of the
best fold's model, beside the v3 lineage cell for the same drug.
""")

code(r"""
def probe_row(j, drug):
    pr = (j.get("probe") or {})
    out = {}
    for k in ("1", "2"):
        s = (pr.get("test_plus_k") or {}).get(k, {}).get(drug)
        if isinstance(s, dict):
            out[f"probe k={k} |dlogit|"] = s["mean_abs_dlogit"]
            out[f"probe k={k} |dP(R)|"] = s["mean_abs_dP_R"]
    h = (pr.get("h37rv_background") or {}).get(drug)
    if isinstance(h, dict):
        by = h.get("by_locus", {})
        out["H37Rv P(R)"] = h.get("P_R_empty")
        if by:
            m = float(np.mean([v["mean_P_R"] for v in by.values()]))
            out["scan mean P(R)"] = m
            out["scan shift"] = m - h.get("P_R_empty", np.nan)
            out["scan max |shift| by locus"] = float(np.max(
                [abs(v["mean_P_R"] - h.get("P_R_empty", np.nan)) for v in by.values()]))
            out["scan frac>0.5"] = float(np.mean([v["frac_above_0.5"] for v in by.values()]))
    return out


recs = []
for name, (sub, tag, p) in expected.items():
    if not p.exists():
        continue
    j = json.loads(p.read_text())
    arm = "tune" if sub.startswith("tune/") else sub
    cell = sub.split("/", 1)[1] if sub.startswith("tune/") else ""
    for drug, per in (j.get("per_drug") or {}).items():
        recs.append({"arm": arm, "cell": cell, "drug": drug,
                     "cv auc": per.get("cv_auc_mean"), "cv std": per.get("cv_auc_std"),
                     "test auc": (per.get("test") or {}).get("auc"),
                     "test auc_pr": (per.get("test") or {}).get("auc_pr"),
                     "test n_R": (per.get("test") or {}).get("n_R"),
                     "hours": round(j.get("seconds", np.nan) / 3600, 2),
                     **probe_row(j, drug)})
res = pd.DataFrame(recs)

v3 = []
for p in sorted(V3_CELL.glob("*.json")):
    if p.name == "weights_location.json":
        continue
    j = json.loads(p.read_text())
    v3.append({"drug": j["drug"], "v3 cv auc": j["cv_auc_mean"],
               "v3 test auc": j["test"]["auc"], "v3 test auc_pr": j["test"]["auc_pr"]})
v3 = pd.DataFrame(v3).set_index("drug")

if res.empty:
    print("no results yet")
else:
    res = res.merge(v3, left_on="drug", right_index=True, how="left")
    res["test - v3"] = res["test auc"] - res["v3 test auc"]
    display(res.set_index(["arm", "cell", "drug"]).round(4))
""")

md(r"""
## v4 against v3, per drug

TEST AUC, each v4 arm against the v3 lineage cell on the same split. `canary`
and `tune` are excluded: they are PZA-only checks.
""")

code(r"""
grid = res[res.arm.isin(ARMS)] if not res.empty else res
if grid.empty:
    print("no arm has finished yet")
else:
    wide = grid.pivot_table(index="drug", columns="arm", values="test auc")
    wide = wide.reindex(index=[d for d in DRUGS if d in wide.index],
                        columns=[a for a in ARMS if a in wide.columns])
    wide.insert(0, "v3", v3["v3 test auc"].reindex(wide.index))
    wide.loc["mean"] = wide.mean()
    display(wide.round(4))

    sdw = wide.drop(index="mean")[[c for c in wide.columns
                                   if c == "v3" or c.startswith("sd19_")]]
    if sdw.shape[1] > 1:
        ax = sdw.plot.bar(figsize=(11, 3.6), width=0.8)
        ax.set_ylabel("TEST AUC")
        ax.set_ylim(max(0.5, float(np.nanmin(sdw.values)) - 0.03), 1.0)
        ax.set_title("TEST AUC by drug and arm, lineage split")
        ax.legend(frameon=False, ncol=5, fontsize=8)
        plt.tight_layout()
        plt.show()
""")

md(r"""
### What each contrast is worth

`aug - noaug` is the augmentation, `nodist - aug` the proximity channels,
`nosyn - aug` dropping synonymous tokens outright, and `noaug - v3` the token
change on its own. Paired over the drugs both arms have finished.
""")

code(r"""
if grid.empty:
    print("no arm has finished yet")
else:
    t = grid.pivot_table(index="drug", columns="arm", values="test auc")
    t["v3"] = v3["v3 test auc"].reindex(t.index)
    pairs = [("sd19_noaug", "v3"), ("sd19_aug", "sd19_noaug"),
             ("sd19_nodist", "sd19_aug"), ("sd19_nosyn", "sd19_aug"),
             ("sd19_v5", "v3"), ("sd19_v5", "sd19_nosyn"),
             ("sd19_v5_nodist", "sd19_v5"),
             ("sd19_v5_fixdna", "sd19_v5"), ("sd19_v5_atg", "sd19_v5"),
             ("sd19_v5_fixed", "sd19_v5"), ("md_aug", "md_noaug")]
    out = []
    for a, b in pairs:
        if a in t.columns and b in t.columns:
            d = (t[a] - t[b]).dropna()
            if len(d):
                out.append({"contrast": f"{SHORT.get(a, a)} - {SHORT.get(b, b)}",
                            "drugs": len(d), "mean": d.mean(), "median": d.median(),
                            "min": d.min(), "max": d.max(),
                            "better": int((d > 0).sum())})
    display(pd.DataFrame(out).set_index("contrast").round(4) if out
            else "no pair of arms shares a finished drug yet")
""")

md(r"""
## Synonymous robustness

The failure v4 exists to fix. For reference, v3 on PZA: an H37Rv background sits
at P(R) 0.14, a single synonymous change moves the mean to 0.52-0.62, and
52-65% of changes are called resistant
(`scripts/pza_substitution_scan.py`, 2026-10-22).

Each run re-runs that scan over all 17 coding loci, plus the same test isolates
with 1 and 2 synthetic changes.

**Read `scan shift`, not `scan frac>0.5`.** The shift is the scan's mean P(R)
minus the same model's P(R) on an untouched H37Rv background: that is what
invariance means, and v3's was +0.38 to +0.48. `frac>0.5` is whatever share of
candidates land above 0.5 in absolute terms, so it reports 1.00 for a model that
is perfectly invariant but whose H37Rv baseline already sits above 0.5 -- which
is the case for AMI, CAP, LEVO and MOXI, whose own operating thresholds are
0.16-0.44.
""")

code(r"""
cols = [c for c in ["H37Rv P(R)", "scan mean P(R)", "scan shift",
                    "scan max |shift| by locus", "scan frac>0.5",
                    "probe k=1 |dlogit|", "probe k=2 |dlogit|", "probe k=1 |dP(R)|"]
        if not res.empty and c in res.columns]
if res.empty or not cols:
    print("no probes yet")
else:
    pr = res.dropna(subset=cols, how="all")
    display(pr.set_index(["arm", "cell", "drug"])[cols].round(4))
    byarm = pr[pr.arm.isin(ARMS)].groupby("arm")[cols].mean()
    if not byarm.empty:
        print("mean over the drugs that have finished")
        display(byarm.round(4))

    scan = pr[pr.arm.isin(ARMS)].dropna(subset=["scan shift"])
    if not scan.empty:
        fig, ax = plt.subplots(figsize=(9, 3.4), constrained_layout=True)
        for arm, g in scan.groupby("arm"):
            ax.scatter(g["drug"], g["scan shift"].abs(), label=arm, s=28)
        ax.axhline(0.38, color="0.4", lw=1, ls="--")
        ax.annotate("v3, PZA: +0.38", (0.995, 0.39), xycoords=("axes fraction", "data"),
                    ha="right", fontsize=8, color="0.4")
        ax.set_yscale("log")
        ax.set_ylabel("|mean P(R) shift| from one synonymous change")
        ax.tick_params(axis="x", rotation=90)
        ax.set_title("P(R) shift from one synonymous change, H37Rv background")
        ax.legend(frameon=False, fontsize=8, ncol=2)
        plt.show()
""")

md(r"""
### Per-locus scan

Every synonymous candidate inserted alone on an H37Rv background, by locus, for
one finished run.
""")

code(r"""
done = [n for n in expected if expected[n][2].exists()]
pick = next((n for n in done if expected[n][0] == "sd19_aug"
             and expected[n][1] == "PYRAZINAMIDE"),
            next((n for n in done if expected[n][0] == "sd19_aug"),
                 next((n for n in done if expected[n][0] == "canary"),
                      done[0] if done else None)))
if pick is None:
    print("no run has finished yet")
else:
    sub, tag, p = expected[pick]
    j = json.loads(p.read_text())
    h = (j.get("probe") or {}).get("h37rv_background") or {}
    for drug, s in h.items():
        by = pd.DataFrame(s["by_locus"]).T
        by.index.name = "locus"
        by["shift"] = by["mean_P_R"] - s["P_R_empty"]
        print(f"{pick}  ({drug}): empty H37Rv background P(R) = {s['P_R_empty']:.4f}")
        display(by.sort_values("mean_P_R", ascending=False)
                .rename(columns={"n": "candidates", "mean_P_R": "mean P(R)",
                                 "frac_above_0.5": "frac > 0.5"}).round(5))
        break
""")

md(r"""
## Tune: augmentation fraction x lambda

PZA only. `lambda = 0` is copies without the consistency term.
""")

code(r"""
tune = res[res.arm == "tune"] if not res.empty else res
if tune.empty:
    print("no tune cell has finished yet")
else:
    t = tune.copy()
    t["frac"] = t.cell.str.extract(r"f([\d.]+)_")[0].astype(float)
    t["lambda"] = t.cell.str.extract(r"_l([\d.]+)$")[0].astype(float)
    for metric in ["test auc", "cv auc", "scan frac>0.5", "probe k=1 |dlogit|"]:
        if metric in t.columns and t[metric].notna().any():
            print(metric)
            display(t.pivot_table(index="frac", columns="lambda", values=metric).round(4))
""")

md(r"""
## Multi-drug

One model, 11 drug queries, 11 proximity channels, on its own lineage split
(`splits/MULTIDRUG.csv`). Its TEST set is a different set of isolates from the
per-drug splits, so the single-drug columns beside it are an indication, not a
paired comparison.
""")

code(r"""
mdres = res[res.arm.str.startswith("md_")] if not res.empty else res
if mdres.empty:
    print("no multi-drug arm has finished yet")
else:
    w = mdres.pivot_table(index="drug", columns="arm", values="test auc")
    if "sd19_aug" in res.arm.values:
        w["sd19_aug (other split)"] = res[res.arm == "sd19_aug"].set_index("drug")["test auc"]
    w["test n_R"] = mdres.groupby("drug")["test n_R"].first()
    display(w.reindex([d for d in DRUGS if d in w.index]).round(4))
""")

md(r"""
## Caveats

- **`noaug` vs the v3 cell is the only clean test of the tokens.** The probes are
  what the consistency term optimises directly, so a flat scan in an `aug` run
  shows the training worked, not that the token change did it.
- **One seed, one partition.** The v3 grid's significance testing is a Wilcoxon
  over 55 folds; nothing here is a significance test, and a per-drug TEST AUC
  difference of a couple of points is within what re-running moves.
- **TEST is scored once**, with the model from the best CV fold, as in the v3
  cells.
- **The multi-drug split is its own split**: TEST is 27% of isolates, and LEVO
  has 66 R in total.
- **Data are the v3 runs' data** -- the insertion-extraction bug at
  eis/embB/ethA/rrs and the TSS-anchored promoter windows are both still there,
  which is what keeps v4 vs v3 matched.
- **`canary` and `tune` are PZA-only** and are excluded from every arm
  aggregate.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                                                  "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).resolve().parents[1] / "notebooks" / "locusfusion_v4" / "locusfusion_v4_results.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(cells)} cells)")
