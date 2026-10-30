"""
The tidy results table, built once from `results/experiments/`.

Both notebook builders read this — `scripts/build_overview.py` (every run,
compared) and `scripts/build_results.py` (the figures). Same rule as
`bigtb_baselines.py`: single source of truth, imported, never copied. A loader
that lives in two places drifts, and the two notebooks then disagree about how
many runs exist.

    from results_frame import load_frame
    frame, meta = load_frame()

`frame` is one row per (task, modality set, model, drug):

    task     'single-drug' | 'single-drug, all loci' | 'multi-drug'
    modset   dna | dna_protein | dna_biophysical | dna_regulatory | all_modalities | ...
    model    architecture (+ '+tf' when a transformer encoder was used)
    cv       5-fold CV AUC mean      cv_sd  its fold SD (NaN for joint runs)
    fold_aucs  the 5 per-fold AUCs, full precision, () where a run kept none
    test     held-out test AUC       n_loci / n_params  the shape it ran at
    n_valid  labelled isolates for that drug   n_R  how many are resistant
             (single-drug only -- the joint summary counts its test split)

Everything is DERIVED from the result JSONs, so a new run folder is picked up
with no code change.
"""
import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent
EXP = PROJECT / "results" / "experiments"

DROP_ARCH = {"setfusion"}                 # degenerate tokens; see token_signal
DROP_RUNS = {
    "full_run", "alllocus_run",           # superseded 150/15/0 schedule
    # Hyperparameter sweeps, not grid cells: their folders are named for the ARM
    # (`a2_lr1e3_multidrug_dna__late_fusion`), so each arm would invent its own
    # modality-set row. They have their own write-ups; keep them out.
    "joint_capacity", "joint_convergence", "setfusion_scaling", "setfusion_warmup",
    "token_signal",
    "newmodels_pilot",                    # 2 drugs, superseded by newmodels_full
    # Multi-seed replicate of two 19-locus cells. Its folders are named
    # `s<seed>_<modset>__<arch>`, so each seed would invent its own modality-set
    # row, and averaging seeds into the seed-0 grid would change every cell it
    # touches. It has its own analysis (seeds_19loci/README.md).
    "seeds_19loci",
    # Repeated-CV replicate (the same two cells on other CV partitions). Folder
    # names are `cv<seed>_<modset>__<arch>`, and pooling a partition into the
    # seed-42 grid would mix folds that are not paired. Read with its compare.py.
    "repartition_19loci",
    # locusfusion_v2 + --lf-unknown-symbols, the full sweep (2026-10-01). Its
    # result JSONs say arch=locusfusion_v2, so loading it here would pool it into
    # the v2 cells. Read with locusfusion_v3/compare.py.
    "locusfusion_v3",
    # The paper grid retrained on lineage-grouped CV folds (Beijing its own fold),
    # a different partition: its cells share folder names with the random-split
    # grid, so loading it here would pool unpaired folds into the paper cells. Read
    # with results/analysis/lineage_split_significance/ and notebooks/lineage_split.ipynb.
    "lineage_split",
}

# Some runs put the TASK in the cell name rather than the run name --
# newmodels_full submits `sd_`, `sd19_` and `md_multidrug_` cells into one
# folder. Strip those so `sd_all_modalities__locusfusion` lands on the
# `all_modalities` row beside every other model. The task itself is still
# derived from which summary file exists and from the locus count, never from
# the prefix.
TASK_PREFIXES = ("md_multidrug_", "multidrug_", "sd19_", "sd_", "md_")

# A cell folder is normally `<modset>__<arch>`. The sparse baselines are named
# for the METHOD alone, because they are not an architecture (L1-logistic /
# gradient boosting on the variant matrix). They still belong in the grid, so
# fall back to the result's own modality tag and use the folder name as the model.
TAG_TO_MODSET = {
    "dna": "dna", "dna+protein": "dna_protein",
    "dna+biophysical": "dna_biophysical", "dna+regulatory": "dna_regulatory",
    "dna+protein+biophysical+regulatory": "all_modalities",
    # `lasso_wholeseq_20260901` is a MODEL family, not an architecture, so its
    # folders are named for (encoding x locus universe) rather than
    # `<modset>__<arch>`. All four arms are DNA-only; the locus universe is
    # already carried by `task`, so only the encoding survives into the model
    # name (below). Without these four the whole run is silently skipped, and
    # it holds the project's strongest baseline.
    "dna-onehot-all": "dna", "dna-onehot-perdrug": "dna",
    "dna-delta-all": "dna", "dna-delta-perdrug": "dna",
    # `lasso_wholegenome_20260903` -- the same model family over all 4.41 Mb.
    "dna-genome-mc1": "dna",
}

# Folder stem -> model name, for runs whose folder name is not `<modset>__<arch>`.
# Falls back to the stem itself (which is what `sparse_baseline` wants).
TAG_TO_MODEL = {
    "dna-onehot-all": "lasso_onehot", "dna-onehot-perdrug": "lasso_onehot",
    "dna-delta-all": "lasso_delta", "dna-delta-perdrug": "lasso_delta",
    "dna-genome-mc1": "lasso_genome",
}

# The task is normally derived from the locus COUNT, which cannot work for a
# whole-genome run: it reports a single pseudo-locus, so it would otherwise be
# filed under the drug's own 2-3 loci -- the opposite of what it is. Runs whose
# locus universe is not expressible as a count declare it here instead.
TAG_TO_TASK = {"dna-genome-mc1": "single-drug, whole genome"}

# (run, cell) pairs to skip: a superseded arm sitting in a folder whose other
# arms are current, so DROP_RUNS is too blunt.
DROP_CELLS = {
    # The ETHIONAMIDE-only pilot, superseded by `all_drugs`; leaving it in
    # duplicates that one drug and drags the macro down by ~0.009.
    ("lasso_wholegenome_20260903", "pilot"),
}

ALL_LOCI = 19                             # what counts as "all loci"
TASKS = ["single-drug", "single-drug, all loci", "single-drug, whole genome",
         "multi-drug"]
STD_MODSETS = ["dna", "dna_protein", "dna_biophysical", "dna_regulatory",
               "all_modalities"]


def strip_task_prefix(name):
    for pre in TASK_PREFIXES:
        if name.startswith(pre):
            return name[len(pre):]
    return name


def _fold_aucs(j):
    """The 5 per-fold CV AUCs, at full precision, or () if the run kept none.

    Every engine writes `cv_folds`; the lasso sweep instead keeps one entry per
    penalty on `c_path` and reports the best, so read that entry's `fold_aucs`.

    These matter because `summary.csv` is rounded to 4 dp and because all runs
    use the SAME `StratifiedKFold(5, shuffle=True, random_state=42)` splits on
    the same cohort -- so fold *i* of two different models is the same held-out
    isolates, and the two are paired.
    """
    folds = j.get("cv_folds")
    if folds:
        aucs = [f.get("auc") for f in folds if f.get("auc") is not None]
        return tuple(aucs)
    path, best = j.get("c_path"), j.get("best_C")
    if path and best is not None:
        for e in path:
            if e.get("C") == best and e.get("fold_aucs"):
                return tuple(e["fold_aucs"])
    return ()


def _per_drug_folds(j):
    """{drug: (auc per fold)} for a JOINT result, which has no single `drug`.

    The multi-drug engine nests one level deeper --
    `cv_folds[i]["per_drug"][DRUG]["auc"]`.

    NOT comparable fold-for-fold with the single-drug runs: `run_multidrug_cv`
    splits with a non-stratified 80/20 + KFold(5), the single-drug engine with a
    stratified split + StratifiedKFold(5). Same seed, different partition. Pair
    these over DRUGS, never over folds.
    """
    out = {}
    for f in j.get("cv_folds") or []:
        for drug, m in (f.get("per_drug") or {}).items():
            if m.get("auc") is not None:
                out.setdefault(drug, []).append(m["auc"])
    return {d: tuple(v) for d, v in out.items()}


def _read(p):
    try:
        j = json.loads(p.read_text())
    except Exception:
        return None
    loci = j.get("loci") or j.get("genes") or []
    enc = set(j.get("encoder_types") or []) | set((j.get("encoders") or {}).values())
    tf = "transformer" in enc
    return dict(arch=j.get("arch"), tf=tf,
                model=f"{j.get('arch')}{'+tf' if tf else ''}",
                n_loci=len(loci), n_params=j.get("n_params"), drug=j.get("drug"),
                fold_aucs=_fold_aucs(j), per_drug_folds=_per_drug_folds(j))


def load_frame(drugs=None, exp_dir=None):
    """-> (frame, meta). `meta` carries MODSETS / MODELS / skipped / dupes."""
    exp = Path(exp_dir) if exp_dir else EXP
    if drugs is None:
        from bigtb_baselines import ALL_DRUGS
        drugs = list(ALL_DRUGS)
    drugs = set(drugs)

    shape = {}                            # (run, cell, drug) -> shape dict
    for run_dir in sorted(p for p in exp.iterdir() if p.is_dir()):
        if run_dir.name in DROP_RUNS:
            continue
        for d in (p for p in run_dir.iterdir() if p.is_dir()):
            for p in d.glob("*.json"):
                if p.name == "weights_location.json":
                    continue
                s = _read(p)
                if s:
                    shape[(run_dir.name, d.name, s.pop("drug"))] = s

    rows, skipped = [], []
    for run_dir in sorted(p for p in exp.iterdir() if p.is_dir()):
        if run_dir.name in DROP_RUNS:
            continue
        for d in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            if (run_dir.name, d.name) in DROP_CELLS:
                continue
            # A canary is a few-epoch plumbing check, never a result; its folder
            # (`canary_sd19_<modset>__<arch>`) would otherwise invent a modset row.
            if d.name.startswith("canary"):
                continue
            # Which summary file exists IS the task, and is authoritative.
            joint = (d / "multidrug_summary.csv").is_file()
            f = d / ("multidrug_summary.csv" if joint else "summary.csv")
            if not f.is_file():
                continue
            stem = strip_task_prefix(d.name)
            forced_task = None            # set below for runs that declare one
            here = {k: v for k, v in shape.items()
                    if k[0] == run_dir.name and k[1] == d.name}
            if not here:                  # no JSON -> cannot be placed
                skipped.append(f"{run_dir.name}/{d.name}")
                continue
            any_shape = next(iter(here.values()))
            if any_shape["arch"] in DROP_ARCH:
                continue
            df = pd.read_csv(f)
            if "__" in stem:
                modset, model = stem.rsplit("__", 1)[0], any_shape["model"]
            else:
                tag = str(df["modalities"].iloc[0])
                modset, model = TAG_TO_MODSET.get(tag), TAG_TO_MODEL.get(tag, stem)
                forced_task = TAG_TO_TASK.get(tag)
                if modset is None:
                    skipped.append(f"{run_dir.name}/{d.name} (unmapped tag {tag!r})")
                    continue
            # "all loci" is taken from the MAX over the cell's drugs, so a
            # 1-locus drug cannot mislabel the cell.
            cell_loci = max(v["n_loci"] for v in here.values())
            task = (forced_task if forced_task else
                    "multi-drug" if joint else
                    "single-drug, all loci" if cell_loci >= ALL_LOCI else "single-drug")
            df = (df[df["drug"] != "MACRO"].rename(columns={"cv_auc": "cv"}) if joint
                  else df.rename(columns={"cv_auc_mean": "cv", "cv_auc_std": "cv_sd"}))
            for _, r in df.iterrows():
                if r["drug"] not in drugs:
                    continue
                exact = here.get((run_dir.name, d.name, r["drug"]))
                s = exact if exact is not None else any_shape
                # Folds are per-DRUG, so they may only ever come from that
                # drug's own JSON -- never from the `any_shape` fallback, which
                # would silently attach a different drug's folds. A joint run
                # has one JSON for every drug, and carries them nested instead.
                folds = ((exact or {}).get("fold_aucs")
                         or any_shape.get("per_drug_folds", {}).get(r["drug"], ()))
                rows.append(dict(task=task, modset=modset, model=model,
                                 fold_aucs=folds,
                                 arch=any_shape["arch"], run=run_dir.name, cell=stem,
                                 drug=r["drug"], n_loci=s["n_loci"],
                                 n_params=s["n_params"], cv=r.get("cv"),
                                 cv_sd=r.get("cv_sd"), test=r.get("test_auc"),
                                 # Cohort size, carried through because how much
                                 # labelled data a drug has explains several of
                                 # the per-drug results. Single-drug summaries
                                 # count the WHOLE cohort; the joint summary
                                 # only counts its test split, so n_valid is
                                 # left NaN there rather than filled with a
                                 # number that means something different.
                                 n_valid=r.get("n_valid"), n_R=r.get("n_R")))

    frame = pd.DataFrame(rows)
    for c in ("cv", "cv_sd", "test", "n_valid", "n_R"):
        frame[c] = pd.to_numeric(frame[c], errors="coerce")

    modsets = ([m for m in STD_MODSETS if m in set(frame.modset)]
               + sorted(set(frame.modset) - set(STD_MODSETS)))
    dupes = (frame.groupby(["task", "modset", "model", "run"]).size()
             .reset_index().groupby(["task", "modset", "model"]).size())
    meta = {"MODSETS": modsets, "MODELS": sorted(set(frame.model)),
            "skipped": skipped,
            "dupes": [f"{a}/{b}/{c}" for a, b, c in dupes[dupes > 1].index],
            "n_rows": len(frame),
            "n_cells": frame.groupby(["task", "modset", "model"]).ngroups}
    return frame, meta
