"""
Checks for ``--cv-seed`` — the CV-partition seed for repeated cross-validation.

A tiny synthetic-fixture integration test (1 epoch, CPU, seconds) plus the
sbatch command generator:

    python tests/test_cv_seed.py

The claims:

  * **The default is the old partition.** Omitting ``cv_seed`` gives exactly the
    folds ``cv_seed=42`` gives — every run before 2026-09-18 used 42.
  * **The partition is deterministic** for a given seed, and **moves** when the
    seed does. Each fold's held-out isolates are recorded as ``val_ids_sha1``, so
    a paired test can check pairing instead of assuming it.
  * **The generator forwards it only when it differs**, so an unchanged grid still
    produces byte-identical commands; and it refuses ``--multidrug``, where the
    partition is not wired.
"""
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from datasets import load_dataset  # noqa: E402
from datasets.fixtures import build_fixture_dataset  # noqa: E402
from training.multimodal import run_modal_cv  # noqa: E402

_RESULTS = []


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
        _RESULTS.append(True)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        _RESULTS.append(False)


def _fingerprints(**kw):
    with tempfile.TemporaryDirectory() as tmp:
        geno, pheno = build_fixture_dataset(tmp, genes=["gyrA"], drugs=["MOXIFLOXACIN"],
                                            n_isolates=120, n_codons=10, seed=3)
        data = load_dataset("MOXIFLOXACIN", ["dna"], geno, pheno, verbose=False)
        res = run_modal_cv(data, epochs=1, n_splits=3, batch_size=64,
                           save_weights="none", **kw)
    return res, [f["val_ids_sha1"] for f in res["cv_folds"]]


def _test_default_is_42_and_seed_moves_partition():
    res_default, fp_default = _fingerprints()
    res_42, fp_42 = _fingerprints(cv_seed=42)
    _, fp_42_again = _fingerprints(cv_seed=42)
    res_7, fp_7 = _fingerprints(cv_seed=7)
    assert res_default["cv_seed"] == 42 and res_42["cv_seed"] == 42 and res_7["cv_seed"] == 7
    assert fp_default == fp_42, "omitting cv_seed changed the partition"
    assert fp_42 == fp_42_again, "the partition is not deterministic"
    assert len(set(fp_42)) == 3, "folds within one partition are not distinct"
    assert fp_7 != fp_42, "a different cv_seed gave the same partition"


def _generator(*extra):
    return subprocess.run(
        [sys.executable, "scripts/sbatch_all_runs.py", "--dry_run",
         "--experiments", "all_modalities__mdcnn", "--drugs", "ISONIAZID", *extra],
        cwd=PROJECT, capture_output=True, text=True)


def _test_generator_forwards_only_when_set():
    plain = _generator()
    assert plain.returncode == 0, plain.stderr
    assert "--cv-seed" not in plain.stdout, "the default was emitted — grids would change"
    moved = _generator("--cv-seed", "101")
    assert moved.returncode == 0, moved.stderr
    assert "--cv-seed 101" in moved.stdout, "a non-default cv-seed was not forwarded"
    md = subprocess.run(
        [sys.executable, "scripts/sbatch_all_runs.py", "--dry_run", "--multidrug",
         "--experiments", "all_modalities__mdcnn", "--cv-seed", "101"],
        cwd=PROJECT, capture_output=True, text=True)
    assert md.returncode != 0 and "single-drug" in (md.stderr + md.stdout), \
        "--multidrug accepted a cv-seed it cannot apply"


if __name__ == "__main__":
    check("default is 42; the partition is deterministic and moves with the seed",
          _test_default_is_42_and_seed_moves_partition)
    check("the generator forwards --cv-seed only when set, and refuses --multidrug",
          _test_generator_forwards_only_when_set)
    n_fail = _RESULTS.count(False)
    print(f"\n{len(_RESULTS) - n_fail}/{len(_RESULTS)} passed")
    sys.exit(1 if n_fail else 0)
