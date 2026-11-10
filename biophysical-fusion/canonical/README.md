# Canonical models

Edit the networks here. The training code imports these files, so the next run
trains whatever these files contain.

| file | model | trained by | old import path (still works) |
|---|---|---|---|
| `locusfusion.py` | LocusFusion v5_fixed, the 19-locus model (`LocusFusionNet`) | `locusfusion_v4/train.py` | `locusfusion_v4.model` (`LocusFusionV4Net`) |
| `genomefusion.py` | GenomeFusion, whole-genome LocusFusion (`GenomeFusionNet`) | `genomefusion/train.py` | `genomefusion.model` |

Each file's docstring lists its train, sweep and test commands. The rest of
each pipeline stays in its package:

- **Tokens:** `locusfusion_v4/tokenizer.py` + `layout.py`, and
  `genomefusion/tokenizer.py` + `layout.py`. A new channel goes in the layout
  and tokenizer first; the model only sees its width.
- **Loci:** `genomefusion/partition.py`.
- **Training loop:** each package's `train.py`.

The LocusFusion in the paper is v3 (`models/locusfusion_v2.py`,
`LocusFusionV2Net`). It's the same network shape, but it tokenizes inside the
model and a 19-locus job needs about 128 GB. v5_fixed is the version
GenomeFusion is compared against.

## After editing

1. Run the model tests. They use synthetic data and take seconds, but run them
   in a job or `salloc` session rather than on a login node:
   `python -m locusfusion_v4.tests.test_model` and
   `python -m genomefusion.tests.test_model`.
2. Print the jobs without submitting: `python -m <package>.sbatch --arms canary --dry-run`.
3. Run the canary (PZA, one job) before any full sweep.

Checkpoints are `{"config": model.config, "state_dict": ...}`. If you rename
or add parameters, earlier weights will no longer load with `strict=True`.
