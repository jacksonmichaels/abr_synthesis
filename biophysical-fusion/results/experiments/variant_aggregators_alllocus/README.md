# variant_aggregators_alllocus — the six aggregators at 19 loci

Reproducible via `submit.sh` in this folder. Submitted **2026-08-26**, 66 GPU
jobs (`63648459`–`63648524`) + 11 CPU jobs (`63648525`–`63648535`). Manifest:
`slurm_logs/manifests/submitted_20260826_172334_485603.json`.

## The question, and why it is open

`variant_aggregators_20260825` ran the six aggregators at **per-drug loci** (2
genes for ISONIAZID), and they lost: 0.889–0.891 against 0.9086 for `mdcnn` on
the identical cell. Read on its own that says the aggregators do not work.

It does not say that, because the whole variant-token family loses in that
regime and wins outside it:

| | 2–3 loci | 19 loci |
|---|---:|---:|
| `mdcnn` / `cisfusion` | 0.9086 | 0.9246 |
| **`locusfusion`** | **0.8920** | **0.9236** |
| `additive` / `deepsets` / `gatedpool` / `fm` | 0.889–0.891 | **never run** |
| `sparse_baseline` (L1-logistic, no GPU) | 0.8788 | **never run** |

At per-drug loci `locusfusion` and the six aggregators are **indistinguishable**
— 0.8892 to 0.8920, a spread of 0.0028, inside single-seed noise. What separated
`locusfusion` from them was not its architecture; it was **+0.032 from the locus
count alone** (`newmodels_full`, `sd_` vs `sd19_`).

So the aggregators have only ever been measured where the family is weakest,
and the family's own behaviour predicts they move with the loci. This run is
what makes "the aggregators are worse" either a finding or a retraction.

**`alllocus_run_v2` and `newmodels_full/sd19_*` are the matched controls** —
same 19 loci, same 300 epochs / patience 30 / `--min-epochs 50` /
`--save-weights best` / seed 0. The only difference is `--arch`.

## The hold that was deliberately overridden

This was written to wait for the `noisyor` rerun under
`variant_aggregators_20260825/`, because that is the one member whose
per-drug-loci number is not yet trustworthy — its first run scored 0.4956 on a
polarity bug (`results/archive/noisyor_polarity_bug_20260825/`). If the fix does
not restore it, 11 of these 66 jobs are wasted.

**Both were launched together anyway, wall-clock being the binding constraint
rather than compute.** The two `noisyor` arms carry the same job-name prefix but
distinct `--run-name`s and do not collide:

| | run folder | loci |
|---|---|---|
| rerun of the failed cell | `variant_aggregators_20260825/all_modalities__noisyor` | per-drug (2–3) |
| this run | `variant_aggregators_alllocus/all_modalities__noisyor` | 19 |

Both read the **fixed** `models/experimental_models.py`, so the risk taken is
only that the fix proves insufficient — not that this run reproduces the bug.
If the per-drug rerun comes back near 0.5, treat this run's `noisyor` column as
uninterpretable and read the other five.

## What each cell is for

The six differ ONLY in how the variant set is aggregated — same tokenizer, same
features, same read-out contract — so any difference is attributable.

| cell | asks |
|---|---|
| `catalogue` | how far does pure memorisation get, at 19 loci? |
| `additive` | what does featurising the variant buy over memorising it? |
| `noisyor` | does "susceptible unless something confers resistance" hold up? |
| `gatedpool` | was softmax normalisation the thing that broke attention? |
| `deepsets` | does attention buy anything at all over plain additivity? |
| `fm` | is epistasis worth anything, priced at O(T·rank)? |

## How to read it

**Two comparisons decide whether the family survives, and they are not the
macro.**

1. **`sparse_baseline` at 19 loci.** It is an L1-logistic on the variant matrix
   with no GPU, and at per-drug loci it was 0.8788 — within 0.03 of the best
   CNN. If it tracks the networks up to ~0.92 here, then the interesting result
   of this whole line of work is that a linear model on variants is enough, and
   the architectures are a footnote. Read this **first**.
2. **`catalogue` vs `additive`.** At per-drug loci the gap was **+0.022** for
   featurising over memorising — the cleanest measurement in the project of
   generalisation to substitutions unseen in training (the *pncA* /
   PYRAZINAMIDE mechanism). At 19 loci the variant vocabulary is ~10x larger and
   the unseen fraction larger with it, so the gap should **widen**. If it
   narrows instead, the featurisation story is weaker than claimed.

The macro over the six is the least interesting number in the run. At per-drug
loci five of them sat within 0.0014 of each other, which already said the
aggregator is not the lever; a repeat of that at 19 loci is a confirmation, not
a discovery.

## What this run deliberately does NOT do

- **No modality ladder.** `all_modalities` only. `newmodels_full` showed
  `locusfusion` moves by 0.0012 across all five modality sets at 19 loci — the
  variant-token family is nearly modality-insensitive there, so five rows would
  cost 5x the jobs to re-measure a flat line.
- **No joint arm.** `locusfusion` lost the multi-drug cell (0.9118 vs 0.9191
  `mdcnn`); the aggregators have no reason to do better and the question here is
  about loci, not task.
- **No knob sweeps.** Every cell runs at `EXPERIMENTAL_DEFAULTS`.
  `setfusion_scaling` spent 62 arms establishing that width is not the lever for
  this family.

## Known quirk of the sparse arm

`sparse_baseline.py` writes `summary.csv` inside its per-drug loop, and one job
per drug shares an `--out`, so the last job to finish leaves a **1-row**
summary. The JSONs are the real record. Rebuild after the run:

```python
rows = [json.loads(f.read_text()) for f in out.glob("*.json")]
```

That is what happened to `variant_aggregators_20260825/sparse_baseline/` and it
was regenerated from the 11 JSONs on 2026-08-26.
