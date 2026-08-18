# setfusion_scaling — is setfusion small, or is it *shaped* wrong?

> **Tooling retired 2026-08-13.** `scripts/sweep_setfusion_scaling.py`,
> `analyze_setfusion_scaling.py` and `watch_setfusion_scaling.sh` were removed in
> the repo cleanup, so `submit.sh` in this folder no longer runs as written and
> the script paths cited below are historical. The results, figures and findings
> here are unaffected. To resume the 616 held-back single-drug jobs, restore the
> scripts first — from the repo root, no hash needed:
>
> ```bash
> for P in biophysical-fusion/scripts/sweep_setfusion_scaling.py \
>          biophysical-fusion/scripts/analyze_setfusion_scaling.py \
>          biophysical-fusion/scripts/watch_setfusion_scaling.sh; do
>     git checkout "$(git log --diff-filter=D -1 --format=%H -- "$P")^" -- "$P"
> done
> ```
>
> The deletions were committed on 2026-08-18 (`27a4a7b`), and the loop above
> finds that commit itself, so it keeps working across rebases and squashes.
> Verified against `sweep_setfusion_scaling.py` on 2026-08-18.

Reproducible via `submit.sh` in this folder, which drives
`scripts/sweep_setfusion_scaling.py` (arms are defined there, once, and the same
definitions produce `arm_params.csv`).

## The question

`setfusion` is the smallest architecture in the grid by two orders of magnitude
and the weakest joint model in `full_run_v2`:

| arch | joint DNA params | joint DNA macro CV |
|---|---|---|
| late_fusion | 45,898,955 | 0.9184 |
| cisfusion | 47,955,211 | 0.9228 |
| **setfusion** | **460,417** | **0.7939** |

Two explanations fit that table. Either the model is simply too small — in which
case scaling it up closes the gap — or it is the *shape* of the model, and no
amount of width helps. The sweep is built to tell those apart, because the four
axes hit structurally different things and the parameter breakdown of the
control says they are not interchangeable:

| component | params | share |
|---|---|---|
| fusion transformer (2 layers) | 264,960 | 57.5% |
| shared per-modality encoders | 91,776 | 19.9% |
| pool_attn (drug queries -> tokens) | 66,048 | 14.3% |
| fc1 read-out | 33,024 | 7.2% |
| **drug_queries — all per-drug capacity** | **1,408** | **0.3%** |

Two things in that table are worth stating before any result comes back:

1. **The read-out is shared across all 11 drugs.** Every drug's pooled vector
   goes through the same `fc1 -> fc_out`; its only private parameters are one
   128-d query. `TODO.md` records the init-time symptom — fused tokens are
   near-collinear (pairwise cosine 0.9968–0.9978), so every query pools nearly
   the same vector and a shared MLP maps them to nearly the same logit. **No
   width knob changes this**, which is why axis D exists.
2. **The information bottleneck is the pooling, not the widths.** A 3,423 bp
   locus is reduced to `2 * out_channels * bins` = 256 numbers before a token
   exists; the whole isolate becomes 19 x 128 = 2,432 floats from 198,230 raw
   input floats, where `late_fusion` flattens 137,952 features and keeps every
   column. Resistance is single-column SNPs. `--enc-bins` is the only knob that
   reopens position resolution, and it is cheap — hence a four-point ladder.

## The control

`results/experiments/full_run_v2/{,multidrug_}{dna,dna_protein}__setfusion`,
already run. No control jobs are submitted here. Every arm reuses that run's
protocol **exactly** — `--epochs 300 --patience 30 --min-epochs 50`, lr `exp(-9)`,
batch 128, seed 0, 5-fold CV, all curated loci, `--save-weights best` — so the
only difference between an arm and its control is the flags in the tables below.

Both modality sets are swept: `dna` (19 tokens joint, 1–2 single) and
`dna_protein` (38 joint, 4 single), the best joint setfusion cell in
`full_run_v2` and the set `joint_capacity` used for the same kind of question.

## The arms

Parameter counts are joint-DNA (`arm_params.csv` has all four cells). Ratios are
against the 460,417-parameter control.

### Axis A — token width (`--d-model`), `nhead` fixed at 4

| arm | change | params | xctl |
|---|---|---|---|
| `a1_d192` | `--d-model 192` | 808,129 | 1.76 |
| `a2_d256` | `--d-model 256` | 1,254,145 | 2.72 |
| `a3_d384` | `--d-model 384` | 2,441,089 | 5.30 |
| `a4_d512` | `--d-model 512` | 4,021,249 | 8.73 |

One knob scales the encoder output, both identity embeddings, the transformer,
the drug queries and the head input together — it is the only axis that widens
every stage at once.

### Axis B — everything after the encoders

| arm | change | params | xctl |
|---|---|---|---|
| `b1_ff512` | `--dim-ff 512` | 592,001 | 1.29 |
| `b2_ff1024` | `--dim-ff 1024` | 855,169 | 1.86 |
| `b3_ff2048` | `--dim-ff 2048` | 1,381,505 | 3.00 |
| `b4_layers3` | `--fusion-layers 3` | 592,897 | 1.29 |
| `b5_layers4` | `--fusion-layers 4` | 725,377 | 1.58 |
| `b6_layers6` | `--fusion-layers 6` | 990,337 | 2.15 |
| `b7_hidden512` | `--hidden 512` | 493,697 | 1.07 |
| `b8_hidden1024` | `--hidden 1024` | 560,257 | 1.22 |

Three separate single-knob ladders (FF width, fusion depth, read-out width), not
one coupled one, so a win is attributable to the layer that produced it.

### Axis C — inside one `SharedBlockEncoder`

| arm | change | params | xctl |
|---|---|---|---|
| `c1_enc96` | `--enc-width 96 --enc-out-channels 48` | 550,017 | 1.19 |
| `c2_enc128` | `--enc-width 128 --enc-out-channels 64` | 668,801 | 1.45 |
| `c3_depth2` | `--enc-depth 2` | 466,625 | 1.01 |
| `c4_depth3` | `--enc-depth 3` | 472,833 | 1.03 |
| `c5_bins8` | `--enc-bins 8` | 493,185 | 1.07 |
| `c6_bins16` | `--enc-bins 16` | 558,721 | 1.21 |
| `c7_bins32` | `--enc-bins 32` | 689,793 | 1.50 |
| `c8_bins64` | `--enc-bins 64` | 951,937 | 2.07 |

`width`/`out_channels` move together in the architecture's own 2:1 ratio, so
"encoder channels" is one knob rather than two. `--enc-depth` is nearly free in
parameters (+1%) but not in FLOPs — it buys receptive field, and it is the arm
to read if depth-not-width is what the encoder lacks. At `--enc-bins 64` the
short protein blocks are oversampled (a 432-aa locus is ~48 positions after
pooling); that is a real limit of the top of this ladder, not a bug.

### Axis D — per-drug read-out capacity (joint only)

| arm | change | params | xctl |
|---|---|---|---|
| `d1_perdrug64` | `--per-drug-hidden 64` | 641,803 | 1.39 |
| `d2_perdrug128` | `--per-drug-hidden 128` | 823,435 | 1.79 |

Each drug gets its own `hidden -> k -> 1` branch off the shared trunk, exactly
as `joint_capacity/b2_perdrug64` did for `late_fusion`. Not submitted
single-drug: with one output there is nothing to separate.

### Axis R — training regime at *baseline* capacity

| arm | change | params | xctl |
|---|---|---|---|
| `r1_cosine` | `--lr-schedule cosine --warmup-epochs 20` | 460,417 | 1.00 |
| `r2_reg` | `--dropout 0.3 --weight-decay 1e-4` | 460,417 | 1.00 |

These exist so that a null result at the top of the ladder is *readable*:
without them, "the big model did not help" and "the big model was not trained,
or overfit" are the same observation. Identical parameter counts to the control
is also the cleanest confirmation that nothing structural moved.

**The base learning rate is deliberately NOT swept.** `joint_convergence`
already measured that on this data: `a2_lr1e3` took joint macro CV from 0.9113
to **0.7350**, and `a3_lr1e3_reg` to **0.5834**. The cosine multiplier never
exceeds 1.0, so `r1` can only ever scale the recorded LR *down*. Likewise
`r2` uses `joint_capacity/b3_reg`'s exact values (0.3 / 1e-4), which **hurt**
there (cisfusion 0.9228 -> 0.8771, late_fusion 0.9184 -> 0.8955) — it is
included because that was measured at 46 M parameters with no capacity change,
and this sweep pairs it with a 29x model where the overfitting argument is
different. Expect it to lose at baseline capacity; it is the control for `x6`.

No gradient clipping and no batch-size change: neither has an observed problem
to fix here, and each would be another uncontrolled knob.

### Axis X — sparse crosses (bundles, not single knobs)

| arm | change | params | xctl |
|---|---|---|---|
| `x1_AB` | d256 + ff1024 + 4 layers | 3,621,633 | 7.87 |
| `x2_AC` | d256 + enc128/64 + bins16 | 1,888,513 | 4.10 |
| `x3_BC` | ff1024 + 4 layers + enc128/64 + bins16 | 1,919,873 | 4.17 |
| `x4_mid` | A+B+C mid | 4,346,753 | 9.44 |
| `x5_big` | top of every ladder, control regime | 13,281,281 | 28.85 |
| `x6_big_tuned` | `x5` + `r1` + `r2` | 13,281,281 | 28.85 |
| `x7_big_perdrug` | `x6` + `d1` (joint only) | 13,642,635 | 29.63 |

Each *pair* of axes appears once, then the triple, then the ceiling with and
without the regime change. Read these only against the single-knob arms —
alone, they are uninterpretable by construction. Note the ceiling is still
**3.5x smaller than `late_fusion`**, so even `x5` does not test "setfusion at
late_fusion's size".

## Cost

31 arms x 2 modality sets x 2 scopes = 118 cells = **678 SLURM jobs** (62 joint,
616 single-drug). For scale, `full_run_v2`'s joint setfusion cells took 4.8–11.4 h
each at 0.46 M params, and the `--enc-width` arms are the ones that will move
wall-clock most: the 12-tap `conv1` runs over the full un-pooled length, so
doubling its channels is ~4x the FLOPs of the layer that dominates the model.
Those arms (marked `heavy` in the script) get double the time limit and
`--constraint vram23`.

Submission is staged rather than fired at once — `submit.sh` partitions the
sweep into `joint` (62), `single-ab` (264), `single-cr` (220) and `single-x`
(132), so the joint question can be answered before the single-drug side is
paid for.

## Reading it

`scripts/analyze_setfusion_scaling.py` does this properly and is safe to run at
any time — unfinished cells are reported as missing rather than silently
dropped, so a partial read never looks like a complete one:

```bash
python scripts/analyze_setfusion_scaling.py                 # joint, to stdout
python scripts/analyze_setfusion_scaling.py --write         # + ANALYSIS_joint.md, .csv, .png
python scripts/analyze_setfusion_scaling.py --scope single  # once those stages run
```

It differences every arm against **its own** control (same modality set, same
scope), judges each delta against a noise band of `max(0.01, control fold SD)`
and reports anything inside it as **unresolved** rather than as a small win,
flags arms whose folds hit the 300-epoch ceiling (measured under undertraining,
so their deltas are lower bounds), and asserts that the R arms have the
control's exact parameter count while the A–D arms do not — a mismatch there
means the flags never reached the model and the cell is measuring nothing.

While it is in flight, `scripts/run_monitors/sweep_status.py` is the per-job
glance — one line per cell, with fold progress for the ones still running:

```bash
python scripts/run_monitors/sweep_status.py            # whatever is live
python scripts/run_monitors/sweep_status.py -d         # list finished cells + their CV
watch -n 120 python scripts/run_monitors/sweep_status.py
```

A watcher is running for the joint stage
(`scripts/watch_setfusion_scaling.sh`, launched detached; progress in
`slurm_logs/watch_setfusion_scaling.log`, job ids in `joint_job_ids.txt`). It
polls until all 62 jobs leave the queue, then writes the analysis and a
`.watch_joint_done` marker. It counts result JSONs afterwards, because a job can
leave the queue by failing and "no longer running" must not be read as "no
error".

The by-hand version, if you want to check the script's arithmetic:

```bash
python - <<'EOF'
import json, glob, os
ctl = {}
for p in glob.glob("../full_run_v2/multidrug_*__setfusion/multidrug__*.json"):
    j = json.load(open(p))
    ctl[os.path.basename(os.path.dirname(p))] = j
for p in sorted(glob.glob("*/multidrug__*.json")):
    j = json.load(open(p))
    arm, cell = os.path.dirname(p).split("_multidrug_")
    c = ctl.get("multidrug_" + cell)
    d = j["cv_macro_auc_mean"] - c["cv_macro_auc_mean"] if c else float("nan")
    print(f"{arm:16s} {cell:14s} cv={j['cv_macro_auc_mean']:.4f} ({d:+.4f}) "
          f"params={j['n_params']/1e6:5.2f}M epochs={[f['best_epoch'] for f in j['cv_folds']]}")
EOF
```

**This is a screen, not a result.** One seed, five folds; joint fold SD in
`full_run` was 0.003–0.030 against the gap being chased. An arm that gains under
~0.01 has not been shown to do anything. The output of this sweep is a shortlist
of 2–3 directions worth multi-seeding, not a winner.

Two more things to carry forward when reading it:

- **Single-drug `dna` cells have only 1–2 blocks per drug** (ISONIAZID:
  `dna:inhA`, `dna:katG`), so a transformer over two tokens is barely being
  asked to fuse anything. Axes A and B are near-vacuous there by construction;
  axis C is not. The `dna_protein` single-drug cells (4 tokens) are the honest
  single-drug read.
- If axis D wins, the next arm is not more width — it is the token-collinearity
  fix `TODO.md` already measured at init (centering tokens before attention took
  per-drug logit spread from 1.5e-04 to 1.5e-02, a 100x change, one line). That
  was left alone by decision; a D win is the evidence that would reopen it.

## Weights

`--save-weights best` to
`/project/pi_mfiterau_umass_edu/abr_model_weights/setfusion_scaling/{arm}_{cell}/`,
each with a `config.json` that fully rebuilds the model — the new capacity knobs
are recorded under `model.setfusion` and round-trip bit-identically
(`tests/test_checkpoint.py::test_roundtrip_setfusion_capacity_knobs`). A config
written before those knobs existed has no `setfusion` key and still rebuilds at
the defaults, so `full_run`/`full_run_v2` checkpoints are unaffected.
