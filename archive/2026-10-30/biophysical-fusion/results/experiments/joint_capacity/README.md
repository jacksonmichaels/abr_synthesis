# joint_capacity — does the joint head have enough capacity for 11 tasks?

Submitted **2026-08-06**, 8 SLURM jobs `62656456–62656463`, reproducible via
`submit.sh` in this folder. Code changes behind it: `../CODE_CHANGES_20260806.md`.

## The question

`full_run`'s substantive finding was that **modality choice matters much more
single-drug than joint** — +0.023–0.031 single-drug against +0.005 for joint
`late_fusion` — explained as a joint model already seeing every locus for every
drug. That explanation is probably right, but there is a mechanical one sitting
underneath it that nobody controlled for.

`MultiDrugNet` is `MultiModalNet` with `out_dim=n_drugs`, so its head is:

```
fc1(236k -> 256) -> fc2(256 -> 256) -> fc_out(256 -> 11)
```

All eleven drugs are read off **one 256-d vector by one shared linear layer**.
A single-drug model gets that same 256 units for one task. And adding a modality
widens `fc1`'s *input* — 39,646 bp of DNA becomes 236k flattened features with
all-modalities — without widening the representation everything must squeeze
through. If that vector is saturated, extra modalities have nowhere to go, which
is exactly the observed symptom.

There is also no regularization anywhere in the stack: no dropout, no
normalization, no weight decay, on ~46M parameters fit to 17,941 isolates.

## The cells

`multidrug dna_protein`, `late_fusion` + `cisfusion` — the best joint modality
set in `full_run` (cisfusion **0.9228**, late_fusion **0.9184**).

**`cisfusion` is the cleaner read of the two.** Its folds converge around epoch
85, so the 150-epoch cap is not binding for it; `late_fusion` had 40% of folds
pinned at the cap, so its arms measure capacity *under* undertraining. Weight
the cisfusion result accordingly.

## The arms — each changes ONE thing; the control is `full_run` itself

Every arm runs at `--epochs 150`, matching `full_run` exactly, so
`full_run/multidrug_dna_protein__{arch}/` is the control and the only difference
is head capacity.

| arm | change | asks |
|---|---|---|
| `b1_hidden512` | `--hidden 512` | is the 256-d shared vector the bottleneck? |
| `b2_perdrug64` | `--per-drug-hidden 64` | or is it the total absence of per-drug capacity? Each drug gets its own `256 → 64 → 1` branch off the shared trunk |
| `b3_reg` | `--dropout 0.3 --weight-decay 1e-4` | how much of the gap is overfitting? (`--weight-decay` > 0 switches Adam → AdamW) |
| `b4_all` | all three | do they compose, or overlap? Only interpretable against b1–b3, which is why those exist |

Parameter counts as submitted, against the `full_run` control:

| arm | late_fusion | cisfusion | vs control |
|---|---|---|---|
| *(full_run control)* | 45,898,955 | 47,955,211 | — |
| `b1_hidden512` | 91,810,251 | 93,784,587 | ~2× — `fc1` is 178k×256 → ×512 |
| `b2_perdrug64` | 46,077,771 | 48,134,027 | **+178,816 (+0.39%)** |
| `b3_reg` | 45,898,955 | 47,955,211 | **identical** — dropout/AdamW add no parameters |
| `b4_all` | 92,166,475 | 94,140,811 | b1 + b2 |

`b3_reg` matching the control to the parameter is also the cleanest available
confirmation that the 2026-08-06 refactor left the model structurally unchanged:
same architecture, same count, only the regularization differs.

`b2` is the interesting number — per-drug capacity for **0.39%** more weights,
exactly `11·(256·64+64) + 11·65 − (256·11+11)`. If it moves CV at all, the shared
output layer was the constraint and the fix is nearly free.

## Reading it

```bash
python - <<'EOF'
import json, glob, os
base = "../full_run"
for p in sorted(glob.glob("*/multidrug__*.json")):
    j = json.load(open(p))
    arm, cell = p.split("/")[0].split("_multidrug_")
    ctrl = json.load(open(f"{base}/multidrug_{cell}/multidrug__{j['tag']}.json"))
    d = j["cv_macro_auc_mean"] - ctrl["cv_macro_auc_mean"]
    print(f"{arm:16s} {cell:28s} cv={j['cv_macro_auc_mean']:.4f} "
          f"(full_run {ctrl['cv_macro_auc_mean']:.4f}, {d:+.4f}) "
          f"params={j['n_params']/1e6:.1f}M epochs={[f['best_epoch'] for f in j['cv_folds']]}")
EOF
```

**Fold SD in `full_run` was 0.003–0.030 against the 0.0116 gap this project is
trying to close.** One seed, five folds. An arm that gains under ~0.01 has not
been shown to do anything; treat b1–b4 as a screen for which direction is worth
multi-seeding, not as a result. See the closing section of
`../CODE_CHANGES_20260806.md`.

## Weights

`--save-weights all` (5 folds each) to
`/project/pi_mfiterau_umass_edu/abr_model_weights/joint_capacity/{arm}_multidrug_dna_protein__{arch}/`,
each with a `config.json` that fully rebuilds the model. The per-drug-head and
wide-head configs round-trip bit-identically (`tests/test_checkpoint.py`).

## Caveat to carry forward

If `../joint_convergence/` shows the joint models were undertrained at 150
epochs, these arms measured capacity in that regime. Capacity and training
budget interact — a bigger head may need more epochs to pay off — so the final
configuration must combine the winner here with the winner there and be
**re-validated jointly**, not assumed additive. Running both changes in one
sweep now would have made neither attributable, which is why they are separate.
