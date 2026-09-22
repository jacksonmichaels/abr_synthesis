# newmodels_pilot — does either new model earn a full sweep?

Submitted **2026-08-25**, 7 SLURM jobs, reproducible via `submit.sh` in this
folder.

## Why this exists

Two models have never been trained on real data:

| model | added | state before this run |
|---|---|---|
| `locusfusion` (`--arch locusfusion`) | 2026-08-25 | built, `tests/test_locusfusion.py` 25/25, **zero training runs** |
| `branched` (`--head branched`, `models.BranchedHead`) | 2026-08-25 | built, `tests/test_branched.py` 25/25, **zero training runs** |

A full sweep of both across the project's three tasks is ~130 jobs and several
hundred GPU-hours. This is 7 jobs on the two cheapest cells that still have an
exact matched control, and it exists to decide whether to spend that.

**`full_run_v2` is the control for both arms.** Same 300 epochs / patience 30 /
`--min-epochs 50` / `--save-weights best` / seed 0, so the only difference in the
locusfusion arm is `--arch` and in the branched arm is `--head`.

## The branched head, and what it is NOT

`models.BranchedHead` is the Learn-to-Branch mechanism from Luo et al., *Sci Rep*
14:6631 (2024) — Branched CALM-Net — with **subject replaced by drug**. A shared
trunk feeds G group nodes; each drug learns a categorical θ over them, sampled
with Gumbel-softmax at a temperature annealed 5.0 → 0.5 so it collapses toward
one-hot. Drugs that end on the same node **are** the discovered cluster. A
generic head trained on every drug's labels is the cold-start path.

**It is a head, not an architecture** — it composes with `late_fusion`, `mdcnn`
and `cisfusion` unchanged. `setfusion` and `locusfusion` build their own
read-outs and the runner refuses the combination rather than silently ignoring
it.

**It only means anything on the multi-drug task.** With one output there are no
tasks to group, so `make_head` returns a plain `DenseHead` and `--head branched`
is a documented no-op on both single-drug tasks. Any future "branched × 3 tasks"
grid is really "branched × 1 task".

### Why the measurement is worth making

From the finished runs, multi-drug minus single-drug-at-19-loci, paired over the
15 (modality × model) cells that exist on both sides:

| | macro | spread | ρ(gain, n_isolates) |
|---|---:|---:|---:|
| Δ from multi-task sharing | **+0.0012** | **0.028** | **−0.47** |

Per drug that is LEVOFLOXACIN **+0.0165** and ETHIONAMIDE **+0.0139** against
KANAMYCIN **−0.0119** and PYRAZINAMIDE **−0.0065**. Uniform sharing helps the
small drugs, dilutes the large ones, and cancels to nothing in the macro. That
cancellation — not an absence of transfer — is what routing targets.

### The bug this pilot already avoided

The head as first written **did not learn**. Probed on a synthetic task with
known ground truth (8 drugs, 4 loci, drugs 0–3 driven by locus 0 and 4–7 by
locus 1, so the true grouping is a 4/4 split), soft Gumbel mixing left
`softmax(θ)` at 0.44/0.56 after 60 epochs, mean `|∇θ| ≈ 5e-4`, and recovered an
arbitrary 1/7 split. Every group node was receiving gradient from every drug, so
the nodes never specialised and the loss was flat in θ — the same
"mechanism present but inert" failure `token_signal` diagnosed in setfusion, and
it would have surfaced as *"branching does nothing"* after three GPU-hours
instead of ninety seconds.

Three changes, all defaults now: **straight-through hard routing**
(`gumbel_softmax(hard=True)`, so a group node is updated only by the drugs
routed to it), **symmetry-breaking init** on θ, and a **×10 LR multiplier** on θ
(architecture parameters get gradients orders of magnitude smaller than weights;
the standard DARTS treatment). `tests/test_branched.py` asserts the disjoint-
gradient property directly, so the inert version cannot come back silently.

## The arms

### locusfusion — 2 jobs

`all_modalities__locusfusion`, single-drug. `all_modalities` is the right first
cell because stage 1 fuses a locus's modalities *before* the loci are fused — a
DNA-only run cannot show what the architecture is for.

| drug | n | why this drug | control (`full_run_v2/all_modalities__mdcnn`) |
|---|---:|---|---:|
| ISONIAZID | 17.4k | the project's core question; where the `token_signal` fix was measured | **0.9645** |
| MOXIFLOXACIN | 2.9k | every `token_signal` collinearity diagnostic was run here | **0.8680** |

### branched — 5 jobs

Multi-drug `dna__mdcnn` — the cheapest joint cell (2.23 h at 300 epochs, 3.87 M
params) and the one the sharing measurement above was made on. Control:
`full_run_v2/multidrug_dna__mdcnn`, macro CV **0.9179**.

| arm | flags | what it is |
|---|---|---|
| `c0_dense` | — | the joint baseline, reproduced at identical settings |
| `c1_perdrug64` | `--per-drug-hidden 64` | **the bar.** Same per-drug capacity, no routing |
| `b2_branch_g2` | `--head branched --branch-groups 2` | smallest split that can express anything |
| `b3_branch_g3` | `--branch-groups 3` | |
| `b4_branch_g4` | `--branch-groups 4` | the paper's default |

`c1_perdrug64` is the comparison that matters. Branching also adds per-drug
MLPs, so beating plain `DenseHead` would confound routing with capacity;
`joint_capacity`'s `b2_perdrug64` already put that arm at **−0.001**, so the
capacity half is known to be worth nothing on its own. G=11 (one group per drug
= no sharing) is deliberately not run — that is the single-drug grid.

## How to judge it

**locusfusion — three things, not one** (the claim is mechanistic, so hold it to
the discipline `token_signal` held itself to):

1. `cv_auc_mean` against the control above.
2. The read-out attention must stop being uniform. Flat 1/n_loci is the same
   collinearity that killed setfusion.
3. `variant_report()` names the locus and column behind every attended token —
   for ISONIAZID the top token should be *katG* codon 315 or the *fabG1–inhA*
   promoter `c-15t`. An arm that raises AUC while attending to neither improved
   for some other reason.

**branched — two things:**

1. macro CV against `c1_perdrug64`, not against `c0_dense`.
2. `branch_assignments` in the result JSON must be a non-trivial partition, and
   it should be checked against **both** mechanism (KAN/AMK/CAP share *rrs*;
   LFX/MOXI share *gyrA*/*gyrB*; INH/ETO share the *fabG1–inhA* promoter) **and**
   phenotype correlation (KAN/AMK co-occur at φ=0.88). Matching mechanism better
   than co-resistance is the interesting result; matching co-resistance is a
   warning that the model is exploiting MDR structure that will not transfer.
   A partition that puts 10 of 11 drugs in one group means routing collapsed and
   the arm is `c1_perdrug64` with extra parameters.

## What this pilot does NOT do

- **No modality sweep, no locus sweep, no seeds.** One cell each. Single seed,
  so nothing under ~0.01 is resolvable — this is a go/no-go on the two models,
  not a result.
- **No branched × locusfusion.** `LocusFusionNet` has per-drug attention queries
  as its read-out; the runner raises rather than ignoring `--head branched`.
- **No cold-start arm.** The generic head is trained (`--branch-generic-weight`
  defaults to 0.3) but leave-one-drug-out is a separate experiment.
