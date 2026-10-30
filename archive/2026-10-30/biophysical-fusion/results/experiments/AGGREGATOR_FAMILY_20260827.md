# The variant-set aggregator family — `models/experimental_models.py`

**Written 2026-08-27.** Covers the six aggregators, their motivations, their
measured results across two runs, and a recommendation on which of them belong
in a write-up.

Runs this document reads:

| run | loci | what it holds |
|---|---|---|
| `results/experiments/variant_aggregators_20260825` | per-drug (1–3) | 5 aggregators + the L1/GBM sparse baseline |
| `results/experiments/variant_aggregators_alllocus` | all 19 | 6 aggregators + the 19-locus sparse baseline |
| `results/archive/noisyor_polarity_bug_20260825` | per-drug | the pre-fix `noisyor` results, archived |

All AUCs below are 5-fold CV, `all_modalities`, identical protocol to
`training.multimodal.run_modal_cv`.

---

## 1. What the file is for

Everything in `net.py` and `locusfusion.py` answers **"how do I read a genomic
block"** — CNN, transformer, patch embedding. These six answer a different
question, and they exist because of what the project's own census says the
problem actually is:

> Censused over 19 loci × 17,943 isolates against each alignment's `MT_H37Rv`
> row, **the median isolate differs from the reference at 0–3 columns of a 2.5 kb
> gene, and at 14 columns across all 19 loci.**

Once you tokenize those differences, **there is no long sequence left to
encode.** The task becomes:

> given a set of ~14 deviations from wild type, most of them neutral lineage
> markers, decide R/S — from 11.5 k labelled training isolates.

That is **sparse evidence aggregation**, not sequence encoding. The encoder
question is already answered; the aggregator is what is left.

### 1.1 The argument against softmax attention

**Softmax normalises, so it is a *relative* selector.** With one informative
token and thirteen neutral ones the weights must sum to 1, so the operator
*cannot* say "this token, absolutely, whatever else is present" — it has to
spend mass on the neutral tokens. That is the mechanism behind the flat 1/8
attention `results/experiments/token_signal` measured. **It is a structural
property of the operator, not a training failure**, which is why three
transformer arms failed and re-tuning them did not help.

A needle detector wants an **absolute, monotone** aggregator: adding a neutral
token must not dilute the signal.

So: same tokenizer, six deliberately far-apart aggregators.

---

## 2. Shared substrate

All six consume an identical, **flat** variant token set — every block's
variants concatenated into one set per isolate, **no locus hierarchy.** That is
a deliberate contrast with `LocusFusionNet`, which groups by locus and fuses in
two stages. Here the grouping is held flat and constant so that **the only thing
varying across the six is the aggregator.**

### `VariantSet` (`experimental_models.py:109`) — zero parameters

Delta blocks → padded token set, `T = n_blocks × max_variants`:

| field | shape | what it is |
|---|---|---|
| `feat` | `(B,T,C_TOK)` | the `locusfusion` feature layout, imported not redefined |
| `valid` | `(B,T)` | which slots are real variants |
| `vid` | `(B,T)` | **exact** `(block, column, alt)` identity into a global vocabulary |
| `coord` | `(B,T)` | codon coordinate, negative upstream |
| `locus`, `modality` | `(B,T)` | |
| `uncovered` | `(B,n_blocks)` | a block differing from reference *everywhere* is a failed assembly, not a hypervariant one |
| `n_occ` | `(B,n_blocks)` | the **true** variant count, before the `max_variants` cap |

`vid` is exact rather than hashed — the vocabulary is `sum_b(L_b × C_b)`, ~48 k
for an isoniazid cell and ~527 k for all 19 loci with every modality. Small
enough to index directly, **so `catalogue` has no hash collisions to explain
away.**

### `VariantEmbedding` (`:222`)

`feature projection + sinusoid(coord) + locus + modality`, LayerNorm, then
zeroed at padded slots so a pooled sum is not polluted. The coordinate encoding
is the parameter-free continuous one: a nucleotide column lands at *k*, *k*+1/3
or *k*+2/3, **so the fractional part is the codon phase and exact position
survives.**

### `_VariantModel` (`:253`)

Base class holding tokenizer, embedding, `forward`, `from_blocks`, drug names,
and one shared extra: a learned `uncovered_w` per `(block, drug)`, because **a
missing gene is not a wild-type gene, and no other architecture in this project
distinguishes them.**

Subclasses implement exactly one method: `aggregate(h, v) -> (B, n_drugs)`.

### Two guardrails

* **All six require `--delta` and per-locus blocks.** On dense input every
  column is occupied, the cap keeps the first `max_variants` columns of each
  block, and the model **silently degenerates while still producing plausible
  numbers.**
* The factory **rejects a knob passed to a model that does not use it** —
  silently ignoring it would make a sweep arm look like it ran when it was
  really the control under a different folder name.

---

## 3. The six variants

### 3.1 `catalogue` — pure memorisation. The control.

```
logit_j = bias_j + Σ_v w[vid_v, j]
```

A learned scalar per **exact** variant identity, summed. `needs_embedding =
False` — it never sees a feature.

**Motivation.** This *is* logistic regression on the isolate × variant presence
matrix, expressed as a module so it trains in the existing loop against the
existing protocol. It is also **a WHO-style resistance catalogue, learned rather
than curated** — which is what TB-Profiler and Mykrobe actually are clinically.

**The point of it.** The weight table is zero-initialised, so **a substitution
that never appears in training contributes exactly nothing at test time.**
`catalogue` can only recognise what it has already seen. It is the *control for
generalisation*, not a contender. Zero init also means an untrained model sits
exactly at the base rate, and `catalogue_entries()` reads the learned table by
name — "katG column 944 → +2.1 toward RESISTANT".

### 3.2 `additive` — featurised additive variant-effect model

```
logit_j = bias_j + Σ_v w_j(features_v),   w = MLP(embedding)
```

**The same aggregator as `catalogue`**, but the per-variant weight comes from
*what the variant is* — locus, exact position, which base or residue it became,
biophysical consequence — rather than from its identity. **So a substitution
never seen in training still gets a weight.**

**Motivation.** `catalogue` and `additive` are a **matched pair, and the
difference between them is a measurement.** The stated hypothesis was *pncA*
loss-of-function for pyrazinamide — hundreds of distinct inactivating
substitutions, most absent from training; "does this substitution break the
protein" generalises where memorising positions cannot. This is the smallest
architecture that can express that. *(Section 5.1: the measurement came back
saying something different, and better.)*

Three further properties, all consequences of staying additive:

* **Exactly interpretable with no attribution machinery.** `contributions(xs)`
  returns signed per-token contributions that sum to the logit. No SHAP, no
  sampling error, no convergence question — which matters given finding 3
  (attribution share badly mis-ranks predictive value). *An additive model does
  not need an attribution method; it is one.*
* **A susceptible isolate is the empty sum**, so the prior for "no deviations
  from H37Rv" is the learned bias, i.e. the base rate.
* **It forbids epistasis by construction.** A real restriction — and also what
  makes it sample-efficient at 11.5 k isolates.

`residual_catalogue=True` adds the exact-identity table on top: memorise what
you have seen, featurise what you have not. Off by default so the clean
measurement comes first. **Never run.**

### 3.3 `noisyor` — multiple-instance learning. "Susceptible unless."

```
log P(S) = logsigmoid(-z_0) + Σ_v logsigmoid(-z_v)
logit    = S - log(-expm1(S))
```

**The biology as an architecture.** Resistance is conferred if *any* variant
confers it — a noisy-OR over independent causes, which is literally the guidance
received: *"a susceptible isolate is the one with nothing in it, and anything
else is resistant."* A wild-type isolate has an empty product and falls back to
the learned background.

**Why this rather than a sum:**

* **Absolute and monotone.** Every `p_v` is judged on its own; adding a neutral
  variant cannot dilute a resistance variant the way a softmax must.
* **It saturates.** Two independent resistance mutations give "resistant, twice
  over", not twice the logit — which is what an additive model gets wrong.
* `p_v` reads directly as **"probability this variant confers resistance"**,
  comparable against the WHO catalogue's own confidence gradings.

**The label convention is load-bearing here, and the first version got it
wrong.** This project encodes **R=0, S=1** (`training/multimodal._metrics`:
`n_R = (y == 0).sum()`), so a higher logit means *more likely susceptible*. A
noisy-OR is monotone increasing in its evidence, so pointing it at P(resistant)
structurally forces every variant-carrying isolate toward the wrong class — and
since resistant isolates carry more variants, it comes out **anti-predictive:
macro CV 0.4956, below chance**, with train loss moving 0.3152 → 0.3023 over 99
epochs against `additive`'s 0.2246 → 0.0875 on the same cell. Compounded by a
`-4.0` output-bias init, where `d/dz logsigmoid(-z) = -sigmoid(z)` attenuates
every gradient by ~50×, on top of BIG-TB's `lr = exp(-9)`. Fixed: the product is
now P(susceptible), init is `-2.0`. Pre-fix results archived to
`results/archive/noisyor_polarity_bug_20260825/`.

**Two structural restrictions, both real costs:**

* **It literally cannot learn a protective variant.** Every factor is in (0,1),
  so adding a variant can only raise P(R). Right prior for resistance; but a
  lineage marker correlating with susceptibility is invisible to it.
* **It assumes causes are independent** — cannot express compensatory epistasis
  (*rpoB* + *rpoC*) or dose-dependence.

### 3.4 `gatedpool` — attention with the softmax deleted

```
pooled_j = Σ_v sigmoid(g_j(h_v)) · V(h_v)     + a max branch
```

**The minimal edit to the thing that failed.** `setfusion` and `locusfusion`
read out with a softmax attention query; this keeps the per-drug gate and
**deletes the normalisation**, so the gate is an *absolute* relevance score
rather than a share of a fixed budget. A sigmoid gate can open for one token and
stay shut for the rest.

The max branch answers *"did any single variant fire hard"* — the needle
question — while the gated sum answers *"how much total evidence is there."*
They disagree exactly when an isolate carries several weak variants, and the
head decides.

**Its decision rule, written before the run:** *"if `gatedpool` closes a gap
that `deepsets` does not, normalisation was the problem; if `deepsets` matches
it, the gating was never doing anything."*

`gates(xs)` returns the un-normalised analogue of an attention map, directly
comparable against `locusfusion`'s, which is forced to sum to 1.

### 3.5 `deepsets` — no attention at all

```
rho([ Σ phi(h_v), max phi(h_v), log1p(count) ])
```

**The ablation the project had never run: does attention buy anything over plain
permutation-invariant aggregation?** No gate, no query, no normalisation. It is
the honest null model for the family.

`setfusion` and `locusfusion` both spend most of their parameters on machinery
whose only job is deciding *which* tokens matter. If summing them and taking a
max does as well, **that machinery is not earning its place — and that finding
is worth more than a fourth-decimal AUC gain.**

The count is deliberate: "how many deviations from H37Rv does this isolate
carry" is a real feature (lineage divergence, assembly quality) that a pure sum
conflates with effect size. It is fed from `n_occ`, **the true count before the
cap** — the one quantity the tokenizer's truncation can distort.

### 3.6 `fm` — pairwise interactions, priced linearly

```
logit = bias + Σ_v w_v + W2 · [ ½ ( (Σ_v e_v)² − Σ_v e_v² ) ]
```

The classical factorization-machine trick: the second term is **every pairwise
interaction** `Σ_{u<v} <e_u, e_v>`, computed in `O(T·rank)` instead of `O(T²)`.
So it asks **"is epistasis worth anything here?"** without paying attention's
quadratic cost *or* its normalisation problem.

**Motivation.** It is the right way to relax `additive`'s hard restriction.
Compensatory resistance is real — *rpoC* mutations restoring fitness lost to
*rpoB* RRDR mutations is the textbook case, and both loci are in the 19 — but
the project's own finding 1 says the signal is dominated by a handful of
specific positions, so interactions should be second-order, and an FM prices
them accordingly.

Factors come from **features, not a per-identity table**, so an unseen
substitution still participates in interactions — the `additive`-over-
`catalogue` argument applied one order up. A per-identity factor table would be
`vocab × rank` = 4 M parameters at 19 loci, almost all never trained.

---

## 4. Results

`all_modalities`, 5-fold CV AUC, macro over 11 drugs.

| model | 2–3 loci | params | 19 loci | params |
|---|---:|---:|---:|---:|
| `gatedpool` | 0.8904 | 98 k | **0.9163** | 100 k |
| `deepsets` | **0.8906** | 115 k | 0.9154 | 117 k |
| `additive` | 0.8892 | 48 k | 0.9098 | 50 k |
| `fm` | 0.8897 | **16 k** | 0.9025 | **18 k** |
| `catalogue` | 0.8672 | 52 k | 0.8854 | 445 k |
| `noisyor` | *— crashed, §6* | — | 0.8629 | 50 k |
| | | | | |
| `locusfusion` | 0.8920 | 646 k | 0.9242 | 659 k |
| `mdcnn` | 0.9086 | 3.5 M | **0.9246** | 6.7 M |
| MD-CNN baseline | — | | 0.9222 | |
| L1-logistic | 0.8788 | *none* | 0.8925 † | *none* |

† 7/11 drugs at time of writing.

**Headline: not one of the six wins at either locus setting.** `mdcnn` and
`locusfusion` are ahead of the best aggregator by 0.008 at 19 loci and 0.018 at
per-drug loci.

`catalogue` is the only model in the family whose parameter count scales with
the data (52 k → 445 k), and it is the worst of the five that ran at both
settings.

---

## 5. Which of these belong in a write-up

**Not as a model comparison. Yes — as three measurements.** Presenting all six
as a leaderboard would be the worst version of this: six models nobody asked
about, none of which wins, and the reader's takeaway is "a lot of things were
tried that did not work."

### 5.1 REPORT — `catalogue` → `additive`. The strongest result in the file.

**+0.0220 at per-drug loci, +0.0244 at 19 loci**, and it is a controlled
measurement — same tokenizer, same aggregator, same optimizer, one thing
changed: whether the variant's weight comes from its *identity* or its
*features*.

**The stated motivation turned out to be wrong, and the truth is better.** The
hypothesis was *pncA*/pyrazinamide. Pyrazinamide's gain at 19 loci is
**+0.0049.** The gain is not about *pncA* at all — it is almost perfectly a
**small-cohort effect**:

| drug | n | gain (19 loci) | gain (per-drug loci) |
|---|---:|---:|---:|
| LEVOFLOXACIN | 269 | **+0.0696** | +0.0388 |
| ETHIONAMIDE | 2,153 | +0.0290 | +0.0388 |
| MOXIFLOXACIN | 2,868 | +0.0386 | +0.0411 |
| KANAMYCIN | 3,576 | +0.0448 | +0.0035 |
| AMIKACIN | 3,705 | +0.0366 | +0.0287 |
| CAPREOMYCIN | 3,854 | +0.0324 | +0.0279 |
| STREPTOMYCIN | 7,682 | +0.0087 | +0.0323 |
| PYRAZINAMIDE | 12,904 | +0.0049 | +0.0250 |
| ETHAMBUTOL | 15,111 | +0.0032 | +0.0035 |
| ISONIAZID | 17,436 | +0.0015 | +0.0039 |
| RIFAMPICIN | 17,582 | −0.0011 | −0.0011 |

**Spearman(n, gain) = −0.900, p = 0.0002** at 19 loci (−0.743, p = 0.009 at
per-drug loci).

The claim to make: *featurising a variant instead of memorising its identity
buys generalisation, and the size of that gain is set almost entirely by how
little labelled data the drug has.* Above ~8 k isolates a learned catalogue has
seen enough; below it, features matter, and at n = 269 they are worth 0.07 AUC.
That is a clean, quantitative, mechanistic result about **sample efficiency**,
and it says something useful about deploying to a new drug or a new region where
labels are scarce.

### 5.2 REPORT — `gatedpool` ≈ `deepsets`. The attention ablation.

The decision rule was written before the run. The verdict:

| | 2–3 loci | 19 loci |
|---|---:|---:|
| `gatedpool` (sigmoid gate, no softmax) | 0.8904 | 0.9163 |
| `deepsets` (no gate at all) | 0.8906 | 0.9154 |
| **difference** | **−0.0002** | **+0.0009** |

Both differences are ~5× smaller than the average per-drug fold SD (0.014).
**Deleting the gating machinery entirely costs nothing.** Combined with
`token_signal`'s flat 1/8 attention, the finding is: *on this task the selection
mechanism is not where the performance comes from — a sum and a max over the
same tokens does the same job.*

Worth reporting precisely **because** it is negative. It explains why three
transformer arms failed, and it means the deep-learning contribution here is the
*representation* (tokenize the variants, featurise them), not the *attention*.

### 5.3 REPORT BRIEFLY — the parameter numbers.

`additive` reaches **0.9098 with 50,365 parameters**, beating every
`late_fusion` cell at 19 loci (best 0.9082, **45.9 M parameters**) — a 900×
reduction. `fm` gets 0.9025 from **18,254 parameters**, 2,500× smaller and 0.006
behind.

The sharpest possible version of "the signal is sparse and mostly additive." One
sentence, one number, then move on.

### 5.4 DESIGN NOTE ONLY — `fm`.

Worse than `additive` at 19 loci (0.9025 vs 0.9098, **−0.0073**) and identical at
per-drug loci (+0.0005). **No evidence that epistasis is worth modelling here** —
the rank-8 interaction term costs more in variance than it recovers in signal,
exactly as finding 1 predicted. One line in the ablation table.

### 5.5 METHODS NOTE ONLY — `noisyor`.

Worst of the six at 0.8629, and catastrophic on LEVOFLOXACIN (**0.6096**,
n = 269). The monotone constraint is a real cost — it cannot learn a protective
variant, and at small n the resulting bias dominates.

There is a good story here, but it is a **methods** story, not a results one: the
R=0/S=1 convention silently inverted a monotone architecture and produced a
below-chance model that trained without error, and it was caught by reading the
training curve rather than the AUC. Tell it if writing about rigour. Not worth a
bar on a chart.

### 5.6 DO NOT REPORT — `residual_catalogue`.

Never run. The docstring calls it "likely the strongest single model in this
file." Either run it or delete the claim — do not leave an unrun hypothesis in a
file being presented.

### 5.7 Suggested framing

One subsection, **"Ablating the aggregator"**, containing:

1. the softmax-normalisation argument (2 sentences — it is the intellectual
   contribution),
2. one 6-row table,
3. the `catalogue` → `additive` cohort-size plot — **this is the figure worth
   making**,
4. one sentence on `gatedpool ≈ deepsets`.

Place it **after** the main results, as *why the architecture is what it is*,
never as a competing leaderboard.

---

## 6. Open gaps

### 6.1 `noisyor` has no per-drug-loci number

All 11 rerun jobs (`63647994`–`63648007`) reported `COMPLETED` in 34 s–4 min but
wrote nothing. Each died with:

```
RuntimeError: CUDA error: an illegal memory access was encountered
  models/experimental_models.py:175  in VariantSet.forward
  models/locusfusion.py:189          in _select_variants  ->  score.topk(k, dim=1)
```

Diagnosis so far — the exact cell reproduces **clean on CPU**:

```
noisyor:  CPU forward OK (256,1) range[-4.874,0.000]  vid_max=46622/47825  locus_max=1/2  valid=0.0254
additive: CPU forward OK (256,1)                      vid_max=46622/47825  locus_max=1/2  valid=0.0254
```

No out-of-bounds index. The identical code path also ran clean for `noisyor` at
19 loci and for the other five aggregators at these same loci. The failing jobs
all landed on `gypsum-gpu083`; the successful 19-locus batch went to
`gypsum-gpu020`. **Reads as node/driver-specific, not a model bug — a plain
resubmit should clear it.**

Fix worth making regardless: **a job that crashes should not exit 0.** The
sbatch wrapper swallowed 11 tracebacks into clean exit codes and an empty output
directory.

### 6.2 No joint (13-drug) arm exists for any of the six

`locusfusion` and all three CNNs have one; the aggregators do not. If the report
claims anything about multi-task sharing, the aggregators cannot be in that
claim.

### 6.3 `residual_catalogue` never run

See §5.6.
