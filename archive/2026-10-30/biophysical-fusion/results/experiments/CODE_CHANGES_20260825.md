# `--arch locusfusion` — a variant-token, two-stage transformer. 2026-08-25

Everything here is additive. No existing default changes, `models/net.py` is
untouched, and every prior checkpoint still loads (`tests/test_locusfusion.py`
asserts the config round-trip, including a config written before this arch
existed). One fixture change is not additive and is called out in §6.

---

## The question this answers

Three transformer arms have now lost in this project, and they lost for reasons
that are written down rather than guessed at:

| arm | result | the diagnosed reason |
|---|---|---|
| `full_run_v2/*__setfusion` | macro 0.8877 vs 0.9086 for `mdcnn` | `token_signal`: **0.14% of an encoded token varies with the genotype**; attention flat at 1/8; a linear probe on the encoder output beat the trained model by 0.011 |
| `setfusion_scaling` (62 arms) | never closed the gap | it was never a width problem |
| `transformer_run` (165 jobs) | DNA-only 0.802 vs 0.892 | patch embedding replaces the flatten, so absolute position is coarsened to a pooled summary |

Two of those three are the *same* failure seen from different angles: a
transformer over a **dense** genomic one-hot spends its tokens on sequence that
is identical in every isolate, and pays for the privilege by giving up the
flatten that carried exact position.

So do not tokenize the sequence. **Tokenize the difference from the reference.**

## The measurement the design is built on

A variant census over all 19 curated loci x 17,943 isolates, each against the
`MT_H37Rv` row in its own alignment (reproduce with the snippet in `README.md`'s
locusfusion section):

| block | L | median diffs | p90 | p99 | >32 |
|---|---:|---:|---:|---:|---:|
| dna:rpoB | 3,716 | 1 | 3 | 5 | 0.00% |
| dna:katG | 2,488 | 1 | 2 | 3 | 0.13% |
| dna:gyrA | 2,620 | 3 | 5 | 7 | 0.00% |
| dna:rrs | 1,813 | 0 | 1 | 26 | 0.80% |
| dna:pncA | 1,067 | 0 | 1 | 5 | 0.54% |
| **all 19 loci** | **~40 kb** | **14** | **26** | **214** | **4.38%** |

*M. tuberculosis* is clonal. **The median isolate differs from the reference at
0-3 columns of a 2.5 kb gene**, and at 14 columns across all 19 loci. A
patch-embedding transformer over that input is spending ~99.9% of its capacity
restating constants; the `token_signal` number is what that looks like from the
inside.

The long tail is not variation. The p99.9 column (katG 285, ethA 576, pncA 761)
is **records that failed to assemble**: 14-91 isolates per locus are all-gap,
which under delta encoding differs from the reference at every column. Those are
missing genes, not hypervariant ones, and §3 says what the model does with them.

---

## 1. `models/locusfusion.py` — `LocusFusionNet`

```
per locus L, per coordinate stream:
    delta block (B, C, len)  --occupancy-->  the columns that differ
                             --gather-->     <= max_variants tokens
token   = tok_proj(modality slots + flags) + pos_proj(sinusoid(coord))
        + locus_emb[L] (+ FiLM adapter[L])
[WT]_L  = wt_emb[L] + wt_proj(variant count, coverage, uncovered)

stage 1   TransformerEncoder over {[WT]_L, variants of L}   -> z_L = out[[WT]]
          KeyedTokenNorm(z_L, key=L)
stage 2   TransformerEncoder over {z_L : every locus}        -> fused
readout   one learned query per drug, cross-attending the fused set
```

**A susceptible isolate is the empty set.** Each locus carries a learned `[WT]`
sentinel — the wild-type null — and every variant token is evidence against it.
A pan-susceptible isolate presents 19 sentinels and nothing else. That is the
hypothesis shape the biology suggests ("the model may be learning what a
sensitive strain looks like, and calling everything else resistant") made
structural instead of accidental, and it is also what keeps a stage-1 attention
row from ever being fully masked.

**Fuse at the gene, then across genes.** Every other arch fuses modalities and
loci in the same step — `late_fusion` at one flatten, `mdcnn` at layer 1,
`setfusion` in one transformer over one token per block. Here all of *rpoB*'s
evidence becomes one locus representation first, and only those talk to each
other. A resistance *mechanism* is a property of a gene; a resistance
*phenotype* is a property of the set of genes.

**Protein and biophysical fuse into ONE token per changed residue.** They come
off the identical translate path with the same `k_max`, so they are exactly
co-indexed: the occupancy is their union and each writes into its own feature
slot at the same selected column. One token then says *"katG residue 315, now
Threonine, and here is what that does to MW / pI / hydrophobicity"*.

**What survives that the pooled archs gave up:**

* *Exact position.* A sinusoidal encoding of a continuous codon coordinate, so
  "column 315 of katG" is preserved to the nucleotide — better than `mdcnn`
  (pools 9-fold) and unlike `setfusion` (4 relative bins per locus).
* *Cheap attention.* Median ~14 tokens per isolate against ~1,300 patches.
* *Exact attribution.* `forward(..., return_attn=True)` plus `variant_report()`
  reads as "drug j attended to katG alignment column 944" — checkable against
  the WHO catalogue without SHAP.

### The one approximation, stated plainly

`datasets/protein.py` DEGAPS each isolate before translating, so protein codon
*k* is the k-th codon of that isolate's own degapped CDS, while the DNA block
stays in shared alignment-column space. The two agree exactly when no indel sits
upstream — the overwhelming majority of a clonal cohort — and drift by the indel
length when one does. **So nt and aa tokens are NOT fused by position.** They
stay separate tokens that within-locus attention may pair, and the constant part
of the offset (where the CDS starts inside the aligned record) is a **learned
per-locus scalar** rather than an assumption. `_test_streams_stay_separate_
across_coordinate_systems` pins this.

## 2. `summary_norm="keyed"` — on by default, and why

A locus summary is read off the `[WT]` slot, and for a wild-type isolate that
slot's *input is identical in every isolate*. Measured on 400 real ISONIAZID
isolates at init, reproducing `token_signal`'s metric:

| stage-2 input | per-isolate spread / mean norm |
|---|---:|
| `summary_norm="none"` | **3.10%** |
| `summary_norm="keyed"` | **9.17%** |

Without the norm this is the same shape of failure `token_signal` diagnosed in
setfusion at 0.14% — a per-locus constant the readout absorbs into its
intercept. `KeyedTokenNorm` (reused from `models/net.py`, keyed by locus here
rather than by (modality, locus)) deletes it. That intervention is worth +0.033
CV AUC on ISONIAZID for setfusion (`token_signal/a1_tokennorm`: 0.9287 ->
0.9621), which is why it is the default rather than a knob.

Read the 9.17% for what it is: a ratio at **random init**, not a claim about a
trained model, and 100% of the model's *input* varies with the genotype by
construction regardless. `summary_norm="none"` creates no module and adds no
`state_dict` keys.

## 3. Uncovered != wild type

Under delta encoding an all-gap record differs from the reference at every
column, so without a flag a gene that failed to assemble reads as the
most-mutated isolate in the cohort. A locus whose occupancy exceeds
`uncovered_frac` (0.5) sets `F_UNCOVERED` on every one of its tokens and reports
it on the `[WT]` token's statistics. No existing architecture in this project
distinguishes those cases at all.

The `[WT]` token also carries `log1p(true variant count)` — before the cap — so
a frameshift that changes 42 downstream residues is not silently read as 16.

## 4. `--delta` is implied, not requested

`--arch locusfusion` turns on reference-difference encoding in both runners. On
a plain one-hot every column is occupied, the cap keeps the first
`max_variants` columns of each block, and the model degenerates into "read the
head of each block" *while still producing plausible numbers*. That is a silent
failure, so `forward` also checks occupancy on the first batch and warns.

## 5. Plumbing

| file | change |
|---|---|
| `models/locusfusion.py` | new — `LocusFusionNet`, `LOCUSFUSION_DEFAULTS`, `LOCUS_ENCODERS`, `SUMMARY_NORMS` |
| `models/__init__.py` | re-exports; `ARCHITECTURES` += `locusfusion`; new `PER_LOCUS_ARCHS` / `DELTA_ARCHS` tuples so the two runners stop hardcoding the same list |
| `training/multimodal.py` | `_build_model` branch, `run_modal_cv(locusfusion=...)`, knobs recorded in the result JSON |
| `training/multidrug.py` | the same, joint side |
| `training/checkpoint.py` | `model_config(locusfusion=...)` + a `build_model_from_config` branch; `.get`, so a config with no `locusfusion` key rebuilds at the defaults |
| `datasets/multidrug.py` | `load_multidrug_dataset(delta=...)` — the single-drug loader had it, the joint one did not, so `--delta` was unreachable from `run_multidrug.py` |
| `scripts/run_{experiment,multidrug}.py` | `--lf-*` argument group (guarded: the flags error unless `--arch locusfusion`), per-locus + `--delta` implication |
| `tests/test_locusfusion.py` | new — 25 checks |

## 6. The one non-additive change: `datasets/fixtures.py`

`make_gene_fixture` now also writes the unmutated sequence as an `MT_H37Rv`
record. Without a reference row `--delta` is a silent no-op on fixtures, so
`--arch locusfusion` could not be smoke-tested synthetically at all.

It is not in the phenotype fixture, so it drops out of every dataset exactly as
the real reference row does (17,942 of 17,943 rpoB isolates intersect) — the
fixture is now *more* faithful, not less. `tests/test_baseline_alignment.py`,
`test_checkpoint.py`, `test_cisfusion.py` and `test_transformer_encoder.py` all
still pass; `test_setfusion.py` has one pre-existing failure (its
"defaults are the full_run configuration" check has not been updated since
`token_norm` was added) that is unrelated and unchanged by this work.

## 7. The run this wants

`full_run_v2` is the matched control (300 epochs, patience 30, `--min-epochs 50`,
`--save-weights best`, same seed, same per-drug loci), and `alllocus_run_v2` for
the 19-locus cells:

```bash
python scripts/run_experiment.py --modalities dna protein biophysical regulatory \
    --drugs all --arch locusfusion --epochs 300 --patience 30 --min-epochs 50 \
    --device cuda --run-name locusfusion_v1
```

`--delta` and per-locus branches come for free. Two things to check that are not
AUC, because the claim is mechanistic — the same discipline `token_signal`
imposed on itself:

* **read-out attention must stop being uniform.** Flat 1/n_loci means the
  summaries are still collinear and the architecture did not do its job.
* **the attended tokens must be the right ones.** `variant_report()` names the
  locus and column of every token; for ISONIAZID the top-attended token should
  be katG codon 315 or the fabG1-inhA promoter `c-15t`. An arm that raises AUC
  while attending to neither has improved for some other reason.

Then the knobs, in the order the evidence says they matter: `--lf-summary-norm`
(the ratio), `--lf-carry-variants 2` (cross-locus epistasis — rpoB+rpoC
compensatory pairs), `--lf-locus-encoder per_locus`, and only then `--lf-d-model`.
Width is last on purpose: `setfusion_scaling` swept four width axes across 62
arms and closed nothing.


---

# The experimental aggregator family — `models/experimental_models.py`

Added the same day, on top of the tokenizer above. Six more `--arch` values,
additive to everything else.

## Why a second family rather than more knobs on `locusfusion`

`locusfusion` answers "tokenize variants instead of sequence". It does **not**
answer "and then how do you combine them" — it inherits `setfusion`'s softmax
read-out for that, which is the operator `token_signal` caught failing.

**Softmax normalises, so it is a relative selector.** With one informative token
and thirteen neutral ones the weights must sum to 1: it cannot say "this token,
absolutely, whatever else is present", it has to spend mass on the neutral
tokens. That is the mechanism behind the flat 1/8 attention, and it is a property
of the operator rather than a training failure. A needle detector wants an
**absolute, monotone** aggregator instead.

So this family holds the tokenizer fixed and varies only `aggregate()`. All six
share `VariantSet` and `VariantEmbedding` and import the token feature layout
from `locusfusion` (``SLOTS``, ``C_TOK``, the flag offsets) rather than
redefining it, so a contribution or gate from any of them is directly comparable
to a `locusfusion` attention weight. The one deliberate difference is that the
token set here is **flat** — every block's variants in one set, no locus
hierarchy — because the variable under test is the aggregator, not the grouping.

| `--arch` | aggregator | params (INH cell) |
|---|---|---:|
| `catalogue` | learned scalar per exact variant id | 29 k |
| `additive` | `sum w(features_v)` | 48 k |
| `noisyor` | `1 - prod(1 - p_v)` | 48 k |
| `gatedpool` | sigmoid gate, no softmax, + max branch | 98 k |
| `deepsets` | sum + max + count | 114 k |
| `fm` | factorization machine, rank 8 | 16 k |

## The three measurements the family is for

1. **`catalogue` vs `additive` — what does featurisation buy?** `catalogue`'s
   per-identity table is zero-initialised, so a substitution absent from training
   contributes exactly nothing; `additive` is the identical aggregator with the
   weight computed from the variant's features. The gap is the *pncA* /
   PYRAZINAMIDE mechanism isolated into one number — hundreds of distinct
   inactivating substitutions, most unseen in training.
   `tests/test_experimental_models.py` pins both halves.
2. **`deepsets` vs `gatedpool` vs `locusfusion` — was attention, or its
   normalisation, ever the problem?** `gatedpool` is the minimal edit (keep the
   per-drug gate, delete the softmax); `deepsets` deletes the selection machinery
   entirely. If `gatedpool` closes a gap `deepsets` does not, normalisation was
   it; if `deepsets` matches both, that machinery never earned its parameters.
3. **`additive` vs `fm` — is epistasis worth anything?** An FM prices every
   pairwise interaction at `O(T * rank)` via the sum-of-squares identity, so the
   question is cheap to ask without attention's quadratic cost.

## `variant_design_matrix()` — the baseline that is not a network

Exports the sparse isolate x variant matrix with readable column names
(`dna:katG@944=2`), so an L1-logistic or a gradient-boosted tree fits in a few
lines and no GPU.

**This should be run before any of the six.** There is no sparse-linear or tree
baseline anywhere in this project, and for TB AMR from a variant matrix those are
the canonical strong methods — TB-Profiler and Mykrobe are catalogue lookups and
are clinically competitive. `token_signal` already found plain logistic
regression on setfusion's own representations beating the trained model by 0.011.
If it reaches ~0.92 that reframes the project, and it is the cheapest possible
way to find out.

## Restrictions, stated up front

* `catalogue` **cannot** generalise to unseen variants. That is its job — it is
  the control, not a contender.
* `noisyor` is **monotone increasing in evidence** (every factor is in (0,1)), so
  it literally cannot learn a protective variant; and it assumes independent
  causes, so it cannot express compensation or dose-dependence.
* `additive` forbids epistasis by construction. `fm` is the cheapest relaxation.
* All six need `--delta` and per-locus blocks, both implied by `--arch`.

## The knob guard, and the bug it had

`EXPERIMENTAL_DEFAULTS` is the union over the family, so callers pass it straight
through. The rule is **changed-and-foreign**, not merely foreign: passing the
whole defaults dict to `--arch deepsets` is fine, passing `fm_rank=16` to it
raises and names `fm`.

The first version guarded on *foreign* alone, which made every run fail the
moment the engine passed the defaults dict — caught by the synthetic end-to-end
smoke run, not by the unit tests, which had only ever exercised the CLI path.
`_test_factory_rejects_a_changed_foreign_knob` now covers both halves.

## Plumbing

| file | change |
|---|---|
| `models/experimental_models.py` | new — `VariantSet`, `VariantEmbedding`, `_VariantModel` + six subclasses, `make_experimental`, `variant_design_matrix` |
| `models/__init__.py` | re-exports; `ARCHITECTURES` / `PER_LOCUS_ARCHS` / `DELTA_ARCHS` extended from `EXPERIMENTAL_MODELS` rather than by hand |
| `training/{multimodal,multidrug}.py` | one `_build_model` branch each (`arch in EXPERIMENTAL_MODELS`), `experimental=` knob dict, recorded in the result JSON |
| `training/checkpoint.py` | `model_config(experimental=...)` + a rebuild branch; `.get`, so older configs are untouched |
| `scripts/run_{experiment,multidrug}.py` | `--xm-*` argument group, guarded to the six archs |
| `tests/test_experimental_models.py` | new — 25 checks, one per design claim |

Full suite after: `test_baseline_alignment` 8/8, `test_branched` 25/25,
`test_checkpoint` 34/34, `test_cisfusion` 16/16, `test_experimental_models`
25/25, `test_locusfusion` 25/25, `test_transformer_encoder` 13/13.
`test_setfusion` 21/22 — the one failure is pre-existing and unrelated (its
"defaults are the full_run configuration" check has not been updated since
`token_norm` was added).
