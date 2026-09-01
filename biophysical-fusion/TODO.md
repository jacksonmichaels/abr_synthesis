# ABR workspace — TODO

High-level on purpose: file layout is in flux, so this names *components* and
*decisions*, not paths. Per-project detail stays in each project's own README.

## Where things stand

- **full_run sweep** — 4 architectures × 5 modality sets × {single-drug,
  multi-drug}, all at 150 epochs. 220 single-drug jobs (one per experiment ×
  drug) + 20 joint jobs, landing in `results/experiments/full_run/`. That
  folder's README describes the grid; `full_run_viewer.ipynb` beside it reads
  whatever has finished and compares against both BIG-TB baselines.
- **MD-CNN reproduction — DONE (2026-08-18).** All 5 folds plus stage 2:
  macro CV **0.9212** vs the authors' **0.9222** over our 11 shared drugs
  (**−0.0010**), test macro **0.8857** vs their **0.8840**. Outputs in
  `mdcnn_eval/training_output/repro_filter12_epoch150/` (`auc.csv`,
  `test_set_auc.csv`). Getting there took two jobs: `62593892` was killed by its
  16 h wall clock 48 epochs into fold 4 (the ~5 h the authors logged does not
  transfer — ~45 min to load plus 3h31m per fold here), so fold 4 + merge + eval
  were resubmitted as `63188024` (`-t 12:00:00`) via
  `mdcnn_eval/scripts/_sbatch_mdcnn_fold4_eval.sh`. CPU-only by design — the
  authors' own published run never got a GPU (cuDNN 9 vs TF 2.14's
  `libcudnn.so.8`), so matching that keeps the numbers comparable.
- **Earlier joint DNA-only run** (60 epochs, per-drug 18-locus union, late
  fusion): macro CV 0.891 vs their 0.922 (−0.031, worse on all 11 drugs); macro
  TEST 0.900 vs their 0.884 (+0.016, better on 9 of 11). Only STREPTOMYCIN and
  MOXIFLOXACIN lose both ways. Its folds peaked *at* the 60-epoch cap in 4/5
  folds, which is why the sweep runs at 150.
- Not a leak story on the multi-drug side: MD-CNN derives train/test with one
  identical call in both crossval and eval. (Unlike SD-CNN — see
  `reference_docs/BASELINE_LEAK_FINDING.md`.)
- **`alllocus_run`** (2026-08-06, 221 jobs) answers open question 4 below for
  `mdcnn` and `cisfusion`: give the SINGLE-drug grid the joint 19-locus input and
  the joint advantage disappears — multi-task sharing is worth ≈+0.002 macro CV
  under `mdcnn`, inside fold SD. It survives only under `late_fusion` (+0.029 to
  +0.031 on the clean cells). Its results folder was lost and has been rebuilt
  from SLURM logs; see `results/experiments/alllocus_run/README.md` for what is
  and is not recoverable.

- **`locusfusion`** — a variant-token, two-stage transformer: one token per
  column that differs from H37Rv rather than one per patch of sequence,
  modalities fused within a locus and then across loci, with a learned `[WT]`
  sentinel per locus so a susceptible isolate is the empty set. It is the design
  that follows from taking findings 3 and 4 in the README seriously at the same
  time — attribution mis-ranks value, and setfusion failed on the
  signal-to-constant ratio rather than on width. Design and the variant census
  in `results/experiments/CODE_CHANGES_20260825.md`.

  - **First run, `newmodels_full` (2026-08-26): macro CV 0.8920** on the
    single-drug per-drug-loci grid, against `mdcnn` 0.9086. It lost.
  - **Its tokenizer was broken, and the break is measured**
    (`results/experiments/CODE_CHANGES_20260901.md`). A nucleotide token landed
    at `column/3` minus a *learned* per-locus scalar, but the aligned FASTAs are
    not bare CDS and the reference row has gaps inside the CDS window, so katG
    S315's DNA token sat 43 codons from its own protein token (rpoB S450: 53;
    pncA S65: 63) — and the scalar read `[-0.0107, +0.0081]` off the trained
    ISONIAZID checkpoint, i.e. it never moved off its zero init. The codon phase
    was wrong for the same reason, and an N call was indistinguishable from a
    match. **So `newmodels_full`'s locusfusion cells measure a mis-registered
    model, not the design.**
  - **`locusfusion_v2` (2026-09-01, 55 jobs) — DONE, and it closes the gap.**
    The rerun with the coordinate computed from the CDS annotation and the H37Rv
    gap pattern, and the token reduced to `alt`/`ref`/`phase`/`coord` over a
    35-symbol vocabulary. Same protocol, same seed, same loci; the only
    difference is the tokenizer, and the parameter count barely moves.
    **Macro over the five modality sets 0.9036 vs the old tokenizer's 0.8863
    (+0.0173) and `mdcnn`'s 0.8995 (+0.0041).** `all_modalities` goes 0.8920 →
    0.9089 against `mdcnn` 0.9086, i.e. from losing by 0.0166 to a tie. **All 55
    cells improved**, by a near-uniform +0.017 — the shape of a fixed input
    representation, not a lucky seed. Read it with
    `results/experiments/locusfusion_v2/README.md`, which also says what the run
    does NOT establish: the 19-locus and joint arms were not rerun and their
    `newmodels_full` numbers still measure the bug, and the coordinate fix and
    the embedding rewrite are not separately attributed.

## Open questions

Why is our joint CV below theirs while our test is above? Ordered by
cost-to-test, not by expected effect:

1. **Non-identical partitions** — both split with seed 42, but over differently
   ordered isolate sets, so the test cohorts genuinely differ (their LFX 20R/36S
   vs our 17R/34S). Align the ordering before trusting any per-drug delta.
2. **Small-n noise** — their CV→test drops −0.038 where ours gains +0.009. On
   ~40-isolate strata, fold-seed variance exceeds the macro gap we're chasing.
   Quantify fold variance before attributing anything to modeling.
3. **Fewer tasks** — we train 11 drugs, they train 13; the two we drop are
   fluoroquinolones sharing gyrA/gyrB with our worst drug (LFX, −0.115).
4. **Locus universe, not multi-task sharing** — single-drug runs see 2–3 loci and
   joint runs see 19, so "joint wins" conflated the two. **Settled for `mdcnn` /
   `cisfusion`**: matched at 19 loci the gap closes to ≈+0.002 / +0.005. Still
   open for `late_fusion`, where +0.03 survives; the untested explanation is
   regularization (36M params on one drug's labels vs 11) rather than transfer.

Epoch budget and locus set are no longer open: the sweep runs 150 epochs, and
the joint runs use every curated locus (below).

## Architectures

`--arch` selects the topology; every result JSON records `arch` and `n_params`.

- **`late_fusion`** (default) — one encoder per feature block, outputs
  concatenated into a shared dense head. `MultiModalNet` (one logit) /
  `MultiDrugNet` (one logit per drug). Loci are concatenated end-to-end, so no
  convolution ever mixes them — only the flatten does, and layer 1 is a 1×1 stem
  (384 params). On the 18-locus joint DNA input that is a 137,952-wide flatten,
  35.4 M parameters, nearly all in one FC layer, over ~11.5 k training isolates.
- **`mdcnn`** — BIG-TB's own SD-CNN / MD-CNN topology (`MDCNNNet`), ported from
  both reference `get_conv_nn`s: every locus is a **channel** on one shared
  zero-padded position axis, `Conv2D(64, (5,12))` across all of them, then
  `Conv1D(64,12) → pool3 → Conv1D(32,3) → Conv1D(32,3) → pool3`, valid padding
  throughout. On their 19-locus 5-channel input it reproduces their shapes
  exactly: 73,024 layer-1 params, 14,336 flatten, 3.87 M total. `n_drugs=1` is
  SD-CNN, `n_drugs=11` is MD-CNN.
- **`setfusion`** — one encoder shared per modality; each block becomes a token
  carrying learned (modality, locus) embeddings, fused by a transformer, read
  out by one attention query per drug. Block count and order stop mattering.
- **`cisfusion`** — promoter ⊕ CDS concatenated per locus into a cis-unit, with
  a segment channel marking which columns are which, then per-branch encoders.
- **`locusfusion`** — one token
  per **variant** rather than per patch: runs on reference-difference input and
  emits a token only where the isolate deviates from H37Rv, so a token varies
  with the genotype by construction. Stage 1 fuses all of a locus's modalities
  into one locus representation, stage 2 fuses the loci; each locus carries a
  learned `[WT]` sentinel, so a susceptible isolate is the empty set. The input
  is O(variants) — median 14 tokens per isolate over 19 loci — rather than
  O(sequence length). Design + the variant census in
  `results/experiments/CODE_CHANGES_20260825.md`; the run it wants is in §7 there.

The last four need per-locus blocks and the runners imply that; `locusfusion`
also implies `--delta`.

Verified identical to the references and not worth re-litigating: LR `exp(-9)`,
batch 128, Adam, masked weighted BCE, per-drug inverse-frequency alpha on the
train split, R=0/S=1 encoding, 5-fold shuffled KFold, `256→256→sigmoid` head.

### The experimental aggregators (`models/experimental_models.py`, 2026-08-25)

Six more `--arch` values — `catalogue`, `additive`, `noisyor`, `gatedpool`,
`deepsets`, `fm` — that share locusfusion's variant tokenizer and differ **only**
in `aggregate()`. Built and tested (25/25), **none trained**.

The premise, and it is a measured one rather than a hunch: once the variants are
tokenized there is no long sequence left to encode, so the remaining question is
how to combine ~14 sparse pieces of evidence — and **softmax normalises, which
makes it a relative selector.** With one informative token among thirteen neutral
ones the weights must sum to 1, so attention has to spend mass on the neutral
tokens. That is the mechanism behind the flat 1/8 attention `token_signal`
measured; it is a property of the operator, not a training failure.

They are a controlled comparison, so they want **one grid, not six runs**. Three
readings are what the grid is for:

1. `catalogue` vs `additive` — what does featurising a variant buy on
   substitutions absent from training? `catalogue`'s zero-initialised
   per-identity table scores an unseen variant at exactly zero; `additive` scores
   it from position + substitution + biophysical change. This is the *pncA* /
   PYRAZINAMIDE mechanism isolated into a single measurement.
2. `deepsets` vs `gatedpool` vs `locusfusion` — was attention, or specifically
   its normalisation, ever the problem? If `gatedpool` closes a gap `deepsets`
   does not, normalisation was it. If `deepsets` matches both, the selection
   machinery was never earning its parameters.
3. `additive` vs `fm` — is epistasis worth anything? An FM prices all pairwise
   interactions at O(T*rank), so this is cheap to ask.

**Run the sparse baseline before any of them.** `variant_design_matrix()` exports
the isolate x variant matrix with readable column names (`dna:katG@944=2`); an
L1-logistic and a LightGBM on it cost an afternoon and no GPU. There is no
sparse-linear or tree baseline anywhere in this project, and for TB AMR from a
variant matrix those are the canonical strong methods — `token_signal` already
found plain logistic regression on setfusion's own representations beating the
trained model by 0.011. If it reaches ~0.92 that reframes the project, and it is
the cheapest possible way to find out.

Known restrictions, each stated in its class docstring: `catalogue` cannot
generalise to unseen variants (that is its job); `noisyor` is monotone in
evidence so it cannot learn a protective variant, and assumes independent causes
so it cannot express compensation; `additive` forbids epistasis by construction.

### locusfusion — next steps

**0. Run it — done twice, and only the second one counts.** `newmodels_full`
(2026-08-26) put it at macro CV 0.8920 against `mdcnn`'s 0.9086, but its
tokenizer mis-registered the nucleotide stream against the protein stream by
43-63 codons, so that number is about the bug, not the design.
`locusfusion_v2` (2026-09-01, 55 jobs) is the rerun:

```bash
bash results/experiments/locusfusion_v2/submit.sh
```

Judge it on **three** things, not one — the claim is mechanistic, so hold it to
the discipline `token_signal` held itself to:

- `cv_auc_mean` against the same cell in `full_run_v2`. The bar that matters is
  `all_modalities__mdcnn` at 0.9086 (per-drug loci) / 0.9246 (19 loci).
- **Read-out attention must stop being uniform.** Flat 1/n_loci means the locus
  summaries are still collinear and the architecture did not do its job — the
  same test that caught setfusion.
- **The attended tokens must be the right ones.** `variant_report()` names the
  locus and alignment column behind every token; for ISONIAZID the top-attended
  token should be katG codon 315 or the fabG1–inhA promoter `c-15t`. An arm that
  raises AUC while attending to neither has improved for some other reason and
  the mechanistic claim is still unsupported.

Then the knobs, in the order the evidence ranks them: `--lf-summary-norm` (the
signal ratio), `--lf-carry-variants 2` (cross-locus epistasis — rpoB+rpoC
compensatory pairs), `--lf-locus-encoder per_locus`, and only then
`--lf-d-model`. Width is last on purpose: `setfusion_scaling` swept four width
axes across 62 arms and closed nothing.

**Refinements, for the second run rather than the first:**

1. **Signed Δproperty for the biophysical modality.** Under `--delta`,
   `datasets/biophysical.py` keeps *the new residue's* z-scored properties at
   changed positions, not `new − reference`. For "does this substitution break
   the protein" — the mechanism behind the *pncA* / PYRAZINAMIDE result, the
   strongest biophysical finding in the project — the signed difference is the
   quantity that generalizes to unseen substitutions, and the current encoding
   makes the model recover it from position. It is a few lines in
   `biophysical.py`, but it changes an existing modality's semantics under
   `--delta` and so needs its own flag and its own matched control (it would
   move `token_signal/a2_delta` too).
2. **The 'N' hole in the uncovered flag.** `one_hot_nt` leaves unknown bases
   all-zero, so an N-filled record is invisible to the occupancy test while a
   gap-filled one is caught. The real records are overwhelmingly `-`, but a
   coverage channel from the loader would close it properly.
3. **Lineage confounding gets more direct, not less.** Neutral lineage markers
   become first-class tokens here. Worth checking whether attention concentrates
   on WHO-catalogue positions or on lineage-defining ones — `variant_report()`
   makes that a one-liner, and it is a sharper version of the standing lineage
   question below.

## Next steps — 2026-08-26

Ordered. Each one is blocked by the one above it, and the top two are cheap.

**1. `noisyor` rerun — IN FLIGHT.** Jobs `63647994`–`63648007`, manifest
`slurm_logs/manifests/submitted_20260826_170414_428834.json`. Its first run
scored macro CV **0.4956** (below chance): the model is monotone in its evidence
and was pointed at P(resistant), while this project encodes **R=0/S=1**, so a
variant pushed every isolate toward the wrong class. It never escaped its init
either (train loss 0.3152 → 0.3023 over 99 epochs, against `additive`'s
0.2246 → 0.0875 on the identical cell). Fixed in
`models/experimental_models.py` — the product is P(susceptible), and the
saturating `-4.0` init is now `-2.0`; the test asserts the DIRECTION now, not
just the monotonicity. Broken results archived with a write-up at
`results/archive/noisyor_polarity_bug_20260825/`.

*When it lands:* it should join the 0.889–0.891 band the other five sit in. Above
that band means the saturating/monotone prior buys something real; still near 0.5
means the polarity was not the whole story and `-2.0` is still too deep for
`lr = exp(-9)`. Then rebuild the overview:

```bash
python scripts/build_overview.py && jupyter nbconvert --to notebook \
    --execute --inplace notebooks/overview.ipynb
```

**2. `variant_aggregators_alllocus` — SUBMITTED 2026-08-26.** 66 GPU jobs
(`63648459`–`63648524`) + 11 CPU (`63648525`–`63648535`); manifest
`slurm_logs/manifests/submitted_20260826_172334_485603.json`. Its README's hold
on step 1 was overridden deliberately — wall-clock, not compute, is the binding
constraint, and both `noisyor` arms read the already-fixed code, so the only risk
taken is that the polarity fix proves insufficient.

This is the run that decides whether the experimental family survives. The six
aggregators have only ever run at **per-drug loci** (2 genes for INH), where the
entire variant-token family loses — and where `locusfusion` and all six are
**indistinguishable** (0.8892–0.8920, spread 0.0028). `locusfusion` gained
**+0.032 from the locus count alone**. So "the aggregators are worse" is
currently a statement about 2 genes, not about the models.

Read two things first, and neither is the macro: **`sparse_baseline` at 19 loci**
(if an L1-logistic with no GPU tracks the networks to ~0.92, the architectures
are the footnote and that is the finding), and **`catalogue` vs `additive`**
(+0.022 at per-drug loci; the vocabulary is ~10x larger at 19 loci so the gap
should widen — if it narrows, the featurisation story is weaker than claimed).

**3. Retire what has answered its question.** Only after 2 is written up.
`deepsets`, `gatedpool` and `fm` exist to establish that the aggregator is not
the lever; five members within 0.0014 already says so. Move them to
`models/legacy.py` (*importable, not in the training path*) rather than deleting
— `alllocus_run` is in the README precisely because results were thrown away
before the finding was recorded. Keep `additive` + `catalogue` (only meaningful
as a pair, and `additive`'s contributions sum to the logit, so it is the one
model in the project whose attribution is exact and free — see finding 3, where
SHAP mis-ranked predictive value and is unconverged below `NSAMPLES` ~128) and
`variant_design_matrix` (deleting the module deletes the sparse baseline).

**4. Still open from before, unchanged in priority by any of the above:**
`build_full_run_viewer.py`'s prose is `full_run`-era and overstates the modality
gains 3–4x; `--token-norm keyed` has only ever run on two drugs in one cell; and
single seed everywhere — multi-seeding the headline cells is a prerequisite for
reporting any of this, including `locusfusion`'s 19-locus parity.

## Loci and regulatory regions

The two reference codebases pick loci differently, and both rules are available:

- **Single-drug** uses SD-CNN's per-drug map (`tb.DRUG_TO_LOCI`), which keeps a
  run locus-matched to the SD-CNN baseline. `datasets.EXTRA_LOCI` (`--extra-loci`,
  off by default) adds the WHO Table-21 tier-1 genes that map omits — fabG1 for
  INH/ETO. Turning it on un-matches the baseline, so it belongs to its own
  experiment.
- **Multi-drug** uses `datasets.loci_on_disk()` — every curated locus FASTA (19,
  including fabG1), which is MD-CNN's own drug-independent rule.
  `--per-drug-loci` selects the per-drug union instead (18).

Regulatory regions come from WHO Table 21 candidate genes; Table 22 supplies a
region's upstream **coordinates**, not its membership. Availability is decided at
load, and any requested region without a FASTA is reported and skipped.

The default region set is **intersected with the loaded loci**, so a run never
carries more promoter windows than coding loci — WHO's list is far longer (INH:
12 regions vs 2 loci) and most of those promoters belong to genes whose CDS is
never loaded. INH gets `inhA` + `katG`; the multi-drug union drops 48 → 16.
`--all-regulatory` restores the full set, and an explicit `--regulatory-loci`
overrides both. Known cost: KANAMYCIN keeps only `rrs` and loses the `eis`
promoter, since `eis` is not one of its coding loci (AMIKACIN and every
multi-drug run keep it).

**The fabG1–inhA operon promoter** is the one carrying the dominant non-*katG*
INH/ETO mechanism (`c-15t`). WHO files it under `inhA` but keys it to
tss=1,673,440, which is fabG1's own CDS start (Rv1483 1673440–1674183; inhA
starts 762 bp later) — so the `inhA` regulatory window **is** that promoter, and
any run with the regulatory modality already sees `c-15t`. `regulatory_msa`
emits the same window under both names (`PROMOTER_ALIASES`) so `fabG1` can be
requested when `inhA` is not in the region set; the two FASTAs are byte-identical
by construction. `datasets.regulatory.REGION_ALIASES` drops the alias from the
per-drug defaults when its primary is present, so no run feeds the model the
same 873 bp twice.

## SetFusionNet: identical logits at init

At initialisation every drug gets the SAME logit. The cause is measured, not
guessed: after fusion the block tokens are near-collinear (pairwise cosine
0.9968–0.9978, spread ‖z_i − z̄‖/‖z‖ ≈ 4%), so every drug query pools essentially
the same vector no matter what its attention weights are, and the per-drug MLP is
shared. Scaling `drug_queries` ×50 does not move it; adding a final LayerNorm to
the fusion encoder does not either (both tested).

Candidate one-line fixes, measured as per-drug logit spread at init on a
12-block / 4-drug setup: as-built `1.5e-04`; `h_in = norm(pooled) + drug_queries`
`3.8e-03`; centering tokens (`z -= z.mean(dim=1, keepdim=True)`) before attention
`1.5e-02`.

Left as-is by decision: it is an init property, not a training failure — each
drug's loss differs, so gradients into the queries differ and they do separate.
Revisit once the sweep shows whether setfusion actually underperforms.

## Reporting caveats

- Their `test_set_auc.csv` **spec/sens columns are garbage** (binarization summed
  across all 13 drugs — their own file shows values >1). Compare AUC / AUC-PR only.
- Our test model is best-of-5 folds by val macro-AUC; theirs is fold 4
  unconditionally. Our test number is favourably selected — say so when quoting it.
- Any macro comparison must be restricted to the 11 shared drugs.
- Single-drug and joint runs use different locus sets by design, so INH/ETO are
  not locus-matched across scopes.

## Constraints

- `pi_annagreen` is **read-only**. Everything we write goes under this workspace.
- Workspace expires **2026-09-03** (extended; `ws_list` is authoritative, 9998
  extensions left). Extend before then with `ws_extend abr_synthesis`.
