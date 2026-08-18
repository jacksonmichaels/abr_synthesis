# ABR workspace — TODO

High-level on purpose: file layout is in flux, so this names *components* and
*decisions*, not paths. Per-project detail stays in each project's own README.

## Where things stand

- **full_run sweep** — 4 architectures × 5 modality sets × {single-drug,
  multi-drug}, all at 150 epochs. 220 single-drug jobs (one per experiment ×
  drug) + 20 joint jobs, landing in `results/experiments/full_run/`. That
  folder's README describes the grid; `full_run_viewer.ipynb` beside it reads
  whatever has finished and compares against both BIG-TB baselines.
- **MD-CNN reproduction** — folds 0–3 done and matching: macro CV **0.9205** vs
  the authors' **0.9212** over the same four folds and our 11 shared drugs,
  **−0.0007**, largest per-drug gap AMIKACIN −0.0069. Job `62593892` was then
  killed by its 16 h wall clock 48 epochs into fold 4 (the ~5 h the authors
  logged does not transfer: ~45 min to load plus 3h31m per fold here), so it
  never wrote the aggregate `auc.csv` and never reached stage 2. Fold 4 + merge +
  eval resubmitted 2026-08-18 as `63188024` (`-t 12:00:00`) via
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

The last three need per-locus blocks and the runners imply that. Verified
identical to the references and not worth re-litigating: LR `exp(-9)`, batch
128, Adam, masked weighted BCE, per-drug inverse-frequency alpha on the train
split, R=0/S=1 encoding, 5-fold shuffled KFold, `256→256→sigmoid` head.

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
