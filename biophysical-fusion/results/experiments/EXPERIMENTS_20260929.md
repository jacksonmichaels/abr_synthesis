# Experiments launched 2026-09-29

Four batches, all single-drug, per-drug loci unless stated, 11 drugs. The launch
scripts named below are the source of truth for flags; this file says why each
run exists, what it is compared against, and what to watch out for.

Paths are relative to `biophysical-fusion/`. SLURM logs are in `slurm_logs/`,
and GPU manifests are in `slurm_logs/manifests/submitted_20260929_*.json`.

---

## 0. Read this first: v2 draft-tokenizer contamination

The locusfusion_v2 grid submitted 2026-09-24 14:13 ran while the tokenizer was
still being edited. Some result folders were trained on a **48-channel draft**,
in which an uncalled base gets its own channel. Everything else used the
**final 41-channel** tokenizer.

| trained on the draft (do not use as controls) | trained on the final tokenizer |
|---|---|
| `locusfusion_v2/sd_dna__*` (all 11) | `locusfusion_v2/sd_dna_regulatory__*` |
| `locusfusion_v2/sd_dna_protein__*` (all 11) | `locusfusion_v2/sd_all_modalities__*` |
| `locusfusion_v2/sd_dna_biophysical__*` (4 of 8; 3 died at import) | every `locusfusion_v2_structure/*` arm, `who_t12` |

How to check a run: look at `branch_specs` in the per-drug JSON. The draft has
dna=6 and protein=23; the final has dna=5 and protein=21. The draft logs also
print `unknown_symbols, implied`.

So the comparison "v2 dna+protein 0.9007 vs. all_modalities 0.8924", read as
"extra modalities hurt v2", is **invalid**: the two sides used different
tokenizers. The `sd41_*` reruns in §1 replace these controls. The draft folders
are kept; draft vs. `sd41_` dna is a free ablation of the uncalled-base channel.

---

## 1. locusfusion_v2: modality ladder + structure on a fair base (GPU, 66 jobs)

Launch script: `results/experiments/locusfusion_v2_ladder/submit.sh`, with arms
`controls`, `ladder` and `structure`.

| arm | output | purpose |
|---|---|---|
| controls | `locusfusion_v2/sd41_{dna,dna_protein,dna_biophysical}__locusfusion_v2` | final-tokenizer controls (jobs 65029523–65029607) |
| ladder | `locusfusion_v2/sd_dna_protein_{regulatory,biophysical}__locusfusion_v2` | which added modality, if any, lowers v2 (jobs 65029272–65029337) |
| structure | `locusfusion_v2_structure/sd_dna_protein_structure_L30__locusfusion_v2` | AlphaFold closeness at 30 Å on the dna+protein base (jobs 65029338–65029350) |

**Comparisons**
- Ladder arms vs. `sd41_dna_protein`.
- Structure vs. `sd41_dna_protein`.
- `sd41_*` vs. `sd_dna_regulatory` and `sd_all_modalities`, all final-tokenizer.

Significance testing is the project default: exact Wilcoxon over folds.

**Context: the structure result so far** (`locusfusion_v2_structure/compare.csv`,
all on the all_modalities base)

| decay length | Δ macro CV | Wilcoxon p |
|---|---|---|
| 10 Å | −0.0021 | 0.06 |
| 20 Å | −0.0019 | 0.41 |
| 30 Å | +0.0001 | 0.70 |

No length gives a gain. The suspected reason is coverage: few isolates have two
or more located variants in the same locus, so the pairwise matrix is mostly
empty. The proposed next feature is a per-token distance to the drug or active
site. It is not implemented yet.

**Grid entries** were added to `_build_grid()` in `scripts/sbatch_all_runs.py`
as explicit v2-only entries, not MODALITY_SETS rows:
- `dna_protein_regulatory`
- `dna_protein_biophysical`
- `dna_protein_structure_L30`

**Protocol:** the same as every v2 run (300 epochs / patience 30 / min 50 /
best weights / seed 0 / cv-seed 42). Code changes after the 09-24 controls
trained only add options (`--locus-set`, `--lf-struct-length`), and their
defaults reproduce the old behaviour.

---

### Result, §1 locusfusion_v2 GPU grid (all 66 done, read 2026-10-01)

Macro over 11 drugs. Δ is the exact Wilcoxon over 55 folds.

| arm (final 41-ch tokenizer unless noted) | CV | test |
|---|---|---|
| dna | 0.8822 | 0.8643 |
| dna + protein | 0.8827 | 0.8655 |
| dna + biophysical | 0.8823 | 0.8613 |
| dna + regulatory | 0.8918 | 0.8758 |
| dna + protein + regulatory | 0.8918 | 0.8781 |
| dna + protein + biophysical | 0.8826 | 0.8617 |
| all modalities | 0.8924 | 0.8757 |
| dna + protein + structure 30 Å | 0.8807 | 0.8631 |
| **DRAFT 48-ch** dna | 0.8999 | 0.8863 |
| **DRAFT 48-ch** dna + protein | 0.9007 | 0.8870 |

- **The uncalled-base channel is worth about 0.018.** Final vs. draft:
  dna −0.0177 (3/55 folds better, p = 3e-9); dna + protein −0.0181 (1/55,
  p = 7e-10). This is the largest single effect measured on v2. The 41-channel
  cleanup removed it.
- **Regulatory helps; protein, biophysical and structure do nothing.**
  - dna + regulatory vs. dna: +0.0096 (p = 0.002).
  - all modalities vs. dna + protein: +0.0098 (p = 0.007).
  - dna + protein + structure vs. dna + protein: −0.0020 (p = 0.44).
- **So the "adding modalities hurts v2" read was entirely the tokenizer
  confound.**

---

## 2. Whole-genome LASSO penalty sweep (CPU, 55 jobs)

Launch scripts: `results/experiments/lasso_wholegenome_sweep_20260929/submit.sh`,
which submits `sbatch_sweep.sh DRUG RATIO`. Output goes to `l1r<ratio>/`, with
one `summary.csv` per ratio.

- **Penalty mix:** `l1_ratio` ∈ {1 (L1), 0.75, 0.5, 0.25 (elastic net, saga), 0 (L2, liblinear)}.
- **Strength:** 18 C values from 1e-4 to 30, a superset of the 09-03 grid.
- **Canary:** LEVOFLOXACIN at 0.5 and 0 (jobs 65030114/5) passed. The grid was
  held with `afterok` until then.

| canary | CV | test |
|---|---|---|
| elastic net 0.5 | 0.8910 | 0.9009 |
| ridge (L2) | 0.8498 | 0.8444 |
| L1, 09-03 run | 0.8937 | 0.8974 |

- **Code:** `--l1-ratio` was added to
  `results/experiments/lasso_wholegenome_20260903/lasso_wholegenome.py`. At the
  default of 1 it calls the parent's `ws._fit`, so the 09-03 run is reproduced
  exactly. Each C entry in `c_path` now records `n_iter_per_fold` and `max_iter`.
- **Convergence:** saga hit `max_iter` (2000) at C ≥ 0.7 on LEVOFLOXACIN.
  **Before trusting an elastic-net cell, check `n_iter_per_fold < max_iter`
  near its `best_C`.**
- `lassosweep-AMIKACIN-1` started before the `n_iter` change, so its JSON lacks
  that field. It is L1 with liblinear, so this is harmless.
- **Selection bias:** choosing the best of 18 C values × 5 ratios by CV inflates
  CV AUC. Compare arms on test AUC.
- **Scope:** "degree of the norm" here means only L1, L2 and their mixes;
  sklearn has no general Lp penalty. A true p-norm with 1 < p < 2 would need a
  custom solver.

---

## 3. VCF variants restricted to the curated loci (CPU, 11 jobs)

Launch scripts: `results/experiments/lasso_vcf_loci_20260929/submit.sh`, which
submits `sbatch_vcf_loci.sh DRUG` (both arms per job). Jobs 65030742–65030752.

**Question:** why is the whole-genome lasso (macro CV 0.9174) below the
19-locus one-hot lasso (0.9241)? It is already *above* its matched 19-locus
variant-only arm (0.9152). So the candidate causes are:
- **(a) encoding:** one-hot pools every allele at a site in one penalised
  wild-type column, while variant-only coding splits rare variants into
  separate columns (72% of genome-wide keys are singletons). This fits the loss
  on pncA/tlyA/gid loss-of-function drugs, PZA −0.013 CV.
- **(b) data source:** the snpEff files keep PASS calls only, missing coverage
  reads as reference, and indel alleles are truncated to 20 nt.

**Design:** the same model, protocol and C grid as the 09-03 whole-genome run,
with columns cut to the H37Rv spans of the aligned loci
(`datasets.cds.genome_span`; 19 loci, 38,482 nt). The option is
`--restrict-loci {all19,perdrug}`.

| output | compare against |
|---|---|
| `all19/` | `lasso_wholeseq_20260901/delta_all` (0.9152) |
| `perdrug/` | `lasso_wholeseq_20260901/delta_perdrug` (0.8613) |

**How to read it**
- About equal: the sources agree, so the gap is the encoding. The next step is
  pooled features (per-site "any non-reference", per-gene frameshift/stop)
  added to the genome-wide matrix.
- Lower: the VCF calls lose information. Investigate the PASS filter first.

### Result (2026-09-29, all 11 drugs finished)

| arm | macro CV (VCF) | macro CV (alignment, delta) | Δ | VCF better | macro test (VCF / aln) |
|---|---|---|---|---|---|
| 19 loci | 0.9172 | 0.9152 | +0.0020 | 6/11 | 0.9183 / 0.9083 |
| per-drug loci | 0.8613 | 0.8613 | 0.0000 | 5/11 | 0.8661 / 0.8549 |

- **The data source is ruled out.** Under the same encoding, VCF calls and
  curated alignments give the same AUC, at both locus universes.
- **The rest of the genome adds nothing on the macro.** The whole genome scores
  0.9174 and VCF at 19 loci scores 0.9172. That macro is a cancellation of
  per-drug effects:
  - The genome helps AMIKACIN (0.8926 → 0.9313) and MOXIFLOXACIN.
  - It hurts the small cohorts, LEVOFLOXACIN (0.9371 → 0.8937) and
    ETHIONAMIDE (0.8241 → 0.8084), and also PYRAZINAMIDE (−0.006).
- **So the gap to 19-locus one-hot (0.9241, about −0.007) is the encoding.**
  L1 on one column per variant cannot pool the alleles at a site the way the
  penalised wild-type column does.

Next step: pooled features on the genome-wide matrix, such as per-site "any
non-reference" and per-gene frameshift/stop. The penalty sweep (§2) is
unaffected by this result.

---

## 4. Pooled features for the whole-genome lasso (CPU, 1 + 11 jobs)

Launch script: `results/experiments/lasso_pooled_20260929/submit.sh`.
- Job 65031785 builds `~/abr_workspace/wholegenome_annot.npz`, the snpEff
  `Annotation` / `Annotation_Impact` per cohort key.
- Jobs 65031786–65031796 run one per drug, held on it with `afterok`.

**Why:** §3 showed the gap is the encoding. These arms add pooled indicators on
top of the per-key columns, via `lasso_wholegenome.py --pool`:

| arm | pooled columns added |
|---|---|
| `site/` | `site:<POS>`: any call covering that base (a deletion covers its REF span) |
| `gene/` | per Gene_ID: `lof:<gene>` = any HIGH-impact call, `nonsyn:<gene>` = any HIGH or MODERATE call |
| `both/` | site + gene |
| `all19_both/` | both, restricted to the 19 loci |

Only groups with at least 2 distinct keys are added. The fit is the same
L1 / 8-value C grid / split / folds as every control.

| control | macro CV |
|---|---|
| whole genome, no pooling | 0.9174 |
| 19 loci, VCF, no pooling | 0.9172 |
| 19 loci, one-hot | 0.9241 |

**Code changes:** `--pool`, `--annot-cache` and `--build-annotation-only`. The
default `--pool none` leaves the fit unchanged. JSONs now carry `pool` and
`n_key_features`.

**Caveat:** `who_mass.py` will not recognise pooled feature names
(`site:`, `lof:`, `nonsyn:`) in `top_features`.

### Result (2026-09-29, all 11 drugs finished; annotation job took 5.5 min)

| arm | macro CV | macro test |
|---|---|---|
| whole genome, no pooling | 0.9174 | 0.9170 |
| whole genome + site | 0.9220 | 0.9255 |
| whole genome + gene | 0.9228 | 0.9270 |
| **whole genome + both** | **0.9265** | **0.9286** |
| 19 loci VCF, no pooling | 0.9172 | 0.9183 |
| 19 loci VCF + both | 0.9230 | 0.9244 |
| 19 loci alignment one-hot | 0.9241 | 0.9287 |

Exact Wilcoxon over 55 folds, at each arm's CV-selected best C:

| comparison | Δ | folds better | p |
|---|---|---|---|
| whole genome: both vs. none | +0.0091 | 42/55 | 3.5e-5 |
| whole genome: gene vs. none | +0.0054 | — | 0.009 |
| whole genome: site vs. none | +0.0046 | — | 0.007 |
| 19 loci: both vs. none | +0.0057 | 43/55 | 6.2e-6 |
| 19 loci: both vs. one-hot | −0.0011 | — | 0.97 |
| whole genome both vs. 19-locus one-hot | +0.0024 | — | 0.13 |
| whole genome both vs. 19 loci both | +0.0035 | — | 0.009 |

- **Pooling recovers the one-hot advantage.** At the same 19 loci, pooled VCF
  features match one-hot (p = 0.97), which confirms §3's encoding diagnosis.
- **Once pooled, the whole genome is ahead of the 19 loci under the same
  encoding** (+0.0035, p = 0.009). It is level-to-ahead of one-hot (+0.0024,
  not significant); before pooling it was −0.0067.
- **Gene pooling does most of the work on loss-of-function drugs.**
  - PZA: 0.9325 → 0.9503.
  - ETO: 0.8084 → 0.8397.
  - STR: 0.9257 → 0.9379.
  - KAN: 0.9294 → 0.9394.

  Site pooling is what rescues LEVOFLOXACIN (0.8937 → 0.9267).
- **AMIKACIN is the one drug where pooling slightly hurts** (0.9313 → 0.9285).
- **Best C at the grid edge (0.001 or 3):** LEVO in site and both, ISON in
  all19_both. The penalty sweep (§2) uses unpooled features, so it does not
  cover this.
- **One seed.** The per-drug margins are unresolved except where the fold-level
  test says otherwise.

---

## 5. Pooled lasso, round 2 (2026-10-01, CPU, 1 canary + 21 jobs)

Launch script: `results/experiments/lasso_pooled_v2_20261001/submit.sh`. The
canary (65126970, LEVOFLOXACIN) runs first; jobs 65126971–65126991 wait on it
with `afterok`.

| folder | change vs. round-1 `both` (0.9265) |
|---|---|
| `both` | indel anchor fix + 18-value C grid |
| `both_fmc3` | + per-fold drop of columns with < 3 training carriers |
| `both_ps2`, `both_ps4` | + pooled columns ×2 / ×4, i.e. L1 penalty ÷2 / ÷4 |
| `both_fmc3_ps2` | both changes |
| `both_l1r0.75` | elastic net 0.75 (saga, 3-day limit) |
| `all19_site` | 19 loci, delta + site only, 8-value grid; like-for-like vs. `onehot_all` 0.9241 |

**Code:** new options in `lasso_wholegenome.py`: `--fold-min-count`,
`--pooled-scale`, and the anchor-stripped `_key_span`. Each was unit-tested on
synthetic keys. Defaults are unchanged.

**Caveat:** the site-span rule changed, so round-1 `site:` results
(`lasso_pooled_20260929`) used the old rule, in which an indel also marked its
unchanged anchor base. JSONs from this round record `site_span_rule`.

**Helpers:**
- `lasso_pooled_20260929/show_design.py` prints real design rows per arm.
- `lasso_pooled_20260929/varpos.py` counts variable positions: 732,067
  (16.6% of the genome); 57% have a single carrier; 75,642 have two or more
  alleles.

## 6. locusfusion_v2 tokenizer ablation (2026-10-01, GPU)

**Why:** the 09-24 48-channel draft beat the final 41-channel tokenizer by
about 0.018 (§1 result). Session transcripts show the draft differed in **two**
features. The user removed both on 09-24 with: "remove the channels for errors
as in go back to having 5 channel DNA representation … remove the reference
values and keep the difference".

Both are now opt-in flags (`models.locusfusion_v2.token_layout`); the defaults
are unchanged at C_TOK 41:

| flag | what it adds | width |
|---|---|---|
| `--lf-unknown-symbols` | N channel in dna/regulatory, `-`/`X` residue channels in protein; an uncalled base makes a token. Requires `load_dataset(unknown_symbols=True)`, which the runner passes. | 45 |
| `--lf-ref-bio` | the H37Rv residue's 3 properties | 44 |
| both | — | 48, the draft's width |

**Grid entries:** `dna_unk`, `dna_refbio` and `dna_unk_refbio` (all
`__locusfusion_v2`).

**Launch:** `results/experiments/locusfusion_v2_tokenizer_ablation/submit.sh`.
- Canary 65136458: LEVOFLOXACIN with both flags.
- After that: `ARMS="unk refbio"`, 22 jobs, writing to
  `locusfusion_v2/sd41_dna_{unk,refbio}__locusfusion_v2`.
- The "neither" corner is `sd41_dna`. The "both" corner is the draft's own
  `sd_dna` (draft code), or `ARMS=both` to rerun it on current code.

**Tests:** `tests/test_locusfusion_v2.py` passes 46/46 (7 new);
`test_locusfusion` 25/25; `test_checkpoint` 34/34. Coordinated with the
structure session (a8), whose structure knobs are untouched.

**Open check:** `locusfusion_v2_ladder/n_signal.py` tests whether the count of
uncalled bases alone predicts resistance per drug, i.e. a batch-effect risk.

## Outage: `/project` unmounted at ~18:56 on 2026-09-29

- **Failures:** six GPU jobs died at data load with `FileNotFoundError` on the
  phenotype CSV, all in `dna_protein_regulatory__locusfusion_v2`:
  - AMIKACIN
  - CAPREOMYCIN
  - ETHAMBUTOL
  - KANAMYCIN
  - PYRAZINAMIDE
  - STREPTOMYCIN

  They show **COMPLETED** in sacct, because `run_experiment.py` catches the
  error and exits 0. They need to be resubmitted:

  ```
  python scripts/sbatch_all_runs.py --experiments dna_protein_regulatory__locusfusion_v2 \
      --drugs AMIKACIN CAPREOMYCIN ETHAMBUTOL KANAMYCIN PYRAZINAMIDE STREPTOMYCIN \
      --epochs 300 --patience 30 --min-epochs 50 --save-weights best \
      --run-prefix locusfusion_v2/sd_ --mem 64G --cpus 4 --gpus 1 --time 24:00:00 \
      --constraint vram23
  ```

- **Held:** all 58 still-pending GPU and sweep jobs were put on hold
  (`scontrol hold`). Release them with `scontrol release <ids>` once `/project`
  is back.
- **Unaffected:** sweep jobs that were already running had loaded their data
  before the unmount.
- **Before trusting any result from this day,** check that its drug JSON exists
  and that its log has no `FAILED`.

### Resumed 2026-10-01 13:34 UTC (`/project` back)

**Outage was partial before it was total.** Some nodes lost `/project` from
16:40 on 09-29, so 9 sweep jobs had already died at load with
`FileNotFoundError` by then.

**What was done on resume:**
- Released the 58 held GPU jobs.
- Resubmitted the 6 failed `dna_protein_regulatory` jobs (65119807–65119813).
- Resubmitted the 9 failed sweep cells (65119814–65119822):
  - ETHAMBUTOL, MOXIFLOXACIN, RIFAMPICIN at 0.5
  - ISONIAZID, KANAMYCIN, LEVOFLOXACIN, MOXIFLOXACIN at 0.25
  - AMIKACIN, ETHAMBUTOL at 0
- Resubmitted the 4 sweep cells that hit 24 h, with a 3-day limit
  (65119823–65119826): ISONIAZID at 0.5; ETHAMBUTOL, PYRAZINAMIDE,
  RIFAMPICIN at 0.25. saga at low l1_ratio is slow on the big cohorts:
  1–2 h per fit at C ≥ 1.

**Sweep, partial (macro over the drugs finished, vs. the 09-03 L1 run on the same drugs):**

| l1_ratio | drugs | CV Δ | test Δ |
|---|---|---|---|
| 1 | 11 | +0.0017 | +0.0001 |
| 0.75 | 11 | +0.0011 | +0.0006 |
| 0.5 | 7 | +0.0012 | +0.0019 |
| 0.25 | 4 | +0.0010 | +0.0014 |
| 0 (ridge) | 9 | −0.0140 | −0.0163 |

- **The penalty barely matters, except that pure ridge is clearly worse.**
  Compare pooling (§4), at +0.0091.
- **Convergence:** saga was unconverged at the best C for LEVO and KANA at 0.75.
- **Grid edge:** LEVOFLOXACIN at L1 picked C = 30, the top of the grid.

## Status at time of writing
Everything was submitted and pending, except `lassosweep-AMIKACIN-1`, which was
running. No results from these batches existed yet.

## Environment notes
- `/project/pi_annagreen_umass_edu/` was mounted again on 2026-09-29 (it was
  missing on 09-27).
- The login-session `python3` has no sklearn or Bio. Use `conda activate abr_env`,
  and run anything heavy through sbatch.
