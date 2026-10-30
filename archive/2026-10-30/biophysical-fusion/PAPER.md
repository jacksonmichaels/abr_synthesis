# PAPER.md — the paper's story, claims and numbers

**Last revised 2026-09-27.** This is the single planning document for the
write-up. Anything that disagrees with it (older notebooks, `claims.ipynb`,
earlier drafts) is stale.

| file | what it is |
|---|---|
| `../paper/outline.tex` | the current outline/draft. The author edits their own copy; changes proposed to it are marked `\sout{old}\note{new}` |
| `notebooks/results.ipynb` (built by `scripts/build_results.py`) | every number below; §9 is the significance family |
| `SIGNIFICANCE_TESTS.md` | the test, the 14-comparison family, its limits |
| `notebooks/display_figures.ipynb` (built by `scripts/build_display_figures.py`) | the paper figures, saved to `results/figures/paper/` as `fig0`–`fig6` + `figA1` (PNG + SVG) |
| `scripts/locusfusion_schematic.py` | draws `fig0_locusfusion_architecture` (fixed tokenizer; no "wild type" wording — the per-locus token is the "[S] summary token") |
| `results/figures/paper/` | the figures above, plus `sig_paper_family.csv`, `sig_outside_family.csv` |
| `results/figures/display/`, `scripts/locusfusion_diagram.py` | **stale** — 2026-09-01 exports and the old-tokenizer diagram. Do not use for the paper |
| `notebooks/claims.ipynb` | **superseded** — bootstrap intervals, old framing, attribution claims no longer in the paper. Do not quote from it |

---

## 1. The story

Deep models for TB drug-resistance prediction keep getting larger, and CNNs are
reported as best, but published comparisons change inputs and architectures
together, so the gains cannot be attributed. We hold data, splits, training and
evaluation fixed on Big-TB (11 drugs, 17,942 isolates) and vary one axis at a
time. Result: **the input moves accuracy, capacity does not.** A LASSO keeps up
with the CNNs when given the same 19 loci; the locus set is the largest effect;
modalities matter mainly on a small locus set; parameter count is uncorrelated
with accuracy. The one model that beats the LASSO, **LocusFusion** (0.66M, a
tenth of the CNN), does so through its input representation — its tokenizer fix
alone accounts for its margin.

Framing rules (agreed with the author):

- **"Capacity", not "architecture", is what does not matter.** Fusion design
  can lose: `late_fusion` vs `cisfusion` at equal size is −0.012.
- **LocusFusion is an input-representation result carried by a small model**,
  not an architecture win. Say its margin comes from the tokenization. The
  advisor's angle (efficient, biologically informed, DL-capable) fits this.
- **The LASSO "is not significantly different" from the CNN at 19 loci** —
  never "matches"/"equivalent" (no equivalence test is run). On per-drug loci
  it loses; always say both.
- **The locus gain is probably not causal** (co-resistance / lineage). Any
  "headroom is in locus selection" statement carries "with lineage controlled".
- Voice: **"we"** throughout.
- **Do not mention the BIG-TB SD-CNN train/test leak.** It has been fixed and
  is not part of the paper.
- Contributions: (1) the LASSO baseline, (2) controlled ablations over locus
  set, modalities and size across 3.5 orders of magnitude of parameters,
  (3) LocusFusion and the representation change behind it. The onehot−delta
  encoding result is **not** a contribution (appendix/methods note at most).

---

## 2. Claims and numbers

All differences are macro Δ CV AUC (mean over drugs of the per-drug mean fold
Δ). Intervals are Hodges–Lehmann 95% CIs over the 55 fold differences. p is the
exact Wilcoxon over the 55, Holm over the 14-row family unless marked
*unadjusted*. `k/11` = drugs where A > B.

**LASSO vs. CNN** (`lasso_onehot`, 228 nonzero of 22,384 columns)
- 19 loci, − `mdcnn`: −0.0005 [−0.0024, +0.0022], 5/11, Holm p = 0.95 — not significant.
- per-drug loci, − `mdcnn`: −0.0336 [−0.0360, −0.0172], 0/11, p < 0.0001.
- vs published MD-CNN (different splits, not paired): +0.0019, 6/11.
- vs SD-CNN, input-matched at per-drug loci: 0.8750 vs 0.8636 (+0.0114, not paired).
  Never compare the 19-locus LASSO with SD-CNN (input mismatch).

**LocusFusion** (fixed tokenizer = `locusfusion_fixed`)
- − LASSO, 19 loci: +0.0103 [+0.0061, +0.0116], 10/11, p < 0.0001.
- − `mdcnn`, 19 loci, 5 modality sets: +0.0117 [+0.0050, +0.0118], 10/11, p < 0.0001.
  (all_modalities alone +0.0098; DNA-only +0.0142, 9/11 — unadjusted.)
- fixed − old tokenizer, 19 loci: +0.0095 [+0.0044, +0.0109], 11/11, p < 0.0001.
  On the old tokenizer it tied the LASSO (0.9236 vs 0.9241).
- Headline cell: 0.9343 (19 loci, all_modalities), 0.66M params.
- Why the tokenizer mattered: the reference symbol on the token, occupancy read
  as alt ≠ ref, a positional band fitted to the data — **not** coordinate
  placement (the first v2 tokenizer had exact coordinates and tied the old v1;
  `results/archive/locusfusion_v2_pretokenfix_20260918/README.md`). The two
  halves of the fix were never ablated separately.

**Locus set** (per-drug 2–3 loci → all 19)
- LASSO +0.0491 [+0.0342, +0.0545] 11/11; LocusFusion +0.0293 [+0.0209, +0.0352]
  11/11; established (mdcnn, cisfusion, late_fusion) +0.0158 [+0.0132, +0.0256]
  10/11; all p < 0.0001.
- Per architecture, unadjusted: mdcnn +0.0217, cisfusion +0.0193, late_fusion +0.0063 (9/11).
- Whole genome − 19 loci (both `delta`): +0.0022 [−0.0035, +0.0070], 6/11, Holm p = 0.71.
- Not causal, probably: 19-locus LASSO top coefficients include rpoB (ISONIAZID)
  and embB/embA (PYRAZINAMIDE). The labels `rpoB:1418`, `embB:319`, `embA:403`
  are **alignment columns, not codons** — map them to H37Rv codons before
  quoting. Genome-wide: 62.4% of |coefficient| mass outside the WHO catalogue
  (all drugs); ETHIONAMIDE's top-60 is 86% on no-resistance-role sequence,
  mostly PE/PPE/esx. ETHIONAMIDE's single largest coefficient is the
  *fabG1–inhA* promoter `1673425:C>T`.

**Modalities** (established models, per-drug loci)
- all four − DNA: +0.0185 [+0.0054, +0.0173], 11/11, p < 0.0001 — about the
  size of the locus effect for the same models (+0.0158). The HL interval sits
  below the macro because two drugs carry most of it.
- Per drug (sizes only — 5 folds cannot reach p < 0.0625): ETHIONAMIDE +0.089 and
  ISONIAZID +0.031 from regulatory; PYRAZINAMIDE +0.020 from protein;
  rifampicin no headroom; rRNA-target drugs nothing.
- At 19 loci the gain shrinks; LocusFusion all − DNA is +0.003.
- Order of effects is conditional: at 19 loci, loci > tokenizer > modalities;
  at per-drug loci modalities ≈ loci.

**Model size**
- Spearman ρ(params, CV) = +0.02, p = 0.91, 26 complete network cells at 19
  loci, 0.018M–59.3M (includes the variant-token aggregators, excludes v2).
- 5-modset matched: mdcnn (5.1M median) − cisfusion (47.9M) +0.0042, 8/11, Holm
  p = 0.056 (n.s.); mdcnn − late_fusion (45.9M) +0.0165, 10/11; cisfusion −
  late_fusion +0.0123, 9/11 (both p < 0.0001). Parameter counts are medians
  over the 5 modality sets; the all_modalities cells are 6.7M / 60.7M / 57.7M.
- Capacity sweeps agree: `joint_capacity`, `setfusion_scaling` (62 arms).

**Appendix: `locusfusion_v2`** (retokenized, per-drug loci)
- − LocusFusion: dna +0.0005 (7/11), dna_protein +0.0000 (4/11), both n.s.;
  dna_regulatory −0.0174 and all_modalities −0.0165, 0/11 each.
- Regulatory cells did not move from the pre-fix v2 run (0.8918/0.8924 vs
  0.8920/0.8926). Untested candidate cause: promoter columns share the
  16-token-per-locus cap. Grid incomplete (dna_biophysical 8/11; 19 loci dna 5/11).
- Structure channels (`locusfusion_v2_structure`) and WHO tier-1+2 loci
  (`who_t12`, +0.012, 10/11) exist but are **not in the paper** (author's call).

**Protocol facts for the Data section**
- 17,942 isolates, 11 drugs (the set shared with the MD-CNN run log; no
  ciprofloxacin/ofloxacin). Cohorts 269 (levofloxacin) to 17,582 (rifampicin).
- Stratified 80/20 held-out split at seed 42; StratifiedKFold(5, shuffle, 42).
- Deviations from the published runs: stratified CV at seed 42 (baseline KFold
  seed 1); output bias **not** initialised to the log-odds (the baseline does);
  early stopping on validation AUC with patience 30 (baseline validation loss,
  patience 5). The baseline settings collapsed to all-susceptible on the
  imbalanced drugs.
- Seeds: `seeds_19loci` — SD across seeds 0.0006 vs across folds 0.004–0.008
  (macro), but its locusfusion arm is the **pre-fix** tokenizer.
- Mean per-drug fold SD 0.0164. Spread across modality sets: cisfusion 0.0129,
  late_fusion 0.0076, mdcnn 0.0075, locusfusion 0.0031.

---

## 3. Significance convention

Exact two-sided Wilcoxon signed-rank over the 55 fold-paired differences
(BIG-TB's method; advisor's decision 2026-09-27 — **do not reintroduce
Nadeau–Bengio or TOST**). Hodges–Lehmann shift with its exact CI. Holm over the
14 comparisons listed in `SIGNIFICANCE_TESTS.md`. Verdicts: "significant" /
"not significant" only. The paper figure (`fig6_significance_by_claim`) draws 13 of the
14, grouped by claim (LASSO vs. CNN, LocusFusion, Locus set, Modalities, Model
size); only the onehot−delta row is tested and tabled but not drawn. The
variant-token aggregators are in no paper figure (appendix material if needed). Figure rules from `~/.claude/CLAUDE.md` apply (neutral titles, no
asides inside plots).

---

## 4. Figures

| file | shows | claim |
|---|---|---|
| fig0 | LocusFusion schematic | model |
| fig1 | macro CV AUC per model, per-drug loci vs 19 loci, published baselines as lines | LASSO tie at 19 / loss at per-drug; locus effect |
| fig2 | heatmap: LocusFusion − LASSO, − MD-CNN, fixed − old tokenizer, per drug by cohort size | LocusFusion separates (10/11); margin from the tokenizer (11/11) |
| fig3 | grouped bars: + protein / + biophysical / + regulatory over DNA per drug, CNN mean, both locus sets | modalities matter on small locus sets, where the mechanism predicts |
| fig4 | LASSO coefficient mass: own vs other drugs' loci (19-locus top-40); WHO membership (genome-wide) | locus gain probably not causal |
| fig5 | parameters vs macro CV AUC, 20 network cells at 19 loci (no aggregators) | size does not buy accuracy (ρ −0.63 over 20 cells; −0.19, p 0.50 CNNs only) |
| fig6 | paired Δ by claim, Wilcoxon + Holm | significance |
| figA1 | v2 vs v1 per modality set | appendix |

The notebook needs `bigtb_baselines` (the `/project` volume) to execute; the PNGs
on disk were rendered 2026-09-27 with the SD-CNN values from that module's
hard-coded table and MD-CNN's documented macro 0.9222 standing in for the log.

## 5. Open items and reviewer risks

Ranked by value to the paper:

1. **Lineage-held-out evaluation** — the biggest effect (locus set) is likely
   confounded; a TB reviewer will ask first. Needs lineage calls (TB-Profiler or
   Big-TB metadata), then re-scoring on lineage-grouped splits.
2. **WHO mutation-catalogue baseline** (rule-based) — the standard comparison in
   this field; missing. The `catalogue` model is a *learned* catalogue, not this.
3. **Repartition LocusFusion (fixed tokenizer)** on cv-seeds 101–104, which
   `repartition_19loci` already has for mdcnn. Blocked: the fixed tokenizer lives
   only on branch `worktree-lf-token-offset-fix`, which lacks `--cv-seed`.
4. Effects are small (~+0.01) and the pooled Wilcoxon treats overlapping folds
   as independent; the ties are robust, the wins less so. State it in Limitations.
5. Tokenizer and two-stage fusion are not separated (fixed tokenizer only ever
   ran inside LocusFusion) — in Limitations.
6. Map LASSO alignment-column coefficients to H37Rv codons; add a drug count to
   the SD-CNN comparison; verify the Big-TB citation; drop or verify the
   "0.0032M–63.8M across every cell" parameter range.

**Data access (2026-09-27):** `/project/pi_annagreen_umass_edu/` is not mounted
on the login node or compute nodes, so `results.ipynb` cannot be executed and no
training can run until it returns. `/work/pi_annagreen_umass_edu/` exists but
does not hold `project_data_curation/`.
