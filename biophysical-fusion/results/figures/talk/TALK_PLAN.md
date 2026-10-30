# Lab talk plan (~45 min, ~28 slides)

Audience: a computational biology lab that has seen the early work (before LocusFusion). Define terms briefly. Recap
the setup quickly and spend the time on LocusFusion and what came after.

**Arc:** what actually makes an ABR model better? → we held everything fixed and varied one thing at a time → the
inputs matter, model size doesn't → LocusFusion, a small model built on that idea → a quick lineage check → what we
have learned since the paper.

**Out of scope:** the whole-genome LASSO, the variant-token aggregators and the data-pipeline fixes.

Figure paths are relative to `results/figures/`. `SLIDE_MAP.md` in this folder lists the same files by slide number.

| Agenda | Time | Slides |
|---|---|---|
| 1. Background and setup | 8 min | 1–6 |
| 2. LocusFusion | 7 min | 7–10 |
| 3. Results | 18 min | 11–19 |
| 4. Beyond the paper | 12 min | 20–28 |

If time runs short, cut slides 6, 13 and 22 first.

---

## 1. Background and setup (8 min)

### Motivation

**1. TB resistance**
- Culture-based testing takes weeks for *M. tuberculosis*.
- Resistance comes almost entirely from chromosomal mutations, so in principle it can be read from the genome.
- Figure: `talk/mockups/s01_resistance_cartoon.png` (rpoB S450L: TCG → TTG, Ser → Leu, rifampicin no longer seats).

**2. Current tools**
- Catalogue tools (WHO catalogue, TB-Profiler, Mykrobe) can't call a variant the catalogue hasn't graded.
- Models learned from genotype–phenotype data fill that gap, and the field is moving toward bigger models.

**3. The problem**
- Each paper changes the loci, the representation and the model at the same time.
- Figure: `talk/mockups/s03_literature_table.png`, one row per paper:
  - Green et al.: 18 loci
  - Big-TB: SD-CNN on each drug's own loci vs MD-CNN on all loci
  - Kulkarni et al.: adds protein features and lineage
  - Serajian et al.: genome-wide k-mers
- Question: if we hold everything else fixed, which change actually moves accuracy?

### Setup

**4. Data pipeline**
- Figure: `manuscript/fig0_pipeline.png`
- 17,942 isolates, 11 drugs, per-drug cohorts of 269 (LFX) to 17,582 (RIF).
- Define CV AUC and macro CV AUC (mean over drugs) in one line.
- Comparisons are paired by fold: Wilcoxon over 55 fold differences, Holm-adjusted.

**5. What we vary**
- Locus set: each drug's own loci vs all 19 loci.
- Modalities: DNA, + protein, + biophysical, + regulatory, all.
- Model size, and input representation.
- Figure: `talk/mockups/s05_axes.png`

**6. The models we compare** *(cut first if short)*
- Stacked CNN (5.1M), cis-fusion (47.9M), late-fusion (45.9M), and the LASSO (228 nonzero weights out of 22,384).
- Figure: `talk/mockups/s06_models.png` (block diagrams; the July diagrams in `diagrams/` are out of date).
- Sanity check: our stacked CNN trained on all drugs reaches 0.9179, our rerun of their code 0.9212, and the published MD-CNN 0.9222.

**6b. What the LASSO is** (1 min)
- Figure: `talk/new/lasso_explainer.png`
- (a) Logistic regression with one weight per one-hot column (position × base). The L1 penalty λΣ|w| drives most weights to exactly zero.
- (b) Rifampicin, 19 loci, real penalty path:
  - 4 weights already give CV AUC 0.922;
  - CV selects 355 weights (AUC 0.981);
  - across the 11 drugs it keeps about 228 of about 22,384 columns on average.
- (c) Reading the weights:
  - the H37Rv base at rpoB codons 432–452 (the rifampicin resistance-determining region) pushes toward susceptible, which is the flip side of "a mutation there means resistant";
  - rpoB 170 T is V170F, a known rifampicin mutation, and pushes toward resistant;
  - gid 96 is outside rifampicin's genes, the kind of co-resistance signal slide 13 comes back to.
- Point to make: the model is cheap, needs no GPU, and its weights can be read directly.

## 2. LocusFusion (7 min)

**7. The idea**
- Turn only the positions that differ from the reference into tokens.
- Figure: `talk/mockups/s07_onehot_vs_tokens.png`

**8–9. Tokenization worked example**
- Figure: `manuscript/fig_tokenization.png`. Build it up in three steps: isolate, then tokens, then model.
- Isoniazid-resistant isolate: fabG1 c.−15C>T (promoter), katG R119R (synonymous), katG S315T, and an uncalled base in the katG 3′ flank.
- Each token is 45-dimensional: base, residue, biophysical properties, promoter base, change in properties from the reference, a summary flag and codon phase.

**10. Architecture**
- Figure: `manuscript/fig1_architecture.png`
- Attention first within each locus (summary token plus variant tokens), then across the loci's summary tokens, then a drug query and an MLP.
- 0.65M parameters.

## 3. Results (18 min)

### Controlled comparisons

**11. The LASSO is a strong baseline at 19 loci**
- At 19 loci it is level with the stacked CNN: 0.924 vs 0.925, p = 0.95.
- On each drug's own loci it loses on every drug (−0.034, 0/11).
- Takeaway: any comparison against the LASSO has to state the locus set.

**12. Locus set**
- Figure: `manuscript/fig2_locus_size.png`, panel a.
- Every model gains from all 19 loci, and the smaller the model, the bigger the gain:
  - LASSO +0.049
  - LocusFusion +0.027
  - established models +0.016

**13. Why more loci help** *(cut if short)*
- Figure: `shap_v3_alltest/shapley_alltest_locus_heatmap.png` or `manuscript/figA1_lasso_coefficients.png`.
- Weight lands on other drugs' genes, which points to co-resistance and lineage rather than causal signal.

**14. Modalities**
- Figure: `manuscript/fig3_modalities.png`
- On each drug's own loci, adding all modalities gives +0.019 for the established models.
- The gains follow known mechanisms:
  - ETO +0.089 and INH +0.031, through the inhA promoter in the regulatory input.
  - PZA +0.020, through protein on pncA.
- At 19 loci the gain roughly halves for the established models and disappears for LocusFusion.

**15. Model size**
- Figure: `manuscript/fig2_locus_size.png`, panel b.
- Bigger is not better: the 5.1M stacked CNN beats the 45.9M late-fusion by 0.017.
- Cis-fusion and late-fusion are almost the same size but differ by 0.012, so how a model fuses its inputs matters where size doesn't.

**16. LocusFusion per drug**
- Figure: `manuscript/fig4_locusfusion_per_drug.png`
- +0.008 over the LASSO (10/11 drugs) and +0.012 over the stacked CNN (9/11), both significant.
- Largest gains on the drugs with few isolates (MXF, LFX).
- Removing the uncalled-base channel costs +0.012 on all 11 drugs, and the model then falls below the LASSO.
- Takeaway: this is a result about the input representation in a small model, not an architecture win.

### Where LocusFusion's margin comes from: the uncalled-base channel

**17. The accident**
- Figure: `talk/new/uncalled_channel_per_drug.png`
- An early draft of the v2 tokenizer gave uncalled bases their own channel, and a cleanup removed it.
- (a) Without it: −0.018 on 11/11 drugs (v2, each drug's own loci, DNA).
- (b) v3 put it back: +0.012 on 11/11 drugs (19 loci, mean of the five modality sets).

**18. Why an uncalled base carries signal**
- Figure: `talk/new/uncalled_hotspots.png`
- Isolates with an N at canonical resistance sites are 57–95% resistant, against 13–33% for isolates with a called base there. The sites are rrs 1401, rpoB 450/435, rpoC 483, katG 315, gyrA 94, embB 306 and pncA 65.
- Not a batch effect: within a source collection, the N count alone predicts resistance at AUC ≈ 0.5.
- A plausible reading is mixed populations (heteroresistance) at resistance sites. Present it as a hypothesis; allele frequencies weren't tested.
- Source: `results/analysis/missingness_v3/`.

### Robustness: lineage split

**19. Holding out lineages**
- Figure: `talk/new/lineage_vs_random.png`
- Design in one sentence: sublineages are kept together, and the Beijing lineage is its own CV fold.
- Say aloud:
  - every model drops about 0.02–0.03 AUC;
  - the order doesn't change, and LocusFusion is still on top (0.932 → 0.913);
  - the gain from using all 19 loci survives.
- If asked: LocusFusion's lead over the stacked CNN holds (0.913 vs 0.902) but is no longer significant (Holm p = 0.087). It still beats the LASSO significantly.
- These will be in the final paper; swap in its figures once they exist.
- Backup: `talk/lineage_split/c4_0.png` (split design) and `talk/lineage_split/c21_0.png` (19-locus gain, per drug).

## 4. Beyond the paper (12 min)

**20. Does the residue token change calls? (PZA, pncA only)**
- Figure: `talk/modality_flips/c14_0.png`
- S66* (a stop codon, never seen in training, resistant isolate):
  - DNA-only model: 0.09, called susceptible.
  - + protein: 0.78; all modalities: 0.82, called resistant.
  - In training, a premature stop in pncA is 91% resistant.
- A25A (synonymous, susceptible isolate):
  - DNA-only model: 0.97. That call traces back to one divergent resistant training isolate carrying the same base.
  - + protein: 0.17, called susceptible.
- This is an illustration, not a test (small n).

**21. Mutating every base of pncA in silico**
- Figure: `talk/modality_flips/c19_0.png` (`c21_0.png` shows synonymous changes along the gene).
- All 1,674 single-base substitutions on an H37Rv background.
- Unseen stop codons: + protein gives 0.96 and calls every one resistant.
- No model learned that synonymous changes are harmless: they average P(resistant) 0.52–0.61.

**22. AlphaFold structure: no gain** *(cut if short)*
- Figure: `talk/new/structure_tokens.png`, on the same isolate as slides 8–9 (katG R119R, S315T and an uncalled base in the 3′ flank).
  - (a) R119 and S315 are 24.9 Å apart on the AlphaFold model.
  - (b) For each locus, a 16 × 16 closeness matrix over the token slots, exp(−d / 10 Å): 1 for the same residue, 0.08 for this pair, 0 for tokens with no residue (flank, promoter, rRNA) and for empty slots.
  - (c) Row i of the matrix is appended to token i (41 + 16 = 57 channels). This ran on LocusFusion v2, before the uncalled-base channel.
- Result: −0.002 macro CV AUC, p = 0.06. Decay lengths of 20 Å and 30 Å, heavy-atom distances and an AlphaFold confidence (PAE) gate didn't help either.
- Why it carried little: most isolates have at most one variant with a residue per gene, so the matrix is mostly the diagonal plus zeros. Panel (b) shows this.
- Few isolates have two or more located variants in the same gene, which motivates distance to the drug site instead.

**22b. How the distances were computed**
- Figure: `talk/mockups/s23_site_distance_method.png`
- Ribosome loci (rrs, rrl, rpsL; streptomycin, kanamycin, amikacin, capreomycin):
  - drug-bound structures from the PDB: 4DR3 (STR, *T. thermophilus* 30S), 8EV7 (KAN) and 8EV6 (AMK) (*T. thermophilus* 70S), 5V93 (CAP, *M. tuberculosis* 70S cryo-EM);
  - keep the 16S, 23S and S12 chains of one ribosome, and the drug copy bound to it (for KAN and AMK, the copy at rrs 1401);
  - align each chain globally to H37Rv rrs, rrl and rpsL;
  - each H37Rv position gets the closest heavy-atom distance from its residue to any drug atom.
- Protein loci (rpoB, katG, inhA, ethR, embB, pncA, gyrA, gyrB, gid, eis):
  - binding-site residues come from the literature review, checked against H37Rv codon numbers;
  - distances are measured on the AlphaFold DB model of each H37Rv protein, which has no drug in it;
  - each codon gets the closest heavy-atom distance to any atom of a site residue (0 Å for a site residue).
- Each observed variant then reads the distance at its position, and variants are binned into ≤4, 4–8, 8–12, 12–20 and >20 Å.
- Caveats to say if asked:
  - the review looks partly machine-written: its gyrB residues were in the wrong numbering (shifted −39) and two eis residues were wrong, all fixed or dropped;
  - three of the four ribosome structures are *T. thermophilus* (about 79% rRNA identity after alignment);
  - AlphaFold monomers have no ligand and only one chain, so the gyrA/gyrB and EmbA/EmbB interfaces are seen from one side.

**23. Binding-site distance: the 16S rRNA**
- Figure: `talk/binding_site_visuals/c4_0.png` (+ `c6_0.png`)
- Streptomycin sits in a different pocket from kanamycin, amikacin and capreomycin, which overlap at the h44 decoding site.

**24. The same variant sits at a different distance from each drug**
- Figure: `talk/new/site_distance_by_drug.png`: one panel per variant, distance to each drug on x, % resistant carriers for that drug on y.
- rrs 517 is 3 Å from streptomycin and about 19 Å from the others; rrs 1401 is the reverse.
- RpsL K43R is 2.9 Å from streptomycin and 97% resistant; at 7.5–8.6 Å from the injectables it is 23–37% resistant.
- rrs 1401G is 3.3 Å from the injectables (81–94% resistant) and 13 Å from streptomycin (69%).
- rrs 492T is far from all four drugs (29–39 Å) and 2–5% resistant.
- The exception is rrs 514C: 16–17 Å from the injectables but still 56–75% resistant, through co-resistance in MDR isolates.
- The case for a distance channel is a multi-drug model.

**25. % resistant by distance, all 11 drugs**
- Figure: `talk/binding_site_visuals/c12_0.png`
- Ribosome drugs: 81–94% resistant within 4 Å, 2–10% beyond 20 Å.
- Ethambutol: a clean gradient, 78% → 11%.
- Pyrazinamide and isoniazid: flat, because resistance is loss of function anywhere in the protein.
- Fluoroquinolones: diluted by gyrA S95T, a lineage marker that sits 1.3 Å from the site.

**26. EmbB in 3D**
- Figure: `presentation/embB_ethambutol_rotating.gif` (static version: `talk/binding_site_visuals/c14_0.png`).
- Resistant variants cluster on the ethambutol pocket; the large susceptible spheres are on the far side.

**27. Distance alone vs LocusFusion**
- Figure: `talk/binding_site_distance/c44_0.png`
- Distance beats "any variant at the site" for every drug (e.g. INH 0.73 → 0.92, LFX 0.50 → 0.85).
- It stays 0.03–0.22 below LocusFusion.

### Wrap-up

**28. Summary**
- Figure: `talk/new/recap_effects.png`: every headline paired comparison as one bar of macro CV AUC difference, coloured by model family (teal LASSO, copper LocusFusion, blue CNNs); hatched = not significant.
- Read it top to bottom:
  - inputs move accuracy: more loci help every model, and modalities help on a drug's own loci;
  - size doesn't: the 5.1M stacked CNN is ahead of the 45.9M late-fusion;
  - LocusFusion's edge is about the size of its uncalled-base channel;
  - every model drops 0.02–0.03 under the lineage split.
- The LocusFusion modality row (†) uses an unadjusted p from the v3 sweep; it wasn't in the Holm family.

**28b. Next steps**
- Figure: `talk/mockups/s28_next_distance_channel.png`: in a multi-drug model, the same token (rrs 1401G) gets a different distance depending on which drug is queried: 13.3 Å for streptomycin, 3.3–3.4 Å for the injectables. The % resistant carriers follow the distance.
- Also on the list:
  - curated binding sites from PDB ligand contacts, replacing the literature review;
  - a WHO-catalogue rule baseline.

**29. Questions**
- Figure: reuse `talk/binding_site_visuals/c4_0.png` (the four ribosome drugs on the *M. tuberculosis* 16S) or the EmbB GIF as a full-bleed background.

---

## Before presenting

- **Don't use binding-site §10** (`talk/binding_site_distance/c48_0.png`). It was drawn when the sd19 prediction sweep had written only 6 of 11 drugs. To use it, check the logs for job array 65965474, resubmit with `--skip-existing`, then rebuild.
- **The paper draft gives the stacked CNN's size two ways:** 5.1M (line 238) and 6.7M (line 329). Use one value on the slides.
- **To rebuild the figures** (`scripts/build_talk_mockups.py` redraws the schematic mock-ups in `talk/mockups/`):
  - `scripts/render_talk_figures.py <notebook>` re-renders notebook figures at slide resolution.
  - `scripts/build_talk_figures.py` rebuilds `talk/new/` (run it with the `ABR_GENOTYPE_DIR` and `ABR_GENBANK` environment variables pointing at `data_snapshot/`).
