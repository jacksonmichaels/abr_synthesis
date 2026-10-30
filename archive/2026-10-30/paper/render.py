import sys; sys.path.insert(0,'.')
from edits import E
import apply  # re-runs the checks and writes paper_draft_revised.tex
H = r"""# paper_draft.tex: revision plan

Lineage-grouped splits are the setting for the base results. Supporting analyses stay on Big-TB's protocol and are labelled as such. LocusFusion is framed as a variant representation read with two-stage attention. The advisor notes and the LASSO downplay are folded in.

This is the single plan document. Earlier versions are in `/tmp/claude-3044/-scratch3-workspace-jacksonmicha-umass-edu-abr-synthesis/f98060c3-da28-4676-a17f-ede6bc9ea641/scratchpad/`.

## How to use
- **§4 lists the find/replace edits** in draft order.
  - Each **Find** block is copied verbatim from `paper/paper_draft.tex` (2026-10-26) and occurs exactly once; a script checks this.
  - Select multi-line blocks whole.
- **Or diff the applied copy.** `paper/paper_draft_revised.tex` is the draft with every edit applied by `paper/apply.py`, which reads `paper/edits.py`:
  ```
  diff paper/paper_draft.tex paper/paper_draft_revised.tex
  ```
- **Copy the generated files into your LaTeX project:**
  - `results/figures/manuscript/tab_family.tex` and `tab_bigtb.tex` go to `tables/`.
  - The regenerated figure PNGs go to `images/` (names in §2).

## 1. Decisions implemented
1. **Lineage-grouped splits are the setting for the base results.** These are Table 1 and every comparison between models in Results: locus set, modalities, model size, and LocusFusion vs the baselines.
   - **Supporting analyses keep the setting they were run on** (Big-TB's protocol) and say so where they appear. These are the comparison with published models (sec:bigtb, Table 2) and the Shapley attribution appendix.
   - **No new runs are planned.**
2. **Published Big-TB comparisons** are made only on Big-TB's own setting (stratified 80/20 split, stratified 5-fold CV, seed 42; their paper and code use no lineage split). The setting is stated each time: `sec:bigtb` and Table 2.
3. **LocusFusion's contribution** is its **variant representation** (one token per variant, replacing the one-hot alignment) and its **two-stage attention** (within each locus, then across loci).
   - The uncalled-base ablation is no longer cited in the abstract, contributions, sec:lf, Discussion or Conclusion.
   - The sec:lf heading no longer claims where the margin comes from.
4. **The LASSO is downplayed** to a baseline.
5. **Advisor notes:**
   - Terms are defined for a CS audience.
   - The "field lacks a baseline" claim is replaced.
   - "Behind by how much" is answered with numbers.
   - Numbers are added to the abstract and the contributions.
6. **The data-pipeline issue is not mentioned.**

## 2. Generated files (from code)

| File | Draft reference | Notes |
|---|---|---|
| `tab_family.tex` | Table 1, `\input{tables/tab_family.tex}` | lineage, 11 rows; same numbers as `results/analysis/lineage_paper_family/paper_family_lineage.csv` |
| `tab_bigtb.tex` | Table 2, `\input{tables/tab_bigtb.tex}` | Big-TB protocol. It may carry an uncalled-base ablation row: delete that row, or keep it and add one sentence in sec:bigtb |
| `fig2_locus_size.png` | `images/locus_and_params.png` | lineage; Big-TB dashed lines removed |
| `fig3_modalities.png` | `images/modalities.png` | lineage |
| `fig4_locusfusion_per_drug.png` | `images/lf.png` | lineage; LF − LASSO and LF − stacked columns only |
| Shapley appendix figures | `images/shapley_alltest_*`, `images/alltest_own_who_share.png` | **Unchanged.** These are the existing figures from Big-TB's protocol, labelled in the captions (edits to L274 and the first appendix caption). The prepared lineage rerun (`results/analysis/shap_locusfusion_lineage_alltest/submit.sh`) stays unused. |

## 3. What the LocusFusion claims rest on
- **On lineage splits**, LocusFusion is significantly ahead of the stacked CNN (+0.0114, 6/11) and the LASSO (+0.0149, 11/11). LocusFusion uses variant tokens plus two-stage attention; both baselines read the same loci as a one-hot alignment.
- **The text claims the combination, not each part separately.** No run in either setting isolates the representation from the attention. §4 therefore says "a representation and a reader together", and Limitations says "our comparisons do not separate the two". Keep that sentence: it is what lets the claim stand without the ablations.
- **No new runs are planned.** A reviewer may ask about two things:
  - The random-split LASSO favoured one-hot over the delta (variant) encoding (+0.0089, 11/11). That is a linear-model result, and it points the other way.
  - The variant-set aggregator runs exist, but they use the old tokenizer.
  - Neither appears in the paper.

## Still to check by hand
- **Kouchaki et al. 2019:** check whether they included an SVM. If so, add it to the L55 edit.
- **Max parameter count:** the draft says 59.3M; the lineage cis-fusion run logs 59,005,825. Check L184/L213.
- **Streptomycin:** check its first-/second-line label in the definitions paragraph.
- **Bibliography:** add `napier2020barcode` (entry below), or drop it from the L137 edit.

```bibtex
@article{napier2020barcode,
  author  = {Napier, Gary and Campino, Susana and Merid, Yared and Abebe, Markos and Woldeamanuel, Yimtubezinash and Aseffa, Abraham and Hibberd, Martin L. and Phelan, Jody and Clark, Taane G.},
  title   = {Robust barcoding and identification of \textit{Mycobacterium tuberculosis} lineages for epidemiological and clinical studies},
  journal = {Genome Medicine}, volume = {12}, pages = {114}, year = {2020},
  doi     = {10.1186/s13073-020-00817-3}
}
```
I wrote this entry from memory, so check the author list and DOI against the journal page.

## Key numbers
- **Lineage-grouped splits, macro CV AUC:**
  - 19 loci: LocusFusion 0.9134, stacked CNN 0.9017, cis-fusion 0.8978, late-fusion 0.8771, LASSO 0.8985.
  - Per-drug loci: 0.8852 / 0.8891 / 0.8879 / 0.8886 / 0.8316.
- **Big-TB protocol, macro CV AUC:**
  - 19 loci: LocusFusion 0.9323, stacked CNN 0.9246, LASSO 0.9241, published MD-CNN 0.9222.
  - Per-drug loci: LocusFusion 0.9115, stacked CNN 0.9086, LASSO 0.8750, published SD-CNN 0.8636.

## 4. Find/replace edits
"""
L=[H]
for i,x in enumerate(E,1):
    L.append(f"### {i}. {x['where']}\n")
    if x['find']=='@@TABLE1@@':
        L.append("**Find:** the whole table environment, from `\\begin{table}[tbp]` through `\\label{tab:family}` and `\\end{table}` (L216–245).\n")
    else:
        L.append("**Find:**\n```latex\n"+x['find'].rstrip('\n')+"\n```\n")
    if x['repl'].strip():
        L.append("**Replace with:**\n```latex\n"+x['repl'].rstrip('\n')+"\n```\n")
    else:
        L.append("**Replace with:** nothing (delete the line).\n")
    if x['why']: L.append(f"*{x['why']}*\n")
open('/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/paper/revision_plan.md','w').write("\n".join(L))
print('md written')
