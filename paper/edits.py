# Find/replace edits for paper_draft.tex (lineage split as base setting). Each find must occur exactly once.
E = []
def e(where, find, repl, why=""):
    E.append(dict(where=where, find=find, repl=repl, why=why))

# ---------------- Abstract
e("Abstract, sentence 1",
"motivating predictive ABR models.",
"motivating models that predict resistance directly from a bacterial genome.")
e("Abstract, sentence 2",
"which loci a model sees has a large effect",
"which regions of the genome (loci) a model sees has a large effect")
e("Abstract, sentences 3–5",
r"""we compare models across locus sets, modalities, model sizes and input representations, holding data, splits and evaluation fixed.
A LASSO model on the 19 curated loci is not significantly different from a stacked CNN trained on the same loci, needs no GPU, and is the baseline new architectures should have to clear on the same loci.
Using these insights, we designed LocusFusion, an ABR prediction model that combines a variant tokenization with multi-stage attention. At an eighth of the stacked CNN's size, it outperforms the LASSO on 10 and the stacked CNN on 9 of 11 drugs, and its margin depends on the uncalled-base channel and filtering to variants in its tokenization.""",
r"""we compare models across locus sets, input modalities, model sizes and input representations, holding data, splits and evaluation fixed, on 17,942 patient samples and 11 drugs split so that no lineage appears in both training and evaluation. Giving a model all 19 resistance loci rather than each drug's own raises its mean cross-validated AUC by 0.015 to 0.067, while a 45.9M-parameter model is 0.021 behind a 5.1M-parameter one.
Using these insights, we designed LocusFusion, an ABR prediction model that combines a variant tokenization with multi-stage attention. At an eighth of the size of a stacked CNN, it reaches an AUC of 0.913, against 0.902 for the stacked CNN and 0.899 for a LASSO on the same loci. On Big-TB's own split protocol it reaches 0.932, against 0.922 for the best published model, and there its margin depends on the uncalled-base channel and filtering to variants in its tokenization.""",
"Lineage numbers from FAM/LOG. The LASSO sentence is removed (downplay). The published comparison is stated on its matched setting.")

# ---------------- Introduction
e("Introduction: new definitions paragraph after L48",
"so they can inform the choice of regimen before months of ineffective treatment are given.\n",
r"""so they can inform the choice of regimen before months of ineffective treatment are given.

We briefly define the biological terms used throughout. An \emph{isolate} is a bacterial sample cultured from one patient, and its \emph{phenotype} for a drug is the laboratory result: resistant or susceptible. Each isolate's genome is compared with H37Rv, the standard reference genome of \textit{M.~tuberculosis}; a \emph{variant} is a position at which the two differ. A \emph{locus} (plural \emph{loci}) is a stretch of the genome, here a gene or group of genes together with its \emph{promoter}, the upstream region that controls how strongly the gene is expressed. Within a gene, each \emph{codon} of three DNA bases codes for one amino acid (a \emph{residue}) of the protein; a \emph{synonymous} variant changes the codon without changing the residue, and a stop codon ends the protein. Isolates that share recent ancestry form a \emph{lineage} and share many variants that have nothing to do with resistance. We predict resistance to 11 antibiotics: the first-line drugs used in standard treatment (isoniazid, rifampicin, ethambutol and pyrazinamide) and seven second-line drugs used mainly when those fail (streptomycin, amikacin, kanamycin, capreomycin, ethionamide, levofloxacin and moxifloxacin).
""", "Advisor: define terms for a CS audience. Check the streptomycin grouping.")
e("Introduction L53: missing space",
r"sees all of them.\citet{kulkarni2026mic}",
r"sees all of them. \citet{kulkarni2026mic}")
e("Introduction L55: the 'tall claim'",
"Further, the field lacks an agreed, efficient baseline that new architectures must clear on the same inputs, motivating our L1-regularized logistic regression (LASSO) model, which is not significantly different from our stacked CNN at 19 loci and scores within 0.002 of the published MD-CNN.",
r"Simple models have been competitive in earlier work, including logistic regression in Big-TB and in several earlier studies \citep{kouchaki2019ml,chen2019wdnn,serajian2024denovo}, but they are rarely trained on exactly the same inputs as the deep models they are compared with. We therefore include an L1-regularized logistic regression (LASSO) on the same one-hot alignment the CNNs see.",
"The Background (L86) already cites these baselines. Check whether Kouchaki 2019 included an SVM; if so, name it.")
e("Introduction L58–67: contributions",
r"""\item a LASSO on the raw alignment that is not significantly different from the stacked CNN at 19 loci, falls behind it on every drug on each drug's own loci, and needs no GPU;
\item controlled ablations over locus set, modalities and model size, across
      models spanning two orders of magnitude of parameter count. The locus set
      has a large effect on every model, the modalities matter mainly when the
      locus set is small, and larger models are not more accurate;
\item LocusFusion, a  0.65M-parameter model that outperforms the LASSO on 10 of 11 drugs and
      the stacked CNN on 9 of 11 drugs, and the input-representation change that
      accounts for its margin.""",
r"""\item controlled ablations over locus set, modalities and model size, across
      models spanning two orders of magnitude of parameter count, with a LASSO on
      the same alignment as a baseline, on splits that hold out whole lineages.
      Giving every model all 19 loci instead of each drug's own raises macro CV
      AUC by 0.015 to 0.067; the added modalities gain 0.022 for the established
      models on per-drug loci and nothing at 19 loci; and larger models are not
      more accurate, with a 45.9M-parameter model 0.021 behind a 5.1M-parameter one;
\item LocusFusion, a 0.65M-parameter model with a macro CV AUC of 0.913 at 19
      loci, significantly ahead of the stacked CNN (0.902) and the LASSO (0.899).
      On Big-TB's own split protocol it reaches 0.932 against 0.922 for the
      published MD-CNN, and there its margin depends on the uncalled-base channel
      of its tokenization.""",
"The LASSO bullet is removed (downplay).")

# ---------------- Data
e("Dataset: label for cross-references",
"\\subsection{Dataset}\n",
"\\subsection{Dataset}\n\\label{sec:data}\n")
e("Dataset L126: define operon",
"defined from MycoBrowser operon coordinates for every",
"defined from MycoBrowser operon coordinates (an operon is a group of adjacent genes transcribed together) for every")
e("Dataset L137: the splits",
"To match the conditions of Big-TB evaluation, we use a stratified 80/20 train/test split with an additional five-fold stratified cross-validation on the training set, giving identical per-drug folds for every model.",
r"""We split isolates by lineage, so that no sublineage appears on both sides of any split. We assign each isolate a lineage with the TB-Profiler barcode \citep{phelan2019tbprofiler,napier2020barcode} and group isolates by level-2 sublineage, splitting any group that holds more than 10\% of a drug's cohort one level deeper until none does; mixed and uncalled isolates form one group. Beijing (lineage 2.2.1), which holds 18--47\% of each drug's cohort and 23--41\% of its resistant isolates and cannot be divided at barcode resolution, forms one cross-validation fold on its own. The remaining groups are assigned whole, balanced on isolate and resistant counts, to a test split of about 20\% and to the other four folds. Every model sees the same per-drug splits. For comparison with published models we also evaluate on Big-TB's own protocol, a stratified 80/20 train/test split with stratified five-fold cross-validation on the training set (Section~\ref{sec:bigtb}).""",
"Add napier2020barcode to the .bib (§5).")

# ---------------- Models
e("Models L175: LASSO",
"As an additional baseline we contribute a simple LASSO logistic regression on the raw one-hot alignment, with 228 nonzero coefficients on average out of 22,384 columns at 19 loci.",
"As a linear baseline we include a LASSO logistic regression on the raw one-hot alignment, with 119 nonzero coefficients on average out of 22,384 columns at 19 loci.",
"119 on lineage splits; 228 was the random-split value.")
e("Models L178: implementation check",
"TAs a check on our implementation, trained jointly on all drugs with Big-TB's 19 loci and DNA, our stacked CNN",
"As a check on our implementation, trained jointly on all drugs with Big-TB's 19 loci, DNA input and Big-TB's own stratified splits, our stacked CNN",
"Also fixes the typo 'TAs'.")
e("Models L207: duplicate sentence",
"An additional token is encoded and prepended to act as a summary token. \n",
"")
e("Models L213",
r"that is significantly ahead of both the LASSO and the stacked CNN at 19 loci (Section~\ref{sec:lf})",
r"that is significantly ahead of both the LASSO and the stacked CNN at 19 loci (Section~\ref{sec:lf}).",
"Missing period. Also check 59.3M against 59.0M (LOG).")

# ---------------- Table 1 -> generated
e("Table 1 (L216–245)", "@@TABLE1@@", r"\input{tables/tab_family.tex}",
"Replace the whole table environment, from \\begin{table}[tbp] to \\end{table}. Copy results/figures/manuscript/tab_family.tex to tables/. The generated file contains its own table environment, caption and \\label{tab:family}.")

# ---------------- Evaluation
e("Evaluation L249",
"We  vary four axes (model size, modalities, locus set and input representation), holding data, splits and evaluation fixed. Every model we trained uses the same stratified five-fold CV (seed 42), and we report macro CV AUC: the mean over drugs of each drug's mean fold AUC. Comparisons are paired by fold (11 drugs $\\times$ 5 folds = 55 differences) and tested with a two-sided exact Wilcoxon signed-rank test, Holm-adjusted over 14 comparisons; we call a difference significant at adjusted $p < 0.05$. The published Big-TB models used different splits and are compared unpaired.",
r"""We vary four axes (model size, modalities, locus set and input representation), holding data, splits and evaluation fixed. Every model we trained uses the same lineage-grouped five-fold CV (Section~\ref{sec:data}), and we report macro CV AUC: the mean over drugs of each drug's mean fold AUC. AUC, the area under the receiver operating characteristic curve, is the probability that a model scores a randomly chosen resistant isolate above a randomly chosen susceptible one: 0.5 for random guessing and 1 for a perfect ranking. Comparisons are paired by fold (11 drugs $\times$ 5 folds = 55 differences) and tested with a two-sided exact Wilcoxon signed-rank test, a paired test that makes no assumption about the distribution of the differences, with Holm's correction for the 11 comparisons of Table~\ref{tab:family}; we call a difference significant at adjusted $p < 0.05$. Comparisons with the published Big-TB models use Big-TB's split protocol and are unpaired, because their fold assignments differ from ours (Section~\ref{sec:bigtb}).""")

# ---------------- Results
e("Results L256",
"Table~\\ref{tab:family} lists every paired test;",
"Table~\\ref{tab:family} lists every paired test on lineage-grouped splits;")
e("Results L258: LASSO heading",
"\\subsection{LASSO is a strong baseline at 19 loci}",
"\\subsection{A linear baseline on the same loci}")
e("Results L259–261: LASSO body",
"""At 19 loci it is not significantly different from
 the stacked CNN ($\\Delta=-0.0005$, 5/11 drugs), and it is $+0.0019$ above the
published MD-CNN (6/11). The picture changes on each drug's own loci: there the stacked CNN is ahead on every drug ($\\Delta=-0.0336$, 0/11, significant). Against the published SD-CNN, which also sees per-drug loci, the LASSO is $+0.0114$ (0.8750 vs.\\ 0.8636; 9/11). Any comparison against the LASSO must therefore state the locus set.""",
r"""At 19 loci it is slightly behind the stacked CNN (0.8985 vs.\ 0.9017; $\Delta=-0.0032$, 2/11 drugs), not significant after correction. On each drug's own loci the gap widens: the stacked CNN is ahead on every drug (0.8316 vs.\ 0.8891; $\Delta=-0.0574$, 0/11, significant).""",
"The published-model comparisons move to sec:bigtb.")
e("Results L272: locus set",
"LASSO $+0.0491$, LocusFusion $+0.0274$ (both 11/11) and the established models $+0.0158$ (10/11), all significant. Within the established models the gain is $+0.0217$ for the stacked CNN, $+0.0193$ for cis-fusion and $+0.0063$ for",
"LASSO $+0.0669$, LocusFusion $+0.0349$ (both 10/11) and the established models $+0.0150$ (9/11), all significant. Within the established models the gain is $+0.0216$ for the stacked CNN, $+0.0193$ for cis-fusion and $+0.0042$ for")
e("Results L274: co-resistance",
r"Part of this gain likely reflects co-resistance and lineage structure shared across drugs rather than causal signal at the added loci: at 19 loci, most of the LASSO's largest coefficients fall on other drugs' loci (Figure~\ref{fig:lasso_coef}), and LocusFusion's attribution concentrates on the same genes whatever the drug (Figure~\ref{fig:shap_locus}).",
r"Because no sublineage spans training and evaluation folds, this gain does not come from recognising the sublineages themselves. Part of it likely reflects co-resistance (isolates resistant to one drug are often resistant to others, because resistance accumulates under multi-drug treatment) rather than causal signal at the added loci: at 19 loci, 69\% of the LASSO's 40 largest coefficients per drug fall on other drugs' loci, and LocusFusion's attribution concentrates on the same genes whatever the drug (Figure~\ref{fig:shapley_locus}).",
"69% is computed from the lineage LASSO's top_features. Removes the undefined fig:lasso_coef and fig:shap_locus refs. The attribution clause stands only once the Shapley appendix is rerun on lineage.")
e("fig:locus_size caption",
r"Dashed lines: published Big-TB SD-CNN and MD-CNN, evaluated on different splits. (b) Each network and modality set at 19 loci (n = 20) against its parameter count; dashed line: LASSO. Spearman $\rho = -0.65$ ($p = 0.002$); among the established models alone $\rho = -0.19$ ($p = 0.50$).}",
r"(b) Each network and modality set at 19 loci (n = 20) against its parameter count; dashed line: LASSO. Spearman $\rho = -0.82$ ($p < 0.001$); among the established models alone $\rho = -0.60$ ($p = 0.017$).}",
"The regenerated figure drops the published dashed lines.")
e("Results L292: modalities",
r"adding all modalities to DNA gains $+0.0185$ for the established models (11/11, significant), about the size of the gain from adding loci to the same models ($+0.0158$). The gains follow known mechanisms. Most come from two drugs, ethionamide ($+0.089$) and isoniazid ($+0.031$), which gain from the regulatory input carrying the \textit{fabG1}--\textit{inhA} promoter; pyrazinamide ($+0.020$) gains from the protein input on \textit{pncA}.",
r"adding all modalities to DNA gains $+0.0215$ for the established models (9/11, significant), more than the gain from adding loci to the same models ($+0.0150$). The gains follow known mechanisms. Most come from three drugs: ethionamide ($+0.085$) and isoniazid ($+0.038$), which gain from the regulatory input carrying the \textit{fabG1}--\textit{inhA} promoter, and pyrazinamide ($+0.036$), which gains from the protein input on \textit{pncA}.")
e("Results L293: rifampicin",
"Rifampicin, already near ceiling, does not move.",
"Rifampicin, already near ceiling, barely moves ($+0.003$).")
e("Results L294: modalities at 19 loci",
"At 19 loci the gain falls to $+0.009$ for the established models, most of it on levofloxacin ($+0.051$) and moxifloxacin ($+0.019$), and for LocusFusion it is $-0.0002$. The modality gain on per-drug loci is concentrated in the drugs whose resistance mechanisms the added modalities encode. At 19 loci, where the additional loci appear to supply much of the same information, it is roughly halved for the established models and absent for LocusFusion.",
"At 19 loci the gain disappears: $-0.0021$ for the established models and $+0.0009$ for LocusFusion. The modality gain on per-drug loci is concentrated in the drugs whose resistance mechanisms the added modalities encode, and at 19 loci the additional loci appear to supply the same information.")
e("Results L301–305: model size",
"""nearly the same size, yet cis-fusion is ahead by $0.0123$ (9/11,
significant). The stacked CNN, at 5.1M, is ahead of late-fusion by $0.0165$
(10/11, significant) and not significantly different from cis-fusion
($+0.0042$, 8/11). Across all 20 networks and modality sets, Spearman $\\rho=-0.65$ ($p=0.002$), driven by LocusFusion, the smallest network; among the established models alone, $\\rho=-0.19$ ($p=0.50$).""",
"""nearly the same size, yet cis-fusion is ahead by $0.0164$ (10/11,
significant). The stacked CNN, at 5.1M, is ahead of late-fusion by $0.0212$
(11/11, significant) and not significantly different from cis-fusion
($+0.0048$, 8/11). Across all 20 networks and modality sets, Spearman $\\rho=-0.82$ ($p<0.001$), and among the established models alone $\\rho=-0.60$ ($p=0.017$): within our grid, larger models were less accurate.""")
e("Results L307: sec:lf heading",
"\\subsection{LocusFusion's margin comes from its tokenization}",
"\\subsection{LocusFusion and its tokenization}",
"The ablation was run on Big-TB's protocol only, so the heading no longer asserts the result for the lineage setting.")
e("Results L310–314: sec:lf body",
"""(Figure~\\ref{fig:lf}). It is $+0.0083$ over the LASSO (10/11 drugs) and $+0.0115$ over the stacked CNN (matched over the five modality sets, 9/11), both significant.
The gain over the LASSO is largest on the low-$n$ drugs, moxifloxacin ($+0.023$)
and levofloxacin ($+0.018$), and smallest on rifampicin ($+0.0005$);
the one loss is ethambutol ($-0.0003$). To separate representation from architecture, we train the same model without the uncalled-base channel. The full token is $+0.0122$ over this ablation (11/11, significant), and without the channel LocusFusion falls below the LASSO (0.9212 vs.\\ 0.9241).
\\\\
\\\\
""",
r"""(Figure~\ref{fig:lf}). It is $+0.0149$ over the LASSO (11/11 drugs) and $+0.0115$ over the stacked CNN (matched over the five modality sets, 6/11), both significant. Its margin over the stacked CNN comes mostly from levofloxacin ($+0.094$), the drug with the fewest labelled isolates; over the other ten drugs it averages $+0.0032$.
The gain over the LASSO is largest on the low-$n$ drugs, levofloxacin ($+0.059$)
and moxifloxacin ($+0.032$), and smallest on amikacin ($+0.0009$). To separate representation from architecture, we train the same model without the uncalled-base channel on Big-TB's split protocol (Section~\ref{sec:bigtb}). There the full token is $+0.0122$ over this ablation (11/11, significant; Table~\ref{tab:bigtb}), and without the channel LocusFusion falls below the LASSO (0.9212 vs.\ 0.9241).

\subsection{Comparison with published models}
\label{sec:bigtb}
The published Big-TB scores come from stratified random splits, not lineage-grouped ones, so we compare against them on that setting: Big-TB's protocol of a stratified 80/20 train/test split with stratified five-fold CV (seed 42), on which we also trained LocusFusion, the stacked CNN and the LASSO (Table~\ref{tab:bigtb}). Fold assignments differ from Big-TB's, so these comparisons are unpaired. At 19 loci LocusFusion reaches 0.9323, against 0.9222 for the published MD-CNN; on each drug's own loci it reaches 0.9115, against 0.8636 for the published SD-CNN. Every model scores higher on this setting than on lineage-grouped splits, which withhold whole sublineages.

\input{tables/tab_bigtb.tex}
""",
"This also removes the stray \\\\ lines. Table 2 comes from code (copy results/figures/manuscript/tab_bigtb.tex). The ablation's 'significant' is confirmed by Table 2's Wilcoxon p. The last sentence of the new subsection is optional.")

# ---------------- Discussion
e("Discussion L320: wrapfigure placement",
"\\begin{wrapfigure}{T}{0.5\\linewidth}",
"\\begin{wrapfigure}{r}{0.5\\linewidth}",
"T is not a valid placement.")
e("fig:lf caption",
"between LocusFusion and the LASSO, the stacked CNN, and LocusFusion without the uncalled-base channel. The LASSO column compares LocusFusion with all modalities to the LASSO on DNA; the other two are matched over the five modality sets.",
"between LocusFusion and the LASSO, and between LocusFusion and the stacked CNN, on lineage-grouped splits. The LASSO column compares LocusFusion with all modalities to the LASSO on DNA; the stacked CNN column is matched over the five modality sets.",
"The regenerated figure drops the ablation column, which is now in Table 2.")
e("Discussion L327",
"cis-fusion was ahead of late-fusion by 0.0123,",
"cis-fusion was ahead of late-fusion by 0.0164,")
e("Discussion L329: LASSO paragraph",
"The LASSO makes the same point from the other direction. With 228 nonzer coefficients on average it is not significantly different from the 6.7M parameter stacked CNN at 19 loci, and its coefficients can be read directly as variant effects. Its result depends on the locus set, however: on each drug's own loci it falls behind the CNN on every drug. We therefore see it as the baseline a new model should clear on the same loci, and any comparison against it should state which loci are used.",
r"The LASSO, consistent with earlier reports that regularized linear models are competitive on this task \citep{kouchaki2019ml,chen2019wdnn}, is only slightly behind the 5.1M-parameter stacked CNN at 19 loci (0.8985 vs.\ 0.9017, not significant) with 119 nonzero coefficients on average. On each drug's own loci, however, it is 0.057 behind (0.8316 vs.\ 0.8891) and lower on all 11 drugs, so how a simple model compares with a deep one depends on the loci both are given.",
"Downplay, plus numbers per the advisor. Fixes 'nonzer' and 6.7M.")
e("Discussion L331",
"Its margin over the LASSO and the stacked CNN depends on the uncalled-base channel of its variant tokens, which lets the model distinguish a base that could not be called from one that matches the reference. Removing this channel lowers its macro CV AUC on all 11 drugs and leaves it level with the stacked CNN.",
"On Big-TB's split protocol, its margin over the LASSO and the stacked CNN depends on the uncalled-base channel of its variant tokens, which lets the model distinguish a base that could not be called from one that matches the reference; removing this channel lowers its macro CV AUC on all 11 drugs and leaves it level with the stacked CNN. On lineage-grouped splits its margin over the stacked CNN, though significant, rests largely on levofloxacin.")
e("Discussion L333: limitations",
"Part of the gain from additional loci likely reflects co-resistance and lineage structure shared across drugs, and our splits do not hold out lineages. We use one dataset and one reference genome. Future works should explore these limitations and include additional genetic loci.",
"Part of the gain from additional loci likely reflects co-resistance shared across drugs. Our splits hold out whole sublineages, but the Beijing fold is larger than the others and richer in resistant isolates, and levofloxacin has 9 to 23 resistant isolates per partition, so its fold AUCs are noisy and it carries much of LocusFusion's margin over the stacked CNN. The uncalled-base ablation and the comparison with published models use Big-TB's stratified splits rather than lineage-grouped ones. The Wilcoxon test treats the 55 fold differences as independent, although folds share training data, and the Spearman correlations are over model configurations trained on the same data. We use one dataset and one reference genome. Future work should explore these limitations and include additional genetic loci.")

# ---------------- Conclusion
e("Conclusion L340: modalities",
"the additional modalities on each drug's own loci gained about as much as the additional loci,",
"the additional modalities on each drug's own loci gained more than the additional loci,")
e("Conclusion L340: LocusFusion",
"Our best model, LocusFusion, has 0.65M parameters, and its margin over the baselines depends on its tokenization: removing the uncalled-base channel lowers its macro CV AUC by 0.0122, and lowers it on all 11 drugs, more than its margin over the stacked CNN (0.0115) or the LASSO (0.0083).",
"Our best model, LocusFusion, has 0.65M parameters and is ahead of both the stacked CNN (0.0115) and the LASSO (0.0149) at 19 loci; on Big-TB's split protocol it reaches 0.932 against 0.922 for the published MD-CNN, and removing its uncalled-base channel there lowers its macro CV AUC by 0.0122, on all 11 drugs.")
e("Conclusion L342–343: LASSO",
"""A LASSO on the raw alignment is not significantly different from the stacked CNN at 19 loci and needs no GPU, which makes it a practical baseline for future comparisons. It falls behind the CNN on every drug when
restricted to each drug's own loci, so any comparison against it should be made on the same locus set.""",
"A LASSO on the same alignment is only slightly behind the stacked CNN at 19 loci, but is 0.057 behind it on each drug's own loci, so comparisons between simple and deep models should be made on the same locus set.")
e("Conclusion L345: future work",
"Because part of the gain from additional loci may reflect co-resistance and lineage structure, future work should test locus selection and input representation with lineage controlled, for example on lineage-grouped splits.",
"Because part of the gain from additional loci may reflect co-resistance shared across drugs, future work should separate it from causal signal at the added loci, and test input representations on further cohorts.")

# ---------------- Appendix
e("Appendix: first Shapley caption (define Shapley)",
r"\caption{Shapley attribution share by input modality for LocusFusion at 19 loci,",
r"\caption{Shapley values divide a prediction among input groups by averaging each group's marginal contribution over all orders of adding the groups. Shapley attribution share by input modality for LocusFusion at 19 loci,",
"The isolate counts in this caption (17,433; 54 to 3,517) change once the appendix is rerun on lineage splits.")
e("Appendix: fig:own_who caption, undefined reference",
r"coefficients (Figure~\ref{fig:lasso_coef}). Dotted line:",
"coefficients. Dotted line:")

# ===== 2026-10-26 reframing: LocusFusion = variant representation + two-stage attention.
# The uncalled-base ablation is no longer a headline claim. Overrides by 'where'.
def _set(where, repl, why=None):
    for x in E:
        if x['where'] == where:
            x['repl'] = repl
            if why is not None: x['why'] = why
            return
    raise KeyError(where)

_set("Abstract, sentences 3–5",
r"""we compare models across locus sets, input modalities, model sizes and input representations, holding data, splits and evaluation fixed, on 17,942 patient samples and 11 drugs split so that no sublineage appears in both training and evaluation. Giving a model all 19 resistance loci rather than each drug's own raises mean cross-validated AUC by 0.015 to 0.067 across model families, while a 45.9M-parameter model is 0.021 behind a 5.1M-parameter one.
Using these insights, we designed LocusFusion, which replaces the one-hot alignment with one token per variant and reads the tokens with two-stage attention, first within each locus and then across loci. At an eighth of the size of a stacked CNN, it reaches a cross-validated AUC of 0.913 at 19 loci, the highest of any model we trained, against 0.902 for the stacked CNN and 0.899 for a LASSO on the one-hot alignment of the same loci. On Big-TB's own split protocol it reaches 0.932, against 0.922 for the best published model on this benchmark.""",
"Lineage numbers. 'Sublineage', 'across model families', 'on this benchmark' and 'highest of any model we trained' are the 2026-10-26 abstract fixes; that last phrase is descriptive, not tested. The LASSO sentence is removed (downplay) and the uncalled-base claim is removed (reframing).")

_set("Introduction L58–67: contributions",
r"""\item controlled ablations over locus set, modalities and model size, across
      models spanning two orders of magnitude of parameter count, with a LASSO on
      the same alignment as a baseline, on splits that hold out whole sublineages.
      Giving every model family all 19 loci instead of each drug's own raises macro
      CV AUC by 0.015 to 0.067; the added modalities gain 0.022 for the established
      models on per-drug loci and nothing at 19 loci; and larger models are not
      more accurate, with a 45.9M-parameter model 0.021 behind a 5.1M-parameter one;
\item LocusFusion, a 0.65M-parameter model that represents each isolate by its
      variants rather than a one-hot alignment and reads them with two-stage
      attention, within and then across loci. At 19 loci it reaches a macro CV AUC
      of 0.913, significantly ahead of the stacked CNN (0.902) and the LASSO
      (0.899) trained on the one-hot alignment; on Big-TB's own split protocol it
      reaches 0.932, against 0.922 for the published MD-CNN.""",
"The LASSO bullet is removed (downplay). The LocusFusion bullet is framed as variant representation plus attention. If the §3 ablations are run, append: 'and ablations that separate the representation from the attention'.")

_set("Results L307: sec:lf heading", "\\subsection{LocusFusion at 19 loci}",
"Neutral heading. The section no longer claims what LocusFusion's margin comes from.")

_set("Results L310–314: sec:lf body",
r"""(Figure~\ref{fig:lf}). It is $+0.0149$ over the LASSO (11/11 drugs) and $+0.0115$ over the stacked CNN (matched over the five modality sets, 6/11), both significant. Its margin over the stacked CNN comes mostly from levofloxacin ($+0.094$), the drug with the fewest labelled isolates; over the other ten drugs it averages $+0.0032$.
The gain over the LASSO is largest on the low-$n$ drugs, levofloxacin ($+0.059$)
and moxifloxacin ($+0.032$), and smallest on amikacin ($+0.0009$). Both baselines read the same loci as a one-hot alignment, so the comparison sets a representation of each isolate by its variants, read with two-stage attention, against one in which every position is encoded whether or not it varies.

\subsection{Comparison with published models}
\label{sec:bigtb}
The published Big-TB scores come from stratified random splits, not lineage-grouped ones, so we compare against them on that setting: Big-TB's protocol of a stratified 80/20 train/test split with stratified five-fold CV (seed 42), on which we also trained LocusFusion, the stacked CNN and the LASSO (Table~\ref{tab:bigtb}). Fold assignments differ from Big-TB's, so these comparisons are unpaired. At 19 loci LocusFusion reaches 0.9323, against 0.9222 for the published MD-CNN; on each drug's own loci it reaches 0.9115, against 0.8636 for the published SD-CNN. Every model scores higher on this setting than on lineage-grouped splits, which withhold whole sublineages.

\input{tables/tab_bigtb.tex}
""",
"The uncalled-base ablation sentence is removed. If tab_bigtb.tex keeps the ablation row, either delete that row or add one sentence here pointing to it. The stray \\\\ lines are removed. The last sentence of the new subsection is optional.")

_set("Discussion L331",
"On lineage-grouped splits, LocusFusion's advantage is that of a representation and a reader together: it encodes only the positions where an isolate differs from the reference, each as a token that carries the base, the residue and their properties, and it reads these tokens with attention first within each locus and then across loci, where the one-hot models must learn to ignore the thousands of positions that match the reference. Its margin over the stacked CNN, though significant, rests largely on levofloxacin.",
"The uncalled-base explanation is removed. If the §3 ablations are run, add one sentence apportioning the margin between representation and attention.")

_set("Discussion L333: limitations",
"Part of the gain from additional loci likely reflects co-resistance shared across drugs. Our splits hold out whole sublineages, but the Beijing fold is larger than the others and richer in resistant isolates, and levofloxacin has 9 to 23 resistant isolates per partition, so its fold AUCs are noisy and it carries much of LocusFusion's margin over the stacked CNN. LocusFusion differs from the one-hot models in both its input representation and its attention, and our comparisons do not separate the two. The comparison with published models uses Big-TB's stratified splits rather than lineage-grouped ones. The Wilcoxon test treats the 55 fold differences as independent, although folds share training data, and the Spearman correlations are over model configurations trained on the same data. We use one dataset and one reference genome. Future work should explore these limitations and include additional genetic loci.",
"Drop the 'do not separate the two' sentence if the §3 ablations are run.")

_set("Conclusion L340: LocusFusion",
"Our best model, LocusFusion, has 0.65M parameters, represents each isolate by its variants rather than a one-hot alignment and reads them with two-stage attention; it is ahead of both the stacked CNN (0.0115) and the LASSO (0.0149) at 19 loci, and on Big-TB's split protocol it reaches 0.932, against 0.922 for the published MD-CNN.")

e_ = dict(where="Background L94: LocusFusion vs prior variant-level models",
          find="It also keeps the aligned loci and their fixed-width structure, so it can be compared with the one-hot models on identical inputs, and a single channel of its token can be ablated.",
          repl="It also keeps the aligned loci and their fixed-width structure, so it can be compared with the one-hot models on identical inputs.",
          why="The channel-ablation clause goes with the de-emphasised uncalled-base result.")
E.insert([i for i,x in enumerate(E) if x['where'].startswith('Introduction L58')][0] + 1, e_)

e_ = dict(where="Models L215: the ablation-method paragraph",
          find=r"To test the uncalled-base channel, we ablate it: the ablation keeps the model and the remaining channels and removes only the uncalled-base symbols (\texttt{N} in the DNA and regulatory slots; deleted and unreadable codons in the residue slot), giving a 41-dimensional token in which an uncalled base reads as matching the reference and produces no token. We call these symbols together the uncalled-base channel. Uncalled bases make up 6.8\% of tokens and occur in 21\% of isolates.",
          repl=r"Uncalled bases (\texttt{N}) make up 6.8\% of tokens and occur in 21\% of isolates.",
          why="The ablation is no longer reported. The one descriptive fact about the token is kept.")
E.insert([i for i,x in enumerate(E) if x['where'].startswith('Table 1')][0], e_)

# ===== 2026-10-26: lineage split only for the base results. Supporting analyses
# (Shapley appendix, published comparison) stay on Big-TB's protocol, labelled.
_set("Results L274: co-resistance",
r"Because no sublineage spans training and evaluation folds, this gain does not come from recognising the sublineages themselves. Part of it likely reflects co-resistance (isolates resistant to one drug are often resistant to others, because resistance accumulates under multi-drug treatment) rather than causal signal at the added loci: at 19 loci, 69\% of the LASSO's 40 largest coefficients per drug fall on other drugs' loci, and LocusFusion's attribution, measured on Big-TB's split protocol, concentrates on the same genes whatever the drug (Figure~\ref{fig:shapley_locus}).",
"The 69% comes from the lineage LASSO's top_features. The Shapley figure is the existing one, labelled with its setting. Removes the undefined fig:lasso_coef and fig:shap_locus references.")
_set("Appendix: first Shapley caption (define Shapley)",
r"\caption{Shapley values divide a prediction among input groups by averaging each group's marginal contribution over all orders of adding the groups. Shapley attribution share by input modality for LocusFusion at 19 loci, trained on Big-TB's split protocol (Section~\ref{sec:bigtb}),",
"Defines Shapley and states the setting. The two later appendix figures use the same isolates and checkpoints.")

# 2026-10-26: LF − stacked is 0.011448 (FAM), so it rounds to 0.0114. Fix the replacements only;
# the finds keep the draft's random-split 0.0115.
for _x in E:
    _x['repl'] = _x['repl'].replace('0.0115', '0.0114')
