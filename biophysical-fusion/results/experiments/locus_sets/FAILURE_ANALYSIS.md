# Locus sets failure analysis (2026-11-05)

**The performance drop with more loci is in the stacked CNN only, and the jobs weren't cut short.** LocusFusion and LASSO stay flat from 19 to 64 loci. The best-supported explanation is that the CNN architecture scales badly with locus count, made worse by too little training on drugs with few resistant isolates. I inferred that from the existing results and haven't tested it with a rerun. No code changed, and no jobs were submitted.

Macro CV / test AUC, averaged over modality sets (LASSO is DNA only):

| | Per-drug (1–3 loci) | 19 loci | WHO (6–14; 54 in multi-drug) | 64 loci |
|---|---|---|---|---|
| **Single-drug LocusFusion** | .878 / .877 | .913 / .883 | .903 / .886 | **.914 / .903** |
| **Single-drug CNN** | .880 / .869 | .901 / .894 | .884 / .875 | **.867 / .859** |
| **Single-drug LASSO** | .832 / .849 | .899 / .904 | .863 / .868 | **.901 / .905** |
| **Multi-drug LocusFusion** | – | .895 / .877 | .897 / .880 | **.898 / .880** |
| **Multi-drug CNN** | – | .896 / .870 | .877 / .860 | **.871 / .862** |

- **This doesn't contradict earlier results:** those tested about 2 → 19 loci (`dna__mdcnn` .892 → .917), and that gain still holds. 19 → 64 is a range we hadn't tested.
- **WHO's dip isn't a "more loci" effect:** in single-drug it uses fewer loci than the 19-locus set.
- **The extra loci add little:** LASSO gains only +.003 in CV, so a flat result was the realistic best case. The 14 loci unique to the 64-locus set target drugs we don't model.

## Were models cut short?

**Not by the scheduler or the epoch cap, but some CNN folds barely trained.**

| Possible cause | Result |
|---|---|
| SLURM kill, OOM or timeout | Ruled out: all 276 cells finished on the first try, each with 5 folds |
| 300-epoch cap | Ruled out: only 2 folds reached it, both in the 19-locus CNN control |
| Early stopping treating 64 loci unfairly | Unlikely: 25% of CNN folds were still at their peak when stopped, vs 31% at 19 loci, so more patience probably won't close the gap |
| Config or fold differences | Ruled out: hyperparameters and validation fold IDs match across arms |
| Repaired FASTAs vs older controls | Ruled out as the main cause: the multi-drug CNN drops too, and its control uses the same FASTAs |

**LocusFusion trained fully and then overfit.** At 64 loci, validation AUC peaks around epoch 20 (.882) and falls to .866 by epoch 79, while train loss keeps dropping to .049. Most folds (85%) stop at the epoch-79 floor set by `min_epochs` + patience.

**The CNN is underfitting.** Its 64 loci include all 19 control loci, yet it trains worse:

- **Train loss:** the stop/epoch-1 loss ratio is worse on 11/11 drugs.
- **Validation AUC:** the gap to the 19-locus run stays about .05 from epoch 20 to 79 in single-drug, so it isn't just learning more slowly.

**Some folds never started learning.** I counted a fold as stuck if its train loss at stop was still at least 90% of its epoch-1 value (my threshold):

| CNN arm | Stuck folds | Levofloxacin share |
|---|---|---|
| Per-drug | 14/275 | 11 |
| 19 loci | 7/275 | 7 |
| WHO | 13/275 | 13 |
| 64 loci | 18/275 (mean AUC .727) | 9 |

- **Levofloxacin:** it gets 2 updates per epoch at lr 1.2e-4, about 168 in total. It is stuck in every arm, not just 64 loci.
- **Other drugs at 64 loci:** stuck folds also appear for ethionamide, moxifloxacin and capreomycin, mostly in `dna` and `dna_regulatory`.

## Why the CNN drops

1. **Architecture dilution (most likely, untested):** each locus is stacked as an input channel on one shared position axis, and the first layer stays at 64 filters.
   - Going to 64 loci adds exactly 172,800 parameters, all in that first layer, so 3.4× more loci share the same 64 feature maps. LocusFusion encodes each locus separately, which would explain why it is unaffected.
   - The odd parameter counts (54 loci larger than 64) are harmless: the flatten width follows the longest locus, and PPE35 at 6,020 bp is only in the WHO set.
2. **Too little training on low-label drugs:** levofloxacin alone is 42% of the macro drop (−.157). Excluding it and all stuck folds, the single-drug drop is still −.018, with 73% of paired differences negative.
3. **Few resistant isolates (independent of update count):** the drop shrinks as resistant counts rise (Spearman ≈ .8 in both single- and multi-drug). Multi-drug gives every drug the same updates, so too few updates can't be the whole story.
4. **Promoters merged into the DNA channels:** `mdcnn_trunk_per_modality=false` puts promoter windows in the same trunk, giving 70 channels in single-drug and 108 in multi-drug. Excluding levofloxacin, `dna_regulatory` falls most (−.041 vs −.030 for `dna`). The other modality sets, which add separate trunks, fall less (−.009 to −.015).

## What this means for the paper

- **Wording:** don't write "more loci hurt". The supported claim is that the stacked CNN degrades at 64 loci while LocusFusion and LASSO don't.
- **LocusFusion vs CNN at 64 loci:** that gap is partly against a baseline that didn't fully train, so a reviewer could challenge it.
- **Levofloxacin CNN numbers:** treat them as fragile in every arm, including the 19-locus control.
- **LocusFusion's flat CV is slightly flattering:** CV reports the peak epoch, and validation AUC gets noisier at 64 loci. Averaged over ±5 epochs around the peak, 19 → 64 is −.008 rather than +.001. Test AUC goes the other way (+.007 excluding levofloxacin, +.020 with it), so the true change is roughly zero.
- **Multi-drug CNN:** the drop is much smaller on test (−.009) than CV (−.025), and is mostly levofloxacin, moxifloxacin and streptomycin.

## What I couldn't check

- **No validation loss:** histories store only `train_loss` and `val_auc`, and only the best weights are saved. Overfitting is inferred from those two curves, and validation-loss curves would need new runs.
- **The 14 re-extracted loci:** I didn't re-validate them. LocusFusion and LASSO staying flat argues against corruption, but both could simply ignore a bad locus.

## Suggested next tests

None of these has been submitted.

1. **Training budget (cheapest):** rerun the CNN at 64 loci on levofloxacin, moxifloxacin and ethionamide, with `min_epochs` set by update count or a higher learning rate. This separates too little training from a real capacity limit.
2. **Locus ladder:** run the CNN on 19 + k loci (k = 0, 5, 15, 30, 45) to see if the loss grows with locus count.
3. **Wider first layer:** scale the filters with locus count (about 215 at 64 loci keeps the 19-locus ratio), which directly tests dilution.
4. **Validation loss:** add it to `history` for future runs.
