# token_signal — make the setfusion mechanisms load-bearing

Reproducible via `submit.sh` in this folder.

## The finding this exists to fix

`full_run_v2/all_modalities__setfusion` was measured directly (MOXIFLOXACIN,
2,868 phenotyped isolates, held-out test = 574 / 78 R):

| what was measured | value |
|---|---|
| per-isolate spread of an encoded token / its norm | **0.00136** |
| attention of the drug query over the 8 blocks | **0.1250 = 1/8, flat to 4 dp** |
| test AUC, `locus_emb := 0` | -0.009  [-0.040, +0.021] |
| test AUC, gyrB<->gyrA embeddings swapped | -0.009  [-0.025, -0.001] |
| test AUC, `modality_emb := 0` | -0.012  [-0.050, +0.028] |

and a linear probe on the model's own representations, same split:

| representation | test AUC |
|---|---|
| `enc` — encoder output | 0.8054 |
| `pre` — after adding both embeddings | 0.8054 |
| `post` — after the fusion transformer | 0.8048 |
| the model's own head | **0.7946** |

Three conclusions, none of them about capacity:

1. **0.14% of a token varies with the genotype.** The other 99.86% is a constant
   that only says which locus this is. The transformer and the drug query see
   nearly the same vector for every isolate, which is why attention is uniform.
2. **The locus embedding is invisible to the readout by construction.** It adds
   the *same* vector to every isolate, so any affine head absorbs it into its
   intercept — `enc -> pre` changes the probe by nothing to four decimals. The
   only route by which it could matter is the fusion nonlinearity, and the
   ablations say that route is dead too.
3. **Everything after the encoder is worth -0.011 AUC.** Plain logistic
   regression on the raw encoder output beats the trained model.

So the mechanisms do not fail for want of width. They fail because the signal
they are supposed to operate on is three orders of magnitude below the constant
they are riding on. `setfusion_scaling` sweeps width; this sweeps the ratio.

## The two changes

**`--delta` (input, all architectures).** *M. tuberculosis* is clonal: isolates
differ at a handful of positions in a 2.6 kb gene, so a one-hot spends ~99.9% of
its columns restating sequence identical in every isolate. `--delta` zeroes every
column matching the **real MT_H37Rv record** in each alignment — not a cohort
consensus, which would be fitted on test isolates too. Shape, channels and
alphabet are unchanged, and no discriminative information is lost: the columns
removed are constant across the cohort. Measured occupancy on MOXIFLOXACIN:

| block | plain | delta |
|---|---|---|
| dna:gyrB | 100.000% | 0.015% |
| dna:gyrA | 99.999% | 0.129% |
| protein:gyrA | 99.878% | 0.375% |
| regulatory:gyrA | 100.000% | 0.020% |

**`--token-norm keyed` (setfusion only).** `KeyedTokenNorm` standardises each
token across the batch with statistics kept per `(modality, locus)`. Subtracting
the per-key mean deletes the locus-constant; dividing by the per-key std puts the
genotype at unit scale. Two consequences, and they are the point: the transformer
starts seeing input that differs between isolates, so attention *can* stop being
uniform; and locus identity stops being carried redundantly by that constant, so
`locus_emb` becomes the only thing saying which locus a token is — load-bearing
instead of decorative. Statistics are keyed rather than positional on purpose, so
`forward(xs, keys=...)` with a subset or reordering still works and setfusion's
count-independence survives.

Both default OFF. `token_norm='none'` creates no module, so no `state_dict` keys
are added and every existing checkpoint still loads.

## The grid

Two drugs x four arms, `all_modalities__setfusion`, at `full_run_v2` training
settings (300 epochs, patience 30, `--min-epochs 50`, one job each).

| arm | flags |
|---|---|
| `a0_control` | — (reproduces the `full_run_v2` cell) |
| `a1_tokennorm` | `--token-norm keyed` |
| `a2_delta` | `--delta` |
| `a3_both` | `--token-norm keyed --delta` |

**MOXIFLOXACIN** is where every diagnostic above was measured, but it is badly
powered — 78 resistant isolates in the test split, which is why the ablation CIs
span +/-0.04. **ISONIAZID** carries ~17.4k phenotyped isolates and is the drug the
project's core question is about; it is the arm to believe. Read `cv_auc_mean`
over the 5 folds first, not the single test number.

## How to tell whether it worked

Not by AUC alone. The claim is mechanistic, so the mechanism is what gets
checked — rerun the ablation probe against each new checkpoint:

* **`locus_emb := 0` should now cost real AUC.** If deleting it is still free,
  the keying is still inert and the arm did not fix the thing it targeted.
* **attention should stop being uniform.** Flat 1/8 means the tokens are still
  collinear.
* **the linear probe on `enc` should no longer beat the model's own head.** If it
  still does, the fusion stack is still destroying more than it adds.

An arm that raises AUC while leaving all three unchanged has improved the model
for some *other* reason, and the mechanistic claim is still unsupported.

## Known risk

`--delta` makes the input extremely sparse (0.015-0.375% occupied). A ReLU CNN
stack with no input normalisation may simply see near-zero activations, and the
BIG-TB learning rate `exp(-9) ~ 1.2e-4` is already ~10x below a normal Adam
setting. If `a2_delta` collapses while `a3_both` does not, that ordering is the
evidence that the sparsity needed the downstream normalisation to be usable.
