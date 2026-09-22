# `lasso_wholegenome_20260903` — L1-logistic on the whole genome

## The question

`lasso_wholeseq_20260901` finding 6 suspects that the 19-locus models read
co-resistance rather than mechanism. `ablation_eto` measured it: for
ETHIONAMIDE, four loci with no ETO mechanism reach CV 0.7241, and adding a
mechanism locus (+0.097) is worth no more than adding a locus that cannot be
causal (+0.085). Neither run can go further, because **all 19 loci on disk are
resistance genes for some drug** — "not this drug's mechanism" and "carries
co-resistance" name the same columns.

Genome-wide they separate. 4.41 Mb of H37Rv is overwhelmingly *not* a resistance
gene, so where the coefficients land is now informative.

## Status

**Complete: 11 of 11 drugs**, run 2026-09-03, in `all_drugs/`. The ETHIONAMIDE
pilot in `pilot/` came first — the smallest cohort (n = 2,153) and the drug with
three existing anchors to compare against — and everything under "Findings" below
that names ETHIONAMIDE is that pilot, unchanged. The full sweep is in
`all_drugs/`, and finding 4 is what it adds.

## Cost, measured rather than estimated

| | 19-locus `onehot`/all | whole genome |
|---|---:|---:|
| nonzeros per isolate | 16,130 | 1,228 |
| columns | 33,218 | 127,697 (ETO cohort, `min_count=1`) |
| seconds per fit | 23 | < 1 |
| cell wall time | 141 s | 91 s, of which 77 s is parsing |

A variant matrix is sparse where a one-hot alignment is dense, so the whole
genome is an order of magnitude **cheaper** in nonzeros than the 19 curated
loci. Parsing dominates and is one-time: 77 s for 2,153 isolates on 4 procs
scales to ~11 min for the full 17,942.

## Result — the ETHIONAMIDE pilot

The 11-drug sweep is finding 4.

| input | CV AUC | test | support |
|---|---:|---:|---:|
| `ethA+ethR+inhA` (BIG-TB per-drug map) | 0.6641 | 0.6656 | 74 |
| `rpoB+pncA+katG+embB` (no ETO mechanism) | 0.7241 | 0.7922 | 491 |
| 19 curated loci | 0.8118 | 0.8591 | 225 |
| **whole genome** | **0.8083 ± 0.0146** | 0.7715 | 192 |

## Findings

### 1. The whole genome is worth nothing over the 19 curated loci (ETHIONAMIDE)

0.8083 against 0.8118, a difference of 0.0035 against a fold spread of 0.0146.
4.41 Mb and 127,697 candidate columns buy no more than the 19 genes somebody
chose by hand. Whatever the extra 99.1% of the genome carries, it is already
present in the curated loci.

Finding 4 confirms this over all 11 drugs.

### 2. It finds the mechanism unaided (ETHIONAMIDE) — and it is the largest coefficient

The single largest coefficient of 127,697 is **`1673425:C>T`**, which snpEff
annotates to the intergenic region `Rv1482c–fabG1`: the *fabG1–inhA* operon
promoter, the canonical *c-15t* ethionamide/isoniazid mechanism. Three more
mechanism features follow (`fabG1:1674048`, `inhA:1674481`, `inhA:1674263`).

So the answer to "does it identify positions in known loci" is yes, and
emphatically: handed the whole genome with no curation, no locus list and no
gene annotation in the objective, the strongest single variant it selects is the
right one. `ablation_eto` finding 3 said `fabG1` alone beats the drug's own
locus map; this says a genome-wide search finds that promoter position first.

### 3. But 86% of its weight is on sequence with no resistance role (ETHIONAMIDE)

Top-60 coefficients by |coef| mass:

| class | mass | features |
|---|---:|---:|
| PE/PPE/esx repeat families | 40.2% | 28 |
| other genes, no resistance role | 35.6% | 22 |
| ETO mechanism (gene or its flank) | 14.3% | 4 |
| resistance gene for another drug | 9.9% | 6 |

The largest class is *PE_PGRS* / *PPE* / *esx* — the hypervariable repeat
families that are the standard lineage markers of *M. tuberculosis*, and also
the regions short-read variant calling handles worst. `PPE54`, `PE_PGRS27`,
`PE_PGRS28`, `PE_PGRS33`, `PE_PGRS51`, `esxK` all carry more mass than *inhA*.

This is the strongest form yet of the parent run's finding 6, and it is no
longer confined to co-resistance: ETHIONAMIDE resistance is predictable at
CV 0.81 with **six sevenths of the model's weight on sequence that cannot
cause resistance to anything**. It also explains `ablation_eto`: if lineage
alone carries this much, then any locus correlated with lineage — `rpoB`,
`pncA` — buys as much as a real mechanism locus, which is exactly what was
measured.

### 4. Over all 11 drugs, the same result — and this is the run's headline

`all_drugs/`, macro CV AUC over the 11 shared drugs, against the matched
19-locus `delta` arm of `lasso_wholeseq_20260901`:

| | macro CV | macro test |
|---|---:|---:|
| lasso, `delta`, 19 curated loci | 0.9152 | 0.9083 |
| **lasso, `delta`, whole genome** | **0.9174** | **0.9170** |
| lasso, `onehot`, 19 curated loci | 0.9241 | 0.9287 |

**+0.0022 CV over its matched control, 6 wins of 11, Wilcoxon p = 0.78.** 110x
the sequence and 27x the columns (426,457 mean against 11,474) buy nothing
measurable, and the `onehot` arm on 0.9% of the genome is 0.0067 *ahead* of it.
The macro is a cancellation rather than per-drug equality: AMIKACIN +0.0397 and
MOXIFLOXACIN +0.0190 against LEVOFLOXACIN −0.0300 (269 isolates, 22 nonzero
coefficients — the cell least able to afford a bigger search space), with the
other eight inside ±0.010.

Cost, mean per drug: **58 s** on 426,457 columns, against 1,405 s for the
19-locus `onehot` arm's 22,384. A variant matrix is sparse where a one-hot
alignment is dense, so genome-wide is an order of magnitude *cheaper* than the
curated loci it is compared against. Support barely moves either — 300 nonzero
coefficients genome-wide against 246 at 19 loci.

### 5. The weight lands outside the WHO catalogue for 9 of 11 drugs

`who_mass.py` matches every nonzero coefficient of the selected model against WHO
2023 Catalogue Table 21 (tier 1 + tier 2, by gene name *and* Rv tag), counting an
intergenic region flanking a Table-21 gene as inside — which is where the
Table-22 promoter windows live. Written to `who_mass.csv`:

| bucket | mean share of \|coef\| mass |
|---|---:|
| the drug's own Table-21 genes | 27.1% |
| another drug's Table-21 genes | 10.6% |
| outside Table 21 entirely | **62.4%** |

Outside is the largest of the three buckets for **9 of 11 drugs** and an outright
majority for 8. The two exceptions are LEVOFLOXACIN (22 coefficients in total)
and STREPTOMYCIN (43.2% on its own genes against 41.8% outside). ETHIONAMIDE is
the extreme at 81.2% outside; PYRAZINAMIDE puts more mass on *other* drugs' genes
(18.6%) than on its own (5.9%), which is finding 6 of the parent run — MDR
co-occurrence — measured genome-wide.

Finding 3's ETO-specific 86% is therefore not an ETO peculiarity. ETO is the
extreme case, but the outside share never falls below 41.8% on any drug, and it
is a majority on eight of eleven.

## What this does not settle

- **Coefficient mass is not a variance decomposition, at 11 drugs either.** 62.4%
  of the weight is not 62.4% of the AUC. Splitting that needs a masked arm (fit
  with all WHO-catalogue sequence removed) or a lineage-held-out split.
- **One seed.** Differences under ~0.01 are unresolved, which includes finding
  4's +0.0022 and every per-drug margin except the three named there.
- **Lineage or mapping artifact?** PE/PPE/esx variants are unreliable under
  short-read calling, so some of that 40.2% may be batch/study signal rather
  than population structure. A `--mask-repeats` arm separates them and is
  cheap.
- **`onehot` is unavailable here.** A wild-type indicator per site makes the
  matrix dense at ~10^6 sites, so this is a `delta`-encoding arm and compares to
  the parent run's `delta` cells, not its 0.9241 `onehot` headline.

## Reproduce

```bash
# from biophysical-fusion/
# the pilot
python results/experiments/lasso_wholegenome_20260903/lasso_wholegenome.py \
    --drugs ETHIONAMIDE --procs 4 \
    --out results/experiments/lasso_wholegenome_20260903/pilot

# all 11 drugs
python results/experiments/lasso_wholegenome_20260903/lasso_wholegenome.py \
    --drugs all --procs 4 \
    --out results/experiments/lasso_wholegenome_20260903/all_drugs

# where the weight lands -> who_mass.csv
python results/experiments/lasso_wholegenome_20260903/who_mass.py \
    --run results/experiments/lasso_wholegenome_20260903/all_drugs
```

`notebooks/wholegenome_lasso.ipynb` is this run on its own; §14 of
`notebooks/results.ipynb` is it in the project's context.

Run interactively on 4 CPUs, 2026-09-03. The script imports `lasso_wholeseq.py`
for `_metrics`, `_fit`, the C grid and the summary schema, so the protocol
cannot drift from the arm it is compared against.
