# `ablation_eto` — where ETHIONAMIDE's 19-locus lift comes from

## The question

ETHIONAMIDE takes the largest per-drug lift in this run: **+0.148** CV AUC from
being handed all 19 loci instead of its own three (0.6641 -> 0.8118). The parent
README reads that through finding 6 — the 19-locus models pick loci that cannot
be causal, so part of the lift is MDR co-occurrence rather than mechanism — and
lists ETO's off-target picks (`pncA` 17% of top-40 coefficient mass, `rpoB` 16%)
as evidence.

There is a competing explanation the parent README does not consider. ETO's
per-drug map is `ethA+ethR+inhA`, and `fabG1` is **not in it**. The dominant ETO
mechanism is the *fabG1–inhA* operon promoter (parent README finding 1), which
lives in the `fabG1` locus FASTA, not the `inhA` CDS. On that reading the lift is
a mechanism locus the per-drug map omits, and co-resistance is incidental.

The two canned locus universes differ by 16 loci at once, so neither can
separate them. These cells add **one locus at a time**.

## The cells

ETHIONAMIDE only, `onehot` only, protocol identical to the parent run. The two
anchors already existed and are quoted, not re-run.

| loci | CV AUC | test | cols | support | what it isolates |
|---|---:|---:|---:|---:|---|
| `ethA+ethR+inhA` | 0.6641 | 0.6656 | 4,923 | 74 | the per-drug map (anchor) |
| `+ fabG1` | **0.7613** | 0.7794 | 4,949 | 109 | one added locus, WHO tier-1 for ETO |
| `+ rpoB` | 0.7493 | 0.7575 | 5,336 | 119 | one added locus, cannot cause ETO resistance |
| `+ pncA` | 0.7272 | 0.7606 | 6,848 | 394 | ditto, and the largest off-target mass |
| `fabG1` alone | 0.7262 | 0.7110 | 26 | 11 | the promoter with no *ethA*/*ethR*/*inhA* at all |
| `rpoB+pncA+katG+embB` | 0.7241 | 0.7922 | 7,652 | 491 | four loci, zero ETO mechanism |
| all 19 | 0.8118 | 0.8591 | 15,974 | 225 | the parent run's cell (anchor) |

Fold spread is 0.010–0.016 across these cells, so differences under ~0.012 are
not resolved on one seed at n=2,153.

## Findings

### 1. Both explanations are true, and they are the same size

Adding `fabG1` buys **+0.097**. Adding `rpoB` — a locus that cannot cause
ethionamide resistance — buys **+0.085**. The gap between them is 0.012, at the
edge of what these folds resolve. A locus with a mechanism and a locus without
one are worth indistinguishable amounts to this model.

### 2. Co-resistance alone predicts ETO at 0.724

`rpoB+pncA+katG+embB` carries no ETO mechanism whatsoever and reaches 0.7241
CV / 0.7922 test — **+0.060 above the drug's own three loci**. Its coefficients
are exactly what that implies: `pncA:687`, `embB:220`, `embB:1197`, `katG:551`,
`rpoB:1628`. Finding 6 of the parent README is confirmed, not merely suspected,
and this is the number to quote for it.

### 3. The per-drug map is worse than one locus it leaves out

`fabG1` alone — **26 columns, 11 nonzero coefficients** — reaches 0.7262,
beating the full `ethA+ethR+inhA` map (0.6641) by +0.062. Its support is the
promoter window and nothing else: `fabG1:85` (the canonical *c-15t* position),
`fabG1:92`, `fabG1:83`, `fabG1:66`. Eleven coefficients get 90% of the way to
the 19-locus model's 0.8118.

The BIG-TB per-drug map is therefore not a neutral locus-matched control for
ETHIONAMIDE — it excludes the drug's dominant mechanism, and the parent run's
0.6641 measures that exclusion, not the model.

### 4. The two effects are roughly additive

0.6641 + 0.097 (mechanism) + 0.060 (co-resistance) = 0.821, against 0.8118 for
all 19 loci. Read as a split of the +0.148, that is ~62% mechanism and ~38%
co-resistance, with the caveat that single-locus additions cannot see
interactions and the residual is within noise of zero.

## What this does not settle

Whether the co-resistance in finding 2 is confounding or legitimate signal —
same limitation as the parent run, same fix (a lineage-held-out split). It is
also confined to ETHIONAMIDE; every other drug's per-drug map may or may not
omit a mechanism locus, and only ETO was tested.

More fundamentally: all 19 loci on disk are resistance genes for *some* drug, so
"cannot cause ETO resistance" and "carries co-resistance" are the same set here.
Separating mechanism from lineage needs input that is mostly neither, i.e. a
genome-wide arm. No run in this project has one.

## Reproduce

`--loci` was added to `lasso_wholeseq.py` for this; the tag is the joined locus
list, so each cell names itself.

```bash
# from biophysical-fusion/
S=results/experiments/lasso_wholeseq_20260901/lasso_wholeseq.py
O=results/experiments/lasso_wholeseq_20260901/ablation_eto
for L in "ethA ethR inhA fabG1" "ethA ethR inhA rpoB" "ethA ethR inhA pncA" \
         "fabG1" "rpoB pncA katG embB"; do
    python -u $S --drugs ETHIONAMIDE --encoding onehot --loci $L --out $O
done
```

Run interactively on 4 CPUs, 2–83 s per cell, 2026-09-03.
