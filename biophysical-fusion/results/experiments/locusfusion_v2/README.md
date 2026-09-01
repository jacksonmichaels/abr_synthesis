# locusfusion_v2 — the same architecture, with the tokenizer fixed

Submitted and completed **2026-09-01**. 55 SLURM jobs (5 modality sets x 11
drugs), single-drug task, per-drug loci. Reproduce with `submit.sh`; read the
tables with `compare.py`; the mechanistic checks are `attribution_check.py`.

**Nothing about the architecture changed.** Same two-stage structure, same `[WT]`
sentinel, same keyed summary norm, same read-out, same `d_model=128`, same
300 epochs / patience 30 / `--min-epochs 50` / `--save-weights best` / seed 0,
same loci, same cohorts. Only the tokenizer underneath it is different, and the
parameter count barely moved (646,273 against 645,763). The write-up of what
changed and why is `../CODE_CHANGES_20260901.md`.

## Result

5-fold CV AUC, macro over the 11 drugs, against both controls — `newmodels_full`
(the same architecture on the old tokenizer) and `full_run_v2`'s `mdcnn`:

| cell | v2 | old tokenizer | delta | mdcnn | v2 − mdcnn |
|---|---:|---:|---:|---:|---:|
| `dna` | 0.8994 | 0.8825 | **+0.0169** | 0.8919 | +0.0074 |
| `dna_protein` | 0.9007 | 0.8830 | **+0.0177** | 0.8968 | +0.0039 |
| `dna_biophysical` | 0.8999 | 0.8828 | **+0.0171** | 0.8977 | +0.0022 |
| `dna_regulatory` | 0.9092 | 0.8913 | **+0.0179** | 0.9025 | +0.0067 |
| `all_modalities` | 0.9089 | 0.8920 | **+0.0169** | 0.9086 | +0.0003 |
| **macro over cells** | **0.9036** | **0.8863** | **+0.0173** | 0.8995 | +0.0041 |

**The gap to `mdcnn` closed and reversed.** On the old tokenizer locusfusion's
`all_modalities` cell lost to `mdcnn` by −0.0166 (0.8920 vs 0.9086); it now
ties it (0.9089 vs 0.9086), and across all five cells locusfusion is ahead by
+0.0041. This is the first time a transformer arm in this project has not lost.

Per drug, `all_modalities`:

| drug | v2 | old | delta | mdcnn |
|---|---:|---:|---:|---:|
| ISONIAZID | 0.9678 | 0.9635 | +0.0043 | 0.9645 |
| RIFAMPICIN | 0.9776 | 0.9735 | +0.0041 | 0.9774 |
| ETHAMBUTOL | 0.9428 | 0.9375 | +0.0053 | 0.9427 |
| PYRAZINAMIDE | 0.9335 | 0.9133 | +0.0202 | 0.9330 |
| STREPTOMYCIN | 0.9385 | 0.9309 | +0.0077 | 0.9323 |
| KANAMYCIN | 0.8785 | 0.8564 | +0.0220 | 0.8810 |
| AMIKACIN | 0.8754 | 0.8454 | +0.0300 | 0.8745 |
| CAPREOMYCIN | 0.8581 | 0.8449 | +0.0132 | 0.8636 |
| LEVOFLOXACIN | 0.9514 | 0.9129 | +0.0385 | 0.9570 |
| MOXIFLOXACIN | 0.8614 | 0.8360 | +0.0255 | 0.8680 |
| ETHIONAMIDE | 0.8131 | 0.7979 | +0.0152 | 0.8009 |

**The tokenizer fix improves every one of the 55 cells** — 11/11 drugs in
`all_modalities`, and the same in the other four. A uniform +0.017 across five
modality sets is what a fixed *input representation* looks like; it is not the
shape of a lucky seed.

Against `mdcnn` it is 6 wins, 1 tie and 4 losses across the 11 drugs. The four
losses are KANAMYCIN, CAPREOMYCIN, LEVOFLOXACIN and MOXIFLOXACIN — three of them
the smallest cohorts in the set, where fold variance exceeds the gap (see
TODO.md open question 2). No claim is made that locusfusion beats `mdcnn` per
drug; the claim is that it no longer loses in aggregate.

## What is NOT established by this run

- **This is the single-drug, per-drug-loci task only.** The 19-locus arm
  (`alllocus_run_v2` as control) and the joint multi-drug arm were not rerun.
  The old tokenizer's numbers for those cells in `newmodels_full` are still on
  the books and are still measuring the bug.
- **The improvement is not attributed between the two changes.** The coordinate
  fix and the embedding rewrite shipped together, and an N-call fix rode along
  with them. Separating them needs an arm with one held back, which this run
  does not have.
- **Nothing here says the architecture is better than `mdcnn`.** +0.0041 over
  five cells, on 11 drugs whose fold SD is larger than that, is a tie stated
  politely.

## The mechanistic checks

The claim is mechanistic, so it gets a mechanistic measurement — the discipline
`token_signal` imposed on itself. `attribution_check.py`, on the trained
ISONIAZID `all_modalities` fold, 512 isolates:

```
read-out attention over 2 keys (uniform would be 0.5000, KL 0)
  ISONIAZID   KL from uniform 0.0308   max weight 0.8230   mean top-1 0.5929
              loci by mean weight: katG 0.543, inhA 0.457

  locus    stream   position      ref -> alt     count
  katG     nt         463.33     nt G -> nt T    201
  katG     aa         463.00     aa R -> aa L    201
  katG     nt         315.33     nt G -> nt C    167
  katG     aa         315.00     aa S -> aa T    167
  katG     nt         767.00     nt T -> nt ?     45
```

Three things to read off that:

1. **The two streams co-register.** katG residue 315 appears through the
   nucleotide path at 315.33 and through the protein path at 315.00, in exactly
   the same 167 isolates; residue 463 likewise in 201. That is S315T (the
   dominant INH resistance mutation, AGC→ACC at the second base, which is what
   the `.33` says) and R463L (the lineage marker). **Under the old tokenizer the
   nucleotide token for S315 sat at 357.3 against the protein token's 314** — 43
   codons apart, so within-locus attention could not have paired them even in
   principle. This is the fix, visible in the trained model.
2. **The N-call fix is load-bearing on real data.** `katG nt 767 T → ?` in 45 of
   512 isolates: unresolved base calls that the old encoding turned into an
   all-zero column and therefore read as wild type.
3. **Read-out attention is not uniform** (KL 0.0308, top weight 0.823 against a
   0.5 uniform), so the locus summaries are not collinear — the failure mode
   `token_signal` measured in setfusion at exactly 1/8. With only two loci in
   this cell this is a weak test; the 19-locus arm would be the real one, and it
   has not been run.

## Cost

44–375 s per cell against `newmodels_full`'s 2,460 s for the largest, on the
same hardware request (64G, 1 GPU, vram23). The blocks are int8 symbol ids
rather than float32 one-hots, so they are 5–20x leaner in host memory and the
tokenizer is an integer gather instead of a channel-wise reduction. The whole
55-job grid finished in ~40 minutes of wall clock.

All 55 jobs exited clean: 55 result files, no tracebacks, no OOM, and — worth
noting specifically — **no occupancy warnings**, which is the guard that fires
if the per-column reference metadata fails to load and every column starts
reading as a variant.
