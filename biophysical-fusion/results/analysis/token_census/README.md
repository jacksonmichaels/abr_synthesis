# locusfusion token census — 2026-09-01

How many variant tokens does `models/locusfusion.py` actually emit, per locus,
per isolate? And does anything upstream — a frameshift, an indel, a broken
alignment record — quietly blow that count up?

Reproduce:

```bash
python scripts/token_census.py --all-regulatory        # ~2 min, writes the CSVs
python scripts/plot_token_census.py                    # writes the four figures
python scripts/token_census.py --loci katG inhA --verify   # the loader cross-check
```

Cohort: all 17,942 isolates that survive the phenotype join, 19 coding loci from
`REAL_GENOTYPE_DIR` and all 49 promoter windows from `REAL_REGULATORY_DIR`.
1,525,070 (isolate, locus, stream) rows.

The census recomputes occupancy from the FASTAs rather than materialising every
delta block (that would be terabytes), so it is only as good as its agreement
with the real loader. `--verify` builds the actual `datasets.load_dataset`
delta blocks for a slice and compares token for token: **14/14 block-streams
agree exactly** (400 ISONIAZID isolates over katG/inhA, 500 STREPTOMYCIN
isolates over rpsL/rrs/gid).

## The headline: the tokenizer is doing what it claims, for ~99.7% of pairs

| stream | (isolate, locus) pairs | median | p99 | over `max_variants` = 16 |
|---|---|---|---|---|
| `nt` (DNA, alignment columns) | 340,898 | 0 | 5 | 361 (0.106%) |
| `aa` (protein + biophysical, residues) | 305,014 | 0 | 4 | 1,338 (0.439%) |
| `reg` (promoter windows) | 879,158 | 0 | 2 | 2,579 (0.293%) |

Summing `nt` + `aa` over all 19 coding loci, an isolate presents a **median of
20 tokens** — the architecture's premise (a clonal cohort collapses to a handful
of tokens, and a susceptible isolate is nearly the empty set) holds. The median
is 0 for 27 of the 36 coding (locus, stream) cells: at most loci most isolates
are wild type and present nothing but the `[WT]` sentinel. `frac_zero` in
`summary.csv` spans the range, from ~0 at gyrA and tlyA (where a lineage marker
is near-fixed in the cohort) to 0.995 at fabG1.

Per-locus min / quantiles / max: `fig_token_distribution.png`.
Upper tail on a log axis: `fig_token_tail.png`.
Machine-readable: `summary.csv`, and `token_census.csv.gz` for the per-isolate rows.

## The tail is real, and it is indels — including in-frame ones

**99.85% of the `aa` pairs that overflow the cap carry a CDS indel.** The
mechanism is in `datasets/biochem.translate_seq`: it degaps each isolate's own
CDS window before translating, so the residue index is that isolate's codon
index, while `delta_zero_columns` compares residue *k* of the isolate against
residue *k* of H37Rv. Any indel upstream slides the whole remainder of the
protein out from under that comparison.

The surprise is which kind of indel hurts more:

| CDS indel | n pairs | median aa tokens | 75th pct | max | share over the cap |
|---|---|---|---|---|---|
| none | 302,619 | 0 | 1 | 101 | 0.0007% |
| **in-frame (len % 3 == 0)** | 354 | **80.5** | 298 | **1,006** | **71.8%** |
| frameshift (len % 3 != 0) | 2,041 | 19 | 30 | 129 | 53.0% |

An in-frame indel is the *worse* case, not the safe one. A true frameshift
usually runs into a stop codon within a few dozen codons and `translate_seq`
truncates the protein there, which bounds the damage at ~129 residues. A 3 bp
in-frame deletion changes nothing about the frame and so hits no early stop — it
just silently re-indexes every downstream residue by one, and the delta then
lights up most of the protein. That is how embA reaches **816 aa tokens off a
DNA stream whose maximum is 15**, and rpoB reaches 1,006 residues.

`fig_frameshift.png` (per-locus rates, and the token count split three ways) and
`fig_nt_vs_aa.png` (every isolate's aa count against its nt count; the indel
classes sit far above the diagonal) are the two plots for this.

Worst loci by CDS-indel rate: gid 4.63%, pncA 3.24%, ethA 2.89%, eis 0.61%,
katG 0.55%, rpoB 0.54%. Those are also the loci where real resistance mutations
*are* frameshifts and truncations (pncA for PZA, gid for STR, ethA for ETO,
katG for INH), so this is not noise that can simply be dropped — it is signal
arriving in a form the residue-indexed comparison mangles.

## A second, separate failure: records that differ almost everywhere

119 (isolate, locus) `nt` pairs, over 108 isolates, differ from H37Rv at **more
than half** of the block's columns — pncA 55, ethR 27, katG 14, ethA 14, eis 8,
gid 1. katG's worst isolate differs at 2,423 of 2,488 columns; ethR's at 751 of
751. These are not variant calls, they are broken or mis-oriented alignment
records, and they are independent of the indel story above (they inflate the
`nt` stream, which has no translation step). They are a small enough set to
inspect by hand from `token_census.csv.gz`.

## What this costs the model, concretely

`_select_variants` keeps the **first 16 occupied columns in positional order**
and discards the rest, so an affected isolate hands stage 1 the first 16
residues of a re-indexed tail as if they were 16 independent point mutations.

* 1,699 (isolate, locus, stream) cells are truncated by the cap.
* **1,424 of 17,942 isolates (7.94%) are truncated somewhere** in their nt/aa
  streams — a much larger number than the 0.1–0.4% per-pair rate, because each
  isolate is exposed at 36 (locus, stream) cells.
* Uncapped, the p99 of total tokens per isolate is 583 and the max 3,658;
  capped, those become 64 and 100. The cap is doing real work — the question is
  whether what it throws away was worth keeping, and for an indel-carrying
  isolate the answer is that what it *keeps* is misleading.

Two mitigations already in the architecture, worth stating so the problem is not
overstated: the `[WT]` sentinel carries the true pre-cap count through
`wt_proj`, so the model is told "there were 675 differing residues here" even
though it only sees 16 of them; and the `nt` stream stays in shared alignment-
column space and is therefore *not* re-indexed, so the DNA view of an
indel-carrying isolate remains correct. The corrupted view is confined to
`protein` + `biophysical`.

## Also noted

`rrs` and `rrl` have no CDS in the reference, so `ProteinModality` falls back to
translating the whole aligned window. The result is a 50- and 46-residue
"protein" (translation hits a stop almost immediately), and it is zero for
over 99% of isolates. Harmless in practice, but it is not a protein and the census refuses
to report an aa stream for those two loci.

## Files

| file | what |
|---|---|
| `token_census.csv.gz` | one row per (isolate, locus, stream): token count, nt sub/del/ins/ambiguous breakdown, `cds_indel`, `indel_class`, `aa_first_diff`, `aa_tail_frac` |
| `summary.csv` | one row per (locus, stream): min, p50/p90/p99/p99.9, max, `frac_zero`, `frac_over_cap`, `frac_indel`, `frac_frameshift`, `frac_block_half` |
| `fig_token_distribution.png` | per locus: min, quantile spans, max — one panel per stream |
| `fig_token_tail.png` | share of isolates above x tokens, log y, per locus |
| `fig_frameshift.png` | CDS-indel rate per locus, and aa tokens by indel class |
| `fig_nt_vs_aa.png` | aa tokens vs nt tokens per (isolate, locus) |
