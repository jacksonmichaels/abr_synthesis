# GenomeFusion

LocusFusion over the whole H37Rv genome. Design and rationale:
`docs/genomefusion/genomefusion_design.html`.

Every base of the genome belongs to exactly one locus — a gene, or the
intergenic segment between two genes. Only the loci an isolate actually changes
enter the network, and a locus holds as many tokens as the isolate has changes
in it. There is no promoter window and no per-locus token cap.

## The partition

Built by `partition.py` from `data_snapshot/genbank/genome.gbff` (NC_000962.3,
4,411,532 bp, 4,008 gene features).

| | |
|---|---|
| loci | **6,776** = 3,749 gene + 3,027 intergenic |
| masked | 402,567 bp (9.1%): the Marin 2022 RLC BED (773 regions) plus 254 genes whose Mycobrowser category is PE/PPE or insertion seqs and phages |
| overlap | 13,121 bases lie in two genes |

**Overlapping genes.** Each shared base goes to the gene whose START codon is
nearer, so start codons stay whole, and the token's `overlap` flag records that
a second gene carries it. The cost is that 840 CDS lose a base of their STOP
codon to a neighbour's start codon (the common ATGA-type overlap) — *pncA*
among them. With RLC masking too, 951 of 3,680 CDS (25.8%) have at least one
incomplete codon, 4,342 codons in all, about 0.3% of the genome's codons. A
token landing in an incomplete codon is called `unreadable`, never a false
nonsense.

**Promoters are not loci.** A fixed 100 bp upstream window would run into a
neighbouring gene for 61% of genes, so the same base would be a token twice.
Instead an intergenic token carries its distance upstream of the next start
codon on each strand. Verified: the *inhA*-promoter position 1,673,425 reads 15
bp upstream of *fabG1* and 126 bp upstream of Rv1482c.

## Tokens

`tokenizer.py` reads the per-isolate Pilon VCFs and emits one token per changed
base or per indel event. QC (`allele_category`) is the same rule as the
project's MSA extractor, so a variant called alt/ref/missing here is the one
v3's tokens would carry. Synonymous changes are dropped by default, as in v5.

Residues are read on the H37Rv codon grid, exactly as v3/v4/v5, so an indel
never turns the rest of a protein into false substitutions. What that grid
cannot show is carried explicitly:

- `consequence` is 8-way: synonymous / missense / nonsense / **frameshift** /
  **inframe_indel** / **start_loss** / **stop_loss** / **unreadable**. An indel
  whose length is not a multiple of 3 is a frameshift;
- `truncation` is the fraction of the CDS after a frameshift, nonsense or
  start-loss;
- the `[NULL]` token carries the locus's **net frame offset** and the
  **fraction of the CDS read out of frame**, so a +1 and a −1 a few codons
  apart show as a restored frame with a garbled stretch between them;
- a deletion that crosses a locus boundary leaves one span token in each locus
  it touches, with `span` = the bases deleted there.

The token is 57 channels (`layout.py`), plus 2 if the population channels are
on and one per drug for site proximity.

### The coverage gap

The VCFs were filtered with `bcftools view --types snps,indels,mnps,other`, so
they hold variant sites only. **A position with no record is reference OR not
covered, and these files cannot tell the two apart.** This is not new — v3's
"uncalled" channel has always meant "low-quality variant call", not "no
coverage" — but it matters more genome-wide. Closing it needs the full Pilon
VCFs or a per-isolate low-coverage BED, which live on the Farhat lab's server.
Until then, "no token" means "matches H37Rv or was not covered".

## Store and batching

`build_tokens.py` runs as a SLURM array (`scripts/sbatch/genomefusion_tokens.sh`),
reading the VCFs straight from the read-only pi_annagreen tree — copying 35,884
files off that spinning disk measured at ~21 hours, while one read pass is ~0.25 s
per isolate.

`data.py` expands every token ONCE into a `(n_tokens, 57)` float16 matrix on the
device, so building a batch is a gather and a scatter with no Python loop. The
batch is **packed**, not dense: the median locus carries one token and the
busiest carries ~90, so a `(B, 6776, 90, 57)` tensor would be mostly padding.
Instead each (isolate, active locus) pair is a row of a flat pair list and the
model buckets pairs by token count, padding each bucket only to its own width.

## The model

`model.py`. Two transformer stages, as in v3–v5:

1. token embedding (linear → *d*), plus a sinusoid of the position within the
   locus, plus the locus embedding;
2. **within** each active locus: a `[NULL]` token, a shared 2-layer encoder with
   a per-locus FiLM, locus vector = the output at `[NULL]`;
3. **across** the active loci: a 2-layer encoder over the locus vectors plus one
   learned **reference token**;
4. one query per drug attends over the loci and the reference token, then an MLP.

**The reference token** is what makes the sparse form behave. A susceptible
isolate may have no relevant locus active, and without it the softmax would have
to put every drug query's weight on some lineage SNP.

**Locus identity is not a free embedding.** At 6,776 loci that would be 2.7M
parameters, mostly for loci that vary in a handful of isolates. The embedding
and the FiLM scale/shift are predicted from a 26-dim locus feature vector
(region kind, Mycobrowser category, DeJesus essentiality with an explicit "no
call" state, length, genome position, strand, pseudogene, WHO-19 membership),
plus a small learned residual with its own dropout.

## Running

From `biophysical-fusion/`, with `conda activate abr_env`.

```bash
python -m genomefusion.partition                       # build the partition cache
sbatch --array=0-15%16 scripts/sbatch/genomefusion_tokens.sh
sbatch --dependency=afterok:<jobid> scripts/sbatch/genomefusion_tokens.sh merge
python -m genomefusion.sbatch --arms canary --dry-run
python -m genomefusion.sbatch --arms canary
python -m genomefusion.sbatch --arms gf19 gf_genome gf_minus19
```

Arms: `gf19` (the v3/v5 loci — a **parity** arm, not a result), `gf_genome`,
`gf_minus19`. Protocol is the lineage-split v3/v5 protocol unchanged, so CV
macro AUCs pair with those over the same 55 folds.

## Tests

```bash
python -m genomefusion.tests.test_model        # synthetic, seconds
python -m genomefusion.tests.test_tokenizer    # real VCFs, ~1 min
```

`test_model.test_bucketing` is the one that matters: bucketed pairs must give
the same locus vectors as one pass at a single width.

## Not built yet

- **multi-drug** — deliberately deferred.
- **`flat` ablation** (no locus hierarchy, drug queries attend over tokens).
  This is what tells us whether the hierarchy pays for itself; most active loci
  hold a single token.
- **`pop` arm** (homoplasy count, lineage-marker flag). Homoplasy needs a
  phylogeny of the cohort, which we do not have; `--population` refuses rather
  than training on two columns of zeros.
- **site proximity** — the per-drug distance channels exist in the model
  (`n_dist`, `site_table`) but the genome-wide site index is not built, so runs
  carry no distance channel yet.
