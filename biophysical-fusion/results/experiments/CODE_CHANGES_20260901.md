# The locusfusion tokenizer, rebuilt. 2026-09-01

Two changes, both to how a variant becomes a token. `locusfusion`'s two-stage
structure, its `[WT]` sentinel, its keyed summary norm and its read-out are all
untouched; so is every other architecture. The aggregator family in
`models/experimental_models.py` is deliberately left on the old layout — its
`variant_aggregators_*` numbers were measured on it, and porting them is
separate work with its own control run.

The run this produces is `locusfusion_v2` (55 jobs; `submit.sh` in that folder).

---

## 1. The coordinate was wrong, and the parameter meant to fix it never moved

`locusfusion` placed a nucleotide token at `column / 3 − coord_offset[locus]`
and a protein token at its codon index, with `coord_offset` a **learned**
per-locus scalar. Both halves of that are broken, for reasons that are
arithmetic rather than arguable.

**The aligned FASTAs are not bare CDS.** Measured over the 19 curated loci:

| locus | record | CDS start col | = codons | ref gaps inside the CDS |
|---|---:|---:|---:|---:|
| katG | 2,488 | 111 | 37.0 | 53 |
| rpoB | 3,716 | 101 | 33.7 | 96 |
| pncA | 1,067 | 102 | 34.0 | **304** |
| gid | 1,052 | 100 | 33.3 | 177 |
| ethA | 1,737 | 0 | 0.0 | 167 |
| rpsL | 588 | 112 | 37.3 | 1 |
| gyrA | 2,620 | 0 | 0.0 | 3 |

The CDS starts at column 100–112 in 10 of the 17 coding loci, so `column / 3` is
33–37 codons past where the protein stream puts the same residue before anything
else happens. Then the reference row's own gaps inside the CDS window bend the
map further, and **non-linearly** — pncA's 304 gap columns mean no single scalar
can be right at both ends of the gene.

What that cost, at the canonical resistance codons:

| variant | alignment column | DNA-token coord | protein-token coord | error |
|---|---:|---:|---:|---:|
| katG S315 | 1,072 | 357.3 | 314 | **+43.3 codons** |
| rpoB S450 | 1,505 | 501.7 | 449 | **+52.7** |
| pncA S65 | 380 | 126.7 | 64 | **+62.7** |
| gyrA D94 | 279 | 93.0 | 93 | 0.0 |

And the learned scalar did not absorb it. Read off `fold3.pt` of the completed
`newmodels_full/sd_all_modalities__locusfusion/ISONIAZID` run:

```
coord_offset = [-0.0107, +0.0081]        # (inhA, katG)
```

against a true katG offset of 37. It initialises at zero and reaches the loss
through a sinusoid whose top wavelength was 6.3 codons, so its gradient is
oscillatory over the distance it needed to travel; ~80 epochs at lr 1.2e-4
moved it by 0.01 codons. It was never going to work, and the model it was
supposed to serve spent every one of its runs reading nucleotide and protein
evidence for the same residue as if they were 43 codons apart.

**The fix** (`datasets/tokens.py`) computes it. Walk the reference row, count
degapped reference bases from the CDS start: `coord = n / 3`, `phase = n % 3`,
negative upstream. That is the H37Rv codon number — the coordinate the WHO
catalogue names a mutation by — it costs zero parameters, and it is exact.
Verified on the real alignments, both streams:

```
locus    res  aa    col  nt coord  aa coord    err phase
katG     315   S   1072    314.00    314.00  +0.00     0
rpoB     450   S   1505    449.00    449.00  +0.00     0
pncA      65   S    380     64.00     64.00  +0.00     0
gyrA      94   D    279     93.00     93.00  +0.00     0
embB     306   M    924    305.00    305.00  +0.00     0
```

The same bug took the **codon phase** with it: `F_PHASE` was `one_hot(column % 3)`,
which is the codon position only when the CDS starts at column 0. With starts at
101/102/111/112 it was a per-locus rotation that an additive `locus_emb` into a
shared `tok_proj` cannot undo, so "third codon position" — the one that is
usually synonymous — was not recoverable at all. It now comes off the same map.

`coord_offset` is deleted.

### What this does NOT fix

Protein codon *k* is still the k-th codon of the **isolate's own degapped** CDS
(`datasets/protein.py` degaps before translating), so it equals reference codon
*k* only when no indel sits upstream. That holds for the overwhelming majority
of a clonal cohort and is a property of the protein featurizer, not of the
tokenizer. nt and aa therefore remain separate tokens that within-locus
attention may pair — they now merely agree about where they are.

## 2. The token is a symbol, not a 42-float slot vector

The old token was a fixed 42-wide float vector. Its content, audited:

| slot | width | verdict |
|---|---:|---|
| `dna` one-hot | 5 | the alt base |
| `regulatory` one-hot | 5 | the alt base — and it can **never** be set at the same time as `dna`, because a token belongs to exactly one coordinate stream |
| `protein` one-hot | 20 | the alt residue |
| `biophysical` | 3 | kept |
| `F_IS_NT/AA/REG` | 3 | derivable from which slot is occupied |
| `F_IS_WT` | 1 | derivable |
| `F_GAP` | 1 | a **literal copy** of the one-hot's `-` channel |
| `F_UNCOVERED` | 1 | a per-LOCUS constant broadcast onto every token |
| `F_PHASE` | 3 | wrong, see above |

Now: a token is `alt` and `ref` symbol ids over one 35-entry vocabulary, a codon
`phase`, and a `coord`. **Per-token input goes from 43 numbers to 2** (plus 3
floats where biophysical is loaded). Embedded as

```
alt_emb[alt] + ref_emb[ref] + phase_emb[phase] + pos_proj(sinusoid(coord))
+ locus_emb[locus]  (+ bio_proj(properties))  (+ uncovered_emb)
```

Stream identity is implied by which range the id falls in, which is strictly
more information than the three flags at zero cost. `pos_dims` drops 64 → 32
because the sinusoid band is now fitted to the data — `[1/3, 4096]` codons
instead of `[2π, 2π·10⁴]`, which spent its bottom half on wavelengths longer
than any locus and never resolved a codon phase at the top.

**One thing was added rather than removed: the reference symbol.** A variant has
two ends and the old layout carried only the alt, so a C>T and a G>T at the same
column were the same token. Given `(locus, column)` the reference is a constant
the model could memorize, but there is no reason to make it.

Parameter accounting at `d_model=128` is a wash — 646,273 against 645,763. The
token shrank; `d_model` sets the model size, and it did not change. That was
deliberate: the point of this run is to attribute an AUC change to the tokenizer.

### The N-call, which used to read as wild type

`one_hot_nt` and `one_hot_aa` map an unknown symbol to the **all-zero column**,
and occupancy was `x.abs().sum(1) > 0`. Under delta encoding a matching column
is also all-zero. So an `N` where the reference has a base, and a residue past a
premature stop, were **indistinguishable from a match** — a failed base call read
as wild type. Occupancy is now `alt != ref` over the ids, which cannot confuse
the two. Visible immediately in the real data: katG's amino-acid stream has a
median of 1 variant and a **max of 104** across 400 ISONIAZID isolates, and the
104 is a truncation the old tokenizer could not see at all.

## 3. What the tokenizer now sees, on real data

400 ISONIAZID isolates, `all_modalities`, no warnings, `n_params = 646,273`:

| locus | stream | median | p90 | p99 | max |
|---|---|---:|---:|---:|---:|
| inhA | nt | 0 | 0 | 1 | 1 |
| inhA | aa | 0 | 0 | 1 | 1 |
| katG | nt | 1 | 2 | 3 | 3 |
| katG | aa | 1 | 2 | 2 | 104 |
| katG | reg | 0 | 0 | 2 | 2 |

which reproduces the variant census in `README.md` exactly (dna:katG median 1,
p90 2, p99 3). The most common katG nucleotide tokens, by coordinate:

```
codon  462.33   x171     katG R463L, the lineage marker
codon  314.33   x106     katG S315T, the dominant INH resistance mutation
codon  766.00    x41
codon  -28.33    x20     upstream of the CDS
```

`314.33` is codon 315 at the second base — which is exactly what S315T (AGC→ACC)
is. The tokenizer is landing on the right biology at the right coordinate, and
`variant_report()` now returns `ref`/`alt` names alongside it, so an attention
map reads as "drug j attended to katG codon 314, Ser→Thr" and joins directly
against `datasets/who_catalogue.py`.

## 4. Plumbing

| file | change |
|---|---|
| `datasets/tokens.py` | **new** — the 35-symbol vocabulary, `nt_symbol_ids` / `aa_symbol_ids`, and the exact per-column `{coord, phase, ref_id}` maps |
| `datasets/base.py` | `FeatureBlock.column_meta` (optional, defaults `None`) |
| `datasets/{dna,protein,regulatory}.py` | a `variant_tokens` mode emitting `(N, 1, L)` int8 symbol ids plus `column_meta` |
| `datasets/biophysical.py` | untouched, on purpose — see below |
| `datasets/{loader,multidrug}.py` | `variant_tokens=` plumbed through both loaders |
| `models/variant_tokens.py` | **new** — the shared `sinusoid` / `select_variants`, and the LEGACY `SLOTS`/`C_TOK`/`F_*` layout, pinned for `experimental_models.py` along with `sinusoid_legacy` so its runs stay reproducible |
| `models/locusfusion.py` | the tokenizer and embedding above; `coord_offset` deleted; `_embed` extracted; a one-hot block set is now **refused** rather than silently read as "the first 16 columns" |
| `models/__init__.py` | `VARIANT_TOKEN_ARCHS` |
| `scripts/run_{experiment,multidrug}.py` | `variant_tokens` implied by `--arch locusfusion`, recorded in the result JSON |
| `tests/test_locusfusion.py` | rewritten, 25 → **36** checks, including the real-alignment coordinate check |

**Biophysical has no symbol-id form, deliberately.** It is the modality whose
whole claim is that three properties stand in for the residue identity, so
handing it that identity would answer its own ablation. In a `dna+biophysical`
cell the amino-acid stream therefore carries properties and coordinates but no
symbol, and falls back to "a residue whose properties moved" — which means it
keeps the old blindness to an unchanged-property substitution. That is the
honest version of that cell, not an oversight.

`locusfusion` stays in `DELTA_ARCHS` for the same reason: biophysical still
needs delta encoding for its occupancy to mean anything.

The `{coord, phase, ref_id}` buffers are **persistent**. `build_model_from_config`
has only keys and specs, so a checkpoint rebuilt for attribution would otherwise
come back with the fallback map and quietly move every token; a test pins this.

## 5. Status

* `tests/test_locusfusion.py` **36/36**.
* `test_baseline_alignment` 8/8, `test_branched` 25/25, `test_checkpoint` 34/34,
  `test_cisfusion` 16/16, `test_experimental_models` 25/25,
  `test_transformer_encoder` 13/13. `test_setfusion` 21/22 — the pre-existing
  "defaults are the full_run configuration" failure recorded in
  `CODE_CHANGES_20260825.md` §6, unchanged by this work.
* Synthetic end-to-end runs pass on both runners.
* `scripts/locusfusion_diagram.py` still draws the OLD token vector and is not
  yet updated; it now fails loudly on the channel-count guard rather than
  drawing something false.
