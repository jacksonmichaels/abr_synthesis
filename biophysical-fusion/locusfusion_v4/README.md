# LocusFusion v4

LocusFusion v4 keeps v3's architecture: per-locus variant tokens, a transformer
within each locus, a transformer across loci, and one learned query per drug.
It changes three things.

1. **Tokens.** The protein residue one-hot is gone. Each token keeps the isolate
   residue's biophysical values and their change from H37Rv, plus two new
   channels:
   - a **consequence** one-hot (synonymous / missense / nonsense / other);
   - the **distance to the drug's binding site**.
2. **Synonymous augmentation.** In training, isolates are copied with synthetic
   synonymous changes. The copy keeps the label, and a consistency term asks
   for the same prediction as the original.
3. **Offline tokens.** Tokens are built once from the aligned FASTAs and cached
   (~0.6 GB for 19 loci). Training never reads a FASTA. A 19-locus v3 job
   needed 128 GB; a v4 job needs a few GB.

It runs as a single-drug model (one drug, one proximity channel) or as a
multi-drug model (eleven drug queries and eleven proximity channels, one per
drug's site).

## The token

| channels | width | what |
|---|---|---|
| `dna` | 6 | A C T G - N: the isolate's base at a gene-record column |
| `regulatory` | 6 | A C T G - N: the isolate's base at a promoter column |
| `bio` | 3 | MW / pI / hydrophobicity (z-scored) of the residue the codon now reads; 0 for STOP, deleted or unreadable |
| `d_bio` | 3 | `bio` minus H37Rv's residue at that codon |
| `consequence` | 4 | synonymous / missense / nonsense / other (frameshift, in-CDS insertion, deleted or uncalled codon, stop-loss); all zero off a coding sequence |
| `phase` | 3 | position in the codon |
| `is_null` | 1 | the per-locus `[NULL]` summary token |
| `site_proximity:<drug>` | 1 per drug | exp(−d / 10 Å) to that drug's site, −1 = no distance |

The token is 27 channels for a single-drug model and 37 for the multi-drug
model. The layout is defined in `layout.py`.

**Shared with v3.** Wherever a channel exists in both versions, it is v3's
tokenizer exactly:
- placement in the genome, origin at the start codon, and promoter/record
  de-duplication come from v3's own `_frame_maps`;
- the N channel for uncalled bases;
- the 16-token cap, applied 5′→3′;
- residues read on the H37Rv codon grid.

`tests/test_parity.py` checks this against v3's `_locus_tokens` on 3,000 real
isolates.

## Binding-site distance

Distances are listed per drug and locus in `binding_sites/distances.json` and
built by `binding_sites/build_distances.py` from `binding_sites/sites.json`.

- **Protein sites.** Residues come from `reference_docs/mtb_drug_binding_sites_literature_review.pdf`,
  corrected as in `notebooks/locusfusion_v4/binding_site_distance.ipynb`: gyrB QRDR-B is
  shifted −39 to the 675-aa grid, and eis E32/F393 are dropped. The distance is
  the closest heavy atom to any site residue on the AlphaFold monomer.
- **Streptomycin, kanamycin, amikacin, capreomycin.** For rrs, rrl and rpsL, the
  distance is to the drug's own atoms in a drug-bound ribosome (4DR3, 8EV7,
  8EV6, 5V93).
- **Missing (−1).** Promoter tokens, loci without a site for that drug (for
  example rpoC for rifampicin), insertion columns outside a gene, and positions
  the structure does not resolve.

Spot checks (Å):

| position | value |
|---|---|
| rpoB S450 (RIF) | 0 |
| rrs 1401 | 3.3 for KAN, 13.3 for STR |
| rpsL K43 | 2.9 for STR, 7.5 for KAN |
| rrs 517 (STR) | 3.0 |

## Synonymous augmentation

```
loss = BCE(batch) + BCE(augmented copies, same labels)
     + λ · mean |logit(copy) − stopgrad(logit(original))|
```

- **Copies.** A fraction `--aug-frac` of each batch is copied. Each copy gets
  k ~ U{1..`--aug-k`} synthetic changes.
- **Drawing a change.** A coding locus is chosen uniformly, then a candidate
  within it. Gene length does not matter: pncA, with 418 candidates, is
  augmented as often as rpoB.
- **Never augmented:**
  - the start and stop codons;
  - a base another loaded locus also carries (neighbouring genes share record
    and promoter sequence);
  - the codons in `augment_exclusions.json` (fabG1 L203L, which creates an
    alternative inhA promoter);
  - a codon the isolate has already changed;
  - a locus with no free token slot.
- **The copy is a real-looking isolate.** The change has its true coordinate,
  phase and site distance, and the locus's changed-column count goes up by one.

There are 24,085 candidates over the 17 coding loci.

## Files

| file | what |
|---|---|
| `layout.py` | the token channels |
| `tokenizer.py` | aligned FASTAs → `TokenSet` (cached `.npz`) and synonymous candidates |
| `binding_sites/` | site definitions, distance builder, built `distances.npz` / `.json` |
| `augment_exclusions.json` | codons never augmented |
| `augment.py` | `SynonymousAugmenter`: inserts candidates into a batch on the GPU |
| `model.py` | `LocusFusionV4Net` and `proximity_table` |
| `train.py` | CV on a lineage split, test, robustness probes, result JSON |
| `make_multidrug_split.py` | one lineage split for the multi-drug model |
| `sbatch.py` | all experiment arms as SLURM jobs, run from a frozen snapshot |
| `tests/` | `test_model` (synthetic, seconds), `test_parity` (real data, ~2 min, ~11 GB) |
| `../scripts/build_tokenization_v4.py` | builds `notebooks/locusfusion_v4/tokenization_v4.ipynb`: v4 vs v3, both tokenizers on the same isolates; tokenizes only, never trains |
| `../scripts/build_locusfusion_v4_results.py` | builds `notebooks/locusfusion_v4/locusfusion_v4_results.ipynb`: the sweep's results, coverage table first; reads JSONs and `sacct` only |

## Running

Run these from `biophysical-fusion/` with `conda activate abr_env`.

```bash
python -m locusfusion_v4.tests.test_model             # synthetic tests
python -m locusfusion_v4.tests.test_parity            # needs ABR_* data env, ~11 GB
python -m locusfusion_v4.binding_sites.build_distances   # rebuild the distance table (seconds)
python -m locusfusion_v4.make_multidrug_split            # rebuild the multi-drug split

python scripts/build_tokenization_v4.py                   # the v4-vs-v3 notebook
python scripts/build_locusfusion_v4_results.py            # the results viewer
jupyter nbconvert --to notebook --execute --inplace notebooks/locusfusion_v4/locusfusion_v4_results.ipynb

python -m locusfusion_v4.sbatch --arms tokens canary --dry-run
python -m locusfusion_v4.sbatch --arms tokens                         # CPU, ~20 min
python -m locusfusion_v4.sbatch --arms canary --dependency afterok:<tokens job id>
```

Arms in `sbatch.py`:

| arm | what |
|---|---|
| `canary` | PZA with augmentation |
| `tune` | PZA, frac × λ grid |
| `noaug` / `aug` / `nodist` / `nosyn` | 11 drugs each, 19 loci, lineage split |
| `md_*` | the same four on the multi-drug model |

**Protocol.** The same as the lineage-split v3 cells: 300 epochs, patience 30,
min 50, Adam at exp(−9), early stopping on validation AUC (macro over drugs for
the multi-drug model), and the best fold's model scored once on TEST.

**Output.** Results go to `results/experiments/locusfusion_v4/<arm>/`:
- `<DRUG|MULTIDRUG>.json`;
- test predictions;
- the per-candidate synonymous scan.

Weights go to `<workspace>/model_weights_v4/`.

**Robustness probes.** These are under `"probe"` in each JSON:
- test isolates with 1 and 2 synthetic synonymous changes: mean |Δlogit| and
  mean ΔP(R) per drug;
- every synonymous candidate alone on an H37Rv background: P(R) per locus
  against the empty background. This is `scripts/pza_substitution_scan.py`
  generalized to every coding locus.

## Caveats

- **The multi-drug split is its own split.** It cannot match the per-drug
  lineage splits: an isolate's place differs between drugs.
  - It uses the same recipe, with Beijing as fold 4, and is balanced on each
    drug's resistant count.
  - TEST is 27% of isolates, not 20%, because sublineage groups go in whole.
  - Levofloxacin has only 66 resistant isolates in total.
- **Single- and multi-drug models see different promoter windows.** A
  single-drug model loads only its own drug's windows, as v3 did; the
  multi-drug model loads the union.
- **Not every site is a drug site.** The site list comes from one review that
  reads as partly machine-written. Two of its "sites" are not drug sites: gid's
  SAM pocket, and ethR's inhibitor tunnel.
- **Distances are within one chain.** No complexes are modelled, so rpoC has no
  distance to the rifampicin pocket and gyrA has none to gyrB's half of the
  fluoroquinolone site.
- **Augmentation assumes synonymous changes are neutral** everywhere except the
  excluded codons. A synonymous change that acts by a mechanism not in the
  exclusion list would be taught as neutral.
- **Data are those of the v3 runs.** The curated alignments still have the
  insertion-extraction bug at eis/embB/ethA/rrs, and the promoter windows are
  still the TSS-anchored ones. This keeps v4 vs v3 a matched comparison.
