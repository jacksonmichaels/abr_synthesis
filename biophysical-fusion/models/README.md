# Model designs

Every architecture here answers one question differently: **given a list of
feature blocks — one per (modality, locus) — how do you combine them into a
resistance call?** Same data, same `ConvBranch` feature extractor, same
`forward(xs)` contract (`bio_input = "blocks"`), so differences are attributable
to the fusion strategy and not to the convs.

---

## Shared building blocks

### `ConvBranch`
- 1D conv feature extractor: `stem 1x1 -> conv(k=12) -> maxpool(3) -> conv(k=3) -> conv(k=3) -> maxpool(3)`, ending at 32 channels.
- Deliberately mirrors BIG-TB's `ProteinCNN1x1` (`protein-tasks/one_hot_encoded/cnn_model.py`), split into extractor + head so the same branch is reused across variants instead of duplicated.
- `out_len(in_len)` traces a zero tensor through the stack — branch widths are computed, never hardcoded.

### `DenseHead`
- `fc1(in -> 256) -> fc2(256 -> 256) -> one logit per output`.
- `dropout=0` and `per_drug_hidden=0` by default, so the forward pass is byte-identical to the run behind `results/experiments/full_run`.
- `dropout` exists because the stack has **no** regularization otherwise — no dropout, no norm, no weight decay — and a joint `late_fusion` net is ~46M params fit on 17.9k isolates.
- `per_drug_hidden=k` gives each drug its own `256 -> k -> 1` branch off the shared trunk (~180k params at k=64, against 46M for the trunk). Only active when `out_dim > 1`; the default makes all 11 drugs read off **one** 256-d vector through **one** shared linear layer.
- `out_bias` initializes the output bias to the class log-odds, so `sigmoid(bias)` starts at the base rate. Calibration only — a constant bias shift cannot change AUC ranking.

### `ENCODERS` registry
- **`cnn`** (`CNNEncoder`) — `ConvBranch` flattened. Strong local motif detector; the default for DNA / regulatory / biophysical.
- **`transformer`** (`TransformerEncoder`) — strided conv chunks the sequence into ~L/9 tokens (ViT-style patch embedding, which is what keeps attention tractable on long genomic sequences), + learned positional embedding, small Transformer encoder, mean-pooled. Models long-range interactions (e.g. epistatic residue pairs) a local CNN misses — the candidate for the protein branch.
- Add an encoder by registering it here; nothing else changes. Only `late_fusion` and `cisfusion` consume the registry — the other architectures define their own encoding.

---

## `MultiModalNet` / `MultiDrugNet` — `--arch late_fusion`

- **Design:** one encoder per block, each flattened, all concatenated, one shared `DenseHead`.
- **Intuition:** every modality/locus is an independent evidence stream; let the dense head weigh them.
- **What it preserves:** full positional resolution. Nothing is pooled away — "column 315 of katG" survives to the flatten as a distinct feature index.
- **What it gives up:** no convolution ever mixes two loci. *All* cross-locus interaction is deferred to a single enormous FC layer (~35M params DNA-only, ~46M all-modalities, nearly all of it in `fc1`).
- **Per-block encoder choice** is the point of `encoder_types`: "DNA uses a CNN, protein uses a Transformer" is expressed as a per-block list. All-CNN reproduces the pre-encoder behavior exactly.
- **`MultiDrugNet`** is the same net with one sigmoid logit per drug (BIG-TB's MD-CNN head shape). Every drug is predicted from the same fused genotype representation, so loci that inform several drugs (`rrs` → KAN/AMK/CAP, `gyrA` → LFX/MFX) share features. Trained with masked multi-drug weighted BCE, so each isolate contributes only its non-missing labels. `drug_names` sets the output width and keeps columns self-describing (`logits[:, i] == drug_names[i]`).
- **Binding assumption:** blocks are bound to the model by **position** — branch *i* owns encoder *i*, and after the flatten each position is a fixed feature index. Block count and order are frozen at construction.

## `MDCNNNet` — `--arch mdcnn`

- **Design:** BIG-TB's own SD-CNN / MD-CNN topology, ported. Blocks are zero-padded to a common length and stacked into the **channel** axis; layer 1 is `Conv2d(n_loci, 64, (C, 12))` across all loci at once, then the 1D conv stack. All padding is `valid`, matching Keras.
- **Intuition:** the exact opposite bet from late fusion — loci should mix at **layer 1**, not at the flatten.
- **Why it is cheap:** 14,336-wide flatten and ~3.9M params on BIG-TB's 19-locus DNA input, against ~35M for `late_fusion` on the same input. The `(5, 12)` kernel spans the whole one-hot height so the height axis collapses to 1.
- **`n_drugs=1` reproduces SD-CNN; `n_drugs=len(drugs)` reproduces MD-CNN** — the two reference models are the same computation with a different head width.
- **Grouping caveat (measured, not hypothetical):** blocks are grouped by channel count by default, one trunk per group, because one kernel cannot span different channel heights. But `dna` and `regulatory` are *both* 5-channel (A,C,T,G,gap), so channel-only grouping puts all 16 promoter windows in the DNA trunk — where the position axis is the longest CDS (`rpoC`, 4,066 bp). An 87–2,060 bp promoter window becomes 95%+ zero padding *and* occupies a whole input channel of the layer-1 conv. That is why `dna_regulatory__mdcnn` scored **below** `dna__mdcnn` in `full_run`.
- **The fix:** pass `block_modalities` (or `from_blocks(..., trunk_per_modality=True)`) to group by **(modality, channels)**, giving promoters their own trunk padded to their own 2,060 bp maximum.
- **Requires per-locus blocks** (`per_modality_branch=False`) — pre-concatenated per-modality blocks defeat the entire point. Ignores `--encoders`.

## `SetFusionNet` — `--arch setfusion`

**The problem it exists to solve.** `MultiModalNet` and `MDCNNNet` both bind a
block by **position**. That silently assumes the block list is a fixed-length,
fixed-order vector — which is exactly where the regulatory modality breaks down.
A drug has e.g. 2 coding loci but 12 WHO promoter windows (only 2 sharing a gene
name; 10 with no CDS loaded at all), and both counts change per drug. Slot *k* of
the DNA branch and slot *k* of the regulatory branch are then unrelated genes.
`SetFusionNet` binds a block by **key** instead.

### How the keying actually works

- **The key is `(modality, locus)`,** parsed off `FeatureBlock.name` by `parse_block_key`:
  - `"protein:katG"` → `("protein", "katG")`
  - `"dna"` (a merged per-modality block) → `("dna", None)`, normalized to the `NO_LOCUS = "<none>"` sentinel.
  - `SetFusionNet.from_blocks(blocks)` reads keys straight off the loader's blocks, so you never hand-maintain the list.
- **The modality half of the key selects the encoder.** `self.encoders` is a `ModuleDict` with **one `SharedBlockEncoder` per modality**, not per block — all 12 regulatory windows go through the same promoter encoder; `inhA` (269 aa) and `katG` (432 aa) hit the same protein weights. Adding a 13th promoter window costs zero new parameters and needs no new branch.
  - Enforced invariant: every block of a modality must have the same channel count, or construction raises — one shared encoder cannot span two channel heights.
- **The locus half of the key is carried into the token as a learned vector.** Two embedding tables are built at construction:
  - `modality_emb : nn.Embedding(n_modalities, d_model)`
  - `locus_emb    : nn.Embedding(len(locus_vocab), d_model)`, where `locus_vocab[0]` is always `NO_LOCUS` and the rest are the distinct gene names, in first-seen order.
  - Both are `trunc_normal_(std=0.02)` init.
- **The keys are resolved to integer rows once** by `_key_ids`, cached as the non-persistent buffer `_default_ids` — an `(n_blocks, 2)` long tensor of `(modality_index, locus_index)`. Unknown modality or locus raises with the vocabulary it *was* built for, rather than silently indexing the wrong row.
- **The token is the sum:**
  ```
  token_i = SharedBlockEncoder[modality_i](x_i) + modality_emb[mod_id_i] + locus_emb[locus_id_i]
  ```
  This is the whole mechanism. **`dna:katG` and `regulatory:katG` are added the identical `locus_emb` row**, so they arrive at the fusion transformer carrying a shared, learnable identity vector. Attention can therefore match them *by content*, regardless of where either sits in the block list or how many blocks separate them. `modality_emb` is what keeps them distinguishable once matched — same gene, different evidence type.
- **Position in the list carries no information at all.** There is no positional encoding on the token sequence — deliberately. The transformer sees a genuine **set**, so it is permutation-invariant and count-agnostic; identity comes only from the key.
- **The key set is not frozen at construction.** `forward(xs, keys=...)` overrides the init-time list: you may pass a **subset**, a **reordering**, or a **repeat** of the blocks seen at init, as long as every key is in the vocabulary. That is what makes a per-drug-varying number of promoter windows expressible at all. Leave `keys=None` for the plain trainer contract.
- **Degenerate case is warned, not silent.** Build it with per-locus blocks (`per_modality_branch=False`). With the merged per-modality layout every block parses to locus `None`, they all share `locus_vocab[0]`, and the locus keying is a no-op — the constructor emits a warning saying exactly that.

### The rest of the pipeline

- **Fusion:** a `TransformerEncoder` (`norm_first=True`, 2 layers, 4 heads by default) over the `(B, n_blocks, d_model)` token set → contextualized tokens.
- **Read-out is one learned query per drug.** `drug_queries` is an `(n_drugs, d_model)` parameter; `pool_attn` cross-attends each query over the fused tokens, so `logits[:, j]` comes from **drug j attending over the locus tokens itself**, instead of every drug reading the same flattened concatenation. Then `LayerNorm -> fc1 -> ReLU -> dropout -> fc_out(hidden -> 1)`, shared across drugs.
- **It is directly interpretable.** `forward(xs, return_attn=True)` returns a `(B, n_drugs, n_blocks)` map of which locus each drug read.
- **No `per_drug_hidden` analogue** — the learned query per drug already *is* the per-drug capacity. `head_dropout` (not `dropout`, which is the transformer/attention dropout) is the knob the runners' `--dropout` maps to.

### The cost, stated plainly

- The shared encoder must be **length-agnostic**, so `SharedBlockEncoder` ends in adaptive pooling rather than a flatten: `out_features` depends only on `d_model`, never on L. That is the price of weight sharing and count-independence.
- Pooling is mean **and** max over `bins=4` equal segments — max answers "did this motif occur", mean "how much of the segment matches", and the segmentation retains coarse relative position a single global pool would destroy.
- Net effect: you keep *"this motif fired in the first quarter of katG"*; you lose *"at column 315"*. A real resolution loss traded for the keying.
- `ceil_mode=True` on both pools so a short block (a small promoter window) cannot collapse the position axis to zero the way a valid-padded stack would.
- **Training note:** it starts near-degenerate and sits at flat loss for ~12 epochs. `full_run`'s patience of 15 fired before it escaped, which is why that run's setfusion row (0.76–0.80) was an artifact rather than an architecture verdict. `full_run_v2` added `--min-epochs 50` warmup.

## `CisFusionNet` — `--arch cisfusion`

- **Design:** rebuilds the block list into one branch per **locus**, concatenating that locus's promoter and CDS along the **length** axis in transcription order (`regulatory` ⊕ `dna`), then runs ordinary per-branch encoders + `DenseHead`.
- **Intuition:** the same promoter/CDS correspondence SetFusion makes *learnable*, made **structural** — and by the cheapest possible route. A WHO Table-22 window is literally the DNA immediately upstream of its gene; `regulatory:katG` and `dna:katG` are neighbouring stretches of one chromosome that the block layout happened to tear apart. Glue them back and a single conv kernel near the junction spans both, with no new architecture at all. *"cis"* is the biology: a promoter acts only on the gene physically adjacent to it on the same DNA molecule.
- **The 6th channel is load-bearing.** Every nucleotide branch gets a segment marker appended — 0 promoter, 1 CDS. Without it the junction is invisible and the model cannot distinguish an upstream −15C>T from a synonymous change at the same offset into the CDS. Three column patterns stay mutually distinguishable: promoter (one-hot set, flag 0), CDS (one-hot set, flag 1), spacer/padding (all zero).
- **Three unit types, all expected:** *paired* (both present — `inhA`, `katG` for INH), *promoter-only* (a WHO window whose gene is not in `DRUG_TO_LOCI` — 10 of INH's 12: `ahpC`, `mshA`, `ndh`, efflux loci; flag all 0), *CDS-only* (a coding locus with no Table-22 window; flag all 1).
- **`spacer=N`** inserts N all-zero columns at the junction. The WHO window (upstream coords + 30 bp flank) is not guaranteed to end exactly at the start codon, so this asserts **order and adjacency**, not exact genomic contiguity — the zero columns are a "gap here" token, not a claim about its true width. Default 0 leaves them flush.
- **Only the two nucleotide modalities merge.** `dna` and `regulatory` share the (A,C,T,G,gap) alphabet, so the concatenation is over one consistent channel space. Protein (20ch) and biophysical (3ch) cannot be spliced onto a nucleotide axis and pass through as their own branches, unchanged.
- **The trade is the mirror image of SetFusion's.** Branches are positional again — branch count fixed at construction, encoders not shared — but **nothing is pooled away**, so full positional resolution survives to the flatten exactly as in `MultiModalNet`. Use `cisfusion` when the locus set is fixed (one drug, one model); use `setfusion` when it varies.
- **`cis_inputs(xs)`** returns the actual `[(name, tensor), ...]` branch inputs rather than burying the concatenation in `forward` — plot one and you can see the promoter, the junction and the CDS in a single matrix.
- Warns if no locus had both a regulatory *and* a dna block, since then it is just `MultiModalNet` with an extra channel.

---

## `LocusFusionNet` — `--arch locusfusion`

**One token per VARIANT, fused within a locus and then across loci.** The only
architecture here whose token count depends on the isolate rather than on the
sequence length, and the only one that fuses the modalities at the gene before
fusing the genes.

### The measurement it is built on

*M. tuberculosis* is clonal. Censused over all 19 curated loci x 17,943
isolates, each against the `MT_H37Rv` row in its own alignment:

| block | L | median diffs | p90 | p99 |
|---|---:|---:|---:|---:|
| dna:rpoB | 3,716 | 1 | 3 | 5 |
| dna:katG | 2,488 | 1 | 2 | 3 |
| dna:gyrA | 2,620 | 3 | 5 | 7 |
| dna:pncA | 1,067 | 0 | 1 | 5 |
| **all 19 loci** | ~40 kb | **14** | **26** | 214 |

**The median isolate differs from the reference at 0-3 columns of a 2.5 kb
gene.** A patch-embedding transformer over that input spends ~99.9% of its
tokens restating constants, which is exactly what `token_signal` measured from
the inside: 0.14% of a `setfusion` token varied with the genotype, attention was
flat at 1/8, and a linear probe on the encoder output beat the trained model.

So this one does not tokenize the sequence. It tokenizes the **difference from
the reference** — it runs on `--delta` input (implied by `--arch`) and emits one
token per surviving column. **100% of a token varies with the genotype by
construction.**

### A susceptible isolate is the empty set

Each locus carries a learned `[WT]` sentinel — the wild-type null — and every
variant token is evidence against it. A pan-susceptible isolate presents one
sentinel per locus and nothing else. That is the hypothesis "the model learns
what a sensitive strain looks like and calls everything else resistant" made
**structural** rather than accidental; it is also what guarantees no stage-1
attention row is ever fully masked.

### The pipeline

```
per locus L, per coordinate stream (nt / aa / reg):
    delta block (B, C, len)  --occupancy-->  the columns that differ
                             --gather-->     <= max_variants tokens
token   = tok_proj(modality slots + flags) + pos_proj(sinusoid(coord))
        + locus_emb[L] (+ FiLM adapter[L])
[WT]_L  = wt_emb[L] + wt_proj(variant count, coverage, uncovered)

stage 1   TransformerEncoder over {[WT]_L, variants of L}  ->  z_L = out[[WT]]
          KeyedTokenNorm(z_L, key=L)
stage 2   TransformerEncoder over {z_L : every locus}       ->  fused
readout   one learned query per drug, cross-attending the fused set
```

- **Stage 1 is the gene-level fusion.** All of *rpoB*'s evidence — CDS one-hot,
  translation, biophysical profile, promoter window — becomes one locus
  representation before any other gene is consulted. A resistance *mechanism* is
  a property of a gene; a resistance *phenotype* is a property of the set.
- **Protein and biophysical fuse into ONE token per changed residue.** They come
  off the identical translate path with the same `k_max`, so they are exactly
  co-indexed: the occupancy is their union, each writes into its own feature
  slot at the same column. One token says *"katG residue 315, now Threonine, and
  here is what that does to MW / pI / hydrophobicity"* — which is the form the
  *pncA* loss-of-function generalization needs (hundreds of distinct
  inactivating substitutions, most unseen in training).
- **The token feature vector is one fixed layout** (`SLOTS`), so a modality a
  run does not load simply leaves its slot zero and two runs' attention maps
  stay comparable. Derived flags are free and are the biology: `is_cds`,
  `is_promoter`, `gap` (a deletion vs a substitution), codon phase, `uncovered`.
- **Position is carried, not pooled.** A sinusoidal encoding of a *continuous*
  codon coordinate: nt columns land at k, k+1/3, k+2/3 so the fractional part is
  the codon phase, and the promoter sits at negative codons upstream. Parameter-
  free, where a learned table would need 1,355 x 128 x 19 = 3.3M weights.
- **`locus_encoder`** is the per-locus specialization knob: `shared` (one stage-1
  encoder, identity from `locus_emb` alone), `adapter` (default — the same
  encoder plus a per-locus FiLM, 2*d_model per locus, ~4.9k for 19 loci), or
  `per_locus` (a separate encoder each; 19x the stage-1 weights, and the loci can
  no longer run as one batched call).
- **`carry_variants=m`** hands each locus's first *m* variant tokens up to stage 2
  alongside its summary, so cross-locus attention can see individual variants
  (rpoB+rpoC compensatory pairs). Default 0 — summaries only.

### `summary_norm="keyed"` — on by default

A locus summary is read off the `[WT]` slot, whose input is **identical in every
isolate**. Measured on 400 real ISONIAZID isolates at init, using
`token_signal`'s own metric:

| stage-2 input | per-isolate spread / mean norm |
|---|---:|
| `summary_norm="none"` | 3.10% |
| `summary_norm="keyed"` | **9.17%** |

Left alone this is the same shape of failure `token_signal` found in setfusion —
a per-locus constant the readout absorbs into its intercept. `KeyedTokenNorm`
(shared with `SetFusionNet`, keyed by locus here rather than by (modality,
locus)) deletes it. That intervention took setfusion's ISONIAZID cell from
0.9287 to **0.9621**, which is why it is the default rather than a knob. Read
9.17% for what it is: a ratio at random **init**, not a claim about a trained
model. `"none"` creates no module and adds no `state_dict` keys.

### Uncovered is not wild type

Under delta encoding an all-gap record differs from the reference at *every*
column, so without a flag a gene that failed to assemble reads as the
most-mutated isolate in the cohort — 14-91 isolates per locus are in that state.
A locus whose occupancy exceeds `uncovered_frac` (0.5) sets `F_UNCOVERED` on all
of its tokens and reports it on the `[WT]` statistics. No other architecture
here distinguishes those cases at all. The `[WT]` token also carries
`log1p(true variant count)` *before* the cap, so a frameshift that changes 42
downstream residues is not silently read as 16.

### The approximation, stated plainly

`datasets/protein.py` DEGAPS each isolate before translating, so protein codon
*k* is the k-th codon of that isolate's own degapped CDS while the DNA block
stays in shared alignment-column space. The two agree exactly when no indel sits
upstream — the overwhelming majority of a clonal cohort — and drift by the indel
length when one does. **So nt and aa tokens are not fused by position.** They
stay separate tokens that within-locus attention may pair, and the constant part
of the offset (where the CDS starts inside the aligned record) is a *learned*
per-locus scalar rather than an assumption.

### What it gives up

- **Anything constant across the cohort**, which by definition carries no
  discriminative signal — but it is a real restriction, not a free lunch.
- **Sequence context** around a variant, beyond what the coordinate says. A conv
  sees the neighbouring bases; this sees the position and the substitution.
- **The tail.** `max_variants=16` per (locus, stream) covers >99% of (isolate,
  locus) pairs; overflow keeps the first 16 in positional order. The true count
  survives on the `[WT]` token, but which columns they were does not.
- It **requires `--delta`** and per-locus blocks. On dense input the tokenizer
  degenerates to "the first 16 columns of each block" while still producing
  plausible numbers, so `forward` warns on the first batch.

### Interpretability

`forward(..., return_attn=True)` gives `(B, n_drugs, n_stage2_tokens)` and
`variant_report(xs)` names the locus and alignment column behind every token, so
an attention map reads as *"MOXIFLOXACIN attended to gyrA column 281"* rather
than *"token 7"* — directly checkable against the WHO catalogue, with no SHAP
and no sampling error. That matters here: finding 3 in the top-level README is
that attribution share badly mis-ranks predictive value.

---

## `BranchedHead` — `--head branched` (a HEAD, not an `--arch`)

**Learn-to-Branch over drugs.** Port of the branching mechanism in Luo et al.,
*Sci Rep* 14:6631 (2024) — Branched CALM-Net — with **subject replaced by
drug**. It replaces `DenseHead`, so it composes with `late_fusion`, `mdcnn` and
`cisfusion` unchanged; `setfusion` and `locusfusion` build their own read-outs
and the runners refuse the combination rather than ignoring it.

### The axis it moves

Every architecture above varies how the **genotype** is encoded and then hands
the result to a head that shares everything between the drugs. `DenseHead` with
`out_dim=11` reads all eleven off ONE 256-d vector through ONE linear layer
(total sharing); a single-drug run trains 11 disjoint models (no sharing).
Neither is obviously right, because the drugs do not share uniformly —
KAN/AMK/CAP all target *rrs*, LFX/MOXI share *gyrA*/*gyrB*, INH/ETO share the
*fabG1–inhA* promoter, and *rpoB* informs only RIF. This is the first thing in
the project on the **task-sharing** axis rather than the representation axis.

### The measurement it is built on

Multi-drug minus single-drug-at-19-loci, paired over the 15 (modality × model)
cells that exist on both sides:

| | macro | per-drug spread | ρ(gain, n_isolates) |
|---|---:|---:|---:|
| Δ from multi-task sharing | +0.0012 | **0.028** | **−0.47** |

LEVOFLOXACIN **+0.0165** and ETHIONAMIDE **+0.0139** against KANAMYCIN
**−0.0119** and PYRAZINAMIDE **−0.0065**. Uniform sharing helps the small drugs,
dilutes the large ones, and cancels to nothing in the macro. That cancellation —
not an absence of transfer — is what routing targets.

### The pipeline

```
fused features -> fc1 -> h ─┬─> group node 1 ┐
                            ├─> group node 2 ├─ theta (Gumbel-softmax, annealed) -> per-drug MLP -> logit
                            ├─> group node G ┘
                            └─> generic head -> all n_drugs logits   (cold-start path)
```

- **`theta` is (n_drugs, G).** Each drug learns a categorical over the group
  nodes; drugs that end on the same node **are** the discovered cluster
  (`branch_assignments()` reports the partition, and it lands in the result JSON).
- **`tau` anneals 5.0 → 0.5 across the run**, stepped once per epoch by
  `training.core.anneal_branch_temperature`. Without the anneal theta never
  sharpens and the head is strictly more parameters doing DenseHead's job.
- **The generic head** is trained by `generic_weight * masked_weighted_bce` on
  every drug's labels (`training.core.branch_aux_loss`), and is what a drug with
  no per-drug parameters would predict from. It is the paper's Eq. 4 minus the
  reconstruction term — we do not port the LSTM autoencoder, which denoises
  irregular sensor data; `token_signal` measured that ~99.86% of an encoded block
  is constant across isolates, so a reconstruction objective would spend itself
  rebuilding that constant. `--delta` is the domain-appropriate version.

### `hard=True` is the default, and the reason is measured

The head as first written **did not learn**. On a synthetic task with known
ground truth (8 drugs, 4 loci; drugs 0–3 driven by locus 0 and 4–7 by locus 1,
so the true grouping is a 4/4 split), soft Gumbel mixing left `softmax(theta)`
at 0.44/0.56 after 60 epochs, mean `|∇theta| ≈ 5e-4`, and recovered an arbitrary
1/7 split. Every group node was receiving gradient from every drug, so the nodes
never specialised and the loss was flat in theta — the same *mechanism present
but inert* failure `token_signal` diagnosed in setfusion.

Straight-through routing (`gumbel_softmax(..., hard=True)`) sends the one-hot
forward and the soft gradient backward, so **a group node is updated only by the
drugs currently routed to it**. That is what breaks the symmetry.
`theta_init_std` breaks the tie at init and `theta_lr_mult` compensates for
theta's gradient being orders of magnitude smaller than the weights' (the
standard treatment for architecture parameters, as in DARTS).
`tests/test_branched.py` asserts the disjoint-gradient property directly, so the
inert version cannot come back silently.

### Cost, and the control that matters

Against `DenseHead` it drops `fc2` and `fc_out` and adds G group nodes, a
per-drug MLP each, theta, and the generic read-out — about **+378k parameters**
at G=4/hidden=256/k=64/11 drugs. That is ~10% of the joint `mdcnn` DNA model and
~1% of joint `late_fusion`, but ~70% of a toy net, so the *ratio* is not the
thing to quote.

**The control is `DenseHead(per_drug_hidden=64)`, not plain `DenseHead`.**
Branching also adds per-drug capacity, so beating the plain head would confound
routing with capacity — and `joint_capacity`'s `b2_perdrug64` already measured
that arm at −0.001, so the capacity half is known to be worth nothing alone.
`--branch-groups 1` is the same control inside the branched code path.

### It only means anything on the multi-drug task

With one output there are no tasks to group: `make_head` returns a plain
`DenseHead` and `--head branched` is a documented no-op on both single-drug
tasks. A "branched × 3 tasks" grid is really "branched × 1 task".

---

# Experimental family (`experimental_models.py`) — the *aggregation* question

Everything above answers "how do I read a genomic block". These six answer a
different one, and they exist because of what this project's own measurements
say the problem actually is.

## The reframing

Censused over 19 loci x 17,943 isolates against each alignment's `MT_H37Rv` row:
**the median isolate differs from the reference at 0-3 columns of a 2.5 kb gene,
and at 14 columns across all 19 loci.** Once you tokenize those differences —
which `LocusFusionNet` does — there is no long sequence left to encode. The task
becomes:

> given a set of ~14 deviations from wild type, most of them neutral lineage
> markers, decide R/S — from 11.5 k labelled training isolates.

That is **sparse evidence aggregation**, not sequence encoding. CNN, transformer
and SSM are all answers to the encoding question. The aggregator is what is left,
and softmax attention is a specifically bad choice for it:

> **Softmax normalises, so it is a RELATIVE selector.** With one informative
> token and thirteen neutral ones the weights must sum to 1, so it cannot say
> "this token, absolutely, whatever else is present" — it has to spend mass on
> the neutral tokens. That is the mechanism behind the flat 1/8 attention
> `token_signal` measured. It is a property of the operator, not a training
> failure.

A needle detector wants an **absolute, monotone** aggregator: adding a neutral
token must not dilute the signal. So: one tokenizer, six aggregators,
deliberately far apart.

| `--arch` | aggregator | the question it asks | params (INH cell) |
|---|---|---|---:|
| `catalogue` | learned scalar per exact variant id | how far does pure memorisation get? | 29 k |
| `additive` | `sum w(features_v)` | does featurising buy generalisation to unseen substitutions? | 48 k |
| `noisyor` | `1 - prod(1 - p_v)` | "susceptible unless something confers resistance", as an architecture | 48 k |
| `gatedpool` | sigmoid gate, no softmax | is normalisation the thing that broke attention here? | 98 k |
| `deepsets` | sum + max + count | does attention buy anything at all over plain additivity? | 114 k |
| `fm` | factorization machine | is epistasis worth anything, at O(T*k) instead of O(T^2)? | 16 k |

All six share `VariantSet` (the flat variant tokenizer) and `VariantEmbedding`,
and differ **only** in `aggregate()`. That is deliberate: it makes the six a
controlled comparison of aggregators rather than six unrelated models. The token
feature layout is imported from `locusfusion` rather than redefined, so a
contribution or gate here is directly comparable to a `locusfusion` attention
weight.

All six require `--delta` and per-locus blocks; `--arch` implies both.

## `catalogue` vs `additive` — a matched pair, and the measurement between them

`catalogue` learns one scalar per **exact** variant identity and sums them. That
is logistic regression on the isolate x variant presence matrix, expressed as a
module so it trains under the existing protocol — and it is a WHO-style
resistance catalogue, learned rather than curated. Its weight table is
zero-initialised, so **a substitution absent from training contributes exactly
nothing.** It can only recognise what it has already seen.

`additive` is the identical aggregator with the weight computed from the
variant's **features** — locus, exact position, which base or residue it became,
the biophysical consequence — so an unseen substitution still gets a weight.

The gap between them is a direct measurement of what featurisation buys, and it
is exactly the *pncA* / PYRAZINAMIDE mechanism: hundreds of distinct inactivating
substitutions, most unseen in training, where "does this substitution break the
protein" generalises and memorising positions cannot.

`additive` is also **its own attribution method**. `contributions(xs)` returns
the signed contribution of every token and they sum to the logit exactly — no
SHAP, no sampling error, no convergence question. That matters given finding 3
in the top-level README (attribution share badly mis-ranks predictive value).
Its restriction is that it forbids epistasis by construction; `fm` is the
cheapest way to relax that.

## `noisyor` — the biology as an architecture

`P(R) = 1 - (1 - p_0) * prod_v (1 - p_v)`. Resistance is conferred if **any**
variant confers it. A wild-type isolate has an empty product and falls back to
the learned background `p_0`, i.e. the base rate. Computed in log space
throughout, so nothing ever evaluates `1 - p` in float.

- **Absolute and monotone.** Each `p_v` is judged on its own; a neutral variant
  cannot dilute a resistance variant the way a softmax must.
- **It saturates.** Two resistance mutations give "resistant", not twice the
  logit — which is what an additive model gets wrong.
- `p_v` reads directly as "probability this variant confers resistance",
  comparable against the WHO catalogue's own confidence gradings.

Two structural restrictions, both worth stating before the run: it is monotone
increasing in evidence, so **it cannot learn a protective variant**; and it
assumes independent causes, so it cannot express compensatory epistasis.

## `gatedpool` — attention minus the softmax

The minimal edit to the thing that failed: keep `setfusion`'s per-drug gate,
delete the normalisation. The gate becomes an absolute relevance score rather
than a share of a fixed budget. A max branch runs alongside the gated sum — "did
any single variant fire hard" is the needle question, "how much evidence is
there in total" is not, and they disagree exactly when an isolate carries several
weak variants.

Run it against `locusfusion` to attribute that model's result: if `gatedpool`
closes a gap `deepsets` does not, normalisation was the problem; if `deepsets`
matches it, the gating was never doing anything.

## `deepsets` — the honest null model

`rho([sum phi(h_v), max phi(h_v), count])`. No gate, no query, no
normalisation. This is the ablation the project has never run: **does attention
buy anything over plain permutation-invariant aggregation?** `setfusion` and
`locusfusion` spend most of their parameters deciding *which* tokens matter; if
a sum and a max do as well, that machinery is not earning its place — and that
finding is worth more than a fourth-decimal AUC gain would be.

The variant count is fed in separately, from the TRUE count before the tokenizer
cap, because "how many deviations does this isolate carry" is a real feature
(lineage divergence, assembly quality) that a pure sum conflates with effect size.

## `fm` — pairwise interactions, priced linearly

`bias + sum_v w_v + W2 . [0.5 * ((sum e_v)^2 - sum e_v^2)]`. The classical
factorization-machine trick: the second term is **every pairwise interaction**,
computed in `O(T * rank)` instead of `O(T^2)`. So it asks "is epistasis worth
anything?" without attention's quadratic cost or its normalisation problem.

Compensatory resistance is real — *rpoC* mutations restoring fitness lost to
*rpoB* RRDR mutations is the textbook case, and both loci are among the 19 — but
finding 1 says the signal is dominated by a handful of specific positions, so
interactions should be second-order and an FM prices them accordingly. Factors
come from the variant's features, not a per-identity table (which would be ~4 M
parameters at 19 loci, almost all never trained), so unseen substitutions
participate in interactions too.

## The baseline that is not a network — run this first

`variant_design_matrix()` builds the sparse isolate x variant matrix these models
tokenize, with column names like `dna:katG@944=2`:

```python
X, names = variant_design_matrix(arrays, block_keys, branch_specs)
LogisticRegression(penalty="l1", solver="liblinear", C=0.1).fit(X[tr], y[tr])
```

There is **no sparse-linear or tree baseline anywhere in this project**, and for
TB AMR from a variant matrix those are the canonical strong methods —
TB-Profiler and Mykrobe are catalogue lookups and are clinically competitive.
`token_signal` already found plain logistic regression on setfusion's own
representations beating the trained model by 0.011, which points straight here.
**If this reaches ~0.92 it reframes the project**, and that is worth more than
any architecture above.

## The knob guard

`EXPERIMENTAL_DEFAULTS` is the union over the family, so callers pass it
straight through and a member ignores what is not its own. What is *not* ignored
is a knob moved **off its default** that belongs to a different member — that
raises, naming the model it belongs to. The rule is *changed-and-foreign*, not
merely *foreign*, for the same reason the setfusion and locusfusion flag groups
are guarded: an arm that quietly ran as its own control is worse than a crashed
job.

---

## Choosing between them

These five vary the **representation**. `BranchedHead` (above) varies the
**task sharing** and is orthogonal to all of them — pick a row, then decide
separately whether the drugs share one head or route to group nodes.

| | mixes modalities at | mixes loci at | positional resolution | input scales with | params |
|---|---|---|---|---|---|
| `late_fusion` | the flatten | the flatten (one FC) | full | sequence length | ~35M |
| `mdcnn` | one trunk per channel height | layer 1 (loci as channels) | full, zero-padded to the group max | sequence length | ~3.9M |
| `setfusion` | one token per block | transformer over tokens | 4 relative bins per locus | sequence length | ~0.46M |
| `cisfusion` | within a cis-unit | the flatten | full | sequence length | per-locus branches |
| `locusfusion` | **within a locus (stage 1)** | **transformer over locus summaries (stage 2)** | **full, per variant** | **variant count** | ~0.65M |

Plus the six experimental aggregators above (`catalogue`, `additive`,
`noisyor`, `gatedpool`, `deepsets`, `fm`), which share `locusfusion`'s variant
tokenizer and vary only the aggregation:

| | mixes modalities at | mixes loci at | aggregator | input scales with | params |
|---|---|---|---|---|---|
| `catalogue` … `fm` | one flat variant set | the same flat set | sum / noisy-OR / gate / set / FM | **variant count** | 16 k – 114 k |

Everything except `late_fusion` requires per-locus blocks and `--arch` implies
that; `locusfusion` and all six experimental archs additionally imply `--delta`.
`--encoders` applies only to `late_fusion` and `cisfusion`.

The last row is the one structural difference: every other architecture's input
is O(sequence length) and ~99.9% constant across a clonal cohort;
`locusfusion`'s is O(variants), median 14 tokens per isolate.

---

## Legacy (`legacy.py`) — importable, not in the training path

These predate the block-list interface and take the old two-argument
`forward(dna_x, bio_xs)`; the `bio_input` class attribute records which
biophysical layout each expects. They share `ConvBranch` / `DenseHead` with the
live models, so a revived variant trains with the same building blocks.

- **`DNAOnlyCNN`** — DNA-only baseline, no biophysical branch. BIG-TB's SD-CNN in spirit and the original control the fusion models were measured against. Current equivalent: `run_experiment.py --modalities dna`, which runs the corrected protocol (missing-phenotype filter, stratified split, train-only alpha).
- **`LateFusionCNN`** — matches Kulkarni et al. 2026's model design and training Methods exactly: DNA branch + one biophysical branch **per gene**, each pooled and flattened independently, then concatenated before the head.
- **`EarlyFusionCNN`** — the ablation for *"does fusing early beat fusing late?"*. Per-gene biophysical arrays are upsampled to nucleotide resolution (`datasets.biochem.upsample_to_nt`) and stacked onto the one-hot **channels** before a single **shared** conv stack. Not from the paper.
- **`CrossAttentionFusionCNN`** — `experimental_plan.pdf`'s "Asymmetrical Cross-attention": the DNA feature map is the attention **Query**; the per-gene biophysical feature maps are concatenated along the sequence axis into one **Key/Value** set. The idea is to let the CNN's own read of the DNA pull in biophysical context *only where attention says it's useful*, instead of always concatenating it. Still an open experiment.
