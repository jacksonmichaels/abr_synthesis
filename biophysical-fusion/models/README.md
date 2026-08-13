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

## Choosing between them

| | mixes loci at | positional resolution | block count/order | params (19-locus DNA) |
|---|---|---|---|---|
| `late_fusion` | the flatten (one FC) | full | fixed, positional | ~35M |
| `mdcnn` | layer 1 (loci as channels) | full, but zero-padded to the group max | fixed, positional | ~3.9M |
| `setfusion` | transformer over tokens | 4 relative bins per locus | **free — keyed** | shared per modality |
| `cisfusion` | within a cis-unit, then the flatten | full | fixed, positional | per-locus branches |

`mdcnn`, `setfusion` and `cisfusion` all require per-locus blocks; `--arch`
implies that. `--encoders` applies only to `late_fusion` and `cisfusion`.

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
