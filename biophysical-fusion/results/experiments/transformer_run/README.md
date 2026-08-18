# transformer_run — the full_run_v2 grid with a transformer encoder, at matched size

Submitted **2026-08-18**, 165 SLURM jobs, reproducible via `submit.sh` in this
folder. Manifests: `slurm_logs/manifests/submitted_20260818_*.json` (three, one
per architecture — each carries its own `--tf-d-model`).

## The question

Every architecture in this project encodes a genomic block with a **local motif
detector** — a 1D CNN, or MD-CNN's conv stack over locus-as-channel input. Does
a **self-attention** encoder do better on the same inputs? Long-range coupling
between distant positions (epistasis between residue pairs, a promoter variant
interacting with a CDS variant) is the thing a fixed-width conv structurally
cannot see and attention can.

**`full_run_v2` is the matched control.** Same loci (per-drug `DRUG_TO_LOCI`),
same 300 epochs / patience 30 / 50-epoch warmup, same `--save-weights best`, same
seed. The only difference is which encoder each branch or trunk uses.

## What had to be built first

`--encoders` / `--default-encoder` already existed, but the transformer path was
not usable for this comparison:

1. **`--arch mdcnn` ignored the encoder choice entirely** and printed
   `encoders: n/a (mdcnn)`. Its topology is the conv stack. So a transformer
   mdcnn did not exist.
2. **The transformer encoder was not tunable at all** — `d_model=64, layers=2,
   dim_ff=128` were hardcoded, and no CLI flag reached them.

Both are fixed (see `../../CODE_CHANGES_20260818.md`); `tests/test_transformer_encoder.py`
covers the new path, 13/13.

### `MDCNNTransformerTrunk` — how mdcnn got a transformer

The point of MD-CNN over late fusion is that **layer 1 spans every locus and the
full channel height at once**, so loci mix before anything else. A transformer
version has to keep that or it is just late fusion with attention.

It turns out MD-CNN's layer 1 already *is* a patch embedding waiting to happen.
It is a `Conv2d(n_loci, 64, (channels, 12))` sliding along the position axis;
setting the stride equal to the kernel width along that axis turns the identical
op into a ViT-style tokenizer:

```
Conv2d(n_loci, d_model, (channels, patch), stride=(1, patch))
   -> (B, d_model, 1, ~L/patch)  ->  + learned pos  ->  TransformerEncoder  ->  mean-pool
```

So one op both mixes the loci exactly as the reference does and chunks the
sequence into tokens. Everything downstream of it — the `Conv1d/pool` stack and
the 14,336-wide flatten — is replaced by attention and a pooled summary. Grouping
by channel height, the zero-padding, and the dense head are untouched.

`tests/test_transformer_encoder.py` asserts the property that matters: perturbing
**any** locus plane moves the logits, i.e. the loci really are still being mixed.

The cost is the same one SetFusionNet pays, and it is a real loss: the flatten is
gone, so absolute position is coarsened. MDCNNTrunk keeps "this motif fired at
column 315"; this keeps "it fired in this ~9 bp token".

`--arch mdcnn` accepts only a **uniform** encoder choice, and raises on a
per-modality mix. Its trunks group by channel height and a trunk can span
modalities (`dna` and `regulatory` are both 5-channel), so "protein=transformer,
dna=cnn" has no well-defined meaning there. Use `--default-encoder`.

## Matching the parameter count — and why it cannot be exact

**A transformer branch is nowhere near a CNN branch at the defaults.**
`CNNEncoder` flattens, so its output width scales with sequence length
(`32 * L/9` — about 12,000 features on a 3.4 kb block, which alone puts ~3.1M
parameters in the dense head). `TransformerEncoder` mean-pools to exactly
`d_model`, whatever the length. At stock settings the two differ by roughly
**30x**, so "swap the encoder" without touching capacity would have compared a
5M-parameter model against a 150K one.

The shape is fixed at standard transformer proportions — `nhead 4`, `layers 4`,
`dim_ff = 4 * d_model`, `patch 9` (unchanged, so the token count matches the
CNN's pooling factor) — and **`d_model` alone is solved per architecture** to
match that architecture's own CNN median in `full_run_v2`:

| arch | CNN median | `d_model` | `dim_ff` | transformer median | median per-run ratio |
|---|---:|---:|---:|---:|---:|
| `late_fusion` | 4,450,561 | 208 | 832 | 4,482,977 | **1.05** |
| `cisfusion` | 4,517,377 | 160 | 640 | 4,145,825 | **1.00** |
| `mdcnn` | 2,491,329 | 176 | 704 | 3,226,737 | **0.99** |

Per-architecture rather than one global value, because the comparison being made
is CNN-vs-transformer *within* an architecture. A single `d_model` would have
mis-sized all three (the best global fit left a median mismatch of ~1.5x).

**The residual is structural. Read the per-cell ratios before quoting any cell.**

| cell | CNN median | transformer median | ratio |
|---|---:|---:|---:|
| `dna__late_fusion` | 3,262,721 | 2,296,273 | 0.70 |
| `dna_protein__late_fusion` | 4,190,913 | 4,496,705 | 1.07 |
| `dna_biophysical__late_fusion` | 4,189,825 | 4,464,881 | 1.07 |
| `dna_regulatory__late_fusion` | 4,656,897 | 4,480,481 | 0.96 |
| `all_modalities__late_fusion` | 6,610,497 | 8,852,017 | **1.34** |
| `dna__cisfusion` | 3,380,673 | 2,724,545 | 0.81 |
| `dna_protein__cisfusion` | 4,395,777 | 5,365,505 | 1.22 |
| `dna_biophysical__cisfusion` | 4,392,513 | 5,316,545 | 1.21 |
| `dna_regulatory__cisfusion` | 4,648,833 | 2,729,345 | **0.59** |
| `all_modalities__cisfusion` | 6,721,025 | 7,962,305 | 1.18 |
| `dna__mdcnn` | 2,360,769 | 1,671,953 | 0.71 |
| `dna_protein__mdcnn` | 3,089,025 | 3,295,553 | 1.07 |
| `dna_biophysical__mdcnn` | 3,062,913 | 3,237,121 | 1.06 |
| `dna_regulatory__mdcnn` | 2,368,449 | 1,690,433 | 0.71 |
| `all_modalities__mdcnn` | 3,713,857 | 4,878,497 | **1.31** |

The reason is a genuine difference in what each architecture charges for:
**CNN cost tracks total sequence length; transformer cost tracks block count.**
A DNA-only cell is one long block, so the transformer is *under*-sized there
(0.70); an all-modalities cell is four blocks each paying a full transformer, so
it is *over*-sized (1.34). No single config removes this, and per-cell tuning
would mean 15 different models and no clean comparison. `n_params` is recorded in
every result JSON — use it, and do not read a ±30% capacity difference as an
architecture effect.

Every result row also records the exact config under `transformer`, so a cell
always states the capacity it ran at.

## The grid — 3 architectures, not 4

`late_fusion`, `cisfusion` and `mdcnn` × 5 modality sets (`dna`, `dna_protein`,
`dna_biophysical`, `dna_regulatory`, `all_modalities`) = 15 cells × 11 drugs.

**`setfusion` is excluded on purpose.** Its fusion stage is *already* a
transformer over locus-keyed tokens — that is the architecture. Its per-block
encoder is a separate thing with its own capacity knobs (`--enc-width`,
`--enc-out-channels`, `--enc-depth`, swept by `../setfusion_scaling/`). Putting
it in this grid would either duplicate `full_run_v2` or move two axes at once.

## Resources, and the two risks

`--mem 64G --cpus 4 --gpus 1 --time 36:00:00 --constraint vram23` — raised from
`full_run_v2`'s single-drug 48G / 16 h.

- **Attention is O(n_tokens²).** The longest block here is ETHAMBUTOL's
  `embC+embA+embB` concatenation, ~10.1 kb → ~1,120 tokens at `patch 9`. That is
  the cell to watch for CUDA OOM. `vram23` keeps the 11 GiB cards out and the
  driver exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Cost is the least certain number here.** `full_run_v2`'s single-drug jobs
  were cheap (median ~0.1 h at 150 epochs), but a transformer epoch over ~1,000
  tokens is far more expensive than a conv one, and 36 h is a guess at the
  ceiling rather than a measurement. If the wide cells saturate it, the honest
  fix is a larger `--tf-patch` (fewer tokens) — which changes capacity, so it
  would need its own re-match rather than being slipped in.

## What to compare, when it lands

Against `../full_run_v2/`, cell for cell, on CV AUC — that is the like-for-like
pair. Three things worth asking, in order:

1. **Does attention help at all, anywhere?** Judge per cell, and only where the
   parameter ratio is near 1 (the `dna_protein` / `dna_biophysical` rows).
2. **Does it help most where the mechanism says it should?** The case for
   attention is long-range coupling, so PYRAZINAMIDE (`pncA` loss-of-function
   scattered across the gene) and the `dna_regulatory` / `all_modalities` cells
   (promoter interacting with CDS) are where a real effect should concentrate.
   RIFAMPICIN is already 0.977 on a tight positional signal in the *rpoB* RRDR —
   exactly the case a local conv already solves, so no headroom.
3. **Did mdcnn's lost flatten cost it?** `dna__mdcnn` is the direct read: same
   locus-as-channel input, same layer-1 locus mixing, conv-stack-plus-flatten
   versus attention-plus-pool.

Standing caveats: single seed, so treat sub-0.01 differences as unresolved; and
these are not locus-matched to `../alllocus_run_v2/`, which is a different axis
(19 loci, CNN).
