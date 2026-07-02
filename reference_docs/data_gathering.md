# MTB Region Extraction Guide (MycoBrowser + MycoExpress)

Goal: for each gene of interest, extract the **coding sequence + strand-correct flanking regulatory (intergenic) windows** as separate-but-adjacency-aware model inputs, and (optionally) attach expression evidence.

## Data sources
- **MycoBrowser** (`mycobrowser.epfl.ch`) — annotation + coordinates + sequence. Reference tool, not analysis.
- **MycoExpress** (`mycoexpress.wpi.edu`) — per-gene expression profile across a condition panel; each row = `(log2FC, p-value)`. Orthogonal expression evidence only.
- **H37Rv reference:** NCBI `NC_000962.3` — use for scriptable, reproducible sequence slicing (preferred over hand-copying from MycoBrowser).

## Per-gene extraction (do this from the reference, not by hand)
For each locus tag (e.g. `Rv0262c`):
1. Get from annotation: `start`, `end`, `strand`, and the two flanking loci.
2. Slice the **coding sequence** from `NC_000962.3` using `[start, end]` + strand.
3. Define **intergenic windows** as the gaps to the adjacent genes:
   - Upstream = promoter/5'UTR side; Downstream = 3' side.
   - Window size: **up to 350 bp, min 50 bp** each side (matches LLMTB approach).
4. Emit coding + upstream + downstream as **separate regions**, but **retain adjacency metadata** (which flank belongs to which gene, and orientation).

## ⚠️ Critical: strand handling (silent-corruption risk in batch)
- `c` suffix (e.g. `Rv0262c`) = **reverse strand**.
- For reverse-strand genes, the **upstream/promoter (5') region is at the HIGHER coordinate**, not the lower one. Downstream is at the lower coordinate.
- Reverse-complement reverse-strand sequences so all inputs are 5'→3' relative to the gene.
- Do NOT hardcode "lower coord = upstream" — branch on strand. This is the #1 batch bug.

## Boundaries: fixed-width vs real transcript ends
- **Default / first pass:** fixed-width flank (e.g. 350 bp) is sufficient and matches tracked literature.
- MycoBrowser ncRNA annotation is **incomplete** — do not rely on it to define true UTR/sRNA edges.
- If real boundaries are needed later, overlay published maps (out of scope for extraction, but leave hooks): TSS maps (Cortes 2013, Shell 2015), sRNA predictions (DeJesus 2017, Gerrick 2018).

## Optional: expression evidence (MycoExpress)
- Query per locus tag → returns condition rows `(log2FC, p-value)`.
- **Always gate on p-value** before trusting a fold change (panel contains many non-significant rows, p ≈ 0.9+).
- Use as: (a) interpretability check (does a resistance-relevant gene respond to the relevant drug?), and (b) justification for *which* regulatory windows to include rather than including all blindly.

## Output contract (suggested)
Per gene: `{ locus_tag, strand, coding_seq, upstream_seq, downstream_seq, upstream_len, downstream_len, adjacency: {upstream_neighbor, downstream_neighbor} }`, all sequences 5'→3' relative to the gene.