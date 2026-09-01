"""
Modality interface. Every modality is a small class that knows how to turn the
shared per-isolate sequences into one or more feature "blocks" (channels-first
arrays). The loader (loader.py) owns the shared work — resolving loci, loading
the aligned FASTAs once, aligning isolates to the phenotype — and hands each
modality a ready ``LoadContext``; the modality only describes its own
featurization.

A modality may emit several blocks (e.g. one biophysical block *per gene*,
matching Kulkarni et al. 2026's separate-channel-per-protein design). Each
block becomes one branch of the multi-modal model.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class LoadContext:
    """Everything a modality needs, prepared once by the loader."""
    drug: str
    loci: List[str]                       # gene loci for dna/protein/biophysical (resolved: user override or DRUG_TO_LOCI)
    regulatory_loci: List[str]            # regions for the regulatory modality (resolved: user override or DRUG_TO_REGULATORY)
    isolates: List[str]                   # shared isolate order (all blocks align to this)
    gene_seqs: Optional[pd.DataFrame]     # index=isolate, cols=gene, aligned nt (None if no gene modality requested)
    genotype_dir: str
    regulatory_dir: str


@dataclass
class FeatureBlock:
    """One model-ready input array plus enough metadata to describe it."""
    name: str                 # unique, e.g. "dna" or "biophysical:katG"
    modality: str             # "dna" | "protein" | "biophysical" | "regulatory"
    array: np.ndarray         # (N, C, L) float32
    channel_names: List[str] = field(default_factory=list)
    note: str = ""
    # Per-COLUMN constants, identical for every isolate, so they ride alongside
    # the array rather than inside it: {"coord", "phase", "ref_id"}, each (L,).
    # Only the variant-token loaders fill it (see datasets/tokens.py); every
    # other path leaves it None and nothing downstream looks at it.
    column_meta: Optional[dict] = None

    @property
    def channels(self) -> int:
        return self.array.shape[1]

    @property
    def length(self) -> int:
        return self.array.shape[2]

    def spec(self):
        """(in_channels, length) — what MultiModalNet needs to build a branch."""
        return self.channels, self.length


def merge_modality_blocks(mod_blocks, modality):
    """Concatenate a modality's per-locus/region blocks along the LENGTH axis
    into ONE block = one branch for that modality (e.g. the protein blocks for
    embC/embA/embB -> a single (N, 20, sum K) protein branch). Every sub-block
    shares the channel axis (true within a modality) and the isolate axis, so the
    concat is well-defined; loci stay in the caller's fixed order so gene k keeps
    the same ordinal slot across modalities. Alignment is ordinal, not positional
    (see datasets/multidrug.py's docstring)."""
    if len(mod_blocks) == 1:
        b = mod_blocks[0]
        return FeatureBlock(name=modality, modality=modality, array=b.array,
                            channel_names=b.channel_names, note=b.note)
    arr = np.concatenate([b.array for b in mod_blocks], axis=2)   # (N, C, sum L)
    units = "+".join(b.name.split(":")[-1] for b in mod_blocks)
    return FeatureBlock(
        name=modality, modality=modality, array=arr,
        channel_names=mod_blocks[0].channel_names,
        note=f"{modality}: {len(mod_blocks)} loci concatenated in order ({units})")


class Modality:
    """Base class. Subclasses set ``name`` and implement ``build``.

    ``uses_genes`` tells the loader whether this modality needs the shared
    per-gene FASTA frame loaded (DNA/protein/biophysical do; regulatory loads
    its own region FASTAs)."""
    name: str = "modality"
    uses_genes: bool = True

    def build(self, ctx: LoadContext) -> List[FeatureBlock]:
        raise NotImplementedError

    def __repr__(self):
        return f"<{type(self).__name__} name={self.name!r}>"
