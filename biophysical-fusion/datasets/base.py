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

    @property
    def channels(self) -> int:
        return self.array.shape[1]

    @property
    def length(self) -> int:
        return self.array.shape[2]

    def spec(self):
        """(in_channels, length) — what MultiModalNet needs to build a branch."""
        return self.channels, self.length


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
