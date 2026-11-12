"""
Modality interface. Every modality is a small class that knows how to turn the
shared per-isolate sequences into one feature "block" per locus (channels-first
arrays). The loaders (loader.py, multidrug.py) own the shared work — resolving
loci, loading the aligned FASTAs once, aligning isolates to the phenotype — and
hand each modality a ready ``LoadContext``; the modality only describes its own
featurization.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class LoadContext:
    """Everything a modality needs, prepared once by the loader."""
    drug: str
    loci: List[str]                       # gene loci for dna/protein/biophysical
    regulatory_loci: List[str]            # regions for the regulatory modality
    isolates: List[str]                   # shared isolate order (all blocks align to this)
    gene_seqs: Optional[pd.DataFrame]     # index=isolate, cols=gene, aligned nt
    genotype_dir: str
    regulatory_dir: str


@dataclass
class FeatureBlock:
    """One model-ready input array plus enough metadata to describe it."""
    name: str                 # unique, e.g. "dna:katG" or "biophysical:katG"
    modality: str             # "dna" | "protein" | "biophysical" | "regulatory"
    array: np.ndarray         # (N, C, L)
    channel_names: List[str] = field(default_factory=list)
    note: str = ""
    # DNA and regulatory blocks carry cds.block_frame (where the block's columns
    # sit in the genome). Recorded in a checkpoint's config.json by
    # training.checkpoint.block_records.
    meta: dict = field(default_factory=dict)

    @property
    def channels(self) -> int:
        return self.array.shape[1]

    @property
    def length(self) -> int:
        return self.array.shape[2]

    def spec(self):
        """(in_channels, length)."""
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
