"""
Dataset package: per-modality loaders (DNA, protein, biophysical, regulatory)
managed by a single dataloader (``load_dataset``). See loader.py.

Locus selection: ``load_dataset(drug, modalities, ..., loci=[...],
regulatory_loci=[...])`` controls which (and how many) loci each modality loads;
defaults are ``DRUG_TO_LOCI[drug]`` for the gene modalities and the WHO-derived
``DRUG_TO_REGULATORY[drug]`` for the regulatory modality. Two opt-in widenings:
``extra_loci=True`` adds the ``EXTRA_LOCI`` overlay (WHO Table 21 tier-1 genes
the per-drug map omits — fabG1 for INH/ETO), and ``loci_on_disk(dir)`` returns
every curated locus, which is MD-CNN's drug-independent rule.
"""
from . import who_catalogue
from .base import FeatureBlock, LoadContext, Modality
from .loader import (DRUG_TO_LOCI, EXTRA_LOCI, MODALITIES, DrugData,
                     drug_loci, load_dataset, loci_on_disk)
from .multidrug import (ALL_DRUGS, MultiDrugData, load_multidrug_dataset,
                        union_loci, union_regulatory)
from .regulatory import DRUG_NAME_TO_WHO, DRUG_TO_REGULATORY

__all__ = [
    "load_dataset", "MODALITIES", "DrugData", "DRUG_TO_LOCI",
    "EXTRA_LOCI", "drug_loci", "loci_on_disk",
    "DRUG_TO_REGULATORY", "DRUG_NAME_TO_WHO", "who_catalogue",
    "FeatureBlock", "LoadContext", "Modality",
    "load_multidrug_dataset", "MultiDrugData", "ALL_DRUGS",
    "union_loci", "union_regulatory",
]
