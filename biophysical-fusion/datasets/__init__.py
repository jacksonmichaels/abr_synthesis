"""
Dataset package: per-modality loaders (DNA, protein, biophysical, regulatory)
managed by a single dataloader (``load_dataset``). See loader.py.

Locus selection: ``load_dataset(drug, modalities, ..., loci=[...],
regulatory_loci=[...])`` controls which (and how many) loci each modality loads;
defaults are ``DRUG_TO_LOCI[drug]`` for the gene modalities and the WHO-derived
``DRUG_TO_REGULATORY[drug]`` for the regulatory modality.
"""
from . import who_catalogue
from .base import FeatureBlock, LoadContext, Modality
from .loader import DRUG_TO_LOCI, MODALITIES, DrugData, load_dataset
from .regulatory import DRUG_NAME_TO_WHO, DRUG_TO_REGULATORY

__all__ = [
    "load_dataset", "MODALITIES", "DrugData", "DRUG_TO_LOCI",
    "DRUG_TO_REGULATORY", "DRUG_NAME_TO_WHO", "who_catalogue",
    "FeatureBlock", "LoadContext", "Modality",
]
