"""
Regulatory-region modality: one-hot nucleotides for the promoter / upstream /
accessory loci that drive resistance beyond the primary coding target. One block
per region.

Per-drug regions come from the **WHO 2023 catalogue** (`who_catalogue.py`,
Tables 21 & 22). The default set for a drug is its WHO Table-21 candidate genes
that also have a **Table-22 promoter entry** — i.e. every candidate for which
WHO delimits an upstream/promoter region. Genes with no Table-22 region are not
regulatory (WHO specifies no upstream window for them).

Each region's `{region}.fasta` is the per-isolate MSA of the **Table-22 promoter
window** (the upstream coordinates + 30 bp flank), extracted separately from the
BIG-TB VCFs into a dedicated dir (``bigtb_ref.REAL_REGULATORY_DIR``, built by
``regulatory_msa/{build_promoter_coords,submit_promoter_msa}.py``). So a region
named e.g. ``embA`` here is the *embA promoter* (the embC–embA intergenic
window), NOT the embA CDS — it is disjoint from the coding sequence the DNA /
protein branches already model, keeping the modalities' information separate.

Only regions whose FASTA is present are loaded; the rest are skipped, and if a
drug has no available region the modality yields nothing and the loader drops it.
"""
from typing import List

from . import who_catalogue as who
from .base import FeatureBlock, LoadContext, Modality
from .sequences import NT_CHANNELS, load_sequence_df, one_hot_nt, stack_padded

# project drug name (DRUG_TO_LOCI keys) -> WHO Table 21 abbreviation.
# Both fluoroquinolones map to WHO's combined 'FQ' row.
DRUG_NAME_TO_WHO = {
    "ISONIAZID": "INH", "RIFAMPICIN": "RIF", "ETHAMBUTOL": "EMB",
    "PYRAZINAMIDE": "PZA", "STREPTOMYCIN": "STM", "KANAMYCIN": "KAN",
    "AMIKACIN": "AMK", "CAPREOMYCIN": "CAP", "LEVOFLOXACIN": "FQ",
    "MOXIFLOXACIN": "FQ", "ETHIONAMIDE": "ETO",
}


def _regulatory_default(drug):
    """WHO Table-21 candidate genes for the drug that have a WHO Table-22
    promoter region — the loci for which the catalogue delimits an upstream
    window. Includes primary-gene promoters (e.g. katG/inhA promoters for INH):
    the promoter window is upstream and disjoint from the CDS, so it is still
    information separate from the DNA/protein branches. Availability on disk is
    filtered later, at load time."""
    abbrev = DRUG_NAME_TO_WHO.get(drug.upper())
    if abbrev is None or abbrev not in who.TABLE_21:
        return []
    genes = [g for g, _rv in who.genes_for_drug(abbrev)]
    return list(dict.fromkeys(g for g in genes if g in who.TABLE_22))


# drug -> default regulatory regions (WHO-derived). Override per call via
# load_dataset(..., regulatory_loci=[...]).
DRUG_TO_REGULATORY = {d: _regulatory_default(d) for d in DRUG_NAME_TO_WHO}


def _region_note(region):
    """Human-readable provenance for a region block, incl. WHO Table-22
    upstream extent + TSS when the catalogue lists one."""
    m = who.TABLE_22.get(region)
    if not m:
        return f"WHO candidate locus {region} (no Table 22 upstream coordinates)"
    seg = "+".join(f"{a}-{b}" for a, b in m["region"])
    tss = f", TSS {m['tss']}" if m["tss"] else ""
    return f"WHO Table 22 upstream region {region} ({seg}{tss})"


class RegulatoryModality(Modality):
    name = "regulatory"
    uses_genes = False

    def build(self, ctx: LoadContext) -> List[FeatureBlock]:
        regions = ctx.regulatory_loci
        if not regions:
            return []
        df, found = load_sequence_df(ctx.regulatory_dir, regions)
        if not found:
            return []
        df = df.reindex(ctx.isolates).fillna("")
        blocks = []
        for region in found:
            mats = [one_hot_nt(df.at[iso, region]).T for iso in ctx.isolates]
            blocks.append(FeatureBlock(
                name=f"regulatory:{region}",
                modality=self.name,
                array=stack_padded(mats),
                channel_names=list(NT_CHANNELS),
                note=_region_note(region),
            ))
        return blocks
