"""
Regulatory-region modality: one-hot nucleotides for the promoter / upstream /
accessory loci that drive resistance beyond the primary coding target. One block
per region.

Per-drug regions come from the **WHO 2023 catalogue** (`who_catalogue.py`,
Tables 21 & 22). The set for a drug is its WHO Table-21 candidate genes; the
loaders keep the ones that are also loaded as coding loci and have a FASTA.

Each region's `{region}.fasta` is the per-isolate MSA of the **Table-22 promoter
window** (the upstream coordinates + 30 bp flank), extracted separately from the
BIG-TB VCFs into a dedicated dir (``paths.REAL_REGULATORY_DIR``). So a region
named e.g. ``embA`` here is the *embA promoter* (the embC–embA intergenic
window), NOT the embA CDS — it is disjoint from the coding sequence the DNA /
protein branches already model, keeping the modalities' information separate.

Only regions whose FASTA is present are loaded; the rest are skipped, and if a
drug has no available region the modality yields nothing and the loader drops it.
"""
from typing import List

from . import cds
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


# A gene whose promoter WHO delimits under ANOTHER gene's name — the two windows
# are literally the same base pairs. WHO's Table-22 `inhA` row is keyed to
# tss=1673440, which is fabG1's own CDS start (Rv1483 1673440-1674183; inhA
# starts 762 bp later), so that row IS the fabG1-inhA operon promoter carrying
# c-15t, filed under inhA. regulatory_msa emits the window under BOTH names so
# either can be requested on its own; the DEFAULT set keeps only the primary,
# because loading both would hand the model the same 873 bp twice.
REGION_ALIASES = {"fabG1": "inhA"}


def _drop_alias_duplicates(regions):
    """Remove an alias whose primary is already present (order preserved)."""
    keep = set(regions)
    return [r for r in regions
            if not (r in REGION_ALIASES and REGION_ALIASES[r] in keep)]


def _regulatory_default(drug):
    """Every WHO Table-21 candidate gene for the drug. Includes primary-gene
    promoters (e.g. katG/inhA for INH): the promoter window is upstream and
    disjoint from the CDS, so it is still information separate from the
    DNA/protein branches. Table 22 supplies a region's upstream coordinates
    (see _region_note), not its membership. Availability on disk is filtered
    at load time."""
    abbrev = DRUG_NAME_TO_WHO.get(drug.upper())
    if abbrev is None or abbrev not in who.TABLE_21:
        return []
    genes = [g for g, _rv in who.genes_for_drug(abbrev)]
    return _drop_alias_duplicates(list(dict.fromkeys(genes)))


# drug -> regulatory regions (WHO-derived)
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
            array = stack_padded([one_hot_nt(df.at[iso, region]).T for iso in ctx.isolates])
            blocks.append(FeatureBlock(
                name=f"regulatory:{region}",
                modality=self.name,
                array=array,
                channel_names=list(NT_CHANNELS),
                note=_region_note(region),
                # where this window sits in the genome (see cds.block_frame)
                meta=cds.block_frame(ctx.regulatory_dir, region, array.shape[2]),
            ))
        return blocks
