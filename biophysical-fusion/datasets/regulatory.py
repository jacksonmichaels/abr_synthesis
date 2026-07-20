"""
Regulatory-region modality: one-hot nucleotides for the promoter / upstream /
accessory loci that drive resistance beyond the primary coding target. One block
per region.

Per-drug regions come from the **WHO 2023 catalogue** (`who_catalogue.py`,
Tables 21 & 22). The default set for a drug is its WHO Table-21 candidate genes
**minus its primary coding loci** (`DRUG_TO_LOCI`) — i.e. the extra candidate
loci that carry the promoter/upstream signal. This reproduces the previous
hand-picked default (fabG1 for isoniazid & ethionamide — the fabG1–inhA operon
promoter) and extends it from the catalogue (e.g. eis for kanamycin — the eis
promoter). Where WHO Table 22 gives an explicit upstream region + TSS for a
region, that is attached to the block as metadata.

Only regions whose aligned `{region}.fasta` is present are loaded; the rest are
skipped, and if a drug has no available region the modality yields nothing and
the loader drops it. Note: this loads the aligned *locus* as the regulatory
input (as the previous fabG1 default did); true promoter-coordinate slicing
using the stored Table-22 region/TSS is a future refinement.
"""
from typing import List

from bigtb_ref import tb
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
    """WHO Table-21 candidate genes for the drug, excluding its primary coding
    loci — the extra candidate/regulatory loci for that drug. Availability on
    disk is filtered later, at load time."""
    abbrev = DRUG_NAME_TO_WHO.get(drug.upper())
    if abbrev is None or abbrev not in who.TABLE_21:
        return []
    coding = set(tb.DRUG_TO_LOCI.get(drug.upper(), []))
    genes = [g for g, _rv in who.genes_for_drug(abbrev)]
    return list(dict.fromkeys(g for g in genes if g not in coding))


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
