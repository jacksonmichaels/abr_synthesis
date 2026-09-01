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

import numpy as np

from . import tokens
from . import who_catalogue as who
from .base import FeatureBlock, LoadContext, Modality
from .sequences import (NT_CHANNELS, delta_one_hot_nt, load_sequence_df,
                        one_hot_nt, reference_row, stack_padded)

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
    DNA/protein branches. Availability on disk is filtered later, at load time.

    Table 22 supplies a region's upstream COORDINATES (see _region_note), not
    its membership — requiring an entry here used to silently drop tier-1 genes
    our Table-22 transcription lacks, notably fabG1 (Rv1483), whose promoter
    carries the dominant non-katG INH/ETO mechanism, and furA. Today that
    exclusion is masked by a second gap: neither gene has a promoter FASTA in
    REAL_REGULATORY_DIR, so both are skipped at load either way and this
    function's output is unchanged on the current data. It matters the moment
    one is curated — with the old filter, a freshly built fabG1 window would
    STILL have been ignored."""
    abbrev = DRUG_NAME_TO_WHO.get(drug.upper())
    if abbrev is None or abbrev not in who.TABLE_21:
        return []
    genes = [g for g, _rv in who.genes_for_drug(abbrev)]
    return _drop_alias_duplicates(list(dict.fromkeys(genes)))


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

    # see datasets/sequences.py — zero the columns matching H37Rv. Promoter
    # windows are the most redundant blocks in the set (regulatory:gyrA takes
    # 8 distinct forms across 2,868 isolates), so this is where it bites hardest.
    delta = False

    # One symbol id per column instead of the 5-channel one-hot; see
    # datasets/tokens.py and DNAModality.variant_tokens.
    variant_tokens = False

    def build(self, ctx: LoadContext) -> List[FeatureBlock]:
        regions = ctx.regulatory_loci
        if not regions:
            return []
        df, found = load_sequence_df(ctx.regulatory_dir, regions)
        if not found:
            return []
        df = df.reindex(ctx.isolates).fillna("")
        blocks = []

        if self.variant_tokens:
            for region in found:
                length = max(len(df.at[iso, region]) for iso in ctx.isolates) or 1
                ids = np.stack([tokens.nt_symbol_ids(df.at[iso, region], "reg", length)
                                for iso in ctx.isolates])[:, None, :]
                blocks.append(FeatureBlock(
                    name=f"regulatory:{region}",
                    modality=self.name,
                    array=ids,
                    channel_names=["symbol_id"],
                    note=_region_note(region) + " [symbol ids vs H37Rv]",
                    column_meta=tokens.region_column_meta(
                        ctx.regulatory_dir, region, length),
                ))
            return blocks

        for region in found:
            ref = reference_row(ctx.regulatory_dir, region) if self.delta else None
            if self.delta and not ref:
                print(f"  [load] delta encoding: no H37Rv row for regulatory "
                      f"{region} — stays plain one-hot")
            mats = [(delta_one_hot_nt(df.at[iso, region], ref) if self.delta
                     else one_hot_nt(df.at[iso, region])).T for iso in ctx.isolates]
            blocks.append(FeatureBlock(
                name=f"regulatory:{region}",
                modality=self.name,
                array=stack_padded(mats),
                channel_names=list(NT_CHANNELS),
                note=_region_note(region) + (" [delta vs H37Rv]" if self.delta else ""),
            ))
        return blocks
