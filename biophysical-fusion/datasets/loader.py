"""
The single dataloader. Given a drug and a set of modality names, it does the
shared work once (resolve loci, load the aligned FASTAs, align isolates to the
phenotype), runs each requested modality, and returns a ``DrugData`` bundle of
model-ready feature blocks plus labels and metadata.

    from datasets import load_dataset, MODALITIES
    data = load_dataset("ISONIAZID", ["dna", "biophysical"],
                        genotype_dir, phenotype_csv)
    data.branch_specs()     # [(5, L), (3, K_inhA), (3, K_katG)]
    data.arrays()           # aligned (N, C, L) arrays, one per branch

Adding a modality is a one-line entry in MODALITIES; nothing else changes.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from bigtb_ref import tb
from .base import FeatureBlock, LoadContext, merge_modality_blocks
from .biophysical import BiophysicalModality
from .dna import DNAModality
from .protein import ProteinModality
from .regulatory import DRUG_TO_REGULATORY, RegulatoryModality
from .sequences import load_phenotype, load_sequence_df, numeric_labels

# the registry — the whole point of the refactor: any subset can be requested
MODALITIES = {
    "dna": DNAModality,
    "protein": ProteinModality,
    "biophysical": BiophysicalModality,
    "regulatory": RegulatoryModality,
}

DRUG_TO_LOCI = tb.DRUG_TO_LOCI

# --- locus-set policy -------------------------------------------------------
# Two reference codebases pick loci two different ways, and we inherited the
# narrower one. SD-CNN uses the PER-DRUG map above; MD-CNN ignores it entirely
# and feeds every locus in its flat `parameters/locus_order.py` (19) to every
# drug. The one locus that differs is fabG1, which has a FASTA on disk but is
# named by no drug in DRUG_TO_LOCI — so our multi-drug union came out at 18.
#
# EXTRA_LOCI is the opt-in overlay: WHO 2023 catalogue Table 21 TIER 1 genes
# that DRUG_TO_LOCI omits. fabG1 (Rv1483) heads the fabG1-inhA operon whose
# promoter carries the dominant non-katG isoniazid / ethionamide resistance
# mechanism, and WHO lists it tier 1 for both. Default OFF: turning it on makes
# a single-drug run no longer locus-matched to BIG-TB's SD-CNN, which is the
# baseline those runs are compared against.
EXTRA_LOCI = {"ISONIAZID": ["fabG1"], "ETHIONAMIDE": ["fabG1"]}


def drug_loci(drug, extra=False):
    """Gene loci for one drug: BIG-TB's DRUG_TO_LOCI, plus EXTRA_LOCI when
    ``extra`` is set (order preserved, no duplicates)."""
    base = list(DRUG_TO_LOCI.get(drug.upper(), []))
    if extra:
        base += [g for g in EXTRA_LOCI.get(drug.upper(), []) if g not in base]
    return base


def loci_on_disk(genotype_dir):
    """Every locus with an aligned FASTA in ``genotype_dir``, sorted.

    MD-CNN's own rule — its locus list is drug-independent, one channel per
    curated locus — so this is what a multi-drug run should default to if the
    goal is comparability with it. Picks up any locus curated later for free.
    Do NOT point this at a directory that also holds regulatory windows (the
    synthetic fixtures do): promoter regions would be read as coding loci."""
    return sorted(p.stem for p in Path(genotype_dir).glob("*.fasta"))


@dataclass
class DrugData:
    drug: str
    modalities: List[str]              # modalities that actually produced blocks
    requested: List[str]               # what the caller asked for
    blocks: List[FeatureBlock]
    y: np.ndarray                      # (N,) 0=R 1=S -1=missing
    isolate_ids: List[str]
    loci: List[str]
    gene_order: List[str]              # gene loci actually found on disk
    dropped: List[str] = field(default_factory=list)  # requested but empty

    @property
    def n(self):
        return len(self.isolate_ids)

    def arrays(self):
        return [b.array for b in self.blocks]

    def branch_specs(self):
        return [b.spec() for b in self.blocks]

    def class_counts(self):
        return {"R": int((self.y == 0).sum()),
                "S": int((self.y == 1).sum()),
                "missing": int((self.y == -1).sum())}

    def modality_tag(self):
        """Compact, stable tag for output filenames, e.g. 'dna+biophysical'."""
        return "+".join(m for m in MODALITIES if m in self.modalities)

    def summary(self):
        return {
            "drug": self.drug,
            "modalities": self.modalities,
            "dropped": self.dropped,
            "n_isolates": self.n,
            "genes": self.gene_order,
            "class_counts": self.class_counts(),
            "blocks": [
                {"name": b.name, "modality": b.modality,
                 "shape": tuple(b.array.shape), "channels": b.channel_names,
                 "note": b.note}
                for b in self.blocks
            ],
        }


def _resolve(modalities):
    unknown = [m for m in modalities if m not in MODALITIES]
    if unknown:
        raise ValueError(
            f"Unknown modalit{'y' if len(unknown) == 1 else 'ies'} {unknown}. "
            f"Available: {list(MODALITIES)}")
    # keep registry order for determinism, dedupe
    return [m for m in MODALITIES if m in set(modalities)]


def load_dataset(drug, modalities, genotype_dir, phenotype_csv,
                 regulatory_dir=None, loci=None, regulatory_loci=None,
                 per_modality_branch=True, dna_per_locus=True,
                 extra_loci=False, all_regulatory=False, delta=False,
                 variant_tokens=False, verbose=True):
    """Build a DrugData bundle for one drug over the requested modalities.

    loci                : which gene loci to load for dna/protein/biophysical.
                          Default = DRUG_TO_LOCI[drug] (the current behavior). Pass
                          a subset/reordering to control which — and how many — are
                          loaded, e.g. loci=['katG'].
    regulatory_loci     : which regulatory regions to load. Default = the WHO-derived
                          DRUG_TO_REGULATORY[drug]. Pass a list to override.
    per_modality_branch : ONE branch per modality (default) — each modality's
                          per-gene/per-region blocks are concatenated along length
                          into a single branch (matches the multi-drug layout). Pass
                          False for the older one-branch-per-gene layout.
    extra_loci          : add the EXTRA_LOCI overlay (WHO Table 21 tier-1 genes
                          DRUG_TO_LOCI omits — fabG1 for INH/ETO) to the default
                          locus set. Ignored when `loci` is given explicitly.
                          Default False: on, a run is no longer locus-matched to
                          BIG-TB's SD-CNN.
    all_regulatory      : keep the FULL WHO region set for the drug. By default the
                          regions are intersected with the loaded `loci`, so a run
                          never carries more promoter windows than coding loci —
                          the WHO candidate list is much longer than DRUG_TO_LOCI
                          (INH: 12 regions vs 2 loci) and most of those promoters
                          belong to genes whose CDS is not loaded at all. Note the
                          strictness cuts real signal in one known case: KANAMYCIN
                          keeps only `rrs` and loses the `eis` promoter, because
                          `eis` is not one of its coding loci. Ignored when
                          `regulatory_loci` is given explicitly.
    delta               : reference-difference encoding — zero every column that
                          matches the H37Rv reference, in every modality (see
                          datasets/sequences.py). Shape, channels and alphabet are
                          unchanged; what goes is the ~99.9% of each sequence that
                          is identical in every isolate and therefore carries no
                          discriminative signal. Default False (it changes the
                          input, so it is not comparable to an existing run).
    variant_tokens      : emit one SYMBOL ID per column instead of a one-hot, with
                          the H37Rv reference id, the exact reference-codon
                          coordinate and the codon phase attached to the block as
                          per-column constants (datasets/tokens.py). What the
                          variant-token architectures read. Biophysical is
                          deliberately excluded — see the note at the call site.
    dna_per_locus       : only when per_modality_branch=False — emit one DNA block
                          per locus instead of one concatenated DNA block. Mirrors
                          load_multidrug_dataset. Needed by the mdcnn architecture,
                          which stacks loci as channels and so must SEE them
                          separately; with it False, DNA stays one block even in
                          the per-locus layout (what this loader used to do).
    Requested loci with no FASTA on disk are skipped (warned when explicit)."""
    drug = drug.upper()
    modalities = _resolve(modalities)
    regulatory_dir = regulatory_dir or genotype_dir

    user_loci = loci is not None
    loci = list(loci) if user_loci else drug_loci(drug, extra_loci)
    user_reg = regulatory_loci is not None
    regulatory_loci = (list(regulatory_loci) if user_reg
                       else list(DRUG_TO_REGULATORY.get(drug, [])))
    if not user_reg and not all_regulatory:
        regulatory_loci = [r for r in regulatory_loci if r in set(loci)]

    mods = [MODALITIES[m]() for m in modalities]
    for m in mods:
        # DNA concatenates its loci internally unless told otherwise (protein /
        # biophysical / regulatory are per-locus by construction), so the
        # per-locus layout has to switch it on explicitly — same rule as
        # load_multidrug_dataset.
        if hasattr(m, "per_locus"):
            m.per_locus = (not per_modality_branch) and dna_per_locus
        m.delta = delta
        # symbol-id blocks for the variant-token architectures. Biophysical has
        # no id form on purpose: it is the modality whose whole claim is that
        # three properties stand in for the residue identity, so handing it that
        # identity would answer its own ablation.
        if hasattr(m, "variant_tokens"):
            m.variant_tokens = variant_tokens
    needs_genes = any(m.uses_genes for m in mods)

    df_phenos = load_phenotype(phenotype_csv)

    # --- establish the shared isolate axis ---------------------------------
    gene_seqs, gene_order = (load_sequence_df(genotype_dir, loci)
                             if needs_genes and loci else (None, []))
    # warn for ANY requested-but-missing locus, not just an explicit --loci: a
    # silently dropped locus is how fabG1 went unnoticed in the first place.
    if needs_genes and verbose:
        missing = [g for g in loci if g not in gene_order]
        if missing:
            src = "--loci" if user_loci else "the default locus set"
            print(f"  [load] {drug}: gene loci from {src} not found on disk, "
                  f"skipped: {missing}")

    if gene_seqs is not None and not gene_seqs.empty:
        isolates = list(gene_seqs.index.intersection(df_phenos.index))
        gene_seqs = gene_seqs.reindex(isolates)
    else:
        # regulatory-only (no gene modality / no gene loci): bootstrap the
        # isolate axis from the regulatory region FASTAs.
        reg_df, _ = load_sequence_df(regulatory_dir, regulatory_loci)
        if reg_df.empty:
            raise ValueError(
                f"No sequence data for {drug} with modalities {modalities} "
                f"(no gene loci found and no regulatory regions).")
        isolates = list(reg_df.index.intersection(df_phenos.index))

    df_phenos = df_phenos.loc[isolates]
    ctx = LoadContext(drug=drug, loci=loci, regulatory_loci=regulatory_loci,
                      isolates=isolates, gene_seqs=gene_seqs,
                      genotype_dir=str(genotype_dir), regulatory_dir=str(regulatory_dir))

    # --- run each modality --------------------------------------------------
    blocks, used, dropped = [], [], []
    for mod in mods:
        mod_blocks = mod.build(ctx)
        # report ANY skipped region, not just an explicit --regulatory-loci:
        # a silently dropped region is how the fabG1 gap stayed invisible.
        if mod.name == "regulatory" and verbose:
            loaded = {b.name.split(":", 1)[1] for b in mod_blocks}
            miss = [r for r in regulatory_loci if r not in loaded]
            if miss:
                src = "--regulatory-loci" if user_reg else "the WHO default set"
                print(f"  [load] {drug}: regulatory regions from {src} not found "
                      f"on disk, skipped: {miss}")
        if mod_blocks:
            # one branch per modality: concatenate this modality's per-gene/region
            # blocks (in their fixed order) into a single branch.
            if per_modality_branch and len(mod_blocks) > 1:
                mod_blocks = [merge_modality_blocks(mod_blocks, mod.name)]
            blocks.extend(mod_blocks)
            used.append(mod.name)
        else:
            dropped.append(mod.name)
            if verbose:
                print(f"  [load] {drug}: modality '{mod.name}' produced no data "
                      f"— dropped.")

    if not blocks:
        raise ValueError(f"No modality produced data for {drug} "
                         f"(requested {modalities}).")

    y = numeric_labels(df_phenos, drug)
    return DrugData(drug=drug, modalities=used, requested=modalities, blocks=blocks,
                    y=y, isolate_ids=isolates, loci=loci, gene_order=gene_order,
                    dropped=dropped)
