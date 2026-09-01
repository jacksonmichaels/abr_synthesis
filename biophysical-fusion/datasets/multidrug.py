"""
Multi-drug dataset: load the UNION of every drug's loci once and attach a
per-isolate label MATRIX (one column per drug) instead of a single-drug vector.
This is the data side of the MD-CNN-style model (``models.MultiDrugNet``): every
drug is predicted simultaneously from the same genotype, so loci shared across
drugs (rrs, gyrA, inhA, …) are loaded once and their features are shared.

    from datasets import load_multidrug_dataset
    data = load_multidrug_dataset(None, ["dna"], geno_dir, pheno_csv)
    data.drugs            # ['ISONIAZID', 'RIFAMPICIN', ...]  (label-column order)
    data.branch_specs()   # one (5, L_g) spec per locus block
    data.Y                # (N, n_drugs) float: 0=R 1=S -1=missing

Reuses the same per-modality machinery as the single-drug ``load_dataset`` (the
modality classes, the shared isolate axis, the aligned-FASTA loader). The only
differences are (1) loci default to the union across the requested drugs and
(2) by default each MODALITY is one branch — every locus of a modality is
concatenated along the length axis into a single (N, C, sum L) block
(``per_modality_branch=True``; pass False for the older one-branch-per-locus
layout).

Positional alignment (what IS and ISN'T aligned)
------------------------------------------------
* Isolate axis: FULLY aligned. Row i of every block is the same isolate, so the
  fused feature vector always describes one strain. This is the only alignment
  the late-fusion head relies on and it is exact.
* Within a branch: each locus comes from an aligned MSA FASTA (fixed length per
  locus across isolates), and loci are concatenated in a single fixed union
  order, so a given column is the same genomic position for every isolate. Conv
  preserves this locally; after global flatten the absolute position becomes a
  fixed feature index (consistent across isolates). Kernels that straddle a
  locus boundary blend two genes — a minor, standard artifact of concatenated-
  input CNNs.
* ACROSS branches / modalities: NOT positionally aligned. Each branch is encoded
  independently and only its POOLED, FLATTENED vector is concatenated for the
  dense head. Nothing registers "DNA codon 315 of katG" to "protein residue 315
  of katG": they land at unrelated indices in their respective flattened vectors,
  and the modalities are at different resolutions/lengths anyway (DNA ~3x protein;
  translation is variable-length). The head can only learn cross-modal relations
  statistically — there is no inductive bias enforcing residue-level correspondence.
  Loci ARE concatenated in the same union order in every modality, so gene k sits
  in the same ordinal slot across branches, but that is ordinal, not positional.
  True residue-level cross-modal alignment would need a different architecture
  (stack modalities as channels on a shared residue axis, or positional
  cross-attention) — see the answer accompanying this change.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from bigtb_ref import tb
from .base import LoadContext, FeatureBlock, merge_modality_blocks
from .loader import MODALITIES, DRUG_TO_LOCI, drug_loci, loci_on_disk, _resolve  # noqa: F401
from .regulatory import DRUG_TO_REGULATORY
from .sequences import load_phenotype, load_sequence_df

# Canonical drug order for the label-matrix columns (the DRUG_TO_LOCI keys).
ALL_DRUGS = list(DRUG_TO_LOCI)


def union_loci(drugs, extra=False):
    """Ordered, de-duplicated union of every drug's gene loci.

    This is the SD-CNN-derived rule (union of the per-drug map). MD-CNN instead
    uses every curated locus regardless of drug — see ``loci_on_disk``, which is
    what the multi-drug runner defaults to on real data. ``extra`` adds the
    EXTRA_LOCI overlay (fabG1 for INH/ETO)."""
    out = []
    for d in drugs:
        for g in drug_loci(d, extra):
            if g not in out:
                out.append(g)
    return out


def union_regulatory(drugs):
    """Ordered, de-duplicated union of every drug's WHO regulatory regions."""
    out = []
    for d in drugs:
        for r in DRUG_TO_REGULATORY.get(d.upper(), []):
            if r not in out:
                out.append(r)
    return out


# per-modality branch concatenation is shared with the single-drug loader:
# datasets.base.merge_modality_blocks.


@dataclass
class MultiDrugData:
    drugs: List[str]                   # label-column order
    modalities: List[str]              # modalities that produced blocks
    requested: List[str]
    blocks: List[FeatureBlock]
    Y: np.ndarray                      # (N, n_drugs) 0=R 1=S -1=missing
    isolate_ids: List[str]
    loci: List[str]                    # union gene loci actually found on disk
    gene_order: List[str]
    dropped: List[str] = field(default_factory=list)

    @property
    def n(self):
        return len(self.isolate_ids)

    @property
    def n_drugs(self):
        return len(self.drugs)

    def arrays(self):
        return [b.array for b in self.blocks]

    def branch_specs(self):
        return [b.spec() for b in self.blocks]

    def modality_tag(self):
        return "+".join(m for m in MODALITIES if m in self.modalities)

    def per_drug_counts(self):
        """{drug: {R, S, missing}} over the label matrix."""
        out = {}
        for j, d in enumerate(self.drugs):
            col = self.Y[:, j]
            out[d] = {"R": int((col == 0).sum()), "S": int((col == 1).sum()),
                      "missing": int((col == -1).sum())}
        return out

    def summary(self):
        return {
            "drugs": self.drugs,
            "modalities": self.modalities,
            "dropped": self.dropped,
            "n_isolates": self.n,
            "loci": self.gene_order,
            "per_drug_counts": self.per_drug_counts(),
            "blocks": [{"name": b.name, "modality": b.modality,
                        "shape": tuple(b.array.shape)} for b in self.blocks],
        }


def load_multidrug_dataset(drugs, modalities, genotype_dir, phenotype_csv,
                           regulatory_dir=None, loci=None, regulatory_loci=None,
                           per_modality_branch=True, dna_per_locus=True,
                           extra_loci=False, all_regulatory=False, delta=False,
                           verbose=True):
    """Build a MultiDrugData bundle predicting ``drugs`` (default: all) from the
    union of their loci.

    drugs               : list of drug names, or None for all 11 (ALL_DRUGS order).
    modalities          : subset of dna/protein/biophysical/regulatory.
    loci                : gene loci to load (default: union across ``drugs``;
                          pass ``loci_on_disk(genotype_dir)`` for MD-CNN's
                          drug-independent all-loci rule, which is what
                          scripts/run_multidrug.py does on real data).
    extra_loci          : add the EXTRA_LOCI overlay (fabG1 for INH/ETO) to the
                          per-drug union. Ignored when ``loci`` is given.
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
    regulatory_loci     : regulatory regions (default: union across ``drugs``).
    per_modality_branch : ONE branch per modality — every locus of a modality is
                          concatenated into a single (N, C, sum L) block (default,
                          4 branches for all-modalities). False -> one branch per
                          locus/region (the ~100-branch layout).
    dna_per_locus       : only when per_modality_branch=False — DNA one block per
                          locus (True) vs one concatenated block (False).
    delta               : reference-difference encoding — zero every column that
                          matches the H37Rv reference, in every modality (see
                          datasets/sequences.py). Same flag, same meaning and the
                          same per-modality plumbing as load_dataset's."""
    drugs = [d.upper() for d in (drugs if drugs else ALL_DRUGS)]
    modalities = _resolve(modalities)
    regulatory_dir = regulatory_dir or genotype_dir

    loci = list(loci) if loci is not None else union_loci(drugs, extra_loci)
    user_reg = regulatory_loci is not None
    regulatory_loci = (list(regulatory_loci) if user_reg
                       else union_regulatory(drugs))
    if not user_reg and not all_regulatory:
        regulatory_loci = [r for r in regulatory_loci if r in set(loci)]

    mods = [MODALITIES[m]() for m in modalities]
    for m in mods:
        m.delta = delta
        if m.name == "dna":
            # one DNA branch (concatenated) when per_modality_branch; else honor
            # dna_per_locus for the per-locus layout.
            m.per_locus = (not per_modality_branch) and dna_per_locus
    needs_genes = any(m.uses_genes for m in mods)

    df_phenos = load_phenotype(phenotype_csv)

    gene_seqs, gene_order = (load_sequence_df(genotype_dir, loci)
                             if needs_genes and loci else (None, []))
    if needs_genes and verbose:
        missing = [g for g in loci if g not in gene_order]
        if missing:
            print(f"  [load-md] gene loci not found on disk, skipped: {missing}")

    if gene_seqs is not None and not gene_seqs.empty:
        isolates = list(gene_seqs.index.intersection(df_phenos.index))
        gene_seqs = gene_seqs.reindex(isolates)
    else:
        reg_df, _ = load_sequence_df(regulatory_dir, regulatory_loci)
        if reg_df.empty:
            raise ValueError("No sequence data for the requested drugs/modalities "
                             "(no gene loci and no regulatory regions found).")
        isolates = list(reg_df.index.intersection(df_phenos.index))

    df_phenos = df_phenos.loc[isolates]
    ctx = LoadContext(drug="+".join(drugs), loci=loci,
                      regulatory_loci=regulatory_loci, isolates=isolates,
                      gene_seqs=gene_seqs, genotype_dir=str(genotype_dir),
                      regulatory_dir=str(regulatory_dir))

    blocks, used, dropped = [], [], []
    for mod in mods:
        mod_blocks = mod.build(ctx)
        if mod_blocks:
            # Downcast one-hot blocks (dna/protein/regulatory are 0/1) to uint8 as
            # EACH modality is built — before the next modality allocates its own
            # float32 arrays — so the load-time peak stays one-modality-sized, not
            # sum-of-all-modalities. Cuts host RAM ~4x (the all-drug all-modality
            # union is ~100 blocks over ~17.9k isolates). biophysical is continuous
            # -> left float32. training.multimodal._batch casts back to float per batch.
            for b in mod_blocks:
                if b.modality in ("dna", "protein", "regulatory") and b.array.dtype != np.uint8:
                    b.array = b.array.astype(np.uint8)
            # one branch per modality: concatenate this modality's per-locus
            # blocks (in the shared union order) into a single branch.
            if per_modality_branch and len(mod_blocks) > 1:
                mod_blocks = [merge_modality_blocks(mod_blocks, mod.name)]
            blocks.extend(mod_blocks)
            used.append(mod.name)
        else:
            dropped.append(mod.name)
            if verbose:
                print(f"  [load-md] modality '{mod.name}' produced no data — dropped.")

    if not blocks:
        raise ValueError(f"No modality produced data (requested {modalities}).")

    # label matrix (N, n_drugs), columns in `drugs` order, via the BIG-TB encoder
    _, Y = tb.rs_encoding_to_numeric(df_phenos, drugs)
    Y = Y.astype(np.float32)

    if verbose:
        print(f"  [load-md] drugs={len(drugs)} loci={len(gene_order)} "
              f"blocks={len(blocks)} isolates={len(isolates)}")
    return MultiDrugData(drugs=drugs, modalities=used, requested=modalities,
                         blocks=blocks, Y=Y, isolate_ids=isolates, loci=loci,
                         gene_order=gene_order, dropped=dropped)
