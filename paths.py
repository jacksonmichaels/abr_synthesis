"""
Every input and output location, in one place. Each one can be overridden with
the environment variable named next to it.

ROOT is the repository root. The data is not in git; link it into ROOT from the
shared workspace (see README.md), so the defaults below resolve in any clone.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _env(name, default):
    return Path(os.environ.get(name) or default)


DATA = _env("ABR_DATA", ROOT / "data_snapshot")

# Per-gene aligned FASTAs (one record per isolate + MT_H37Rv) and the phenotype
# table, joined on New_ID. Local copy of
# /project/pi_annagreen_umass_edu/saishradha/project_data_curation.
REAL_GENOTYPE_DIR = str(_env("ABR_GENOTYPE_DIR", DATA / "genomic_data" / "aligned"))
REAL_PHENOTYPE_CSV = str(_env("ABR_PHENOTYPE_CSV",
                              DATA / "phenotype_data" / "master_resistance_table.csv"))
GENBANK = str(_env("ABR_GENBANK", DATA / "genbank" / "genome.gbff"))

# The 19 curated loci with the MSA insertion bug repaired (eis/embB/ethA/rrs
# rebuilt, the other 15 symlinked to the originals). LocusFusion v5_fixed reads these.
FIXED_GENOTYPE_DIR = str(_env("ABR_FIXED_GENOTYPE_DIR", ROOT / "curated_fix" / "aligned19"))

# Per-region promoter FASTAs for the regulatory modality (WHO Table-22 windows,
# extracted from the BIG-TB VCFs by regulatory_msa/). `aligned` is the
# TSS-anchored build; `aligned_atg` is anchored on the start codon (v5_fixed).
REAL_REGULATORY_DIR = str(_env("ABR_REGULATORY_DIR", ROOT / "regulatory_msa" / "aligned"))
ATG_REGULATORY_DIR = str(_env("ABR_ATG_REGULATORY_DIR", ROOT / "regulatory_msa" / "aligned_atg"))

# AlphaFold models of the H37Rv proteins on the H37Rv codon grid (tracked in git).
REAL_STRUCTURE_DIR = str(ROOT / "structures" / "alphafold")

# TB-Profiler lineage calls, keyed by New_ID (splits/call_lineages.py).
LINEAGE_CSV = str(_env("ABR_LINEAGE_CSV", DATA / "lineage" / "lineage_calls.csv"))

# Lineage-grouped splits: one {DRUG}.csv per drug (New_ID, role, fold) and the
# multi-drug MULTIDRUG.csv (splits/build_lineage_splits.py).
SPLIT_DIR = str(_env("ABR_SPLIT_DIR", ROOT / "results" / "splits"))

RESULTS_DIR = ROOT / "results"

# Stacked-CNN checkpoints (training/checkpoint.py). LocusFusion and GenomeFusion
# write theirs to ROOT/model_weights_{lf,gf}.
MODEL_WEIGHTS_DIR = str(_env("ABR_MODEL_WEIGHTS_DIR", ROOT / "model_weights_cnn"))
