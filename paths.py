"""
Every input and output location, in one place. Each one can be overridden with
the environment variable named next to it.

ROOT is the repository root.

Shared data comes from Sai's curation folder (SAI), which is the source of
record: the aligned gene MSAs, the phenotype table and the GenBank reference.
/project is not mounted on every compute node; to run from a local mirror of
that folder, set ABR_SAI to it. This repo only adds the inputs her folder does
not have: regulatory promoter windows (msa/), AlphaFold structures and
binding-site distances (structures/, locusfusion/binding_sites/), and the
GenomeFusion references (genomefusion_refs/).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _env(name, default):
    return Path(os.environ.get(name) or default)


SAI = _env("ABR_SAI", "/project/pi_annagreen_umass_edu/saishradha/project_data_curation")

# Per-gene aligned FASTAs (one record per isolate + MT_H37Rv; 64 loci) and the
# phenotype table, joined on New_ID.
REAL_GENOTYPE_DIR = str(_env("ABR_GENOTYPE_DIR", SAI / "genomic_data" / "aligned"))
REAL_PHENOTYPE_CSV = str(_env("ABR_PHENOTYPE_CSV",
                              SAI / "phenotype_data" / "master_resistance_table.csv"))
GENBANK = str(_env("ABR_GENBANK", SAI / "make_msa" / "genbank_reference_data" / "genome.gbff"))

# Per-region promoter FASTAs for the regulatory modality: WHO Table-22 windows
# anchored on the start codon, extracted from the BIG-TB VCFs by msa/.
REAL_REGULATORY_DIR = str(_env("ABR_REGULATORY_DIR", ROOT / "regulatory_msa" / "aligned_atg"))
ATG_REGULATORY_DIR = REAL_REGULATORY_DIR

# AlphaFold models of the H37Rv proteins on the H37Rv codon grid (tracked in git).
REAL_STRUCTURE_DIR = str(ROOT / "structures" / "alphafold")

# Sai's lineage calls (ROLLINGDB_ID = New_ID, top-level `Lineage`), used by splits/.
LINEAGE_CSV = str(_env("ABR_LINEAGE_CSV", "/project/pi_annagreen_umass_edu/saishradha/"
                       "Data-Curation-for-MTB__sm_lineage_aware_sdcnn/BIG_TB_isolates_with_lineages.csv"))

# Leave-one-major-lineage-out splits: heldout_{L}/{DRUG}.csv (New_ID, role, fold) and
# heldout_{L}/MULTIDRUG.csv (splits/build_holdout_splits.py).
SPLIT_DIR = str(_env("ABR_SPLIT_DIR", ROOT / "results" / "splits"))

RESULTS_DIR = ROOT / "results"

# Stacked-CNN checkpoints (training/checkpoint.py). LocusFusion and GenomeFusion
# write theirs to ROOT/model_weights_{lf,gf}.
MODEL_WEIGHTS_DIR = str(_env("ABR_MODEL_WEIGHTS_DIR", ROOT / "model_weights_cnn"))

# GenomeFusion. The partition cache holds partition.npz (built from GENBANK and
# GF_REFS on first use); runs read genomefusion_cache_v3, the build with tlyA in
# the 19. GF_TOKENS is the token store directory (tokens.npz, built from the
# BIG-TB VCFs in GF_VCF_DIR by genomefusion.build_tokens). Results go to
# GF_RESULTS_DIR/<arm>/, weights to GF_WEIGHTS_DIR/<arm>/<drug>/best.pt.
GF_REFS = _env("ABR_GF_REFS", ROOT / "genomefusion_refs")
# WHO Table-22 promoter regions in absolute coordinates, anchored on the start
# codon (msa/build_promoter_coords.py). Not the TSS-anchored
# promoter_windows.json, which shifts katG's window by 481 bp.
GF_PROMOTERS = _env("ABR_GF_PROMOTERS", ROOT / "msa" / "windows" / "promoter_windows_atg.json")
GF_CACHE = _env("ABR_GF_CACHE", ROOT / "genomefusion_cache_v3")
GF_TOKENS = _env("ABR_GF_TOKENS", ROOT / "genomefusion_tokens")
GF_VCF_DIR = _env("ABR_GF_VCF_DIR", "/project/pi_annagreen_umass_edu/anna/BIG-TB_v2/isolates/"
                                    "BIG-TB_download_2025_07_01")
GF_RESULTS_DIR = RESULTS_DIR / "genomefusion"
GF_WEIGHTS_DIR = _env("ABR_GF_WEIGHTS_DIR", ROOT / "model_weights_gf")
