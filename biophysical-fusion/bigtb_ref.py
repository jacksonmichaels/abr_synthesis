"""
Single point of contact with the BIG-TB reference codebase. Everything here
is reused directly instead of re-implemented (BASE_TO_COLUMN, DRUG_TO_LOCI,
get_one_hot, make_genotype_df, rs_encoding_to_numeric, alpha_mat,
get_threshold_val all come straight from tb_cnn_codebase.py).
"""
import os
import sys
from pathlib import Path

# tb_cnn_codebase imports TensorFlow at module load even though we only reuse its
# numpy/pandas utilities (never TF compute). Silence TF's C++ backend startup
# noise (oneDNN / cpu_feature_guard / absl INFO+WARNING lines) so training output
# stays clean. These must be set BEFORE TensorFlow is first imported — hence the
# top of this module, which is imported before anything runs TF.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

_TB_CNN_DIR = (
    Path(__file__).resolve().parent.parent
    / "Big-TB-benchmark" / "dna-tasks" / "SD-CNN"
    / "model_training" / "parameters"
)
if str(_TB_CNN_DIR) not in sys.path:
    sys.path.insert(0, str(_TB_CNN_DIR))

import tb_cnn_codebase as tb  # noqa: E402  (import after sys.path edit, by design)

# Real genotype/phenotype data — UMass Unity cluster, pi_annagreen allocation
# (access granted 2026-07-07). These two paths are the ones marked "Invariant
# to antibiotic" in every SD-CNN parameter file, e.g.
#   Big-TB-benchmark/dna-tasks/SD-CNN/model_training/parameter_files/
#     optimized_epochs/RIF_ccp_epoch_60.txt  (lines `genotype_input_directory:`
#     / `phenotype_file:`).
# `data.build_dataset` joins them exactly as tb_cnn_codebase.make_geno_pheno_dataset
# does: phenotype indexed on New_ID, intersected with the FASTA record IDs
# (sample accessions like SAMN…/SAMEA…). Verified 2026-07-07: 17,942/17,943
# rpoB isolates intersect (only the MT_H37Rv reference row drops out).
REAL_GENOTYPE_DIR = (
    "/project/pi_annagreen_umass_edu/saishradha/project_data_curation"
    "/genomic_data/aligned"
)
REAL_PHENOTYPE_CSV = (
    "/project/pi_annagreen_umass_edu/saishradha/project_data_curation"
    "/phenotype_data/master_resistance_table.csv"
)
