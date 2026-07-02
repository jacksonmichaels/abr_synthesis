"""
Single point of contact with the BIG-TB reference codebase. Everything here
is reused directly instead of re-implemented (BASE_TO_COLUMN, DRUG_TO_LOCI,
get_one_hot, make_genotype_df, rs_encoding_to_numeric, alpha_mat,
get_threshold_val all come straight from tb_cnn_codebase.py).
"""
import sys
from pathlib import Path

_TB_CNN_DIR = (
    Path(__file__).resolve().parent.parent
    / "Big-TB-benchmark" / "dna-tasks" / "SD-CNN"
    / "model_training" / "parameters"
)
if str(_TB_CNN_DIR) not in sys.path:
    sys.path.insert(0, str(_TB_CNN_DIR))

import tb_cnn_codebase as tb  # noqa: E402  (import after sys.path edit, by design)
