"""
The seven helpers this codebase uses from BIG-TB's tb_cnn_codebase.py
(Big-TB-benchmark/dna-tasks/SD-CNN/model_training/parameters/tb_cnn_codebase.py),
copied verbatim. tb_cnn_codebase imports TensorFlow, h5py, sparse and ipdb at
module load; none of that is needed for these numpy/pandas functions, so they
live here and the Big-TB clone is not needed at run time.

Authors of the originals: Michael Chen, Anna G. Green, Chang-ho Yoon,
Saishradha Mohanty.
"""
import numpy as np
import pandas as pd
from Bio import SeqIO

BASE_TO_COLUMN = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '-': 4}

DRUG_TO_LOCI = {
    'ISONIAZID': ['inhA', 'katG'],
    'RIFAMPICIN': ['rpoB', 'rpoC'],
    'ETHAMBUTOL': ['embC', 'embA', 'embB'],
    'PYRAZINAMIDE': ['pncA'],
    'STREPTOMYCIN': ['rpsL', 'rrs', 'gid'],
    'KANAMYCIN': ['rrs'],
    'AMIKACIN': ['rrs', 'eis'],
    'CAPREOMYCIN': ['rrs', 'rrl', 'tlyA'],
    'LEVOFLOXACIN': ['gyrB', 'gyrA'],
    'MOXIFLOXACIN': ['gyrB', 'gyrA'],
    'ETHIONAMIDE': ['inhA', 'ethA', 'ethR'],
}


def get_one_hot(sequence):
    """Return one-hot encoding of a DNA sequence.
    Shape: (L_seq, len(one hot encoding letters))
    """
    seq_len = len(sequence)
    one_hot = np.zeros((seq_len, len(BASE_TO_COLUMN)), dtype=int)
    for i, base in enumerate(sequence):
        if base in BASE_TO_COLUMN:
            one_hot[i, BASE_TO_COLUMN[base]] = 1
    return one_hot


def sequence_dictionary(filename):
    """Load one FASTA file and return a DataFrame with isolate IDs and sequences."""
    seq_dict = SeqIO.to_dict(
        SeqIO.parse(filename, "fasta"),
        key_function=lambda x: x.id.split("/")[-1].split(".cut")[0]
    )

    # create a dictionary of identifier: sequence
    for identifier, sequence in seq_dict.items():
        seq_dict[str(identifier)] = str(sequence.seq)

    df = pd.DataFrame.from_dict(seq_dict, orient='index')
    gene_name = filename.split("/")[-1].split("_")[0]
    df.columns = [gene_name]

    return df


def rs_encoding_to_numeric(df_geno_pheno, drugs_list):
    """Convert R/S/-1 to numeric {R:0, S:1, missing:-1} encoding."""
    if isinstance(drugs_list, str):
        drugs_list = [drugs_list]

    y_all_rs = df_geno_pheno[drugs_list].fillna('-1').astype(str)
    mapping = {'R': 0, 'S': 1, '-1.0': -1, '-1': -1}
    y_all = y_all_rs.replace(mapping)
    y_all.index = list(range(0, y_all.shape[0]))
    return y_all, y_all.values


def alpha_mat(subset_y, df_geno_pheno, weight=1.0, drug_name=None):
    """
    Create alpha matrix for a single drug (reflects proportion of resistant/sensitive strains). NaN fixed
    """
    if subset_y.ndim == 1:
        subset_y = subset_y.reshape(-1, 1)

    # Mask missing (-1)
    valid_idx = np.where(subset_y != -1)[0]
    y_valid = subset_y[valid_idx]

    resistant_num = np.sum(y_valid == 0)
    sensitive_num = np.sum(y_valid == 1)

    if resistant_num + sensitive_num == 0:
        raise ValueError("No valid phenotype data found for this drug!")

    alpha_val = resistant_num / float(resistant_num + sensitive_num)
    if drug_name:
        print(f"{drug_name}: {resistant_num} resistant, {sensitive_num} sensitive (α={alpha_val:.3f})")

    # Initialize alpha matrix (same shape as y)
    alpha_matrix = np.zeros_like(subset_y, dtype=np.float32)
    alpha_matrix[np.where(subset_y == 1.0)] = weight * alpha_val   # sensitive → +α
    alpha_matrix[np.where(subset_y == 0.0)] = -alpha_val           # resistant → −α
    alpha_matrix[np.where(subset_y == -1.0)] = 0.0                 # missing → 0

    # Safety: replace NaNs
    alpha_matrix = np.nan_to_num(alpha_matrix, nan=0.0)

    return alpha_matrix


def get_threshold_val(y_true, y_pred):
    """Compute optimal threshold maximizing (sensitivity + specificity)."""
    thresholds = np.linspace(0, 1, 101)
    num_sensitive = np.sum(y_true == 1)
    num_resistant = np.sum(y_true == 0)
    fpr_, tpr_ = [], []

    for t in thresholds:
        fp = np.sum((y_pred < t) & (y_true == 1))
        tp = np.sum((y_pred < t) & (y_true == 0))
        fpr_.append(fp / float(num_sensitive))
        tpr_.append(tp / float(num_resistant))

    fpr_, tpr_ = np.array(fpr_), np.array(tpr_)
    sens_spec_sum = (1 - fpr_) + tpr_
    best_idx = np.argmax(sens_spec_sum)
    return {
        'threshold': thresholds[best_idx],
        'spec': 1 - fpr_[best_idx],
        'sens': tpr_[best_idx]
    }
