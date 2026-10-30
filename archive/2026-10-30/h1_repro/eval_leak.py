"""
H1: quantify the train/test leak in the BIG-TB SD-CNN pipeline for MOXIFLOXACIN.
NO training. Loads THEIR precomputed X_sparse + THEIR saved best model, and
evaluates it on:
  (a) the crossval split test set   -> non-stratified, seed 42  (truly held out)
  (b) the assess   split test set   -> stratified,     seed 42  (what they report)
The crossval script trains/saves the model on split (a)'s train portion; the
assess script re-splits (b) with the SAME seed but stratify=y, giving a DIFFERENT
partition. We measure how many of (b)'s test isolates were in (a)'s train set
(the leak) and the AUC on each.
Read-only: their code/files are untouched.
"""
import os, sys
import numpy as np, pandas as pd, sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

REF = "/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/Big-TB-benchmark/dna-tasks/SD-CNN/model_training"
sys.path.insert(0, os.path.join(REF, "parameters"))
from tb_cnn_codebase import (masked_multi_weighted_bce, masked_weighted_accuracy,
                             rs_encoding_to_numeric)

PICK = "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/SD-CNN/model_training/pickle_files/MOXI_ccp_60"
OUT  = "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/SD-CNN/training_output/MOXI_ccp_60"
DRUG, SEED, TEST_SIZE = "MOXIFLOXACIN", 42, 0.2

# --- data (their exact build) ------------------------------------------------
df = pd.read_parquet(os.path.join(PICK, "geno_pheno_metadata.parquet")).reset_index(drop=True)
all_idx = df.index.values
y_df, _ = rs_encoding_to_numeric(df, DRUG)          # R->0, S->1, missing->-1
y = y_df.values.ravel().astype(int)
valid = (y != -1)
print(f"Universe: {len(df)} isolates | MOXI-valid: {valid.sum()} "
      f"(R={int((y==0).sum())}, S={int((y==1).sum())})")

# --- reproduce BOTH splits ---------------------------------------------------
tr_cv, te_cv = train_test_split(all_idx, test_size=TEST_SIZE, random_state=SEED)          # crossval: NO stratify
tr_as, te_as = train_test_split(all_idx, test_size=TEST_SIZE, random_state=SEED,
                                stratify=df[DRUG].values)                                   # assess: stratify
tr_cv_s, te_cv_s, te_as_s = set(tr_cv), set(te_cv), set(te_as)

# leak = assess-test isolates that were in crossval-train (model saw them)
te_as_valid = [i for i in te_as if valid[i]]
leaked = [i for i in te_as_valid if i in tr_cv_s]
overlap_te = te_cv_s & te_as_s
print(f"\nassess-test MOXI-valid isolates: {len(te_as_valid)}")
print(f"  of those, in crossval-TRAIN (leaked/seen by model): {len(leaked)} "
      f"({100*len(leaked)/len(te_as_valid):.1f}%)")
print(f"  test sets overlap (te_cv ∩ te_as): {len(overlap_te)} / {len(te_as)} "
      f"({100*len(overlap_te)/len(te_as):.1f}%)")

# --- load THEIR best model + THEIR X_sparse, evaluate on each test set --------
from tensorflow.keras.models import load_model
X = sparse.load_npz(os.path.join(PICK, "MOXI_X_sparse.npz"))
# compile=False: we only need forward inference, skip loading their custom
# loss/metrics (which trip Keras-3 legacy-h5 deserialization).
model = load_model(os.path.join(OUT, "saved_models", "sd-cnn_model_best.h5"),
                   compile=False)

def auc_on(idx):
    idx = np.asarray(idx)
    m = valid[idx]
    iv = idx[m]
    Xd = X[iv, :].todense()
    p = np.squeeze(model.predict(Xd, batch_size=256, verbose=0))
    return roc_auc_score(y[iv], p), len(iv)

auc_cv, n_cv = auc_on(te_cv)   # clean held-out (what the model was actually held out from)
auc_as, n_as = auc_on(te_as)   # leaky (their reported protocol)
print(f"\nBEST MODEL evaluated on:")
print(f"  assess-test  (leaky, their reported split) : AUC = {auc_as:.4f}  (n={n_as})   [their csv: 0.8861]")
print(f"  crossval-test(clean, truly held out)       : AUC = {auc_cv:.4f}  (n={n_cv})")
print(f"\n  => leak inflation = {auc_as-auc_cv:+.4f}")
print(f"  their clean CV mean (MOXI_auc.csv) = 0.8190 ; our clean CV mean = 0.853")
