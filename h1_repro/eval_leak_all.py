"""
(c) Quantify the BIG-TB SD-CNN crossval/assess train-test leak for EVERY drug.
For each drug's param file: reproduce their two splits, load THEIR saved best
model + THEIR X_sparse, and report:
  reported(leaky)  = best model on assess (stratified) split  [reproduces their csv]
  clean(held-out)  = best model on crossval (non-strat) split it was held out from
  clean_CV_mean    = their own 5-fold val AUC (their *_auc.csv)
  leak%            = assess-test isolates that were in crossval-train
NO training, inference only, their code/files untouched.
"""
import os, sys, glob, yaml, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

REF = "/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/Big-TB-benchmark/dna-tasks/SD-CNN/model_training"
sys.path.insert(0, os.path.join(REF, "parameters"))
from tb_cnn_codebase import rs_encoding_to_numeric, DRUG_TO_LOCI
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

PARAM_DIR = os.path.join(REF, "parameter_files", "optimized_epochs")
rows = []

for pf in sorted(glob.glob(os.path.join(PARAM_DIR, "*.txt"))):
    kw = yaml.safe_load(open(pf))
    drug = kw["drug"]; seed = kw["random_seed"]; ts = kw["test_size"]
    meta = kw["metadata_path"]; xsp = kw["X_sparse_path"]
    best = os.path.join(kw["saved_model_path"], "sd-cnn_model_best.h5")
    out = kw["output_path"]
    try:
        if not (os.path.isfile(meta) and os.path.isfile(xsp) and os.path.isfile(best)):
            print(f"[{drug}] missing artifacts, skip"); continue
        df = pd.read_parquet(meta).reset_index(drop=True)
        y = rs_encoding_to_numeric(df, drug)[0].values.ravel().astype(int)
        valid = (y != -1)
        strat = df[drug].fillna("-1").astype(str).values
        idx = df.index.values
        tr_cv, te_cv = train_test_split(idx, test_size=ts, random_state=seed)
        tr_as, te_as = train_test_split(idx, test_size=ts, random_state=seed, stratify=strat)
        tr_cv_s = set(tr_cv)

        te_as_valid = [i for i in te_as if valid[i]]
        leaked = sum(1 for i in te_as_valid if i in tr_cv_s)
        leak_pct = 100 * leaked / max(len(te_as_valid), 1)

        X = sparse.load_npz(xsp)
        model = load_model(best, compile=False)
        def auc_on(ix):
            ix = np.asarray(ix); m = valid[ix]; iv = ix[m]
            if len(np.unique(y[iv])) < 2:
                return float("nan"), len(iv)
            p = np.squeeze(model.predict(X[iv, :].todense(), batch_size=256, verbose=0))
            return roc_auc_score(y[iv], p), len(iv)
        auc_leaky, n_as = auc_on(te_as)
        auc_clean, n_cv = auc_on(te_cv)
        del X; K.clear_session()

        # their own reported csvs
        rep = np.nan; cvm = np.nan
        tcsv = out + "_test_set_drug_auc.csv"; acsv = out + "_auc.csv"
        if os.path.isfile(tcsv):
            rep = float(pd.read_csv(tcsv)["AUC"].iloc[0])
        if os.path.isfile(acsv):
            cvm = float(pd.read_csv(acsv)["AUC"].mean())

        nR = int((y == 0).sum()); nS = int((y == 1).sum())
        rows.append(dict(drug=drug, loci="+".join(DRUG_TO_LOCI.get(drug, [])),
                         R=nR, S=nS, pctR=round(100*nR/(nR+nS), 1),
                         reported=round(rep, 4), leaky_repro=round(auc_leaky, 4),
                         clean_test=round(auc_clean, 4), clean_CV=round(cvm, 4),
                         leak_pct=round(leak_pct, 1),
                         inflation=round(auc_leaky - auc_clean, 4)))
        print(f"[{drug:13s}] reported={rep:.3f} leaky_repro={auc_leaky:.3f} "
              f"clean_test={auc_clean:.3f} clean_CV={cvm:.3f} leak={leak_pct:.0f}% "
              f"infl={auc_leaky-auc_clean:+.3f}", flush=True)
    except Exception as e:
        print(f"[{drug}] ERROR: {type(e).__name__}: {e}", flush=True)

res = pd.DataFrame(rows)
res.to_csv("/home/jacksonmicha_umass_edu/abr_workspace/h1_repro/leak_all.csv", index=False)
print("\n==== SUMMARY (sorted by inflation) ====")
print(res.sort_values("inflation", ascending=False).to_string(index=False))
print(f"\nmean inflation across drugs: {res['inflation'].mean():+.4f}")
