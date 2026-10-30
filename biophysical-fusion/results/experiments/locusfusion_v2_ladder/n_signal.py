"""Does missingness (uncalled bases, N) in a drug's own loci predict its phenotype
by itself? Per drug: AUC of (a) the isolate's N count over its loci, (b) any-N,
for predicting RESISTANCE, plus the share of isolates with any N, split by R/S.
    python results/experiments/locusfusion_v2_ladder/n_signal.py
"""
import sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
import numpy as np
from sklearn.metrics import roc_auc_score
from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, tb
from datasets.loader import drug_loci
from datasets.sequences import load_phenotype, load_sequence_df, numeric_labels

ph = load_phenotype(REAL_PHENOTYPE_CSV)
print(f"{'drug':13} {'n':>6} {'%R':>5} | {'anyN R':>7} {'anyN S':>7} | {'AUC Ncount':>10} {'AUC anyN':>8}")
for drug in tb.DRUG_TO_LOCI:
    loci = drug_loci(drug)
    seq_df, _ = load_sequence_df(REAL_GENOTYPE_DIR, loci)
    isos = [i for i in ph.index if i in seq_df.index]
    y = numeric_labels(ph.loc[isos], drug)
    keep = y != -1
    isos = [i for i, k in zip(isos, keep) if k]; y = y[keep]
    res = (y == 0).astype(int)                     # 1 = resistant
    s = seq_df.loc[isos, loci].fillna("")
    ncount = np.zeros(len(isos))
    for g in loci:
        ncount += s[g].map(lambda q: sum(ch not in "ACGT-" for ch in q)).to_numpy()
    anyn = (ncount > 0).astype(int)
    print(f"{drug:13} {len(isos):6d} {res.mean():5.1%} | {anyn[res==1].mean():7.1%} {anyn[res==0].mean():7.1%} | "
          f"{roc_auc_score(res, ncount):10.3f} {roc_auc_score(res, anyn):8.3f}", flush=True)
