"""Is the uncalled-base signal biology or batch?

locusfusion_v3's only change from v2 is that an uncalled base (`N` in the
aligned FASTA) makes a token, and that is worth ~+0.018 macro CV. Two readings:

  batch    N rate tracks the source collection / sequencing run, and source
           tracks resistance prevalence, so N predicts phenotype by proxy.
  biology  the pipeline sets mixed calls (alt AF 0.25-0.75) to missing, so an N
           at a resistance codon can be a heteroresistant (mixed) infection.

Three tests, per drug, on that drug's own loci (DRUG_TO_LOCI):
  A  N burden by source collection (`category`, 4 sets).
  B  AUC for resistance of: the isolate's N count; the source collection alone
     (5-fold CV logistic); and the N count WITHIN each collection (weighted
     mean) -- batch predicts B1 ~ B2 and B3 ~ 0.5.
  C  the alignment columns where N is most enriched in resistant isolates,
     mapped to H37Rv CDS codons -- biology predicts the known resistance codons
     (rpoB 450, katG 315, inhA promoter, pncA, ...).

    sbatch scripts/sbatch/missingness_check.sh
Writes results/analysis/missingness_v3/{SUMMARY.md, *.csv}.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV  # noqa: E402
from datasets.cds import _reference_row, cds_columns  # noqa: E402
from datasets.loader import DRUG_TO_LOCI  # noqa: E402
from datasets.multidrug import ALL_DRUGS  # noqa: E402
from datasets.sequences import load_sequence_df  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "analysis" / "missingness_v3"
OUT.mkdir(parents=True, exist_ok=True)
MIN_N = 20                     # a column needs >= this many N isolates to be ranked

pheno = pd.read_csv(REAL_PHENOTYPE_CSV, index_col="New_ID", dtype=str, low_memory=False)
loci = sorted({l for d in ALL_DRUGS for l in DRUG_TO_LOCI[d]})
seqs, loci = load_sequence_df(REAL_GENOTYPE_DIR, loci)
seqs = seqs.loc[seqs.index.intersection(pheno.index)]
seqs = seqs[~seqs.index.duplicated()]
pheno = pheno.loc[seqs.index]
src = pheno["category"].fillna("unknown")
print(f"{len(seqs)} isolates, loci {loci}", flush=True)

nmask = {}                     # locus -> (N, L) bool, True where the base is N
for l in loci:
    s = seqs[l].fillna("").astype(str)
    L = int(s.str.len().max())
    arr = np.frombuffer("".join(x.ljust(L, "-") for x in s).encode(), dtype=np.uint8)
    nmask[l] = arr.reshape(len(s), L) == ord("N")
n_by_locus = pd.DataFrame({l: m.sum(1) for l, m in nmask.items()}, index=seqs.index)


def codon_of(locus, col):
    """alignment column -> (H37Rv CDS nt, codon), or None outside the CDS."""
    win = cds_columns(REAL_GENOTYPE_DIR, locus)
    if not win:
        return None
    ref = _reference_row(sorted(Path(REAL_GENOTYPE_DIR).glob(f"{locus}*.fasta"))[0].as_posix())
    off = len(ref[:col].replace("-", "")) - len(ref[:win[0]].replace("-", ""))
    if 0 <= off < len(ref[win[0]:win[1]].replace("-", "")):
        return off + 1, off // 3 + 1
    return ("upstream" if off < 0 else "downstream"), off


# --- A: N burden by source ---------------------------------------------------
tot = n_by_locus.sum(1)
A = pd.DataFrame({"isolates": src.value_counts(),
                  "any N (share)": (tot > 0).groupby(src).mean(),
                  "mean N per isolate": tot.groupby(src).mean(),
                  "median": tot.groupby(src).median()})
A.to_csv(OUT / "A_by_source.csv")

# --- B and C, per drug ---------------------------------------------------------
B, C = [], []
for drug in ALL_DRUGS:
    y = pheno[drug].map({"R": 1, "S": 0})
    ok = y.notna()
    yv = y[ok].astype(int).values
    dl = [l for l in DRUG_TO_LOCI[drug] if l in nmask]
    n = n_by_locus.loc[ok, dl].sum(1).values
    s = src[ok]
    auc_n = roc_auc_score(yv, n) if n.any() else 0.5
    X = pd.get_dummies(s).values.astype(float)
    p_src = cross_val_predict(LogisticRegression(max_iter=1000), X, yv,
                              cv=StratifiedKFold(5, shuffle=True, random_state=42),
                              method="predict_proba")[:, 1]
    auc_src = roc_auc_score(yv, p_src)
    within, w = [], []
    for g in s.unique():
        m = (s == g).values
        if len(set(yv[m])) == 2 and n[m].any():
            within.append(roc_auc_score(yv[m], n[m])); w.append(m.sum())
    auc_within = np.average(within, weights=w) if within else np.nan
    B.append({"drug": drug, "loci": "+".join(dl), "n": int(ok.sum()),
              "R share": yv.mean(), "isolates with any N": int((n > 0).sum()),
              "R share | any N": yv[n > 0].mean() if (n > 0).any() else np.nan,
              "R share | no N": yv[n == 0].mean(),
              "AUC(N count)": auc_n, "AUC(source only, CV)": auc_src,
              "AUC(N count) within source": auc_within})
    for l in dl:
        M = nmask[l][ok.values]
        cnt = M.sum(0)
        for col in np.where(cnt >= MIN_N)[0]:
            a = int((M[:, col] & (yv == 1)).sum()); b = int(cnt[col]) - a
            c = int(((~M[:, col]) & (yv == 1)).sum()); d = int((~M[:, col]).sum()) - c
            odds, p = stats.fisher_exact([[a, b], [c, d]])
            C.append({"drug": drug, "locus": l, "column": int(col), "N isolates": int(cnt[col]),
                      "R share | N here": a / cnt[col], "R share | called": c / max(c + d, 1),
                      "odds ratio": odds, "fisher p": p})
    print(f"{drug}: AUC N {auc_n:.3f}  source {auc_src:.3f}  within {auc_within:.3f}", flush=True)

B = pd.DataFrame(B)
B.to_csv(OUT / "B_auc.csv", index=False)
C = pd.DataFrame(C)
top = (C[C["odds ratio"] > 1].sort_values("fisher p").groupby("drug").head(5)
       if len(C) else C)
if len(top):
    top = top.assign(h37rv=[codon_of(r.locus, r.column) for r in top.itertuples()])
C.to_csv(OUT / "C_columns.csv", index=False)
top.to_csv(OUT / "C_top.csv", index=False)

md = ["# Uncalled-base (N) check", "",
      f"{len(seqs)} isolates; per-drug loci (DRUG_TO_LOCI). Source = phenotype "
      "table `category`.", "",
      "## A. N burden by source collection", "", "```\n" + A.round(3).to_string() + "\n```", "",
      "## B. Does N predict resistance, and is it the source?", "",
      "Batch predicts AUC(N) ~ AUC(source) and within-source AUC(N) ~ 0.5.", "",
      "```\n" + B.round(3).to_string(index=False) + "\n```", "",
      f"## C. Columns where N is most enriched in resistant isolates (>= {MIN_N} N isolates, "
      "top 5 per drug by Fisher p)", "",
      "`h37rv` = (CDS nucleotide, codon), or (upstream/downstream, offset in nt).", "",
      ("```\n" + top.round(4).to_string(index=False) + "\n```") if len(top) else "(none)", ""]
(OUT / "SUMMARY.md").write_text("\n".join(md) + "\n")
print(f"wrote {OUT / 'SUMMARY.md'}")
