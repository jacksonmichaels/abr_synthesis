#!/usr/bin/env python3
"""Print exactly what each lasso arm is given, at one real site, for real isolates.

    python results/experiments/lasso_pooled_20260929/show_design.py [--drug PYRAZINAMIDE --locus pncA]

Builds the columns with the SAME functions the runs used -- lasso_wholeseq's
_locus_block for the alignment arms, lasso_wholegenome's build_cohort /
drug_design / pooled_features for the VCF arms -- then picks the position in the
locus with the most distinct alleles in the drug's cohort and prints, for a
handful of isolates (one carrier per allele, a wild type, a deletion over the
site, an uncalled base if any), the value of every column that touches that site.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "results" / "experiments" / "lasso_wholeseq_20260901"))
sys.path.insert(0, str(PROJECT / "results" / "experiments" / "lasso_wholegenome_20260903"))

import numpy as np  # noqa: E402

import lasso_wholeseq as ws  # noqa: E402
import lasso_wholegenome as lw  # noqa: E402
from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV  # noqa: E402
from datasets.cds import genome_span  # noqa: E402
from datasets.sequences import load_phenotype, load_sequence_df, numeric_labels  # noqa: E402

COMP = str.maketrans("ACGT", "TGCA")

ap = argparse.ArgumentParser()
ap.add_argument("--drug", default="PYRAZINAMIDE")
ap.add_argument("--locus", default="pncA")
ap.add_argument("--gene-id", default="Rv2043c",
                help="pick the site inside this snpEff Gene_ID (pncA's CDS is Rv2043c)")
ap.add_argument("--cache", default=str(Path.home() / "abr_workspace" / "wholegenome_cache.npz"))
ap.add_argument("--annot-cache", default=str(Path.home() / "abr_workspace" / "wholegenome_annot.npz"))
a = ap.parse_args()

# ---------------------------------------------------------------- cohort + labels
X_all, iso_all, keys_all, gene_of, geneid_of = lw.build_cohort(lw.VCF_TSV_DIR, a.cache, 2)
_ann, imp = lw.build_annotation(lw.VCF_TSV_DIR, keys_all, a.annot_cache, 2)
impact_of = dict(zip(keys_all, imp))
row_of = {iso: i for i, iso in enumerate(iso_all)}
ph = load_phenotype(REAL_PHENOTYPE_CSV)
isos = [i for i in ph.index if i in row_of]
y = numeric_labels(ph.loc[isos], a.drug)
keep = np.nonzero(y != -1)[0]
isos, y = [isos[i] for i in keep], y[keep]
label = {iso: ("R" if v == 0 else "S") for iso, v in zip(isos, y)}

# The real whole-genome designs for this drug: per-key columns, then pooled.
Xk, names = lw.drug_design(X_all, iso_all, keys_all, [row_of[i] for i in isos], 1)
Xp, pnames = lw.pooled_features(Xk, names, "both", geneid_of, impact_of)
col_p = {n: j for j, n in enumerate(pnames)}
print(f"\n{a.drug}: {len(isos)} isolates")
print(f"  whole genome, delta (per key)     : {Xk.shape[1]:>9,} columns")
print(f"  whole genome, pooled 'both'       : {Xp.shape[1]:>9,} columns "
      f"(= {Xk.shape[1]:,} keys + {Xp.shape[1] - Xk.shape[1]:,} pooled)")

# ---------------------------------------------------------------- pick the site
start, end, strand = genome_span(REAL_GENOTYPE_DIR, a.locus)
spans = {}
for j, k in enumerate(names):
    p0, p1 = lw._key_span(k)
    if p0 < end and p1 > start:
        spans[k] = (j, p0, p1)
by_pos = {}
for k, (j, p0, p1) in spans.items():
    if (p1 - p0 == 1 and len(k.split(">")[1]) == 1           # SNVs, to pick the site
            and geneid_of.get(k) == a.gene_id):
        by_pos.setdefault(p0, []).append(k)
pos0 = max(by_pos, key=lambda p: len(by_pos[p]))
POS = pos0 + 1
print(f"\nsite: H37Rv position {POS:,} (VCF POS, 1-based), {a.locus} span "
      f"{start + 1:,}-{end:,} on the {strand} strand; {len(by_pos[pos0])} distinct SNV alleles here")

# Every per-key column that overlaps the site (SNVs AND indels covering it).
over = sorted((k for k, (j, p0, p1) in spans.items() if p0 <= pos0 < p1),
              key=lambda k: (len(k), k))
gid = a.gene_id

# ---------------------------------------------------------------- alignment arms
seq_df, _ = load_sequence_df(REAL_GENOTYPE_DIR, [a.locus])
seq_df = seq_df.reindex(isos)
seqs = seq_df[a.locus].fillna("").tolist()
ref_row = ws.reference_row(REAL_GENOTYPE_DIR, a.locus)
# alignment column holding this genome base: the i-th non-gap column of the H37Rv
# row, i counted 5'->3' along the record ('-' strand records are reverse-complemented)
i = (end - 1 - pos0) if strand == "-" else (pos0 - start)
nongap = [c for c, ch in enumerate(ref_row) if ch != "-"]
acol = nongap[i]
ref_base_aln = ref_row[acol]
Xo, no = ws._locus_block(REAL_GENOTYPE_DIR, a.locus, seqs, "onehot", 1)
Xd, nd = ws._locus_block(REAL_GENOTYPE_DIR, a.locus, seqs, "delta", 1)
oh_cols = [(n, j) for j, n in enumerate(no) if n.startswith(f"{a.locus}:{acol}:")]
de_cols = [(n, j) for j, n in enumerate(nd) if n.startswith(f"{a.locus}:{acol}:")]
print(f"alignment: {a.locus}.fasta column {acol} (0-based); H37Rv base there '{ref_base_aln}'"
      f"{' = complement of the + strand base' if strand == '-' else ''}")

# ---------------------------------------------------------------- pick isolates
irow = {iso: r for r, iso in enumerate(isos)}
Xk_csr = Xk.tocsr()


def keys_of(iso):
    r = Xk_csr[irow[iso]]
    return {names[j] for j in r.indices}


chosen = []
for k in over:                                     # one carrier per overlapping key
    j = spans[k][0]
    rows = Xk_csr[:, j].nonzero()[0]
    if len(rows):
        chosen.append((f"carries {k}", isos[rows[0]]))
# two carriers of DIFFERENT high-impact keys elsewhere in the gene, to show what
# lof: pools across sites
_high = [k for k in names if geneid_of.get(k) == gid and impact_of.get(k) == "HIGH"
         and k not in over]
for k in _high[:2]:
    rows = Xk_csr[:, spans[k][0] if k in spans else names.index(k)].nonzero()[0]
    if len(rows):
        chosen.append((f"carries {k} elsewhere in {gid} [HIGH: {_ann[keys_all.index(k)]}]",
                       isos[rows[0]]))
for r, iso in enumerate(isos):                     # a wild type: no key over the site
    if not (keys_of(iso) & set(over)) and seqs[r] and seqs[r][acol] == ref_base_aln:
        chosen.append(("wild type here", iso)); break
for r, iso in enumerate(isos):                     # an uncalled base in the alignment
    if seqs[r] and len(seqs[r]) > acol and seqs[r][acol] not in "ACGT-":
        chosen.append((f"alignment has '{seqs[r][acol]}' here", iso)); break
chosen = chosen[:10]

# ---------------------------------------------------------------- print
lof, nons = f"lof:{gid}", f"nonsyn:{gid}"


def v(M, r, j):
    return int(M[r, j])


print("\n" + "=" * 100)
for why, iso in chosen:
    r = irow[iso]
    s = seqs[r]
    print(f"\nisolate {iso}  ({a.drug}: {label[iso]})  -- {why}")
    print(f"  alignment char at column {acol}: '{s[acol] if len(s) > acol else ''}'")
    print("  [onehot · 19 loci]   " + "  ".join(f"{n}={v(Xo, r, j)}" for n, j in oh_cols))
    print("  [delta  · 19 loci]   " + "  ".join(f"{n}={v(Xd, r, j)}" for n, j in de_cols))
    ks = keys_of(iso)
    print("  [delta · VCF]        " + "  ".join(f"{k}={int(k in ks)}" for k in over))
    sc = f"site:{POS}"
    extra = [f"{sc}={v(Xp, r, col_p[sc])}" if sc in col_p else f"{sc}=(not a column)"]
    for g in (lof, nons):
        extra.append(f"{g}={v(Xp, r, col_p[g])}" if g in col_p else f"{g}=(not a column)")
    print("  [pooled columns]     " + "  ".join(extra))
    fed = sorted(k for k in ks if geneid_of.get(k) == gid and impact_of.get(k) in ("HIGH", "MODERATE"))
    print(f"  {gid} HIGH/MODERATE keys this isolate carries (feed {lof}/{nons}): "
          + (", ".join(f"{k} [{impact_of[k]}]" for k in fed) or "none"))

# group sizes for the pooled columns at this site / gene
for g in (f"site:{POS}", lof, nons):
    if g in col_p:
        print(f"\n{g}: set in {int(Xp[:, col_p[g]].sum()):,} of {len(isos):,} isolates")
members = [k for k in names if geneid_of.get(k) == gid]
print(f"{gid}: {len(members)} distinct keys in this cohort; "
      f"{sum(impact_of.get(k) == 'HIGH' for k in members)} HIGH, "
      f"{sum(impact_of.get(k) == 'MODERATE' for k in members)} MODERATE")
