"""Call M. tuberculosis (sub)lineages for every isolate from the TB-Profiler SNP barcode.

    python scripts/call_lineages.py            # (sbatch scripts/sbatch/call_lineages.sh)

Barcode: results/experiments/lineage_split/ref/barcode.bed (jodyphelan/tbdb, commit
pinned in ref/SOURCE.txt; Napier et al. 2020). Each row is one marker SNP: BED start
(0-based), end (= 1-based H37Rv position), the (sub)lineage it defines, and the
allele that marks it.

Genotypes: the whole-genome LASSO's parsed cohort cache
(~/abr_workspace/wholegenome_cache.npz: CSR of PASS calls, columns keyed
"POS:REF>ALT", rows = all 17,942 isolates), so /project is never read. A marker is
carried when the isolate's PASS calls put the marker allele at that position; a
marker whose allele IS the H37Rv base is carried when the isolate has no PASS
substitution there (no-calls are indistinguishable from reference in this cache).

Calling: a label is present when >= --min-frac of its markers are carried. The
call is the deepest present label whose present ancestors form one chain
(lineage4 -> lineage4.3 -> lineage4.3.4 ...). Present labels on two branches
that are not ancestor/descendant (e.g. lineage2.2 and lineage4.3) mark the
isolate MIXED. No present label -> UNCALLED.

Writes results/experiments/lineage_split/:
  lineage_calls.csv        isolate, lineage (deepest), level1, level2, level3, status,
                           n_present, support (fraction of the call's markers carried)
  lineage_summary.csv      counts per level1 / level2
  lineage_by_drug.csv      level2 x drug x R/S counts (phenotyped isolates only)
  lineage_by_source.csv    level1 x phenotype `category` (data source) counts
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

PROJECT = Path(__file__).resolve().parent.parent
EXP = PROJECT / "results" / "experiments" / "lineage_split"
SNAP = Path("/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot")
DRUGS = ["AMIKACIN", "CAPREOMYCIN", "ETHAMBUTOL", "ETHIONAMIDE", "ISONIAZID", "KANAMYCIN",
         "LEVOFLOXACIN", "MOXIFLOXACIN", "PYRAZINAMIDE", "RIFAMPICIN", "STREPTOMYCIN"]


def parent(label):
    """lineage4.3.4 -> lineage4.3 -> lineage4 -> None (same for La1.8.1, etc.)."""
    head, _, tail = label.rpartition(".")
    if not head:
        return None
    # "lineage4" has no dot; "La1.8" -> "La1"; "M.canetti" is a leaf with no parent
    if re.fullmatch(r"[A-Za-z]+", head):
        return None
    return head


def ancestors(label):
    out, p = [], parent(label)
    while p:
        out.append(p)
        p = parent(p)
    return out


def level(label, k):
    """First k dotted components after the lineage root (lineage4.3.4, k=2 -> lineage4.3)."""
    if not label.startswith("lineage"):
        return label.split(".")[0] if k == 1 else ".".join(label.split(".")[:k])
    parts = label[len("lineage"):].split(".")
    return "lineage" + ".".join(parts[:k])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--barcode", default=str(EXP / "ref" / "barcode.bed"))
    ap.add_argument("--cache", default=str(Path.home() / "abr_workspace" / "wholegenome_cache.npz"))
    ap.add_argument("--genbank", default=str(SNAP / "genbank" / "genome.gbff"))
    ap.add_argument("--phenotype-csv", default=str(SNAP / "phenotype_data" / "master_resistance_table.csv"))
    ap.add_argument("--min-frac", type=float, default=0.5)
    ap.add_argument("--out", default=str(EXP))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ref = str(next(SeqIO.parse(args.genbank, "genbank")).seq).upper()
    bc = pd.read_csv(args.barcode, sep="\t", header=None,
                     names=["chrom", "start", "pos", "label", "allele", "family", "spoligo", "rd"])
    bc["ref"] = [ref[p - 1] for p in bc.pos]
    bc["is_ref_allele"] = bc.allele == bc.ref
    print(f"barcode: {len(bc)} markers, {bc.label.nunique()} labels, "
          f"{int(bc.is_ref_allele.sum())} markers carry the H37Rv base")
    want = set(bc.pos)

    z = np.load(args.cache, allow_pickle=False)
    keys, isolates = z["keys"], z["isolates"]
    indptr, indices = z["indptr"], z["indices"]
    print(f"cache: {len(isolates)} isolates, {len(keys)} keys, {len(indices)} nonzeros")

    # column -> list of (barcode position, alt base) it asserts; substitutions only
    col_hits = {}
    for j, k in enumerate(keys):
        pos_s, _, change = str(k).partition(":")
        r, _, a = change.partition(">")
        if not pos_s.isdigit() or len(r) != len(a) or not r:
            continue
        p0 = int(pos_s)
        hits = [(p0 + o, a[o]) for o in range(len(r))
                if r[o] != a[o] and (p0 + o) in want]
        if hits:
            col_hits[j] = hits
    print(f"{len(col_hits)} cohort columns touch a barcode position")

    # per isolate: {position: alt base} at barcode positions
    hit_cols = np.fromiter(col_hits.keys(), dtype=np.int64)
    is_hit = np.zeros(len(keys), dtype=bool)
    is_hit[hit_cols] = True
    markers = list(zip(bc.pos, bc.allele, bc.is_ref_allele, bc.label))
    by_label = defaultdict(list)
    for i, (_, _, _, lab) in enumerate(markers):
        by_label[lab].append(i)
    labels = sorted(by_label)

    rows = []
    for r in range(len(isolates)):
        cols = indices[indptr[r]:indptr[r + 1]]
        alt = {}
        for j in cols[is_hit[cols]]:
            for p, b in col_hits[int(j)]:
                alt[p] = b
        carried = np.array([(p not in alt) if isref else (alt.get(p) == a)
                            for p, a, isref, _ in markers])
        frac = {lab: carried[by_label[lab]].mean() for lab in labels}
        present = [lab for lab in labels if frac[lab] >= args.min_frac]
        if not present:
            rows.append(dict(isolate=str(isolates[r]), lineage="UNCALLED", status="uncalled",
                             n_present=0, support=np.nan))
            continue
        # leaves of the present set: present labels with no present descendant
        pset = set(present)
        leaves = [l for l in present
                  if not any(l in ancestors(m) for m in pset if m != l)]
        status = "ok" if len(leaves) == 1 else "mixed"
        call = max(leaves, key=lambda l: (len(ancestors(l)), frac[l]))
        rows.append(dict(isolate=str(isolates[r]), lineage=call, status=status,
                         n_present=len(present), support=frac[call],
                         other_leaves=";".join(sorted(set(leaves) - {call}))))
    calls = pd.DataFrame(rows)
    ok = calls.lineage != "UNCALLED"
    for k in (1, 2, 3):
        calls[f"level{k}"] = np.where(ok, [level(l, k) for l in calls.lineage], "UNCALLED")
    calls.to_csv(out / "lineage_calls.csv", index=False)

    summ = pd.concat({
        "level1": calls.groupby(["level1", "status"]).size().unstack(fill_value=0),
        "level2": calls.groupby(["level2", "status"]).size().unstack(fill_value=0)},
        names=["level", "label"])
    summ.to_csv(out / "lineage_summary.csv")
    print(calls.status.value_counts().to_string())
    print(calls.groupby("level1").size().sort_values(ascending=False).to_string())

    ph = pd.read_csv(args.phenotype_csv, low_memory=False).set_index("New_ID")
    j = calls.set_index("isolate").join(ph, how="inner")
    print(f"{len(j)} of {len(calls)} called isolates join the phenotype table on New_ID")
    recs = []
    for d in DRUGS:
        s = j[j[d].isin(["R", "S"])]
        t = s.groupby(["level2", d]).size().unstack(fill_value=0)
        for g, rr in t.iterrows():
            recs.append(dict(drug=d, level2=g, R=int(rr.get("R", 0)), S=int(rr.get("S", 0))))
    pd.DataFrame(recs).to_csv(out / "lineage_by_drug.csv", index=False)
    j.groupby(["level1", "category"]).size().unstack(fill_value=0).to_csv(out / "lineage_by_source.csv")
    print(f"wrote {out}/lineage_calls.csv, lineage_summary.csv, lineage_by_drug.csv, lineage_by_source.csv")


if __name__ == "__main__":
    main()
