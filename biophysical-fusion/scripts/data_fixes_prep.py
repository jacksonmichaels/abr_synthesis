"""Precomputes the tables notebooks/analysis/data_fixes.ipynb reads, so the notebook
itself only touches a handful of files.

    python scripts/data_fixes_prep.py            # job: scripts/sbatch/build_data_fixes_notebook.sh

Writes results/analysis/data_fixes/:
  alignment_scan.csv   every aligned FASTA in every input folder: records, width,
                       insertion columns, how many of them no isolate fills
  vcf_records.tsv      raw VCF lines inside each curated window, for every isolate
                       whose eis/embB/ethA/rrs row changed in the repair, plus 50
                       unchanged isolates per locus (seed 0)
  widths.csv           per reference position of eis/embB/ethA/rrs: the width the
                       2026-10-22 bookkeeping gave it vs the width the cells need
  eis_regression.json  rebuild of eis from its pickled cells vs the Oct-22 file

Reads only; the VCFs and the labmate's FASTAs under /project are read-only.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W = Path("/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis")
OUT = W / "biophysical-fusion" / "results" / "analysis" / "data_fixes"
LAB = Path("/project/pi_annagreen_umass_edu/saishradha/project_data_curation")
VCF_LIST = LAB / "make_msa" / "path_list" / "big_tb_file_paths.txt"
REF_ID = "MT_H37Rv"
FOUR = ["eis", "embB", "ethA", "rrs"]
LAB14 = ["atpE", "ddn", "fbiA", "fbiB", "fbiC", "fgd1", "mmpL5", "mmpS5", "pepQ", "rplC",
         "Rv0678", "Rv1979c", "Rv2983", "tsnR"]

sys.path.insert(0, str(W / "regulatory_msa"))
from aln_widths import pad_cells, pad_reference, realized_widths  # noqa: E402

# (label, folder, file filter) -- the folder each set of model inputs was read from
FOLDERS = [
    ("curated 19, original (paper runs)", W / "data_snapshot/genomic_data/aligned", None),
    ("curated 19, repaired", W / "curated_fix/aligned19", None),
    ("who_t12 genes, original", W / "archive/2026-10-30/gene_msa/aligned", None),
    ("who_t12 genes, re-extracted", W / "gene_msa/aligned_fixed", None),
    ("promoters, TSS-anchored", W / "regulatory_msa/aligned", None),
    ("promoters, ATG-anchored", W / "regulatory_msa/aligned_atg", None),
    ("labmate's 14 extra loci", LAB / "genomic_data/aligned", set(LAB14)),
]


def read_fasta(path):
    ids, seqs, buf = [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if ids:
                    seqs.append("".join(buf))
                ids.append(line[1:].strip())
                buf = []
            else:
                buf.append(line.strip())
    if ids:
        seqs.append("".join(buf))
    return ids, seqs


def as_matrix(seqs):
    w = len(seqs[0])
    return np.frombuffer("".join(seqs).encode(), dtype="S1").reshape(len(seqs), w)


def scan():
    rows = []
    for label, d, only in FOLDERS:
        if not d.is_dir():
            print(f"[scan] absent: {d}", flush=True)
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".fasta")
                       and (only is None or f[:-6] in only))
        for f in files:
            ids, seqs = read_fasta(d / f)
            ref = seqs[ids.index(REF_ID)] if REF_ID in ids else None
            widths = {len(s) for s in seqs}
            row = {"folder": label, "locus": f[:-6], "n_records": len(ids),
                   "width": len(ref) if ref else -1, "ragged": len(widths) > 1,
                   "ins_cols": 0, "ins_unfilled": 0, "ins_filled_bases": 0}
            if ref and len(widths) == 1:
                M = as_matrix(seqs)
                iso = M[np.array([i != REF_ID for i in ids])]
                ins = np.frombuffer(ref.encode(), dtype="S1") == b"-"
                filled = iso[:, ins] != b"-"
                row.update(ins_cols=int(ins.sum()),
                           ins_unfilled=int((~filled.any(0)).sum()),
                           ins_filled_bases=int(filled.sum()))
            rows.append(row)
        print(f"[scan] {label}: {len(files)} files", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "alignment_scan.csv", index=False)


def changed_isolates():
    """-> {locus: [isolates whose ungapped row differs between the two files]}"""
    out = {}
    rng = np.random.default_rng(0)
    for g in FOUR:
        io, so = read_fasta(W / f"data_snapshot/genomic_data/aligned/{g}.fasta")
        i_n, sn = read_fasta(W / f"curated_fix/aligned/{g}.fasta")
        assert io == i_n, g
        ch = [k for k, a, b in zip(io, so, sn)
              if k != REF_ID and a.replace("-", "") != b.replace("-", "")]
        same = [k for k, a, b in zip(io, so, sn)
                if k != REF_ID and a.replace("-", "") == b.replace("-", "")]
        out[g] = (ch, list(rng.choice(same, 50, replace=False)))
        print(f"[changed] {g}: {len(ch)} changed", flush=True)
    return out


def vcf_cache(sel):
    win = json.load(open(W / "curated_fix/curated_windows.json"))["windows"]
    paths = {os.path.basename(p).replace("_variants.vcf", ""): p
             for p in pd.read_csv(VCF_LIST, sep="\t", header=None)[0]}
    rows = []
    for g, (ch, same) in sel.items():
        start, end = win[g]["start"], win[g]["end"]
        for iso in ch + same:
            n0 = len(rows)
            with open(paths[iso]) as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    f = line.rstrip("\n").split("\t")
                    pos, ref = int(f[1]), f[3]
                    if pos <= end and pos + len(ref) - 1 >= start:
                        rows.append({"locus": g, "isolate": iso, "changed": iso in ch,
                                     "POS": pos, "REF": ref, "ALT": f[4], "QUAL": f[5],
                                     "FILTER": f[6], "INFO": f[7]})
            # an isolate with no records in the window still needs a row
            if len(rows) == n0:
                rows.append({"locus": g, "isolate": iso, "changed": iso in ch, "POS": -1,
                             "REF": "", "ALT": "", "QUAL": "", "FILTER": "", "INFO": ""})
        print(f"[vcf] {g}: {len(ch) + len(same)} isolates read", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "vcf_records.tsv", sep="\t", index=False)


def widths_and_regression():
    rows = []
    for g in FOUR:
        sites = W / f"curated_fix/aligned/sites/{g}"
        df = pd.read_pickle(sites / f"{g}_seq_dict.pkl")
        n_pos = df.shape[0]
        book = np.ones(n_pos, int)
        ins = pd.read_csv(sites / f"{g}_insertion_sites.csv")
        book[ins["aln_idx"].values] += ins["len_insertion"].values
        real = realized_widths((df[c].values for c in df.columns), n_pos)
        # which isolate (first) needs the widest cell where the bookkeeping is short
        short = np.nonzero(np.array(real) > book)[0]
        who = {}
        for c in df.columns:
            cells = df[c].values
            for p in short:
                if p not in who and len(cells[p].replace("X", "")) == real[p]:
                    who[p] = c
        for p in range(n_pos):
            if book[p] > 1 or real[p] > 1:
                rows.append({"locus": g, "pos": p, "bookkeeping": int(book[p]),
                             "realized": int(real[p]), "widest_isolate": who.get(p, "")})
        if g == "eis":
            ids, seqs = read_fasta(W / "curated_fix/aligned/eis.fasta")
            old = dict(zip(ids, seqs))
            bases = old[REF_ID].replace("-", "")
            comp = str.maketrans("ACGTN-", "TGCAN-")
            rc = lambda s: s.translate(comp)[::-1]
            ref_aln = rc(pad_reference(rc(bases), real))
            same = int(ref_aln == old[REF_ID])
            for c in df.columns:
                same += rc(pad_cells(df[c].values, real)) == old[c]
            json.dump({"records": len(ids), "identical": same},
                      open(OUT / "eis_regression.json", "w"))
            print(f"[eis] {same} of {len(ids)} records identical", flush=True)
        del df
        print(f"[widths] {g}: {len(short)} positions where the cells outgrow the bookkeeping",
              flush=True)
    pd.DataFrame(rows).to_csv(OUT / "widths.csv", index=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if not VCF_LIST.exists():
        raise SystemExit(f"{VCF_LIST} not visible -- /project is not mounted on {os.uname()[1]}")
    widths_and_regression()
    vcf_cache(changed_isolates())
    scan()
