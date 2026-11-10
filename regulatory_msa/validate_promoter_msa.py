#!/usr/bin/env python3
"""
Validate extracted promoter-window FASTAs against their window JSON, the H37Rv
reference and the WHO 2023 variant table. Read-only; prints one row per region
and exits non-zero if any hard check fails.

Hard checks (per region):
  ids     record IDs == the isolate IDs of a reference FASTA (default: the
          curated katG alignment), plus the MT_H37Rv row
  width   every record has the same aligned width
  ref     MT_H37Rv row, gaps removed, == the genome slice [start, end]
          (reverse-complemented for neg-sense windows)
  ins     if the alignment has insertion columns, some isolate must fill them
          (all-gap insertion columns are the signature of the Jul-2025 upstream
          overwrite bug fixed in make_aligned_msa.py on 2026-10-22)
  vcf     up to --vcf-k isolates carrying an insertion, rebuilt from H37Rv +
          their own VCF, must equal their FASTA row (gaps removed; N in the
          FASTA matches anything, since the extractor writes low-confidence
          calls as N). Only isolates whose in-window records are all PASS 1/1
          and fully inside the window are scored.

Reported only (warnings; the 2026-07/08 compare build has the insertion bug,
so it legitimately differs wherever an isolate carries an insertion):
  same    windows identical to the --compare build: isolates that differ
  snps    for each WHO promoter SNP in the window, the number of isolates
          carrying the alt base, and whether counts in the region shared with
          the compare build's window agree.

    python validate_promoter_msa.py                      # every FASTA in aligned_atg/
    python validate_promoter_msa.py --genes katG
    python validate_promoter_msa.py --windows promoter_windows.json --dir aligned --no-compare
"""
import argparse
import json
import os
import re
import sys

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

HERE = os.path.dirname(os.path.abspath(__file__))
GENBANK = ("/project/pi_annagreen_umass_edu/saishradha/project_data_curation/"
           "make_msa/genbank_reference_data/genome.gbff")
WHO_VARIANTS = ("/project/pi_annagreen_umass_edu/saishradha/project_data_curation/"
                "make_vcf-who_map/data/WHO_variants_data_2023/WHO_resistance_variants_all_2023.csv")
VCF_PATHS = ("/project/pi_annagreen_umass_edu/saishradha/project_data_curation/"
             "make_msa/path_list/big_tb_file_paths.txt")
REF_ID = "MT_H37Rv"
COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def read_fasta(path):
    out, k, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if k is not None:
                    out[k] = "".join(buf)
                k, buf = line[1:].split()[0], []
            else:
                buf.append(line)
    if k is not None:
        out[k] = "".join(buf)
    return out


def read_ids(path):
    with open(path) as fh:
        return {l[1:].split()[0] for l in fh if l.startswith(">")}


def column_of(w, ref_row):
    """genome position -> alignment column, from the H37Rv row's non-gap bases."""
    cols = [i for i, c in enumerate(ref_row) if c != "-"]
    pos = list(range(w["start"], w["end"] + 1))
    if w["sense"] == "neg":
        pos = pos[::-1]
    return dict(zip(pos, cols))


def promoter_snps(variants, gene, w):
    d = variants[(variants["gene"] == gene)
                 & variants["variant"].str.contains(r"_[cn]\.-\d+[ACGT]>[ACGT]$", regex=True)
                 & (variants["ref_nt"].str.len() == 1) & (variants["alt_nt"].str.len() == 1)]
    d = d[(d["genome_index"] >= w["start"]) & (d["genome_index"] <= w["end"])]
    return d.drop_duplicates(["variant"])


def snp_counts(seqs, w, snps):
    cmap = column_of(w, seqs[REF_ID])
    out = {}
    for _, r in snps.iterrows():
        pos = int(r["genome_index"])
        alt = r["alt_nt"] if w["sense"] == "pos" else r["alt_nt"].translate(COMP)
        col = cmap[pos]
        out[r["variant"]] = sum(1 for k, s in seqs.items() if k != REF_ID and s[col] == alt)
    return out


def vcf_paths():
    out = {}
    with open(VCF_PATHS) as fh:
        for line in fh:
            p = line.strip()
            out[os.path.basename(p).replace(".eff", "").replace(".vcf", "").replace("_variants", "")] = p
    return out


def vcf_spot_check(seqs, w, genome, paths, k):
    """(scored, mismatched isolate IDs) for up to k insertion carriers."""
    ref = seqs[REF_ID]
    cols = [i for i, c in enumerate(ref) if c == "-"]
    carriers = [s for s in seqs if s != REF_ID and any(seqs[s][i] != "-" for i in cols)]
    win = genome[w["start"] - 1:w["end"]]
    scored, bad = 0, []
    for iso in carriers:
        if scored >= k:
            break
        recs, clean = [], True
        with open(paths[iso]) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                pos, r, a = int(f[1]), f[3], f[4]
                last = pos + len(r) - 1
                if last < w["start"] or pos > w["end"]:
                    continue
                if (pos < w["start"] or last > w["end"] or f[6] != "PASS"
                        or not f[9].startswith("1/1") or "," in a):
                    clean = False
                    break
                recs.append((pos, r, a))
        if not clean or not recs:
            continue
        s = win
        for pos, r, a in sorted(recs, reverse=True):
            i = pos - w["start"]
            s = s[:i] + a + s[i + len(r):]
        if w["sense"] == "neg":
            s = str(Seq(s).reverse_complement())
        got = seqs[iso].replace("-", "")
        scored += 1
        if len(got) != len(s) or any(x != y and x != "N" for x, y in zip(got, s)):
            bad.append(iso)
    return scored, bad


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows", default=os.path.join(HERE, "promoter_windows_atg.json"))
    p.add_argument("--dir", default=os.path.join(HERE, "aligned_atg"))
    p.add_argument("--compare-windows", default=os.path.join(HERE, "promoter_windows.json"))
    p.add_argument("--compare-dir", default=os.path.join(HERE, "aligned"))
    p.add_argument("--no-compare", action="store_true")
    p.add_argument("--id-ref", default="/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/"
                                       "data_snapshot/genomic_data/aligned/katG.fasta")
    p.add_argument("--genes", nargs="+")
    p.add_argument("--vcf-k", type=int, default=10, help="insertion carriers to rebuild from VCF per region")
    a = p.parse_args()
    paths = vcf_paths()

    windows = json.load(open(a.windows))["windows"]
    cmp_windows = {} if a.no_compare else json.load(open(a.compare_windows))["windows"]
    genes = a.genes or sorted(g for g in windows if os.path.isfile(os.path.join(a.dir, f"{g}.fasta")))
    genome = str(next(SeqIO.parse(GENBANK, "genbank")).seq).upper()
    iso_ids = read_ids(a.id_ref) - {REF_ID}
    variants = pd.read_csv(WHO_VARIANTS).dropna(subset=["genome_index"])
    variants["genome_index"] = variants["genome_index"].astype(int)

    failed = []
    for g in genes:
        w = windows[g]
        seqs = read_fasta(os.path.join(a.dir, f"{g}.fasta"))
        errs = []
        ids = set(seqs) - {REF_ID}
        if REF_ID not in seqs:
            errs.append("no MT_H37Rv row")
        if ids != iso_ids:
            errs.append(f"ids: {len(ids - iso_ids)} extra, {len(iso_ids - ids)} missing")
        widths = {len(s) for s in seqs.values()}
        if len(widths) != 1:
            errs.append(f"widths {sorted(widths)[:5]}")
        expect = genome[w["start"] - 1:w["end"]]
        if w["sense"] == "neg":
            expect = str(Seq(expect).reverse_complement())
        if REF_ID in seqs and seqs[REF_ID].replace("-", "").upper() != expect:
            errs.append("ref row != genome slice")
        vcf_txt = ""
        if not errs:
            ref = seqs[REF_ID]
            cols = [i for i, c in enumerate(ref) if c == "-"]
            if cols and not any(s[i] != "-" for k, s in seqs.items() if k != REF_ID for i in cols):
                errs.append(f"{len(cols)} insertion columns never filled (overwrite bug)")
            elif cols:
                scored, bad = vcf_spot_check(seqs, w, genome, paths, a.vcf_k)
                vcf_txt = f"vcf {scored - len(bad)}/{scored} insertion carriers match"
                if bad:
                    errs.append(f"vcf mismatch {bad[:3]}")
            else:
                vcf_txt = "no insertions"

        note, snp_txt = "", ""
        cw = cmp_windows.get(g)
        cpath = os.path.join(a.compare_dir, f"{g}.fasta")
        cseqs = read_fasta(cpath) if cw and os.path.isfile(cpath) else None
        if not errs:
            snps = promoter_snps(variants, w.get("who_region_from", g), w)
            counts = snp_counts(seqs, w, snps)
            n_carriers = sum(counts.values())
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:2]
            snp_txt = f"{len(counts)} WHO SNPs, {n_carriers} carrier calls; top {top}"
            if cseqs is not None:
                same_win = (cw["start"], cw["end"], cw["sense"]) == (w["start"], w["end"], w["sense"])
                if same_win:
                    ndiff = sum(cseqs.get(k, "").replace("-", "") != s.replace("-", "")
                                for k, s in seqs.items())
                    note = (f"window unchanged; {ndiff} isolates differ (ungapped) from compare build"
                            if ndiff else "window unchanged, sequences identical")
                else:
                    shared = promoter_snps(variants, w.get("who_region_from", g), cw)
                    shared = shared[shared["variant"].isin(counts)]
                    old = snp_counts(cseqs, cw, shared)
                    bad = {v: (counts[v], c) for v, c in old.items() if counts[v] != c}
                    note = (f"window moved; {len(old)} shared WHO SNPs, counts "
                            + ("match" if not bad else f"WARN differ {list(bad.items())[:2]}"))
        status = "FAIL " + "; ".join(errs) if errs else "ok"
        print(f"{g:9s} n={len(ids):5d} width={sorted(widths)[0]:5d}  {status}  | {vcf_txt} | {note} | {snp_txt}")
        if errs:
            failed.append(g)
    print(f"\n{len(genes) - len(failed)}/{len(genes)} regions pass" + (f"; FAILED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
