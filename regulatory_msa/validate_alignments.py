#!/usr/bin/env python3
"""
Audit aligned FASTA directories for the two ways a build goes wrong silently.

1. TRUNCATION -- fewer records than the cohort. `datasets.sequences.load_sequence_df`
   reindexes on the phenotype cohort and fills what is missing with "", and an
   empty sequence tokenizes as "no variants", i.e. the isolate silently reads as
   H37Rv. A job that died halfway through writing a FASTA therefore does not
   fail a run; it quietly removes variants from it. embB / ethA / rrs sat in
   curated_fix/aligned/ like that from 2026-10-22 to 2026-10-28.

2. THE INSERTION BUG -- every insertion column all-GAP, the signature of the
   Jul-2025 extractor that overwrote downstream reference bases instead of
   inserting them (see aln_widths.py). A column no isolate fills is legitimate
   on its own; ALL of them unfilled is the bug. An `N` in the column is a
   low-confidence insertion call, i.e. the column IS filled.

Read-only. Exits non-zero if any directory has a hard failure.

    python validate_alignments.py                      # every known directory
    python validate_alignments.py --dirs <dir> [...]    # just these
    python validate_alignments.py --expect 17943        # required record count
"""
import argparse
import os
import sys

WORKSPACE = "/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis"
HOME_WS = "/home/jacksonmicha_umass_edu/abr_workspace"
KNOWN = [
    f"{WORKSPACE}/data_snapshot/genomic_data/aligned",      # the curated 19, every paper run
    f"{WORKSPACE}/curated_fix/aligned",                     # the 4 rebuilt
    f"{WORKSPACE}/curated_fix/aligned19",                   # 15 + 4, the repaired set
    f"{HOME_WS}/regulatory_msa/aligned",                    # old TSS-anchored promoters
    f"{HOME_WS}/regulatory_msa/aligned_atg",                # start-codon-anchored promoters
    f"{WORKSPACE}/archive/2026-10-30/gene_msa/aligned",     # who_t12, pre-fix (archived)
    f"{WORKSPACE}/gene_msa/aligned_fixed",                  # who_t12, re-extracted
]
REF_ID = "MT_H37Rv"


def scan(path):
    """-> (n_records, ref_row, widths_vary, per-record widths equal?) in one pass."""
    n, ref, name, width_set, buf = 0, None, None, set(), []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    seq = "".join(buf)
                    width_set.add(len(seq))
                    if name == REF_ID:
                        ref = seq
                    n += 1
                name, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
    if name is not None:
        seq = "".join(buf)
        width_set.add(len(seq))
        if name == REF_ID:
            ref = seq
        n += 1
    return n, ref, width_set


def insertion_report(path, ref):
    """(insertion columns, how many no isolate fills) -- streamed, one row at a time."""
    ins = [i for i, c in enumerate(ref) if c == "-"]
    if not ins:
        return 0, 0
    filled = [False] * len(ins)
    with open(path) as fh:
        name, buf = None, []

        def consume():
            if name is None or name == REF_ID:
                return
            seq = "".join(buf)
            for k, col in enumerate(ins):
                # the bug leaves the column ALL GAP; an N there is a real
                # low-confidence insertion call, so it counts as filled
                if not filled[k] and col < len(seq) and seq[col] != "-":
                    filled[k] = True

        for line in fh:
            if line.startswith(">"):
                consume()
                name, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
        consume()
    return len(ins), sum(1 for f in filled if not f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=None)
    ap.add_argument("--expect", type=int, default=17943,
                    help="required record count (17,942 isolates + MT_H37Rv)")
    args = ap.parse_args()

    bad = 0
    for d in (args.dirs or KNOWN):
        if not os.path.isdir(d):
            print(f"\n== {d}\n   (absent)")
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith((".fasta", ".fa")))
        print(f"\n== {d}  ({len(files)} files)")
        soft = []
        for f in files:
            p = os.path.join(d, f)
            n, ref, widths = scan(p)
            if ref is None:
                print(f"   {f:28s} FAIL no {REF_ID} row")
                bad += 1
                continue
            n_ins, unfilled = insertion_report(p, ref)
            notes = []
            if n != args.expect:
                notes.append(f"TRUNCATED {n} records (want {args.expect})")
            if len(widths) != 1:
                notes.append(f"RAGGED widths {sorted(widths)[:4]}")
            # one or two unfilled columns can be legitimate (the only carrier was
            # filtered out); every column of a sizeable set unfilled is the bug
            if n_ins >= 5 and unfilled == n_ins:
                notes.append(f"INSERTION BUG all {n_ins} insertion columns unfilled")
            elif n_ins and unfilled == n_ins:
                soft.append(f"{f} ({n_ins} unfilled)")
            print(f"   {f:28s} n={n:6d} cols={len(ref):6d} ins={n_ins:5d} unfilled={unfilled:5d}"
                  f"  {'; '.join(notes) if notes else 'ok'}")
            bad += bool(notes)
        if soft:
            print(f"   (few-column insertion sets nothing fills, probably legitimate: "
                  f"{', '.join(soft)})")
    print(f"\n{bad} file(s) with a hard failure")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
