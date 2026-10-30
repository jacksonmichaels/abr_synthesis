#!/usr/bin/env python3
"""
Rebuild an aligned FASTA from the cells `make_aligned_msa.py` already pickled,
without re-reading a single VCF.

Step 1 of the extractor (reading ~18k VCFs over /project, the slow part) writes
`<out>/sites/<GENE>/<gene>_seq_dict.pkl`. Steps 2-3 then pad those cells to a
common width -- and that is where the embB / ethA / rrs re-extractions died
(see aln_widths.py). This redoes only steps 2-3, with the width taken from the
cells, so a failed build is minutes of CPU instead of hours of VCF reading.

The reference row comes from an EXISTING alignment of the same window (its
H37Rv row, ungapped), so the rebuilt file covers exactly the same genome span
as the alignment it replaces -- which is what keeps `datasets.cds.block_frame`
and every locus definition downstream unchanged.

    python rebuild_alignment.py --gene eis \
        --pickle  /scratch3/.../curated_fix/aligned/sites/eis/eis_seq_dict.pkl \
        --ref-fasta /scratch3/.../data_snapshot/genomic_data/aligned/eis.fasta \
        --sense neg --out /scratch3/.../curated_fix/aligned/eis.fasta

    python rebuild_alignment.py ... --check-against <existing fasta>   # regression
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aln_widths import pad_cells, pad_reference, realized_widths  # noqa: E402

COMPLEMENT = str.maketrans("ACGTacgtNn-", "TGCAtgcaNn-")


def reverse_complement(s):
    return s.translate(COMPLEMENT)[::-1]


def read_fasta(path):
    recs, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    recs[name] = "".join(buf)
                name, buf = line[1:].strip(), []
            else:
                buf.append(line)
    if name is not None:
        recs[name] = "".join(buf)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", required=True)
    ap.add_argument("--pickle", required=True, help="<gene>_seq_dict.pkl from step 1")
    ap.add_argument("--ref-fasta", required=True,
                    help="an existing alignment of the same window; its H37Rv row sets the span")
    ap.add_argument("--sense", choices=["pos", "neg"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-id", default="MT_H37Rv")
    ap.add_argument("--check-against", default=None,
                    help="compare the rebuilt rows to this alignment and report")
    args = ap.parse_args()

    ref_row = read_fasta(args.ref_fasta)[args.ref_id]
    bases = ref_row.replace("-", "")
    # the cells are in GENOME orientation; a neg-sense alignment is its reverse complement
    if args.sense == "neg":
        bases = reverse_complement(bases)

    df = pd.read_pickle(args.pickle)
    n_pos, n_iso = df.shape
    print(f"{args.gene}: {n_iso} isolates x {n_pos} reference positions", flush=True)
    if n_pos != len(bases):
        raise SystemExit(f"{n_pos} positions but the reference window is {len(bases)} bp")

    cols = list(df.columns)
    # two passes straight off the frame: materializing every cell list at once
    # costs several GB on a locus the size of embB
    widths = realized_widths((df[iso].values for iso in cols), n_pos)
    extra = sum(w - 1 for w in widths)
    print(f"{args.gene}: {sum(1 for w in widths if w > 1)} positions take extra columns, "
          f"{extra} extra in all -> width {sum(widths)}", flush=True)

    ref_aln = pad_reference(bases, widths)
    if args.sense == "neg":
        ref_aln = reverse_complement(ref_aln)

    tmp = args.out + f".{os.getpid()}.tmp"
    n = 0
    with open(tmp, "w") as fh:
        fh.write(f">{args.ref_id}\n{ref_aln}\n")
        for iso in cols:
            row = pad_cells(df[iso].values, widths)
            if len(row) != len(ref_aln):
                raise SystemExit(f"{iso}: {len(row)} != reference {len(ref_aln)}")
            if args.sense == "neg":
                row = reverse_complement(row)
            fh.write(f">{iso}\n{row}\n")
            n += 1
    os.replace(tmp, args.out)
    print(f"{args.gene}: wrote {n + 1} records ({n} isolates + {args.ref_id}) to {args.out}",
          flush=True)

    if args.check_against:
        old = read_fasta(args.check_against)
        new = read_fasta(args.out)
        shared = [k for k in new if k in old]
        same = sum(1 for k in shared if old[k] == new[k])
        ungapped = sum(1 for k in shared
                       if old[k].replace("-", "").upper() == new[k].replace("-", "").upper())
        print(f"{args.gene}: vs {os.path.basename(args.check_against)} -- "
              f"{len(shared)} shared records, {same} identical, "
              f"{ungapped} identical ignoring gaps", flush=True)


if __name__ == "__main__":
    main()
