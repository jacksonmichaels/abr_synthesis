"""Builds notebooks/analysis/data_fixes.ipynb -- what was wrong with the input
alignments, mechanically, and the evidence that the repaired files are right.

    python scripts/data_fixes_prep.py      # once: results/analysis/data_fixes/
    python scripts/build_data_fixes.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/analysis/data_fixes.ipynb

Job: scripts/sbatch/build_data_fixes_notebook.sh (does all three). Trains nothing.
"""
import json
from pathlib import Path

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": t.strip("\n").splitlines(keepends=True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})


# =============================================================================
md(r"""
# Input-data fixes: what broke, the mechanism, and why the repaired files are right

The models never read a VCF. They read one **aligned FASTA per locus**: a row per
isolate, every row the same width, with the columns lined up against H37Rv. This
notebook covers the four things that went wrong in building those files, between
July and October 2026.

| # | problem | which files | fix |
|---|---|---|---|
| 1 | **insertion bug**: an inserted allele was written *over* the bases after it instead of being inserted | curated eis/embB/ethA/rrs (DNA branch of every paper run), all who_t12 gene files, all promoter files, the labmate's Oct-2025 loci | extractor line fixed 2026-10-22; files re-extracted |
| 2 | **width bookkeeping**: the first fix crashed on embB/ethA/rrs when variants were nested | the 2026-10-22 re-extraction | widths taken from the data, 2026-10-28 (`aln_widths.py`) |
| 3 | **truncated files**: jobs that died mid-write left short FASTAs, and the loader accepted them silently | 3 files plus embB/ethA/rrs between Oct 22 and Oct 28 | rebuilt; the tokenizer now refuses short files |
| 4 | **promoter windows** placed relative to the TSS when WHO counts from the start codon | every promoter (regulatory) file | windows rebuilt from the ATG (`aligned_atg/`) |
| 5 | **wrong windows in the labmate's set**: `pepQ.fasta` holds a different gene; `mmpS5` is shifted by 100 bp | lab64 locus set only | re-extraction from the VCFs (running) |

Each section gives the mechanism on a toy example that runs the real code
lines, then shows it on a real isolate, then gives the check that the new file
is right. Section 6 lists which results were trained on which version.
""")

code(r"""
import json
import sys
import tempfile
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from IPython.display import display
from Bio import SeqIO
from Bio.Seq import Seq

PROJECT = next(p for p in [Path.cwd(), *Path.cwd().parents,
                           Path("/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion")]
               if (p / "paper_style.py").exists())
W = PROJECT.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(W / "regulatory_msa"))
from paper_style import MODEL, MODALITY, NEUTRAL, apply
from aln_widths import pad_cells, pad_reference, realized_widths   # the repaired extractor's own padding
apply()
pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_colwidth", 80)

PREP = PROJECT / "results" / "analysis" / "data_fixes"
OLD_DIR = W / "data_snapshot" / "genomic_data" / "aligned"   # what every paper run read
NEW_DIR = W / "curated_fix" / "aligned"                      # the 4 rebuilt loci
WIN = json.load(open(W / "curated_fix" / "curated_windows.json"))["windows"]
REF_ID = "MT_H37Rv"
GENOME = SeqIO.read(W / "data_snapshot" / "genbank" / "genome.gbff", "genbank")
G = str(GENOME.seq)
COMP = str.maketrans("ACGTNacgtn-", "TGCANtgcan-")
rc = lambda s: s.translate(COMP)[::-1]

# cell status colours, taken from the paper palette (this is a diagnostic notebook,
# so these do not carry their paper meanings)
STATUS = {"same as H37Rv": NEUTRAL["grid"], "gap": NEUTRAL["surface"],
          "inserted base": MODEL["lasso"], "differs from H37Rv": MODEL["locusfusion"],
          "N (uncalled)": MODALITY["uncalled"]}


def read_fasta(path, limit=None):
    out, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(buf)
                    if limit and len(out) >= limit:
                        return out
                name, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
    if name is not None:
        out[name] = "".join(buf)
    return out


def status_row(row, ref):
    out = []
    for ch, r in zip(row, ref):
        if ch == "-":
            out.append("gap")
        elif ch == "N":
            out.append("N (uncalled)")
        elif r == "-":
            out.append("inserted base")
        elif ch != r:
            out.append("differs from H37Rv")
        else:
            out.append("same as H37Rv")
    return out


def draw_rows(ax, rows, ref=None, fontsize=10):
    # rows: [(label, aligned string)]; colours compare each column to `ref` (default: first row)
    ref = ref if ref is not None else rows[0][1]
    for r, (lab, s) in enumerate(rows):
        st = status_row(s, ref) if lab != "H37Rv" else ["gap" if c == "-" else "same as H37Rv" for c in s]
        for c, (ch, k) in enumerate(zip(s, st)):
            ax.add_patch(Rectangle((c, -r - 1), 1, 0.9, facecolor=STATUS[k],
                                   edgecolor=NEUTRAL["axis"], lw=0.6))
            ax.text(c + 0.5, -r - 0.55, ch, ha="center", va="center", fontsize=fontsize,
                    family="monospace", color=NEUTRAL["ink"])
        ax.text(-0.4, -r - 0.55, lab, ha="right", va="center", fontsize=fontsize)
    ax.set_xlim(0, max(len(s) for _, s in rows))
    ax.set_ylim(-len(rows) - 0.1, 0.1)
    ax.axis("off")


def status_legend(fig, keys=None, **kw):
    keys = keys or list(STATUS)
    fig.legend([Patch(facecolor=STATUS[k], edgecolor=NEUTRAL["axis"]) for k in keys], keys,
               ncol=len(keys), frameon=False, loc="lower center", **kw)
""")

# =============================================================================
md(r"""
## 0. How an aligned FASTA is built from a VCF

`make_aligned_msa.py` does this for each locus window (the gene ±100 bp):

1. **Cells.** Each isolate starts as a list with one **cell** per H37Rv position,
   holding H37Rv's base. Each VCF record edits cells. A SNP replaces a cell's base,
   and a deletion replaces cells with `-`. An insertion has to fit several bases
   into one position, so the *whole* ALT allele goes into the cell at the
   variant's position. The rest of the REF span becomes the placeholder `X`,
   which is deleted at write time.
2. **Widths.** A position gets as many alignment columns as its widest cell
   across all isolates. H37Rv gets its base, then a `-` for each extra column.
3. **Padding.** Each isolate's cells are padded with `-` to their position's
   width and concatenated. Every row then has the same length, and column *j*
   means the same thing in every row.

The toy below runs the extractor's own padding functions (`aln_widths.py`) on
two isolates: one with a SNP, and one with an insertion `C → CAA` at position 4.
""")

code(r"""
ref = "ATGCCGTACGGA"
snp = list(ref); snp[2] = "T"                  # G -> T at position 2
ins = list(ref); ins[4] = "CAA"                # C -> CAA at position 4: the whole ALT in one cell
widths = realized_widths([list(ref), snp, ins], len(ref))
H = pad_reference(ref, widths)
rows = [("H37Rv", H), ("isolate 1 (SNP)", pad_cells(snp, widths)), ("isolate 2 (insertion)", pad_cells(ins, widths))]
print("widths per H37Rv position:", widths)
for lab, s in rows:
    print(f"{lab:24s} {s}")

fig, axes = plt.subplots(2, 1, figsize=(9, 3.4), gridspec_kw={"height_ratios": [1, 1.2]})
ax = axes[0]
x = 0
for i, (w, c) in enumerate(zip(widths, ins)):
    ax.add_patch(Rectangle((x, 0), w, 0.9, facecolor=STATUS["inserted base"] if len(c) > 1 else NEUTRAL["grid"],
                           edgecolor=NEUTRAL["ink2"], lw=1))
    ax.text(x + w / 2, 0.45, c, ha="center", va="center", family="monospace", fontsize=10)
    ax.text(x + w / 2, 1.05, str(i), ha="center", va="bottom", fontsize=8, color=NEUTRAL["ink2"])
    x += w
ax.text(-0.4, 0.45, "isolate 2 cells", ha="right", va="center", fontsize=10)
ax.set_xlim(0, sum(widths)); ax.set_ylim(-0.1, 1.5); ax.axis("off")
ax.set_title("Toy locus: one cell per H37Rv position (top), padded alignment rows (bottom)", fontsize=10)
draw_rows(axes[1], rows)
status_legend(fig, ["same as H37Rv", "gap", "inserted base", "differs from H37Rv"], bbox_to_anchor=(0.5, -0.06))
plt.tight_layout(); plt.show()
""")

md(r"""
Position 4 gets 3 columns because isolate 2's cell there holds 3 bases. H37Rv
and isolate 1 get `--` in the two new columns. Columns that are `-` in the H37Rv
row are **insertion columns**: they exist because some isolate carries an
insertion there.
""")

# =============================================================================
md(r"""
## 1. The insertion bug

### 1.1 The line

The labmate's extractor was edited on 2025-07-22. The edit changed how an
insertion is written into the cells:

```python
# before (Jul-19 2025 upstream) and after our 2026-10-22 fix -- correct
new_seq[idx] = alt_allele                          # whole ALT into ONE cell

# Jul-22 2025 upstream (the bug)
new_seq[idx:idx+len(alt_allele)] = list(alt_allele)   # one ALT base per cell
```

A list slice assignment with as many items as slots *replaces* those slots.
For `C → CAA`, the buggy line writes `C`, `A`, `A` into cells 4, 5 and 6. The
inserted `AA` overwrites the H37Rv bases at 5 and 6 instead of going between
4 and 5. Every cell still holds one base, so the isolate is never wider than
H37Rv. Step 2 still adds insertion columns, because the bookkeeping that
records insertion lengths was untouched, but no isolate has bases to put in
them. **Every insertion column comes out all-gap.** That is the signature
used in 1.5.
""")

code(r"""
def insert_jul2025(cells, idx, REF, ALT):     # the buggy upstream line
    s = cells.copy(); s[idx:idx + len(REF)] = ["X"] * len(REF); s[idx:idx + len(ALT)] = list(ALT); return s

def insert_fixed(cells, idx, REF, ALT):       # make_aligned_msa.py after 2026-10-22
    s = cells.copy(); s[idx:idx + len(REF)] = ["X"] * len(REF); s[idx] = ALT; return s

def pad_bookkeeping(cells, book):
    # step 3 as it was before 2026-10-28: pad a cell up to 1 + recorded insertion length;
    # a cell already longer is left alone; X placeholders are stripped
    return "".join(c + "-" * (w - len(c)) if len(c) < w else c for c, w in zip(cells, book)).replace("X", "")

idx, REF, ALT = 4, "C", "CAA"
book = [1] * len(ref); book[idx] += len(ALT) - len(REF)     # insertion_dict: +2 columns at position 4
bug, fix = insert_jul2025(list(ref), idx, REF, ALT), insert_fixed(list(ref), idx, REF, ALT)
truth = ref[:idx] + ALT + ref[idx + len(REF):]
print("cells, buggy :", bug)
print("cells, fixed :", fix)
print(f"ungapped, buggy: {''.join(bug).replace('X', '')}   (length {len(''.join(bug))})")
print(f"ungapped, fixed: {''.join(fix).replace('X', '')}   (length {len(''.join(fix).replace('X', ''))})")
print(f"H37Rv with CAA spliced in: {truth}")

H = pad_reference(ref, book)
fig, ax = plt.subplots(figsize=(9, 1.9))
draw_rows(ax, [("H37Rv", H), ("Jul-2025 extractor", pad_bookkeeping(bug, book)),
               ("fixed extractor", pad_bookkeeping(fix, book))])
ax.set_title("Toy insertion C→CAA at position 4: aligned rows from the buggy and fixed lines", fontsize=10)
status_legend(fig, bbox_to_anchor=(0.5, -0.15)); plt.show()
""")

md(r"""
The buggy row has the right length but the wrong content. The inserted bases
are missing, the two insertion columns are empty, and H37Rv's `GT` after the
insertion reads `AA`. A model sees two SNPs where there is an insertion. If the
inserted bases match the bases they land on, as in a homopolymer run (`C → CT`
before a `T`), the overwrite changes nothing and the isolate reads as H37Rv at
that site.

### 1.2 The same thing on real isolates

Each case shows one real isolate's VCF record and three rows on the repaired
file's column grid: H37Rv, the file every paper run read (`data_snapshot`, built
with the Jul-2025 extractor), and the repaired file. The isolates are picked
from the cached VCF records: one insertion in a repeat, so the overwrite is
invisible, and one where the overwrite changes bases.
""")

code(r"""
VCF = pd.read_csv(PREP / "vcf_records.tsv", sep="\t", dtype={"QUAL": str, "INFO": str, "FILTER": str})
VCF = VCF.fillna({"REF": "", "ALT": "", "INFO": "", "FILTER": "", "QUAL": ""})
FOUR = ["eis", "embB", "ethA", "rrs"]
OLD = {g: read_fasta(OLD_DIR / f"{g}.fasta") for g in FOUR}
NEW = {g: read_fasta(NEW_DIR / f"{g}.fasta") for g in FOUR}
for g in FOUR:
    print(f"{g:5s} sense={WIN[g]['sense']}  window {WIN[g]['start']}-{WIN[g]['end']}  "
          f"columns {len(OLD[g][REF_ID])} -> {len(NEW[g][REF_ID])}")


def genome_rows(g, iso):
    # H37Rv / old / new rows in GENOME orientation on the NEW file's column grid
    o, n, ro, rn = OLD[g][iso], NEW[g][iso], OLD[g][REF_ID], NEW[g][REF_ID]
    if WIN[g]["sense"] == "neg":
        o, n, ro, rn = rc(o), rc(n), rc(ro), rc(rn)
    at_ref = [c for c, r in zip(o, ro) if r != "-"]            # old row, one char per H37Rv position
    assert all(c == "-" for c, r in zip(o, ro) if r == "-")    # old insertion columns: empty
    it = iter(at_ref)
    o_on_new = "".join(next(it) if r != "-" else "-" for r in rn)
    return rn, o_on_new, n


def col_of_genome_pos(g, pos):
    # column in the genome-oriented new grid that holds H37Rv position `pos` (1-based)
    rn = NEW[g][REF_ID]
    rn = rc(rn) if WIN[g]["sense"] == "neg" else rn
    k = pos - WIN[g]["start"]
    return [j for j, r in enumerate(rn) if r != "-"][k]


def simple_insertions():
    # changed isolates whose window has exactly one record, a PASS insertion with REF of 1 base
    out = []
    for (g, iso), d in VCF[VCF.changed & (VCF.POS > 0)].groupby(["locus", "isolate"]):
        if len(d) == 1:
            r = d.iloc[0]
            if r.FILTER == "PASS" and len(r.REF) == 1 and len(r.ALT) > 1 and "," not in r.ALT:
                out.append((g, iso, r))
    return out


cases = {}
for g, iso, r in simple_insertions():
    k = len(r.ALT) - 1
    downstream = G[r.POS:r.POS + k]                 # the k H37Rv bases right after the anchor
    kind = "repeat" if downstream == r.ALT[1:] else "overwrite"
    cases.setdefault(kind, (g, iso, r))
print({k: (v[0], v[1], f"{v[2].POS} {v[2].REF}>{v[2].ALT}") for k, v in cases.items()})

fig, axes = plt.subplots(len(cases), 1, figsize=(11, 2.0 * len(cases)))
for ax, (kind, (g, iso, r)) in zip(np.atleast_1d(axes), sorted(cases.items())):
    rn, o, n = genome_rows(g, iso)
    c = col_of_genome_pos(g, r.POS)
    lo, hi = max(0, c - 8), c + len(r.ALT) + 10
    draw_rows(ax, [("H37Rv", rn[lo:hi]), ("data_snapshot (Jul-2025)", o[lo:hi]), ("repaired", n[lo:hi])], fontsize=9)
    ax.set_title(f"{g} {iso}: VCF {r.POS} {r.REF}>{r.ALT} ({kind}), genome orientation, "
                 f"columns {lo}-{hi}", fontsize=10)
status_legend(fig, bbox_to_anchor=(0.5, -0.04)); plt.tight_layout(); plt.show()
""")

md(r"""
In both cases the `data_snapshot` row has nothing in the insertion column. In
the repeat case the overwrite writes the same base back, so this isolate's
insertion has vanished and the locus reads as H37Rv. In the other case the
bases after the anchor are replaced, so the file reports substitutions that are
not in the VCF.

### 1.3 Replaying the extractor from the raw VCFs

To show that the old file *is* the buggy line's output and the new file *is*
the fixed line's output, the next cell re-implements step 1 of
`make_aligned_msa.py` (`allele_category` and `introduce_snps_indels_single_seq`).
It follows the source line by line, and the insertion branch is switchable. It
replays every cached isolate from its own VCF records: all 395 whose row changed,
plus 50 unchanged ones per locus. The replay is compared to each file
**ungapped**, because widths depend on the whole cohort, not on one isolate.

A third check does not use the cell machinery: H37Rv with each called variant
spliced in as a plain string edit. It applies to isolates whose records do not
overlap each other.
""")

code(r"""
Rec = namedtuple("Rec", "POS REF ALT QUAL FILTER INFO")


def parse_info(s):
    d = {}
    for kv in s.split(";"):
        if kv:
            k, eq, v = kv.partition("=")
            d[k] = v if eq else True
    return d


def allele_category(rec, qualThresh=10, presentThresh=0.75):
    # make_aligned_msa.allele_category, with pyvcf's parsed fields read off the raw line
    alt = rec.ALT.replace(",", "")                 # "".join(record.ALT)
    info = parse_info(rec.INFO)
    if "AF" not in info:
        af = 0.76
    else:
        afl = str(info["AF"]).split(",")
        if len(afl) > 1:
            return "missing"
        af = float(afl[0])
    qual = 11 if rec.QUAL in ("", ".") else float(rec.QUAL)
    if "IMPRECISE" in info:
        return "ref"
    filt = [] if rec.FILTER in ("PASS", ".", "") else rec.FILTER.split(";")
    if len(filt) > 0 and "Amb" not in filt:
        return "missing"
    if "N" in rec.REF or "N" in alt:
        return "missing"
    if not alt.isalnum():
        return "missing"
    if qual < qualThresh:
        return "missing"
    if "DP" in info and float(info["DP"]) < 5:
        return "missing"
    if "MQ" in info and float(info["MQ"]) < 30:
        return "missing"
    if len(rec.REF) == len(alt) and "BQ" in info and float(info["BQ"]) < 20:
        return "missing"
    if "AF" in info:
        if af <= 1 - presentThresh:
            return "ref"
        elif af > presentThresh:
            return "alt"
        return "missing"
    return "alt"


def extract_cells(records, START, END, region, buggy):
    # make_aligned_msa.introduce_snps_indels_single_seq, step 1; START 0-based, END 1-based
    new_seq = list(region)
    for rec in records:
        alt, ref_a = rec.ALT.replace(",", ""), rec.REF
        cat = allele_category(rec)
        if cat == "ref":
            continue
        idx = rec.POS - (START + 1)
        if len(ref_a) == len(alt):
            if idx < 0:
                continue
            new_seq[idx:idx + len(ref_a)] = list(alt) if cat == "alt" else ["N"] * len(alt)
            if len(ref_a) > (END - rec.POS) + 1:
                new_seq = new_seq[:len(region)]
        elif len(alt) > len(ref_a):
            if idx < 0 or idx >= len(region):
                continue
            if cat == "alt" or (cat == "missing" and len(alt) - len(ref_a) <= 100):
                if cat == "missing":
                    alt = "N" * len(alt)
                new_seq[idx:idx + len(ref_a)] = ["X"] * len(ref_a)
                if buggy:
                    new_seq[idx:idx + len(alt)] = list(alt)
                else:
                    new_seq[idx] = alt
        else:
            if cat == "alt":
                if rec.POS <= START or rec.POS > END:
                    end_del = rec.POS + len(ref_a) - 1
                    if end_del > END:
                        new_seq = ["-"] * len(region)
                    else:
                        new_seq[:end_del - START] = ["-"] * (end_del - START)
                else:
                    old_len = len(new_seq)
                    new_seq[idx:idx + len(ref_a)] = list(alt) + ["-"] * (len(ref_a) - len(alt))
                    new_seq = new_seq[:old_len]
    return new_seq[:len(region)]


def ungapped(cells, sense):
    s = "".join(cells).replace("X", "").replace("-", "")
    return rc(s) if sense == "neg" else s


def splice_truth(records, START, region):
    # H37Rv with every called variant spliced in as a string edit; None if records overlap
    calls = [(r, allele_category(r)) for r in records]
    spans = sorted((r.POS, r.POS + len(r.REF) - 1) for r, c in calls if c != "ref")
    if any(b[0] <= a[1] for a, b in zip(spans, spans[1:])):
        return None
    s = region
    for r, c in sorted(calls, key=lambda t: -t[0].POS):
        alt, i = r.ALT.replace(",", ""), r.POS - (START + 1)
        if c == "ref" or i < 0 or i + len(r.REF) > len(region):
            if c != "ref":
                return None                          # boundary-crossing record: out of scope here
            continue
        if len(alt) == len(r.REF):
            new = alt if c == "alt" else "N" * len(alt)
        elif len(alt) > len(r.REF):
            if c == "missing" and len(alt) - len(r.REF) > 100:
                continue
            new = alt if c == "alt" else "N" * len(alt)
        else:
            if c != "alt":
                continue
            new = alt
        s = s[:i] + new + s[i + len(r.REF):]
    return s


rows = []
for (g, iso), d in VCF.groupby(["locus", "isolate"]):
    w = WIN[g]; START, END = w["start"] - 1, w["end"]
    region = G[START:END]
    recs = [Rec(int(r.POS), r.REF, r.ALT, r.QUAL, r.FILTER, r.INFO) for r in d.itertuples() if r.POS > 0]
    old_u, new_u = OLD[g][iso].replace("-", ""), NEW[g][iso].replace("-", "")
    t = splice_truth(recs, START, region)
    rows.append({"locus": g, "isolate": iso, "changed": bool(d.changed.iloc[0]),
                 "buggy replay == old file": ungapped(extract_cells(recs, START, END, region, True), w["sense"]) == old_u,
                 "fixed replay == new file": ungapped(extract_cells(recs, START, END, region, False), w["sense"]) == new_u,
                 "splice checkable": t is not None,
                 "splice == new file": t is not None and (rc(t) if w["sense"] == "neg" else t) == new_u,
                 "splice == old file": t is not None and (rc(t) if w["sense"] == "neg" else t) == old_u})
REPLAY = pd.DataFrame(rows)
summary = (REPLAY.groupby(["locus", "changed"])
           .agg(isolates=("isolate", "size"),
                buggy_eq_old=("buggy replay == old file", "sum"),
                fixed_eq_new=("fixed replay == new file", "sum"),
                splice_checkable=("splice checkable", "sum"),
                splice_eq_new=("splice == new file", "sum"),
                splice_eq_old=("splice == old file", "sum")))
display(summary)
bad = REPLAY[~REPLAY["fixed replay == new file"] | ~REPLAY["buggy replay == old file"]]
print(f"{len(bad)} isolates where a replay disagrees with its file")
display(bad.head(10))
""")

code(r"""
s = summary.reset_index()
s = s[s.changed]
x = np.arange(len(s)); w = 0.26
fig, ax = plt.subplots(figsize=(9, 3.4))
for k, (col, lab, colr) in enumerate([("buggy_eq_old", "buggy-line replay = data_snapshot file", MODEL["locusfusion"]),
                                      ("fixed_eq_new", "fixed-line replay = repaired file", MODEL["lasso"]),
                                      ("splice_eq_new", "plain splice = repaired file", MODEL["cnn"])]):
    denom = s["splice_checkable"] if col == "splice_eq_new" else s["isolates"]
    ax.bar(x + (k - 1) * w, s[col] / denom.clip(lower=1), w, color=colr, label=lab)
    for xi, v, dd in zip(x + (k - 1) * w, s[col], denom):
        ax.text(xi, v / max(dd, 1) + 0.02, f"{v}/{dd}", ha="center", fontsize=7)
ax.set_xticks(x, s["locus"]); ax.set_ylim(0, 1.12); ax.set_ylabel("fraction of isolates")
ax.set_title("Changed isolates: replay of each extractor version vs each file, by locus", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
plt.tight_layout(); plt.show()
""")

md(r"""
Read-out:
- If the buggy replay reproduces `data_snapshot` and the fixed replay reproduces
  the repaired file for every changed isolate, the overwrite line fully explains
  the old file, and the fix does exactly what it claims. Any isolate where a
  replay disagrees is listed above.
- For isolates whose records do not overlap, the plain string splice agrees with
  the repaired file. This is the independent check: H37Rv plus the called
  variants, without the cell machinery. Isolates with overlapping records are
  covered in section 2.
- The unchanged sample confirms that isolates without insertions are untouched.

### 1.4 How much damage, and where it matters

These counts are over all 17,942 isolates and both files. "Inserted bases" are
non-gap characters in insertion columns. "Overwritten" positions are H37Rv
positions where the two files disagree and the repaired file has a base there.
""")

code(r"""
dmg = []
for g in FOUR:
    ro, rn = OLD[g][REF_ID], NEW[g][REF_ID]
    ref_cols_o = np.array([c != "-" for c in ro]); ref_cols_n = np.array([c != "-" for c in rn])
    ins_n = ~ref_cols_n
    ids = [k for k in NEW[g] if k != REF_ID]
    O = np.frombuffer("".join(OLD[g][k] for k in ids).encode(), "S1").reshape(len(ids), -1)
    N = np.frombuffer("".join(NEW[g][k] for k in ids).encode(), "S1").reshape(len(ids), -1)
    Oref, Nref = O[:, ref_cols_o], N[:, ref_cols_n]
    insb = (N[:, ins_n] != b"-").sum(1)
    over = (Oref != Nref) & (Nref != b"-")
    href = np.frombuffer(ro.replace("-", "").encode(), "S1")
    old_is_wt = (Oref == href[None, :]).all(1)
    dmg.append({"locus": g, "insertion columns (repaired)": int(ins_n.sum()),
                "filled in data_snapshot": int((O[:, ~ref_cols_o] != b"-").any(0).sum()),
                "isolates with inserted bases": int((insb > 0).sum()),
                "inserted bases lost": int(insb.sum()),
                "isolates with overwritten positions": int(over.any(1).sum()),
                "overwritten by N": int((over & (Oref == b"N")).sum()),
                "overwritten by a base": int((over & (Oref != b"N")).sum()),
                "carriers that read exactly H37Rv in data_snapshot": int(((insb > 0) & old_is_wt).sum())})
DMG = pd.DataFrame(dmg).set_index("locus")
display(DMG)
""")

md(r"""
The worst case is **ethA**, the ethionamide-activating gene. Its insertions are
mostly in the coding sequence, and an insertion whose length is not a multiple
of 3 shifts the reading frame. A frameshift that knocks out EthA is a classic
ETO resistance mechanism. The next cell counts the ethA frameshift-insertion
carriers, how many of them read exactly as H37Rv in the old file, and their
ETO phenotypes.
""")

code(r"""
ethA = next(f for f in GENOME.features if f.type == "CDS" and f.qualifiers.get("gene", [""])[0] == "ethA")
cds0, cds1 = int(ethA.location.start), int(ethA.location.end)        # 0-based half-open, genome
g = "ethA"; rn = rc(NEW[g][REF_ID])                                    # genome orientation
gpos, p = [], WIN[g]["start"] - 1
for r in rn:                                                           # genome position each column sits after
    if r != "-":
        p += 1
    gpos.append(p)
gpos = np.array(gpos)
ins_cds = np.array([r == "-" for r in rn]) & (gpos > cds0) & (gpos <= cds1)   # 1-based positions in the CDS
pheno = pd.read_csv(W / "data_snapshot/phenotype_data/master_resistance_table.csv",
                    index_col="New_ID", usecols=["New_ID", "ETHIONAMIDE"])["ETHIONAMIDE"]
pheno = pheno[~pheno.index.duplicated()]
ids = [k for k in NEW[g] if k != REF_ID]
M = np.frombuffer("".join(NEW[g][k] for k in ids).encode(), "S1").reshape(len(ids), -1)
cols_file = M.shape[1] - 1 - np.nonzero(ins_cds)[0]                  # genome-orientation column -> file column
href_old = OLD[g][REF_ID].replace("-", "")
E = pd.DataFrame({"cds_inserted": (M[:, cols_file] != b"-").sum(1),
                  "old_is_h37rv": [OLD[g][k].replace("-", "") == href_old for k in ids]}, index=ids)
E["frameshift_ins"] = (E.cds_inserted % 3) != 0
E["ETO"] = pheno.reindex(E.index)
grp = {"all phenotyped": E, "ethA frameshift-insertion carriers": E[E.frameshift_ins]}
tab = pd.DataFrame({k: {"isolates": len(v), "ETO phenotyped": int(v.ETO.isin(["R", "S"]).sum()),
                        "ETO R": int((v.ETO == "R").sum()),
                        "read exactly as H37Rv in data_snapshot": int(v.old_is_h37rv.sum())}
                    for k, v in grp.items()}).T
tab["% R of phenotyped"] = 100 * tab["ETO R"] / tab["ETO phenotyped"].clip(lower=1)
display(tab)

fig, ax = plt.subplots(figsize=(5.5, 2.4))
ax.barh(range(len(tab)), tab["% R of phenotyped"], color=[NEUTRAL["muted"], MODEL["locusfusion"]])
for i, (v, n) in enumerate(zip(tab["% R of phenotyped"], tab["ETO phenotyped"])):
    ax.text(v + 1, i, f"{v:.0f}% (n={n})", va="center", fontsize=9)
ax.set_yticks(range(len(tab)), tab.index); ax.set_xlim(0, 110); ax.set_xlabel("% ETO-resistant")
ax.set_title("ETO resistance: all phenotyped isolates vs ethA frameshift-insertion carriers", fontsize=10)
plt.tight_layout(); plt.show()
""")

md(r"""
The rows that "read exactly as H37Rv" are isolates whose only ethA variant was
an insertion. In the old file they carry no variant at all, so a model trained
on it cannot tell them apart from wild type.

### 1.5 The signature, across every input folder

A correct alignment has insertion columns because some isolate fills them. The
bug leaves every insertion column of a file empty. The prep job scanned every
aligned FASTA in every folder the models have read. Each dot is one file: the
fraction of its insertion columns that at least one isolate fills. Files with no
insertion columns are left out (counts in the table).
""")

code(r"""
SCAN = pd.read_csv(PREP / "alignment_scan.csv")
order = list(dict.fromkeys(SCAN.folder))
SCAN["filled frac"] = 1 - SCAN.ins_unfilled / SCAN.ins_cols.where(SCAN.ins_cols > 0)
tab = SCAN.groupby("folder", sort=False).agg(
    files=("locus", "size"), with_insertion_cols=("ins_cols", lambda s: int((s > 0).sum())),
    all_insertion_cols_empty=("filled frac", lambda s: int((s == 0).sum())),
    n_records_not_17943=("n_records", lambda s: int((s != 17943).sum())),
    ragged=("ragged", "sum")).reindex(order)
display(tab)

fig, ax = plt.subplots(figsize=(8, 3.6))
rng = np.random.default_rng(0)
for i, f in enumerate(order):
    d = SCAN[(SCAN.folder == f) & (SCAN.ins_cols > 0)]
    y = i + rng.uniform(-0.18, 0.18, len(d))
    ax.scatter(d["filled frac"], y, s=14, color=np.where(d["filled frac"] == 0, MODEL["locusfusion"], MODEL["lasso"]),
               edgecolor="none", alpha=0.85)
ax.set_yticks(range(len(order)), order); ax.invert_yaxis()
ax.set_xlabel("fraction of insertion columns filled by ≥1 isolate"); ax.set_xlim(-0.05, 1.05)
ax.set_title("Insertion columns filled, per aligned FASTA, by input folder", fontsize=10)
plt.tight_layout(); plt.show()
""")

md(r"""
In a folder built with the Jul-2025 line, every file with insertions sits at 0.
In a re-extracted folder they sit at or near 1. A file can sit below 1
legitimately: an insertion column can exist only because of a
low-confidence call that was later masked. The 14 extra loci in the labmate's
set are scanned from her own files, which the bug also affected (see 5.2).
""")

# =============================================================================
md(r"""
## 2. Why the first fix crashed: widths from bookkeeping vs from the cells

The 2026-10-22 fix restored `new_seq[idx] = alt_allele`, but it still sized each
position from bookkeeping: `insertion_dict[idx] = max(len(ALT) − len(REF))`,
width `1 + insertion_dict[idx]`. That is right when REF is a single base. Pilon
also emits **long-REF** insertions: a 148-bp REF replaced by a 151-bp ALT is
"+3" to the bookkeeping. Now suppose a *second* record falls inside that REF
span. On embB that second record is a low-confidence SNP, which writes `N` over
one of the `X` placeholders. That cell is no longer stripped, so the row gains a
character the bookkeeping never counted. The isolate comes out longer than the
reference, and `assert len(seq) == len(new_ref_seq)` kills the job partway
through writing the FASTA. That is how embB, ethA and rrs died, leaving the
truncated files of section 3.

The toy below uses a 4-base REF, a +3 ALT, and a masked SNP inside the span.
""")

code(r"""
ref = "ACGTGCATGGAC"
cells = list(ref)
cells = insert_fixed(cells, 3, "TGCA", "TGGTGCA")    # long-REF insertion: +3 by the bookkeeping
cells[5] = "N"                                         # a masked SNP inside that REF span
print("cells:", cells)
book = [1] * len(ref); book[3] += 3
H_book = pad_reference(ref, book)
row_book = pad_bookkeeping(cells, book)
print(f"bookkeeping: reference {len(H_book)} columns, isolate {len(row_book)} -> "
      f"{'assert FAILS' if len(row_book) != len(H_book) else 'ok'}")
real = realized_widths([list(ref), cells], len(ref))
H_real, row_real = pad_reference(ref, real), pad_cells(cells, real)
print(f"realized   : widths {real}\n             reference {len(H_real)}, isolate {len(row_real)}")
print("H37Rv  :", H_real); print("isolate:", row_real)

fig, ax = plt.subplots(figsize=(9, 1.5))
draw_rows(ax, [("H37Rv", H_real), ("isolate", row_real)])
ax.set_title("Toy nested variants, padded with widths from the cells (aln_widths.realized_widths)", fontsize=10)
status_legend(fig, bbox_to_anchor=(0.5, -0.2)); plt.show()
""")

md(r"""
`aln_widths.realized_widths` sets each position's width to its **longest
realized cell** across isolates (minimum 1). By construction no row can overflow
and nothing is dropped. For the ordinary single-base-REF insertion this equals
`1 + max insertion`, which is the bookkeeping, so files that already built
cleanly do not change. The first check below confirms it: eis was rebuilt from
its pickled cells and compared with its Oct-22 file.

The embB isolate that killed the job, and its records in the window:
""")

code(r"""
reg = json.load(open(PREP / "eis_regression.json"))
print(f"eis rebuilt with realized widths vs Oct-22 file: {reg['identical']} of {reg['records']} records byte-identical")

d = VCF[(VCF.locus == "embB") & (VCF.isolate == "SAMN08892416") & (VCF.POS > 0)].copy()
d["category"] = [allele_category(Rec(int(r.POS), r.REF, r.ALT, r.QUAL, r.FILTER, r.INFO)) for r in d.itertuples()]
d["len REF"], d["len ALT"] = d.REF.str.len(), d.ALT.str.len()
d["REF span end"] = d.POS + d["len REF"] - 1
d["REF"], d["ALT"] = d.REF.str.slice(0, 20), d.ALT.str.slice(0, 20)
display(d[["POS", "REF span end", "len REF", "len ALT", "REF", "ALT", "FILTER", "category"]])

WID = pd.read_csv(PREP / "widths.csv")
short = WID[WID.realized > WID.bookkeeping]
print("positions where a cell is wider than the bookkeeping allowed:")
display(short)
""")

md(r"""
The 148-bp record spans 4,247,427–4,247,574. The two `Del;Amb` SNPs at
4,247,573/4 lie inside it and are low-confidence (`missing`), so each writes an
`N` over an `X`. That gives 2 extra characters, the "3425 vs 3423" in the job's
assertion. Before 2026-10-22 the bug hid this: it wrote the ALT one base per
cell, so no cell was ever wider than one base.

Width, in columns beyond the window length, per locus and per version:
""")

code(r"""
wrows = []
for g in FOUR:
    L = WIN[g]["end"] - WIN[g]["start"] + 1
    wg = WID[WID.locus == g]
    wrows.append({"locus": g, "Jul-2025 file (data_snapshot)": len(OLD[g][REF_ID]) - L,
                  "2026-10-22 bookkeeping": int((wg.bookkeeping - 1).sum()),
                  "2026-10-28 realized (repaired file)": int((wg.realized - 1).sum()),
                  "repaired file, measured": len(NEW[g][REF_ID]) - L})
WT = pd.DataFrame(wrows).set_index("locus")
display(WT)
fig, ax = plt.subplots(figsize=(7, 3))
cols = ["Jul-2025 file (data_snapshot)", "2026-10-22 bookkeeping", "2026-10-28 realized (repaired file)"]
x = np.arange(len(WT)); w = 0.26
for k, (c, colr) in enumerate(zip(cols, [NEUTRAL["muted"], MODEL["cnn"], MODEL["lasso"]])):
    ax.bar(x + (k - 1) * w, WT[c], w, color=colr, label=c)
ax.set_xticks(x, WT.index); ax.set_ylabel("insertion columns")
ax.set_title("Alignment width beyond the H37Rv window, by locus and extractor version", fontsize=10)
ax.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.show()
""")

md(r"""
- The bookkeeping and realized widths differ only at the few positions in the
  table above. Almost all of the realized width over the bookkeeping is one
  long-REF cell per locus, which holds its whole ALT at one position: embB's
  151-base cell adds about 150 columns where the bookkeeping counted 3.
- The Jul-2025 file and the 2026-10-22 bookkeeping have the same width. The
  bug never changed how many columns there were, only left them empty. All the
  growth in the repaired files comes from the five long-REF cells above.
""")

# =============================================================================
md(r"""
## 3. Truncated files, and why the loader did not notice

A job that dies while writing a FASTA leaves a well-formed file that ends early.
The 2026-10-28 audit (`regulatory_msa/validate_alignments.py`) found these. The
numbers come from that audit; the short files have since been overwritten by
rebuilds:

| file | records | cause |
|---|---|---|
| `curated_fix/aligned/{embB,ethA,rrs}` (Oct 22–28) | partial | the section-2 assertion |
| `regulatory_msa/aligned_atg/whiB6` | 2,265 of 17,943 | job died mid-write |
| `gene_msa/aligned_fixed/dnaA` | 8,726 | job died mid-write |
| `gene_msa/aligned_fixed/whiB6` | 2,265 | job died mid-write |

The loader joins per-locus tables on isolate ID. The toy below gives it two
loci, one complete and one cut short, and runs the project's
`load_sequence_df` and the v4 tokenizer's new guard on them.
""")

code(r"""
from datasets.sequences import load_sequence_df
from locusfusion_v4.tokenizer import _check_cohort

tmp = Path(tempfile.mkdtemp())
isolates = [f"ISO{i}" for i in range(1, 7)]
def write(locus, ids, ref="ACGTACGT", alt="ACGAACGT"):
    with open(tmp / f"{locus}.fasta", "w") as fh:
        fh.write(f">{REF_ID}\n{ref}\n")
        for i in ids:
            fh.write(f">{i}\n{alt}\n")
write("locA", isolates)
write("locB", isolates[:3])                               # died after 3 isolates

both, _ = load_sequence_df(tmp, ["locA", "locB"])
print("two loci, joined (what build_tokens does next is .fillna('')):")
display(both.fillna("<NaN>"))
one, _ = load_sequence_df(tmp, ["locB"])
print(f"locB alone: {len(one) - 1} isolates (+ H37Rv) -- the cohort silently shrinks from 6 to 3")
try:
    _check_cohort(both, isolates, ["locA", "locB"], tmp)
except ValueError as e:
    print("\n_check_cohort:", e)
""")

md(r"""
There are two silent failure modes:
- **Several loci:** a missing row becomes `""`. A token is a column where the
  isolate differs from H37Rv, so an empty row has no tokens. Isolates 4–6 read
  as wild type at locB, a direct analogue of the insertion bug.
- **One locus:** the isolate index comes from the FASTA, so the cohort becomes
  whichever isolates were written. No error is raised, and a 2,265-isolate
  whiB6 would train on 2,265 isolates.

The tokenizer now runs `_check_cohort` (each locus must cover ≥99% of the
best-covered locus's isolates) and `_check_coverage` (≥99% of rows at the
reference width). Both raise with the file path. In the current scan, every
folder has 0 files with a record count other than 17,943 (table in 1.5).
""")

# =============================================================================
md(r"""
## 4. Promoter windows: counted from the start codon, not the TSS

The regulatory modality reads a per-isolate alignment of each WHO Table-22
promoter region (+30 bp flank). Table 22 gives each region as offsets
(`1–532` for katG). The July build counted the offsets from the gene's
**transcription start site** (TSS). WHO counts them from the **start codon**:
`c.-1` is the base before the ATG, the numbering every WHO promoter variant
uses (`fabG1 c.-15C>T`, `eis c.-10G>A`). Where the TSS lies upstream of the ATG,
the old window sits that many bases too far upstream.

The evidence is in Table 22 itself. Where WHO lists a TSS, the far end of the
region is almost always the TSS-to-ATG distance plus 51. The region runs from
the ATG to 50 bp beyond the TSS, which only makes sense counted from the ATG.
""")

code(r"""
A = json.load(open(W / "regulatory_msa/promoter_windows.json"))["windows"]
B = json.load(open(W / "regulatory_msa/promoter_windows_atg.json"))["windows"]
rows = []
for gname in A:
    a, b = A[gname], B[gname]
    s = 1 if b["sense"] == "pos" else -1
    # genome -> position relative to the ATG, on the circular chromosome (dnaA's upstream is its end)
    c = lambda x: s * (((x - b["atg"]) + len(G) // 2) % len(G) - len(G) // 2)
    seg = b["segments"]
    lo_off, hi_off = min(x[0] for x in seg), max(x[1] for x in seg)
    old = sorted([c(a["start"]), c(a["end"])]); new = sorted([c(b["start"]), c(b["end"])])
    who = (-hi_off, -lo_off)
    cover = lambda w: max(0, min(w[1], who[1]) - max(w[0], who[0]) + 1) / (who[1] - who[0] + 1)
    rows.append({"gene": gname, "sense": b["sense"], "tss_to_atg": abs(c(b["tss"])) if b.get("tss") else np.nan,
                 "who_far_end": hi_off, "old_window": tuple(old), "new_window": tuple(new),
                 "who_region": who, "old covers WHO": cover(old), "new covers WHO": cover(new),
                 "segments": len(seg), "old dropped segments": len(a.get("dropped_segments", []))})
P = pd.DataFrame(rows).set_index("gene")
on_line = (P.who_far_end - P.tss_to_atg == 51).sum()
print(f"{on_line} of {P.tss_to_atg.notna().sum()} genes with a TSS: far end = TSS-to-ATG + 51")
print(f"new windows cover the whole WHO region for {(P['new covers WHO'] == 1).sum()} of {len(P)} genes; "
      f"old windows for {(P['old covers WHO'] == 1).sum()}")

fig, ax = plt.subplots(figsize=(5, 4))
d = P.dropna(subset=["tss_to_atg"])
on = d.who_far_end - d.tss_to_atg == 51
ax.scatter(d.tss_to_atg[on], d.who_far_end[on], s=18, color=MODALITY["regulatory"], label="far end = TSS-to-ATG + 51")
ax.scatter(d.tss_to_atg[~on], d.who_far_end[~on], s=18, color=MODEL["locusfusion"], label="other")
for gname, r in d[~on].iterrows():
    ax.annotate(gname, (r.tss_to_atg, r.who_far_end), fontsize=7, xytext=(3, 3), textcoords="offset points")
xx = np.array([0, d.tss_to_atg.max()])
ax.plot(xx, xx + 51, color=NEUTRAL["muted"], lw=1, zorder=0, label="TSS-to-ATG + 51")
ax.set_xlabel("TSS-to-ATG distance (bp)"); ax.set_ylabel("far end of Table-22 region (bp)")
ax.set_title("WHO Table 22: region far end vs TSS-to-ATG distance", fontsize=10)
ax.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.show()
""")

code(r"""
show = [g for g in ["katG", "rpoB", "ahpC", "eis", "embA", "pncA"] if g in P.index]
KNOWN = {"eis": {"c.-37": -37, "c.-14": -14, "c.-10": -10}}      # well-known WHO eis promoter variants
fig, axes = plt.subplots(len(show), 1, figsize=(10, 1.15 * len(show)), sharex=True)
for ax, gname in zip(axes, show):
    r, b = P.loc[gname], B[gname]
    tss = -r.tss_to_atg
    for y, (lab, (lo, hi), colr) in enumerate([("WHO region", r.who_region, NEUTRAL["ink2"]),
                                               ("old (TSS)", r.old_window, MODEL["locusfusion"]),
                                               ("new (ATG)", r.new_window, MODEL["lasso"])]):
        ax.add_patch(Rectangle((lo, -y - 0.8), hi - lo + 1, 0.6, color=colr))
        ax.text(-1290, -y - 0.5, lab, ha="left", va="center", fontsize=7, color=colr)
    ax.axvline(0, color=NEUTRAL["ink"], lw=1)
    if tss != 0:
        ax.axvline(tss, color=NEUTRAL["muted"], lw=1, ls="--")
        ax.text(tss - 3, 0.1, "TSS", fontsize=7, va="bottom", ha="right", color=NEUTRAL["ink2"])
    ax.text(3, 0.1, "ATG" if tss != 0 else "ATG = TSS", fontsize=7, va="bottom")
    if gname in KNOWN:
        for v in KNOWN[gname].values():
            ax.plot([v, v], [-3, 0], color=MODALITY["regulatory"], lw=1)
        ax.text(min(KNOWN[gname].values()) - 5, -3.0, " ".join(KNOWN[gname]), fontsize=7, ha="right",
                va="center", color=MODALITY["regulatory"])
    ax.set_ylim(-3.6, 0.6); ax.set_yticks([]); ax.set_ylabel(gname, rotation=0, ha="right", va="center")
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
axes[0].set_xlim(-1300, 150)
axes[-1].set_xlabel("position relative to the start codon (bp; negative = upstream)")
axes[0].set_title("Promoter windows vs the WHO region, start-codon coordinates", fontsize=10)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(11, 3))
o = P.sort_values("old covers WHO")
ax.bar(range(len(o)), o["old covers WHO"], color=MODEL["locusfusion"], label="old (TSS-anchored) window")
ax.scatter(range(len(o)), o["new covers WHO"], color=MODEL["lasso"], s=10, zorder=3, label="new (ATG-anchored) window")
ax.set_xticks(range(len(o)), o.index, rotation=90, fontsize=7); ax.set_ylabel("fraction of WHO region covered")
ax.set_title("Coverage of each gene's WHO Table-22 region, old vs new window (49 shared regions)", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower right"); plt.tight_layout(); plt.show()
display(P[P["old covers WHO"] < 1][["sense", "tss_to_atg", "who_region", "old_window", "new_window",
                                    "old covers WHO", "old dropped segments"]].sort_values("old covers WHO"))
""")

md(r"""
- Genes where the TSS *is* the ATG (embA, pncA here) have identical windows in
  both builds. Everything else moved by its TSS-to-ATG distance. For katG the old
  window misses nearly the whole WHO region.
- The best-known promoter mutations (eis c.-10/-14/-37, fabG1 c.-15, ahpC) lie
  within ~50 bp of the ATG. The 30-bp pad kept most of them inside the old
  windows, which is why the old regulatory results were not obviously broken.
- The old build also dropped the second segment of two-segment regions (last
  column of the table).
- The old promoter files also carry the insertion bug (section 1.5). The new
  `aligned_atg/` files were extracted after both fixes and pass
  `validate_promoter_msa.py` (ids, width, reference row, insertions, VCF spot
  checks).
""")

# =============================================================================
md(r"""
## 5. The labmate's extra loci: `pepQ` is a different gene, `mmpS5` is shifted

The lab64 locus set adds 14 loci from the labmate's Oct-2025 folder. To check
what each file contains, the next cell degaps its H37Rv row and searches for it
in the genome, on both strands. It reads `locus_sets/rebuilt/`, whose H37Rv rows
are identical to hers once degapped.
""")

code(r"""
def locate(path):
    ref = read_fasta(path, limit=1)[REF_ID].replace("-", "")
    i = G.find(ref)
    strand = "+"
    if i < 0:
        i, strand = G.find(rc(ref)), "-"
    return i, i + len(ref), strand, len(ref)

def genes_in(lo, hi):
    out = []
    for f in GENOME.features:
        if f.type == "CDS" and int(f.location.end) > lo and int(f.location.start) < hi:
            out.append((f.qualifiers.get("gene", [""])[0] or f.qualifiers["locus_tag"][0],
                        f.qualifiers["locus_tag"][0], int(f.location.start), int(f.location.end),
                        "+" if f.location.strand == 1 else "-"))
    return out

def cds_of(name):
    f = next(f for f in GENOME.features if f.type == "CDS" and name in
             (f.qualifiers.get("gene", [""])[0], f.qualifiers.get("locus_tag", [""])[0]))
    return int(f.location.start), int(f.location.end), "+" if f.location.strand == 1 else "-"

for gname in ["pepQ", "mmpS5"]:
    lo, hi, st, L = locate(W / f"locus_sets/rebuilt/{gname}.fasta")
    print(f"{gname}.fasta H37Rv row: {L} bp = genome [{lo}, {hi}) strand {st}")
    print(f"   real {gname} CDS: {cds_of(gname)}")
    print(f"   CDSs overlapping the file's window: {genes_in(lo, hi)}")
""")

code(r"""
fig, axes = plt.subplots(2, 1, figsize=(10, 3.6))
ax = axes[0]
ax.plot([0, len(G)], [0, 0], color=NEUTRAL["axis"], lw=3)
plo, phi, _, _ = locate(W / "locus_sets/rebuilt/pepQ.fasta")
q0, q1, _ = cds_of("pepQ")
for x, lab, colr in [((q0 + q1) / 2, "pepQ (Rv2535c) CDS", MODEL["lasso"]), ((plo + phi) / 2, "pepQ.fasta window (Rv3698)", MODEL["locusfusion"])]:
    ax.plot([x, x], [-0.3, 0.3], color=colr, lw=3); ax.text(x, 0.4, lab, ha="center", fontsize=8, color=colr)
ax.set_xlim(0, len(G)); ax.set_ylim(-0.6, 0.9); ax.set_yticks([])
ax.set_xticks(np.arange(0, 4.5e6, 5e5), [f"{v/1e6:.1f}" for v in np.arange(0, 4.5e6, 5e5)])
ax.set_xlabel("H37Rv position (Mb)"); ax.set_title("pepQ: real CDS vs the window in pepQ.fasta", fontsize=10)

ax = axes[1]
mlo, mhi, _, _ = locate(W / "locus_sets/rebuilt/mmpS5.fasta")
m0, m1, _ = cds_of("mmpS5")
for name in ["mmpL5", "mmpS5", "Rv0678"]:
    a0, a1, st = cds_of(name)
    ax.add_patch(Rectangle((a0, 0.5), a1 - a0, 0.5, color=NEUTRAL["muted"]))
    ax.text((max(a0, 778200) + min(a1, 779600)) / 2, 0.75, f"{name} ({st})", ha="center", va="center", fontsize=8,
            color=NEUTRAL["surface"])
ax.add_patch(Rectangle((mlo, -0.3), mhi - mlo, 0.4, color=MODEL["locusfusion"]))
ax.text(mhi + 15, -0.1, "mmpS5.fasta window", fontsize=8, va="center", color=MODEL["locusfusion"])
ax.add_patch(Rectangle((m0 - 100, -1.1), (m1 + 100) - (m0 - 100), 0.4, color=MODEL["lasso"]))
ax.text(m1 + 115, -0.9, "mmpS5 ± 100 bp (re-extraction)", fontsize=8, va="center", color=MODEL["lasso"])
ax.set_xlim(778200, 779600); ax.set_ylim(-1.4, 1.5); ax.set_yticks([])
ax.set_xlabel("H37Rv position (bp)"); ax.set_title("mmpS5 region: genes, labmate's window, corrected window", fontsize=10)
plt.tight_layout(); plt.show()

# what the protein modality would read: the window's own translation
for gname in ["mmpS5", "pepQ"]:
    lo, hi, st, _ = locate(W / f"locus_sets/rebuilt/{gname}.fasta")
    c0, c1, cst = cds_of(gname)
    inside = max(0, min(hi, c1) - max(lo, c0))
    print(f"{gname}: {inside} of {c1 - c0} CDS bases inside the file's window")
m_seq = Seq(G[m0:m1]).reverse_complement()
print(f"mmpS5 CDS ends with stop codon {m_seq[-3:]}; window covers genome down to {mlo}, CDS 3' end (minus strand) is at {m0}"
      f" -> the last {mlo - m0} bp of the CDS, stop codon included, are outside the window")
""")

md(r"""
- `pepQ.fasta` covers Rv3698 ±100 bp at 4.14 Mb, about 1.3 Mb from the real
  pepQ. Her coordinate file labels that window "Rv3698". Every modality,
  including DNA, reads the wrong gene.
- `mmpS5.fasta` is the gene shifted 100 bp upstream. It has 100 bp extra
  upstream and is missing the last 100 bp of the CDS, including the stop codon.
  The CDS finder therefore fails, and protein and biophysical translate the
  whole window out of frame.
- These 14 files also carry the insertion bug. It happened in step 1, so the
  damage is already in her pickled cells and cannot be rebuilt from them (her
  14 in the 1.5 scan). Array 66380513 is re-extracting all 14 from the VCFs with
  the fixed extractor and corrected windows (`locus_sets/extract/windows.json`:
  pepQ [2859200, 2860518] −, mmpS5 [778377, 779005] −). Until it finishes and
  validates, no lab64 job should run.
""")

# =============================================================================
md(r"""
## 6. Which results used which version (as of 2026-10-30)

| input | file version read | problems it carries | used by |
|---|---|---|---|
| curated 19 DNA | `data_snapshot/genomic_data/aligned` | insertion bug in eis, embB, ethA, rrs | **every paper run** (the DNA input of every model) |
| curated 19 DNA | `curated_fix/aligned19` (15 originals + 4 rebuilt) | none known | locus-sets sweep (who_t12, lab64) |
| promoter windows | `regulatory_msa/aligned` (TSS-anchored, 49) | wrong anchor; insertion bug | every regulatory / all-modalities result (`bigtb_ref.REAL_REGULATORY_DIR` still points here) |
| promoter windows | `regulatory_msa/aligned_atg` (ATG-anchored, 61) | none known | LocusFusion v4 `v5_atg` arm only |
| who_t12 genes | `gene_msa/aligned` (archived) | insertion bug | 2026-09-24 who_t12 experiment |
| who_t12 genes | `gene_msa/aligned_fixed` | none known | locus-sets sweep |
| labmate's 14 extra | `locus_sets/rebuilt` | insertion bug; pepQ wrong gene; mmpS5 shifted | nothing yet (lab64 waits for re-extraction) |

The paper's DNA inputs differ from the repaired ones only for isolates in the
1.4 table: 17 (eis), 6 (embB), 225 (ethA) and 147 (rrs) of 17,942. The ethA rows
fall on one drug, ethionamide.
""")

md(r"""
## Caveats

- **Long-REF insertions are aligned validly but not minimally.** With realized
  widths, a cell like embB's 151-base ALT sits at one position, followed by gap
  cells across its REF span. The ungapped sequence is right, but positionally
  the isolate reads as a long insertion plus a long deletion, not a 3-bp
  insertion. This affects a handful of isolates per locus (section 2 table).
- **Nested records are double-counted.** A masked SNP inside a long REF span
  adds an `N` on top of an ALT that already covers that base (embB
  SAMN08892416: 2 extra `N`). This is how the extractor resolves overlapping
  Pilon records, kept as is. The plain-splice check in 1.3 excludes isolates
  with overlapping records for this reason.
- **The replay in 1.3 reimplements step 1; it does not run the extractor.** It
  follows the source line by line with pyvcf's field parsing done by hand, and
  it covers the 395 changed isolates and 200 unchanged ones, not all 17,942.
- **The truncation counts in section 3 come from the 2026-10-28 audit.** The
  short files themselves were overwritten by the rebuilds.
- **ETO counts in 1.4 are associations among carriers**, not a measured change
  in model AUC. No paper run has been re-trained on the repaired curated set.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "analysis" / "data_fixes.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
