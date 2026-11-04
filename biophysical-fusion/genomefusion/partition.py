"""
The genome partition: every base of H37Rv in exactly one locus.

A locus is a gene (CDS / rRNA / tRNA / ncRNA, as annotated in the GenBank
record) or an intergenic segment between two genes. There are no promoter
windows: a 100 bp upstream window would run into a neighbouring gene for 61% of
genes, and the same base would then be a token in two loci. The regulatory role
of an intergenic base is carried by the token instead, as its distance upstream
of the next start codon on each strand (`up_plus` / `up_minus`).

    from genomefusion.partition import build_partition
    p = build_partition()          # cached under <workspace>/genomefusion_cache/
    p.loci                         # LocusInfo per locus, genome order
    p.locus_of[pos]                # 0-based genome position -> locus index, -1 masked

Overlapping genes: 13,121 bases of H37Rv lie in two gene features. Each goes to
the gene whose START codon is nearer, so start codons stay whole; the token's
`overlap` flag records that a second gene also carries it.

Masking: a base is dropped (locus -1) when it falls in the Marin et al. 2022
refined low-confidence (RLC) BED, or in a gene whose Mycobrowser functional
category is "PE/PPE" or "insertion seqs and phages". A locus with no unmasked
base is dropped entirely.
"""
from dataclasses import dataclass, field
from pathlib import Path
import csv
import json
import re

import numpy as np

import os

PROJECT = Path(__file__).resolve().parents[1]
# A SLURM job runs from a frozen snapshot, where PROJECT.parent is the snapshot
# directory and holds none of the shared data. These therefore come from the
# environment, with the live workspace only as a fallback, and are checked
# rather than silently skipped: a partition built without the mask or the
# annotation would look fine and be wrong.
WORKSPACE = Path(os.environ.get("ABR_WORKSPACE", PROJECT.parent))
REFS = Path(os.environ.get("ABR_GF_REFS", WORKSPACE / "genomefusion_refs"))
CACHE = Path(os.environ.get("ABR_GF_CACHE", WORKSPACE / "genomefusion_cache"))

GENBANK = Path(os.environ.get(
    "ABR_GENBANK", WORKSPACE / "data_snapshot" / "genbank" / "genome.gbff"))
RLC_BED = REFS / "rlc" / "RLC_Regions.H37Rv.bed"
MYCOBROWSER = REFS / "mycobrowser" / "H37Rv_r5.txt"
ESSENTIALITY = REFS / "essentiality" / "dejesus_S3_orf_essentiality.tsv"

MASK_CATEGORIES = ("PE/PPE", "insertion seqs and phages")

# locus feature vector -------------------------------------------------------
REGION_KINDS = ("cds", "rna", "intergenic")
CATEGORIES = ("conserved hypotheticals", "intermediary metabolism and respiration",
              "cell wall and cell processes", "lipid metabolism", "information pathways",
              "virulence, detoxification, adaptation", "regulatory proteins",
              "PE/PPE", "insertion seqs and phages", "unknown")
ESSENTIAL_CALLS = ("ES", "ESD", "GD", "NE", "GA", "Uncertain")
#: the 19 loci LocusFusion v3/v5 use, by gene name
LOCI_19 = ("ahpC", "eis", "embA", "embB", "embC", "ethA", "ethR", "fabG1", "gid", "gyrA",
           "gyrB", "inhA", "katG", "pncA", "rpoB", "rpoC", "rpsL", "rrl", "rrs")


@dataclass
class LocusInfo:
    index: int
    name: str                 # locus_tag, or "<up>-<down>" for an intergenic segment
    gene: str                 # gene name where annotated, else ""
    kind: str                 # one of REGION_KINDS
    start: int                # 0-based inclusive, genome coordinates
    end: int                  # 0-based exclusive
    strand: int               # +1 / -1; 0 for intergenic
    n_bases: int              # unmasked bases assigned to this locus
    category: str = ""
    essential: str = ""
    pseudo: bool = False
    cds_len: int = 0          # coding length in bases (0 if not a CDS)
    is_19: bool = False
    up_genes: tuple = ()      # for an intergenic segment: (plus-strand gene, minus-strand gene)


@dataclass
class Partition:
    loci: list
    locus_of: np.ndarray      # (genome_len,) int32, -1 = masked / unassigned
    overlap: np.ndarray       # (genome_len,) bool, base lies in a second gene
    up_plus: np.ndarray       # (genome_len,) int32, bp to next + strand start codon; -1 none
    up_minus: np.ndarray
    codon_of: np.ndarray      # (genome_len,) int32 codon index within its CDS; -1 off a CDS
    phase_of: np.ndarray      # (genome_len,) int8 0/1/2; -1 off a CDS
    genome_len: int
    meta: dict = field(default_factory=dict)

    def feature_matrix(self):
        """(n_loci, F) float32 of the locus features the embedding is predicted from."""
        rows = []
        for L in self.loci:
            kind = [float(L.kind == k) for k in REGION_KINDS]
            cat = [float(L.category == c) for c in CATEGORIES]
            ess = [float(L.essential == e) for e in ESSENTIAL_CALLS]
            ess.append(float(L.essential == ""))          # explicit "no call"
            rows.append([*kind, *cat, *ess,
                         np.log1p(L.end - L.start) / 10.0,
                         (L.start + L.end) / 2.0 / self.genome_len,
                         float(L.strand > 0), float(L.strand < 0),
                         float(L.pseudo), float(L.is_19)])
        return np.asarray(rows, np.float32)

    @property
    def feature_names(self):
        return ([f"kind:{k}" for k in REGION_KINDS] + [f"cat:{c}" for c in CATEGORIES]
                + [f"ess:{e}" for e in ESSENTIAL_CALLS] + ["ess:none"]
                + ["log_len", "genome_frac", "strand_plus", "strand_minus", "pseudo", "is_19"])


# ---- GenBank ---------------------------------------------------------------
_FEATURE_RE = re.compile(r"^     (\S+)\s+(.*)$")
GENE_KINDS = {"CDS": "cds", "rRNA": "rna", "tRNA": "rna", "ncRNA": "rna",
              "tmRNA": "rna", "misc_RNA": "rna"}


def read_genbank(path=GENBANK):
    """[(locus_tag, gene, start0, end, strand, kind, pseudo)] plus the genome length.

    `gene` features give the span; the kind comes from the CDS/rRNA/... feature
    with the same locus_tag. Spans written as join(...) are skipped (there are
    none in NC_000962.3) and reported in `skipped`.
    """
    txt = Path(path).read_text()
    glen = int(re.search(r"^LOCUS\s+\S+\s+(\d+) bp", txt, re.M).group(1))
    kinds = {}
    for m in re.finditer(r"^     (CDS|rRNA|tRNA|ncRNA|tmRNA|misc_RNA)\s+.*?\n"
                         r"((?:                     .*\n)+)", txt, re.M):
        tag = re.search(r'/locus_tag="([^"]+)"', m.group(2))
        if tag:
            kinds[tag.group(1)] = GENE_KINDS[m.group(1)]
    genes, skipped, cur = [], [], None
    for line in txt.split("\n"):
        m = _FEATURE_RE.match(line)
        if m:
            cur = None
            if m.group(1) == "gene":
                loc = m.group(2)
                nums = [int(x) for x in re.findall(r"\d+", loc)]
                if "join" in loc or len(nums) != 2:
                    skipped.append(loc)
                    continue
                cur = {"start": min(nums) - 1, "end": max(nums),
                       "strand": -1 if "complement" in loc else 1,
                       "tag": None, "gene": "", "pseudo": False}
                genes.append(cur)
        elif cur is not None:
            q = line.strip()
            if q.startswith("/locus_tag="):
                cur["tag"] = q.split('"')[1]
            elif q.startswith("/gene="):
                cur["gene"] = q.split('"')[1]
            elif q.startswith("/pseudo"):
                cur["pseudo"] = True
    for g in genes:
        g["kind"] = kinds.get(g["tag"], "cds")
    return genes, glen, skipped


def read_bed(path):
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith(("#", "track", "browser")):
            continue
        f = line.split("\t")
        out.append((int(f[1]), int(f[2])))      # 0-based half-open
    return out


def read_mycobrowser(path=MYCOBROWSER):
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            tag = (r.get("Locus") or "").strip()
            if tag:
                out[tag.lower()] = {"category": (r.get("Functional_Category") or "").strip(),
                                    "pseudo": (r.get("Is_Pseudogene") or "").strip() == "Yes"}
    return out


def read_essentiality(path=ESSENTIALITY):
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            tag = (r.get("ORF ID") or "").strip()
            if tag:
                out[tag.lower()] = (r.get("Final Call") or "").strip()
    return out


# ---- the partition ---------------------------------------------------------
def build_partition(genbank=GENBANK, rlc_bed=RLC_BED, mask_categories=MASK_CATEGORIES,
                    min_locus_bases=1, verbose=True, require_refs=True):
    missing = [str(p) for p in (genbank, rlc_bed, MYCOBROWSER, ESSENTIALITY)
               if p and not Path(p).exists()]
    if missing and require_refs:
        raise FileNotFoundError(
            "GenomeFusion reference data is missing, so the partition would be built without "
            "the mask / categories / essentiality and would silently differ:\n  "
            + "\n  ".join(missing)
            + "\nSet ABR_GF_REFS (and ABR_GENBANK) to the real workspace, or pass "
              "require_refs=False to build an unmasked partition on purpose.")
    genes, glen, skipped = read_genbank(genbank)
    myco = read_mycobrowser() if MYCOBROWSER.exists() else {}
    ess = read_essentiality() if ESSENTIALITY.exists() else {}

    # masked bases: RLC BED + genes in a masked functional category
    masked = np.zeros(glen, bool)
    n_rlc = 0
    if rlc_bed and Path(rlc_bed).exists():
        for s, e in read_bed(rlc_bed):
            masked[max(s, 0):min(e, glen)] = True
            n_rlc += 1
    n_cat = 0
    for g in genes:
        cat = myco.get((g["tag"] or "").lower(), {}).get("category", "")
        if cat in mask_categories:
            masked[g["start"]:g["end"]] = True
            n_cat += 1

    # assign each base to the gene whose start codon is nearest
    owner = np.full(glen, -1, np.int32)
    overlap = np.zeros(glen, bool)
    dist = np.full(glen, np.iinfo(np.int64).max, np.int64)
    for gi, g in enumerate(genes):
        sl = slice(g["start"], g["end"])
        start_base = g["start"] if g["strand"] > 0 else g["end"] - 1
        d = np.abs(np.arange(g["start"], g["end"], dtype=np.int64) - start_base)
        taken = owner[sl] >= 0
        overlap[sl] |= taken
        win = d < dist[sl]
        owner[sl] = np.where(win, gi, owner[sl])
        dist[sl] = np.where(win, d, dist[sl])

    # codon grid and phase, per CDS, on its own bases only
    codon_of = np.full(glen, -1, np.int32)
    phase_of = np.full(glen, -1, np.int8)
    cds_len = {}
    for gi, g in enumerate(genes):
        if g["kind"] != "cds":
            continue
        pos = np.arange(g["start"], g["end"], dtype=np.int64)
        own = owner[pos] == gi
        n = g["end"] - g["start"]
        off = pos - g["start"] if g["strand"] > 0 else (g["end"] - 1 - pos)
        codon_of[pos[own]] = (off[own] // 3).astype(np.int32)
        phase_of[pos[own]] = (off[own] % 3).astype(np.int8)
        cds_len[gi] = n

    # distance upstream of the next start codon, per strand
    up_plus = np.full(glen, -1, np.int32)
    up_minus = np.full(glen, -1, np.int32)
    starts_plus = sorted(g["start"] for g in genes if g["strand"] > 0 and g["kind"] == "cds")
    starts_minus = sorted(g["end"] - 1 for g in genes if g["strand"] < 0 and g["kind"] == "cds")
    if starts_plus:
        sp = np.asarray(starts_plus)
        idx = np.searchsorted(sp, np.arange(glen), side="left")
        ok = idx < len(sp)
        up_plus[ok] = sp[idx[ok]] - np.arange(glen)[ok]
    if starts_minus:
        sm = np.asarray(starts_minus)
        idx = np.searchsorted(sm, np.arange(glen), side="right") - 1
        ok = idx >= 0
        up_minus[ok] = np.arange(glen)[ok] - sm[idx[ok]]

    # loci: genes, then the gaps between them
    loci, locus_of = [], np.full(glen, -1, np.int32)
    order = sorted(range(len(genes)), key=lambda i: (genes[i]["start"], genes[i]["end"]))
    gene_locus = {}
    for gi in order:
        g = genes[gi]
        own = (owner == gi) & ~masked
        n = int(own.sum())
        if n < min_locus_bases:
            continue
        tag = g["tag"] or f"gene{gi}"
        info = myco.get(tag.lower(), {})
        li = len(loci)
        gene_locus[gi] = li
        locus_of[own] = li
        loci.append(LocusInfo(
            index=li, name=tag, gene=g["gene"], kind=g["kind"],
            start=g["start"], end=g["end"], strand=g["strand"], n_bases=n,
            category=info.get("category", ""), essential=ess.get(tag.lower(), ""),
            pseudo=bool(g["pseudo"] or info.get("pseudo")),
            cds_len=cds_len.get(gi, 0),
            is_19=g["gene"] in LOCI_19))

    # intergenic runs over bases no gene claimed
    free = owner < 0
    edges = np.flatnonzero(np.diff(np.concatenate(([0], free.view(np.int8), [0]))))
    n_ig_dropped = 0
    # nearest flanking gene on each side, by binary search over sorted spans
    by_end = sorted(range(len(genes)), key=lambda i: genes[i]["end"])
    ends = np.asarray([genes[i]["end"] for i in by_end])
    by_start = sorted(range(len(genes)), key=lambda i: genes[i]["start"])
    starts = np.asarray([genes[i]["start"] for i in by_start])
    for s, e in zip(edges[::2], edges[1::2]):
        own = np.zeros(glen, bool)
        own[s:e] = True
        own &= ~masked
        n = int(own.sum())
        if n < min_locus_bases:
            n_ig_dropped += 1
            continue
        j = int(np.searchsorted(ends, s, side="right")) - 1      # last gene ending at/before s
        up = genes[by_end[j]] if j >= 0 else None
        k = int(np.searchsorted(starts, e, side="left"))         # first gene starting at/after e
        dn = genes[by_start[k]] if k < len(starts) else None
        up_t = (up or {}).get("tag", "start")
        dn_t = (dn or {}).get("tag", "end")
        li = len(loci)
        locus_of[own] = li
        loci.append(LocusInfo(index=li, name=f"{up_t}-{dn_t}", gene="", kind="intergenic",
                              start=int(s), end=int(e), strand=0, n_bases=n,
                              up_genes=(up_t, dn_t)))

    meta = {"genome_len": glen, "n_genes": len(genes), "n_loci": len(loci),
            "n_gene_loci": sum(L.kind != "intergenic" for L in loci),
            "n_intergenic_loci": sum(L.kind == "intergenic" for L in loci),
            "masked_bases": int(masked.sum()), "masked_frac": float(masked.mean()),
            "rlc_regions": n_rlc, "masked_category_genes": n_cat,
            "overlap_bases": int(overlap.sum()), "unassigned_bases": int((locus_of < 0).sum()),
            "intergenic_dropped": n_ig_dropped, "genbank_skipped": skipped,
            "mask_categories": list(mask_categories), "genbank": str(genbank),
            "rlc_bed": str(rlc_bed), "refs": str(REFS),
            "has_categories": bool(myco), "has_essentiality": bool(ess)}
    if verbose:
        print(f"[partition] {meta['n_loci']:,} loci "
              f"({meta['n_gene_loci']:,} gene + {meta['n_intergenic_loci']:,} intergenic); "
              f"masked {meta['masked_bases']:,} bp ({meta['masked_frac']:.1%}); "
              f"overlap {meta['overlap_bases']:,} bp", flush=True)
    return Partition(loci=loci, locus_of=locus_of, overlap=overlap, up_plus=up_plus,
                     up_minus=up_minus, codon_of=codon_of, phase_of=phase_of,
                     genome_len=glen, meta=meta)


def save(p, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, locus_of=p.locus_of, overlap=p.overlap, up_plus=p.up_plus, up_minus=p.up_minus,
        codon_of=p.codon_of, phase_of=p.phase_of,
        loci=json.dumps([vars(L) for L in p.loci]), meta=json.dumps(p.meta))


def load(path):
    z = np.load(path, allow_pickle=False)
    loci = [LocusInfo(**{**d, "up_genes": tuple(d.get("up_genes", ()))})
            for d in json.loads(str(z["loci"]))]
    meta = json.loads(str(z["meta"]))
    return Partition(loci=loci, locus_of=z["locus_of"], overlap=z["overlap"],
                     up_plus=z["up_plus"], up_minus=z["up_minus"], codon_of=z["codon_of"],
                     phase_of=z["phase_of"], genome_len=meta["genome_len"], meta=meta)


def load_or_build(cache=CACHE, **kw):
    path = Path(cache) / "partition.npz"
    if path.exists():
        return load(path)
    p = build_partition(**kw)
    save(p, path)
    return p


if __name__ == "__main__":
    import sys
    p = build_partition()
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else CACHE / "partition.npz"
    save(p, out)
    print(json.dumps(p.meta, indent=1))
    print(f"wrote {out}")
