"""
BIG-TB VCFs -> GenomeFusion tokens.

One token per changed base, or per indel event. Nothing is capped: a locus
carries as many tokens as the isolate has changes in it, and loci the isolate
does not change produce nothing at all.

    from genomefusion.tokenizer import tokenize_isolate
    tok = tokenize_isolate(vcf_path, partition, np.frombuffer(ref, np.uint8))

The VCFs are Pilon 1.24 output, normalised with `bcftools norm` and filtered
with `bcftools view --types snps,indels,mnps,other`, so they hold variant sites
only: **a position with no record is reference OR not covered**, and the two
cannot be told apart from these files. The QC below (`allele_category`) is the
same rule as the project's MSA extractor (msa/make_aligned_msa.py),
so a variant this module calls alt / ref / missing is the one LocusFusion's
tokens would have carried. Synonymous changes are dropped.

Residues are read on the H37Rv codon grid, as in LocusFusion: codon k is the
isolate's bases at reference codon k's three genome positions, so an indel never
turns the rest of a protein into false substitutions. What the grid cannot show
-- that the frame shifted, and how much protein is lost -- is carried by the
`frameshift` consequence, the `truncation` channel, and the per-locus frame
offset on the [NULL] token.
"""
from dataclasses import dataclass
import gzip
import re

import numpy as np

from datasets.biochem import CODON_TABLE
from genomefusion.layout import (FRAMESHIFT, INFRAME_INDEL, MISSENSE, NONSENSE, NT_SYMBOLS,
                                 R_GAP, R_STOP, R_UNK, RESIDUE_SYMBOLS, START_LOSS, STOP_LOSS,
                                 SYN, UNREADABLE)
from genomefusion.reference import COMPLEMENT, codon_positions

NT_INDEX = {c: i for i, c in enumerate(NT_SYMBOLS)}
RES_INDEX = {c: i for i, c in enumerate(RESIDUE_SYMBOLS)}
FLAG_OVERLAP, FLAG_HOMOPOLYMER, FLAG_IMPRECISE = 1, 2, 4
HOMOPOLYMER_MIN = 5

_INFO_RE = re.compile(r"([^=;]+)(?:=([^;]*))?")


@dataclass
class IsolateTokens:
    """One isolate's tokens, as parallel arrays in locus order."""
    locus: np.ndarray        # int32
    pos: np.ndarray          # int32, genome 0-based
    base: np.ndarray         # int8, index into NT_SYMBOLS
    residue: np.ndarray      # int8, index into RESIDUE_SYMBOLS; -1 off a codon
    ref_residue: np.ndarray  # int8
    consequence: np.ndarray  # int8; -1 off a coding sequence
    phase: np.ndarray        # int8; -1 off a codon
    span: np.ndarray         # int32, bases the event covers (1 for a substitution)
    region: np.ndarray       # int8, index into layout.REGIONS
    up_plus: np.ndarray      # int32, bp upstream of the next + strand start codon; -1
    up_minus: np.ndarray
    truncation: np.ndarray   # float32
    flags: np.ndarray        # uint8 bitfield
    codon: np.ndarray        # int32, codon index within the CDS; -1
    # per (isolate, locus) summary, for the [NULL] token
    sum_locus: np.ndarray    # int32, the active loci
    sum_n: np.ndarray        # int32, tokens in that locus
    sum_span: np.ndarray     # int32, total bases covered
    sum_frame: np.ndarray    # int32, net frame offset at the end of the CDS
    sum_oof: np.ndarray      # float32, fraction of the CDS read out of frame
    stats: dict = None


def parse_info(s):
    out = {}
    for m in _INFO_RE.finditer(s):
        out[m.group(1)] = m.group(2)
    return out


def allele_category(ref, alt, qual, filt, info, qual_thresh=10, present_thresh=0.75):
    """"alt", "ref" or "missing" -- the same rule as make_aligned_msa.allele_category.

    "ref" means leave the reference base in place (a confident reference call,
    or a change we cannot place); "missing" means an uncalled base (N).
    """
    af = info.get("AF")
    if af is None:
        af_val = 0.76
    else:
        vals = af.split(",")
        if len(vals) > 1:
            return "missing"
        af_val = float(vals[0])
    q = 11.0 if qual is None else qual
    if "IMPRECISE" in info:
        return "ref"
    filts = [f for f in filt.split(";") if f not in ("", ".", "PASS")]
    if filts and "Amb" not in filts:
        return "missing"
    if "N" in ref or "N" in alt:
        return "missing"
    if not alt.isalnum():
        return "missing"
    if q < qual_thresh:
        return "missing"
    if "DP" in info and int(info["DP"]) < 5:
        return "missing"
    if "MQ" in info and int(info["MQ"]) < 30:
        return "missing"
    if len(ref) == len(alt) and "BQ" in info and int(info["BQ"]) < 20:
        return "missing"
    if "AF" in info:
        if af_val <= (1 - present_thresh):
            return "ref"
        if af_val > present_thresh:
            return "alt"
        return "missing"
    return "alt"


def read_vcf(path, max_indel=100):
    """[(pos0, ref, alt, category, imprecise)] for one isolate, QC applied.

    Indels longer than `max_indel` that are not confidently called are left as
    reference, as the MSA extractor does, to avoid long synthetic runs of N.
    """
    out = []
    op = gzip.open if str(path).endswith((".gz", ".bgz")) else open
    with op(path, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            pos, ref, alt, qual, filt, info_s = f[1], f[3].upper(), f[4].upper(), f[5], f[6], f[7]
            if alt.startswith("<"):          # symbolic ALT (DUP); no sequence to place
                continue
            alt = alt.split(",")[0] if "," in alt else alt
            info = parse_info(info_s)
            q = None if qual in (".", "") else float(qual)
            cat = allele_category(ref, alt, q, filt, info)
            if cat == "ref":
                continue
            if cat == "missing" and abs(len(alt) - len(ref)) > max_indel:
                continue
            out.append((int(pos) - 1, ref, alt, cat, "IMPRECISE" in info))
    return out


def _homopolymer(ref_arr, p, min_run=HOMOPOLYMER_MIN):
    """True if genome position p sits in a run of >= min_run identical bases."""
    b = ref_arr[p]
    i = p
    while i > 0 and ref_arr[i - 1] == b:
        i -= 1
    j = p
    n = len(ref_arr)
    while j + 1 < n and ref_arr[j + 1] == b:
        j += 1
    return (j - i + 1) >= min_run


def _translate(codon_bytes):
    return CODON_TABLE.get(codon_bytes.decode(), "X")


def tokenize_isolate(vcf_path, partition, ref_arr, codon_cache=None):
    """One isolate's tokens. `ref_arr` is the H37Rv sequence as uint8."""
    records = read_vcf(vcf_path)
    locus_of, codon_of, phase_of = partition.locus_of, partition.codon_of, partition.phase_of

    subs = {}          # genome pos -> isolate base character (substitutions and N)
    deleted = set()    # genome positions whose base is gone
    events = []        # (pos0, kind, span, imprecise)  kind in sub/ins/del
    for pos0, r, a, cat, imprecise in records:
        if len(r) == len(a):
            for i, (rb, ab) in enumerate(zip(r, a)):
                p = pos0 + i
                if cat == "missing":
                    subs[p] = "N"
                    events.append((p, "sub", 1, imprecise))
                elif rb != ab:
                    subs[p] = ab
                    events.append((p, "sub", 1, imprecise))
        elif len(a) > len(r):                       # insertion
            n = len(a) - len(r)
            if cat == "missing":
                subs[pos0] = "N"
            events.append((pos0, "ins", n, imprecise))
        else:                                        # deletion
            n = len(r) - len(a)
            first = pos0 + len(a)
            span_pos = np.arange(first, min(first + n, len(locus_of)))
            if cat == "alt":
                deleted.update(int(p) for p in span_pos)
            else:
                for p in span_pos:
                    subs[int(p)] = "N"
            # a deletion that crosses a locus boundary leaves one span token in
            # each locus it touches, at that locus's first deleted base
            if len(span_pos):
                li_span = locus_of[span_pos]
                for li in np.unique(li_span):
                    sel = span_pos[li_span == li]
                    events.append((int(sel[0]), "del", int(len(sel)), imprecise))

    # ---- emit a token per event, grouped by locus ----
    cols = {k: [] for k in ("locus", "pos", "base", "residue", "ref_residue", "consequence",
                            "phase", "span", "region", "up_plus", "up_minus", "truncation",
                            "flags", "codon")}
    per_locus = {}
    n_masked = 0
    codon_cache = {} if codon_cache is None else codon_cache

    for pos0, kind, span, imprecise in events:
        li = int(locus_of[pos0]) if 0 <= pos0 < len(locus_of) else -1
        if li < 0:
            n_masked += 1
            continue
        L = partition.loci[li]
        region = {"cds": 0, "rna": 1, "intergenic": 2}[L.kind]
        cod = int(codon_of[pos0])
        ph = int(phase_of[pos0])
        res = ref_res = -1
        cons = -1
        trunc = 0.0

        if L.kind == "cds" and cod >= 0:
            key = li
            if key not in codon_cache:
                codon_cache[key] = codon_positions(L, partition)
            cp = codon_cache[key]
            n_cod = len(cp)
            if cod < n_cod:
                positions = cp[cod]
                if (positions < 0).any():
                    cons, res, ref_res = UNREADABLE, R_UNK, R_UNK
                else:
                    rb = bytearray(ref_arr[positions].tobytes())
                    ab = bytearray(rb)
                    for k, q in enumerate(positions):
                        q = int(q)
                        if q in deleted:
                            ab[k] = ord("-")
                        elif q in subs:
                            ab[k] = ord(subs[q])
                    if L.strand < 0:
                        rb = bytes(rb).translate(COMPLEMENT)
                        ab = bytes(ab).translate(COMPLEMENT)
                    else:
                        rb, ab = bytes(rb), bytes(ab)
                    r_aa = _translate(rb)
                    a_aa = "-" if b"-" in ab else ("X" if b"N" in ab else _translate(ab))
                    res = RES_INDEX.get(a_aa, R_UNK)
                    ref_res = RES_INDEX.get(r_aa, R_UNK)
                    if kind in ("ins", "del"):
                        cons = FRAMESHIFT if span % 3 else INFRAME_INDEL
                    elif res in (R_GAP, R_UNK) or ref_res in (R_GAP, R_UNK):
                        cons = UNREADABLE
                    elif res == ref_res:
                        cons = SYN
                    elif ref_res == R_STOP:
                        cons = STOP_LOSS
                    elif res == R_STOP:
                        cons = NONSENSE
                    elif cod == 0:
                        cons = START_LOSS
                    else:
                        cons = MISSENSE
                    if cons in (FRAMESHIFT, NONSENSE, START_LOSS) and n_cod:
                        trunc = float(n_cod - cod) / n_cod
            else:
                cons, res, ref_res = UNREADABLE, R_UNK, R_UNK

        if cons == SYN:
            continue

        ch = subs.get(pos0)
        if kind == "del" or pos0 in deleted:
            b = NT_INDEX["-"]
        elif ch is not None:
            b = NT_INDEX.get(ch, NT_INDEX["N"])
        else:
            b = NT_INDEX.get(chr(ref_arr[pos0]), NT_INDEX["N"])

        flags = 0
        if partition.overlap[pos0]:
            flags |= FLAG_OVERLAP
        if kind in ("ins", "del") and _homopolymer(ref_arr, pos0):
            flags |= FLAG_HOMOPOLYMER
        if imprecise:
            flags |= FLAG_IMPRECISE
            cons = UNREADABLE if L.kind == "cds" else cons

        cols["locus"].append(li)
        cols["pos"].append(pos0)
        cols["base"].append(b)
        cols["residue"].append(res)
        cols["ref_residue"].append(ref_res)
        cols["consequence"].append(cons)
        cols["phase"].append(ph)
        cols["span"].append(span)
        cols["region"].append(region)
        cols["up_plus"].append(int(partition.up_plus[pos0]))
        cols["up_minus"].append(int(partition.up_minus[pos0]))
        cols["truncation"].append(trunc)
        cols["flags"].append(flags)
        cols["codon"].append(cod)

        s = per_locus.setdefault(li, {"n": 0, "span": 0, "frame": 0, "indels": []})
        s["n"] += 1
        s["span"] += span
        if L.kind == "cds" and kind in ("ins", "del") and cod >= 0:
            d = span if kind == "ins" else -span
            s["frame"] += d
            s["indels"].append((cod, d))

    # per-locus frame summary: net offset, and the fraction of the CDS out of frame
    sum_locus, sum_n, sum_span, sum_frame, sum_oof = [], [], [], [], []
    for li, s in sorted(per_locus.items()):
        L = partition.loci[li]
        n_cod = max(L.cds_len // 3, 1)
        oof = 0.0
        if s["indels"]:
            run, last, covered = 0, 0, 0
            for cod, d in sorted(s["indels"]):
                if run % 3:
                    covered += max(cod - last, 0)
                run += d
                last = cod
            if run % 3:
                covered += max(n_cod - last, 0)
            oof = min(covered / n_cod, 1.0)
        sum_locus.append(li)
        sum_n.append(s["n"])
        sum_span.append(s["span"])
        sum_frame.append(s["frame"])
        sum_oof.append(oof)

    order = np.argsort(np.asarray(cols["locus"], np.int64), kind="stable") if cols["locus"] \
        else np.zeros(0, np.int64)

    def arr(k, dt):
        a = np.asarray(cols[k], dt) if cols[k] else np.zeros(0, dt)
        return a[order] if len(a) else a

    return IsolateTokens(
        locus=arr("locus", np.int32), pos=arr("pos", np.int32), base=arr("base", np.int8),
        residue=arr("residue", np.int8), ref_residue=arr("ref_residue", np.int8),
        consequence=arr("consequence", np.int8), phase=arr("phase", np.int8),
        span=arr("span", np.int32), region=arr("region", np.int8),
        up_plus=arr("up_plus", np.int32), up_minus=arr("up_minus", np.int32),
        truncation=arr("truncation", np.float32), flags=arr("flags", np.uint8),
        codon=arr("codon", np.int32),
        sum_locus=np.asarray(sum_locus, np.int32), sum_n=np.asarray(sum_n, np.int32),
        sum_span=np.asarray(sum_span, np.int32), sum_frame=np.asarray(sum_frame, np.int32),
        sum_oof=np.asarray(sum_oof, np.float32),
        stats={"n_records": len(records), "n_events": len(events),
               "n_masked": n_masked,
               "n_tokens": len(cols["locus"]), "n_active_loci": len(sum_locus)})
