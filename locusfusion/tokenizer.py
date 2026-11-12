"""
Aligned FASTAs -> LocusFusion tokens, once, offline.

The tokens are built once from the aligned sequence strings and cached: for
17,942 isolates x 19 loci x 16 slots the whole token set is ~1 GB. Everything
the model needs is in the cache (TokenSet), so training never touches a FASTA.

The column geometry (genome placement, start-codon origin, codon and phase per
column, the promoter/record de-duplication) is `_frame_maps` below, and the
base / residue / property values follow these rules:

  * a token is a column whose base differs from H37Rv's (upper-cased), within
    the isolate's own sequence; an uncalled base (anything outside A C T G -)
    gets the N channel;
  * synonymous changes produce no token at all (dropped before the cap);
  * per locus the columns run [promoter window | gene record], 5' -> 3', and the
    first `MAX_VARIANTS` occupied ones are kept; [NULL] reports the true count;
  * the residue a coding token reads is the isolate's codon on the H37Rv codon
    grid (datasets.biochem.translate_reference_frame): '-' if a base of it is
    deleted, 'X' if unreadable, '*' for STOP; bio = its three properties (0 for
    '-', 'X', '*'), d_bio = bio minus H37Rv's residue's; `residue` = its one-hot.

`consequence` (synonymous / missense / nonsense / other, see layout.py) and
`site_pos`, the token's index into the binding-site distance tables (a residue
index on protein loci, a nucleotide offset on rrs / rrl, -1 elsewhere), are
stored per token as well.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import bigtb as tb
from datasets import cds
from datasets.biochem import AA_PROPERTY, CODON_TABLE
from datasets.loader import LOCI_19 as _LOCI_19
from datasets.regulatory import DRUG_TO_REGULATORY
from datasets.sequences import BASE_TO_COLUMN, load_phenotype, load_sequence_df, reference_row

from .layout import C_BASE, MISSENSE, NONSENSE, OTHER, SLOTS, SYN

# the 19 curated loci (MD-CNN's set; the paper's "19 loci"), in loader order
LOCI_19 = list(_LOCI_19)
ALL_DRUGS = list(tb.DRUG_TO_LOCI)        # label-column order of the multi-drug loader
MAX_VARIANTS = 16                        # token slots per locus

# residue alphabet of the cache: 20 residues, STOP, deleted codon, unreadable
RESIDUES = "ACDEFGHIKLMNPQRSTVWY*-X"
R_STOP, R_GAP, R_UNK = 20, 21, 22
PROPS = np.zeros((len(RESIDUES), 3), np.float32)
for _i, _r in enumerate(RESIDUES):
    if _r in AA_PROPERTY:
        PROPS[_i] = AA_PROPERTY[_r]

# base byte -> nucleotide channel (A C T G - N): datasets.sequences.one_hot_nt's
# mapping, read on the RAW byte, with everything else -> N
_NT_LUT = np.full(256, 5, np.int64)
for _b, _c in BASE_TO_COLUMN.items():
    _NT_LUT[ord(_b)] = _c
# base byte (upper-cased) -> codon digit: A C G T = 0..3, '-' = -1, else -2
_CODON_LUT = np.full(256, -2, np.int64)
for _i, _b in enumerate("ACGT"):
    _CODON_LUT[ord(_b)] = _i
_CODON_LUT[ord("-")] = -1
_CODON_RES = np.array([RESIDUES.index(CODON_TABLE[a + b + c])
                       for a in "ACGT" for b in "ACGT" for c in "ACGT"], np.int64)


def regulatory_for(drugs, loci):
    """Promoter windows a run loads: the drugs' WHO regions that are also loaded
    loci (union, in first-seen order)."""
    out = []
    for d in drugs:
        for r in DRUG_TO_REGULATORY.get(d.upper(), []):
            if r in set(loci) and r not in out:
                out.append(r)
    return out


# A FASTA that a dying job wrote only part of is INVISIBLE downstream:
# load_sequence_df reindexes on the phenotype cohort and fills the missing rows
# with "", which tokenizes as "no variants", i.e. the isolate reads as H37Rv.
# Three such files sat on disk for six days in Oct 2026 (curated_fix embB/ethA/
# rrs, aligned_atg whiB6, gene_msa dnaA). Every real alignment covers 100% of
# the cohort, so refuse anything that does not.
MIN_COVERAGE = 0.99


def _check_cohort(gene_seqs, isolates, loci, genotype_dir, minimum=MIN_COVERAGE):
    """Every locus must carry (nearly) the same isolates as the best-covered one.

    A truncated FASTA does not leave empty rows -- the isolates it never got to
    are simply absent, so the cohort silently SHRINKS instead (a 1,000-record
    pncA trains on 1,000 isolates and reports it as the cohort). Loci are each
    other's reference: they are all extracted from the same VCF list."""
    have = {l: int(gene_seqs[l].reindex(isolates).notna().sum()) if l in gene_seqs else 0
            for l in loci}
    best = max(have.values()) if have else 0
    short = {l: n for l, n in have.items() if best and n < minimum * best}
    if short:
        detail = ", ".join(f"{l} {n}" for l, n in sorted(short.items(), key=lambda t: t[1]))
        raise ValueError(
            f"alignment(s) in {genotype_dir} cover fewer isolates than the rest: {detail} "
            f"(best-covered locus has {best}). A truncated alignment silently shrinks the cohort. "
            f"Check it with regulatory_msa/validate_alignments.py.")


def _check_coverage(locus, kind, path, lens, width, n, minimum=MIN_COVERAGE):
    cov = float((lens == width).mean()) if n else 1.0
    if cov < minimum:
        raise ValueError(
            f"{locus} ({kind}): only {int((lens == width).sum())} of {n} cohort isolates have a "
            f"{width}-column row in {path} ({cov:.3%}). A truncated or ragged alignment would be "
            f"read as 'no variants' for the isolates it is missing. Check it with "
            f"regulatory_msa/validate_alignments.py.")
    return cov


def _char_matrix(seqs):
    """list of N strings -> (raw (N, W) uint8, upper (N, W) uint8, lengths (N,))."""
    lens = np.array([len(s) for s in seqs], np.int64)
    W = max(int(lens.max()) if len(lens) else 0, 1)
    raw = np.zeros((len(seqs), W), np.uint8)
    for i, s in enumerate(seqs):
        if s:
            raw[i, :len(s)] = np.frombuffer(s.encode("ascii", "replace"), np.uint8)
    up = raw.copy()
    low = (up >= 97) & (up <= 122)
    up[low] -= 32
    return raw, up, lens


def _occupied(up, lens, ref):
    """(N, W) bool: the column differs from H37Rv (datasets.sequences.delta_one_hot_nt)."""
    N, W = up.shape
    occ = np.ones((N, W), bool)
    if ref:
        rb = np.frombuffer(ref.encode("ascii", "replace"), np.uint8)
        n = min(W, len(rb))
        occ[:, :n] = up[:, :n] != rb[None, :n]
    return occ & (np.arange(W)[None, :] < lens[:, None])


def _translate(up, lens, rows, cols3):
    """Residue index of each (row, 3 columns) codon; columns past a row's end read '-'."""
    ch = up[rows[:, None], cols3]
    ch = np.where(cols3 < lens[rows][:, None], ch, ord("-"))
    d = _CODON_LUT[ch]
    out = _CODON_RES[np.clip(d, 0, 3) @ np.array([16, 4, 1])]
    out = np.where((d == -2).any(1), R_UNK, out)
    return np.where((d == -1).any(1), R_GAP, out)


def _frame_maps(p):
    """(coord, codon, phase, keep, note) over one locus's merged column axis.

    Every column is placed by its GENOME position. A block's reference row
    degaps to an exact genome substring (``datasets.cds.block_frame``), so
    column -> count of reference bases -> genome position is exact, and the
    gene record and its promoter window land on one axis measured from the
    start codon rather than assumed to abut it. Three things follow:

    * a promoter variant sits at its true distance — inhA's window is 732 bp
      upstream of the inhA CDS, being the fabG1 promoter, not adjacent to it;
    * a column the record already covers is dropped from the promoter block
      (``keep``), so the 30-130 bp the two share on 10 of 16 loci produce one
      token, not two. That includes the 30 bp of CDS most WHO windows reach
      into: the record covers them, so the record's token carries their base
      and residue, and the kept promoter columns are all upstream of it;
    * the kept axis is monotonic in coordinate, 5' -> 3'.

    Codon and phase count reference bases from the start codon, which is the
    index `datasets/protein.py` translates on. Without a genome span the
    fallback is v1's convention: the promoter counts backwards from the end
    of its own window and nothing is deduplicated.
    """
    total, reg_len = p["total"], p["reg_len"]
    coord = np.zeros(total, dtype=np.float32)
    codon = np.full(total, -1, dtype=np.int64)
    phase = np.full(total, -1, dtype=np.int64)
    keep = np.ones(total, dtype=bool)
    note = {"genome_anchored": False, "cds_start": None, "has_cds": None,
            "n_codons": None, "duplicate_columns": 0, "promoter_offset": None}

    def bases_before(b):
        """(n, is_base): reference bases strictly before each column."""
        L = b["length"]
        is_base = np.ones(L, dtype=np.int64)
        gaps = np.asarray(b["frame"].get("ref_gap_cols", ()), dtype=np.int64)
        if gaps.size:
            is_base[gaps[gaps < L]] = 0
        return np.concatenate([[0], np.cumsum(is_base)])[:L], is_base

    def positions(b):
        """(genome position per column, is_base), or (None, is_base)."""
        n, is_base = bases_before(b)
        f = b["frame"]
        if f.get("genome_start") is None:
            return None, is_base
        return (f["genome_start"] + n if f["strand"] == "+"
                else f["genome_end"] - 1 - n), is_base

    rec = next((b for b in p["nt"] if b["modality"] == "dna"), None)
    prom = next((b for b in p["nt"] if b["modality"] == "regulatory"), None)

    # --- the origin: the start codon of this locus's coding record -------
    rec_pos = rec_is_base = None
    cds_len = 0
    if rec is not None:
        rec_pos, rec_is_base = positions(rec)
        n, _ = bases_before(rec)
        f = rec["frame"]
        start = int(f.get("cds_start", 0))
        end = int(f.get("cds_end", rec["length"]))
        # an rRNA gene (rrs, rrl) has a start -- the origin -- but no reading
        # frame: its tokens carry a base and a position and nothing else,
        # whatever the protein block's whole-record fallback translated.
        # Frames written without the key predate it and were all coding.
        has_cds = bool(f.get("has_cds", True))
        n_full = np.concatenate([n, [n[-1] + rec_is_base[-1]]])  # bases before col, 0..L
        cds_len = int(n_full[min(end, rec["length"])] - n_full[min(start, rec["length"])]) \
            if has_cds else 0
        note["cds_start"] = start
        note["has_cds"] = has_cds
        # reference codons in the translated window, the terminal stop
        # included — the width of a reference-frame protein block
        n_win = int(n_full[min(end, rec["length"])] - n_full[min(start, rec["length"])])
        note["n_codons"] = n_win // 3
        rel = n - n[min(start, rec["length"] - 1)]          # bases from the start codon
        lo, L = rec["offset"], rec["length"]
        coord[lo:lo + L] = rel / 3.0
        inside = ((np.arange(L) >= start) & (np.arange(L) < end) & (rec_is_base > 0)
                  & has_cds)
        codon[lo:lo + L] = np.where(inside, rel // 3, -1)
        phase[lo:lo + L] = np.where(inside, rel % 3, -1)
        if rec_pos is not None:
            note["genome_anchored"] = True
            g0 = rec_pos[min(start, L - 1)]
            strand = rec["frame"]["strand"]

    # --- the promoter, placed against that origin ------------------------
    if prom is not None:
        lo, L = prom["offset"], prom["length"]
        prom_pos, prom_is_base = positions(prom)
        anchored = (rec_pos is not None and prom_pos is not None
                    and prom["frame"]["strand"] == rec["frame"]["strand"])
        if anchored:
            off = (prom_pos - g0) if strand == "+" else (g0 - prom_pos)
            coord[lo:lo + L] = off / 3.0
            # the WHO window's 3' flank reaches into the CDS on most loci;
            # those columns are coding and are read as such
            inside = (off >= 0) & (off < cds_len) & (prom_is_base > 0)
            codon[lo:lo + L] = np.where(inside, off // 3, -1)
            phase[lo:lo + L] = np.where(inside, off % 3, -1)
            # one base, one token: drop what the record already covers. Gap
            # columns go too -- a gap sits at the position of the base after
            # it, so one inside the shared span is the record's to report,
            # and keeping it would put a promoter token inside the record's
            # stretch of the axis.
            f = rec["frame"]
            dup = (prom_pos >= f["genome_start"]) & (prom_pos < f["genome_end"])
            keep[lo:lo + L] = ~dup
            note["duplicate_columns"] = int(dup.sum())
            note["promoter_offset"] = float(off[0] / 3.0)
        else:
            note["genome_anchored"] = False
            coord[lo:lo + L] = (np.arange(L, dtype=np.float32) - L) / 3.0
    return coord, codon, phase, keep, note


@dataclass
class LocusGeometry:
    """Everything fixed per locus, cut from the frame maps."""
    name: str
    kind: str                    # 'protein' | 'rrna' | 'noncoding'
    reg_len: int
    total: int
    kept: int                    # columns left after the promoter/record de-dup
    coord: np.ndarray            # (total,) codons from the start codon
    codon: np.ndarray            # (total,) -1 off a reference codon
    phase: np.ndarray
    keep: np.ndarray
    codon_cols: np.ndarray       # (n_codons, 3) record columns of each reference codon
    ref_res: np.ndarray          # (n_codons,) H37Rv residue index
    window: tuple                # record columns of the CDS / rRNA gene, [start, end)
    window_bases: int
    note: dict


def locus_geometry(locus, dna_len, ref, genotype_dir, reg_len=0, regulatory_dir=None):
    fr_dna = cds.block_frame(genotype_dir, locus, dna_len, translated=True)
    nt = []
    if reg_len:
        nt.append({"modality": "regulatory", "length": reg_len, "offset": 0,
                   "frame": cds.block_frame(regulatory_dir, locus, reg_len)})
    nt.append({"modality": "dna", "length": dna_len, "offset": reg_len, "frame": fr_dna})
    p = {"locus": locus, "nt": nt, "total": reg_len + dna_len, "reg_len": reg_len}
    coord, codon, phase, keep, note = _frame_maps(p)
    start, end = int(fr_dna.get("cds_start", 0)), int(fr_dna.get("cds_end", dna_len))
    has_cds = bool(fr_dna.get("has_cds", True)) and "cds_start" in fr_dna
    base_cols = [c for c in range(start, min(end, len(ref))) if ref[c] != "-"]
    if has_cds:
        n_cod = len(base_cols) // 3
        codon_cols = np.array(base_cols[:3 * n_cod], np.int64).reshape(n_cod, 3)
        rb = np.frombuffer(ref.encode("ascii"), np.uint8)
        ref_res = _translate(rb[None, :], np.array([len(ref)]),
                             np.zeros(n_cod, np.int64), codon_cols)
        kind = "protein"
    else:
        codon_cols, ref_res = np.zeros((0, 3), np.int64), np.zeros(0, np.int64)
        kind = "rrna" if "cds_start" in fr_dna else "noncoding"
    return LocusGeometry(name=locus, kind=kind, reg_len=reg_len, total=reg_len + dna_len,
                         kept=int(keep.sum()), coord=coord, codon=codon, phase=phase,
                         keep=keep, codon_cols=codon_cols, ref_res=ref_res,
                         window=(start, end), window_bases=len(base_cols), note=note)


@dataclass
class TokenSet:
    """The cached tokens of the 19 loci over every phenotyped isolate."""
    feat: np.ndarray             # (N, L, V, C_BASE) float32
    coord: np.ndarray            # (N, L, V) float32, codons from the start codon
    valid: np.ndarray            # (N, L, V) bool
    site_pos: np.ndarray         # (N, L, V) int16, index into the distance tables
    n_occ: np.ndarray            # (N, L) int32, changed columns before the cap
    kept: np.ndarray             # (L,) int32, columns per locus (the [NULL] fraction's denominator)
    Y: np.ndarray                # (N, n_drugs) float32, 0 = R, 1 = S, -1 = missing
    isolate_ids: list
    drugs: list
    loci: list
    locus_kind: list
    regulatory: list
    config: dict

    ARRAYS = ("feat", "coord", "valid", "site_pos", "n_occ", "kept", "Y")

    @property
    def n(self):
        return len(self.isolate_ids)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")   # concurrent builders
        np.savez(tmp, **{k: getattr(self, k) for k in self.ARRAYS},
                 meta=np.array(json.dumps({
                     "isolate_ids": self.isolate_ids, "drugs": self.drugs, "loci": self.loci,
                     "locus_kind": self.locus_kind, "regulatory": self.regulatory,
                     "config": self.config})))
        tmp.rename(path)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(**{k: z[k] for k in cls.ARRAYS},
                   isolate_ids=meta["isolate_ids"], drugs=meta["drugs"], loci=meta["loci"],
                   locus_kind=meta["locus_kind"], regulatory=meta["regulatory"],
                   config=meta["config"])


def build_tokens(drugs, genotype_dir, phenotype_csv, regulatory_dir, regulatory, verbose=True):
    """Tokenize every phenotyped isolate over LOCI_19 (+ the `regulatory` promoter
    windows, each keyed to the locus of the same name)."""
    t0 = time.time()
    loci = LOCI_19
    drugs = [d.upper() for d in drugs]
    gene_seqs, found = load_sequence_df(genotype_dir, loci)
    missing = [l for l in loci if l not in found]
    if missing:
        raise ValueError(f"no aligned FASTA in {genotype_dir} for {missing}")
    phen = load_phenotype(phenotype_csv)
    isolates = list(gene_seqs.index.intersection(phen.index))
    _check_cohort(gene_seqs, isolates, loci, genotype_dir)
    gene_seqs = gene_seqs.reindex(isolates).fillna("")
    reg_df, reg_found = (load_sequence_df(regulatory_dir, regulatory) if regulatory
                         else (None, []))
    if reg_df is not None:
        reg_df = reg_df.reindex(isolates).fillna("")
    _, Y = tb.rs_encoding_to_numeric(phen.loc[isolates], drugs)
    N, L, V = len(isolates), len(loci), MAX_VARIANTS

    width = C_BASE
    feat = np.zeros((N, L, V, width), np.float32)
    coord = np.zeros((N, L, V), np.float32)
    valid = np.zeros((N, L, V), bool)
    site_t = np.full((N, L, V), -1, np.int16)
    n_occ = np.zeros((N, L), np.int32)
    kept = np.zeros(L, np.int32)
    geoms = []
    for li, locus in enumerate(loci):
        ref = reference_row(genotype_dir, locus)
        if not ref:
            raise ValueError(f"{locus}: no H37Rv row in {genotype_dir}")
        raw, up, lens = _char_matrix(list(gene_seqs[locus].values))
        _check_coverage(locus, "gene", Path(genotype_dir) / f"{locus}.fasta", lens, len(ref), N)
        occ_d = _occupied(up, lens, ref)
        if locus in reg_found:
            rref = reference_row(regulatory_dir, locus)
            rraw, rup, rlens = _char_matrix(list(reg_df[locus].values))
            _check_coverage(locus, "promoter", Path(regulatory_dir) / f"{locus}.fasta",
                            rlens, len(rref), N)
            occ_r = _occupied(rup, rlens, rref)
        else:
            rraw = np.zeros((N, 0), np.uint8)
            occ_r = np.zeros((N, 0), bool)
        g = locus_geometry(locus, raw.shape[1], ref, genotype_dir, occ_r.shape[1], regulatory_dir)
        geoms.append(g)
        kept[li] = g.kept

        occ = np.concatenate([occ_r, occ_d], 1) & g.keep[None, :]
        rows, cols = np.nonzero(occ)
        R = g.reg_len
        in_rec = cols >= R
        rc = cols - R                                     # record column (valid where in_rec)
        base = np.where(in_rec, _NT_LUT[raw[rows, np.clip(rc, 0, raw.shape[1] - 1)]],
                        _NT_LUT[rraw[rows, np.clip(cols, 0, max(R - 1, 0))]] if R else 0)
        cod = g.codon[cols]
        ph = g.phase[cols]
        n_cod = len(g.codon_cols)
        on_codon = in_rec & (cod >= 0) & (cod < n_cod)
        alt = np.full(len(rows), -1, np.int64)
        refr = np.full(len(rows), -1, np.int64)
        if on_codon.any():
            alt[on_codon] = _translate(up, lens, rows[on_codon], g.codon_cols[cod[on_codon]])
            refr[on_codon] = g.ref_res[cod[on_codon]]
        # consequence: per codon; an insertion inside the CDS is 'other' too
        cons = np.full(len(rows), -1, np.int64)
        if g.kind == "protein":
            s, e = g.window
            ins = in_rec & (rc >= s) & (rc < e) & (cod < 0)
            cons[ins] = OTHER
            a, r = alt[on_codon], refr[on_codon]
            c = np.full(len(a), MISSENSE)
            c[a == r] = SYN
            c[(a == R_STOP) & (r != R_STOP)] = NONSENSE
            c[(a == R_GAP) | (a == R_UNK) | ((r == R_STOP) & (a != R_STOP))] = OTHER
            cons[on_codon] = c
        # synonymous changes are not tokens
        k = cons != SYN
        rows, cols, in_rec, rc, base, cod, ph, on_codon, alt, refr, cons = (
            x[k] for x in (rows, cols, in_rec, rc, base, cod, ph, on_codon, alt, refr, cons))
        n_occ[:, li] = np.bincount(rows, minlength=N)
        start = np.searchsorted(rows, np.arange(N))
        rank = np.arange(len(rows)) - start[rows]
        sel = rank < V
        rows, slot = rows[sel], rank[sel]
        cols, in_rec, rc, base, cod, ph, on_codon, alt, refr, cons = (
            x[sel] for x in (cols, in_rec, rc, base, cod, ph, on_codon, alt, refr, cons))

        f = np.zeros((len(rows), width), np.float32)
        nt0 = np.where(in_rec, SLOTS["dna"][0], SLOTS["regulatory"][0])
        f[np.arange(len(rows)), nt0 + base] = 1.0
        bio = np.where(on_codon[:, None], PROPS[np.clip(alt, 0, None)], 0.0)
        f[:, SLOTS["bio"][0]:SLOTS["bio"][1]] = bio
        f[:, SLOTS["d_bio"][0]:SLOTS["d_bio"][1]] = np.where(
            on_codon[:, None], bio - PROPS[np.clip(refr, 0, None)], 0.0)
        hc = cons >= 0
        f[np.nonzero(hc)[0], SLOTS["consequence"][0] + cons[hc]] = 1.0
        coding = cod >= 0
        f[np.nonzero(coding)[0], SLOTS["phase"][0] + ph[coding]] = 1.0
        f[np.nonzero(on_codon)[0], SLOTS["residue"][0] + alt[on_codon]] = 1.0

        # where the token sits in the distance tables
        rel = np.rint(g.coord[cols] * 3).astype(np.int64)
        sp = np.full(len(rows), -1, np.int64)
        if g.kind == "protein":
            ok = in_rec & (rel >= 0) & (rel < 3 * n_cod)
            sp[ok] = rel[ok] // 3
        elif g.kind == "rrna":
            ok = in_rec & (rel >= 0) & (rel < g.window_bases)
            sp[ok] = rel[ok]

        feat[rows, li, slot] = f
        coord[rows, li, slot] = g.coord[cols]
        valid[rows, li, slot] = True
        site_t[rows, li, slot] = sp
        if verbose:
            print(f"  [tokens] {locus:6s} {g.kind:8s} cols={g.total:5d} kept={g.kept:5d} "
                  f"reg={R:4d} tokens={int(sel.sum()):7d} over-cap isolates="
                  f"{int((n_occ[:, li] > V).sum())}", flush=True)

    # drop_synonymous / keep_residue are always on; they stay in the config so it
    # (and the result JSON's "tokens") reads as it always has
    config = {"loci": list(loci), "regulatory": [r for r in regulatory if r in reg_found],
              "drugs": drugs, "max_variants": V, "drop_synonymous": True,
              "keep_residue": True, "token_channels": int(width),
              "genotype_dir": str(genotype_dir), "regulatory_dir": str(regulatory_dir),
              "phenotype_csv": str(phenotype_csv), "n_isolates": N,
              "frames": {g.name: g.note for g in geoms},
              "built": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1)}
    if verbose:
        print(f"  [tokens] {N} isolates x {L} loci, {config['seconds']} s", flush=True)
    return TokenSet(feat=feat, coord=coord, valid=valid, site_pos=site_t,
                    n_occ=n_occ, kept=kept, Y=Y.astype(np.float32), isolate_ids=isolates,
                    drugs=drugs, loci=list(loci), locus_kind=[g.kind for g in geoms],
                    regulatory=config["regulatory"], config=config)


def cache_key(**cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def load_or_build(cache_dir, drugs, genotype_dir, phenotype_csv, regulatory_dir):
    """The cached TokenSet for these drugs and inputs, building it on first use.

    The key and file name are computed exactly as when synonymous dropping, the
    residue one-hot and the 16-slot cap were options, so caches built then are
    found and reused. The key hashes the path STRINGS, so the same directory
    reached by another path gets its own cache."""
    drugs = [d.upper() for d in drugs]
    regulatory = regulatory_for(drugs, LOCI_19)
    key = cache_key(drugs=drugs, loci=list(LOCI_19), regulatory=regulatory, V=MAX_VARIANTS,
                    drop=True, res=True,
                    geno=str(genotype_dir), reg=str(regulatory_dir),
                    pheno=str(phenotype_csv), version=1)
    tag = drugs[0] if len(drugs) == 1 else f"multidrug{len(drugs)}"
    path = Path(cache_dir) / f"tokens_{tag}_nosyn_res_{key}.npz"
    if path.exists():
        print(f"[tokens] cache {path}", flush=True)
        return TokenSet.load(path)
    print(f"[tokens] building {path}", flush=True)
    ts = build_tokens(drugs, genotype_dir, phenotype_csv, regulatory_dir, regulatory)
    ts.save(path)
    return ts
