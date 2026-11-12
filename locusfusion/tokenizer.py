"""
Aligned FASTAs -> v4 tokens, once, offline.

v3 rebuilt its tokens inside the model from (N, C, L) one-hot blocks on every
batch. v4 builds them once from the aligned sequence strings and caches them:
for 17,942 isolates x 19 loci x 16 slots the whole token set is ~0.5 GB, where
v3's 19-locus all-modality blocks needed a 128 GB job. Everything the model and
the augmentation need is in the cache (TokenSet), so training never touches a
FASTA.

Same tokenizer as v3 where v3 and v4 share a channel, by construction rather
than by re-implementation: the column geometry (genome placement, start-codon
origin, codon and phase per column, the promoter/record de-duplication) is
v3's own `LocusFusionV2Net._frame_maps`, and the base / residue / property
values follow v3's rules exactly (tests/test_parity.py checks them against v3's
`_locus_tokens` on real isolates):

  * a token is a column whose base differs from H37Rv's (upper-cased), within
    the isolate's own sequence; an uncalled base (anything outside A C T G -)
    gets the N channel;
  * per locus the columns run [promoter window | gene record], 5' -> 3', and the
    first `max_variants` occupied ones are kept; [NULL] reports the true count;
  * the residue a coding token reads is the isolate's codon on the H37Rv codon
    grid (datasets.biochem.translate_reference_frame): '-' if a base of it is
    deleted, 'X' if unreadable, '*' for STOP; bio = its three properties (0 for
    '-', 'X', '*'), d_bio = bio minus H37Rv's residue's.

New in v4: `consequence` (synonymous / missense / nonsense / other, see
layout.py) and `site_pos`, the token's index into the binding-site distance
tables (a residue index on protein loci, a nucleotide offset on rrs / rrl, -1
elsewhere). `codon` is stored per token too, which the augmentation uses to
leave already-varied codons alone.

Synonymous candidates (for augment.py) are built here as well, from the same
geometry: every single-base change that keeps H37Rv's residue, at a codon that
is neither the start codon nor the stop, at a base no OTHER locus also carries
(neighbouring loci share record and promoter sequence -- a base there can be
regulatory for the other gene; fabG1 L203L is exactly that), minus the
explicit exclusions in augment_exclusions.json.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bigtb_ref import tb
from datasets import cds
from datasets.biochem import AA_PROPERTY, CODON_TABLE
from datasets.regulatory import DRUG_TO_REGULATORY
from datasets.sequences import BASE_TO_COLUMN, load_phenotype, load_sequence_df, reference_row
from models.locusfusion_v2 import LocusFusionV2Net

from .layout import (C_BASE, MISSENSE, NONSENSE, OTHER, RESIDUE_SLOT, SLOTS, SYN,
                     c_base as _c_base)

HERE = Path(__file__).resolve().parent

# the 19 curated loci (MD-CNN's set; the paper's "19 loci"), in loader order
LOCI_19 = ["eis", "embA", "embB", "embC", "ethA", "ethR", "fabG1", "gid", "gyrA", "gyrB",
           "inhA", "katG", "pncA", "rpoB", "rpoC", "rpsL", "rrl", "rrs", "tlyA"]
ALL_DRUGS = list(tb.DRUG_TO_LOCI)        # label-column order of the multi-drug loader

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
    loci -- load_dataset's default for one drug, load_multidrug_dataset's for
    several (union, in first-seen order)."""
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


def _genome_positions(frame, length):
    """Genome position of each column of a block (-1 for a reference gap), or None."""
    if frame.get("genome_start") is None:
        return None
    is_base = np.ones(length, np.int64)
    gaps = np.asarray(frame.get("ref_gap_cols", ()), np.int64)
    if gaps.size:
        is_base[gaps[gaps < length]] = 0
    n = np.concatenate([[0], np.cumsum(is_base)])[:length]
    pos = frame["genome_start"] + n if frame["strand"] == "+" else frame["genome_end"] - 1 - n
    return np.where(is_base > 0, pos, -1)


@dataclass
class LocusGeometry:
    """Everything fixed per locus, cut from v3's frame maps."""
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
    genome_pos: np.ndarray = None                # (dna_len,) record column -> genome, -1 gap
    ref_codons: list = field(default_factory=list)
    covered: set = field(default_factory=set)   # genome positions this locus carries


def locus_geometry(locus, dna_len, ref, genotype_dir, reg_len=0, regulatory_dir=None):
    fr_dna = cds.block_frame(genotype_dir, locus, dna_len, translated=True)
    nt = []
    if reg_len:
        nt.append({"modality": "regulatory", "length": reg_len, "offset": 0,
                   "frame": cds.block_frame(regulatory_dir, locus, reg_len)})
    nt.append({"modality": "dna", "length": dna_len, "offset": reg_len, "frame": fr_dna})
    p = {"locus": locus, "nt": nt, "total": reg_len + dna_len, "reg_len": reg_len}
    coord, codon, phase, keep, note = LocusFusionV2Net._frame_maps(p)
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
    covered = set()
    for b in nt:
        pos = _genome_positions(b["frame"], b["length"])
        if pos is not None:
            k = keep[b["offset"]:b["offset"] + b["length"]]
            covered |= set(pos[(pos >= 0) & k].tolist())
    return LocusGeometry(name=locus, kind=kind, reg_len=reg_len, total=reg_len + dna_len,
                         kept=int(keep.sum()), coord=coord, codon=codon, phase=phase,
                         keep=keep, codon_cols=codon_cols, ref_res=ref_res,
                         window=(start, end), window_bases=len(base_cols), note=note,
                         genome_pos=_genome_positions(fr_dna, dna_len),
                         ref_codons=["".join(ref[c] for c in cc) for cc in codon_cols],
                         covered=covered)


@dataclass
class TokenSet:
    """The cached tokens of one locus set over every phenotyped isolate."""
    feat: np.ndarray             # (N, L, V, C_BASE) float32
    coord: np.ndarray            # (N, L, V) float32, codons from the start codon
    valid: np.ndarray            # (N, L, V) bool
    codon: np.ndarray            # (N, L, V) int16, -1 off a reference codon
    site_pos: np.ndarray         # (N, L, V) int16, index into the distance tables
    n_occ: np.ndarray            # (N, L) int32, changed columns before the cap
    kept: np.ndarray             # (L,) int32, columns per locus (the [NULL] fraction's denominator)
    Y: np.ndarray                # (N, n_drugs) float32, 0 = R, 1 = S, -1 = missing
    isolate_ids: list
    drugs: list
    loci: list
    locus_kind: list
    regulatory: list
    cand: dict                   # synonymous candidates (augment.py)
    config: dict

    ARRAYS = ("feat", "coord", "valid", "codon", "site_pos", "n_occ", "kept", "Y")

    @property
    def n(self):
        return len(self.isolate_ids)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")   # concurrent builders
        np.savez(tmp, **{k: getattr(self, k) for k in self.ARRAYS},
                 **{f"cand_{k}": v for k, v in self.cand.items()},
                 meta=np.array(json.dumps({
                     "isolate_ids": self.isolate_ids, "drugs": self.drugs, "loci": self.loci,
                     "locus_kind": self.locus_kind, "regulatory": self.regulatory,
                     "config": self.config})))
        tmp.rename(path)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        cand = {k[5:]: z[k] for k in z.files if k.startswith("cand_")}
        return cls(**{k: z[k] for k in cls.ARRAYS}, cand=cand,
                   isolate_ids=meta["isolate_ids"], drugs=meta["drugs"], loci=meta["loci"],
                   locus_kind=meta["locus_kind"], regulatory=meta["regulatory"],
                   config=meta["config"])


def _exclusions(path=HERE / "augment_exclusions.json"):
    rows = json.loads(Path(path).read_text())["exclude"]
    return {(r["locus"], int(r["codon"]) - 1) for r in rows}


def synonymous_candidates(geoms, exclude, keep_residue=False):
    """Every allowed single-base synonymous substitution, as ready-made tokens."""
    out = {k: [] for k in ("locus", "codon", "phase", "col", "base", "coord", "site_pos")}
    for li, g in enumerate(geoms):
        if g.kind != "protein":
            continue
        others = set().union(*(h.covered for h in geoms if h is not g))
        n_cod = len(g.codon_cols)
        for k in range(1, n_cod - 1):               # not the start codon, not the stop
            if (g.name, k) in exclude or g.ref_res[k] >= R_STOP:
                continue
            for ph in range(3):
                col = int(g.codon_cols[k, ph])
                if g.genome_pos is not None and g.genome_pos[col] in others:
                    continue
                for b in "ACGT":
                    cod = list(g.ref_codons[k])
                    if cod[ph] == b:
                        continue
                    cod[ph] = b
                    if RESIDUES.index(CODON_TABLE["".join(cod)]) != g.ref_res[k]:
                        continue
                    out["locus"].append(li)
                    out["codon"].append(k)
                    out["phase"].append(ph)
                    out["col"].append(col)
                    out["base"].append(BASE_TO_COLUMN[b])
                    out["coord"].append(float(g.coord[g.reg_len + col]))
                    out["site_pos"].append(k)
    M = len(out["locus"])
    feat = np.zeros((M, _c_base(keep_residue)), np.float32)
    if M:
        res = np.array([geoms[l].ref_res[k] for l, k in zip(out["locus"], out["codon"])])
        feat[np.arange(M), SLOTS["dna"][0] + np.array(out["base"])] = 1.0
        feat[:, SLOTS["bio"][0]:SLOTS["bio"][1]] = PROPS[res]      # d_bio stays 0
        feat[:, SLOTS["consequence"][0] + SYN] = 1.0
        feat[np.arange(M), SLOTS["phase"][0] + np.array(out["phase"])] = 1.0
        if keep_residue:                       # synonymous: H37Rv's own residue
            feat[np.arange(M), RESIDUE_SLOT[0] + res] = 1.0
    return {"locus": np.array(out["locus"], np.int16), "codon": np.array(out["codon"], np.int16),
            "col": np.array(out["col"], np.int32), "coord": np.array(out["coord"], np.float32),
            "site_pos": np.array(out["site_pos"], np.int16), "feat": feat}


def build_tokens(drugs, genotype_dir, phenotype_csv, regulatory_dir, loci=LOCI_19,
                 regulatory=None, max_variants=16, drop_synonymous=False,
                 keep_residue=False, verbose=True):
    """Tokenize every phenotyped isolate over `loci` (+ promoter windows).

    regulatory : promoter windows to load (each keyed to the locus of the same
                 name); default regulatory_for(drugs, loci).
    drop_synonymous : leave synonymous changes out entirely, before the cap --
                 the exact-invariance arm.
    keep_residue : also carry v3's 23-channel residue one-hot (layout.RESIDUE_SLOT),
                 so the token is v3's plus consequence and distance.
    """
    t0 = time.time()
    drugs = [d.upper() for d in drugs]
    regulatory = list(regulatory) if regulatory is not None else regulatory_for(drugs, loci)
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
    N, L, V = len(isolates), len(loci), int(max_variants)

    width = _c_base(keep_residue)
    feat = np.zeros((N, L, V, width), np.float32)
    coord = np.zeros((N, L, V), np.float32)
    valid = np.zeros((N, L, V), bool)
    codon_t = np.full((N, L, V), -1, np.int16)
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
        if drop_synonymous:
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
        if keep_residue:       # v3's protein slot: the residue the codon now reads
            f[np.nonzero(on_codon)[0], RESIDUE_SLOT[0] + alt[on_codon]] = 1.0

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
        codon_t[rows, li, slot] = np.where(on_codon, cod, -1)
        site_t[rows, li, slot] = sp
        if verbose:
            print(f"  [tokens] {locus:6s} {g.kind:8s} cols={g.total:5d} kept={g.kept:5d} "
                  f"reg={R:4d} tokens={int(sel.sum()):7d} over-cap isolates="
                  f"{int((n_occ[:, li] > V).sum())}", flush=True)

    cand = synonymous_candidates(geoms, _exclusions(), keep_residue)
    config = {"loci": list(loci), "regulatory": [r for r in regulatory if r in reg_found],
              "drugs": drugs, "max_variants": V, "drop_synonymous": bool(drop_synonymous),
              "keep_residue": bool(keep_residue), "token_channels": int(width),
              "genotype_dir": str(genotype_dir), "regulatory_dir": str(regulatory_dir),
              "phenotype_csv": str(phenotype_csv), "n_isolates": N,
              "n_synonymous_candidates": int(len(cand["locus"])),
              "frames": {g.name: g.note for g in geoms},
              "built": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1)}
    if verbose:
        print(f"  [tokens] {N} isolates x {L} loci, {len(cand['locus'])} synonymous "
              f"candidates, {config['seconds']} s", flush=True)
    return TokenSet(feat=feat, coord=coord, valid=valid, codon=codon_t, site_pos=site_t,
                    n_occ=n_occ, kept=kept, Y=Y.astype(np.float32), isolate_ids=isolates,
                    drugs=drugs, loci=list(loci), locus_kind=[g.kind for g in geoms],
                    regulatory=config["regulatory"], cand=cand, config=config)


def cache_key(**cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def load_or_build(cache_dir, drugs, genotype_dir, phenotype_csv, regulatory_dir,
                  loci=LOCI_19, regulatory=None, max_variants=16, drop_synonymous=False,
                  keep_residue=False):
    """The cached TokenSet for this configuration, building it on first use."""
    drugs = [d.upper() for d in drugs]
    regulatory = list(regulatory) if regulatory is not None else regulatory_for(drugs, loci)
    key = cache_key(drugs=drugs, loci=list(loci), regulatory=regulatory, V=int(max_variants),
                    drop=bool(drop_synonymous), res=bool(keep_residue),
                    geno=str(genotype_dir), reg=str(regulatory_dir),
                    pheno=str(phenotype_csv), version=1)
    tag = drugs[0] if len(drugs) == 1 else f"multidrug{len(drugs)}"
    path = (Path(cache_dir) / f"tokens_{tag}{'_nosyn' if drop_synonymous else ''}"
            f"{'_res' if keep_residue else ''}_{key}.npz")
    if path.exists():
        print(f"[tokens] cache {path}", flush=True)
        return TokenSet.load(path)
    print(f"[tokens] building {path}", flush=True)
    ts = build_tokens(drugs, genotype_dir, phenotype_csv, regulatory_dir, loci=loci,
                      regulatory=regulatory, max_variants=max_variants,
                      drop_synonymous=drop_synonymous, keep_residue=keep_residue)
    ts.save(path)
    return ts
