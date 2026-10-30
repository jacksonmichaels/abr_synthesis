"""Run the LocusFusion v3 tokenizer on real ISONIAZID data and keep one isolate.

    python scripts/tokenization_example_extract.py        # ~ a few GB RAM; batch or allocation

Writes results/figures/manuscript/tokenization_example.json, which
scripts/tokenization_example.py draws. Kept separate so the figure can be
restyled without reloading 17k isolates.

v3 = LocusFusionV2Net(unknown_symbols=True) on load_dataset(..., delta=True,
protein_frame="reference", unknown_symbols=True), the same inputs the v3 sweep
trained on. The per-drug locus set for ISONIAZID is katG + inhA (with inhA's
promoter window). The isolate is chosen to show every kind of token at once:
katG S315T, a promoter base, a synonymous change and an uncalled base, with few
enough tokens to draw (the uncalled base and S315T weighted double: they are
the two the paper discusses). Every number in the JSON is read off the model's own
feature tensor; nothing is re-derived.
"""
from pathlib import Path
import json
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR  # noqa: E402
from datasets import biochem, cds, load_dataset  # noqa: E402
from datasets.sequences import NT_CHANNELS_WITH_UNKNOWN, reference_row  # noqa: E402
from models import LocusFusionV2Net  # noqa: E402

DRUG = "ISONIAZID"
OUT = PROJECT / "results" / "figures" / "manuscript" / "tokenization_example.json"
CHUNK = 2000

data = load_dataset(DRUG, ["dna", "protein", "biophysical", "regulatory"],
                    REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, regulatory_dir=REAL_REGULATORY_DIR,
                    per_modality_branch=False, delta=True, protein_frame="reference",
                    unknown_symbols=True, verbose=False)
net = LocusFusionV2Net.from_blocks(data.blocks, n_drugs=1, unknown_symbols=True).eval()
L = net.layout
RES = [str(r) for r in list(biochem.RESIDUES_WITH_STOP) + ["-", "X"]]
NT = NT_CHANNELS_WITH_UNKNOWN
print("blocks:", [b.name for b in data.blocks], "c_tok", net.c_tok, "loci", net.loci)

REF_NT = {g: reference_row(REAL_GENOTYPE_DIR, g) for g in net.loci}
REF_REG = {g: reference_row(REAL_REGULATORY_DIR, g) for g in net.loci}
REF_AA = {g: biochem.translate_reference_frame(REF_NT[g], REF_NT[g],
                                               cds.cds_columns(REAL_GENOTYPE_DIR, g))
          for g in net.loci if net.frames[g]["has_cds"]}


def letter(v, alphabet):
    j = int(np.argmax(v))
    return alphabet[j] if v[j] > 0 else None


def decode(feat, rep, b, li, locus):
    """Every token of isolate b at locus li, read off the feature tensor."""
    s = L["slots"]
    out = []
    for t in range(feat.shape[2]):
        f = feat[b, li, t].numpy()
        if t == 0:
            out.append({"kind": "null", "feat": f.round(4).tolist()})
            continue
        if not bool(rep["valid"][b, t - 1]):
            continue
        region = "cds" if int(rep["region"][b, t - 1]) else "promoter"
        col = int(rep["column_in_block"][b, t - 1])
        codon = int(rep["codon"][b, t - 1])
        coord = float(rep["coord"][b, t - 1])
        base = letter(f[slice(*s["dna"])], NT) or letter(f[slice(*s["regulatory"])], NT)
        ref = (REF_NT if region == "cds" else REF_REG)[locus][col]
        aa = letter(f[slice(*s["protein"])], RES)
        ref_aa = REF_AA[locus][codon] if (aa and locus in REF_AA and codon >= 0) else None
        out.append({"kind": "variant", "region": region, "column": col, "codon": codon,
                    "coord_codons": coord, "ref": ref, "alt": base, "ref_aa": ref_aa, "alt_aa": aa,
                    "phase": letter(f[L["f_phase"]:L["f_phase"] + 3], [0, 1, 2]),
                    "feat": f.round(4).tolist()})
    return out


n = len(data.isolate_ids)
best = None
for a in range(0, n, CHUNK):
    xs = [torch.from_numpy(blk.array[a:a + CHUNK]).float() for blk in data.blocks]
    with torch.no_grad():
        feat = net._locus_tokens(xs)[0]
    rep = net.variant_report(xs)
    for b in range(feat.shape[0]):
        if data.y[a + b] != 0:                      # resistant isolates only
            continue
        n_tok = sum(int(r["valid"][b].sum()) for r in rep)
        if not 4 <= n_tok <= 10:
            continue
        toks = {loc: decode(feat, rep[li], b, li, loc) for li, loc in enumerate(net.loci)}
        var = [t for ts in toks.values() for t in ts if t["kind"] == "variant"]
        katg = [t for t in toks.get("katG", []) if t["kind"] == "variant"]
        has = {
            "s315": any(t["codon"] == 314 and t["alt_aa"] not in (None, "S") for t in katg),
            "promoter": any(t["region"] == "promoter" for t in var),
            "unknown": any(t["alt"] == "N" for t in var),
            "synonymous": any(t["alt_aa"] and t["alt_aa"] == t["ref_aa"] and t["alt"] != "N" for t in var),
            "missense": sum(bool(t["alt_aa"]) and t["alt_aa"] != t["ref_aa"] and t["alt"] != "N"
                            for t in var) >= 2,
        }
        score = sum(has.values()) + has["unknown"] + has["s315"] - 0.05 * n_tok
        if best is None or score > best[0]:
            best = (score, a + b, toks, has, n_tok)
    print(f"{min(a + CHUNK, n)}/{n}  best so far {best[0] if best else None}", flush=True)

score, i, toks, has, n_tok = best
print("chose", data.isolate_ids[i], has, n_tok, "tokens")
OUT.write_text(json.dumps({
    "drug": DRUG, "isolate": data.isolate_ids[i], "phenotype": "R", "criteria": has,
    "loci": net.loci, "c_tok": net.c_tok, "max_variants": net.max_variants,
    "layout": {"slots": {k: list(v) for k, v in L["slots"].items()}, "d_bio": list(L["d_bio"]),
               "f_is_null": L["f_is_null"], "f_phase": L["f_phase"]},
    "alphabets": {"nt": NT, "protein": RES},
    "locus_lengths": {g: {"cds": len(REF_NT[g]), "promoter": len(REF_REG[g] or "")} for g in net.loci},
    "frames": {g: {"has_cds": bool(net.frames[g]["has_cds"])} for g in net.loci},
    "tokens": toks}, indent=1))
print("wrote", OUT)
