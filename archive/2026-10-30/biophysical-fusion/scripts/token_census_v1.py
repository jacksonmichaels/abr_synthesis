"""How often does locusfusion (v1) hit its 16-token cap? The v1 tokenizer caps
each (locus, coordinate stream) separately — nt (DNA), aa (protein +
biophysical, co-indexed) and reg (promoter) — so this counts occupied columns
per (isolate, locus, stream) on delta input, as `_stream_tokens` does.

Protein occupancy stands in for the aa stream: a biophysical column can only
differ from the reference where the residue does. Loaded one modality at a time
to keep memory down. Companion to scripts/token_census.py (which is v2's merged
axis).

    sbatch scripts/sbatch/token_census_v1.sh
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR  # noqa: E402
from datasets import ALL_DRUGS, load_multidrug_dataset, loci_on_disk  # noqa: E402
from datasets.loader import DRUG_TO_LOCI  # noqa: E402

CAP = 16
UNCOVERED_FRAC = 0.5          # LOCUSFUSION_DEFAULTS["uncovered_frac"]
OUT = Path(__file__).resolve().parents[1] / "results" / "analysis" / "token_census_v1"
OUT.mkdir(parents=True, exist_ok=True)

loci = loci_on_disk(REAL_GENOTYPE_DIR)
regs = sorted(p.stem for p in Path(REAL_REGULATORY_DIR).glob("*.fasta") if p.stem in set(loci))

counts, frac, ids = {}, {}, None
for modality, stream in (("dna", "nt"), ("protein", "aa"), ("regulatory", "reg")):
    data = load_multidrug_dataset(ALL_DRUGS, [modality], REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV,
                                  regulatory_dir=REAL_REGULATORY_DIR, loci=loci,
                                  regulatory_loci=regs, per_modality_branch=False,
                                  delta=True, verbose=True)
    if ids is None:
        ids = list(data.isolate_ids)
    assert list(data.isolate_ids) == ids, "isolate axis differs between modalities"
    for b in data.blocks:
        locus = b.name.split(":", 1)[1]
        occ = np.abs(b.array).sum(1) > 0                      # (N, L)
        counts[(locus, stream)] = occ.sum(1).astype(np.int32)
        frac[(locus, stream)] = occ.mean(1)
    del data
    print(f"{modality}: done", flush=True)

N = len(ids)
uncovered = {l: np.maximum.reduce([frac[k] for k in frac if k[0] == l]) > UNCOVERED_FRAC
             for l in loci}

rows = []
for (l, s), n in sorted(counts.items()):
    over = n > CAP
    rows.append({"locus": l, "stream": s, "p99": np.percentile(n, 99), "max": int(n.max()),
                 "pairs > 16": int(over.sum()),
                 "pairs > 16, covered": int((over & ~uncovered[l]).sum())})
per_stream = pd.DataFrame(rows).set_index(["locus", "stream"])
per_stream.to_csv(OUT / "per_stream.csv")

def over_mask(ls, covered_only):
    m = np.stack([np.stack([counts[k] > CAP for k in counts if k[0] == l]).any(0) for l in ls])
    if covered_only:
        m &= ~np.stack([uncovered[l] for l in ls])
    return m                                                   # (loci, N)

summ = []
for name, ls in [("all 19 loci", loci)] + [(f"{d} per-drug", [l for l in DRUG_TO_LOCI[d] if l in loci])
                                           for d in ALL_DRUGS]:
    for cov in (False, True):
        m = over_mask(ls, cov)
        summ.append({"loci": name, "excl. uncovered": cov,
                     "(isolate, locus) pairs > 16": int(m.sum()), "pairs": m.size,
                     "pair share": m.mean(),
                     "isolates with any locus > 16": int(m.any(0).sum()),
                     "isolate share": m.any(0).mean()})
summary = pd.DataFrame(summ)
summary.to_csv(OUT / "summary.csv", index=False)

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print(f"\n{N} isolates, {len(loci)} loci\n")
print(per_stream[per_stream["pairs > 16"] > 0].to_string())
print()
print(summary.to_string(index=False))
