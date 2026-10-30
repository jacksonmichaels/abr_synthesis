"""How many variant tokens does locusfusion_v3 (locusfusion_v2 with
--lf-unknown-symbols) see per (isolate, locus), before its 16-token cap?

Same counting as scripts/token_census.py (v2: one merged promoter+coding axis
per locus, one base one token), with the uncalled-base channel on, so an `N`
now makes a token. Counts are split into substitutions/gaps and uncalled bases.
Single-drug loader (the multi-drug one has no unknown_symbols); the drug only
picks the loader call, every isolate with sequence data is counted.

    sbatch scripts/sbatch/token_census_v3.sh

Writes results/analysis/token_census_v3/{per_locus.csv, summary.txt}.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR  # noqa: E402
from datasets import load_dataset, loci_on_disk  # noqa: E402
from datasets.loader import DRUG_TO_LOCI  # noqa: E402
from datasets.multidrug import ALL_DRUGS  # noqa: E402
from datasets.sequences import UNKNOWN_CHANNEL  # noqa: E402
from models import LocusFusionV2Net  # noqa: E402

CAP = 16
OUT = Path(__file__).resolve().parents[1] / "results" / "analysis" / "token_census_v3"
OUT.mkdir(parents=True, exist_ok=True)

loci = loci_on_disk(REAL_GENOTYPE_DIR)
regs = sorted(p.stem for p in Path(REAL_REGULATORY_DIR).glob("*.fasta") if p.stem in set(loci))
data = load_dataset("RIFAMPICIN", ["dna", "regulatory"], REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV,
                    regulatory_dir=REAL_REGULATORY_DIR, loci=loci, regulatory_loci=regs,
                    per_modality_branch=False, delta=True, unknown_symbols=True, verbose=True)
net = LocusFusionV2Net.from_blocks(data.blocks, n_drugs=1, unknown_symbols=True)
N = data.blocks[0].array.shape[0]
print(f"{N} isolates, loci {net.loci}, promoter windows {regs}", flush=True)

tot = {l: np.zeros(N, np.int32) for l in net.loci}
unk = {l: np.zeros(N, np.int32) for l in net.loci}
for a in range(0, N, 2000):
    for li, p in enumerate(net._plan):
        occ, isn = [], []
        for b in p["nt"]:
            blk = data.blocks[b["index"]]
            x = torch.from_numpy(blk.array[a:a + 2000]).float()
            occ.append(x.abs().sum(1) > 0)
            n_ch = blk.channel_names.index(UNKNOWN_CHANNEL)
            isn.append(x[:, n_ch] > 0)
        occ = torch.cat(occ, 1) & getattr(net, f"keep_map_{li}")
        isn = torch.cat(isn, 1) & occ
        tot[p["locus"]][a:a + 2000] = occ.sum(1).numpy()
        unk[p["locus"]][a:a + 2000] = isn.sum(1).numpy()

rows = []
for l in net.loci:
    n, u = tot[l], unk[l]
    over, over_wo = n > CAP, (n - u) > CAP
    rows.append({"locus": l, "median": np.median(n), "p99": np.percentile(n, 99),
                 "max": int(n.max()), "isolates with any N token": int((u > 0).sum()),
                 "isolates > 16": int(over.sum()),
                 "isolates > 16 without N tokens": int(over_wo.sum())})
per_locus = pd.DataFrame(rows).set_index("locus")
per_locus.to_csv(OUT / "per_locus.csv")

T = np.stack([tot[l] for l in net.loci])
U = np.stack([unk[l] for l in net.loci])
lines = [f"{N} isolates, {len(net.loci)} loci",
         f"(isolate, locus) pairs > 16: {int((T > CAP).sum())} of {T.size} "
         f"({(T > CAP).mean():.2%}); without N tokens it would be "
         f"{int(((T - U) > CAP).sum())} ({((T - U) > CAP).mean():.2%})",
         f"isolates with any locus > 16, 19 loci: {int((T > CAP).any(0).sum())} "
         f"({(T > CAP).any(0).mean():.2%})",
         f"share of all tokens that are uncalled bases: {U.sum() / max(T.sum(), 1):.2%}",
         f"isolates with at least one N token: {int((U > 0).any(0).sum())} "
         f"({(U > 0).any(0).mean():.2%})"]
for drug in ALL_DRUGS:
    idx = [net.loci.index(l) for l in DRUG_TO_LOCI[drug] if l in net.loci]
    m = T[idx] > CAP
    lines.append(f"  {drug} per-drug loci {DRUG_TO_LOCI[drug]}: isolates > 16 "
                 f"{int(m.any(0).sum())} ({m.any(0).mean():.2%}); pairs {m.mean():.2%}")
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")
pd.set_option("display.width", 220)
print(per_locus.to_string())
print("\n".join(lines))
