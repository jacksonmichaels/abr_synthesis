"""How many variant tokens does locusfusion_v2 see per (isolate, locus), before
its 16-token cap? All 17,942 isolates, all 19 loci, each coding record merged
with its promoter window (shared bases counted once), exactly as the tokenizer
builds its column axis.

    python scripts/token_census.py      # real data; run via sbatch on the cluster

Writes results/analysis/token_census/{per_locus.csv, per_drug.csv, counts.npz}.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR  # noqa: E402
from datasets import ALL_DRUGS, load_multidrug_dataset, loci_on_disk  # noqa: E402
from datasets.loader import DRUG_TO_LOCI  # noqa: E402
from models import LocusFusionV2Net  # noqa: E402

CAP = 16
OUT = Path(__file__).resolve().parents[1] / "results" / "analysis" / "token_census"
OUT.mkdir(parents=True, exist_ok=True)

loci = loci_on_disk(REAL_GENOTYPE_DIR)
regs = sorted(p.stem for p in Path(REAL_REGULATORY_DIR).glob("*.fasta") if p.stem in set(loci))
data = load_multidrug_dataset(ALL_DRUGS, ["dna", "regulatory"], REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV,
                              regulatory_dir=REAL_REGULATORY_DIR, loci=loci, regulatory_loci=regs,
                              per_modality_branch=False, delta=True, verbose=True)
net = LocusFusionV2Net.from_blocks(data.blocks, n_drugs=1)
N = len(data.isolate_ids)
print(f"{N} isolates, loci {net.loci}, promoter windows {regs}", flush=True)

rec = {l: np.zeros(N, np.int32) for l in net.loci}
prom = {l: np.zeros(N, np.int32) for l in net.loci}
for a in range(0, N, 2000):
    for li, p in enumerate(net._plan):
        occ = torch.cat([torch.from_numpy(data.blocks[b["index"]].array[a:a + 2000]).float()
                         .abs().sum(1) > 0 for b in p["nt"]], dim=1)
        occ &= getattr(net, f"keep_map_{li}")
        rl = p["reg_len"]
        prom[p["locus"]][a:a + 2000] = occ[:, :rl].sum(1).numpy()
        rec[p["locus"]][a:a + 2000] = occ[:, rl:].sum(1).numpy()

rows = []
for l in net.loci:
    n = rec[l] + prom[l]
    over = n > CAP
    rows.append({"locus": l, "promoter window": l in regs,
                 "isolates with ≥1 token": int((n > 0).sum()),
                 "mean": n.mean(), "median": np.median(n), "p99": np.percentile(n, 99),
                 "p99.9": np.percentile(n, 99.9), "max": int(n.max()),
                 "isolates > 16": int(over.sum()), "share > 16": over.mean(),
                 "tokens dropped (total)": int((n[over] - CAP).sum()),
                 "max from promoter": int(prom[l].max())})
per_locus = pd.DataFrame(rows).set_index("locus")
per_locus.to_csv(OUT / "per_locus.csv")

hist = pd.DataFrame({l: np.bincount(np.minimum(rec[l] + prom[l], 60), minlength=61) for l in net.loci},
                    index=[str(i) if i < 60 else "60+" for i in range(61)])
hist.to_csv(OUT / "histogram.csv")

drows = []
for drug in ALL_DRUGS:
    for arm, dl in (("per-drug loci", DRUG_TO_LOCI[drug]), ("all 19 loci", net.loci)):
        n = np.stack([rec[l] + prom[l] for l in dl])            # (loci, N)
        drows.append({"drug": drug, "loci": arm, "n loci": len(dl),
                      "isolates with any locus > 16": int((n > CAP).any(0).sum()),
                      "share": float((n > CAP).any(0).mean()),
                      "loci that overflow": ", ".join(l for l, row in zip(dl, n) if (row > CAP).any())})
per_drug = pd.DataFrame(drows)
per_drug.to_csv(OUT / "per_drug.csv", index=False)
np.savez_compressed(OUT / "counts.npz", isolates=np.array(data.isolate_ids),
                    **{f"record_{l}": rec[l] for l in net.loci}, **{f"promoter_{l}": prom[l] for l in net.loci})

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print("\nPER LOCUS (tokens = changed record + promoter columns per isolate, before the cap)\n")
print(per_locus.round(3).to_string())
print("\nPER DRUG\n")
print(per_drug.to_string(index=False))
allN = np.stack([rec[l] + prom[l] for l in net.loci])
print(f"\nany locus > 16, any isolate: {int((allN > CAP).any(0).sum())} of {N} isolates "
      f"({(allN > CAP).any(0).mean():.2%}); (isolate, locus) pairs over: {int((allN > CAP).sum())} "
      f"of {allN.size}")
