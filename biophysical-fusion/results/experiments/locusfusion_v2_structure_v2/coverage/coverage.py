"""Located-token coverage on the sweep's inputs (dna tokens, per-drug loci).
Per (isolate, coding locus with >=1 token): how many tokens are located, and
why the others are not. Writes coverage.csv + reasons.csv here."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV
from datasets import load_dataset
from datasets.structure import load_structure
from models.locusfusion_v2 import LocusFusionV2Net

DRUGS = ["AMIKACIN", "CAPREOMYCIN", "ETHAMBUTOL", "ETHIONAMIDE", "ISONIAZID",
         "LEVOFLOXACIN", "MOXIFLOXACIN", "PYRAZINAMIDE", "RIFAMPICIN", "STREPTOMYCIN"]
DRUGS = sys.argv[1:] or DRUGS
cov, why = [], []
for drug in DRUGS:
    d = load_dataset(drug, ["dna", "structure"], REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV,
                     per_modality_branch=False, delta=True, verbose=False)
    m = LocusFusionV2Net.from_blocks(d.blocks, n_drugs=1).eval()
    xs = [torch.from_numpy(b.array).float() for b in d.blocks]
    for r in m.variant_report(xs):
        s = load_structure(r["locus"])
        v, c = r["valid"], r["codon"]
        if s is None:                                   # rRNA
            why.append({"drug": drug, "locus": r["locus"], "reason": "rRNA locus (no protein)",
                        "tokens": int(v.sum())})
            continue
        n = s["ca"].shape[0]
        loc = r["struct_mask"].diagonal(dim1=1, dim2=2)
        has = r["valid"].any(1)
        nloc = loc.sum(1)[has]
        over = (r["n_variants"] > m.max_variants)[has]
        cov.append({"drug": drug, "locus": r["locus"], "isolates_with_token": int(has.sum()),
                    "loc0": int((nloc == 0).sum()), "loc1": int((nloc == 1).sum()),
                    "loc2plus": int((nloc >= 2).sum()), "over_cap": int(over.sum())})
        un = v & ~loc
        reason = np.where((c < 0) & (r["region"] == 1), "non-codon column in CDS record (flank/insertion)",
                 np.where(c < 0, "promoter", np.where(c == n - 1, "stop codon", "residue without coordinate")))
        for k in np.unique(reason[un.numpy()]):
            why.append({"drug": drug, "locus": r["locus"], "reason": k,
                        "tokens": int((un.numpy() & (reason == k)).sum())})
        why.append({"drug": drug, "locus": r["locus"], "reason": "located", "tokens": int(loc.sum())})
    print(drug, "done", flush=True)
    del d, xs, m
out = Path(__file__).parent
tag = "_".join(DRUGS) if len(DRUGS) == 1 else "all"
pd.DataFrame(cov).to_csv(out / f"coverage_{tag}.csv", index=False)
pd.DataFrame(why).to_csv(out / f"reasons_{tag}.csv", index=False)
print("written")
