#!/usr/bin/env python
"""
The two non-AUC checks locusfusion's design commits itself to, run against a
trained locusfusion_v2 checkpoint.

    python results/experiments/locusfusion_v2/attribution_check.py \\
        --drug ISONIAZID --cell all_modalities

**1. Read-out attention must stop being uniform.** Flat 1/n_loci means the locus
summaries are still collinear and the two-stage structure did nothing — the
failure `results/experiments/token_signal` measured in setfusion (attention
pinned at exactly 1/8). Reported as the mean KL from uniform, per drug.

**2. The tokens the model attends to must be real variants at real
coordinates.** This is what the coordinate fix bought: a token's coordinate is
the H37Rv codon index, so the most-attended tokens can be named — "katG residue
315, Ser->Thr" — and checked against what is known to cause resistance, without
SHAP. Before the fix the same read-out would have pointed at residue 358.

NOTE the off-by-one, because it is the one way to misread this output: the
model's `coord` is **0-based** (residue k of the protein block), while the WHO
catalogue and every paper number residues from 1. This script prints the
**1-based** residue, so `katG S315T` appears as 315. The fractional part is the
codon phase and is left on the 0-based coordinate: `.33` is the second base of
the codon, `.67` the third.

Neither is a pass/fail gate. They are the discipline `token_signal` imposed on
itself: a mechanistic claim gets a mechanistic measurement, separate from AUC.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from datasets import load_dataset  # noqa: E402
from datasets import tokens as tok  # noqa: E402
from training.checkpoint import build_model_from_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug", default="ISONIAZID")
    ap.add_argument("--cell", default="all_modalities")
    ap.add_argument("--run", default="locusfusion_v2")
    ap.add_argument("--n", type=int, default=512, help="isolates to read")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    folder = here.parent / args.run / f"sd_{args.cell}__locusfusion"
    pointer = json.load(open(folder / "weights_location.json"))
    key = next(k for k in pointer if k.startswith(args.drug + "__"))
    wdir = Path(pointer[key])
    cfg = json.load(open(wdir / "config.json"))
    fold = sorted(wdir.glob("fold*.pt"))[0]

    model = build_model_from_config(cfg)
    model.load_state_dict(torch.load(fold, map_location="cpu"))
    model.eval()
    print(f"{args.drug} / {args.cell}: {fold.name}, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")

    dcfg = cfg["data"]
    # `delta` in the config records the FLAG, not the resolved value — locusfusion
    # implies it (models.DELTA_ARCHS), which is the same quirk
    # scripts/shap_attribution.py documents. Use the resolved value.
    data = load_dataset(args.drug, dcfg["modalities_requested"],
                        dcfg["genotype_dir"], dcfg["phenotype_csv"],
                        regulatory_dir=dcfg["regulatory_dir"],
                        loci=dcfg.get("loci_override"),
                        regulatory_loci=dcfg.get("regulatory_loci_override"),
                        per_modality_branch=False, delta=True,
                        variant_tokens=dcfg.get("variant_tokens", True),
                        verbose=False)
    idx = np.arange(min(args.n, data.blocks[0].array.shape[0]))
    xs = [torch.from_numpy(b.array[idx]).float() for b in data.blocks]
    with torch.no_grad():
        _logits, attn = model(xs, return_attn=True)      # (B, n_drugs, n_keys)
        report = model.variant_report(xs)

    # --- 1. is the read-out attention uniform? ------------------------------
    n_keys = attn.shape[-1]
    uniform = 1.0 / n_keys
    kl = (attn * (attn.clamp_min(1e-12) / uniform).log()).sum(-1)   # (B, n_drugs)
    names = model.drug_names or [args.drug]
    print(f"\nread-out attention over {n_keys} keys "
          f"(uniform would be {uniform:.4f}, KL 0)")
    for j, nm in enumerate(names):
        a = attn[:, j]
        print(f"  {nm:14} KL from uniform {float(kl[:, j].mean()):.4f}   "
              f"max weight {float(a.max()):.4f}   "
              f"mean top-1 {float(a.max(-1).values.mean()):.4f}")
        top = a.mean(0).argsort(descending=True)[:5]
        print("                 loci by mean weight: "
              + ", ".join(f"{model.loci[int(t)] if int(t) < len(model.loci) else int(t)}"
                          f" {float(a.mean(0)[t]):.3f}" for t in top))

    # --- 2. what are the tokens, by coordinate? -----------------------------
    sym = tok.symbol_names()
    counts = collections.Counter()
    for r in report:
        valid = r["valid"].numpy()
        coord = r["coord"].numpy()
        ref, alt = r["ref"].numpy(), r["alt"].numpy()
        for b in range(valid.shape[0]):
            for k in range(valid.shape[1]):
                if valid[b, k]:
                    counts[(r["locus"], r["stream"], round(float(coord[b, k]), 2),
                            sym[int(ref[b, k])], sym[int(alt[b, k])])] += 1
    print(f"\nmost common variant tokens over {len(idx)} isolates")
    print("  nt/aa: residue is 1-BASED (WHO / catalogue numbering); the model's\n"
          "  own coord is 0-based. The fraction is the codon phase: .33 = 2nd\n"
          "  base of the codon, .67 = 3rd.\n"
          "  reg: printed RAW. A promoter coordinate is codons upstream of the\n"
          "  extracted window's 3' end, not of the transcription start, because\n"
          "  the window carries a 30 bp flank whose orientation is not recorded\n"
          "  (datasets/tokens.py). So it is NOT the catalogue's c-15 numbering\n"
          "  and must not be read as if it were.")
    print(f"  {'locus':8} {'stream':6} {'position':>10}  {'ref':>7} -> {'alt':<7} count")
    for (locus, stream, coord, r, a), n in counts.most_common(args.top):
        shown = coord + 1 if stream in ("nt", "aa") else coord
        print(f"  {locus:8} {stream:6} {shown:10.2f}  {r:>7} -> {a:<7} {n}")


if __name__ == "__main__":
    main()
