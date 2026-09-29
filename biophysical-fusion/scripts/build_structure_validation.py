"""Builds notebooks/structure_validation.ipynb -- checks the AlphaFold C-alpha
coordinates written by scripts/fetch_af_structures.py, and the 16x16 token
distance matrices datasets/structure.py builds from them for locusfusion_v2.

    python scripts/fetch_af_structures.py                # the data, once
    python scripts/build_structure_validation.py         # write the notebook
    jupyter nbconvert --to notebook --execute --inplace notebooks/structure_validation.ipynb

Needs the real data (bigtb_ref.REAL_*) and no trained model."""
import json
from pathlib import Path

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})

md(r"""
# Structure validation: AlphaFold C-alpha distances on the `locusfusion_v2` token grid

Every coding token in `locusfusion_v2` carries an H37Rv codon index k. The
fetch script gives each codon a C-alpha coordinate from the AlphaFold DB model of
that protein. A locus's 16 variant tokens then have a 16x16 distance matrix. Any
token with no coordinate is masked: promoter, flank, insertion, rRNA, the stop
codon, or padding.

§1–4 check the coordinates at the residue level for all 17 coding loci.
§5–6 check the token matrices on real isolates.
""")

code(r"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from IPython.display import display

PROJECT = next(p for p in [Path.cwd().parent, Path.cwd(),
                           Path("/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion")]
               if (p / "training" / "checkpoint.py").exists())
sys.path.insert(0, str(PROJECT))

from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR, REAL_STRUCTURE_DIR
from datasets import biochem, cds, load_dataset
from datasets import structure as st
from models import LocusFusionV2Net
from models.locusfusion_v2 import _RES_ORDER

manifest = json.loads((Path(REAL_STRUCTURE_DIR) / "manifest.json").read_text())
CODING = [l for l, r in manifest.items() if r["status"] == "ok"]
print(f"{len(CODING)} coding loci with a structure; masked: "
      f"{[l for l, r in manifest.items() if r['status'] != 'ok']}")
""")

md(r"""
## 1. Where each structure came from

One AlphaFold DB model per locus. `identical` is whether the model's sequence is
exactly the H37Rv protein translated from the GenBank record the alignments were
cut from. `mapped` counts the codons that got a coordinate.
""")

code(r"""
man = pd.DataFrame([manifest[l] for l in CODING]).set_index("locus")
display(man[["locus_tag", "uniprot", "reviewed", "model_id", "afdb_version", "ref_length",
             "model_length", "sequence_identical", "aligned_identity", "mapped_codons",
             "mapped_frac", "mean_plddt"]])
""")

md(r"""
rpoB is the only non-identical sequence. UniProt uses the newer 1,178-aa start
annotation, 6 residues upstream of this GenBank record's 1,172-aa start. The
alignment maps every codon at 100% identity. This grid is the numbering the
WHO catalogue uses (S450 is residue 450 here). UniProt/AlphaFold residue
numbers for rpoB run 6 higher.
""")

md(r"""
## 2. Residue identity on the tokenizer's own grid

This is the check that matters most. The tokenizer gets its reference residue
per codon from the **alignment FASTA's H37Rv row**
(`biochem.translate_reference_frame` over `cds.cds_columns`). The structure gets
its residue from **GenBank → UniProt → AlphaFold**. The two paths are
independent. If they agree at every codon, a token's codon index points at the
right atom.
""")

code(r"""
rows = []
for locus in CODING:
    fasta = sorted(Path(REAL_GENOTYPE_DIR).glob(f"{locus}*.fasta"))[0]
    ref_row = cds._reference_row(fasta.as_posix())
    grid = biochem.translate_reference_frame(ref_row, ref_row, cds.cds_columns(REAL_GENOTYPE_DIR, locus))
    s = st.load_structure(locus)
    model = s["model_residue"]
    n = min(len(grid), len(model))
    have = np.array([m != "" for m in model[:n]])
    same = np.array([g == m for g, m in zip(grid[:n], model[:n])])
    # a GTG/TTG start reads V/L on the grid and is Met in the protein
    alt_start = not same[0] and grid[0] in "VL" and model[0] == "M"
    same[0] |= alt_start
    rows.append({"locus": locus, "grid codons": len(grid), "structure codons": len(model),
                 "start (grid→protein)": f"{grid[0]}→{model[0]}",
                 "with coordinate": int(have.sum()), "residue match": int((same & have).sum()),
                 "mismatch": int((~same & have).sum()),
                 "no coordinate": ", ".join(f"{k}:{grid[k]}" for k in np.where(~have)[0])})
ident = pd.DataFrame(rows).set_index("locus")
display(ident)
assert (ident["mismatch"] == 0).all(), "a codon points at the wrong residue"
assert (ident["grid codons"] == ident["structure codons"]).all()
""")

md(r"""
Every locus should show zero mismatches, and exactly one codon with no
coordinate: the terminal stop (`*`). Loci whose start reads `V→M` or `L→M` have
a GTG or TTG start codon. The tokenizer translates codon 0 literally, while the
protein and the model begin with Met. The coordinate is the same residue, so
those loci count as matches.
""")

md(r"""
## 3. Geometry sanity

Consecutive C-alpha atoms in a real chain sit 3.8 Å apart (about 2.9 Å for a
*cis*-proline). A large deviation would mean the per-codon coordinates are out
of order or are coming from the wrong atoms. This check caught the one mapping
bug found so far: rpoB's TTG start codon had been paired with UniProt's
upstream Met, 22 Å from residue 2.
""")

code(r"""
rows, steps = [], []
for locus in CODING:
    ca = st.load_structure(locus)["ca"]
    d = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    d = d[~np.isnan(d)]
    steps.append(d)
    rows.append({"locus": locus, "bonds": len(d), "median Å": np.median(d), "min Å": d.min(),
                 "max Å": d.max(), "outside 3.6–4.0 Å": int(((d < 3.6) | (d > 4.0)).sum())})
geo = pd.DataFrame(rows).set_index("locus")
display(geo.round(3))
assert (geo["max Å"] < 4.2).all(), "a consecutive-codon step no chain can make: mapping error"

fig, ax = plt.subplots(figsize=(6.4, 2.8), constrained_layout=True)
ax.hist(np.concatenate(steps), bins=np.arange(2.6, 4.2, 0.02), color="#4c72b0")
ax.set_yscale("log")
ax.set_xlabel("C-alpha(k) to C-alpha(k+1) distance (Å)")
ax.set_ylabel("residue pairs")
ax.set_title("Consecutive C-alpha distance, all 17 coding loci")
plt.show()
""")

md(r"""
## 4. Confidence, and known sites

pLDDT is AlphaFold's per-residue confidence (0–100). Below about 50 the chain is
usually disordered, and distances there don't mean much.
""")

code(r"""
n = len(CODING)
fig, axes = plt.subplots((n + 3) // 4, 4, figsize=(11, 1.6 * ((n + 3) // 4)),
                         constrained_layout=True, sharey=True)
for ax, locus in zip(axes.flat, CODING):
    p = st.load_structure(locus)["plddt"]
    ax.plot(np.arange(len(p)), p, lw=0.7, color="#4c72b0")
    ax.axhline(70, color="0.6", lw=0.6, ls="--")
    ax.set_title(locus, fontsize=9)
    ax.set_ylim(0, 100)
for ax in list(axes.flat)[n:]:
    ax.axis("off")
fig.supxlabel("codon (H37Rv grid)")
fig.supylabel("pLDDT")
fig.suptitle("AlphaFold pLDDT by codon, per locus (dashed: pLDDT 70)")
plt.show()

low = pd.DataFrame({l: {"pLDDT < 50": float(np.nanmean(st.load_structure(l)["plddt"] < 50)),
                        "pLDDT < 70": float(np.nanmean(st.load_structure(l)["plddt"] < 70))}
                    for l in CODING}).T
display(low.style.format("{:.1%}"))
""")

md(r"""
Next are residue pairs with known function that sit far apart in sequence. The
residue letters check the numbering; the distances check the fold. Numbers are
**residue numbers on the H37Rv GenBank grid** (codon = residue − 1). For rpoB
this is also WHO numbering.
""")

code(r"""
SITES = [   # (locus, [(label, residue number on this grid, expected residue)], what)
    ("pncA", [("D8", 8, "D"), ("K96", 96, "K"), ("C138", 138, "C")], "catalytic triad"),
    ("pncA", [("D49", 49, "D"), ("H51", 51, "H"), ("H57", 57, "H"), ("H71", 71, "H")], "metal site"),
    ("rpoB", [("D435", 435, "D"), ("H445", 445, "H"), ("S450", 450, "S")], "RRDR"),
    ("katG", [("H108", 108, "H"), ("H270", 270, "H"), ("S315", 315, "S")], "distal His / proximal His / S315"),
    ("inhA", [("F149", 149, "F"), ("Y158", 158, "Y"), ("K165", 165, "K")], "active site"),
]
rows = []
for locus, res, what in SITES:
    s = st.load_structure(locus)
    D = st.residue_distances(locus)
    for i, (la, ra, ea) in enumerate(res):
        for lb, rb, eb in res[i + 1:]:
            rows.append({"locus": locus, "site": what, "pair": f"{la}–{lb}",
                         "residues on grid": s["model_residue"][ra - 1] + s["model_residue"][rb - 1],
                         "expected": ea + eb, "sequence separation": rb - ra,
                         "C-alpha distance Å": round(float(D[ra - 1, rb - 1]), 1)})
sites = pd.DataFrame(rows)
display(sites)
assert (sites["residues on grid"] == sites["expected"]).all(), "site numbering is off"
""")

md(r"""
If the structures are right, every pair above should be much closer in space
than its sequence separation would suggest. Pairs 40+ residues apart should sit
within roughly 5–15 Å.
""")

code(r"""
D = st.residue_distances("pncA")
fig, ax = plt.subplots(figsize=(4.6, 4.0), constrained_layout=True)
im = ax.imshow(D, cmap="viridis_r", vmax=40)
fig.colorbar(im, ax=ax, label="C-alpha distance (Å)")
ax.set_xlabel("codon")
ax.set_ylabel("codon")
ax.set_title("pncA: residue-residue C-alpha distance")
plt.show()
""")

md(r"""
## 5. Token distances on real isolates

Next, the tokenizer runs on two drugs. ISONIAZID with the extra loci covers
katG, inhA and fabG1, with promoter windows. STREPTOMYCIN covers rpsL and gid,
plus rrs, which should be fully masked. For every isolate the code builds
`(B, n_loci, 16, 16)` with `structure.report_distances`.
""")

code(r"""
N = 6000     # first N isolates per drug

def tokens_for(drug, extra):
    data = load_dataset(drug, ["dna", "protein", "regulatory", "structure"], REAL_GENOTYPE_DIR,
                        REAL_PHENOTYPE_CSV, regulatory_dir=REAL_REGULATORY_DIR,
                        per_modality_branch=False, delta=True, protein_frame="reference",
                        extra_loci=extra, verbose=False)
    xs = [torch.from_numpy(b.array[:N]).float() for b in data.blocks]
    net = LocusFusionV2Net.from_blocks(data.blocks, n_drugs=1)
    rep = net.variant_report(xs)
    dist, mask = st.report_distances(rep, net.max_variants)
    return (net, rep, dist, mask), xs

loaded = {d: tokens_for(d, e) for d, e in [("ISONIAZID", True), ("STREPTOMYCIN", False)]}
runs = {d: v[0] for d, v in loaded.items()}
runs_xs = {d: v[1] for d, v in loaded.items()}      # kept for the §7 model check
for d, (net, rep, dist, mask) in runs.items():
    print(f"{d}: loci {net.loci}  dist {tuple(dist.shape)}")
""")

md(r"""
The first table below says why each token is masked or unmasked. The second
repeats the §2 check on the tokens themselves: at every unmasked token, the
structure's residue must be the residue the model reads as H37Rv's at that codon
(its `ref_res` buffer, the independent path).
""")

code(r"""
rows = []
for drug, (net, rep, dist, mask) in runs.items():
    for li, r in enumerate(rep):
        v, c = r["valid"], r["codon"]
        s = st.load_structure(r["locus"])
        has = mask[:, li].diagonal(dim1=1, dim2=2)[:, :v.shape[1]]
        n_cod = s["ca"].shape[0] if s is not None else 0
        rows.append({"drug": drug, "locus": r["locus"],
                     "tokens": int(v.sum()),
                     "promoter": int((v & (r["region"] == 0) & (c < 0)).sum()),
                     "record, no codon": int((v & (r["region"] == 1) & (c < 0)).sum()),
                     "stop codon": int((v & (c == n_cod - 1)).sum()) if s is not None else 0,
                     "with coordinate": int(has.sum()),
                     "isolates with ≥2 located tokens": int((has.sum(1) >= 2).sum())})
tok = pd.DataFrame(rows).set_index(["drug", "locus"])
display(tok)
""")

code(r"""
rows = []
for drug, (net, rep, dist, mask) in runs.items():
    for li, r in enumerate(rep):
        s = st.load_structure(r["locus"])
        if s is None:
            continue
        has = mask[:, li].diagonal(dim1=1, dim2=2)[:, :r["codon"].shape[1]]
        codons = r["codon"][has].numpy()
        ref_res = getattr(net, f"ref_res_{net.loci.index(r['locus'])}").numpy()
        model_says = np.array([_RES_ORDER[j] if j >= 0 else "?" for j in ref_res[codons]])
        struct_says = s["model_residue"][codons].copy()
        alt = (codons == 0) & np.isin(model_says, ["V", "L"]) & (struct_says == "M")
        struct_says[alt] = model_says[alt]          # GTG/TTG start, see §2
        rows.append({"drug": drug, "locus": r["locus"], "located tokens": len(codons),
                     "residue match": int((model_says == struct_says).sum()),
                     "mismatch": int((model_says != struct_says).sum())})
chk = pd.DataFrame(rows).set_index(["drug", "locus"])
display(chk)
assert (chk["mismatch"] == 0).all()

for drug, (net, rep, dist, mask) in runs.items():      # structural invariants
    assert torch.equal(dist, dist.transpose(-1, -2)) and torch.equal(mask, mask.transpose(-1, -2))
    assert (dist[~mask] == 0).all() and (dist[mask] >= 0).all()
    rrs = [i for i, r in enumerate(rep) if r["locus"] in ("rrs", "rrl")]
    assert not mask[:, rrs].any(), "rRNA tokens must be fully masked"
print("symmetric, zero where masked, rRNA fully masked: ok")
""")

md(r"""
## 6. What the matrices look like

Below is one isolate per locus: the one with the most located tokens. Masked
cells are blank. Each axis label is the token's codon and its H37Rv residue, or
the token's region when it has no codon.
""")

code(r"""
net, rep, dist, mask = runs["ISONIAZID"]
fig, axes = plt.subplots(1, len(rep), figsize=(4.4 * len(rep), 4.0), constrained_layout=True)
for ax, (li, r) in zip(np.atleast_1d(axes), enumerate(rep)):
    has = mask[:, li].diagonal(dim1=1, dim2=2)
    b = int(has.sum(1).argmax())
    k = int(r["valid"][b].sum())
    s = st.load_structure(r["locus"])
    lab = [f"{int(c) + 1}{s['model_residue'][int(c)]}" if has[b, j] else
           ("prom" if int(r["region"][b, j]) == 0 else "rec")
           for j, c in enumerate(r["codon"][b, :k])]
    M = np.where(mask[b, li, :k, :k].numpy(), dist[b, li, :k, :k].numpy(), np.nan)
    im = ax.imshow(M, cmap="viridis_r", vmin=0, vmax=40)
    ax.set_xticks(range(k), lab, rotation=90, fontsize=7)
    ax.set_yticks(range(k), lab, fontsize=7)
    ax.set_title(f"{r['locus']}: token C-alpha distance, isolate {b}", fontsize=9)
fig.colorbar(im, ax=axes, label="C-alpha distance (Å)", shrink=0.8)
plt.show()
""")

md(r"""
The next plot compares 3D distance with codon separation for every located
token pair across all isolates. Pairs that are close in space but far apart in
sequence are what 3D distance adds over the existing 1D positional encoding.
""")

code(r"""
xs_, ys_ = [], []
for drug, (net, rep, dist, mask) in runs.items():
    for li, r in enumerate(rep):
        k = r["codon"].shape[1]
        m = mask[:, li, :k, :k] & torch.triu(torch.ones(k, k, dtype=torch.bool), 1)
        c = r["codon"].float()
        sep = (c.unsqueeze(2) - c.unsqueeze(1)).abs()
        xs_.append(sep[m].numpy()); ys_.append(dist[:, li, :k, :k][m].numpy())
sep, d3 = np.concatenate(xs_), np.concatenate(ys_)
fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
hb = ax.hexbin(np.log10(sep + 1), d3, gridsize=50, bins="log", cmap="Blues", mincnt=1)
fig.colorbar(hb, ax=ax, label="token pairs")
ax.set_xlabel("log10(codon separation + 1)")
ax.set_ylabel("C-alpha distance (Å)")
ax.set_title("Token pairs: 3D distance vs codon separation, ISONIAZID + STREPTOMYCIN")
plt.show()
r = np.corrcoef(np.log10(sep + 1), d3)[0, 1]
print(f"{len(d3):,} token pairs; Pearson r(log sep, distance) = {r:.2f}; "
      f"pairs >50 codons apart but <12 Å: {int(((sep > 50) & (d3 < 12)).sum()):,}")
""")


md(r"""
## 7. Encoding the matrix as token features (tentative)

> **Under discussion.** The five encodings below are candidates, not a settled
> choice. The model currently uses **closeness, exp(−d / 10 Å)**, 16 channels.
> The full comparison, with three more encodings (exp(−d/30), linear, percentile),
> coverage numbers and a 6-token example, is `notebooks/tokenization_v2.ipynb` §6.

Every option is slot-aligned: variant token *i* carries row *i*, so channel *j*
describes token *i* → token *j*. They differ in three things:

- how many channels they add;
- what a missing distance (promoter, rRNA, stop codon, padding) looks like;
- how much resolution they keep at long range.
""")

code(r"""
from models.locusfusion_v2 import STRUCT_LENGTH
CONTACT = 8.0                       # Å, a common C-alpha contact threshold

# name -> (channels, encode(dist, mask) -> per-channel-block list of (B, k, k) tensors)
ENCODINGS = {
    "distance + mask":           (32, lambda d, m: [d / 10 * m, m.float()]),
    "distance only":             (16, lambda d, m: [d / 10 * m]),
    f"closeness exp(-d/{STRUCT_LENGTH:g})": (16, lambda d, m: [torch.exp(-d / STRUCT_LENGTH) * m]),
    "closeness 1/(1+d/10)":      (16, lambda d, m: [1 / (1 + d / 10) * m]),
    f"contact d<{CONTACT:g} Å":  (16, lambda d, m: [((d < CONTACT) & m).float()]),
}

def value_at(name, d_A):
    return float(ENCODINGS[name][1](torch.tensor([[d_A]]), torch.tensor([[True]]))[0])

# own slot is identifiable when exactly one entry in the row reads "same residue"
def own_slot_unique(name, d, m):
    blocks = ENCODINGS[name][1](d, m)
    if name == "distance + mask":
        same = (blocks[0] == 0) & (blocks[1] == 1)
    else:
        same = blocks[0] == blocks[0].diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
    located = m.diagonal(dim1=-2, dim2=-1)
    return (same.sum(-1) == 1)[located].float().mean().item()

pooled_d, pooled_m = [], []
for drug, (net, rep, dist, mask) in runs.items():
    for li, r in enumerate(rep):
        if mask[:, li].any():
            pooled_d.append(dist[:, li]); pooled_m.append(mask[:, li])
D_all, M_all = torch.cat(pooled_d), torch.cat(pooled_m)

rows = []
for name, (ch, _) in ENCODINGS.items():
    rows.append({"encoding": name, "channels": ch,
                 "same residue (0 Å)": value_at(name, 0.0),
                 "adjacent (3.8 Å)": value_at(name, 3.8),
                 "20 Å": value_at(name, 20.0), "40 Å": value_at(name, 40.0),
                 "missing partner": 0.0,
                 "|Δ| 30→40 Å": abs(value_at(name, 30.0) - value_at(name, 40.0)),
                 "own slot unique": own_slot_unique(name, D_all, M_all)})
enc = pd.DataFrame(rows).set_index("encoding")
display(enc.style.format({c: "{:.3f}" for c in enc.columns if c not in ("channels", "own slot unique")}
                         | {"own slot unique": "{:.1%}"}))
""")

md(r"""
How to read the table:

- For distance + mask, the value columns show the distance channel (Å/10); the
  mask channel reads 1 for every real entry and 0 for a missing one.
- `own slot unique` is the share of located tokens whose row contains exactly
  one "same residue" entry, i.e. its own slot. It's measured on the real tokens
  from §5.
- **Distance only:** the token's own slot, a missing partner and every padding
  slot all read 0, so its own slot is never unique (0%). This is the case the
  mask channels exist for.
- **The other four** leave the own slot unique for about 98% of located tokens.
  The remaining ~2% share a codon with another token, and those tokens have
  identical rows, so the tie hides nothing.
- **Contact** stays at 97.7% only because most located tokens have no other
  located token within 8 Å. It carries no distance within or beyond 8 Å.
- **Closeness:** a missing partner reads the same as an infinitely distant one.
  `|Δ| 30→40 Å` shows how much long-range resolution each option keeps (the
  distance rows are in Å/10).
""")

code(r"""
fig, (ax, axh) = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True, constrained_layout=True,
                              gridspec_kw={"height_ratios": [2, 1]})
d = torch.linspace(0, 60, 601)
one = torch.ones_like(d, dtype=torch.bool)
curves = {f"exp(-d/{L:g})": torch.exp(-d / L) for L in (5.0, STRUCT_LENGTH, 20.0)}
curves["1/(1+d/10)"] = 1 / (1 + d / 10)
curves[f"contact d<{CONTACT:g} Å"] = (d < CONTACT).float()
for (label, y), ls in zip(curves.items(), ["--", "-", ":", "-.", "-"]):
    ax.plot(d, y, ls, lw=1.6 if label == f"exp(-d/{STRUCT_LENGTH:g})" else 1.1, label=label)
ax.set_ylabel("encoded value")
ax.set_title("Encoded value vs C-alpha distance")
ax.legend(frameon=False, fontsize=8)
upper = torch.triu(torch.ones(D_all.shape[-1], D_all.shape[-1], dtype=torch.bool), 1)
real = D_all[M_all & upper].numpy()
axh.hist(real, bins=np.arange(0, 61, 1), color="#4c72b0")
axh.set_xlabel("C-alpha distance (Å)")
axh.set_ylabel("token pairs")
axh.set_title(f"Located token-pair distances, ISONIAZID + STREPTOMYCIN (n={len(real):,})")
plt.show()
print(f"token pairs: median {np.median(real):.1f} Å; ≤{CONTACT:g} Å {np.mean(real <= CONTACT):.1%}; "
      f"> 30 Å {np.mean(real > 30):.1%}")
""")

md(r"""
Below is one katG isolate under each encoding. It's the isolate with the most
distinct located codons that also has an unlocated token, so every case shows
up. Row *i* is what token *i* carries.
""")

code(r"""
net, rep, dist, mask = runs["ISONIAZID"]
li = net.loci.index("katG")
r = rep[li]
located = mask[:, li].diagonal(dim1=1, dim2=2)
score = [len(set(r["codon"][b][located[b]].tolist())) if (r["valid"][b] & ~located[b]).any() else -1
         for b in range(located.shape[0])]
b = int(np.argmax(score)); k = int(r["valid"][b].sum())
s = st.load_structure("katG")
lab = [f"{int(c) + 1}{s['model_residue'][int(c)]}" if located[b, j] else
       ("prom" if int(r["region"][b, j]) == 0 else "rec") for j, c in enumerate(r["codon"][b, :k])]
d_b, m_b = dist[b:b + 1, li, :k, :k], mask[b:b + 1, li, :k, :k]

panels = [("distance + mask: distance (Å/10)", ENCODINGS["distance + mask"][1](d_b, m_b)[0], m_b),
          ("distance + mask: mask", ENCODINGS["distance + mask"][1](d_b, m_b)[1], None),
          ("distance only (Å/10)", ENCODINGS["distance only"][1](d_b, m_b)[0], None)]
panels += [(n, ENCODINGS[n][1](d_b, m_b)[0], None) for n in list(ENCODINGS)[2:]]
fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), constrained_layout=True)
for ax, (title, M, blank) in zip(axes.flat, panels):
    A = M[0].numpy().astype(float)
    if blank is not None:
        A = np.where(blank[0].numpy(), A, np.nan)
    im = ax.imshow(A, cmap="viridis", vmin=0, vmax=max(1.0, np.nanmax(A)))
    ax.set_xticks(range(k), lab, rotation=90, fontsize=6)
    ax.set_yticks(range(k), lab, fontsize=6)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.7)
fig.suptitle(f"katG isolate {b}: token feature rows by encoding ({k} tokens)")
plt.show()
""")

md(r"""
Last, a check that the model's token channels are exactly the current choice,
exp(−d / 10 Å), slot-aligned, and zero for the `[NULL]` slot.
""")

code(r"""
for drug, (net, rep, dist, mask) in runs.items():
    xs = runs_xs[drug]
    feat = net._locus_tokens(xs)[0]                  # (B, n_loci, 17, c_tok)
    V = net.max_variants
    for li, r in enumerate(rep):
        got = feat[:, li, 1:, net.f_struct:net.f_struct + V]
        want = torch.exp(-dist[:, li] / STRUCT_LENGTH) * mask[:, li]
        assert torch.allclose(got, want, atol=1e-5), (drug, r["locus"])
        assert not feat[:, li, 0, net.f_struct:].any(), "[NULL] carries structure"
    print(f"{drug}: c_tok {net.c_tok} = {net.c_tok - V} base + {V} closeness; "
          "token channels == exp(-d/10) · mask, every locus")
""")

md(r"""
## Caveats

- **Monomer models.** Each distance comes from a single-chain AlphaFold model:
  no ligands (heme, NAD, DNA, drug), no partner subunits, no cross-gene
  geometry. Residues at an oligomer or complex interface are placed as if the
  partner were absent.
- **Low-pLDDT regions** (§4) are kept, not masked, and pLDDT is not a model
  input. AlphaFold is a stand-in here until more reliable structures replace it.
- **§7 is tentative.** The token encoding (closeness, exp(−d / 10 Å)) is the
  current choice under discussion, not a result. The 10 Å length scale is also
  a choice, not a fit.
- **One coordinate per codon.** Two tokens in the same codon are 0 Å apart. A
  change in codon k and a change in codon k+1 are always about 3.8 Å apart,
  whatever the residues are.
- **rpoB numbering.** The grid (and WHO) use GenBank's 1,172-aa start, while
  UniProt/AlphaFold use a 1,178-aa one. Only raw model residue numbers are
  affected; the mapping is by alignment.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "structure_validation.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}")
