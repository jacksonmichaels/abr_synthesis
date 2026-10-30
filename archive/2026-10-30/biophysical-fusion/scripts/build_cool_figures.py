"""Builds notebooks/cool_figures.ipynb -- figures only, added on request.

    python scripts/build_cool_figures.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/cool_figures.ipynb
    (or: sbatch scripts/sbatch/build_cool_figures.sh -- the data load needs a compute node)

Needs the real data (bigtb_ref.REAL_*) and the AlphaFold structures
(scripts/fetch_af_structures.py)."""
import json
from pathlib import Path

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})

md(r"""
# Cool figures
""")

code(r"""
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

PROJECT = next(p for p in [Path.cwd().parent, Path.cwd(),
                           Path("/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion")]
               if (p / "training" / "checkpoint.py").exists())
sys.path.insert(0, str(PROJECT))
warnings.filterwarnings("ignore")

from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV
from datasets import load_dataset
from datasets import structure as st
from datasets.biochem import RESIDUES_WITH_STOP
from models import LocusFusionV2Net

GENES = [p.stem for p in sorted(Path(st._default_dir()).glob("*.npz"))]   # the 17 coding genes
GAP = 4                                                                     # '-' channel of A C T G -
""")

md(r"""
## Mutated residues on the AlphaFold fold, one isolate per gene

For each gene, the isolate with the most mutated residues, i.e. distinct codons
that have a C-alpha coordinate. Ties go to the isolate whose residues are
furthest apart. All 17,942 isolates are searched, within the `locusfusion_v2`
16-token cap. rrs and rrl have no protein structure and are not shown.

Labels read H37Rv residue, position, then the isolate's residue. `del` means
the codon lost a base; a synonymous change repeats the H37Rv residue.
""")

code(r"""
picks = {}
for gene in GENES:
    data = load_dataset("ISONIAZID", ["dna", "protein", "structure"], REAL_GENOTYPE_DIR,
                        REAL_PHENOTYPE_CSV, loci=[gene], per_modality_branch=False, delta=True,
                        protein_frame="reference", verbose=False)
    blocks = {b.modality: b for b in data.blocks}
    net = LocusFusionV2Net.from_blocks(data.blocks, n_drugs=1)
    xs = [torch.from_numpy(b.array).float() for b in data.blocks]
    r = net.variant_report(xs)[0]
    located = r["struct_mask"].diagonal(dim1=1, dim2=2)
    dist = r["struct_dist"]

    def score(b):
        cod = sorted(set(r["codon"][b][located[b]].tolist()))
        if not cod:
            return (0, 0.0)
        idx = [int(torch.nonzero(r["codon"][b] == c)[0]) for c in cod]
        return (len(cod), float(dist[b][idx][:, idx].max()))
    cand = torch.nonzero(located.any(1)).flatten().tolist()
    b = max(cand, key=score)

    ref_aa = blocks["protein"].meta["ref_residues"]
    prot = blocks["protein"].array[b]                     # (21, n_codons), delta-encoded
    dna = blocks["dna"].array[b]                          # (5, record columns)
    cod = sorted(set(r["codon"][b][located[b]].tolist()))
    first = [int(torch.nonzero(r["codon"][b] == c)[0]) for c in cod]
    labels = []
    for c in cod:
        cols = r["column_in_block"][b][(r["codon"][b] == c) & r["valid"][b]].tolist()
        if np.abs(prot[:, c]).sum() > 0:
            alt = RESIDUES_WITH_STOP[int(np.argmax(prot[:, c]))]
        elif any(dna[GAP, col] > 0 for col in cols):
            alt = "del"
        else:
            alt = ref_aa[c]
        labels.append(f"{ref_aa[c]}{c + 1}{alt}")
    picks[gene] = {"isolate": data.isolate_ids[b], "codons": cod, "labels": labels,
                   "D": dist[b][first][:, first].numpy(),
                   "n_tokens": int(r["valid"][b].sum()),
                   "n_with_located": int(located.any(1).sum())}
    del data, blocks, net, xs, r, located, dist

display(pd.DataFrame({g: {"isolate": p["isolate"], "mutated residues": len(p["codons"]),
                          "tokens": p["n_tokens"], "isolates with ≥1 located residue": p["n_with_located"],
                          "residues": ", ".join(p["labels"])}
                      for g, p in picks.items()}).T.rename_axis("gene"))
""")

md(r"""
### Figure 1. C-alpha distance between every pair of residues

One panel per gene: the full residue-by-residue C-alpha distance map of its
AlphaFold model, with both axes in residue number. Orange points on the
diagonal mark the residues mutated in that gene's isolate (the isolate in
Figure 2). All panels share one colour scale.
""")

code(r"""
NCOL = 5
nrow = -(-len(GENES) // NCOL)
full = {g: st.residue_distances(g)[:-1, :-1] for g in GENES}      # drop the stop codon (no residue)
vmax = max(float(np.nanmax(D)) for D in full.values())
fig, axes = plt.subplots(nrow, NCOL, figsize=(5.0 * NCOL, 4.8 * nrow), constrained_layout=True)
for ax, gene in zip(axes.flat, GENES):
    D, p = full[gene], picks[gene]
    n = D.shape[0]
    im = ax.imshow(D, cmap="viridis_r", vmin=0, vmax=vmax, extent=(0.5, n + 0.5, n + 0.5, 0.5),
                   interpolation="nearest")
    res = np.array(p["codons"]) + 1
    ax.scatter(res, res, s=26, color="#eb6834", edgecolor="white", linewidth=0.8, zorder=3,
               label="mutated residues")
    ax.set_xlabel("residue", fontsize=8); ax.set_ylabel("residue", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{gene}  ·  {n} residues", fontsize=11)
for ax in list(axes.flat)[len(GENES):]:
    ax.axis("off")
axes.flat[0].legend(loc="lower left", fontsize=7, frameon=True)
fig.colorbar(im, ax=axes, shrink=0.5, label="C-alpha distance (Å)")
fig.suptitle("C-alpha distance between every pair of residues, per gene (AlphaFold, H37Rv)", fontsize=15)
plt.show()
""")

md(r"""
### Figure 2. Mutated residues on the AlphaFold C-alpha trace

Same genes and isolates as Figure 1. The grey line is the gene's C-alpha trace,
orange points are the mutated residues, and lines join every pair of them,
coloured by distance on Figure 1's scale. A run of three or more consecutive
mutated residues gets one label.
""")

code(r"""
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def label_groups(codons, labels):
    # [(anchor index, text)]: single residues keep their label; a run of >= 3
    # consecutive codons gets one label at its middle residue
    runs, cur = [], [0]
    for i in range(1, len(codons)):
        if codons[i] == codons[i - 1] + 1:
            cur.append(i)
        else:
            runs.append(cur); cur = [i]
    runs.append(cur)
    out = []
    for run in runs:
        if len(run) < 3:
            out += [(i, labels[i]) for i in run]
            continue
        a, b = labels[run[0]], labels[run[-1]]
        first, last = a[:-3] if a.endswith("del") else a[:-1], b[:-3] if b.endswith("del") else b[:-1]
        kind = "del" if all(labels[i].endswith("del") for i in run) else f"{len(run)} changed"
        out.append((run[len(run) // 2], f"{first}–{last} {kind}"))
    return out

norm = Normalize(0, vmax)
cmap = plt.get_cmap("viridis_r")
NC3 = 3
nr3 = -(-len(GENES) // NC3)
fig = plt.figure(figsize=(9.5 * NC3, 8.8 * nr3))
for k, gene in enumerate(GENES):
    p = picks[gene]
    ca = st.load_structure(gene)["ca"]
    ok = ~np.isnan(ca).any(1)
    ax = fig.add_subplot(nr3, NC3, k + 1, projection="3d")
    ax.plot(*ca[ok].T, color="#8f8e87", lw=0.8, alpha=0.9)
    pts = ca[p["codons"]]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            seg = np.stack([pts[i], pts[j]])
            ax.plot(*seg.T, color=cmap(norm(p["D"][i, j])), lw=1.6, alpha=0.9)
    ax.scatter(*pts.T, s=110, color="#eb6834", edgecolor="white", linewidth=1.0, depthshade=False,
               zorder=5)
    for i, text in label_groups(p["codons"], p["labels"]):
        ax.text(*pts[i], f"  {text}", fontsize=12, fontweight="bold", color="#1a1a19", zorder=6)
    span = np.ptp(ca[ok], axis=0)
    ax.set_box_aspect(span / span.max(), zoom=1.35)
    ax.set_axis_off()
    ax.set_title(f"{gene}  ·  {p['isolate']}  ·  {len(p['codons'])} residues", fontsize=15, pad=0)
fig.subplots_adjust(left=0.0, right=0.94, bottom=0.01, top=0.96, wspace=0.0, hspace=0.04)
cax = fig.add_axes([0.95, 0.35, 0.01, 0.3])
fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="residue-pair distance (Å)")
fig.suptitle("Mutated residues on the AlphaFold C-alpha trace, per gene, isolate with the most mutated residues",
             fontsize=20)
plt.show()
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                  "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "cool_figures.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}")
