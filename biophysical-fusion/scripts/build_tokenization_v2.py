"""Builds notebooks/tokenization_v2.ipynb -- locusfusion's tokenizer against
locusfusion_v2's, on real ISONIAZID isolates.

    python scripts/build_tokenization_v2.py            # write the notebook
    jupyter nbconvert --to notebook --execute --inplace notebooks/tokenization_v2.ipynb

Needs the real data (bigtb_ref.REAL_*) and no trained model: every number in it
comes from the two tokenizers, not from a run."""
import json, sys
from pathlib import Path

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})

md(r"""
# Tokenization: `locusfusion` vs `locusfusion_v2`

Both models tokenize the **columns where an isolate differs from H37Rv** and
feed the tokens through the same two-stage transformer. They differ in what a
token *is*.

| | `locusfusion` | `locusfusion_v2` |
|---|---|---|
| what makes a token | any modality's changed column | a changed **nucleotide** column |
| streams | `nt`, `aa`, `reg` tokenized separately | one merged 5'->3' column axis per locus |
| cap | 16 per (locus, stream) | 16 per **locus** |
| protein / biophysical | their own tokens, on their own axis | channels on the DNA token, read at that column's codon |
| feature width | 42 | 38 |
| column -> codon | a LEARNED per-locus offset | the real map, off `FeatureBlock.meta` |
| protein residue k | the isolate's own k-th codon, degapped | H37Rv codon k in every isolate; STOP is a residue |
| promoter position | its own stream, counted back from the window's end | its H37Rv genome position; columns the gene record also covers are dropped |

The last two rows are what make the rest possible. `datasets.cds.block_frame`
places every nucleotide block in the H37Rv genome by exact search and gives a
coding record its CDS window. A DNA column's codon is then the count of
**non-gap reference columns** since the start codon — the H37Rv row carries gaps
inside the CDS window (katG: 53 in a 2,276-column span holding a 2,223 bp CDS),
so `(column - start) // 3` is wrong — and a promoter column sits at its real
distance from that start codon (§4). The protein row is what keeps the residue on
that codon when an isolate carries an indel (§5).
""")

code(r"""
import sys
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

from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR
from datasets import biochem, cds, load_dataset
from datasets.sequences import NT_CHANNELS, reference_row
from models import LocusFusionNet, LocusFusionV2Net
from models import locusfusion as lf1
from models import locusfusion_v2 as lf2

DRUG, N = "ISONIAZID", 4000          # first N isolates, enough for every count below
MODALITIES = ["dna", "protein", "biophysical", "regulatory"]

# v1 reads the isolate-frame protein/biophysical blocks it was built on; v2 reads
# the same residues on the H37Rv codon grid. DNA and promoter blocks are shared.
data = load_dataset(DRUG, MODALITIES, REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV,
                    regulatory_dir=REAL_REGULATORY_DIR,
                    per_modality_branch=False, delta=True, verbose=False)
grid = load_dataset(DRUG, ["protein", "biophysical"], REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV,
                    per_modality_branch=False, delta=True, protein_frame="reference",
                    verbose=False)
assert grid.isolate_ids == data.isolate_ids
blocks_v1 = data.blocks
blocks_v2 = [{b.name: b for b in grid.blocks}.get(b.name, b) for b in data.blocks]
xs1 = [torch.from_numpy(b.array[:N]).float() for b in blocks_v1]
xs2 = [torch.from_numpy(b.array[:N]).float() for b in blocks_v2]

v1 = LocusFusionNet.from_blocks(blocks_v1, n_drugs=1)
v2 = LocusFusionV2Net.from_blocks(blocks_v2, n_drugs=1)

print(f"{DRUG}: {len(data.isolate_ids):,} isolates, {N:,} used in §1-3")
print("protein/biophysical  v1:", [(b.name, b.array.shape[1:]) for b in blocks_v1
                                   if b.modality in ("protein", "biophysical")])
print("                     v2:", [(b.name, b.array.shape[1:]) for b in blocks_v2
                                   if b.modality in ("protein", "biophysical")])
print(f"parameters  v1 {sum(p.numel() for p in v1.parameters()):,}"
      f"   v2 {sum(p.numel() for p in v2.parameters()):,}")
display(pd.DataFrame(v2.frames).T.rename_axis("locus"))
""")

md(r"""
## 1. The token budget

Slots are what the model pads to; tokens are what an isolate actually fills.
""")

code(r"""
rep1, rep2 = v1.variant_report(xs1), v2.variant_report(xs2)

rows = []
for locus in v2.loci:
    r1 = [r for r in rep1 if r["locus"] == locus]
    r2 = [r for r in rep2 if r["locus"] == locus][0]
    n1 = sum(r["valid"].sum(1) for r in r1).float()      # tokens per isolate, v1
    n2 = r2["valid"].sum(1).float()
    rows.append({
        "locus": locus,
        "streams (v1)": " + ".join(r["stream"] for r in r1),
        "slots v1": v1.tokens_per_locus, "slots v2": v2.tokens_per_locus,
        "tokens/isolate v1": n1.mean().item(), "tokens/isolate v2": n2.mean().item(),
        "max v1": int(n1.max()), "max v2": int(n2.max()),
        "truncated v2": (r2["n_variants"] > v2.max_variants).float().mean().item(),
    })
budget = pd.DataFrame(rows)
display(budget.style.format({"tokens/isolate v1": "{:.2f}", "tokens/isolate v2": "{:.2f}",
                             "truncated v2": "{:.2%}"}).hide(axis="index"))
print(f"slots per isolate:  v1 {v1.tokens_per_locus * len(v1.loci)}"
      f"   v2 {v2.tokens_per_locus * len(v2.loci)}")
print(f"tokens per isolate: v1 {budget['tokens/isolate v1'].sum():.2f}"
      f"   v2 {budget['tokens/isolate v2'].sum():.2f}")
""")

md(r"""
The v1 count is larger because a nonsynonymous SNP is two tokens there — one
nucleotide token, and one residue token carrying the biophysics — against one
here. The per-locus summary token (`[WT]` in v1, `[NULL]` in v2) is excluded
from both counts.
""")

code(r"""
tok1 = sum(r["valid"].sum(1) for r in rep1).numpy()
tok2 = sum(r["valid"].sum(1) for r in rep2).numpy()
hi = int(max(tok1.max(), tok2.max()))
fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
bins = np.arange(0, hi + 2) - 0.5
ax.hist([tok1, tok2], bins=bins, label=["locusfusion", "locusfusion_v2"],
        color=["#3B6FA0", "#C4703A"])
ax.set_xlabel("variant tokens per isolate (both loci, per-locus summary token excluded)")
ax.set_ylabel("isolates")
ax.set_xlim(-0.5, min(hi, 24) + 0.5)
ax.set_title(f"{DRUG}: variant tokens per isolate, {N:,} isolates")
ax.legend(frameon=False)
plt.show()
""")

md(r"""
## 2. The feature vector

Same table for both, drawn from `SLOTS` and the flag constants in each module.
""")

code(r"""
def layout(mod):
    rows = [{"dims": f"{lo}:{hi}", "n": hi - lo, "what": f"{m} one-hot / values"}
            for m, (lo, hi) in mod.SLOTS.items()]
    flags = [(n[2:].lower(), getattr(mod, n)) for n in dir(mod)
             if n.startswith("F_") and n != "F_PHASE"]
    for name, i in sorted(flags, key=lambda t: t[1]):
        rows.append({"dims": str(i), "n": 1, "what": f"flag: {name}"})
    rows.append({"dims": f"{mod.F_PHASE}:{mod.F_PHASE + 3}", "n": 3, "what": "codon phase"})
    out = pd.DataFrame(rows).sort_values("dims", key=lambda s: s.str.split(":").str[0].astype(int))
    return out.astype({"n": str})

a, b = layout(lf1), layout(lf2)
a.columns = pd.MultiIndex.from_product([[f"locusfusion (C_TOK={lf1.C_TOK})"], a.columns])
b.columns = pd.MultiIndex.from_product([[f"locusfusion_v2 (C_TOK={lf2.C_TOK})"], b.columns])
display(pd.concat([a.reset_index(drop=True), b.reset_index(drop=True)], axis=1).fillna(""))
""")

md(r"""
The five channels v2 drops: `is_nt` / `is_aa` / `is_reg` said which stream a
token came from and there are no streams; `is_gap` was a copy of the `-` channel
of the nucleotide one-hot, which is still there; `uncovered` was a locus-level
threshold broadcast onto every token, and the per-locus `[NULL]` token still
carries the occupied fraction it was thresholded from. `is_null` is v1's `is_wt`
under a neutral name: the token is always present, is all a locus shows when it
matches H37Rv everywhere, and is the slot the locus summary is read from.

## 3. One isolate, both tokenizers

The decoder below turns a token back into a variant: locus, alignment column,
codon, reference base and residue against the isolate's.
""")

code(r"""
REF_NT = {g: reference_row(REAL_GENOTYPE_DIR, g) for g in v2.loci}
REF_AA = {g: biochem.translate_reference_frame(REF_NT[g], REF_NT[g],
                                               cds.cds_columns(REAL_GENOTYPE_DIR, g))
          for g in v2.loci if REF_NT[g]}          # reference codon k -> residue, v2's numbering
REF_REG = {g: reference_row(REAL_REGULATORY_DIR, g) for g in v2.loci}
S2, S1 = lf2.SLOTS, lf1.SLOTS

def span(f, slot):
    return f[slot[0]:slot[1]]

def letter(vec, alphabet):
    # the one-hot's letter, or '' when the slot is empty (nothing changed there)
    j = int(np.argmax(vec))
    return alphabet[j] if vec[j] > 0 else ""

def v2_tokens(i):
    feat, coord, valid, stats = v2._locus_tokens([x[i:i + 1] for x in xs2])
    out = []
    for li, locus in enumerate(v2.loci):
        r = [r for r in rep2 if r["locus"] == locus][0]
        for t in range(1, v2.tokens_per_locus):
            if not valid[0, li, t]:
                continue
            f = feat[0, li, t].numpy()
            col, codon = int(r["column_in_block"][i, t - 1]), int(r["codon"][i, t - 1])
            region = "cds" if int(r["region"][i, t - 1]) else "promoter"
            aa = letter(span(f, S2["protein"]), biochem.RESIDUES_WITH_STOP)
            base = letter(span(f, S2["dna"]), NT_CHANNELS) or letter(span(f, S2["regulatory"]), NT_CHANNELS)
            ph = f[lf2.F_PHASE:lf2.F_PHASE + 3]
            out.append({
                "locus": locus, "region": region, "column": col, "codon": codon,
                "coord (codons)": float(coord[0, li, t]),
                "base": f"{(REF_NT if region == 'cds' else REF_REG)[locus][col]}->{base}",
                "residue": (f"{REF_AA[locus][codon]}{codon + 1}{aa}"
                            if aa and locus in REF_AA and codon < len(REF_AA[locus]) else ""),
                "biophysical": (", ".join(f"{v:+.2f}" for v in span(f, S2["biophysical"])) if aa else ""),
                "phase": int(np.argmax(ph)) if ph.sum() else -1,
            })
    return pd.DataFrame(out)

def v1_tokens(i):
    feat, coord, valid, stats = v1._locus_tokens([x[i:i + 1] for x in xs1])
    out, cursor = [], {}
    for li, locus in enumerate(v1.loci):
        t = 1
        for stream, members, length in v1._plan[li]:
            mods = [m for _, m in members]
            r = [r for r in rep1 if r["locus"] == locus and r["stream"] == stream][0]
            for j in range(v1.max_variants):
                if valid[0, li, t]:
                    f = feat[0, li, t].numpy()
                    out.append({"locus": locus, "stream": stream,
                                "modalities": "+".join(mods),
                                "index": int(r["columns"][i, j]),
                                "coord (codons)": float(coord[0, li, t]),
                                "nt": letter(span(f, S1["dna"]), NT_CHANNELS),
                                "aa": letter(span(f, S1["protein"]), biochem.AMINO_ACIDS),
                                "reg": letter(span(f, S1["regulatory"]), NT_CHANNELS)})
                t += 1
    return pd.DataFrame(out)

# an isolate carrying the katG S315T token plus a promoter variant
cand = [i for i in range(N)
        if 2 <= sum(int(r["valid"][i].sum()) for r in rep2) <= 4
        and any(int(c) == 314 for r in rep2 if r["locus"] == "katG"
                for c in r["codon"][i][r["valid"][i]])]
ISO = cand[0]
print(f"isolate {data.isolate_ids[ISO]}  ({DRUG} = "
      f"{ {0: 'R', 1: 'S'}.get(int(data.y[ISO]), '?') })")
print("\nlocusfusion_v2 — one token per changed base pair:")
display(v2_tokens(ISO).style.hide(axis="index"))
print("locusfusion — the same isolate, one token per changed column per stream:")
display(v1_tokens(ISO).style.hide(axis="index"))
""")

md(r"""
Read the two tables together: v2's single katG row carries the new base, the
residue that codon now codes for and that residue's biophysics; v1 spends a
separate token on each and leaves relating them to attention.

## 4. Where each promoter window sits

Every locus with both a gene record and a WHO promoter window, placed by H37Rv
genome position. Built from the reference rows alone, so it covers all 18 loci
(16 with a promoter window) rather than ISONIAZID's two. Positions are codons
from the gene's own first base, in its 5'->3' direction; `dropped` is the
promoter columns the gene record already covers. *rrs* and *rrl* are placed on
their GenBank rRNA feature and have no codons — their tokens carry a base and a
position only.
""")

code(r"""
from datasets.sequences import reference_row as _ref

ALL = sorted({g for gs in __import__("bigtb_ref").tb.DRUG_TO_LOCI.values() for g in gs})
keys, specs, frames = [], [], []
for g in ALL:
    r = _ref(REAL_GENOTYPE_DIR, g)
    keys.append(("dna", g)); specs.append((5, len(r)))
    frames.append(cds.block_frame(REAL_GENOTYPE_DIR, g, len(r), translated=True))
    p = _ref(REAL_REGULATORY_DIR, g)
    if p:
        keys.append(("regulatory", g)); specs.append((5, len(p)))
        frames.append(cds.block_frame(REAL_REGULATORY_DIR, g, len(p)))
v2_all = LocusFusionV2Net(keys, specs, block_frames=frames)

rows, spans = [], []
for li, p in enumerate(v2_all._plan):
    coord = getattr(v2_all, f"coord_map_{li}").numpy()
    codon = getattr(v2_all, f"codon_map_{li}").numpy()
    keep = getattr(v2_all, f"keep_map_{li}").numpy()
    rec = next(b for b in p["nt"] if b["modality"] == "dna")
    prom = next((b for b in p["nt"] if b["modality"] == "regulatory"), None)
    rc = coord[rec["offset"]:rec["offset"] + rec["length"]]
    fr = v2_all.frames[p["locus"]]
    gene_end = float(rc[rec["frame"]["cds_end"] - 1]) + 1 / 3
    span = {"locus": p["locus"], "record": (rc.min(), rc.max()),
            "cds": (0.0, gene_end) if fr["has_cds"] else None,
            "rna": None if fr["has_cds"] else (0.0, gene_end),
            "kept": None, "dropped": None}
    row = {"locus": p["locus"], "gene": "CDS" if fr["has_cds"] else "rRNA",
           "strand": rec["frame"]["strand"],
           "codons": int(codon.max()) + 1 if fr["has_cds"] else None,
           "record from": float(rc.min())}
    if prom is not None:
        pc = coord[prom["offset"]:prom["offset"] + prom["length"]]
        pk = keep[prom["offset"]:prom["offset"] + prom["length"]]
        span["kept"] = (pc[pk].min(), pc[pk].max()) if pk.any() else None
        span["dropped"] = (pc[~pk].min(), pc[~pk].max()) if (~pk).any() else None
        row.update({"promoter start": float(pc.min()), "promoter end": float(pc.max()),
                    "promoter cols": prom["length"], "dropped": int((~pk).sum())})
    rows.append(row); spans.append(span)

placement = pd.DataFrame(rows).set_index("locus")
display(placement.style.format({"record from": "{:.1f}", "promoter start": "{:.1f}",
                                "codons": "{:.0f}",
                                "promoter end": "{:.1f}", "dropped": "{:.0f}",
                                "promoter cols": "{:.0f}"}, na_rep=""))
print(f"loci whose window overlaps the gene record: {(placement['dropped'] > 0).sum()}"
      f" of {placement['dropped'].notna().sum()}")
""")

code(r"""
C_CDS, C_KEPT, C_DROP, C_RNA, C_REC = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#c9c8c1"
order = sorted(spans, key=lambda d: d["kept"][0] if d["kept"] else d["record"][0])
fig, ax = plt.subplots(figsize=(8.4, 0.34 * len(order) + 1.3), constrained_layout=True)
for y, d in enumerate(order):
    def bar(rng, color, h, label=None):
        if rng is not None:
            ax.barh(y, rng[1] - rng[0], left=rng[0], height=h, color=color,
                    edgecolor="white", linewidth=1, label=label)
    bar(d["record"], C_REC, 0.62, "gene record" if y == 0 else None)
    bar(d["cds"], C_CDS, 0.36, "CDS" if y == 0 else None)
    bar(d["rna"], C_RNA, 0.36)
    bar(d["kept"], C_KEPT, 0.36, "promoter window, kept" if y == 0 else None)
    bar(d["dropped"], C_DROP, 0.36, "promoter window, dropped" if y == 0 else None)
ax.set_yticks(range(len(order)), [d["locus"] for d in order])
ax.axvline(0, color="#6b6a64", linewidth=1)
ax.set_xlabel("codons from the start codon")
ax.set_title("Gene record, gene body and promoter window positions, 18 loci")
from matplotlib.patches import Patch
series = [("gene record", C_REC), ("CDS", C_CDS), ("rRNA gene", C_RNA),
          ("promoter window, kept", C_KEPT), ("promoter window, dropped", C_DROP)]
ax.legend([Patch(color=c) for _, c in series], [n for n, _ in series], frameon=False,
          fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.07))
ax.grid(axis="x", color="#e6e5df", linewidth=0.8); ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
plt.show()
""")

md(r"""
Six windows end upstream of their CDS — inhA's by 732 bp, because the WHO
`inhA` row is the fabG1 promoter — and ten overlap the gene record by 30-130 bp.
Under the old adjacency convention the first group sat hundreds of codons from
their true position and the second tokenized one base twice.

What the drop does to real isolates, on the drug whose loci overlap:
""")

code(r"""
sm = load_dataset("STREPTOMYCIN", ["dna", "regulatory"], REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV,
                  regulatory_dir=REAL_REGULATORY_DIR,
                  per_modality_branch=False, delta=True, verbose=False)
sm_model = LocusFusionV2Net.from_blocks(sm.blocks, n_drugs=1)
sx = [torch.from_numpy(b.array[:N]).float() for b in sm.blocks]
rows = []
for li, p in enumerate(sm_model._plan):
    occ = torch.cat([sx[b["index"]].abs().sum(1) > 0 for b in p["nt"]], dim=1)
    keep = getattr(sm_model, f"keep_map_{li}")
    rows.append({"locus": p["locus"],
                 "promoter columns dropped": sm_model.frames[p["locus"]]["duplicate_columns"],
                 "variant columns/isolate, both copies": occ.sum(1).float().mean().item(),
                 "after the drop": (occ & keep).sum(1).float().mean().item(),
                 "isolates losing a duplicate token": ((occ & ~keep).sum(1) > 0).float().mean().item()})
display(pd.DataFrame(rows).style.format({"variant columns/isolate, both copies": "{:.3f}",
                                         "after the drop": "{:.3f}",
                                         "isolates losing a duplicate token": "{:.2%}"})
        .hide(axis="index"))
""")

md(r"""
## 5. Does the residue land on the right token?

Every codon whose residue changed should be reachable from one of that isolate's
DNA tokens. Same DNA tokens, both protein frames, all isolates (in chunks):
`isolate` is the degapped translation v1 and every other architecture read,
`reference` is the H37Rv codon grid v2 reads.
""")

code(r"""
CHUNK = 4000
n_all = len(data.isolate_ids)
nt_idx = sorted({b["index"] for p in v2._plan for b in p["nt"]})

# per locus, per isolate: the codons v2's DNA tokens sit on, and the true variant count
reach = {l: ([], []) for l in v2.loci}
for a in range(0, n_all, CHUNK):
    chunk = [None] * len(blocks_v2)
    for k in nt_idx:
        chunk[k] = torch.from_numpy(blocks_v2[k].array[a:a + CHUNK]).float()
    for r in v2.variant_report(chunk):
        cod, ok = r["codon"].numpy(), r["valid"].numpy()
        reach[r["locus"]][0].extend(c[v] for c, v in zip(cod, ok))
        reach[r["locus"]][1].extend(r["n_variants"].numpy())

rows = []
for locus in v2.loci:
    codons, n_var = reach[locus]
    row = {"locus": locus}
    for frame, blocks in (("isolate", blocks_v1), ("reference", blocks_v2)):
        prot = next(b.array for b in blocks if b.name == f"protein:{locus}")
        changed = np.abs(prot).sum(1) > 0
        hit = [i for i in np.flatnonzero(changed.any(1))
               if not np.isin(np.flatnonzero(changed[i]), codons[i][codons[i] >= 0]).all()]
        row[f"changed residue, {frame}"] = int(changed.any(1).sum())
        row[f"unreachable, {frame}"] = len(hit)
        if frame == "reference":
            row["...of those over the 16-token cap"] = int(sum(n_var[i] > v2.max_variants for i in hit))
            row["stop gained"] = int((prot[:, 20] > 0).any(1).sum())
    rows.append(row)
display(pd.DataFrame(rows).style.hide(axis="index"))
""")

md(r"""
In the isolate frame an indel slides every residue after it off the reference
numbering, and a frameshift turns the rest of the protein into phantom
substitutions at bases that never changed; those are the isolate-frame misses. On
the reference grid a residue can only change where one of its own columns did, so
what is left is the 16-token cap: an isolate with more changed columns than that
keeps the first 16 in 5'->3' order and loses the rest. `stop gained` is new with
the reference frame — a nonsense mutation used to be an empty residue slot, the
same as a synonymous change.

## Caveats

* **"locusfusion" here is this branch's `models/locusfusion.py` — the tokenizer
  `newmodels_full` ran, not `locusfusion_fixed`'s.** The fixed tokenizer (exact
  coordinates, symbol ids) lives on the branch `worktree-lf-token-offset-fix`. The
  v1 coordinate mismatch in §3 (katG S315 at 357.7 against 314) is the bug that
  branch fixed, so §1-3 compare v2 against the pre-fix v1, and the trained control
  for v2 is `results/experiments/locusfusion_fixed`, not this notebook's v1.
* v1 and v2 read different protein/biophysical blocks, each the input it was
  built for: v1 the isolate frame (20 channels, left-aligned), v2 the H37Rv codon
  grid (21 channels with STOP, one column per reference codon). DNA and promoter
  blocks are identical.
* §1-3 are one drug (`ISONIAZID`, two loci) on the first 4,000 isolates, §5 on
  all of them;
  §4 places all 18 loci from their reference rows and measures the drop on
  STREPTOMYCIN. Token counts per locus differ across the 19-locus runs — *rrs*
  and *rrl* are hypervariable and are where the 16-token cap binds.
* ISONIAZID's two loci do not overlap their promoter windows, so the drop in §4
  leaves every ISONIAZID number in §1 and §5 unchanged.
* No model here is trained. This notebook compares what the two tokenizers
  *emit*; whether the merged token trains better is a run, not a table.
* The residue attached to a token is the isolate's residue at that codon, which
  is the *changed* one only because the protein block is delta-encoded — a
  synonymous substitution leaves the protein slots zero by construction. A codon
  that lost a base to a deletion reads `-` and carries no residue; the deletion
  is its gap token.
* Placement needs each block's reference row to be an exact H37Rv substring.
  It is for all 18 loci here; a block that is not falls back to the old
  convention — promoter counted back from the window's end, nothing dropped —
  and `LocusFusionV2Net` warns naming the locus.
* `coord` for a promoter column is its genome distance from the start codon in
  base pairs / 3. That is a codon *unit*, not a codon — there is no reading frame
  upstream, which is why promoter tokens carry no phase.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
      "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(sys.argv[1] if len(sys.argv) > 1 else
           Path(__file__).resolve().parent.parent / "notebooks" / "tokenization_v2.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
