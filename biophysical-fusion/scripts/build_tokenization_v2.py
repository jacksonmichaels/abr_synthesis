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
| the reference | not on the token | the change in residue properties from H37Rv's; H37Rv's residue where a codon did not change |
| an uncalled base (`N`) | the all-zero column, i.e. "no variant" | the same: no channel of its own, so no token |
| feature width | 42 | 41 (57 with `structure`, §6) |
| column -> codon | a LEARNED per-locus offset | the real map, off `FeatureBlock.meta` |
| protein residue k | the isolate's own k-th codon, degapped | H37Rv codon k in every isolate; STOP is a residue |
| promoter position | its own stream, counted back from the window's end | its H37Rv genome position; columns the gene record also covers are dropped |
| positional encoding | wavelengths 2π–2π·10⁴ codons, 64 dims | 1/3–4096 codons, the range the data occupies, 32 dims |

The last two rows are what make the rest possible. `datasets.cds.block_frame`
places every nucleotide block in the H37Rv genome by exact search and gives a
coding record its CDS window. A DNA column's codon is then the count of
**non-gap reference columns** since the start codon — the H37Rv row carries gaps
inside the CDS window (katG: 53 in a 2,276-column span holding a 2,223 bp CDS),
so `(column - start) // 3` is wrong — and a promoter column sits at its real
distance from that start codon (§4). The protein row is what keeps the residue on
that codon when an isolate carries an indel (§5).

**A token states a change, not a state.** v2's token carries the isolate's base
and residue, that residue's properties, and how those properties differ from
H37Rv's residue at that codon. `S315T` reads as "threonine, +0.47 molecular
weight, +0.49 hydrophobicity" without the model having to learn the
subtraction. Where a substitution left the codon's residue unchanged, the
reference fills the slot, so a synonymous substitution says "still serine" with
a zero delta. Otherwise it would leave an empty slot that looks like no residue
at all.
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
v2data = load_dataset(DRUG, MODALITIES, REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV,
                      regulatory_dir=REAL_REGULATORY_DIR,
                      per_modality_branch=False, delta=True,
                      protein_frame="reference", verbose=False)
assert v2data.isolate_ids == data.isolate_ids
blocks_v1 = data.blocks
blocks_v2 = v2data.blocks
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
    span_ = getattr(mod, "D_BIO", None)                # v2 only
    if span_:
        rows.append({"dims": f"{span_[0]}:{span_[1]}", "n": span_[1] - span_[0],
                     "what": "properties, isolate - H37Rv"})
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
What v2 adds: three channels for the change in residue properties (isolate
minus H37Rv) and a 21st residue channel, STOP. It has no channel for a symbol
that could not be read. An `N` is the all-zero column in both models, which
under delta encoding is exactly what a position *matching* the reference looks
like, so a failed base call reads as "no variant here".

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
V2_RESIDUES = lf2._RES_ORDER          # 20 residues + STOP

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
            aa = letter(span(f, S2["protein"]), V2_RESIDUES)
            base = (letter(span(f, S2["dna"]), NT_CHANNELS)
                    or letter(span(f, S2["regulatory"]), NT_CHANNELS))
            ph = f[lf2.F_PHASE:lf2.F_PHASE + 3]
            out.append({
                "locus": locus, "region": region, "column": col, "codon": codon,
                "coord (codons)": float(coord[0, li, t]),
                "base": f"{(REF_NT if region == 'cds' else REF_REG)[locus][col]}->{base}",
                "residue": (f"{REF_AA[locus][codon]}{codon + 1}{aa}"
                            if aa and locus in REF_AA and codon < len(REF_AA[locus]) else ""),
                "properties (isolate)": (", ".join(f"{v:+.2f}" for v in span(f, S2["biophysical"]))
                                         if aa else ""),
                "change": (", ".join(f"{v:+.2f}" for v in f[lf2.D_BIO[0]:lf2.D_BIO[1]])
                           if aa else ""),
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
    keys.append(("dna", g)); specs.append((6, len(r)))
    frames.append(cds.block_frame(REAL_GENOTYPE_DIR, g, len(r), translated=True))
    p = _ref(REAL_REGULATORY_DIR, g)
    if p:
        keys.append(("regulatory", g)); specs.append((6, len(p)))
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
                  per_modality_branch=False, delta=True,
                  verbose=False)
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
""")

md(r"""
## 6. Structural features (tentative)

> **Tentative — for discussion.** Nothing here is settled or comes from a trained
> model. The model currently uses **closeness exp(−d / 10 Å)**. This section lays
> out the alternatives.

When `structure` is loaded (`--modalities ... structure`), each coding locus gets
the C-alpha coordinates of its H37Rv AlphaFold model, which gives a distance *d*
between any two of a locus's variant tokens.

- **Slot-aligned.** Every candidate encoding gives variant token *i* row *i* of
  the locus's 16×16 matrix, so channel *j* describes token *i* → token *j*.
- **No position.** Promoter, flank, insertion, rRNA, stop-codon and padding
  tokens, and `[NULL]`, have no C-alpha. Their row, and every other token's
  entry for them, reads as "missing".
- **Base layout unchanged.** The channels are appended after it.

| encoding | channels | value | own slot / same codon | missing |
|---|---|---|---|---|
| distance + mask | 32 | d / 10 Å, plus 1 where d exists | 0, mask 1 | 0, mask 0 |
| distance only | 16 | d / 10 Å | 0 | 0 |
| **closeness exp(−d/10)** *(current)* | 16 | exp(−d / 10 Å) | 1 | 0 |
| closeness exp(−d/30) | 16 | exp(−d / 30 Å) | 1 | 0 |
| closeness 1/(1+d/10) | 16 | 1 / (1 + d / 10 Å) | 1 | 0 |
| closeness linear | 16 | 1 − d / (the protein's largest C-alpha distance) | 1 | 0 |
| closeness percentile | 16 | 1 − (share of the protein's residue pairs closer than d) | 1 | 0 |
| contact | 16 | 1 if d < 8 Å | 1 | 0 |

The two protein-relative encodings (linear, percentile) rescale per gene, so the
same value means "near for this protein" in a small protein and a large one.
""")

code(r"""
from datasets import structure as st
from models.locusfusion_v2 import STRUCT_LENGTH

sdata = load_dataset(DRUG, ["dna", "structure"], REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV,
                     per_modality_branch=False, delta=True, verbose=False)
assert sdata.isolate_ids == v2data.isolate_ids
blocks_s = blocks_v2 + [b for b in sdata.blocks if b.modality == "structure"]
xs_s = xs2 + [torch.from_numpy(b.array[:N]).float() for b in blocks_s[len(blocks_v2):]]
v2s = LocusFusionV2Net.from_blocks(blocks_s, n_drugs=1)
V = v2s.max_variants
del sdata

feat_s = v2s._locus_tokens(xs_s)[0]
assert torch.equal(feat_s[..., :lf2.C_TOK], v2._locus_tokens(xs2)[0]), "structure changed a base channel"
rep_s = v2s.variant_report(xs_s)
for li, r in enumerate(rep_s):                       # the model's channels ARE the current encoding
    if "struct_dist" in r:
        k = r["codon"].shape[1]
        want = st.ENCODINGS[st.CURRENT_ENCODING][1](r["struct_dist"], r["struct_mask"], r["locus"])[0]
        assert torch.allclose(feat_s[:, li, 1:1 + k, v2s.f_struct:v2s.f_struct + k], want, atol=1e-5)
print(f"token width {v2.c_tok} -> {v2s.c_tok} ({lf2.C_TOK} base + {V} structure); base channels "
      f"identical with and without structure; model channels == '{st.CURRENT_ENCODING}'")
""")

md(r"""
### 6.1 Which isolates have geometry to encode

A row only says something when a locus has **two or more located tokens**. The
table also counts the isolates whose tokens are in different codons at least 4
codons apart. Tokens in the same codon or adjacent codons are always 0–4 Å apart,
which the 1D position already says.
""")

code(r"""
rows = []
for li, r in enumerate(rep_s):
    v = r["valid"]
    if "struct_mask" not in r:
        rows.append({"locus": r["locus"], "tokens": int(v.sum()), "located": 0})
        continue
    loc = r["struct_mask"].diagonal(dim1=1, dim2=2)
    c = r["codon"]
    far = (c.unsqueeze(2) - c.unsqueeze(1)).abs() >= 4
    informative = (r["struct_mask"] & far).flatten(1).any(1)
    rows.append({"locus": r["locus"], "tokens": int(v.sum()), "located": int(loc.sum()),
                 "isolates with a token": int(v.any(1).sum()),
                 "with ≥2 located": int((loc.sum(1) >= 2).sum()),
                 "with a pair ≥4 codons apart": int(informative.sum())})
cov = pd.DataFrame(rows).set_index("locus")
display(cov.fillna(0).astype(int))

kr = rep_s[v2s.loci.index("katG")]
pairs = {}
for b in torch.nonzero((kr["struct_mask"] & ((kr["codon"].unsqueeze(2) - kr["codon"].unsqueeze(1)).abs() >= 4))
                       .flatten(1).any(1)).flatten().tolist():
    cod = sorted({int(x) + 1 for x in kr["codon"][b][kr["struct_mask"][b].diagonal()]})
    pairs[tuple(cod)] = pairs.get(tuple(cod), 0) + 1
top = sorted(pairs.items(), key=lambda t: -t[1])[:5]
print("katG, most common located codon sets among those isolates:")
for cod, n in top:
    print(f"  {n:5d} isolates  codons {list(cod)}")
""")

md(r"""
Most isolates have no located pair, so their structure channels only mark their
own slot. The isolates that do have one mostly share the same few codon sets,
e.g. katG 315 + 463 (S315T with R463L, a lineage marker). A fixed pair has the
same distance in every carrier, so it tells the model little beyond which pair is
present.

### 6.2 The encodings against distance

Each curve is what one channel reads at a given C-alpha distance. The histogram
underneath shows where this drug's real located token pairs fall; only pairs in
different codons are counted.
""")

code(r"""
CLOSE = [n for n in st.ENCODINGS if n.startswith("closeness") or n.startswith("contact")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]   # categorical slots, fixed order

pairs_d, pairs_l = [], []
for r in rep_s:
    if "struct_dist" not in r:
        continue
    c = r["codon"]
    k = c.shape[1]
    m = r["struct_mask"] & (c.unsqueeze(2) != c.unsqueeze(1)) & torch.triu(torch.ones(k, k, dtype=torch.bool), 1)
    pairs_d.append(r["struct_dist"][m]); pairs_l += [r["locus"]] * int(m.sum())
pairs_d = torch.cat(pairs_d)

d = torch.linspace(0, 90, 451)
one = torch.ones_like(d, dtype=torch.bool)
fig, (ax, axd, axh) = plt.subplots(3, 1, figsize=(7.6, 8.0), sharex=True, constrained_layout=True,
                                   gridspec_kw={"height_ratios": [2.2, 1, 1.2]})
for name, col in zip(CLOSE, COLORS):
    y = st.ENCODINGS[name][1](d, one, "katG")[0]
    ax.plot(d, y, color=col, lw=2.6 if name == st.CURRENT_ENCODING else 1.6,
            label=name + (" (current)" if name == st.CURRENT_ENCODING else "")
                  + (" — katG" if "linear" in name or "percentile" in name else ""))
ax.set_ylabel("channel value")
ax.set_title("Closeness and contact encodings vs C-alpha distance")
ax.legend(frameon=False, fontsize=8)
axd.plot(d, d / 10, color="#008300", lw=1.6, label="distance / 10 Å (distance only, distance + mask)")
axd.set_ylabel("channel value")
axd.set_title("Distance encodings vs C-alpha distance")
axd.legend(frameon=False, fontsize=8)
axh.hist(pairs_d.numpy(), bins=np.arange(0, 91, 2), color="#2a78d6")
axh.set_xlabel("C-alpha distance (Å)")
axh.set_ylabel("token pairs")
axh.set_title(f"{DRUG}: located token pairs in different codons, {N:,} isolates (n={len(pairs_d):,})")
for a in (ax, axd, axh):
    a.grid(color="#e6e5df", linewidth=0.8); a.set_axisbelow(True)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
plt.show()
q = np.percentile(pairs_d.numpy(), [10, 25, 50, 75, 90])
print("pair distance percentiles 10/25/50/75/90 (Å):", " / ".join(f"{v:.1f}" for v in q))
""")

code(r"""
def own_slot_unique(name):
    hits = tot = 0
    for r in rep_s:
        if "struct_dist" not in r:
            continue
        blocks = st.ENCODINGS[name][1](r["struct_dist"], r["struct_mask"], r["locus"])
        loc = r["struct_mask"].diagonal(dim1=1, dim2=2)
        if name == "distance + mask":
            same = (blocks[0] == 0) & (blocks[1] == 1)
        else:
            same = blocks[0] == blocks[0].diagonal(dim1=1, dim2=2).unsqueeze(-1)
        hits += int(((same.sum(-1) == 1) & loc).sum()); tot += int(loc.sum())
    return hits / max(tot, 1)

def at(name, dist, locus="katG"):
    return float(st.ENCODINGS[name][1](torch.tensor([dist]), torch.tensor([True]), locus)[0][0])

rows = []
for name, (ch, fn) in st.ENCODINGS.items():
    vals = torch.cat([fn(pairs_d[i:i + 1], torch.ones(1, dtype=torch.bool), l)[0]
                      for i, l in enumerate(pairs_l)]) if name in CLOSE else None
    rows.append({"encoding": name + (" (current)" if name == st.CURRENT_ENCODING else ""),
                 "channels": ch,
                 **{f"{x:g} Å": at(name, x) for x in (0, 3.8, 10, 20, 40, 60)},
                 "real pairs in 0.05–0.95": (float(((vals > 0.05) & (vals < 0.95)).float().mean())
                                             if vals is not None else np.nan),
                 "own slot unique": own_slot_unique(name),
                 "missing ≠ far": "yes" if name == "distance + mask" else "no"})
res = pd.DataFrame(rows).set_index("encoding")
display(res.style.format({c: "{:.3f}" for c in res.columns if c.endswith("Å")}
                         | {"real pairs in 0.05–0.95": "{:.0%}", "own slot unique": "{:.1%}"},
                         na_rep="—"))
""")

md(r"""
How to read the table:

- **Value columns** are what one channel reads at that distance (katG for the
  protein-relative encodings).
- **`real pairs in 0.05–0.95`** is the share of this drug's real token pairs
  that land where a closeness channel still separates distances. Near 0 or near
  1, pairs at different distances look alike. The distance encodings have no
  ceiling, so the column doesn't apply to them.
- **`own slot unique`** is the share of located tokens whose row has exactly one
  "same residue" entry. That entry is how a token can find its own slot.
- **`missing ≠ far`** is whether a missing partner (promoter, rRNA, padding)
  can be told apart from a very distant one.

### 6.3 One isolate under each encoding

This example is from ETHAMBUTOL's embA, not this drug: ISONIAZID has no isolate
with six or more located tokens in separate, non-adjacent codons, because its
multi-token isolates are deletions through consecutive codons. The pick is the
embA isolate with at least six distinct located codons whose tokens are furthest
apart on the fold.
""")

code(r"""
EX_DRUG, EX_LOCUS = "ETHAMBUTOL", "embA"
ex = load_dataset(EX_DRUG, ["dna", "structure"], REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV,
                  loci=[EX_LOCUS], per_modality_branch=False, delta=True, verbose=False)
ex_net = LocusFusionV2Net.from_blocks(ex.blocks, n_drugs=1)
ex_rep = ex_net.variant_report([torch.from_numpy(b.array).float() for b in ex.blocks])[0]
ex_ids = ex.isolate_ids
del ex

loc = ex_rep["struct_mask"].diagonal(dim1=1, dim2=2)
def score(b):
    cod = sorted(set(ex_rep["codon"][b][loc[b]].tolist()))
    if len(cod) < 6:
        return -1
    D = ex_rep["struct_dist"][b][loc[b]][:, loc[b]]
    return float(D[torch.triu(torch.ones_like(D, dtype=torch.bool), 1)].median())
cand = torch.nonzero(loc.sum(1) >= 6).flatten().tolist()
EX = max(cand, key=score)
k = int(ex_rep["valid"][EX].sum())
s = st.load_structure(EX_LOCUS)
ref_row = reference_row(REAL_GENOTYPE_DIR, EX_LOCUS)
EX_AA = biochem.translate_reference_frame(ref_row, ref_row, cds.cds_columns(REAL_GENOTYPE_DIR, EX_LOCUS))
codons = [int(c) for c in ex_rep["codon"][EX, :k]]
located = [bool(x) for x in loc[EX, :k]]
labels = [f"{EX_AA[c]}{c + 1}" if ok else ("promoter" if int(ex_rep["region"][EX, j]) == 0 else "no codon")
          for j, (c, ok) in enumerate(zip(codons, located))]
D_ex = ex_rep["struct_dist"][EX:EX + 1, :k, :k]
M_ex = ex_rep["struct_mask"][EX:EX + 1, :k, :k]
print(f"{EX_DRUG} / {EX_LOCUS}: isolate {ex_ids[EX]}, {k} tokens, {sum(located)} located: {labels}")

tab = pd.DataFrame(np.where(M_ex[0].numpy(), D_ex[0].numpy(), np.nan),
                   index=[f"{j + 1}: {l}" for j, l in enumerate(labels)],
                   columns=[f"→{j + 1}" for j in range(k)])
display(tab.style.format("{:.1f}", na_rep="·").background_gradient(cmap="Blues_r", vmin=0, vmax=80)
        .set_caption("C-alpha distance (Å) between the tokens; · = no distance"))
""")

code(r"""
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ca = s["ca"]
ok = ~np.isnan(ca).any(1)
pts = [(j, codons[j]) for j in range(k) if located[j]]
norm = Normalize(0, 80)
cmap = plt.get_cmap("viridis_r")
fig = plt.figure(figsize=(7.6, 6.4), constrained_layout=True)
ax = fig.add_subplot(projection="3d")
ax.plot(*ca[ok].T, color="#c9c8c1", lw=0.8)
for a in range(len(pts)):
    for b in range(a + 1, len(pts)):
        (ja, ca_a), (jb, ca_b) = pts[a], pts[b]
        if ca_a == ca_b:
            continue
        seg = np.stack([ca[ca_a], ca[ca_b]])
        ax.plot(*seg.T, color=cmap(norm(np.linalg.norm(seg[0] - seg[1]))), lw=1.4, alpha=0.9)
for j, c in pts:
    ax.scatter(*ca[c], s=60, color="#eb6834", edgecolor="white", linewidth=1.0, depthshade=False)
    ax.text(*ca[c], f"  {j + 1}: {labels[j]}", fontsize=8)
ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
ax.set_title(f"{EX_LOCUS} C-alpha trace and isolate {ex_ids[EX]}'s located tokens")
fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, label="token-pair distance (Å)")
plt.show()
""")

code(r"""
panels = []
for name, (ch, fn) in st.ENCODINGS.items():
    blocks = fn(D_ex, M_ex, EX_LOCUS)
    tag = " (current)" if name == st.CURRENT_ENCODING else ""
    if name == "distance + mask":
        panels += [(f"{name}: distance / 10 Å", blocks[0][0], "dist"), (f"{name}: mask", blocks[1][0], "unit")]
    else:
        panels.append((name + tag, blocks[0][0], "dist" if name.startswith("distance") else "unit"))
fig, axes = plt.subplots(3, 3, figsize=(12.5, 12), constrained_layout=True)
short = [f"{j + 1}:{l}" for j, l in enumerate(labels)]
for ax, (title, A, kind) in zip(axes.flat, panels):
    A = A.numpy()
    im = ax.imshow(A, cmap="Blues" if kind == "unit" else "Greens_r",
                   vmin=0, vmax=1 if kind == "unit" else max(float(A.max()), 1e-6))
    ax.set_xticks(range(k), short, rotation=90, fontsize=7)
    ax.set_yticks(range(k), short, fontsize=7)
    ax.set_title(title, fontsize=9)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if (A[i, j] > 0.6 if kind == "unit" else A[i, j] < 0.4 * A.max()) else "#1a1a19")
    fig.colorbar(im, ax=ax, shrink=0.7)
for ax in list(axes.flat)[len(panels):]:
    ax.axis("off")
fig.suptitle(f"{EX_LOCUS}, isolate {ex_ids[EX]}: the channels token i carries (row i), by encoding")
plt.show()
""")

md(r"""
Row *i* of each panel is exactly what token *i* would carry under that encoding.
For the far pairs in this isolate, exp(−d/10) reads close to 0 while exp(−d/30),
linear and percentile still separate them. Contact keeps only "within 8 Å".
""")

md(r"""
## Caveats

* **"locusfusion" here is this branch's `models/locusfusion.py` — the tokenizer
  `newmodels_full` ran, not `locusfusion_fixed`'s.** The fixed tokenizer (exact
  coordinates, symbol ids) lives on the branch `worktree-lf-token-offset-fix`. The
  v1 coordinate mismatch in §3 (katG S315 at 357.7 against 314) is the bug that
  branch fixed, so §1-3 compare v2 against the pre-fix v1, and the trained control
  for v2 is `results/experiments/locusfusion_fixed`, not this notebook's v1.
* v1 and v2 read different protein blocks, each the input it was built for:
  v1 the isolate frame (20 residue channels, left-aligned), v2 the H37Rv codon
  grid (21 channels: 20 residues and STOP, one column per reference codon). Both
  read the same 5-channel nucleotide alphabet (A C T G -).
* **An uncalled base makes no token** in either model. `N` is the all-zero
  column, and under delta encoding that is what a column matching H37Rv looks
  like.
* §1-3 are one drug (`ISONIAZID`, two loci) on the first 4,000 isolates, §5 on
  all of them;
  §4 places all 18 loci from their reference rows and measures the drop on
  STREPTOMYCIN. Token counts per locus differ across the 19-locus runs — *rrs*
  and *rrl* are hypervariable and are where the 16-token cap binds.
* ISONIAZID's two loci do not overlap their promoter windows, so the drop in §4
  leaves every ISONIAZID number in §1 and §5 unchanged.
* The `locusfusion_fixed` run this branch compares against also carries the
  reference on its token, as a symbol id rather than as a property delta, and it
  detects an uncalled base, which v2 does not.
* No model here is trained. This notebook compares what the two tokenizers
  *emit*. Whether the merged token (or `structure`) trains better is a question
  for a training run, not this notebook.
* The residue attached to a token is the isolate's residue at that codon. Where
  the codon did not change, the delta-encoded block is zero and the REFERENCE
  residue fills the slot instead, so "unchanged" and "no residue here" are no
  longer the same token; the property delta is what separates a synonymous
  change from a substitution.
* **Which residues changed is read off the protein block, never off the
  properties.** STOP and a deleted codon have all-zero properties legitimately.
  In a `dna+biophysical` cell, with no protein block loaded, those two are not
  separable from "unchanged" — the one cell where a stop gain reads as a bare
  base.
* Placement needs each block's reference row to be an exact H37Rv substring.
  It is for all 18 loci here; a block that is not falls back to the old
  convention — promoter counted back from the window's end, nothing dropped —
  and `LocusFusionV2Net` warns naming the locus.
* `coord` for a promoter column is its genome distance from the start codon in
  base pairs / 3. That is a codon *unit*, not a codon — there is no reading frame
  upstream, which is why promoter tokens carry no phase.
* **§6 is tentative.** The encoding (the model's current exp(−d/10) included),
  its length scales and AlphaFold as the source of structures are all
  placeholders for discussion. The AlphaFold models are single chains with no
  ligands, pLDDT is not a model input, and only distances within a locus exist.
  §6.3's example is from ETHAMBUTOL/embA, not ISONIAZID.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
      "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(sys.argv[1] if len(sys.argv) > 1 else
           Path(__file__).resolve().parent.parent / "notebooks" / "tokenization_v2.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
