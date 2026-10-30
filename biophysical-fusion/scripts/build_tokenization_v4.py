"""Builds notebooks/locusfusion_v4/tokenization_v4.ipynb -- what v4's tokenizer changes from
v3's, on real isolates, with both tokenizers running side by side.

    python scripts/build_tokenization_v4.py      # write the notebook
    jupyter nbconvert --to notebook --execute --inplace notebooks/locusfusion_v4/tokenization_v4.ipynb

v3 here is `LocusFusionV2Net(unknown_symbols=True)`, the paper's model.

The notebook TOKENIZES and nothing else: it runs the real tokenizer, the real
distance table and the real augmenter to build its tables and figures, and it
never trains, fine-tunes or scores a model. Every number about a run is read
out of that run's result JSON on disk (SS7), so a cell that has not been
submitted shows as missing rather than being computed here.

Needs the real data (ABR_* env vars / bigtb_ref.REAL_*). Three loci and 3,000
isolates: v3's loader holds every modality as (N, C, L) one-hot, so the five
loci of `locusfusion_v4/tests/test_parity.py` peak over 13 GB and this peaks at
~6.
"""
import json
from pathlib import Path

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": t.strip("\n").splitlines(keepends=True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})


md(r"""
# v4 vs v3: what changed in the token

Same architecture, same data, same splits. Four changes:

1. **The 23-channel residue one-hot is gone**, replaced by a 4-channel
   consequence flag (synonymous / missense / nonsense / other). §3
2. **A binding-site proximity channel per drug**, `exp(-d / 10 A)`. §4
3. **Tokens are built once, offline**, not rebuilt inside `forward` per batch. §5
4. **Synonymous augmentation** at training time, with a consistency term. §6

Everything else is v3's, by construction rather than re-implementation: the
column geometry is v3's own `_frame_maps`, and the base and property values
follow v3's rules. §2 runs both tokenizers on the same isolates and diffs them.

This notebook only tokenizes. It trains nothing and scores nothing; the run
numbers in §7 are read from the result JSONs on disk.
""")

code(r"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from IPython.display import display

PROJECT = next(p for p in [Path.cwd(), *Path.cwd().parents,
                           Path("/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion")]
               if (p / "training" / "checkpoint.py").exists())
sys.path.insert(0, str(PROJECT))

from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR
from datasets import load_dataset
from locusfusion_v4 import tokenizer as tk
from locusfusion_v4.augment import SynonymousAugmenter
from locusfusion_v4.binding_sites.build_distances import load_distances
from locusfusion_v4.layout import (C_BASE, CONSEQUENCES, DIST_LENGTH, NT_SYMBOLS,
                                   SLOTS, channel_names)
from locusfusion_v4.model import LocusFusionV4Net, proximity_table
from models.locusfusion_v2 import LocusFusionV2Net

pd.set_option("display.width", 220, "display.max_columns", 60)

LOCI = ["pncA", "fabG1", "inhA"]      # tests/test_parity.py adds katG and rrs
DRUGS = ["PYRAZINAMIDE", "ISONIAZID"]
N_CHECK = 3000
REG = tk.regulatory_for(DRUGS, LOCI)

# v3: the loader's blocks, and the net only as a tokenizer (_locus_tokens)
data = load_dataset("ISONIAZID", ["dna", "protein", "biophysical", "regulatory"],
                    REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV,
                    regulatory_dir=REAL_REGULATORY_DIR, loci=LOCI, regulatory_loci=REG,
                    per_modality_branch=False, delta=True, protein_frame="reference",
                    unknown_symbols=True, verbose=False)
v3 = LocusFusionV2Net.from_blocks(data.blocks, unknown_symbols=True, n_drugs=1)

# v4: the cached-token builder
ts = tk.build_tokens(DRUGS, REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR,
                     loci=LOCI, regulatory=REG, verbose=False)
assert ts.isolate_ids == data.isolate_ids

# the slice test_parity checks: the busiest 1,500 isolates and 1,500 at random
rng = np.random.default_rng(0)
rows = np.unique(np.concatenate([np.argsort(-ts.n_occ.sum(1))[:N_CHECK // 2],
                                 rng.choice(ts.n, N_CHECK // 2, replace=False)]))
xs = [torch.from_numpy(a[rows]).float() for a in data.arrays()]
with torch.no_grad():
    feat3, coord3, valid3, stats3 = v3._locus_tokens(xs)
n_blocks3, bytes3 = len(data.blocks), sum(b.array.nbytes for b in data.blocks)
order = [v3.loci.index(l) for l in ts.loci]
feat3, coord3 = feat3[:, order].numpy(), coord3[:, order].numpy()
valid3, stats3 = valid3[:, order].numpy(), stats3[:, order].numpy()
lay3 = v3.layout
del xs, data
gc.collect()

print(f"{len(ts.isolate_ids):,} isolates, {len(rows):,} compared, loci {ts.loci}")
print(f"v3 token {lay3['c_tok']} channels   v4 token {C_BASE} + one per drug")
""")

md(r"""
## 1. The two layouts

v3's width is 45, not the 41 in the paper's table: the paper's model is
`--lf-unknown-symbols`, which widens `dna` and `regulatory` to 6 for an uncalled
base and `protein` to 23 for a deleted or unreadable codon.
""")

code(r"""
def blocks3(lay):
    s = lay["slots"]
    out = [("protein / consequence", *s["protein"]), ("dna", *s["dna"]),
           ("regulatory", *s["regulatory"]), ("bio", *s["biophysical"]),
           ("d_bio", *lay["d_bio"]), ("phase", lay["f_phase"], lay["f_phase"] + 3),
           ("is_null", lay["f_is_null"], lay["f_is_null"] + 1)]
    return [(n, b - a) for n, a, b in out]


def blocks4(n_drugs):
    out = [(n, *SLOTS[n]) for n in ("dna", "regulatory", "bio", "d_bio",
                                    "consequence", "phase", "is_null")]
    out = [("protein / consequence" if n == "consequence" else n, b - a) for n, a, b in out]
    return out + [("site_proximity", n_drugs)]


cmp = pd.DataFrame({"v3": dict(blocks3(lay3)), "v4": dict(blocks4(1))})
cmp = cmp.reindex(["dna", "regulatory", "bio", "d_bio", "protein / consequence",
                   "phase", "is_null", "site_proximity"]).fillna(0).astype(int)
cmp["change"] = cmp["v4"] - cmp["v3"]
display(cmp)
print(f"v3 {lay3['c_tok']}  ->  v4 {C_BASE + 1} single-drug, {C_BASE + 11} multi-drug")

ORDER = ["dna", "regulatory", "protein / consequence", "bio", "d_bio", "phase",
         "is_null", "site_proximity"]
fig, ax = plt.subplots(figsize=(10, 2.0), constrained_layout=True)
colors = dict(zip(ORDER, plt.get_cmap("tab10").colors))
for y, (name, bl) in enumerate([("v4, single-drug", blocks4(1)), ("v3", blocks3(lay3))]):
    x = 0
    for n, w in sorted(bl, key=lambda t: ORDER.index(t[0])):
        ax.barh(y, w, left=x, height=0.6, color=colors[n], edgecolor="white")
        if w >= 3:
            ax.text(x + w / 2, y, f"{n.split(' /')[0]}\n{w}", ha="center", va="center",
                    fontsize=7.5)
        x += w
ax.set_yticks([0, 1], ["v4, single-drug", "v3"])
ax.set_xlabel("channel")
ax.set_title("Token channel layout, v3 and v4")
plt.show()
""")

md(r"""
## 2. The shared channels are identical

`dna`, `regulatory`, `bio`, `d_bio`, `phase`, the coordinate, which slots are
filled, and the `[NULL]` statistics -- over 3,000 isolates and every filled
slot of `pncA fabG1 inhA` -- a coding locus, the promoter/record de-duplication
case, and a promoter window. `tests/test_parity.py` is the same comparison over
`pncA katG rrs fabG1 inhA`, including the rRNA locus, and asserts rather than
prints; it needs ~13 GB, so it runs outside this notebook.

Any v4-vs-v3 difference in results therefore comes from the four changes, not
from a re-tokenization.
""")

code(r"""
v = ts.valid[rows]
f4, f3 = ts.feat[rows][v], feat3[:, :, 1:][v]
pairs = {"dna": lay3["slots"]["dna"], "regulatory": lay3["slots"]["regulatory"],
         "bio": lay3["slots"]["biophysical"], "d_bio": lay3["d_bio"],
         "phase": (lay3["f_phase"], lay3["f_phase"] + 3)}

out = []
for name, (a, b) in pairs.items():
    lo, hi = SLOTS[name]
    d = np.abs(f4[:, lo:hi] - f3[:, a:b])
    out.append({"compared": name, "v3 channels": f"{a}:{b}", "v4 channels": f"{lo}:{hi}",
                "max |v4 - v3|": d.max(), "tokens differing": int((d > 1e-5).any(1).sum())})


def row(name, a, b, tol=1e-5):
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    return {"compared": name, "v3 channels": "-", "v4 channels": "-",
            "max |v4 - v3|": d.max(), "tokens differing": int((d > tol).sum())}


out += [row("coord", ts.coord[rows][v], coord3[:, :, 1:][v], 1e-4),
        row("valid (filled slots)", ts.valid[rows], valid3[:, :, 1:]),
        row("[NULL] log1p(n_occ)", np.log1p(ts.n_occ[rows]), stats3[..., 0]),
        row("[NULL] n_occ / kept", ts.n_occ[rows] / np.maximum(ts.kept, 1), stats3[..., 1])]
display(pd.DataFrame(out).set_index("compared"))
print(f"{int(v.sum()):,} filled slots over {len(rows):,} isolates")
""")

md(r"""
## 3. 23 residue channels out, 4 consequence channels in

What v3's one-hot said and v4's flag does not: which residue. What v4's flag
says and v3's one-hot did not: whether the codon changed at all.

In v3 a synonymous token is a residue one-hot with `d_bio = 0`, so recognising
"nothing happened to the protein" is a conjunction across 23 channels the model
has to learn. In v4 it is one channel. Below, v3's residue against v4's
consequence on the same tokens.
""")

code(r"""
RES3 = list("ACDEFGHIKLMNPQRSTVWY") + ["*", "-", "X"]
p0, p1 = lay3["slots"]["protein"]
oh = f3[:, p0:p1]
res = np.where(oh.sum(1) > 0, oh.argmax(1), -1)
cons = f4[:, slice(*SLOTS["consequence"])]
cl = np.where(cons.sum(1) > 0, cons.argmax(1), -1)

ct = pd.crosstab(pd.Series([RES3[r] if r >= 0 else "(none)" for r in res], name="v3 residue"),
                 pd.Series([CONSEQUENCES[c] if c >= 0 else "(none)" for c in cl],
                           name="v4 consequence"))
display(ct.reindex(columns=[c for c in list(CONSEQUENCES) + ["(none)"] if c in ct.columns]))

n_syn = int((cl == 0).sum())
print(f"{n_syn:,} of {len(cl):,} filled slots ({100 * n_syn / len(cl):.1f}%) are synonymous")
print(f"identity cost v3 {p1 - p0} channels; the flag costs v4 "
      f"{SLOTS['consequence'][1] - SLOTS['consequence'][0]}")
""")

md(r"""
Reading the cross-tab: v3's STOP channel becomes `nonsense`; its deleted (`-`)
and unreadable (`X`) residues become `other`; and the tokens v3 left with an
all-zero protein slot split in two -- an in-CDS insertion is `other`, a promoter
or non-coding token keeps no consequence at all.

Two distinctions the one-hot carried and the flag does not: **which residue**
(`bio` cannot recover it -- leucine and isoleucine have the same three
properties), and **STOP vs. deleted vs. unreadable**, which are one `other`
class with `bio = 0`. Whether either mattered is what the `nosyn` arm tests.
""")

code(r"""
li = {l: i for i, l in enumerate(ts.loci)}
CONSC = {c: SLOTS["consequence"][0] + i for i, c in enumerate(CONSEQUENCES)}

seqs = tk.load_sequence_df(REAL_GENOTYPE_DIR, LOCI)[0].reindex(ts.isolate_ids).fillna("")
geoms = {}
for locus in LOCI:
    ref = tk.reference_row(REAL_GENOTYPE_DIR, locus)
    rl = len(tk.reference_row(REAL_REGULATORY_DIR, locus)) if locus in REG else 0
    geoms[locus] = tk.locus_geometry(locus, len(ref), ref, REAL_GENOTYPE_DIR,
                                     reg_len=rl, regulatory_dir=REAL_REGULATORY_DIR)


def mutation(n, locus, k):
    # the isolate's own codon at H37Rv codon k, read back off the alignment
    g, s = geoms[locus], seqs.loc[ts.isolate_ids[n], locus].upper()
    cod = "".join(s[c] if c < len(s) else "-" for c in g.codon_cols[k])
    alt = "-" if "-" in cod else tk.CODON_TABLE.get(cod, "X")
    return f"{tk.RESIDUES[g.ref_res[k]]}{k + 1}{alt}", f"{g.ref_codons[k]}>{cod}"


def find(locus, consequence, codon=None, alt=None):
    L = li[locus]
    m = ts.valid[:, L] & (ts.feat[:, L, :, CONSC[consequence]] == 1)
    if codon is not None:
        m &= ts.codon[:, L] == codon
    for n, s in np.argwhere(m):
        if alt is None or mutation(n, locus, int(ts.codon[n, L, s]))[0].endswith(alt):
            return int(n), L, int(s)
    raise LookupError(f"no {consequence} token in {locus}")


picks = {"inhA S94A": find("inhA", "missense", codon=93, alt="A"),
         "pncA synonymous": find("pncA", "synonymous"),
         "pncA nonsense": find("pncA", "nonsense"),
         "inhA uncalled base": find("inhA", "other")}

tok4 = pd.DataFrame({k: ts.feat[n, L, s] for k, (n, L, s) in picks.items()},
                    index=channel_names())
display(tok4.loc[(tok4 != 0).any(axis=1)].round(3))

for lab, (n, L, s) in picks.items():
    k = int(ts.codon[n, L, s])
    mut = "  ".join(mutation(n, ts.loci[L], k)) if k >= 0 and ts.locus_kind[L] == "protein" else ""
    print(f"{lab:22s} {ts.isolate_ids[n]:14s} {ts.loci[L]:6s} codon {k:5d} "
          f"coord {ts.coord[n, L, s]:8.2f} site_pos {ts.site_pos[n, L, s]:5d}  {mut}")
""")

md(r"""
## 4. Binding-site proximity

`site_pos` indexes a per-(drug, locus) distance table -- a residue index on a
protein locus, a nucleotide offset on `rrs` / `rrl`, `-1` otherwise. The model
writes `exp(-d / 10 A)` per drug, and `-1` for "no distance": a promoter token, a
locus with no site for that drug, a position the structure does not resolve.

Protein distances are the closest heavy atom to a site residue on the AlphaFold
monomer, residues from the literature review with the corrections in
`notebooks/locusfusion_v4/binding_site_distance.ipynb` (gyrB shifted -39, eis E32/F393
dropped). For `rrs` / `rrl` / `rpsL` under STR / KAN / AMK / CAP it is the
distance to the drug's own atoms in a drug-bound ribosome (4DR3, 8EV7, 8EV6,
5V93).
""")

code(r"""
dist = load_distances()
rows_d = []
for (d, l) in sorted(dist):
    a = dist[(d, l)]
    fin = a[np.isfinite(a)]
    rows_d.append({"drug": d, "locus": l, "positions": len(a), "resolved": len(fin),
                   "min A": round(float(fin.min()), 2) if len(fin) else None,
                   "median A": round(float(np.median(fin)), 1) if len(fin) else None})
dtab = pd.DataFrame(rows_d).pivot_table(index="locus", columns="drug", values="min A")
print("closest site distance per (locus, drug), A -- blank = no site for that drug")
display(dtab.round(2))
print(f"{len(rows_d)} (drug, locus) tables in distances.npz; "
      f"{sum(r['positions'] - r['resolved'] for r in rows_d)} positions unresolved")

prox = proximity_table(DRUGS, ts.loci, dist)
n, L, s = picks["inhA S94A"]
sp = int(ts.site_pos[n, L, s])
print(f"the inhA S94A token (site_pos {sp}), read by each drug's channel:")
for i, drug in enumerate(DRUGS):
    a = dist.get((drug, ts.loci[L]))
    print(f"  {drug:14s} {prox[i, L, sp]:6.2f}   "
          f"{'no site on this locus' if a is None else f'{a[sp]:.2f} A'}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)
dd = np.linspace(0, 50, 400)
axes[0].plot(dd, np.exp(-dd / DIST_LENGTH))
for x0 in (0, 5, 10, 30):
    axes[0].plot([x0], [np.exp(-x0 / DIST_LENGTH)], "o", ms=4)
    axes[0].annotate(f"{x0} A, {np.exp(-x0 / DIST_LENGTH):.2f}",
                     (x0, np.exp(-x0 / DIST_LENGTH)), textcoords="offset points",
                     xytext=(6, 6), fontsize=8)
axes[0].set_xlabel("distance to the site (A)")
axes[0].set_ylabel("channel value")
axes[0].set_title(f"Proximity channel, exp(-d / {DIST_LENGTH:.0f} A)")

a = dist[("ISONIAZID", "inhA")]
axes[1].plot(np.arange(1, len(a) + 1), np.exp(-a / DIST_LENGTH), lw=0.8)
axes[1].set_xlabel("inhA residue")
axes[1].set_ylabel("channel value")
axes[1].set_title("inhA proximity to isoniazid, by residue")
plt.show()
""")

md(r"""
## 5. Tokens are built once, not per batch

v3 holds `(N, C, L)` one-hot blocks per modality and rebuilds its tokens inside
`forward`; the 19-locus all-modality cells needed a 128 GB job. v4's `TokenSet`
is the tokens themselves, cached, and training reads nothing else.
""")

code(r"""
tsb = sum(getattr(ts, k).nbytes for k in ts.ARRAYS)
cache = Path("/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/token_cache_v4")
cached = sorted(cache.glob("tokens_PYRAZINAMIDE_*.npz")) if cache.exists() else []

print(f"v3 input blocks, {len(LOCI)} loci  {bytes3 / 1e9:6.2f} GB  "
      f"({n_blocks3} blocks, re-tokenized every batch)")
print(f"v4 TokenSet,     {len(LOCI)} loci  {tsb / 1e9:6.2f} GB  (tokenized once, "
      f"{ts.config['seconds']} s)")
for p in cached:
    print(f"v4 cache on disk, 19 loci     {p.stat().st_size / 1e9:6.2f} GB  {p.name}")

v4 = LocusFusionV4Net(ts.loci, ts.kept, ["PYRAZINAMIDE"], prox=prox[:1],
                      dist_drugs=["PYRAZINAMIDE"], max_variants=ts.feat.shape[2])
print(f"\nparameters  v3 {sum(p.numel() for p in v3.parameters()):,}"
      f"   v4 {sum(p.numel() for p in v4.parameters()):,}"
      f"   (the input projection: {lay3['c_tok']} vs {C_BASE + 1} channels into d_model)")

B = 8
batch = {k: torch.as_tensor(getattr(ts, k)[:B]) for k in
         ("feat", "coord", "valid", "codon", "site_pos", "n_occ")}
batch["valid"] = batch["valid"].bool()
with torch.no_grad():
    x, _, _ = v4.tokens(batch["feat"], batch["coord"], batch["valid"], batch["site_pos"])
print(f"model input {tuple(x.shape)} = (batch, locus, [NULL] + {ts.feat.shape[2]} slots, "
      f"{C_BASE} + 1)")
""")

md(r"""
## 6. Synonymous augmentation

The reason for v4. In the pncA scan
(`scripts/pza_substitution_scan.py`, 2026-10-22) v3 put an H37Rv background at
P(R) 0.14, and a single synonymous change moved the mean to 0.52-0.62; across
modality sets 52-65% of synonymous changes were called resistant.

`synonymous_candidates` enumerates every single-base change that keeps H37Rv's
residue, excluding the start and stop codons, bases another loaded locus also
carries (fabG1 L203L is why), and `augment_exclusions.json`. A copy draws a
coding locus uniformly and then a candidate within it, so gene length does not
decide how often a locus is augmented.

```
loss = BCE(batch) + BCE(copies, same labels)
     + lambda * mean |logit(copy) - stopgrad(logit(original))|
```
""")

code(r"""
cand = ts.cand
print(f"{len(cand['locus']):,} candidates over these {len(LOCI)} loci; "
      f"23,978 over the 19 (v4_tokens job 66215735)")
display(pd.DataFrame({
    "candidates": pd.Series([ts.loci[i] for i in cand["locus"]]).value_counts(),
    "codons": pd.Series({l: len(geoms[l].codon_cols) for l in ts.loci})}).dropna().astype(int))

aug = SynonymousAugmenter(cand, device="cpu", locus_weight="uniform")
out, inserted = aug(batch, exact_k=2, gen=torch.Generator().manual_seed(0))
b = int((inserted == 2).nonzero()[0])
print(f"\nisolate {ts.isolate_ids[b]}, k = 2:")
for L in [int(l) for l in (out["n_occ"][b] != batch["n_occ"][b]).nonzero()]:
    s = int(batch["valid"][b, L].sum())
    f = out["feat"][b, L, s].numpy()
    print(f"  {ts.loci[L]:6s} slot {s}: codon {out['codon'][b, L, s].item():4d} "
          f"coord {out['coord'][b, L, s].item():8.2f} "
          f"base {NT_SYMBOLS[int(f[slice(*SLOTS['dna'])].argmax())]} "
          f"d_bio {f[slice(*SLOTS['d_bio'])].round(2).tolist()} "
          f"{CONSEQUENCES[int(f[slice(*SLOTS['consequence'])].argmax())]}, "
          f"n_occ {batch['n_occ'][b, L].item()} -> {out['n_occ'][b, L].item()}")
print("\nthe copy is a well-formed isolate: real coordinate, phase and site distance, "
      "n_occ incremented")
""")

md(r"""
## 7. The arms, and what has finished

Every arm is 11 drugs on the per-drug lineage splits, protocol identical to the
v3 cells (300 epochs, patience 30, min 50, Adam at exp(-9), early stop on
validation AUC, best fold scored once on TEST). Each isolates one change.

| arm | what it isolates |
|---|---|
| `noaug` | the token change alone, against the v3 lineage cell |
| `aug` | the augmentation, against `noaug` |
| `nodist` | the proximity channels, removed from `aug` |
| `nosyn` | whether the flag is needed at all: synonymous tokens dropped before the cap |
| `md_*` | the same four as one multi-drug model: 11 drug queries, 11 proximity channels |

Both probes run after every arm, inside the training job: test isolates with 1
and 2 synthetic synonymous changes (mean |Dlogit|, mean DP(R)), and every
candidate alone on an H37Rv background -- the pncA scan over all 17 coding loci.

The tables below are read from the result JSONs; `notebooks/locusfusion_v4/locusfusion_v4_results.ipynb`
is the full results view, with a coverage table of what has run. `canary` and
`tune` are plumbing and hyperparameter checks, not results, and are excluded.
""")

code(r"""
V4 = PROJECT / "results" / "experiments" / "locusfusion_v4"
V3_CELL = (PROJECT / "results" / "experiments" / "lineage_split"
           / "sd19_all_modalities_unk__locusfusion_v2")
ARMS = ["sd19_noaug", "sd19_aug", "sd19_nodist", "sd19_nosyn",
        "md_noaug", "md_aug", "md_nodist", "md_nosyn"]

v3_auc = {}
for p in sorted(V3_CELL.glob("*.json")):
    if p.name == "weights_location.json":
        continue
    j = json.loads(p.read_text())
    v3_auc[j["drug"]] = {"v3 CV AUC": j["cv_auc_mean"], "v3 TEST AUC": j["test"]["auc"]}

found, missing = [], []
for arm in ARMS:
    d = V4 / arm
    js = sorted(d.glob("*.json")) if d.is_dir() else []
    if not js:
        missing.append(arm)
    for p in js:
        j = json.loads(p.read_text())
        drug = j.get("drug", "MULTIDRUG")
        pk = (j.get("probe") or {}).get("test_plus_k", {}).get("1", {})
        dl = np.mean([v["mean_abs_dlogit"] for k, v in pk.items() if isinstance(v, dict)]) \
            if pk else np.nan
        found.append({"arm": arm, "drug": drug,
                      "v4 CV AUC": j.get("cv_auc_mean"),
                      "v4 TEST AUC": (j.get("test") or {}).get("auc", j.get("test_macro_auc")),
                      "probe k=1 mean |dlogit|": dl,
                      **v3_auc.get(drug, {})})

if found:
    f = pd.DataFrame(found)
    f["v4 - v3 TEST"] = f["v4 TEST AUC"] - f["v3 TEST AUC"]
    display(f.set_index(["arm", "drug"]).round(4))
else:
    print("no arm has written a result yet")
print("arms with no results on disk:", ", ".join(missing) if missing else "none")

sub = []
for mp in sorted((V4 / "logs").glob("manifest_*.json")):
    m = json.loads(mp.read_text())
    for j in m.get("jobs", []):
        sub.append({"stamp": m["stamp"], "arms": ",".join(m["arms"]), "job": j["name"],
                    "job id": j.get("job_id"), "kind": j.get("kind"),
                    "dependency": m.get("dependency") or ""})
print("\nsubmitted (from the sbatch manifests):")
display(pd.DataFrame(sub).set_index("job id"))
print(f"snapshot the jobs run from: {m['snapshot']}")
""")

md(r"""
## Caveats

- **The multi-drug split is its own split** -- the per-drug lineage splits
  disagree about where an isolate sits. Same recipe, Beijing as fold 4, balanced
  on each drug's R count; TEST is 27% because sublineage groups go in whole;
  LEVO has 66 R in total.
- **Two entries in the site table are not drug sites**: gid's SAM pocket and
  ethR's inhibitor tunnel. The review they come from reads as partly
  machine-written.
- **Distances are within one chain**, so rpoC has none to the rifampicin pocket
  and gyrA none to gyrB's half of the fluoroquinolone site.
- **Augmentation assumes synonymous changes are neutral** outside the exclusion
  list.
- **Data are the v3 runs' data** -- the insertion-extraction bug at
  eis/embB/ethA/rrs and the TSS-anchored promoter windows are both still there,
  which is what keeps v4 vs v3 matched.
- **Three loci here, 19 in the runs.** The parity diff, the candidate counts and
  the byte counts above are for `pncA fabG1 inhA`; the 19-locus numbers quoted
  in SS5 and SS6 come from the `v4_tokens` job (66215735).
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                                                  "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).resolve().parents[1] / "notebooks" / "locusfusion_v4" / "tokenization_v4.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(cells)} cells)")
