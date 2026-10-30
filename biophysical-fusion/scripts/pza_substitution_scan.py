"""
Single-substitution scan of pncA through the per-drug LocusFusion v3
pyrazinamide checkpoints (inference only, saved weights, no attribution).

Every possible single-base substitution in the pncA CDS (186 codons x 3
positions x 3 alternative bases, the terminal stop codon excluded) is put on an
H37Rv background -- an isolate identical to the reference everywhere else -- and
scored by each per-drug modality-set checkpoint (sd_*_unk). Each substitution
is classed synonymous / missense / nonsense on the H37Rv codon grid, and tagged
with how many training isolates (train + val roles of that checkpoint's split)
carry that exact base change.

Inputs are built with the loader's own encoders (delta one-hot DNA,
reference-frame residues with delta-zeroed protein / biophysical columns) from
the edited aligned sequence. Before scoring, the encoder is checked against the
loaded arrays for real isolates, and the two presentation isolates (A25A, S66*)
must reproduce their stored predictions.

Writes results/analysis/pza_substitution_scan/scan.csv (+ checks.json).

    python scripts/pza_substitution_scan.py        # ~10 min, ~4 GB, CPU
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict_locusfusion_v3 as plv  # noqa: E402  (frozen v3 snapshot code + data env)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from Bio import SeqIO  # noqa: E402
from Bio.Seq import Seq  # noqa: E402

from datasets import biochem, cds  # noqa: E402
from datasets.sequences import delta_one_hot_nt, delta_zero_columns, reference_row  # noqa: E402

DRUG, GENE = "PYRAZINAMIDE", "pncA"
MODSETS = ["dna", "dna_protein", "dna_biophysical", "dna_regulatory", "all_modalities"]
EXAMPLES = {"SAMEA1118274": "A25A", "SAMN05915262": "S66*"}
OUT = plv.PROJECT_DIR / "results" / "analysis" / "pza_substitution_scan"
GD = str(plv.DATA / "genomic_data" / "aligned")


class Encoder:
    """Aligned pncA row -> the four per-drug blocks, as datasets.load_dataset builds them."""

    def __init__(self, shapes):
        self.ref = reference_row(GD, GENE)
        self.win = cds.cds_columns(GD, GENE)
        self.ref_aa = biochem.translate_reference_frame(self.ref, self.ref, self.win)
        self.shapes = shapes

    def __call__(self, seq):
        p = biochem.translate_reference_frame(seq, self.ref, self.win)
        w = self.shapes[f"protein:{GENE}"][1]
        return {
            f"dna:{GENE}": delta_one_hot_nt(seq, self.ref, with_unknown=True).T,
            f"protein:{GENE}": delta_zero_columns(
                biochem.one_hot_aa(p, pad_to=w, with_stop=True, with_unknown=True), p, self.ref_aa),
            f"biophysical:{GENE}": delta_zero_columns(biochem.biophysical_matrix(p, pad_to=w), p, self.ref_aa),
            # a CDS substitution never reaches the promoter-only columns of this block
            f"regulatory:{GENE}": np.zeros(self.shapes[f"regulatory:{GENE}"], dtype=np.float32),
        }


def substitutions(enc):
    """Every single-base CDS substitution: (column, codon, position, ref, alt, ref_aa, alt_aa, kind, seq)."""
    ref = enc.ref.upper()
    cols = [c for c in range(*enc.win) if ref[c] != "-"]
    out = []
    for k in range(len(cols) // 3 - 1):                      # last codon is the terminal stop
        t = cols[3 * k:3 * k + 3]
        rc = "".join(ref[c] for c in t)
        ra = str(Seq(rc).translate())
        for j, c in enumerate(t):
            for b in "ACGT":
                if b == ref[c]:
                    continue
                ac = rc[:j] + b + rc[j + 1:]
                aa = str(Seq(ac).translate())
                kind = "synonymous" if aa == ra else "nonsense" if aa == "*" else "missense"
                out.append(dict(column=c, codon=k + 1, codon_pos=j + 1, ref_base=ref[c], alt_base=b,
                                ref_codon=rc, alt_codon=ac, ref_aa=ra, alt_aa=aa, kind=kind,
                                change=f"{ra}{k + 1}{aa}", seq=ref[:c] + b + ref[c + 1:]))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models, cfgs = {}, {}
    for ms in MODSETS:
        cell = f"sd_{ms}_unk__locusfusion_v2"
        stem = plv.stem_of(cell, DRUG)
        models[ms], cfgs[ms] = plv.load_model(f"locusfusion_v3/{cell}", stem, weights_dir=plv.WEIGHTS_ROOT)
    cfg = cfgs["all_modalities"]
    bundle = plv.load_single(cfg)
    ids = list(bundle["ids"])
    shapes = {n: a.shape[1:] for n, a in bundle["blocks"].items()}
    enc = Encoder(shapes)
    seqs = {r.id: str(r.seq) for r in SeqIO.parse(f"{GD}/{GENE}.fasta", "fasta")}

    # 1. the encoder against the loader, on real isolates
    rng = np.random.default_rng(0)
    sample = list(EXAMPLES) + list(rng.choice(ids, 300, replace=False))
    exact = {}
    for iso in sample:
        e, i = enc(seqs[iso]), ids.index(iso)
        exact[iso] = all(np.array_equal(e[n], bundle["blocks"][n][i]) for n in e)
    for iso in EXAMPLES:
        assert exact[iso], f"encoder does not reproduce the loaded arrays for {iso}"
    n_exact = sum(exact.values())
    print(f"encoder: {n_exact}/{len(sample)} sampled isolates reproduce the loaded arrays exactly", flush=True)

    def score(arrays):
        out = {}
        for ms, m in models.items():
            names = [b["name"] for b in cfgs[ms]["model"]["blocks"]]
            out[ms] = 1 - plv.predict(m, [arrays[n] for n in names])[:, 0]
        return out

    # 2. the two presentation isolates reproduce their stored predictions
    stored = {ms: pd.read_csv(plv.OUT / f"sd_{ms}_unk__locusfusion_v2" / f"{DRUG}.csv.gz").set_index("isolate_id")
              for ms in MODSETS}
    ex = score({n: np.stack([enc(seqs[i])[n] for i in EXAMPLES]) for n in shapes})
    for ms in MODSETS:
        for j, iso in enumerate(EXAMPLES):
            assert abs(ex[ms][j] - stored[ms].p_resistant[iso]) < 1e-4, (ms, iso, ex[ms][j])
    print("examples reproduce their stored predictions", flush=True)

    # 3. the scan, on the H37Rv background
    subs = substitutions(enc)
    wt = enc(enc.ref)
    assert all(not a.any() for a in wt.values()), "H37Rv background should encode to all-zero blocks"
    arrays = {n: np.stack([enc(s["seq"])[n] for s in subs]) for n in shapes}
    p = score(arrays)
    p_wt = score({n: a[None] for n, a in wt.items()})

    # training carriers of each exact base change (train + val of this split)
    role = stored["dna"].role.reindex(ids).values
    y = bundle["y"]
    dna = bundle["blocks"][f"dna:{GENE}"]
    tr = role != "test"
    df = pd.DataFrame([{k: v for k, v in s.items() if k != "seq"} for s in subs])
    ch = {b: i for i, b in enumerate("ACTG")}                 # NT_CHANNELS order: A C T G - N
    carriers = np.array([dna[tr, ch[s["alt_base"]], s["column"]] > 0 for s in subs])     # (n_subs, n_train)
    df["train_carriers"] = carriers.sum(1)
    df["train_R"] = (carriers & (y[tr] == 0)[None]).sum(1)
    for ms in MODSETS:
        df[f"p_{ms}"] = p[ms]
    df.to_csv(OUT / "scan.csv", index=False)
    checks = {"encoder_exact": f"{n_exact}/{len(sample)}",
              "encoder_mismatch_isolates": [i for i, v in exact.items() if not v],
              "p_resistant_H37Rv_background": {ms: float(v[0]) for ms, v in p_wt.items()},
              "n_substitutions": len(df), "by_kind": df.kind.value_counts().to_dict()}
    (OUT / "checks.json").write_text(json.dumps(checks, indent=1))
    print(json.dumps(checks, indent=1))
    print(df.groupby(["kind", df.train_carriers > 0])[[f"p_{m}" for m in MODSETS]].mean().round(3))


if __name__ == "__main__":
    main()
