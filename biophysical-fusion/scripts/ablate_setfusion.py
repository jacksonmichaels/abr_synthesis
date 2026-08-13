#!/usr/bin/env python3
"""
Is a SetFusionNet's locus keying actually doing anything?

A separation in an embedding plot proves nothing: adding a distinct learned
vector per locus separates the loci BY CONSTRUCTION and cannot fail to. The only
evidence that the mechanism matters is that breaking it costs held-out AUC. This
script breaks it, four ways, and reports the cost with a paired bootstrap.

Run it against any setfusion checkpoint::

    python scripts/ablate_setfusion.py --run full_run_v2/all_modalities__setfusion \
        --stem MOXIFLOXACIN__dna+protein+biophysical+regulatory

What it measures, and why each one is here:

1. **Embedding ablations** — zero the locus embedding, zero the modality
   embedding, zero both, and (the controlled one) SWAP two loci's embeddings.
   The swap is the cleanest of the four: it keeps the vectors and their
   statistics and only misassigns them, so it cannot be dismissed as "you moved
   the model off its training distribution".
2. **Input ablations** — permute one locus's feature blocks across isolates,
   destroying that locus's per-isolate signal while preserving its marginal
   statistics. This separates "the model needs to know WHICH locus" from "the
   model needs the locus at all", which the embedding ablations cannot.
3. **Attention** — the mean attention of each drug query over the blocks. Flat
   attention (1/n_blocks) means the query is not selecting, so a keying it could
   select with is inert whatever the embeddings look like.
4. **Linear probe** — logistic regression on `enc` / `pre` / `post`, fit on the
   checkpoint's own train split and scored on its own test isolates. Two things
   fall out: `enc -> pre` is a null result BY ALGEBRA whenever the readout is
   affine (the embedding adds the same vector to every isolate, so it lands in
   the intercept), and comparing the `enc` probe to the model's own head says
   whether the fusion stack adds anything at all.

On full_run_v2/all_modalities__setfusion (MOXIFLOXACIN) every one of these came
back negative: 0.14% of a token varied with the genotype, attention was flat at
0.1250, deleting the locus embedding cost 0.009 AUC [-0.040, +0.021], and the
`enc` probe (0.8054) BEAT the trained head (0.7946). That is the baseline any
proposed fix has to move — see results/experiments/token_signal/.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler             # noqa: E402

from bigtb_ref import (REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV,  # noqa: E402
                       REAL_REGULATORY_DIR)
from datasets import load_dataset                            # noqa: E402
from training.checkpoint import load_model                   # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run name, e.g. full_run_v2/…__setfusion")
    p.add_argument("--stem", required=True, help="{DRUG}__{tag}")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--boot", type=int, default=2000, help="bootstrap resamples (0 to skip)")
    p.add_argument("--no-probe", action="store_true", help="skip the linear probe")
    p.add_argument("--out", default=None, help="write the ablation table here as CSV")
    return p.parse_args()


def load(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.run, args.stem, map_location=dev)
    model.to(dev).eval()
    if not hasattr(model, "locus_emb"):
        raise SystemExit(f"{args.run}/{args.stem} is a {cfg['model']['arch']} model; "
                         "this probe only applies to --arch setfusion")
    drug = cfg["drug"]
    data = load_dataset(drug, cfg["data"]["modalities_used"], REAL_GENOTYPE_DIR,
                        REAL_PHENOTYPE_CSV, regulatory_dir=REAL_REGULATORY_DIR,
                        per_modality_branch=False,
                        delta=bool(cfg["data"].get("delta", False)), verbose=False)
    keep = np.nonzero(data.y != -1)[0]
    y = data.y[keep]
    arrays = [a[keep] for a in data.arrays()]
    ids = [data.isolate_ids[i] for i in keep]
    test = np.array([i in set(cfg["split"]["test_isolate_ids"]) for i in ids])
    return model, cfg, drug, y, arrays, test, dev


def predict(model, arrays, n, batch, dev, shuffle=None, seed=0):
    """Sigmoid predictions. `shuffle` permutes those block indices across
    isolates, which destroys their per-isolate signal but keeps their marginal
    distribution — an input ablation that cannot be confused with a scale change."""
    arrs = arrays
    if shuffle:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        arrs = [a[perm] if i in set(shuffle) else a for i, a in enumerate(arrays)]
    out = []
    with torch.no_grad():
        for s in range(0, n, batch):
            xs = [torch.from_numpy(a[s:s + batch]).float().to(dev) for a in arrs]
            out.append(torch.sigmoid(model(xs)).cpu().numpy().reshape(-1))
    return np.concatenate(out)


def auc_pr(y, p, m):
    """(AUC, AUC-PR) on the masked subset. Labels are 0=R, 1=S; AUC-PR takes
    RESISTANT as the positive class, matching training.multimodal._metrics."""
    return (roc_auc_score(y[m], p[m]), average_precision_score(1 - y[m], 1 - p[m]))


def main():
    args = parse_args()
    model, cfg, drug, y, arrays, test, dev = load(args)
    keys = model.block_keys
    n = len(y)
    print(f"{cfg['model']['arch']}  {drug}  fold {cfg['best_fold']}  "
          f"delta={bool(cfg['data'].get('delta', False))}  "
          f"token_norm={getattr(model, 'token_norm', 'none')}")
    print(f"{n:,} phenotyped, test={test.sum()} "
          f"(R={int((y[test] == 0).sum())}, S={int((y[test] == 1).sum())})\n")

    LE = model.locus_emb.weight.data.clone()
    ME = model.modality_emb.weight.data.clone()
    loci = [l for l in model.locus_vocab if l != "<none>"]

    def restore():
        model.locus_emb.weight.data.copy_(LE)
        model.modality_emb.weight.data.copy_(ME)

    variants = {"baseline": lambda: None,
                "locus_emb := 0": lambda: model.locus_emb.weight.data.zero_(),
                "modality_emb := 0": lambda: model.modality_emb.weight.data.zero_(),
                "both := 0": lambda: (model.locus_emb.weight.data.zero_(),
                                      model.modality_emb.weight.data.zero_())}
    if len(loci) >= 2:
        a, b = model._locus_ix[loci[0]], model._locus_ix[loci[1]]
        variants[f"locus_emb {loci[0]}<->{loci[1]} swapped"] = (
            lambda: model.locus_emb.weight.data.__setitem__([a, b], LE[[b, a]]))

    rows, preds = [], {}
    for name, fn in variants.items():
        restore()
        fn()
        p = preds[name] = predict(model, arrays, n, args.batch, dev)
        at, pt = auc_pr(y, p, test)
        af, pf = auc_pr(y, p, np.ones(n, bool))
        rows.append({"variant": name, "test_AUC": at, "test_AUC_PR": pt,
                     "all_AUC": af, "all_AUC_PR": pf})
    restore()

    # input ablations — "does the model need this locus" vs "which locus is it"
    base = preds["baseline"]
    for locus in loci:
        idx = [i for i, (_m, l) in enumerate(keys) if l == locus]
        p = preds[f"input {locus} destroyed"] = predict(
            model, arrays, n, args.batch, dev, shuffle=idx)
        at, pt = auc_pr(y, p, test)
        af, pf = auc_pr(y, p, np.ones(n, bool))
        rows.append({"variant": f"input {locus} destroyed ({len(idx)} blocks)",
                     "test_AUC": at, "test_AUC_PR": pt, "all_AUC": af, "all_AUC_PR": pf})

    df = pd.DataFrame(rows)
    df["d_test_AUC"] = df.test_AUC - df.test_AUC.iloc[0]
    print(df.round(4).to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\n-> {args.out}")

    if args.boot:
        rng = np.random.default_rng(0)
        yt, bv = y[test], base[test]
        print(f"\npaired bootstrap of dAUC vs baseline "
              f"(test split, {args.boot} resamples):")
        for name, p in preds.items():
            if name == "baseline":
                continue
            pv, d = p[test], []
            for _ in range(args.boot):
                s = rng.integers(0, len(yt), len(yt))
                if len(np.unique(yt[s])) < 2:
                    continue
                d.append(roc_auc_score(yt[s], pv[s]) - roc_auc_score(yt[s], bv[s]))
            d = np.array(d)
            flag = "" if (d < 0).mean() > 0.975 else "   (CI spans 0)"
            print(f"  {name:38s} d={d.mean():+.4f}  95% CI "
                  f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]{flag}")

    # --- attention: is the query selecting at all? ---------------------------
    xs = [torch.from_numpy(a[:args.batch]).float().to(dev) for a in arrays]
    with torch.no_grad():
        _, attn = model(xs, return_attn=True)
    w = attn.mean(0).cpu().numpy()
    print(f"\nmean attention over blocks (uniform would be {1 / len(keys):.4f}):")
    for j in range(w.shape[0]):
        label = (model.drug_names or [drug])[j] if model.drug_names else drug
        top = sorted(zip(keys, w[j]), key=lambda t: -t[1])
        spread = w[j].max() - w[j].min()
        print(f"  {label}: spread {spread:.5f}  " +
              "  ".join(f"{m}:{l}={v:.4f}" for (m, l), v in top[:4]))

    # --- linear probe --------------------------------------------------------
    if args.no_probe:
        return
    ids_t = model._default_ids
    mods = [model.modalities[i] for i in ids_t[:, 0].tolist()]
    enc, pre, post = [], [], []
    with torch.no_grad():
        for s in range(0, n, args.batch):
            xb = [torch.from_numpy(a[s:s + args.batch]).float().to(dev) for a in arrays]
            e = torch.stack([model.encoders[m](x) for m, x in zip(mods, xb)], dim=1)
            t = e if model.tok_norm is None else model.tok_norm(
                e, ids_t[:, 0] * model.n_locus + ids_t[:, 1])
            pr = t + model.modality_emb(ids_t[:, 0]) + model.locus_emb(ids_t[:, 1])
            enc.append(e.cpu().numpy()); pre.append(pr.cpu().numpy())
            post.append(model.fusion(pr).cpu().numpy())
    enc, pre, post = (np.concatenate(v) for v in (enc, pre, post))

    print("\nsignal-to-constant ratio (per-isolate spread / token norm):")
    for tag, tok in (("enc", enc), ("pre", pre), ("post", post)):
        norm = np.linalg.norm(tok, axis=-1).mean()
        spread = np.linalg.norm(tok - tok.mean(0), axis=-1).mean()
        print(f"  {tag:5s} |token|={norm:8.4f}  spread={spread:.5f}  "
              f"ratio={spread / max(norm, 1e-12):.5f}")

    print("\nlinear probe on the model's own representations "
          "(fit on its train split, scored on its test isolates):")
    tr = ~test
    for tag, tok in (("enc  (encoder output)", enc), ("pre  (+embeddings)", pre),
                     ("post (after fusion)", post)):
        X = tok.reshape(n, -1)
        sc = StandardScaler().fit(X[tr])
        Xs = sc.transform(X)
        lr = LogisticRegression(max_iter=3000, C=0.1).fit(Xs[tr], y[tr])
        print(f"  {tag:24s} test AUC {roc_auc_score(y[test], lr.predict_proba(Xs[test])[:, 1]):.4f}")
    print(f"  {'model’s own head':24s} test AUC {roc_auc_score(y[test], base[test]):.4f}")


if __name__ == "__main__":
    main()
