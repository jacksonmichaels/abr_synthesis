"""
LocusFusion v3 SHAP with the biophysical attribution split into its two token
slots: the residue-properties slot (``biophysical``: the isolate's residue's
MW / pI / hydrophobicity) and the change-from-reference slot (``D_BIO``:
isolate minus H37Rv).

Why a separate run
------------------
In a v3 all-modalities model both slots are filled inside the model's forward
(``LocusFusionV2Net._locus_tokens``) from the SAME biophysical input block:
``alt_props = got["biophysical"]`` goes into the residue slot as is, and into
D_BIO as ``alt_props - ref_props``. Input-block SHAP
(scripts/shap_locusfusion_v3.py) therefore credits both to one block.

Method: a path decomposition of the same expected gradients
-----------------------------------------------------------
The biophysical input x reaches the token by two paths. By the chain rule
``dF/dx = g_res + g_dbio`` (gradient through the residue slot, gradient through
D_BIO), so the expected-gradients attribution splits exactly:

    phi_x = E[(x - x') * g_res] + E[(x - x') * g_dbio].

In ONE forward the D_BIO path reads a second leaf tensor ``xd`` that holds the
same values as x (``_locus_tokens`` is re-built from the snapshot's own source
with three textual insertions, asserted to match exactly once; the snapshot is
not edited). Backward then gives ``g_res`` on x and ``g_dbio`` on xd from the
same samples. A second, unmodified forward/backward on the SAME interpolated
samples gives the undecomposed gradient, which is the check:
``phi_res + phi_dbio == phi_full`` (float round-off), and every non-biophysical
block's attribution is identical between the two passes.

Settings are the existing study's: train background of 100 training-split
isolates, ``--n-explain 300 --nsamples 64``, seed 0, the config's best_fold,
POSITIVE = toward RESISTANT, AUC gate (|rescored - reported| <= 5e-3). The data
load, background pick and explained-isolate pick reuse
scripts/shap_locusfusion_v3.py's functions with the same seeded rng, so the
background and explained isolates are the existing run's. Only the expected-
gradients draws (background row and alpha per sample) are new: shap's
GradientExplainer is replaced by the same estimator written out here
(sample a background row and t ~ U(0,1), input t*x + (1-t)*x', average
grad * (x - x')) so both passes share every draw.

Shares, as the existing channel_table: per (block, channel), sum over positions
of |phi|, mean over isolates; a modality's share is its sum over the total.
|a + b| != |a| + |b|, so:
  * primary (``share_*``): total = |phi| of every non-biophysical channel
    + |phi_res| + |phi_dbio|. The six shares sum to 1 (uncalled is a subset of
    dna / protein / regulatory).
  * alternative (``alt_share_*``): the existing denominator, total = non-bio
    + |phi_full| of the biophysical blocks; ``alt_share_biophysical`` is then
    directly comparable to modality_shares.csv, and the residue / delta parts
    over that total sum to more than it by the cancellation between the paths
    (``bio_cancellation`` = (|res| + |delta|) / |full|).

Outputs, under results/analysis/shap_locusfusion_v3/token_split/:
  <DRUG>/<cell>__<bg>__channels_split.csv   per block x channel
  <DRUG>/<cell>__<bg>__meta.json            shares, checks, gate, timing
  modality_shares_split.csv                 (--aggregate)

Run on a GPU node only (see scripts/sbatch/shap_locusfusion_v3_token_split.sh):
    python scripts/shap_locusfusion_v3_token_split.py --drugs LEVOFLOXACIN
    python scripts/shap_locusfusion_v3_token_split.py --aggregate
"""
import argparse
import gc
import importlib.util
import inspect
import json
import sys
import textwrap
import time
import types
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# the existing v3 SHAP script: sets sys.path to the frozen v3 snapshot, the
# ABR_* env defaults, and asserts the imported code is the snapshot's
_spec = importlib.util.spec_from_file_location(
    "shap_locusfusion_v3", PROJECT_DIR / "scripts" / "shap_locusfusion_v3.py")
V3 = importlib.util.module_from_spec(_spec)
sys.modules["shap_locusfusion_v3"] = V3
_spec.loader.exec_module(V3)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import models.locusfusion_v2 as LF2  # noqa: E402
from training.checkpoint import load_model  # noqa: E402
from shap_attribution import resolve_stem  # noqa: E402

assert Path(LF2.__file__).resolve().is_relative_to(V3.SNAPSHOT), LF2.__file__

DEFAULT_OUT = V3.DEFAULT_OUT / "token_split"
EXISTING_SHARES = V3.DEFAULT_OUT / "modality_shares.csv"
NONBIO = ("dna", "protein", "regulatory")


# ---------------------------------------------------------------------------
# the split tokenizer: snapshot source + three insertions
# ---------------------------------------------------------------------------

_INSERT_GD = '''\
            gd = None
            _xd = getattr(self, "_split_xd", None)
            if _xd is not None and "biophysical" in aa:
                _b = aa["biophysical"]
                _here = coding & (codon < _b["length"])
                _kk = codon.clamp(0, _b["length"] - 1)
                _x = _xd[_b["index"]]
                gd = _x.transpose(1, 2).gather(
                    1, _kk.unsqueeze(-1).expand(-1, -1, _x.shape[1])
                ) * _here.unsqueeze(-1).to(dt)
'''
_PATCHES = [
    # 1. the D_BIO path's own copy of the gathered biophysical values, read off xd
    ('            if "protein" in got:\n'
     '                changed = got["protein"].abs().sum(-1, keepdim=True) > 0\n',
     _INSERT_GD +
     '            if "protein" in got:\n'
     '                changed = got["protein"].abs().sum(-1, keepdim=True) > 0\n'),
    # 2. the same reference fill at unchanged-residue substitutions
    ('                    got["biophysical"] = torch.where(fill, ref_props, got["biophysical"])\n',
     '                    got["biophysical"] = torch.where(fill, ref_props, got["biophysical"])\n'
     '                    if gd is not None:\n'
     '                        gd = torch.where(fill, ref_props, gd)\n'),
    # 3. D_BIO = alt - ref reads the xd copy
    ('                alt_props = got["biophysical"]\n',
     '                alt_props = got["biophysical"] if gd is None else gd\n'),
]


def build_split_tokens():
    src = textwrap.dedent(inspect.getsource(LF2.LocusFusionV2Net._locus_tokens))
    src = textwrap.indent(src, "    ")          # patterns are written class-indented
    for old, new in _PATCHES:
        n = src.count(old)
        if n != 1:
            raise RuntimeError(f"snapshot _locus_tokens: patch anchor found {n}x: {old!r}")
        src = src.replace(old, new)
    ns = {}
    exec(compile(textwrap.dedent(src), "<split _locus_tokens>", "exec"), vars(LF2), ns)
    return ns["_locus_tokens"]


SPLIT_TOKENS = build_split_tokens()


# ---------------------------------------------------------------------------
# expected gradients, two passes on shared draws
# ---------------------------------------------------------------------------

def _logit(model, xs):
    return model(xs).reshape(len(xs[0]), -1)[:, 0]


def split_expected_gradients(model, cfg, x_test, bg, nsamples, device, rng, chunk):
    """-> dict of per-block phi arrays (n, C, L), float64, + RESISTANT:
        full    unmodified model (block order)
        split   the split pass, every block's grad on its own input
        dbio    the split pass, bio blocks only: grad on the D_BIO copy
       and the forward-equality check."""
    kinds = [b["modality"] for b in cfg["model"]["blocks"]]
    bio = [i for i, m in enumerate(kinds) if m == "biophysical"]
    bg_t = [torch.from_numpy(a).to(device).float() for a in bg]
    n, nb = len(x_test[0]), len(bg[0])
    out = {k: [np.zeros(a.shape, dtype=np.float64) for a in x_test] for k in ("full", "split")}
    out["dbio"] = {i: np.zeros(x_test[i].shape, dtype=np.float64) for i in bio}
    max_fwd_diff = 0.0
    split_method = types.MethodType(SPLIT_TOKENS, model)
    for j in range(n):
        rind = rng.integers(0, nb, size=nsamples)
        t = rng.uniform(size=nsamples)
        x = [torch.from_numpy(a[j:j + 1]).to(device).float() for a in x_test]
        acc = {k: [torch.zeros(a.shape[1:], dtype=torch.float64, device=device)
                   for a in x_test] for k in ("full", "split")}
        acc_d = {i: torch.zeros(x_test[i].shape[1:], dtype=torch.float64, device=device)
                 for i in bio}
        for s in range(0, nsamples, chunk):
            ri = torch.from_numpy(rind[s:s + chunk]).to(device)
            tt = torch.from_numpy(t[s:s + chunk]).to(device).float()
            ref = [b_[ri] for b_ in bg_t]
            tb = tt.view(-1, 1, 1)
            samp = [(tb * xi + (1 - tb) * r).detach() for xi, r in zip(x, ref)]
            delta = [(xi - r).double() for xi, r in zip(x, ref)]

            # pass A: unmodified model
            model.__dict__.pop("_locus_tokens", None)
            XA = [s_.clone().requires_grad_() for s_ in samp]
            ya = _logit(model, XA)
            ga = torch.autograd.grad(ya.sum(), XA)

            # pass B: D_BIO reads its own copy of each biophysical block
            model._locus_tokens = split_method
            XB = [s_.clone().requires_grad_() for s_ in samp]
            XD = {i: samp[i].clone().requires_grad_() for i in bio}
            model._split_xd = XD
            try:
                yb = _logit(model, XB)
                gb = torch.autograd.grad(yb.sum(), XB + [XD[i] for i in bio])
            finally:
                model._split_xd = None
                model.__dict__.pop("_locus_tokens", None)
            max_fwd_diff = max(max_fwd_diff, float((ya - yb).abs().max()))

            for k in range(len(x)):
                acc["full"][k] += (ga[k].double() * delta[k]).sum(0)
                acc["split"][k] += (gb[k].double() * delta[k]).sum(0)
            for q, i in enumerate(bio):
                acc_d[i] += (gb[len(x) + q].double() * delta[i]).sum(0)
        for k in range(len(x)):
            for key in ("full", "split"):
                out[key][k][j] = -(acc[key][k] / nsamples).cpu().numpy()
        for i in bio:
            out["dbio"][i][j] = -(acc_d[i] / nsamples).cpu().numpy()
    out["max_fwd_diff"] = max_fwd_diff
    return out


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def _abs_per_channel(p):
    return np.abs(p).sum(axis=2).mean(axis=0)          # (C,) mean over isolates


def _signed_per_channel(p):
    return p.sum(axis=2).mean(axis=0)


def split_channel_table(cfg, phi):
    """One row per (block, channel). For a biophysical block: |full|, |residue|
    (split pass, grad via the residue slot) and |delta| (via D_BIO); for every
    other block the split-pass value is reported beside the full one."""
    rows = []
    for i, (b, pf, ps) in enumerate(zip(cfg["model"]["blocks"], phi["full"], phi["split"])):
        names = b["channel_names"] or [f"ch{c}" for c in range(pf.shape[1])]
        isbio = b["modality"] == "biophysical"
        pd_ = phi["dbio"][i] if isbio else None
        cols = {"abs_full": _abs_per_channel(pf), "signed_full": _signed_per_channel(pf),
                "abs_split_pass": _abs_per_channel(ps)}
        if isbio:
            cols.update(abs_residue=_abs_per_channel(ps), signed_residue=_signed_per_channel(ps),
                        abs_delta=_abs_per_channel(pd_), signed_delta=_signed_per_channel(pd_))
        for c, nm in enumerate(names):
            r = {"block": b["name"], "modality": b["modality"], "locus": b["locus"],
                 "channel": nm, "channel_index": c,
                 "uncalled": (b["modality"], nm) in V3.UNCALLED}
            for k, v in cols.items():
                r[k] = float(v[c])
            rows.append(r)
    return pd.DataFrame(rows)


def shares(ch):
    nb = ch[ch["modality"] != "biophysical"]
    bi = ch[ch["modality"] == "biophysical"]
    by = nb.groupby("modality")["abs_full"].sum()
    res = float(bi["abs_residue"].sum()) if len(bi) else 0.0
    dlt = float(bi["abs_delta"].sum()) if len(bi) else 0.0
    full = float(bi["abs_full"].sum()) if len(bi) else 0.0
    unc = float(nb.loc[nb["uncalled"], "abs_full"].sum())
    tot_a = float(by.sum()) + res + dlt
    tot_b = float(by.sum()) + full
    s = {f"share_{m}": float(by.get(m, 0.0)) / tot_a for m in NONBIO}
    s.update(share_biophysical_residue=res / tot_a, share_biophysical_delta=dlt / tot_a,
             share_uncalled=unc / tot_a)
    s.update({f"alt_share_{m}": float(by.get(m, 0.0)) / tot_b for m in NONBIO})
    s.update(alt_share_biophysical=full / tot_b,
             alt_share_biophysical_residue=res / tot_b,
             alt_share_biophysical_delta=dlt / tot_b,
             alt_share_uncalled=unc / tot_b,
             bio_cancellation=(res + dlt) / full if full > 0 else float("nan"),
             residue_fraction_of_split=res / (res + dlt) if res + dlt > 0 else float("nan"))
    return s


def checks(cfg, phi):
    kinds = [b["modality"] for b in cfg["model"]["blocks"]]
    bio = [i for i, m in enumerate(kinds) if m == "biophysical"]
    nonbio = [i for i, m in enumerate(kinds) if m != "biophysical"]
    full_b = np.concatenate([phi["full"][i].ravel() for i in bio])
    sum_b = np.concatenate([(phi["split"][i] + phi["dbio"][i]).ravel() for i in bio])
    fn = np.concatenate([phi["full"][i].ravel() for i in nonbio])
    sn = np.concatenate([phi["split"][i].ravel() for i in nonbio])
    l1 = lambda a: float(np.abs(a).sum())                     # noqa: E731
    return {
        "decomp_rel_err_l1": l1(sum_b - full_b) / max(l1(full_b), 1e-300),
        "decomp_max_abs_err": float(np.abs(sum_b - full_b).max()),
        "decomp_max_abs_full": float(np.abs(full_b).max()),
        "nonbio_rel_diff_l1": l1(sn - fn) / max(l1(fn), 1e-300),
        "nonbio_max_abs_diff": float(np.abs(sn - fn).max()),
    }


# ---------------------------------------------------------------------------
# one drug
# ---------------------------------------------------------------------------

def run_drug(drug, cell, args):
    run_name = f"{args.run}/{cell}__{args.arch}"
    run_dir = V3.RESULTS_DIR / run_name
    stem = resolve_stem(run_dir, drug)
    weights = V3.WEIGHTS_ROOT / run_name / stem
    model, cfg = load_model(run_name, stem, fold=args.fold, weights_dir=V3.WEIGHTS_ROOT,
                            map_location=args.device)
    model = model.to(args.device).eval()
    if cfg["model"]["arch"] != "locusfusion_v2" or not model.unknown_symbols:
        raise RuntimeError(f"{cell}: not a v3 (locusfusion_v2 + unknown_symbols) model")
    if model.layout["ref_bio"] is not None:
        raise RuntimeError(f"{cell}/{drug}: ref_bio model; the two-path split assumes ref_bio=False")
    reported = float(json.loads((run_dir / f"{stem}.json").read_text())["test"]["auc"])

    t0 = time.time()
    bundle = V3.load_union(cfg, cfg["data"]["modalities_requested"])
    print(f"  [{drug}] load {cfg['data']['modalities_requested']} in {time.time() - t0:.0f}s "
          f"({len(bundle['y'])} labelled isolates)", flush=True)
    rng = np.random.default_rng(args.seed)          # as scripts/shap_locusfusion_v3.py
    data = V3.cell_inputs(cfg, weights, bundle, args.n_background, rng)
    del bundle
    gc.collect()
    p, got = V3.gate(model, data, reported, args.device, args.auc_tol, f"{cell}/{drug}")
    n = min(args.n_explain, len(data["y"]))
    pick = np.sort(rng.choice(len(data["y"]), size=n, replace=False))
    x_test = [a[pick] for a in data["test"]]
    bg = V3.background_arrays(data, args.background)

    t1 = time.time()
    eg_rng = np.random.default_rng([args.seed, 1])   # the expected-gradients draws
    phi = split_expected_gradients(model, cfg, x_test, bg, args.nsamples, args.device,
                                   eg_rng, args.grad_batch)
    secs = time.time() - t1

    ch = split_channel_table(cfg, phi)
    for col, val in (("background", args.background), ("cell", cell), ("drug", drug)):
        ch.insert(0, col, val)
    sh = shares(ch)
    ck = checks(cfg, phi)
    ck["max_fwd_logit_diff"] = phi["max_fwd_diff"]

    existing = {}
    if EXISTING_SHARES.exists():
        ex = pd.read_csv(EXISTING_SHARES)
        row = ex[(ex["drug"] == drug) & (ex["cell"] == cell) & (ex["background"] == args.background)]
        if len(row):
            existing = {f"existing_{k}": float(row.iloc[0][k]) for k in
                        ("share_dna", "share_protein", "share_biophysical",
                         "share_regulatory", "share_uncalled")}

    out_dir = Path(args.out) / drug
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = f"{cell}__{args.background}"
    ch.to_csv(out_dir / f"{pre}__channels_split.csv", index=False)
    meta = {"drug": drug, "cell": cell, "background": args.background,
            "run": args.run, "arch": args.arch, "stem": stem, "weights": str(weights),
            "fold": cfg["best_fold"] if args.fold is None else args.fold,
            "n_test": int(len(data["y"])), "n_explained": int(n),
            "n_background": 1 if args.background == "wt" else int(len(data["bg"][0])),
            "nsamples": args.nsamples, "seed": args.seed,
            "test_auc_reported": reported, "test_auc_rescored": got,
            "auc_abs_diff": abs(got - reported), "gate_tol": args.auc_tol, "gate_pass": True,
            **sh, **ck, **existing, "shap_seconds": round(secs, 1)}
    (out_dir / f"{pre}__meta.json").write_text(json.dumps(meta, indent=2))

    print(f"  [{drug}/{cell}/{args.background}] AUC {got:.4f} (reported {reported:.4f}) | "
          f"{n} isolates in {secs:.0f}s", flush=True)
    print(f"    decomposition: rel L1 err {ck['decomp_rel_err_l1']:.2e}, max abs err "
          f"{ck['decomp_max_abs_err']:.2e} (max |phi| {ck['decomp_max_abs_full']:.2e}) | "
          f"non-bio rel diff {ck['nonbio_rel_diff_l1']:.2e} | fwd logit diff "
          f"{ck['max_fwd_logit_diff']:.2e}", flush=True)
    print("    shares (A): " + ", ".join(f"{k[6:]}={v:.4f}" for k, v in sh.items()
                                         if k.startswith("share_")), flush=True)
    print(f"    alt (existing denominator): biophysical={sh['alt_share_biophysical']:.4f} "
          f"residue={sh['alt_share_biophysical_residue']:.4f} "
          f"delta={sh['alt_share_biophysical_delta']:.4f} "
          f"cancellation={sh['bio_cancellation']:.3f}", flush=True)
    if existing:
        print("    existing modality_shares.csv: " + ", ".join(
            f"{k[9:]}={v:.4f}" for k, v in existing.items()), flush=True)
    del model, data, phi
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return meta


def aggregate(out):
    out = Path(out)
    metas = [json.loads(p.read_text()) for p in sorted(out.glob("*/*__meta.json"))]
    if not metas:
        print("  nothing to aggregate")
        return
    df = pd.DataFrame(metas).sort_values(["drug", "cell", "background"])
    first = ["drug", "cell", "background", "share_dna", "share_protein",
             "share_biophysical_residue", "share_biophysical_delta", "share_regulatory",
             "share_uncalled", "test_auc_rescored", "n_explained"]
    rest = [c for c in df.columns if c.startswith(("alt_share_", "existing_share_"))] + \
        [c for c in ("bio_cancellation", "residue_fraction_of_split", "decomp_rel_err_l1",
                     "decomp_max_abs_err", "nonbio_rel_diff_l1", "max_fwd_logit_diff",
                     "test_auc_reported", "fold", "nsamples", "seed", "shap_seconds")
         if c in df.columns]
    df[first + rest].to_csv(out / "modality_shares_split.csv", index=False)
    pd.concat([pd.read_csv(p) for p in sorted(out.glob("*/*__channels_split.csv"))],
              ignore_index=True).to_csv(out / "channels_split_all.csv", index=False)
    print(f"  modality_shares_split.csv: {len(df)} row(s) -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="locusfusion_v3")
    ap.add_argument("--arch", default="locusfusion_v2")
    ap.add_argument("--cell", default="sd19_all_modalities_unk")
    ap.add_argument("--drugs", nargs="+", default=["LEVOFLOXACIN"], metavar="DRUG")
    ap.add_argument("--background", default="train", choices=V3.BACKGROUNDS)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n-background", type=int, default=100)
    ap.add_argument("--n-explain", type=int, default=300)
    ap.add_argument("--nsamples", type=int, default=64)
    ap.add_argument("--grad-batch", type=int, default=32,
                    help="interpolated samples per forward (memory only; not the estimator)")
    ap.add_argument("--auc-tol", type=float, default=5e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.out)
        return
    drugs = [d.upper() for d in args.drugs]
    bad = [d for d in drugs if d not in V3.ALL_DRUGS]
    if bad:
        ap.error(f"unknown drug(s) {bad}")
    print(f"SHAP v3 token split | snapshot={V3.SNAPSHOT} | cell={args.cell} drugs={drugs} "
          f"background={args.background} device={args.device}")
    print(f"  n_background={args.n_background} n_explain={args.n_explain} "
          f"nsamples={args.nsamples} seed={args.seed}", flush=True)
    failures = []
    for drug in drugs:
        if args.skip_existing and (Path(args.out) / drug /
                                   f"{args.cell}__{args.background}__meta.json").exists():
            print(f"  [{drug}] exists, skip", flush=True)
            continue
        try:
            run_drug(drug, args.cell, args)
        except Exception as e:                          # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  [{drug}/{args.cell}] FAILED: {type(e).__name__}: {e}", flush=True)
            failures.append(drug)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
