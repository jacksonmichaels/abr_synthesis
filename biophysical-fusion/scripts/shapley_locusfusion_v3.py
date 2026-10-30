"""
Shapley attributions for the LocusFusion v3 sd19 checkpoints, at modality and
locus granularity, whose sums equal the prediction BY CONSTRUCTION.

Companion to scripts/shap_locusfusion_v3.py (expected gradients). EG's
completeness on these models was poor (median r ~0.35, some runs negative):
the tokenizer runs inside the forward pass, so variant tokens appear and
vanish along the interpolation path and the path integral no longer adds up.
Here the players are whole input groups switched on or off, so the value
function only ever sees inputs of the kind the tokenizer was built for
(an isolate's own delta, or zeros = "identical to H37Rv").

Value function
--------------
v(S) = the model's logit toward RESISTANT (= -logit(P(susceptible))) on the
isolate's delta input with every block/channel owned by a player NOT in S set
to zero. v(all) = f(x); v(empty) = f(H37Rv-like all-zero input). Efficiency:
sum_i phi_i = f(x) - f(0), asserted per isolate.

Zeroing a modality the model was trained with is out of distribution: a value
is "what the model loses without it", not "what the model would have learned
without it". Note also that the v2 tokenizer finds variant positions from the
NUCLEOTIDE blocks only (dna + regulatory) and reads protein / biophysical at
those positions, so protein and biophysical act only through interaction with
dna; Shapley splits that interaction evenly.

Games (per explained isolate)
-----------------------------
modality         players = the cell's modalities (1-4); exact, 2^k coalitions.
called_uncalled  each modality split into called channels and uncalled-base
                 channels (dna N, protein '-' and 'X', regulatory N);
                 biophysical has none and stays whole. Exact, k <= 7.
locus            players = the 19 loci, each owning all of its blocks in every
                 modality incl. its regulatory window. Antithetic permutation
                 sampling (--n-perm), with coalition values memoised per
                 isolate; computed EXACTLY instead when the isolate has
                 2^m <= --exact-max non-null loci coalitions.
In every game a player whose owned input is all zero for this isolate is a
null player (switching it changes nothing), gets phi = 0 exactly, and is left
out of the enumeration/sampling.

Explained isolates are the SAME as the EG study: the same seed, the same
background draw consumed from the rng first, then the same pick, so the rows
match scripts/shap_locusfusion_v3.py's __isolates.csv (checked when present).

`--split lineage` (or ABR_SHAP_SPLIT=lineage) attributes the lineage-split
checkpoints instead (model_weights_lineage/, run lineage_split, default out
results/analysis/shap_locusfusion_lineage_alltest/shapley); see
shap_locusfusion_v3.SPLITS.

Writes per (drug, cell) under {out}/per_cell/{DRUG}/ and, with --aggregate,
{out}/modality_shapley.csv, locus_shapley.csv, gate_shapley.csv
(+ perm_check.csv when --perm-check was run).

    python scripts/shapley_locusfusion_v3.py --drugs LEVOFLOXACIN \
        --cells sd19_all_modalities_unk --perm-check 32 64 128 256   # GPU node
    python scripts/shapley_locusfusion_v3.py --aggregate             # CSV only
"""
import argparse
import gc
import json
import sys
import time
from math import factorial
from pathlib import Path

import shap_locusfusion_v3 as eg  # sets up the v3 snapshot paths + ABR_* env

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from shap_attribution import resolve_stem  # noqa: E402  (from the snapshot)
from training.checkpoint import load_model  # noqa: E402

DEFAULT_OUT = eg.DEFAULT_OUT / "shapley"
MODALITIES = eg.MODALITIES
UNCALLED = eg.UNCALLED
OOD_NOTE = ("zeroing a player's input is out of distribution for a model trained "
            "with it; values mean 'what the model loses without it'")


# ---------------------------------------------------------------------------
# players
# ---------------------------------------------------------------------------

def players_for(cfg, game):
    """-> (names, owners): owners[b] is a (n_players, C_b) 0/1 array saying which
    channels of block b each player owns."""
    blocks = cfg["model"]["blocks"]
    if game == "modality":
        names = [m for m in MODALITIES if any(b["modality"] == m for b in blocks)]
        own = lambda b, p, c: b["modality"] == p                         # noqa: E731
    elif game == "called_uncalled":
        names = []
        for m in MODALITIES:
            bs = [b for b in blocks if b["modality"] == m]
            if not bs:
                continue
            names.append(f"{m}:called")
            if any((m, c) in UNCALLED for c in bs[0]["channel_names"]):
                names.append(f"{m}:uncalled")

        def own(b, p, c):
            m, part = p.split(":")
            if b["modality"] != m:
                return False
            unc = (m, c) in UNCALLED
            return unc if part == "uncalled" else not unc
    elif game == "locus":
        names = list(dict.fromkeys(b["locus"] for b in blocks))
        own = lambda b, p, c: b["locus"] == p                            # noqa: E731
    else:
        raise ValueError(game)
    owners = []
    for b in blocks:
        chans = b["channel_names"] or [f"ch{i}" for i in range(b["channels"])]
        owners.append(np.array([[own(b, p, c) for c in chans] for p in names],
                               dtype=np.float32))
    # every channel of every block is owned by exactly one player
    for b, o in zip(blocks, owners):
        if not np.all(o.sum(0) == 1):
            raise RuntimeError(f"{game}: block {b['name']} channels not partitioned")
    return names, owners


def nonnull(xs_iso, owners):
    """(n_iso, n_players) bool: does the player own any non-zero input entry?"""
    n = xs_iso[0].shape[0]
    out = np.zeros((n, owners[0].shape[0]), dtype=bool)
    for x, o in zip(xs_iso, owners):
        ch_any = np.abs(x).sum(axis=2) > 0                     # (n, C)
        out |= (ch_any.astype(np.float32) @ o.T) > 0
    return out


# ---------------------------------------------------------------------------
# batched value function
# ---------------------------------------------------------------------------

class Evaluator:
    """v(iso, Z) for rows of (isolate index, coalition indicator over players)."""

    def __init__(self, model, xs_iso, owners, device, budget_floats):
        self.model, self.device = model, device
        self.X = [torch.from_numpy(a).to(device).float() for a in xs_iso]
        self.P = [torch.from_numpy(o).to(device) for o in owners]
        per_row = sum(a.shape[1] * a.shape[2] for a in xs_iso)
        self.batch = int(max(8, min(2048, budget_floats // per_row)))
        self.n_forward = 0

    @torch.no_grad()
    def __call__(self, iso, Z):
        iso = np.asarray(iso)
        Z = np.asarray(Z, dtype=np.float32)
        out = np.empty(len(iso), dtype=np.float64)
        for s in range(0, len(iso), self.batch):
            ii = torch.from_numpy(iso[s:s + self.batch]).to(self.device)
            zz = torch.from_numpy(Z[s:s + self.batch]).to(self.device)
            xs = [x[ii] * (zz @ p).clamp_(max=1.0).unsqueeze(-1)
                  for x, p in zip(self.X, self.P)]
            out[s:s + len(ii)] = (-self.model(xs).reshape(-1)).double().cpu().numpy()
            self.n_forward += len(ii)
        return out


def bits_to_Z(masks, nn_idx, n_players):
    """int bitmasks over the non-null players -> (len, n_players) indicator."""
    masks = np.asarray(masks, dtype=np.int64)
    Z = np.zeros((len(masks), n_players), dtype=np.float32)
    for j, p in enumerate(nn_idx):
        Z[:, p] = (masks >> j) & 1
    return Z


def exact_from_values(v, m):
    """Exact Shapley for m players from v over all 2^m bitmasks."""
    masks = np.arange(1 << m)
    pc = np.array([bin(x).count("1") for x in masks])
    w = np.array([factorial(s) * factorial(m - s - 1) / factorial(m) for s in range(m)])
    phi = np.zeros(m)
    for i in range(m):
        sel = masks[((masks >> i) & 1) == 0]
        phi[i] = np.sum(w[pc[sel]] * (v[sel | (1 << i)] - v[sel]))
    return phi


# ---------------------------------------------------------------------------
# games
# ---------------------------------------------------------------------------

def exact_game(ev, nn, n_players):
    """Exact Shapley for every isolate (null players skipped). -> phi (n, P)."""
    n = nn.shape[0]
    rows_iso, rows_Z = [], []
    for i in range(n):
        idx = np.nonzero(nn[i])[0]
        m = len(idx)
        masks = np.arange(1 << m)
        rows_iso.append(np.full(len(masks), i))
        rows_Z.append(bits_to_Z(masks, idx, n_players))
    v = ev(np.concatenate(rows_iso), np.concatenate(rows_Z))
    phi = np.zeros((n, n_players))
    vfull, vempty = np.zeros(n), np.zeros(n)
    off = 0
    for i in range(n):
        idx = np.nonzero(nn[i])[0]
        m = len(idx)
        vi = v[off:off + (1 << m)]
        off += 1 << m
        if m:
            phi[i, idx] = exact_from_values(vi, m)
        vempty[i], vfull[i] = vi[0], vi[-1]
    return phi, vfull, vempty


def locus_game(ev, nn, n_players, n_perm, exact_max, seed, check=()):
    """Antithetic permutation Shapley over the non-null loci, memoised per
    isolate; exact when 2^m <= exact_max. Returns phi (n, P), se (n, P),
    exact flag (n,), v(full), v(empty), and {k: (phi_k, se_k)} for the nested
    first-k-permutation estimates, k in `check`."""
    n = nn.shape[0]
    n_pairs = n_perm // 2
    plans, rows_iso, rows_Z = [], [], []
    for i in range(n):
        idx = np.nonzero(nn[i])[0]
        m = len(idx)
        full = (1 << m) - 1
        if (1 << m) <= exact_max:
            masks = np.arange(1 << m)
            plans.append(("exact", idx, masks, None))
        else:
            rng = np.random.default_rng([seed, i])
            base = np.array([rng.permutation(m) for _ in range(n_pairs)])
            perms = np.empty((2 * n_pairs, m), dtype=np.int64)
            perms[0::2], perms[1::2] = base, base[:, ::-1]          # antithetic pairs
            pref = np.zeros((len(perms), m + 1), dtype=np.int64)
            for k in range(m):
                pref[:, k + 1] = pref[:, k] | (np.int64(1) << perms[:, k])
            masks, inv = np.unique(pref, return_inverse=True)
            plans.append(("perm", idx, masks, (perms, inv.reshape(pref.shape))))
        assert masks[0] == 0 and masks[-1] == full
        rows_iso.append(np.full(len(masks), i))
        rows_Z.append(bits_to_Z(masks, idx, n_players))
    v = ev(np.concatenate(rows_iso), np.concatenate(rows_Z))

    phi = np.zeros((n, n_players))
    se = np.zeros((n, n_players))
    exact = np.zeros(n, dtype=bool)
    vfull, vempty = np.zeros(n), np.zeros(n)
    chk = {k: (np.zeros((n, n_players)), np.zeros((n, n_players))) for k in check}
    off = 0
    for i, (kind, idx, masks, extra) in enumerate(plans):
        vi = v[off:off + len(masks)]
        off += len(masks)
        vempty[i], vfull[i] = vi[0], vi[-1]
        m = len(idx)
        if m == 0:
            exact[i] = True
            continue
        if kind == "exact":
            exact[i] = True
            phi[i, idx] = exact_from_values(vi, m)
            for k in check:
                chk[k][0][i, idx] = phi[i, idx]
            continue
        perms, inv = extra
        vals = vi[inv]                                   # (n_perm, m+1)
        marg = np.zeros((len(perms), m))
        np.put_along_axis(marg, perms, np.diff(vals, axis=1), axis=1)
        pair = 0.5 * (marg[0::2] + marg[1::2])           # iid across pairs
        phi[i, idx] = pair.mean(0)
        se[i, idx] = pair.std(0, ddof=1) / np.sqrt(len(pair)) if len(pair) > 1 else np.nan
        for k in check:
            pk = pair[:k // 2]
            chk[k][0][i, idx] = pk.mean(0)
            chk[k][1][i, idx] = (pk.std(0, ddof=1) / np.sqrt(len(pk))
                                 if len(pk) > 1 else np.nan)
    return phi, se, exact, vfull, vempty, chk


# ---------------------------------------------------------------------------
# one (drug, cell)
# ---------------------------------------------------------------------------

def summarise(phi, names, **extra):
    a = np.abs(phi).mean(0)
    tot = a.sum()
    return pd.DataFrame({"player": names, "mean_abs_phi": a,
                         "share": a / tot if tot > 0 else np.nan,
                         "mean_signed": phi.mean(0), **extra})


def run_cell(cell, drug, args, cache):
    run_name = f"{args.run}/{cell}__{args.arch}"
    run_dir = eg.RESULTS_DIR / run_name
    stem = resolve_stem(run_dir, drug)
    weights = eg.WEIGHTS_ROOT / run_name / stem
    model, cfg = load_model(run_name, stem, fold=args.fold, weights_dir=eg.WEIGHTS_ROOT,
                            map_location=args.device)
    model = model.to(args.device).eval()
    if cfg["model"]["arch"] != "locusfusion_v2" or not model.unknown_symbols:
        raise RuntimeError(f"{cell}: not a v3 (locusfusion_v2 + unknown_symbols) model")
    reported = float(json.loads((run_dir / f"{stem}.json").read_text())["test"]["auc"])

    # identical rng consumption to shap_locusfusion_v3.run_cell -> same pick
    rng = np.random.default_rng(args.seed)
    union = cache.get("bundle")
    data, shared = None, union is not None
    if shared:
        try:
            data = eg.cell_inputs(cfg, weights, union, args.n_background, rng)
            p, got = eg.gate(model, data, reported, args.device, args.auc_tol,
                             f"{cell}/{drug} (shared load)")
        except RuntimeError as e:
            print(f"  [{drug}/{cell}] shared load rejected ({e}); per-cell reload",
                  flush=True)
            data, shared = None, False
    if data is None:
        own = eg.load_union(cfg, cfg["data"]["modalities_requested"])
        rng = np.random.default_rng(args.seed)
        data = eg.cell_inputs(cfg, weights, own, args.n_background, rng)
        del own
        gc.collect()
        p, got = eg.gate(model, data, reported, args.device, args.auc_tol, f"{cell}/{drug}")
    n = min(args.n_explain, len(data["y"]))
    pick = np.sort(rng.choice(len(data["y"]), size=n, replace=False))
    data.pop("bg", None)
    xs = [a[pick] for a in data["test"]]
    ids = data["ids"][pick]
    y_res = (data["y"][pick] == 0).astype(int)

    # EG pick parity
    eg_iso = (Path(args.eg_out) / args.run / args.arch / drug / f"{cell}__train__isolates.csv")
    pick_match = None
    if eg_iso.exists():
        eg_ids = [str(s) for s in pd.read_csv(eg_iso)["isolate_id"]]
        if sorted(eg_ids) != sorted(str(s) for s in ids):
            raise RuntimeError(f"{drug}/{cell}: explained isolates differ from EG's")
        pick_match = True

    pr = np.clip(1.0 - p[pick], 1e-7, 1 - 1e-7)
    f_x = np.log(pr / (1 - pr))                         # logit toward resistant
    with torch.no_grad():
        f0 = float(-model([torch.zeros((1,) + a.shape[1:], device=args.device)
                           for a in xs]).reshape(-1)[0])
    counts, _trunc = eg.token_census(model, xs, args.device)

    out_dir = Path(args.out) / "per_cell" / drug
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = f"{cell}"
    meta = {"drug": drug, "cell": cell, "split": eg.SPLIT, "run": args.run,
            "arch": args.arch, "stem": stem, "weights": str(weights),
            "fold": cfg["best_fold"] if args.fold is None else args.fold,
            "shared_load": shared, "n_test": int(len(data["y"])), "n_explained": int(n),
            "seed": args.seed, "pick_matches_eg": pick_match,
            "test_auc_reported": reported, "test_auc_rescored": got,
            "auc_abs_diff": abs(got - reported), "gate_tol": args.auc_tol,
            "f_all_absent": f0, "f_x_mean": float(f_x.mean()),
            "f_x_mean_R": float(f_x[y_res == 1].mean()) if y_res.any() else None,
            "f_x_mean_S": float(f_x[y_res == 0].mean()) if (y_res == 0).any() else None,
            "note": OOD_NOTE}
    mod_tabs, npz = [], {"ids": ids.astype(str), "f_x": f_x, "y_resistant": y_res}
    t_all = time.time()

    # ---- modality + called/uncalled (exact) -------------------------------
    for game in ("modality", "called_uncalled"):
        t0 = time.time()
        names, owners = players_for(cfg, game)
        nn = nonnull(xs, owners)
        ev = Evaluator(model, xs, owners, args.device, args.budget_floats)
        phi, vfull, vempty = exact_game(ev, nn, len(names))
        err = np.abs(phi.sum(1) - (f_x - f0))
        ref_err = max(np.abs(vfull - f_x).max(), np.abs(vempty - f0).max())
        meta[f"{game}_max_eff_err"] = float(err.max())
        meta[f"{game}_max_endpoint_err"] = float(ref_err)
        meta[f"{game}_n_players"] = len(names)
        meta[f"{game}_forwards"] = ev.n_forward
        meta[f"{game}_seconds"] = round(time.time() - t0, 1)
        if err.max() > args.eff_tol:
            raise AssertionError(f"{drug}/{cell}/{game}: efficiency error {err.max():.2e}")
        t = summarise(phi, names, game=game, frac_isolates_nonnull=nn.mean(0))
        mod_tabs.append(t)
        npz[f"phi_{game}"], npz[f"players_{game}"] = phi, np.array(names)
        del ev
        print(f"  [{drug}/{cell}/{game}] {len(names)} players, {meta[f'{game}_forwards']} "
              f"forwards in {meta[f'{game}_seconds']}s | max eff err {err.max():.1e} | "
              + ", ".join(f"{a}={s:.3f}" for a, s in zip(t['player'], t['share'])),
              flush=True)

    # ---- locus (permutation / exact) --------------------------------------
    t0 = time.time()
    names, owners = players_for(cfg, "locus")
    nn = nonnull(xs, owners)
    ev = Evaluator(model, xs, owners, args.device, args.budget_floats)
    check = sorted(set(args.perm_check) | {args.n_perm})
    _phi_max, _se_max, exact, vfull, vempty, chk = locus_game(
        ev, nn, len(names), max(check), args.exact_max, args.seed, check)
    phi, se = chk[args.n_perm]          # the nested first-n_perm estimate
    err = np.abs(phi.sum(1) - (f_x - f0))
    meta.update({"locus_max_eff_err": float(err.max()),
                 "locus_max_endpoint_err": float(max(np.abs(vfull - f_x).max(),
                                                     np.abs(vempty - f0).max())),
                 "locus_n_perm": args.n_perm, "locus_exact_max": args.exact_max,
                 "locus_frac_exact": float(exact.mean()),
                 "locus_mean_nonnull": float(nn.sum(1).mean()),
                 "locus_max_nonnull": int(nn.sum(1).max()),
                 "locus_forwards": ev.n_forward,
                 "locus_seconds": round(time.time() - t0, 1)})
    if err.max() > args.eff_tol:
        raise AssertionError(f"{drug}/{cell}/locus: efficiency error {err.max():.2e}")
    a = np.abs(phi).mean(0)
    tot = a.sum()
    li = {l: i for i, l in enumerate(model.loci)}
    loc = pd.DataFrame({
        "locus": names, "share": a / tot, "mean_abs_phi": a, "mean_signed": phi.mean(0),
        # Monte-Carlo SE of the cell share (permutation noise only; isolates
        # independent; ignores the denominator's own noise)
        "se": np.sqrt((se ** 2).sum(0)) / len(phi) / tot,
        "se_phi_isolate_mean": np.nanmean(np.where(nn & ~exact[:, None], se, np.nan), 0)
        if (~exact).any() else np.zeros(len(names)),
        "frac_isolates_nonnull": nn.mean(0),
        "mean_variant_tokens": [counts[:, li[l]].mean() if l in li else np.nan
                                for l in names]})
    loc["rank"] = loc["share"].rank(ascending=False, method="first").astype(int)
    npz.update({"phi_locus": phi, "se_locus": se, "players_locus": np.array(names),
                "locus_exact": exact})
    print(f"  [{drug}/{cell}/locus] {ev.n_forward} forwards in {meta['locus_seconds']}s | "
          f"exact {exact.mean():.0%} | non-null loci mean {nn.sum(1).mean():.1f} max "
          f"{nn.sum(1).max()} | max eff err {err.max():.1e} | top "
          + ", ".join(f"{l}={s:.3f}" for l, s in
                      loc.nlargest(3, "share")[["locus", "share"]].values), flush=True)

    pc_rows = []
    if args.perm_check:
        shares = {}
        for k in sorted(args.perm_check):
            ak = np.abs(chk[k][0]).mean(0)
            shares[k] = ak / ak.sum()
        ref = max(shares)
        for k in sorted(shares):
            for l, s, r in zip(names, shares[k], shares[ref]):
                pc_rows.append({"drug": drug, "cell": cell, "n_perm": k, "locus": l,
                                "share": s, "share_at_max": r, "abs_diff": abs(s - r)})
        print("  perm check (max |share - share@%d|): " % ref + ", ".join(
            f"{k}: {np.abs(shares[k] - shares[ref]).max():.4f}" for k in sorted(shares)),
            flush=True)
        if len(shares) > 1:
            ks = sorted(shares)
            print("  perm check (successive): " + ", ".join(
                f"{a}->{b}: {np.abs(shares[a] - shares[b]).max():.4f}"
                for a, b in zip(ks, ks[1:])), flush=True)
    meta["total_seconds"] = round(time.time() - t_all, 1)

    keys = {"drug": drug, "cell": cell}
    mods = pd.concat(mod_tabs, ignore_index=True)
    for k, v in reversed(list(keys.items())):
        mods.insert(0, k, v)
        loc.insert(0, k, v)
    loc.insert(2, "background", "shapley")
    mods.to_csv(out_dir / f"{pre}__modality.csv", index=False)
    loc.sort_values("rank").to_csv(out_dir / f"{pre}__locus.csv", index=False)
    if pc_rows:
        pd.DataFrame(pc_rows).to_csv(out_dir / f"{pre}__perm_check.csv", index=False)
    np.savez_compressed(out_dir / f"{pre}__phi.npz", **npz)
    (out_dir / f"{pre}__meta.json").write_text(json.dumps(meta, indent=2, default=str))

    del model, data, xs, ev
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return meta


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def aggregate(out):
    out = Path(out)
    root = out / "per_cell"
    for name, dest in (("modality", "modality_shapley.csv"), ("locus", "locus_shapley.csv"),
                       ("perm_check", "perm_check.csv")):
        paths = sorted(root.glob(f"*/*__{name}.csv"))
        if paths:
            df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
            if name == "locus":
                df = df.sort_values(["drug", "cell", "rank"])
            df.to_csv(out / dest, index=False)
            print(f"  {dest}: {len(df)} rows from {len(paths)} file(s)")
    metas = [json.loads(p.read_text()) for p in sorted(root.glob("*/*__meta.json"))]
    if metas:
        g = pd.DataFrame(metas).sort_values(["drug", "cell"])
        g.to_csv(out / "gate_shapley.csv", index=False)
        effc = [c for c in g if c.endswith("_max_eff_err")]
        print(f"  gate_shapley.csv: {len(g)} rows | max efficiency error "
              f"{g[effc].max().max():.2e} | max AUC diff {g['auc_abs_diff'].max():.1e}")
    fails = sorted(root.glob("failures_*.json"))
    if fails:
        print(f"  NOTE: {len(fails)} failure file(s): " + ", ".join(p.name for p in fails))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=sorted(eg.SPLITS), default=eg.SPLIT,
                    help="trained grid (see shap_locusfusion_v3.SPLITS); read before "
                         "import; also $ABR_SHAP_SPLIT")
    ap.add_argument("--run", default=eg.DEFAULT_RUN)
    ap.add_argument("--arch", default="locusfusion_v2")
    ap.add_argument("--cells", nargs="+", default=eg.DEFAULT_CELLS, metavar="CELL")
    ap.add_argument("--drugs", nargs="+", default=["LEVOFLOXACIN"], metavar="DRUG")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n-background", type=int, default=100,
                    help="only consumed from the rng so the pick matches EG")
    ap.add_argument("--n-explain", type=int, default=300)
    ap.add_argument("--n-perm", type=int, default=64, help="locus-game permutations (even)")
    ap.add_argument("--perm-check", type=int, nargs="*", default=[],
                    help="also report locus shares at these nested permutation counts")
    ap.add_argument("--exact-max", type=int, default=512,
                    help="locus game exact when 2^(non-null loci) <= this")
    ap.add_argument("--budget-floats", type=float, default=2.5e8,
                    help="input floats per forward batch (sets the batch size)")
    ap.add_argument("--eff-tol", type=float, default=1e-3)
    ap.add_argument("--auc-tol", type=float, default=5e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--eg-out", default=str(eg.DEFAULT_OUT))
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    if args.aggregate:
        print(f"aggregating {args.out}")
        aggregate(args.out)
        return
    if args.n_perm % 2 or any(k % 2 for k in args.perm_check):
        ap.error("permutation counts must be even (antithetic pairs)")
    drugs = (eg.ALL_DRUGS if any(d.lower() == "all" for d in args.drugs)
             else [d.upper() for d in args.drugs])
    bad = [d for d in drugs if d not in eg.ALL_DRUGS]
    if bad:
        ap.error(f"unknown drug(s) {bad}")
    print(f"Shapley v3 | split={eg.SPLIT} snapshot={eg.SNAPSHOT} weights={eg.WEIGHTS_ROOT} "
          f"run={args.run} out={args.out} | cells={args.cells} drugs={drugs} "
          f"n_explain={args.n_explain} n_perm={args.n_perm} perm_check={args.perm_check} "
          f"exact_max={args.exact_max} device={args.device}", flush=True)

    root = Path(args.out) / "per_cell"
    root.mkdir(parents=True, exist_ok=True)
    metas, failures = [], []
    for drug in drugs:
        todo = [c for c in args.cells if not (
            args.skip_existing and (root / drug / f"{c}__meta.json").exists())]
        if not todo:
            print(f"  [{drug}] all cells exist, skip", flush=True)
            continue
        cache = {}
        try:
            cfgs = {}
            for c in todo:
                rd = eg.RESULTS_DIR / args.run / f"{c}__{args.arch}"
                stem = resolve_stem(rd, drug)
                cfgs[c] = json.loads((eg.WEIGHTS_ROOT / args.run / f"{c}__{args.arch}"
                                      / stem / "config.json").read_text())
            widest = max(cfgs, key=lambda c: len(cfgs[c]["data"]["modalities_requested"]))
            mods = [m for m in MODALITIES
                    if any(m in cfgs[c]["data"]["modalities_requested"] for c in todo)]
            t0 = time.time()
            cache["bundle"] = eg.load_union(cfgs[widest], mods)
            print(f"  [{drug}] shared load {mods} in {time.time() - t0:.0f}s "
                  f"({len(cache['bundle']['y'])} labelled isolates)", flush=True)
        except Exception as e:                          # noqa: BLE001
            print(f"  [{drug}] shared load failed ({type(e).__name__}: {e}); "
                  "per-cell loads", flush=True)
        for cell in todo:
            try:
                metas.append(run_cell(cell, drug, args, cache))
            except Exception as e:                      # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"  [{drug}/{cell}] FAILED: {type(e).__name__}: {e}", flush=True)
                failures.append({"drug": drug, "cell": cell,
                                 "error": f"{type(e).__name__}: {e}"})
        cache.clear()
        gc.collect()

    print(f"\nwrote {len(metas)} result(s) -> {root}")
    if failures:
        tag = "_".join(sorted({f["drug"] for f in failures}))
        (root / f"failures_{tag}.json").write_text(json.dumps(failures, indent=2))
        print(f"\n{len(failures)} FAILED:")
        for f in failures:
            print(f"  {f['drug']}/{f['cell']}: {f['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
