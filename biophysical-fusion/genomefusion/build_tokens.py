"""
Build the GenomeFusion token store from the BIG-TB VCFs, as a SLURM array.

    # one shard (array task)
    python -m genomefusion.build_tokens --shard 0 --n-shards 16 --out <dir>
    # merge the shards into one store
    python -m genomefusion.build_tokens --merge --out <dir>

Each task tokenizes its slice of the cohort and writes `shard_<i>.npz`: every
isolate's tokens concatenated, plus the offsets that cut them back apart. The
merge step concatenates the shards in cohort order into `tokens.npz`.

The VCFs are read straight from the read-only pi_annagreen tree -- copying
35,884 files off that spinning disk costs far more than reading them once.
Nothing is written anywhere but `--out`.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from genomefusion.partition import CACHE, WORKSPACE, load_or_build
from genomefusion.reference import load_reference
from genomefusion.tokenizer import tokenize_isolate

VCF_DIR = Path("/project/pi_annagreen_umass_edu/anna/BIG-TB_v2/isolates/BIG-TB_download_2025_07_01")
VCF_SUFFIX = "_variants.vcf.bgz"

TOKEN_FIELDS = ("locus", "pos", "base", "residue", "ref_residue", "consequence", "phase",
                "span", "region", "up_plus", "up_minus", "truncation", "flags", "codon")
SUMMARY_FIELDS = ("sum_locus", "sum_n", "sum_span", "sum_frame", "sum_oof")


def phenotype_ids(pheno_csv=None, cache=CACHE):
    """The cohort: isolates with a phenotype row and a VCF, in sorted order.

    The VCF side comes from ONE directory listing, not a stat per isolate:
    /project is a spinning disk where 18k stats cost minutes. Cached, because
    even the one listing is slow and the cohort never changes.
    """
    path = Path(cache) / "cohort.json"
    if path.exists():
        return json.loads(path.read_text())
    from bigtb_ref import REAL_PHENOTYPE_CSV
    df = pd.read_csv(pheno_csv or REAL_PHENOTYPE_CSV, dtype={"New_ID": str}, low_memory=False)
    have = {e.name[:-len(VCF_SUFFIX)] for e in __import__("os").scandir(VCF_DIR)
            if e.name.endswith(VCF_SUFFIX)}
    ids = sorted(set(df["New_ID"].dropna()) & have)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ids))
    return ids


def build_shard(ids, partition, ref, out_path, drop_synonymous=True, log_every=200):
    ref_arr = np.frombuffer(ref, np.uint8)
    cache = {}
    cols = {k: [] for k in TOKEN_FIELDS}
    sums = {k: [] for k in SUMMARY_FIELDS}
    n_tok, n_sum, stats = [], [], []
    t0 = time.time()
    for i, sid in enumerate(ids):
        tk = tokenize_isolate(VCF_DIR / f"{sid}{VCF_SUFFIX}", partition, ref,
                              drop_synonymous=drop_synonymous, ref_arr=ref_arr,
                              codon_cache=cache)
        for k in TOKEN_FIELDS:
            cols[k].append(getattr(tk, k))
        for k in SUMMARY_FIELDS:
            sums[k].append(getattr(tk, k))
        n_tok.append(len(tk.locus))
        n_sum.append(len(tk.sum_locus))
        stats.append(tk.stats)
        if (i + 1) % log_every == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(ids)}  {el:.0f}s  {el / (i + 1):.2f}s/isolate", flush=True)
    data = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    data.update({k: (np.concatenate(v) if v else np.zeros(0)) for k, v in sums.items()})
    data["n_tok"] = np.asarray(n_tok, np.int64)
    data["n_sum"] = np.asarray(n_sum, np.int64)
    data["ids"] = np.asarray(ids)
    data["stats"] = json.dumps({
        "n_isolates": len(ids),
        "tokens_total": int(sum(n_tok)),
        "tokens_mean": float(np.mean(n_tok)) if n_tok else 0.0,
        "tokens_median": float(np.median(n_tok)) if n_tok else 0.0,
        "tokens_max": int(max(n_tok)) if n_tok else 0,
        "active_loci_mean": float(np.mean(n_sum)) if n_sum else 0.0,
        "active_loci_max": int(max(n_sum)) if n_sum else 0,
        "masked_events_mean": float(np.mean([s["n_masked"] for s in stats])) if stats else 0.0,
        "seconds": round(time.time() - t0, 1)})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **data)
    print(f"[shard] wrote {out_path}: {len(ids)} isolates, {sum(n_tok):,} tokens "
          f"({time.time() - t0:.0f}s)", flush=True)
    return data


def merge(out_dir, n_shards):
    out_dir = Path(out_dir)
    parts = [np.load(out_dir / f"shard_{i}.npz", allow_pickle=False) for i in range(n_shards)]
    data = {}
    for k in TOKEN_FIELDS + SUMMARY_FIELDS + ("n_tok", "n_sum"):
        data[k] = np.concatenate([p[k] for p in parts])
    data["ids"] = np.concatenate([p["ids"] for p in parts])
    n_tok, n_sum = data["n_tok"], data["n_sum"]
    data["tok_off"] = np.concatenate([[0], np.cumsum(n_tok)]).astype(np.int64)
    data["sum_off"] = np.concatenate([[0], np.cumsum(n_sum)]).astype(np.int64)
    stats = {"n_isolates": len(data["ids"]), "tokens_total": int(n_tok.sum()),
             "tokens_mean": float(n_tok.mean()), "tokens_median": float(np.median(n_tok)),
             "tokens_p95": float(np.percentile(n_tok, 95)), "tokens_max": int(n_tok.max()),
             "active_loci_mean": float(n_sum.mean()), "active_loci_max": int(n_sum.max()),
             "shards": [json.loads(str(p["stats"])) for p in parts]}
    data["stats"] = json.dumps(stats)
    path = out_dir / "tokens.npz"
    np.savez(path, **data)
    print(f"[merge] {path}: {stats['n_isolates']:,} isolates, {stats['tokens_total']:,} tokens, "
          f"mean {stats['tokens_mean']:.0f}/isolate, mean {stats['active_loci_mean']:.0f} "
          f"active loci", flush=True)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(WORKSPACE / "genomefusion_tokens"))
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--n-shards", type=int, default=16)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N isolates (a smoke test)")
    ap.add_argument("--keep-synonymous", action="store_true")
    ap.add_argument("--partition-cache", default=str(CACHE))
    args = ap.parse_args(argv)
    out = Path(args.out)

    if args.merge:
        stats = merge(out, args.n_shards)
        (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
        return

    partition = load_or_build(args.partition_cache)
    ref = load_reference(args.partition_cache)
    ids = phenotype_ids()
    if args.limit:
        ids = ids[:args.limit]
    print(f"[build] cohort {len(ids):,} isolates with a phenotype row and a VCF", flush=True)
    sl = ids if args.shard is None else ids[args.shard::args.n_shards]
    name = "shard_all.npz" if args.shard is None else f"shard_{args.shard}.npz"
    print(f"[build] shard {args.shard}/{args.n_shards}: {len(sl)} isolates -> {out / name}",
          flush=True)
    build_shard(sl, partition, ref, out / name, drop_synonymous=not args.keep_synonymous)


if __name__ == "__main__":
    main()
