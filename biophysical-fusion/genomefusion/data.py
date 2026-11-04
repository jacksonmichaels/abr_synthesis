"""
The token store -> padded batches.

The store holds one row per token, concatenated over isolates, with the offsets
that cut it back apart. Channels are stored as compact integer codes (a residue
index, not a 23-wide one-hot), because 17.9k isolates x ~800 tokens x 57 float
channels is 3 GB while the codes are ~90 MB.

`BatchBuilder` expands every token ONCE into one (n_tokens, C) float16 matrix on
the device, so building a batch is a gather and a scatter, with no Python loop
over tokens. At ~14M tokens that matrix is about 1.6 GB in fp16.

A batch pads only to its own maximum, never to the 6,776 loci or to a fixed
token count:

    feat      (B, Lmax, Tmax, C)   token channels
    valid     (B, Lmax, Tmax)      real token
    lvalid    (B, Lmax)            real slot; slot 0 is the learned reference token
    locus_id  (B, Lmax)            compact locus id, -1 in the reference slot
    coord     (B, Lmax, Tmax)      position within the locus, for the sinusoid
    site_pos  (B, Lmax, Tmax)      index into the per-drug distance table, -1 none
    stats     (B, Lmax, 4)         the [NULL] token's per-locus numbers

A length-grouped sampler keeps the padding small: isolates are shuffled inside
buckets of similar active-locus count, so one isolate with 1,500 active loci
does not pad a whole batch of isolates with 300.
"""
import json

import numpy as np
import pandas as pd
import torch

from genomefusion.layout import (C_BASE, N_NULL_STATS, SLOTS, UPSTREAM_LENGTH, UPSTREAM_MISSING,
                                 residue_property_table)

TOKEN_FIELDS = ("locus", "pos", "base", "residue", "ref_residue", "consequence", "phase",
                "span", "region", "up_plus", "up_minus", "truncation", "flags", "codon")


class TokenStore:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.ids = [str(x) for x in z["ids"]]
        self.index = {s: i for i, s in enumerate(self.ids)}
        self.tok = {k: z[k] for k in TOKEN_FIELDS}
        self.sum_locus, self.sum_n = z["sum_locus"], z["sum_n"]
        self.sum_span, self.sum_frame, self.sum_oof = z["sum_span"], z["sum_frame"], z["sum_oof"]
        self.tok_off, self.sum_off = z["tok_off"], z["sum_off"]
        self.stats = json.loads(str(z["stats"]))
        self.path = str(path)

    def __len__(self):
        return len(self.ids)


def _run_index(keys):
    """Index of each element within its run of equal keys (keys sorted in runs)."""
    if len(keys) == 0:
        return np.zeros(0, np.int64)
    new = np.empty(len(keys), bool)
    new[0] = True
    new[1:] = keys[1:] != keys[:-1]
    run_start = np.flatnonzero(new)
    starts = np.repeat(run_start, np.diff(np.append(run_start, len(keys))))
    return np.arange(len(keys), dtype=np.int64) - starts


class BatchBuilder:
    """Expands the store once, then builds padded batches by gather/scatter."""

    def __init__(self, store, partition, loci=None, population=False, device="cpu",
                 site_index=None, max_tokens_per_locus=None):
        self.store, self.partition, self.device = store, partition, device
        self.population = bool(population)
        self.c_tok = C_BASE + (2 if population else 0)

        keep = np.zeros(len(partition.loci), bool)
        keep[np.asarray(list(loci), np.int64) if loci is not None
             else np.arange(len(partition.loci))] = True
        self.keep = keep
        self.kept_loci = np.flatnonzero(keep)
        remap = np.full(len(partition.loci), -1, np.int32)
        remap[self.kept_loci] = np.arange(len(self.kept_loci), dtype=np.int32)
        self.remap = remap
        self.n_loci = len(self.kept_loci)

        T = store.tok
        tok_keep = keep[T["locus"]]
        sum_keep = keep[store.sum_locus]

        # --- per-isolate offsets after filtering to the kept loci -------------
        n_iso = len(store)
        tok_iso = np.repeat(np.arange(n_iso), np.diff(store.tok_off))
        sum_iso = np.repeat(np.arange(n_iso), np.diff(store.sum_off))
        self.tok_rows = np.flatnonzero(tok_keep)
        self.sum_rows = np.flatnonzero(sum_keep)
        tki, ski = tok_iso[self.tok_rows], sum_iso[self.sum_rows]
        self.tok_off = np.concatenate([[0], np.cumsum(np.bincount(tki, minlength=n_iso))])
        self.sum_off = np.concatenate([[0], np.cumsum(np.bincount(ski, minlength=n_iso))])
        self.n_active = np.diff(self.sum_off)

        # --- each token's slot: which active locus, and which token inside it --
        sl = store.sum_locus[self.sum_rows]
        tl = T["locus"][self.tok_rows]
        # the summary lists each isolate's active loci in ascending locus order,
        # and the tokens are sorted the same way, so a token's locus slot is the
        # run index of its locus within the isolate
        lslot = _run_index_of_locus(tl, tki)
        within = _run_index(tl + (tki.astype(np.int64) << 32))
        if max_tokens_per_locus:
            k = within < max_tokens_per_locus
            self.tok_rows, tl, tki, lslot, within = (x[k] for x in
                                                     (self.tok_rows, tl, tki, lslot, within))
            self.tok_off = np.concatenate([[0], np.cumsum(np.bincount(tki, minlength=n_iso))])
        self.tok_lslot = lslot
        self.tok_within = within
        self.tok_iso = tki
        self.sum_iso = ski
        self._check(sl, ski, tl, tki, lslot)

        # --- the expanded feature matrix, built once --------------------------
        feat, coord, site = self._expand(self.tok_rows, site_index)
        self.feat = torch.as_tensor(feat, device=device)
        self.coord = torch.as_tensor(coord, device=device)
        self.site = torch.as_tensor(site, device=device)
        self.t_lslot = torch.as_tensor(lslot, device=device)
        self.t_within = torch.as_tensor(within, device=device)

        # --- the [NULL] stats, per active locus -------------------------------
        lens = np.asarray([max(partition.loci[i].end - partition.loci[i].start, 1)
                           for i in sl], np.float32)
        nn = store.sum_n[self.sum_rows].astype(np.float32)
        st = np.stack([np.log1p(nn), nn / lens,
                       store.sum_frame[self.sum_rows] / 3.0,
                       store.sum_oof[self.sum_rows]], 1).astype(np.float32)
        self.sum_stats = torch.as_tensor(st, device=device)
        self.sum_lid = torch.as_tensor(remap[sl].astype(np.int64), device=device)

    def _check(self, sl, ski, tl, tki, lslot):
        """A token's locus slot must point at the same locus the summary lists."""
        if not len(tl):
            return
        n = min(len(tl), 100_000)
        want = sl[self.sum_off[tki[:n]] + lslot[:n]]
        if not np.array_equal(want, tl[:n]):
            bad = int(np.flatnonzero(want != tl[:n])[0])
            raise AssertionError(
                f"token/summary mismatch at {bad}: token locus {tl[bad]} but summary slot "
                f"{lslot[bad]} of isolate {tki[bad]} is {want[bad]}")

    def _expand(self, rows, site_index):
        T = self.store.tok
        n, C = len(rows), self.c_tok
        feat = np.zeros((n, C), np.float16)
        prop = residue_property_table()
        a, _ = SLOTS["base"]
        feat[np.arange(n), a + T["base"][rows].astype(np.int64)] = 1.0
        res = T["residue"][rows].astype(np.int64)
        hr = res >= 0
        a, _ = SLOTS["residue"]
        feat[np.flatnonzero(hr), a + res[hr]] = 1.0
        a, _ = SLOTS["bio"]
        feat[hr, a:a + 3] = prop[res[hr]]
        rres = T["ref_residue"][rows].astype(np.int64)
        hrr = hr & (rres >= 0)
        a, _ = SLOTS["d_bio"]
        feat[hrr, a:a + 3] = prop[res[hrr]] - prop[rres[hrr]]
        cons = T["consequence"][rows].astype(np.int64)
        hc = cons >= 0
        a, _ = SLOTS["consequence"]
        feat[np.flatnonzero(hc), a + cons[hc]] = 1.0
        a, _ = SLOTS["truncation"]
        feat[:, a] = T["truncation"][rows]
        a, _ = SLOTS["region"]
        feat[np.arange(n), a + T["region"][rows].astype(np.int64)] = 1.0
        a, _ = SLOTS["upstream"]
        for j, k in enumerate(("up_plus", "up_minus")):
            d = T[k][rows].astype(np.float32)
            feat[:, a + j] = np.where(d < 0, UPSTREAM_MISSING, np.exp(-np.abs(d) / UPSTREAM_LENGTH))
        a, _ = SLOTS["span"]
        feat[:, a] = np.log1p(T["span"][rows].astype(np.float32))
        ph = T["phase"][rows].astype(np.int64)
        hp = ph >= 0
        a, _ = SLOTS["phase"]
        feat[np.flatnonzero(hp), a + ph[hp]] = 1.0
        a, _ = SLOTS["flags"]
        fl = T["flags"][rows].astype(np.int64)
        for j in range(3):
            feat[:, a + j] = (fl >> j) & 1
        starts = np.asarray([self.partition.loci[i].start for i in self.partition_idx(rows)],
                            np.int64)
        coord = (T["pos"][rows] - starts).astype(np.float32)
        site = (site_index[T["pos"][rows]].astype(np.int64) if site_index is not None
                else np.full(n, -1, np.int64))
        return feat, coord, site

    def partition_idx(self, rows):
        return self.store.tok["locus"][rows]

    def build(self, rows):
        """rows: store row indices -> a PACKED batch.

        Nothing is padded to (B, n_loci, max_tokens): the median locus carries
        one token and the busiest carries ~90, so that tensor would be ~80x the
        real data. Instead every (isolate, active locus) pair in the batch is one
        row of a flat pair list, and the tokens are a flat list pointing into it.
        The model buckets pairs by token count, so the within-locus encoder pads
        each bucket only to its own width. Only the across-loci stage uses a
        (B, Lmax, d) tensor, which is small.

            tok_*   (N,)    one row per token, in pair order
            pair_*  (P,)    one row per (isolate, active locus)
            Lmax            slots per isolate, slot 0 = the reference token
        """
        rows = np.asarray(rows, np.int64)
        B = len(rows)
        dev = self.device
        ss, se = self.sum_off[rows], self.sum_off[rows + 1]
        sc = se - ss
        P = int(sc.sum())
        Lmax = (int(sc.max()) if B else 0) + 1           # +1 for the reference slot
        if P:
            sgi = np.repeat(ss, sc) + (np.arange(P) - np.repeat(
                np.concatenate([[0], np.cumsum(sc)])[:-1], sc))
            pair_bi = np.repeat(np.arange(B), sc)
            pair_slot = _run_index(pair_bi) + 1
        else:
            sgi = pair_bi = pair_slot = np.zeros(0, np.int64)

        ts, te = self.tok_off[rows], self.tok_off[rows + 1]
        tc = te - ts
        N = int(tc.sum())
        if N:
            gi = np.repeat(ts, tc) + (np.arange(N) - np.repeat(
                np.concatenate([[0], np.cumsum(tc)])[:-1], tc))
            tok_bi = np.repeat(np.arange(B), tc)
            # pair index = where this isolate's pairs start + the token's locus slot
            pair_start = np.concatenate([[0], np.cumsum(sc)])[:-1]
            tok_pair = pair_start[tok_bi] + self.tok_lslot[gi]
        else:
            gi = tok_bi = tok_pair = np.zeros(0, np.int64)

        t = lambda a, dt=torch.long: torch.as_tensor(np.asarray(a), device=dev, dtype=dt)  # noqa: E731
        t_gi = t(gi)
        t_sgi = t(sgi)
        return {
            "B": B, "Lmax": Lmax, "P": P,
            "tok_feat": self.feat[t_gi],
            "tok_coord": self.coord[t_gi],
            "tok_site": self.site[t_gi],
            "tok_pair": t(tok_pair),
            "pair_bi": t(pair_bi),
            "pair_slot": t(pair_slot),
            "pair_lid": self.sum_lid[t_sgi],
            "pair_stats": self.sum_stats[t_sgi],
            # counted from the tokens actually in this batch, so a per-locus cap
            # or a locus filter cannot leave it stale
            "pair_n": t(np.bincount(tok_pair, minlength=P) if P else np.zeros(0)),
        }


def _run_index_of_locus(tl, tki):
    """Each token's index among its isolate's distinct loci (both sorted)."""
    if not len(tl):
        return np.zeros(0, np.int64)
    key = tl.astype(np.int64) + (tki.astype(np.int64) << 32)
    new = np.empty(len(key), bool)
    new[0] = True
    new[1:] = key[1:] != key[:-1]
    # restart the count at each new isolate
    iso_new = np.empty(len(key), bool)
    iso_new[0] = True
    iso_new[1:] = tki[1:] != tki[:-1]
    idx = np.cumsum(new) - 1
    base = np.maximum.accumulate(np.where(iso_new, idx, 0))
    return idx - base


def length_grouped_batches(n_active, rows, batch_size, rng, bucket=8):
    """Batches of `rows` grouped so each holds isolates of similar active-locus count."""
    order = rows[np.argsort(n_active[rows], kind="stable")]
    big = batch_size * bucket
    chunks = []
    for s in range(0, len(order), big):
        c = order[s:s + big]
        c = c[rng.permutation(len(c))]
        chunks.extend(c[i:i + batch_size] for i in range(0, len(c), batch_size))
    rng.shuffle(chunks)
    return chunks


class Labels:
    def __init__(self, store, drugs, pheno_csv):
        df = pd.read_csv(pheno_csv, dtype={"New_ID": str}, low_memory=False)
        df = df.drop_duplicates("New_ID").set_index("New_ID")
        Y = np.full((len(store), len(drugs)), -1, np.int8)
        for j, d in enumerate(drugs):
            if d not in df.columns:
                raise KeyError(f"{d} not a column of {pheno_csv}")
            col = df[d].reindex(store.ids)
            v = col.astype("string").str.strip().str.upper()
            # a missing label is NA, not False, so fill before indexing
            Y[(v == "R").fillna(False).to_numpy(dtype=bool), j] = 1
            Y[(v == "S").fillna(False).to_numpy(dtype=bool), j] = 0
        self.Y = Y
        self.drugs = list(drugs)


def split_rows(store, split_file):
    """(train rows, test rows, [(train, val)] per fold) for a lineage split file."""
    sp = pd.read_csv(split_file, dtype={"New_ID": str}).drop_duplicates("New_ID")
    sp = sp[sp["New_ID"].isin(store.index)]
    rows = np.asarray([store.index[i] for i in sp["New_ID"]], np.int64)
    role, fold = sp["role"].values, sp["fold"].values
    train, test = rows[role == "train"], rows[role == "test"]
    tf = fold[role == "train"]
    folds = [(train[tf != k], train[tf == k]) for k in sorted(set(tf))]
    return train, test, folds
