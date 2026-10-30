"""
Per-drug distance from every H37Rv position to that drug's binding site.

    python -m locusfusion_v4.binding_sites.build_distances      # from biophysical-fusion/

Writes, next to this file:
  distances.npz   one float32 array per (drug, locus), key "DRUG|locus", in A.
                  Protein loci are indexed by residue (0-based = the tokenizer's
                  codon index); rrs / rrl by nucleotide offset from the gene's
                  first base. NaN = no distance at that position.
  distances.json  what each array is the distance TO, from which structure, and
                  its indexing -- the provenance table for the README.

Two sources (the recipe of notebooks/binding_site_distance.ipynb §2 and §7):
  * protein sites (sites.json "protein_sites"): closest heavy-atom distance from
    each residue to any site residue, on the AlphaFold model of the H37Rv
    protein (structures/alphafold/{locus}.npz). 0 A for a site residue itself.
    Within one chain only: no complex, no ligand.
  * ribosome drugs (sites.json "ribosome"): closest heavy-atom distance from each
    nucleotide / residue to the DRUG's atoms in a drug-bound ribosome
    (structures/ribosome/raw), mapped to H37Rv numbering by pairwise alignment.

Every residue named in sites.json is asserted against the AlphaFold sequence.
Light: reads ~30 MB of structure files, runs in seconds on a login node.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                   # biophysical-fusion/.. = the workspace
ALPHAFOLD = REPO / "structures" / "alphafold"
RIBOSOME = REPO / "structures" / "ribosome" / "raw"
GENBANK = os.environ.get("ABR_GENBANK") or str(REPO / "data_snapshot" / "genbank" / "genome.gbff")


def _alphafold(locus):
    z = np.load(ALPHAFOLD / f"{locus}.npz")
    return str(z["ref_protein"]), z["heavy"]


def protein_site_distance(locus, residues):
    """(distance per residue in A, NaN where the residue has no atoms)."""
    seq, heavy = _alphafold(locus)
    site = []
    for tok in residues.split():
        aa, k = tok[0], int(tok[1:]) - 1
        if seq[k] != aa:
            raise ValueError(f"{locus} {tok}: H37Rv has {seq[k]} at {k + 1}")
        site.append(k)
    sa = heavy[site].reshape(-1, 3)
    sa = sa[~np.isnan(sa).any(1)]
    d = np.full(len(seq), np.nan, dtype=np.float32)
    for k in range(len(seq)):
        a = heavy[k][~np.isnan(heavy[k]).any(1)]
        if len(a):
            d[k] = np.sqrt(((a[:, None] - sa[None]) ** 2).sum(-1)).min()
    return d


@lru_cache(None)
def _read_chain(pdb, asym):
    """(canonical sequence, label_seq_id per heavy atom, xyz per heavy atom)."""
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    d = MMCIF2Dict(str(RIBOSOME / f"{pdb}_{asym}.cif"))
    seq = d.get("_entity_poly.pdbx_seq_one_letter_code_can", [""])
    seq = "".join((seq[0] if isinstance(seq, list) else seq).split())
    model = np.array(d["_atom_site.pdbx_PDB_model_num"])
    keep = (np.array(d["_atom_site.type_symbol"]) != "H") & (model == model[0])
    sid = np.array([int(x) if x not in ".?" else 0
                    for x in np.array(d["_atom_site.label_seq_id"])[keep]])
    xyz = np.stack([np.array(d[f"_atom_site.Cartn_{c}"], float)[keep] for c in "xyz"], 1)
    return seq, sid, xyz


@lru_cache(None)
def _h37rv_rrna(gene):
    from Bio import SeqIO
    for rec in SeqIO.parse(GENBANK, "genbank"):
        for f in rec.features:
            if f.type == "rRNA" and f.qualifiers.get("gene", [""])[0] == gene:
                return str(f.extract(rec.seq))
    raise KeyError(f"no rRNA feature {gene} in {GENBANK}")


def _align_map(ref, struct, protein):
    """H37Rv index (0-based) -> structure label_seq_id (0 = unaligned), identity."""
    from Bio import Align
    from Bio.Align import substitution_matrices
    a = Align.PairwiseAligner(mode="global")
    if protein:
        a.substitution_matrix = substitution_matrices.load("BLOSUM62")
        a.open_gap_score, a.extend_gap_score = -10, -0.5
    else:
        a.match_score, a.mismatch_score, a.open_gap_score, a.extend_gap_score = 2, -1, -5, -0.5
    a.end_gap_score = 0
    aln = a.align(ref, struct)[0]
    m, same = np.zeros(len(ref), int), []
    for (r0, r1), (s0, _) in zip(*aln.aligned):
        for k in range(r1 - r0):
            m[r0 + k] = s0 + k + 1
            same.append(ref[r0 + k].replace("T", "U") == struct[s0 + k].replace("T", "U"))
    return m, float(np.mean(same))


def ribosome_distances(spec):
    """{locus: distance per H37Rv position}, {locus: alignment identity}, drug copies used."""
    from scipy.spatial import cKDTree
    pdb, chains, ligs, anchor = spec["pdb"], spec["chains"], spec["ligands"], spec["anchor"]
    ref = {l: (_alphafold(l)[0] if l == "rpsL" else _h37rv_rrna(l)) for l in chains}
    data = {l: _read_chain(pdb, a) for l, a in chains.items()}
    maps = {l: _align_map(ref[l], data[l][0], l == "rpsL") for l in chains}
    tree = cKDTree(np.concatenate([data[l][2] for l in chains]))
    copies = {a: _read_chain(pdb, a)[2] for a in ligs}
    copies = {a: x for a, x in copies.items() if tree.query(x)[0].min() <= 8}  # bound here
    if anchor and len(copies) > 1:
        l, pos = anchor
        at = data[l][2][data[l][1] == maps[l][0][pos - 1]]
        best = min(copies, key=lambda a: cKDTree(copies[a]).query(at)[0].min())
        copies = {best: copies[best]}
    lig = cKDTree(np.concatenate(list(copies.values())))
    out = {}
    for l, (seq, sid, xyz) in data.items():
        per = np.full(len(seq) + 1, np.nan)
        np.fmin.at(per, sid, lig.query(xyz)[0])
        m = maps[l][0]
        out[l] = np.where(m > 0, per[m], np.nan).astype(np.float32)
    return out, {l: round(maps[l][1], 3) for l in chains}, sorted(copies)


def build(sites_path=HERE / "sites.json"):
    cfg = json.loads(Path(sites_path).read_text())
    arrays, prov = {}, {}
    for drug, spec in cfg["ribosome"].items():
        if drug.startswith("_"):
            continue
        dist, ident, used = ribosome_distances(spec)
        for l, d in dist.items():
            arrays[f"{drug}|{l}"] = d
            prov[f"{drug}|{l}"] = {
                "to": f"{drug.lower()} atoms", "structure": f"{spec['pdb']} (drug-bound)",
                "ligand_copies": used, "alignment_identity": ident[l],
                "index": "residue (0-based)" if l == "rpsL" else "nucleotide offset from gene start",
                "n": int(len(d)), "n_with_distance": int(np.isfinite(d).sum())}
    site_cache = {}
    for drug, loci in cfg["drug_protein_sites"].items():
        for l in loci:
            key = f"{drug}|{l}"
            if key in arrays:              # the ribosome distance wins (rpsL / STR)
                continue
            if l not in site_cache:
                site_cache[l] = protein_site_distance(l, cfg["protein_sites"][l]["residues"])
            arrays[key] = site_cache[l]
            s = cfg["protein_sites"][l]
            prov[key] = {"to": s["site"], "structure": f"AlphaFold; site from {s['source']}",
                         "index": "residue (0-based)", "n": int(len(site_cache[l])),
                         "n_with_distance": int(np.isfinite(site_cache[l]).sum())}
    np.savez_compressed(HERE / "distances.npz", **arrays)
    (HERE / "distances.json").write_text(json.dumps(prov, indent=1) + "\n")
    return arrays, prov


def load_distances(path=HERE / "distances.npz"):
    """{(drug, locus): float32 distance array in A} from distances.npz."""
    z = np.load(path)
    return {tuple(k.split("|")): z[k] for k in z.files}


if __name__ == "__main__":
    arrays, prov = build()
    for k, p in prov.items():
        d = arrays[k]
        print(f"{k:28s} n={p['n']:5d} within4A={int((d <= 4).sum()):4d} "
              f"median={np.nanmedian(d):6.1f}A  to {p['to']} ({p['structure']})")
