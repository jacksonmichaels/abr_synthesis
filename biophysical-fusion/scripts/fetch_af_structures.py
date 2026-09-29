"""Download AlphaFold DB models for every protein-coding locus and put their
C-alpha coordinates on the H37Rv codon grid ``locusfusion_v2`` tokenizes on.

    python scripts/fetch_af_structures.py                 # all coding loci on disk
    python scripts/fetch_af_structures.py --loci katG pncA
    python scripts/fetch_af_structures.py --force         # re-download

Writes ``bigtb_ref.REAL_STRUCTURE_DIR/{locus}.npz`` plus ``manifest.json``, and
keeps the raw model files under ``raw/`` so every number can be re-derived.

The grid. A ``locusfusion_v2`` token's ``codon`` is k = (reference bases from
the start codon) // 3, i.e. H37Rv CDS nucleotides [3k, 3k+3). So codon k is
residue k+1 of the H37Rv protein, and the terminal stop is codon n_codons-1 with
no residue. Every array written here is indexed by that k, length n_codons, and
NaN where there is no coordinate (the stop codon, or a residue the model does
not cover).

The mapping. The H37Rv protein is translated from the SAME GenBank record the
aligned FASTAs were cut from (``datasets.cds.GENBANK``), so it is the sequence
the codon grid is defined on. The AlphaFold model is of the UniProt sequence.
When the two are identical (the expected case) residue i maps to codon i-1.
Otherwise they are globally aligned and only identical aligned pairs are mapped;
the manifest records the identity and the mapped fraction per locus.

rRNA loci (rrs, rrl) have no CDS and are skipped: their tokens are masked.
"""
import argparse
import json
import sys
import time
import urllib.parse
from pathlib import Path

import numpy as np
import requests
from Bio import Align, SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bigtb_ref import REAL_GENOTYPE_DIR, REAL_STRUCTURE_DIR  # noqa: E402
from datasets.cds import GENBANK  # noqa: E402
from datasets.loader import loci_on_disk  # noqa: E402

UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
AFDB = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
TAXON = 83332                   # M. tuberculosis H37Rv


def genbank_cds():
    """gene -> {locus_tag, nt, protein} for every CDS in the H37Rv record.

    ``protein`` is translated here, literally (table 11), rather than read off
    /translation, so it is exactly the codon grid of ``nt``: a GTG/TTG start
    reads V/L, as it does in the tokenizer's reference-frame protein block."""
    rec = next(SeqIO.parse(GENBANK, "genbank"))
    out = {}
    for f in rec.features:
        if f.type != "CDS":
            continue
        nt = f.extract(rec.seq)
        tag = f.qualifiers.get("locus_tag", [None])[0]
        prot = str(nt.translate(table=11, to_stop=False))
        for key in ("gene", "locus_tag"):
            name = f.qualifiers.get(key, [None])[0]
            if name and name not in out:
                out[name] = {"locus_tag": tag, "nt": str(nt), "protein": prot}
    return out


def uniprot_candidates(locus_tag):
    """[(accession, reviewed, sequence)] for an H37Rv locus tag, reviewed first."""
    q = f"gene:{locus_tag} AND organism_id:{TAXON}"
    r = requests.get(UNIPROT, params={"query": q, "fields": "accession,reviewed,sequence",
                                      "format": "json", "size": 10}, timeout=60)
    r.raise_for_status()
    hits = [(e["primaryAccession"], e["entryType"].startswith("UniProtKB reviewed"),
             e["sequence"]["value"]) for e in r.json().get("results", [])]
    return sorted(hits, key=lambda h: not h[1])


def afdb_model(acc, raw_dir, force=False):
    """(pdb_path, api_entry) for an accession's AlphaFold DB model, or None."""
    r = requests.get(AFDB.format(acc=acc), timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    entry = r.json()[0]
    url = entry["pdbUrl"]
    path = raw_dir / Path(urllib.parse.urlparse(url).path).name
    if force or not path.exists():
        f = requests.get(url, timeout=120)
        f.raise_for_status()
        path.write_bytes(f.content)
    return path, entry


THREE_TO_ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
                "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
                "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
                "TYR": "Y", "VAL": "V"}


def read_ca(pdb_path):
    """(sequence, ca (L,3), plddt (L,)) from the C-alpha ATOM records.

    AlphaFold writes pLDDT into the B-factor column, one model, one chain."""
    seq, xyz, bf = [], [], []
    for line in Path(pdb_path).read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            seq.append(THREE_TO_ONE.get(line[17:20], "X"))
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            bf.append(float(line[60:66]))
    return "".join(seq), np.asarray(xyz, np.float32), np.asarray(bf, np.float32)


def _same(ref, i, model, j):
    """Residue identity, where a GTG/TTG start codon (V/L at codon 0) counts as
    the Met the protein actually begins with."""
    return ref[i] == model[j] or (i == 0 and ref[0] in "VL" and model[j] == "M")


def map_to_grid(ref, model):
    """codon index per model residue (-1 unmapped), and the aligned identity.

    ``ref`` is the literal translation. Identical sequences (start codon aside)
    map one-to-one. Otherwise a global alignment with free end gaps, on the
    literal residues -- forcing codon 0 to Met first would let it pair with an
    upstream Met from a longer annotation (rpoB: UniProt starts 6 residues
    before this GenBank record's TTG, whose Leu is the model's residue 7). Only
    aligned pairs with the SAME residue are mapped -- a mismatch means the model
    is of a different sequence there, so its coordinate is not H37Rv's."""
    if len(ref) == len(model) and all(_same(ref, i, model, i) for i in range(len(ref))):
        return np.arange(len(model)), 1.0
    al = Align.PairwiseAligner(mode="global", open_gap_score=-10, extend_gap_score=-0.5,
                               end_gap_score=0,
                               substitution_matrix=Align.substitution_matrices.load("BLOSUM62"))
    a = al.align(ref, model)[0]
    idx = np.full(len(model), -1)
    same = aligned = 0
    for (r0, r1), (m0, m1) in zip(*a.aligned):
        for i, j in zip(range(r0, r1), range(m0, m1)):
            aligned += 1
            if _same(ref, i, model, j):
                idx[j] = i
                same += 1
    return idx, same / max(aligned, 1)


def build_locus(locus, cds, raw_dir, out_dir, force=False):
    info = cds.get(locus)
    if info is None:
        return {"locus": locus, "status": "no CDS (rRNA or unannotated): masked"}
    ref = info["protein"].rstrip("*")
    n_codons = len(info["nt"]) // 3
    row = {"locus": locus, "locus_tag": info["locus_tag"], "n_codons": n_codons,
           "ref_length": len(ref), "internal_stops": ref.count("*")}

    hit = None
    met = "M" + ref[1:]                         # the protein as expressed
    for acc, reviewed, useq in uniprot_candidates(info["locus_tag"]):
        got = afdb_model(acc, raw_dir, force)
        if got is not None:
            hit = (acc, reviewed, useq, *got)
            if useq == met:
                break                           # exact H37Rv sequence: take it
        time.sleep(0.2)
    if hit is None:
        row["status"] = "no AlphaFold DB model"
        return row
    acc, reviewed, useq, pdb_path, entry = hit

    mseq, ca_m, pl_m = read_ca(pdb_path)
    idx, identity = map_to_grid(ref, mseq)
    ca = np.full((n_codons, 3), np.nan, np.float32)
    plddt = np.full(n_codons, np.nan, np.float32)
    ok = idx >= 0
    ca[idx[ok]] = ca_m[ok]
    plddt[idx[ok]] = pl_m[ok]
    model_residue = np.full(n_codons, "", dtype="<U1")
    model_residue[idx[ok]] = np.array(list(mseq))[ok]

    np.savez_compressed(
        out_dir / f"{locus}.npz", ca=ca, plddt=plddt, ref_protein=ref,
        model_residue=model_residue, uniprot=acc, model_id=entry["modelEntityId"],
        afdb_version=int(entry["latestVersion"]), pdb_file=pdb_path.name)
    row.update(status="ok", uniprot=acc, reviewed=reviewed, model_id=entry["modelEntityId"],
               afdb_version=int(entry["latestVersion"]), model_length=len(mseq),
               sequence_identical=(mseq == met), aligned_identity=round(identity, 4),
               mapped_codons=int(ok.sum()), mapped_frac=round(ok.sum() / max(len(ref), 1), 4),
               mean_plddt=round(float(np.nanmean(plddt)), 2),
               global_plddt=entry.get("globalMetricValue"))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--loci", nargs="*", default=None,
                    help="default: every locus with an aligned FASTA in REAL_GENOTYPE_DIR")
    ap.add_argument("--out", default=REAL_STRUCTURE_DIR)
    ap.add_argument("--force", action="store_true", help="re-download model files")
    args = ap.parse_args()

    out_dir = Path(args.out)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    loci = args.loci or loci_on_disk(REAL_GENOTYPE_DIR)
    cds = genbank_cds()

    rows = []
    for locus in loci:
        row = build_locus(locus, cds, raw_dir, out_dir, args.force)
        rows.append(row)
        print(f"{locus:>7}  {row['status']:<40} "
              + (f"{row['uniprot']}  identical={row['sequence_identical']}  "
                 f"mapped={row['mapped_codons']}/{row['ref_length']}  "
                 f"pLDDT={row['mean_plddt']}" if row["status"] == "ok" else ""))

    manifest = out_dir / "manifest.json"
    old = json.loads(manifest.read_text()) if manifest.exists() else {}
    old.update({r["locus"]: r for r in rows})
    manifest.write_text(json.dumps(old, indent=1))
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
