#!/usr/bin/env python3
"""Gene-body windows for the WHO 2023 Table-21 tier 1 + 2 candidate genes of the
project's 11 drugs that have no aligned FASTA yet -> gene_windows.json.

Window = the GenBank `gene` feature +/- PAD bp (clamped to the genome), the same
cut as the existing 19 records (gene +/- 100 bp). `sense` is the gene's strand, so
the extractor writes each record 5'->3' like the existing minus-strand loci.

    ~/.conda/envs/abr_env/bin/python build_gene_windows.py
"""
import json, os, sys
from Bio import SeqIO

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "biophysical-fusion"))
from datasets.who_catalogue import genes_for_drug            # noqa: E402
from datasets.regulatory import DRUG_NAME_TO_WHO             # noqa: E402

GENBANK = ("/project/pi_annagreen_umass_edu/saishradha/project_data_curation/"
           "make_msa/genbank_reference_data/genome.gbff")
EXISTING = "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/genomic_data/aligned"
DRUGS = ["ISONIAZID", "RIFAMPICIN", "ETHAMBUTOL", "PYRAZINAMIDE", "STREPTOMYCIN", "KANAMYCIN",
         "AMIKACIN", "CAPREOMYCIN", "LEVOFLOXACIN", "MOXIFLOXACIN", "ETHIONAMIDE"]
PAD = 100

have = {f[:-6] for f in os.listdir(EXISTING) if f.endswith(".fasta")}
want = {}
for d in DRUGS:
    for gene, rv in genes_for_drug(DRUG_NAME_TO_WHO[d]):
        if gene not in have:
            want.setdefault(gene, {"rv": rv, "drugs": []})["drugs"].append(d)

rec = next(SeqIO.parse(GENBANK, "genbank"))
L = len(rec.seq)
by_name = {}
for f in rec.features:
    if f.type != "gene":
        continue
    for key in ("gene", "locus_tag"):
        n = f.qualifiers.get(key, [None])[0]
        if n:
            by_name.setdefault(n, f)

windows, missing = {}, []
for gene, info in sorted(want.items()):
    f = by_name.get(gene) or by_name.get(info["rv"])
    if f is None:
        missing.append(gene); continue
    a, b = int(f.location.start), int(f.location.end)          # 0-based half-open
    s, e = max(0, a - PAD), min(L, b + PAD)
    windows[gene] = {"gene": gene, "rv": info["rv"], "drugs": info["drugs"],
                     "gene_start0": a, "gene_end0": b,
                     # extractor convention: -start 1-based inclusive, -end 1-based inclusive
                     "start": s + 1, "end": e, "sense": "pos" if f.location.strand == 1 else "neg",
                     "length": e - s, "pad": PAD}
json.dump({"chrom": rec.id, "pad": PAD, "windows": windows},
          open(os.path.join(HERE, "gene_windows.json"), "w"), indent=1)
print(f"{len(windows)} gene windows -> gene_windows.json; not found in GenBank: {missing}")
for g, w in windows.items():
    print(f"  {g:8} {w['rv']:8} {w['sense']}  [{w['start']}, {w['end']}]  {w['length']} bp  {','.join(x[:4] for x in w['drugs'])}")
