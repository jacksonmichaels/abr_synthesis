"""
Derive per-region promoter windows (genome coordinates) for the regulatory
modality, straight from WHO 2023 catalogue Table 22 + the H37Rv GenBank
reference.

Design (confirmed with the user):
  - Regions = every WHO Table-21 candidate gene (any drug, tier 1 or 2) that has
    a Table-22 promoter entry, PLUS the PROMOTER_ALIASES below.
  - PROMOTER_ALIASES covers a gene whose promoter WHO delimits under a DIFFERENT
    name. fabG1 is the case that matters: Table 22 has no `fabG1` row, but its
    `inhA` row is keyed to tss=1673440, which is fabG1's OWN CDS start
    (Rv1483: 1673440-1674183; inhA/Rv1484 starts 762 bp later at 1674202). That
    row is therefore the fabG1-inhA OPERON promoter — the one carrying c-15t —
    filed under inhA. Emitting it a second time as `fabG1` makes the region
    reachable when inhA is not in the requested region set; the two windows are
    identical by construction, so do not load both and expect independent
    information.
  - Window = the Table-22 promoter region mapped to absolute NC_000962.3
    coordinates, PLUS 30 bp of flank each side (context for the CNN).
        pos strand : [tss - hi_off, tss - lo_off]   (upstream = lower coords)
        neg strand : [tss + lo_off, tss + hi_off]   (upstream = higher coords)
    where (lo_off, hi_off) is the enclosing span of the region segments.
  - Multi-segment regions (panD, Rv2752c, Rv3236c, tlyA): use the PRIMARY
    (first) upstream segment only. The distant secondary segments span the gene
    body, which would reintroduce coding sequence — kept out to keep the
    regulatory modality strictly upstream / disjoint from the DNA branch.
  - tss = None  : fall back to the GenBank CDS transcription-start end as a
                  proxy TSS (promoter = N bp upstream of the CDS start).
  - A gene with neither a TSS nor a resolvable Rv tag (rrl) is skipped.

Output: promoter_windows.json  {gene: {start, end, sense, ...}}  — 1-indexed,
inclusive, the exact -start/-end passed to make_aligned_msa.py.

Validated: embA -> [4243117, 4243262] (core [4243147,4243232] lands in the
embC..embA intergenic gap); eis core reproduces the known upstream window.
"""
import json
import os

from Bio import SeqIO

# import the project's catalogue tables (run with the repo on PYTHONPATH)
import sys
sys.path.insert(0, "/home/jacksonmicha_umass_edu/abr_workspace/biophysical-fusion")
from datasets import who_catalogue as who
from datasets.regulatory import DRUG_NAME_TO_WHO

PAD = 30

# gene -> the Table-22 entry that delimits ITS promoter (see module docstring).
# The window is emitted under the KEY's name, using the VALUE's WHO region/TSS.
PROMOTER_ALIASES = {"fabG1": "inhA"}
GENBANK = ("/project/pi_annagreen_umass_edu/saishradha/project_data_curation/"
           "make_msa/genbank_reference_data/genome.gbff")   # READ-ONLY
OUT = os.path.join(os.path.dirname(__file__), "promoter_windows.json")


def genbank_gene_coords(path):
    """locus_tag -> (start_1based, end_1based, strand +1/-1) for every gene."""
    rec = next(SeqIO.parse(path, "genbank"))
    tags = {}
    for f in rec.features:
        if f.type == "gene":
            lt = f.qualifiers.get("locus_tag", [None])[0]
            if lt:
                tags[lt] = (int(f.location.start) + 1, int(f.location.end),
                            1 if f.location.strand == 1 else -1)
    return tags, rec.id


def rv_of_gene():
    out = {}
    for e in who.TABLE_21.values():
        for g, rv in e["tier1"] + e["tier2"]:
            out[g] = rv
    return out


def regulatory_genes():
    """Distinct Table-21 candidate genes (any drug) with a Table-22 entry of
    their own or via PROMOTER_ALIASES."""
    genes = []
    for abbr in DRUG_NAME_TO_WHO.values():
        for g, _rv in who.genes_for_drug(abbr):
            if (g in who.TABLE_22 or g in PROMOTER_ALIASES) and g not in genes:
                genes.append(g)
    return genes


def window(gene, tags, rvof):
    # an aliased gene borrows the WHO region/TSS filed under another name, but
    # keeps its own Rv tag so the provenance in the JSON stays honest.
    source = PROMOTER_ALIASES.get(gene, gene)
    m = who.TABLE_22[source]
    segs, tss = m["region"], m["tss"]
    rv = rvof.get(gene)
    cds = tags.get(rv)
    strand = cds[2] if cds is not None else 1        # rRNA/unknown default: + strand
    proxy = False
    tss_used = tss
    if tss is None:
        if cds is None:
            return None                              # cannot derive (e.g. rrl)
        tss_used = cds[0] if strand == 1 else cds[1]  # CDS transcription-start end
        proxy = True
    lo, hi = segs[0]                                  # primary upstream segment only
    if strand == 1:
        start, end = tss_used - hi, tss_used - lo
    else:
        start, end = tss_used + lo, tss_used + hi
    return {
        "gene": gene, "rv": rv, "who_region_from": source,
        "start": int(start - PAD), "end": int(end + PAD),
        "sense": "pos" if strand == 1 else "neg",
        "core_start": int(start), "core_end": int(end),
        "tss": tss_used, "tss_is_proxy": proxy,
        "segments": segs, "primary_segment": list(segs[0]),
        "dropped_segments": [list(s) for s in segs[1:]], "pad": PAD,
        "length": int((end + PAD) - (start - PAD) + 1),
    }


def main():
    tags, chrom = genbank_gene_coords(GENBANK)
    rvof = rv_of_gene()
    genes = regulatory_genes()
    windows, skipped = {}, []
    for g in genes:
        w = window(g, tags, rvof)
        if w is None:
            skipped.append(g)
        else:
            windows[g] = w
    with open(OUT, "w") as f:
        json.dump({"chrom": chrom, "pad": PAD, "windows": windows,
                   "skipped": skipped}, f, indent=2)
    print(f"reference: {chrom}")
    print(f"{len(windows)} promoter windows written to {OUT}")
    print(f"skipped (no TSS + no Rv tag): {skipped}\n")
    proxies = [g for g, w in windows.items() if w["tss_is_proxy"]]
    print(f"CDS-start proxy TSS ({len(proxies)}): {proxies}")
    multseg = [g for g, w in windows.items() if len(w["segments"]) > 1]
    print(f"multi-segment, primary segment kept ({len(multseg)}): {multseg}\n")
    for g, w in sorted(windows.items(), key=lambda kv: kv[1]["start"]):
        flag = "  [proxy-TSS]" if w["tss_is_proxy"] else ""
        print(f"  {g:<9} {w['sense']}  [{w['start']}, {w['end']}]  "
              f"len={w['length']}{flag}")


if __name__ == "__main__":
    main()
