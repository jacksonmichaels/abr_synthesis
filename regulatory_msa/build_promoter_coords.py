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
  - Table-22 offsets count upstream from the gene's START CODON (c.-1 is the
    base before the ATG), not from the TSS: the far end of a region is almost
    always (TSS-to-ATG distance + 51), i.e. WHO's promoter runs from the ATG to
    50 bp past the TSS, and the WHO variant table's c.-N promoter mutations
    sit inside it. The TSS is kept in the JSON for provenance only.
  - Window = the Table-22 region mapped to absolute NC_000962.3 coordinates,
    PLUS 30 bp of flank each side (context for the CNN).
        pos strand : [atg - hi_off, atg - lo_off]   (upstream = lower coords)
        neg strand : [atg + lo_off, atg + hi_off]   (upstream = higher coords)
    where (lo_off, hi_off) is the enclosing span of the region segments, so a
    two-segment region (panD, Rv2752c, Rv3236c, tlyA, rplC) also keeps the
    bases between its segments.
  - dnaA starts at position 1, so its upstream region is the END of the
    circular chromosome; the window is placed there and its downstream flank
    (which would cross the origin) is clamped off.
  - Genes are anchored by Rv tag, falling back to the GenBank gene name for
    the rRNAs (WHO tags rrs/rrl MTB000019/20; GenBank uses Rvnr01/02).

SUPERSEDES the 2026-07/08 TSS-anchored build (promoter_windows.json ->
aligned/), which applied the same offsets from the TSS and so shifted every
window whose TSS is not the ATG upstream by that distance (katG 481 bp, rpoB
212, ...), and dropped second segments. That JSON and aligned/ are left as-is
because existing results were trained on them.

Output: promoter_windows_atg.json  {gene: {start, end, sense, ...}}  —
1-indexed, inclusive, the exact -start/-end passed to make_aligned_msa.py.
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

# WHO drugs outside DRUG_NAME_TO_WHO (no usable phenotype labels in BIG-TB)
# whose candidate genes still get a promoter window, so the new/repurposed-drug
# loci curated 2026-10-15 (atpE, ddn, rplC, ...) have a regulatory region.
# Appended after the project drugs, so every earlier window is unchanged.
EXTRA_WHO_DRUGS = ("BDQ", "LZD", "CFZ", "DLM")
GENBANK = ("/project/pi_annagreen_umass_edu/saishradha/project_data_curation/"
           "make_msa/genbank_reference_data/genome.gbff")   # READ-ONLY
OUT = os.path.join(os.path.dirname(__file__), "promoter_windows_atg.json")


def genbank_gene_coords(path):
    """locus_tag -> (start_1based, end_1based, strand +1/-1) for every gene,
    plus "name:<gene>" keys for genes whose WHO tag is not a GenBank locus_tag
    (rrs/rrl: WHO MTB000019/20, GenBank Rvnr01/02)."""
    rec = next(SeqIO.parse(path, "genbank"))
    tags = {}
    for f in rec.features:
        if f.type == "gene":
            coords = (int(f.location.start) + 1, int(f.location.end),
                      1 if f.location.strand == 1 else -1)
            lt = f.qualifiers.get("locus_tag", [None])[0]
            if lt:
                tags[lt] = coords
            name = f.qualifiers.get("gene", [None])[0]
            if name:
                tags[f"name:{name}"] = coords
    return tags, rec.id, len(rec.seq)


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
    for abbr in list(DRUG_NAME_TO_WHO.values()) + list(EXTRA_WHO_DRUGS):
        for g, _rv in who.genes_for_drug(abbr):
            if (g in who.TABLE_22 or g in PROMOTER_ALIASES) and g not in genes:
                genes.append(g)
    return genes


def window(gene, tags, rvof, genome_len):
    # an aliased gene borrows the WHO region filed under another name, and is
    # anchored on THAT gene's start codon (the offsets are numbered from it);
    # it keeps its own Rv tag in the JSON so the provenance stays honest.
    source = PROMOTER_ALIASES.get(gene, gene)
    m = who.TABLE_22[source]
    segs = m["region"]
    rv = rvof.get(gene)
    cds = tags.get(rvof.get(source)) or tags.get(f"name:{source}")
    if cds is None:
        return None                                   # cannot anchor
    strand = cds[2]
    atg = cds[0] if strand == 1 else cds[1]
    lo = min(a for a, _ in segs)                      # enclosing span of all segments
    hi = max(b for _, b in segs)
    if strand == 1:
        start, end = atg - hi, atg - lo
    else:
        start, end = atg + lo, atg + hi
    if end < 1:
        # upstream of a gene at the origin (dnaA): the region lies at the END
        # of the circular chromosome. Shift it there; the downstream flank
        # would wrap into the gene body across the origin and is dropped.
        start, end = start + genome_len, end + genome_len
    w_start, w_end = max(1, start - PAD), min(genome_len, end + PAD)
    return {
        "gene": gene, "rv": rv, "who_region_from": source,
        "start": int(w_start), "end": int(w_end),
        "sense": "pos" if strand == 1 else "neg",
        "core_start": int(start), "core_end": int(end),
        "anchor": "start_codon", "atg": int(atg), "tss": m["tss"],
        "segments": segs, "pad": PAD,
        "clamped": bool(w_start != start - PAD or w_end != end + PAD),
        "length": int(w_end - w_start + 1),
    }


def main():
    tags, chrom, genome_len = genbank_gene_coords(GENBANK)
    rvof = rv_of_gene()
    genes = regulatory_genes()
    windows, skipped = {}, []
    for g in genes:
        w = window(g, tags, rvof, genome_len)
        if w is None:
            skipped.append(g)
        else:
            windows[g] = w
    with open(OUT, "w") as f:
        json.dump({"chrom": chrom, "pad": PAD, "windows": windows,
                   "skipped": skipped}, f, indent=2)
    print(f"reference: {chrom}")
    print(f"{len(windows)} promoter windows written to {OUT}")
    print(f"skipped (no Rv tag in GenBank): {skipped}\n")
    clamped = [g for g, w in windows.items() if w["clamped"]]
    print(f"clamped at a chromosome end ({len(clamped)}): {clamped}")
    multseg = [g for g, w in windows.items() if len(w["segments"]) > 1]
    print(f"multi-segment, enclosing span kept ({len(multseg)}): {multseg}\n")
    for g, w in sorted(windows.items(), key=lambda kv: kv[1]["start"]):
        flag = "  [clamped]" if w["clamped"] else ""
        print(f"  {g:<9} {w['sense']}  [{w['start']}, {w['end']}]  "
              f"len={w['length']}{flag}")


if __name__ == "__main__":
    main()
