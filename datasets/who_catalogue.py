"""
WHO 2023 Catalogue of Mutations in Mycobacterium tuberculosis complex
(2nd edition) -- Tables 21 and 22, transcribed as Python data.

Table 21: Candidate resistance genes per drug (Tier 1 / Tier 2), with H37Rv locus tags.
Table 22: Upstream/promoter regions of candidate resistance genes, with the primary
          transcriptional start site (TSS) in absolute H37Rv coordinates (NC_000962.3).

Source: WHO/UCN/TB/2023.5, ISBN 9789240082410, Tables 21 & 22.

Notes on Table 21 transcription:
- The PDF omits the Rv locus tag for a gene when it was already introduced for an
  earlier drug (e.g. 'glpK', 'ndh', 'mmpL5'). Those tags are filled in here from
  their first appearance so every gene carries its tag.
- 'FQ' in Table 21 = fluoroquinolones (LFX & MFX in the interpretation tables).

Notes on Table 22 transcription:
- 'region' is the upstream/promoter extent, numbered relative to the gene start,
  counting upstream. It is NOT a genome interval and is strand-dependent: for genes
  on the minus strand ('c' suffix, e.g. Rv2416c) 'upstream' runs toward HIGHER
  genome coordinates. To convert to a genome (BED) interval you still need the gene
  strand + start from the H37Rv GFF / Mycobrowser.
- Some genes have two disjoint segments, stored as a list of (start, end) tuples,
  e.g. panD: [(1, 51), (1838, 1949)].
- 'tss' is the primary transcriptional start site (absolute H37Rv coordinate) where
  WHO listed one; None where the PDF left it blank or gave only a footnote.
- Footnote markers in the source (a-e) are recorded in 'note' where present.
"""

# ---------------------------------------------------------------------------
# TABLE 21 -- Candidate resistance genes
# ---------------------------------------------------------------------------
# Each gene is (gene_name, rv_tag). rv_tag is None only where the catalogue
# itself provides no Rv-style tag (none such here; all resolved).

TABLE_21 = {
    "INH": {
        "tier1": [
            ("fabG1", "Rv1483"), ("inhA", "Rv1484"), ("katG", "Rv1908c"),
            ("furA", "Rv1909c"), ("ahpC", "Rv2428"),
        ],
        "tier2": [
            ("dnaA", "Rv0001"), ("Rv0010c", "Rv0010c"), ("mshA", "Rv0486"),
            ("hadA", "Rv0635"), ("Rv1129c", "Rv1129c"), ("Rv1258c", "Rv1258c"),
            ("ndh", "Rv1854c"), ("Rv2752c", "Rv2752c"), ("glpK", "Rv3696c"),
        ],
        "references": "49,111,131-141",
    },
    "RIF": {
        "tier1": [
            ("rpoB", "Rv0667"),
        ],
        "tier2": [
            ("nusG", "Rv0639"), ("rpoC", "Rv0668"), ("Rv1129c", "Rv1129c"),
            ("Rv2477c", "Rv2477c"), ("Rv2752c", "Rv2752c"), ("lpqB", "Rv3244c"),
            ("mtrB", "Rv3245c"), ("mtrA", "Rv3246c"), ("rpoA", "Rv3457c"),
            ("glpK", "Rv3696c"),
        ],
        "references": "6,76,135,136,139-142",
    },
    "EMB": {
        "tier1": [
            ("aftA", "Rv3792"), ("embC", "Rv3793"), ("embA", "Rv3794"),
            ("embB", "Rv3795"), ("ubiA", "Rv3806c"),
        ],
        "tier2": [
            ("embR", "Rv1267c"), ("Rv2477c", "Rv2477c"), ("Rv2752c", "Rv2752c"),
            ("glpK", "Rv3696c"), ("aftB", "Rv3805c"),
        ],
        "references": "54,76,135,13-141,143",  # ref list as printed in source
    },
    "PZA": {
        "tier1": [
            ("pncA", "Rv2043c"), ("clpC1", "Rv3596c"), ("panD", "Rv3601c"),
        ],
        "tier2": [
            ("Rv1258c", "Rv1258c"), ("rpsA", "Rv1630"), ("sigE", "Rv1221"),
            ("PPE35", "Rv1918c"), ("Rv3236c", "Rv3236c"),
        ],
        "references": "6,137,139,144-146",
    },
    "FQ": {  # fluoroquinolones (LFX & MFX)
        "tier1": [
            ("gyrB", "Rv0005"), ("gyrA", "Rv0006"),
        ],
        "tier2": [
            ("Rv1129c", "Rv1129c"), ("Rv2477c", "Rv2477c"),
            ("Rv2752c", "Rv2752c"), ("glpK", "Rv3696c"),
        ],
        "references": "6,76,136,140,141",
    },
    "BDQ": {
        "tier1": [
            ("mmpL5", "Rv0676c"), ("mmpS5", "Rv0677c"), ("Rv0678", "Rv0678"),
            ("atpE", "Rv1305"), ("pepQ", "Rv2535c"),
        ],
        "tier2": [
            ("Rv1979c", "Rv1979c"), ("lpqB", "Rv3244c"),
            ("mtrB", "Rv3245c"), ("mtrA", "Rv3246c"),
        ],
        "references": "13,76",
    },
    "LZD": {
        "tier1": [
            ("rplC", "Rv0701"), ("rrl", "MTB000020"),
        ],
        "tier2": [
            ("tsnR", "Rv1644"),
        ],
        "references": "13,76",
    },
    "CFZ": {
        "tier1": [
            ("mmpL5", "Rv0676c"), ("mmpS5", "Rv0677c"), ("Rv0678", "Rv0678"),
            ("Rv1979c", "Rv1979c"), ("pepQ", "Rv2535c"),
        ],
        "tier2": [
            ("fgd1", "Rv0407"), ("fbiC", "Rv1173"), ("Rv2983", "Rv2983"),
            ("fbiA", "Rv3261"), ("fbiB", "Rv3262"),
        ],
        "references": "13,84,147",
    },
    "DLM": {
        "tier1": [
            ("fgd1", "Rv0407"), ("ddn", "Rv3547"), ("fbiC", "Rv1173"),
            ("Rv2983", "Rv2983"), ("fbiA", "Rv3261"), ("fbiB", "Rv3262"),
        ],
        "tier2": [
            ("ndh", "Rv1854c"),
        ],
        "references": "86,148",
    },
    "AMK": {
        "tier1": [
            ("rrs", "MTB000019"), ("eis", "Rv2416c"), ("whiB7", "Rv3197A"),
        ],
        "tier2": [
            ("ccsA", "Rv0529"), ("bacA", "Rv1819c"), ("Rv2477c", "Rv2477c"),
            ("whiB6", "Rv3862c"),
        ],
        "references": "15,76,139,149",
    },
    "STM": {
        "tier1": [
            ("rpsL", "Rv0682"), ("Rv1258c", "Rv1258c"), ("rrs", "MTB000019"),
            ("whiB7", "Rv3197A"), ("gid", "Rv3919c"),
        ],
        "tier2": [
            ("bacA", "Rv1819c"), ("Rv2477c", "Rv2477c"), ("glpK", "Rv3696c"),
        ],
        "references": "6,76,137,149,150",
    },
    "ETO": {
        "tier1": [
            ("mshA", "Rv0486"), ("fabG1", "Rv1483"), ("inhA", "Rv1484"),
            ("ethA", "Rv3854c"),
        ],
        "tier2": [
            ("Rv0565c", "Rv0565c"), ("ndh", "Rv1854c"), ("Rv3083", "Rv3083"),
            ("ethR", "Rv3855"),
        ],
        "references": "49,97,133,134,138,139,151,152",
    },
    "KAN": {
        "tier1": [
            ("rrs", "MTB000019"), ("eis", "Rv2416c"), ("whiB7", "Rv3197A"),
        ],
        "tier2": [
            ("ccsA", "Rv0529"), ("bacA", "Rv1819c"), ("Rv2477c", "Rv2477c"),
            ("whiB6", "Rv3862c"),
        ],
        "references": "6,76,139,149",
    },
    "CAP": {
        "tier1": [
            ("rrs", "MTB000019"), ("tlyA", "Rv1694"),
        ],
        "tier2": [
            ("ccsA", "Rv0529"), ("rrl", "MTB000020"), ("bacA", "Rv1819c"),
            ("Rv2680", "Rv2680"), ("Rv2681", "Rv2681"), ("whiB6", "Rv3862c"),
        ],
        "references": "6,76,135,139,153",
    },
}


# ---------------------------------------------------------------------------
# TABLE 22 -- Upstream/promoter regions of candidate resistance genes
# ---------------------------------------------------------------------------
# 'region' : list of (start, end) segments, numbered upstream-relative to the gene.
# 'tss'    : primary transcriptional start site, absolute H37Rv coordinate, or None.
# 'ref'    : reference number as printed, or None.
# 'note'   : source footnote marker (a-e) where present, else None.

TABLE_22 = {
    "aftB":    {"region": [(1, 129)],              "tss": 4268914, "ref": "154", "note": None},
    "ahpC":    {"region": [(1, 93)],               "tss": 2726151, "ref": "131", "note": None},
    "atpE":    {"region": [(1, 51)],               "tss": 1461045, "ref": "154", "note": None},
    "bacA":    {"region": [(1, 81)],               "tss": 2064758, "ref": "155", "note": None},
    "ccsA":    {"region": [(1, 191)],              "tss": 619751,  "ref": "155", "note": None},
    "clpC1":   {"region": [(1, 106)],              "tss": 4040759, "ref": "155", "note": None},
    "ddn":     {"region": [(1, 51)],               "tss": 3986844, "ref": "155", "note": None},
    "dnaA":    {"region": [(1, 314)],              "tss": 4411270, "ref": "155", "note": None},
    "eis":     {"region": [(1, 84)],               "tss": 2715365, "ref": "156", "note": None},
    "embA":    {"region": [(1, 86)],               "tss": 4243233, "ref": "155", "note": None},
    "embC":    {"region": [(1, 1982)],             "tss": None,    "ref": None,  "note": "a"},
    "embR":    {"region": [(1, 103)],              "tss": 1417399, "ref": "155", "note": None},
    "ethA":    {"region": [(1, 51)],               "tss": 4327473, "ref": "155", "note": None},
    "ethR":    {"region": [(1, 26)],               "tss": 4327505, "ref": "154", "note": None},
    "fbiA":    {"region": [(1, 138)],              "tss": 3640456, "ref": "155", "note": None},
    "fbiC":    {"region": [(1, 127)],              "tss": 1302855, "ref": "155", "note": None},
    "fgd1":    {"region": [(1, 51)],               "tss": 490783,  "ref": "155", "note": None},
    "gid":     {"region": [(1, 79)],               "tss": 4408230, "ref": "157", "note": None},
    "glpK":    {"region": [(1, 52)],               "tss": 4139756, "ref": "155", "note": None},
    "gyrA":    {"region": [(1, 35)],               "tss": None,    "ref": None,  "note": None},
    "gyrB":    {"region": [(1, 108)],              "tss": 5183,    "ref": "155", "note": None},
    "hadA":    {"region": [(1, 51)],               "tss": 731930,  "ref": "155", "note": None},
    "inhA":    {"region": [(1, 813)],              "tss": 1673440, "ref": "155", "note": "b"},
    "katG":    {"region": [(1, 532)],              "tss": 2156592, "ref": "158", "note": "c"},
    "mmpS5":   {"region": [(1, 85)],               "tss": 778965,  "ref": "155", "note": "d"},
    "mshA":    {"region": [(1, 669)],              "tss": 574730,  "ref": "155", "note": None},
    "mtrA":    {"region": [(1, 376)],              "tss": 3627674, "ref": "155", "note": None},
    "mtrB":    {"region": [(1, 50)],               "tss": 3626821, "ref": "155", "note": None},
    "ndh":     {"region": [(1, 96)],               "tss": 2103087, "ref": "155", "note": None},
    "nusG":    {"region": [(1, 201)],              "tss": 734104,  "ref": "155", "note": None},
    "panD":    {"region": [(1, 51), (1838, 1949)], "tss": 4046179, "ref": "155", "note": None},
    "pepQ":    {"region": [(1, 51)],               "tss": 2860418, "ref": "155", "note": None},
    "pncA":    {"region": [(1, 51)],               "tss": 2289241, "ref": "155", "note": None},
    "PPE35":   {"region": [(1, 122)],              "tss": 2170683, "ref": "155", "note": None},
    "rplC":    {"region": [(1, 51), (323, 503)],   "tss": 800357,  "ref": "154", "note": None},
    "rpoA":    {"region": [(1, 536)],              "tss": 3878992, "ref": "154", "note": None},
    "rpoB":    {"region": [(1, 263)],              "tss": 759595,  "ref": "155", "note": None},
    "rpoC":    {"region": [(1, 45)],               "tss": None,    "ref": None,  "note": None},
    "rpsA":    {"region": [(1, 100)],              "tss": 1833493, "ref": "155", "note": None},
    "rpsL":    {"region": [(1, 234)],              "tss": 781377,  "ref": "155", "note": None},
    "rrl":     {"region": [(1, 51)],               "tss": None,    "ref": None,  "note": None},
    "rrs":     {"region": [(1, 151)],              "tss": 1471746, "ref": "3",   "note": None},
    "Rv0010c": {"region": [(1, 156)],              "tss": None,    "ref": None,  "note": None},
    "Rv0565c": {"region": [(1, 78)],               "tss": 657497,  "ref": "155", "note": None},
    "Rv1129c": {"region": [(1, 51)],               "tss": 1254510, "ref": "154", "note": None},
    "Rv1258c": {"region": [(1, 58)],               "tss": 1407347, "ref": "155", "note": None},
    "Rv1979c": {"region": [(1, 470)],              "tss": 2223583, "ref": "154", "note": None},
    "Rv2477c": {"region": [(1, 88)],               "tss": 2784079, "ref": "155", "note": None},
    "Rv2680":  {"region": [(1, 153)],              "tss": 2996003, "ref": "155", "note": None},
    "Rv2681":  {"region": [(1, 2)],                "tss": None,    "ref": None,  "note": None},
    "Rv2752c": {"region": [(1, 51), (934, 984)],   "tss": 3067124, "ref": "155", "note": None},
    "Rv2983":  {"region": [(1, 51)],               "tss": 3339118, "ref": "155", "note": None},
    "Rv3083":  {"region": [(1, 51)],               "tss": 3448504, "ref": "155", "note": None},
    "Rv3236c": {"region": [(1, 51), (488, 538)],   "tss": 3613603, "ref": "155", "note": None},
    "sigE":    {"region": [(1, 51)],               "tss": 1364413, "ref": "155", "note": None},
    "tlyA":    {"region": [(1, 51), (185, 236)],   "tss": 1917755, "ref": "155", "note": None},
    "tsnR":    {"region": [(1, 51)],               "tss": None,    "ref": None,  "note": None},
    "ubiA":    {"region": [(1, 51)],               "tss": None,    "ref": None,  "note": None},
    "whiB6":   {"region": [(1, 126)],              "tss": 4338596, "ref": None,  "note": "e"},
    "whiB7":   {"region": [(1, 404)],              "tss": 3569032, "ref": "159", "note": None},
}


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def genes_for_drug(drug, tier=None):
    """Return [(gene, rv_tag), ...] for a drug. tier in {1, 2, None(=both)}."""
    d = TABLE_21[drug]
    if tier == 1:
        return list(d["tier1"])
    if tier == 2:
        return list(d["tier2"])
    return list(d["tier1"]) + list(d["tier2"])


def drug_gene_promoter_join():
    """
    Join Table 21 and Table 22: yields one dict per (drug, gene) pair that has a
    promoter entry, carrying the Rv tag, tier, and the promoter region + TSS.
    This is the drug -> coding-gene -> regulatory-region mapping.
    """
    rows = []
    for drug, d in TABLE_21.items():
        for tier_num, key in ((1, "tier1"), (2, "tier2")):
            for gene, rv in d[key]:
                prom = TABLE_22.get(gene)
                if prom is None:
                    continue
                rows.append({
                    "drug": drug,
                    "gene": gene,
                    "rv_tag": rv,
                    "tier": tier_num,
                    "promoter_region": prom["region"],
                    "tss": prom["tss"],
                    "promoter_ref": prom["ref"],
                    "promoter_note": prom["note"],
                })
    return rows


if __name__ == "__main__":
    print(f"Table 21: {len(TABLE_21)} drugs")
    print(f"Table 22: {len(TABLE_22)} genes with promoter regions")
    joined = drug_gene_promoter_join()
    print(f"Joined drug-gene-promoter rows: {len(joined)}\n")
    print("Example -- INH genes with promoter data:")
    for r in joined:
        if r["drug"] == "INH":
            print(f"  {r['gene']:8s} {r['rv_tag']:9s} tier{r['tier']} "
                  f"region={r['promoter_region']} TSS={r['tss']}")
