"""
The LocusFusion token: one changed base pair, every channel a named quantity.

Fixed layout, the same for every run. The binding-site distance channels are
not in this table: there is one per drug the model reads distances for (one for
a single-drug model, eleven for the multi-drug model), appended after C_BASE by
the model at batch time from each token's `site_pos` (see models/locusfusion.py).

    channel   width  what
    dna         6    A C T G - N    the isolate's base, at a gene-record column
    regulatory  6    A C T G - N    the isolate's base, at a promoter column
    bio         3    MW / pI / hydrophobicity (z-scored) of the residue the
                     isolate's codon now reads; 0 for STOP, a deleted codon and
                     an unreadable one
    d_bio       3    bio minus H37Rv's residue at that codon
    consequence 4    synonymous / missense / nonsense / other; all zero for a
                     non-coding token (promoter, flank, rRNA)
    phase       3    position within the codon (0/1/2); zero off a codon
    is_null     1    the per-locus [NULL] summary token
    residue    23    one-hot of the residue the codon now reads: 20 residues,
                     STOP, deleted codon, unreadable (tokenizer.RESIDUES order)
    ---------------
    C_BASE     49    + n_dist distance channels: exp(-d / 10 A), -1 = no distance
"""
NT_SYMBOLS = ("A", "C", "T", "G", "-", "N")          # datasets.sequences order + N
CONSEQUENCES = ("synonymous", "missense", "nonsense", "other")
PROPERTY_NAMES = ("molecular_weight", "isoelectric_point", "hydrophobicity")
RESIDUE_SYMBOLS = tuple("ACDEFGHIKLMNPQRSTVWY*-X")   # 20 + STOP + deleted + unreadable

SLOTS = {
    "dna": (0, 6),
    "regulatory": (6, 12),
    "bio": (12, 15),
    "d_bio": (15, 18),
    "consequence": (18, 22),
    "phase": (22, 25),
    "is_null": (25, 26),
    "residue": (26, 49),
}
C_BASE = 49
SYN, MISSENSE, NONSENSE, OTHER = range(4)        # index within the consequence slot

# binding-site distance -> channel value
DIST_LENGTH = 10.0      # A; proximity = exp(-d / DIST_LENGTH)
DIST_MISSING = -1.0     # no distance: promoter, no site on this locus for this drug, ...


def channel_names(dist_drugs=()):
    """Every channel's name, in order, for a model reading `dist_drugs`."""
    names = [f"dna:{s}" for s in NT_SYMBOLS] + [f"regulatory:{s}" for s in NT_SYMBOLS]
    names += [f"bio:{p}" for p in PROPERTY_NAMES] + [f"d_bio:{p}" for p in PROPERTY_NAMES]
    names += [f"consequence:{c}" for c in CONSEQUENCES]
    names += [f"phase:{i}" for i in range(3)] + ["is_null"]
    names += [f"residue:{r}" for r in RESIDUE_SYMBOLS]
    names += [f"site_proximity:{d}" for d in dist_drugs]
    assert len(names) == C_BASE + len(dist_drugs)
    return names
