"""
The GenomeFusion token: one changed base or one indel event, every channel a
named quantity.

This is LocusFusion v5's token (v3's residue one-hot + consequence + site
proximity, synonymous changes dropped) with v5's separate promoter-base slot
folded into one base slot -- `region` now says where the base is -- plus the
channels a whole-genome model needs.

    channel       width  what
    base            6    A C T G - N    the isolate's base
    residue        23    20 residues + STOP + deleted + unreadable (v3's slot)
    bio             3    MW / pI / hydrophobicity (z-scored) of that residue
    d_bio           3    bio minus H37Rv's residue at the same codon
    consequence     8    synonymous / missense / nonsense / frameshift /
                         inframe_indel / start_loss / stop_loss / unreadable;
                         all zero off a coding sequence
    truncation      1    fraction of the CDS after the event, for frameshift,
                         nonsense and start-loss; 0 otherwise
    region          3    coding / RNA gene / intergenic
    upstream        2    exp(-d / 50 bp) to the next start codon on the + and
                         the - strand; -1 = not upstream of a gene on that strand
    span            1    log1p(length) of a collapsed indel or uncalled run;
                         0 for a single base
    phase           3    position within the codon; zero off a codon
    flags           3    overlap / homopolymer / imprecise
    is_null         1    the per-locus [NULL] summary token
    ------------------
    C_BASE         57    + population (2) + one site_proximity per drug

`population` (homoplasy count, lineage-marker flag) is an optional arm; the
proximity channels are appended by the model at batch time from each token's
`site_pos`, exactly as in v4. A single-drug model with population channels is
57 + 2 + 1 = 60 wide.

The [NULL] token carries four per-locus numbers through `null_proj`:
log1p(token count), token count / locus length, net frame offset (/3) and the
fraction of the CDS read out of frame.
"""
import numpy as np

NT_SYMBOLS = ("A", "C", "T", "G", "-", "N")          # datasets.sequences order + N
RESIDUE_SYMBOLS = tuple("ACDEFGHIKLMNPQRSTVWY*-X")   # 20 + STOP + deleted + unreadable
CONSEQUENCES = ("synonymous", "missense", "nonsense", "frameshift",
                "inframe_indel", "start_loss", "stop_loss", "unreadable")
REGIONS = ("cds", "rna", "intergenic")
FLAGS = ("overlap", "homopolymer", "imprecise")
PROPERTY_NAMES = ("molecular_weight", "isoelectric_point", "hydrophobicity")

SLOTS = {
    "base": (0, 6),
    "residue": (6, 29),
    "bio": (29, 32),
    "d_bio": (32, 35),
    "consequence": (35, 43),
    "truncation": (43, 44),
    "region": (44, 47),
    "upstream": (47, 49),
    "span": (49, 50),
    "phase": (50, 53),
    "flags": (53, 56),
    "is_null": (56, 57),
}
C_BASE = 57
POPULATION_SLOT = (57, 59)       # homoplasy (log1p), lineage-marker flag
C_BASE_POP = 59

# consequence indices
SYN, MISSENSE, NONSENSE, FRAMESHIFT, INFRAME_INDEL, START_LOSS, STOP_LOSS, UNREADABLE = range(8)
# residue sentinels
R_STOP = RESIDUE_SYMBOLS.index("*")
R_GAP = RESIDUE_SYMBOLS.index("-")
R_UNK = RESIDUE_SYMBOLS.index("X")

UPSTREAM_LENGTH = 50.0           # bp; upstream channel = exp(-d / UPSTREAM_LENGTH)
UPSTREAM_MISSING = -1.0
DIST_LENGTH = 10.0               # A; proximity = exp(-d / DIST_LENGTH), as v4
DIST_MISSING = -1.0

N_NULL_STATS = 4                 # the [NULL] token's per-locus numbers


def c_base(population=False):
    return C_BASE_POP if population else C_BASE


def channel_names(dist_drugs=(), population=False):
    names = [f"base:{s}" for s in NT_SYMBOLS] + [f"residue:{r}" for r in RESIDUE_SYMBOLS]
    names += [f"bio:{p}" for p in PROPERTY_NAMES] + [f"d_bio:{p}" for p in PROPERTY_NAMES]
    names += [f"consequence:{c}" for c in CONSEQUENCES] + ["truncation"]
    names += [f"region:{r}" for r in REGIONS] + ["upstream:plus", "upstream:minus"]
    names += ["span"] + [f"phase:{i}" for i in range(3)] + [f"flag:{f}" for f in FLAGS]
    names += ["is_null"]
    if population:
        names += ["population:homoplasy", "population:lineage_marker"]
    names += [f"site_proximity:{d}" for d in dist_drugs]
    assert len(names) == c_base(population) + len(dist_drugs), (len(names), c_base(population))
    return names


# ---- the property table, as datasets.biochem defines it ---------------------
def residue_property_table():
    """(23, 3) float32: the three z-scored properties per residue symbol.

    STOP, deleted and unreadable are zero -- the same rule as v3/v4, where a
    residue with no biophysics contributes nothing rather than an average one.
    """
    from datasets.biochem import AA_PROPERTY, PROPERTY_NAMES as BIO_NAMES
    assert tuple(BIO_NAMES) == PROPERTY_NAMES, (BIO_NAMES, PROPERTY_NAMES)
    tab = np.zeros((len(RESIDUE_SYMBOLS), 3), np.float32)
    for i, r in enumerate(RESIDUE_SYMBOLS):
        if r in AA_PROPERTY:
            tab[i] = AA_PROPERTY[r]
    return tab
