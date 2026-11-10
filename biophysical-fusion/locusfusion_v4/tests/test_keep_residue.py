"""
`--keep-residue` puts v3's residue one-hot back on the v4 token, unchanged.

    cd biophysical-fusion
    python -m locusfusion_v4.tests.test_keep_residue      # ~1 min, ~3 GB

The v4 sweep (2026-10-28) found that replacing v3's 23-channel residue one-hot
with the 4-channel consequence flag was a wash on CV, so `--keep-residue` keeps
both. This checks the restored channel against v3's own `_locus_tokens` on one
locus -- pncA alone, so the loader fits in a few GB, unlike test_parity's five.

Needs the real data (ABR_GENOTYPE_DIR / ABR_PHENOTYPE_CSV / ABR_GENBANK, or the
/project defaults); skipped without it.
"""
import functools
from pathlib import Path

import numpy as np
import torch

from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR
from locusfusion_v4.layout import C_BASE_RES, RESIDUE_SLOT, RESIDUE_SYMBOLS, SLOTS, channel_names
from locusfusion_v4.tokenizer import PROPS, RESIDUES, build_tokens

LOCI = ["pncA"]
DRUGS = ["PYRAZINAMIDE"]
N_CHECK = 3000

HAVE_DATA = Path(REAL_GENOTYPE_DIR).exists()


@functools.cache
def tokens():
    return build_tokens(DRUGS, REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR,
                        loci=LOCI, regulatory=[], keep_residue=True, verbose=False)


@functools.cache
def v3_tokens():
    from datasets import load_dataset
    from models.locusfusion_v2 import LocusFusionV2Net
    ts = tokens()
    data = load_dataset(DRUGS[0], ["dna", "protein", "biophysical"],
                        REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV,
                        regulatory_dir=REAL_REGULATORY_DIR, loci=LOCI, regulatory_loci=[],
                        per_modality_branch=False, delta=True, protein_frame="reference",
                        unknown_symbols=True, verbose=False)
    net = LocusFusionV2Net.from_blocks(data.blocks, unknown_symbols=True, n_drugs=1)
    assert data.isolate_ids == ts.isolate_ids
    rng = np.random.default_rng(0)
    rows = np.unique(np.concatenate([np.argsort(-ts.n_occ.sum(1))[:N_CHECK // 2],
                                     rng.choice(ts.n, N_CHECK // 2, replace=False)]))
    xs = [torch.from_numpy(a[rows]).float() for a in data.arrays()]
    with torch.no_grad():
        feat3, _, _, _ = net._locus_tokens(xs)
    return rows, net.layout, feat3[:, :, 1:].numpy()


def test_width_and_names():
    ts = tokens()
    assert ts.feat.shape[3] == C_BASE_RES == 49
    assert ts.config["keep_residue"] and ts.config["token_channels"] == 49
    names = channel_names(DRUGS, keep_residue=True)
    assert len(names) == 50 and names[RESIDUE_SLOT[0]] == "residue:A"
    assert tuple(RESIDUES) == RESIDUE_SYMBOLS       # v3's protein-slot order


def test_one_residue_per_coding_token():
    ts = tokens()
    f = ts.feat[ts.valid]
    oh = f[:, slice(*RESIDUE_SLOT)]
    coding = ts.codon[ts.valid] >= 0
    assert set(np.unique(oh.sum(1))) <= {0.0, 1.0}
    assert (oh[coding].sum(1) == 1).all(), "a coding token with no residue"
    assert (oh[~coding].sum(1) == 0).all(), "a non-coding token with a residue"
    assert np.allclose(PROPS[oh[coding].argmax(1)], f[:, slice(*SLOTS["bio"])][coding], atol=1e-6)


def test_candidates_carry_the_reference_residue():
    cand = tokens().cand
    assert cand["feat"].shape[1] == C_BASE_RES
    oh = cand["feat"][:, slice(*RESIDUE_SLOT)]
    assert (oh.sum(1) == 1).all()
    assert np.allclose(cand["feat"][:, slice(*SLOTS["d_bio"])], 0)
    assert cand["feat"][:, SLOTS["consequence"][0]].all()


def test_matches_v3_protein_slot():
    ts = tokens()
    rows, lay, feat3 = v3_tokens()
    m = ts.valid[rows]
    a = ts.feat[rows][m][:, slice(*RESIDUE_SLOT)]
    b = feat3[m][:, slice(*lay["slots"]["protein"])]
    bad = ~np.isclose(a, b, atol=1e-5).all(1)
    assert not bad.any(), f"{int(bad.sum())} of {len(bad)} tokens differ from v3"


if __name__ == "__main__":
    from locusfusion_v4.tests._run import run
    run(globals(), skip_reason=None if HAVE_DATA else "no real data at REAL_GENOTYPE_DIR")
