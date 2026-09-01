"""
Pipeline entry point — modality-selectable.

Loads any subset of modalities (DNA / protein / biophysical / regulatory) via
the single dataloader (datasets.load_dataset), builds a MultiModalNet with one
branch per feature block, and runs BIG-TB's SD-CNN CV + held-out-test protocol
(training.multimodal.run_modal_cv). Results are written per drug under a run
folder, named by drug *and* the modality set so runs are self-describing:

    results/experiments/{run_name}/{DRUG}__{modality-tag}.json
    results/experiments/{run_name}/summary.csv

Modes:
  --real       (default) real BIG-TB data on Unity (bigtb_ref.REAL_*).
  --synthetic  small synthetic fixtures — proves the wiring, numbers meaningless.

Pass `all` to --modalities and/or --drugs to run every modality / every drug.

Examples (run from the project root):
    python scripts/run_experiment.py --modalities dna
    python scripts/run_experiment.py --modalities dna biophysical --drugs ISONIAZID
    python scripts/run_experiment.py --modalities all --drugs all --device cuda
    python scripts/run_experiment.py --modalities dna protein biophysical regulatory \
        --drugs ISONIAZID --epochs 60 --device cuda
    python scripts/run_experiment.py --synthetic --modalities all --drugs all
"""
import argparse
import json
import sys
import time
from pathlib import Path

# this file lives in scripts/; put the project root on the path so the package
# imports below resolve no matter which directory the job was launched from
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd  # noqa: E402  (import after the sys.path bootstrap, by design)
import torch  # noqa: E402

from bigtb_ref import (MODEL_WEIGHTS_DIR, REAL_GENOTYPE_DIR,  # noqa: E402
                       REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR, tb)
from datasets import DRUG_TO_REGULATORY, MODALITIES, load_dataset  # noqa: E402
from training.checkpoint import SAVE_CHOICES, write_pointer  # noqa: E402
from datasets.fixtures import build_fixture_dataset  # noqa: E402
from models import (ARCHITECTURES, DELTA_ARCHS, ENCODERS,  # noqa: E402
                    EXPERIMENTAL_MODELS, LOCUS_ENCODERS,
                    PER_LOCUS_ARCHS, SUMMARY_NORMS,
                    TOKEN_NORMS)
from training.curves import save_curves  # noqa: E402
from training.multimodal import run_modal_cv  # noqa: E402

RESULTS_DIR = PROJECT_DIR / "results" / "experiments"
ALL_MODALITIES = list(MODALITIES)
ALL_DRUGS = list(tb.DRUG_TO_LOCI)


def _resolve_choices(requested, universe, kind, ap):
    """Expand the sentinel 'all' to the full universe, else validate the
    requested items against it. `universe` is the ordered list of valid names."""
    if any(x.lower() == "all" for x in requested):
        return list(universe)
    picked = [x.upper() if kind == "drug" else x.lower() for x in requested]
    unknown = [x for x in picked if x not in universe]
    if unknown:
        ap.error(f"unknown {kind}(s) {unknown}; choose from {universe} or 'all'")
    return picked


def _parse_encoders(tokens, ap):
    """Parse ['dna=cnn', 'protein=transformer'] into {modality: encoder}."""
    if not tokens:
        return {}
    mapping = {}
    for tok in tokens:
        if "=" not in tok:
            ap.error(f"--encoders expects MODALITY=TYPE tokens, got {tok!r}")
        mod, enc = (s.strip().lower() for s in tok.split("=", 1))
        if mod not in MODALITIES:
            ap.error(f"--encoders: unknown modality {mod!r}; choose from {list(MODALITIES)}")
        if enc not in ENCODERS:
            ap.error(f"--encoders: unknown encoder {enc!r}; choose from {list(ENCODERS)}")
        mapping[mod] = enc
    return mapping


def _summary_row(r):
    return {
        "drug": r["drug"], "modalities": r["tag"],
        "genes": "+".join(r["genes"]), "n_valid": r["n_valid"],
        "n_R": r["n_resistant"], "n_S": r["n_susceptible"],
        "cv_auc_mean": r["cv_auc_mean"], "cv_auc_std": r["cv_auc_std"],
        "cv_auc_pr_mean": r["cv_auc_pr_mean"],
        "test_auc": r["test"]["auc"], "test_auc_pr": r["test"]["auc_pr"],
        "test_sens": r["test"]["sens"], "test_spec": r["test"]["spec"],
        "seconds": r["seconds"],
    }


def _write_summary(run_dir):
    rows = []
    for jf in sorted(run_dir.glob("*.json")):
        r = json.loads(jf.read_text())
        if "cv_auc_mean" in r:
            rows.append(_summary_row(r))
    if rows:
        pd.DataFrame(rows).round(4).to_csv(run_dir / "summary.csv", index=False)


def _make_synthetic(tmp, drugs, modalities, loci, regulatory_loci):
    """One fixture set covering the requested gene loci (+ regulatory regions if
    that modality is requested). Honors --loci / --regulatory-loci overrides."""
    genes = sorted(set(loci)) if loci else sorted(
        {g for d in drugs for g in tb.DRUG_TO_LOCI[d.upper()]})
    regions = []
    if "regulatory" in modalities:
        regions = sorted(set(regulatory_loci)) if regulatory_loci else sorted(
            {r for d in drugs for r in DRUG_TO_REGULATORY.get(d.upper(), [])})
        regions = [r for r in regions if r not in set(genes)]  # don't write a locus twice
    return build_fixture_dataset(
        tmp, genes=genes, drugs=[d.upper() for d in drugs],
        n_isolates=80, n_codons=40, seed=0, regulatory_regions=regions,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modalities", nargs="+", default=["dna"], metavar="M",
                    help=f"any subset of {ALL_MODALITIES}, or 'all' (default: dna)")
    ap.add_argument("--drugs", nargs="+", default=None, metavar="DRUG",
                    help="drug(s) to run, or 'all' (default: ISONIAZID RIFAMPICIN)")
    ap.add_argument("--loci", nargs="+", default=None, metavar="GENE",
                    help="which gene loci to load for dna/protein/biophysical "
                         "(default: the drug's DRUG_TO_LOCI). Applies to all "
                         "selected drugs, so usually pair with a single --drugs.")
    ap.add_argument("--all-regulatory", action="store_true",
                    help="keep the FULL WHO region set. By default regulatory "
                         "regions are intersected with the loaded loci, so a run "
                         "never has more promoter windows than coding loci "
                         "(KANAMYCIN then keeps rrs only and loses the eis "
                         "promoter). Ignored when --regulatory-loci is given.")
    ap.add_argument("--extra-loci", action="store_true",
                    help="add the EXTRA_LOCI overlay to the drug's default loci "
                         "(WHO Table 21 tier-1 genes DRUG_TO_LOCI omits: fabG1 "
                         "for ISONIAZID/ETHIONAMIDE). Off by default — on, the "
                         "run is no longer locus-matched to BIG-TB's SD-CNN. "
                         "Ignored when --loci is given.")
    ap.add_argument("--regulatory-loci", nargs="+", default=None, metavar="REGION",
                    help="which regulatory regions to load (default: the "
                         "WHO-derived per-drug set).")
    ap.add_argument("--encoders", nargs="+", default=None, metavar="MODALITY=TYPE",
                    help="per-modality encoder, e.g. --encoders dna=cnn "
                         f"protein=transformer. Types: {list(ENCODERS)}. "
                         "Modalities not named use --default-encoder.")
    ap.add_argument("--default-encoder", default="cnn", choices=list(ENCODERS),
                    help="encoder for modalities not named in --encoders (default: cnn)")
    real = ap.add_mutually_exclusive_group()
    real.add_argument("--real", dest="real", action="store_true", default=True,
                      help="run on the real BIG-TB data (default)")
    real.add_argument("--synthetic", dest="real", action="store_false",
                      help="run on small synthetic fixtures instead")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--monitor", default="auc", choices=["auc", "loss"],
                    help="early-stopping metric (default: auc — 'loss' stalls at "
                         "the majority-class collapse on imbalanced drugs)")
    ap.add_argument("--patience", type=int, default=15,
                    help="early-stopping patience (default: 15)")
    ap.add_argument("--min-epochs", type=int, default=0,
                    help="warmup: hold early stopping off for this many epochs "
                         "(default: 0 = off). Needed by architectures that start "
                         "from a degenerate init and plateau before they learn — "
                         "setfusion sat at a flat loss for ~12 epochs and patience "
                         "fired before it broke out. Best-weight restore still "
                         "runs, so a warmup can only help.")
    ap.add_argument("--min-delta", type=float, default=1e-4,
                    help="early-stopping min improvement (default: 1e-4)")
    # --- optimizer / capacity (all default to the full_run values) ------------
    ap.add_argument("--lr", type=float, default=None,
                    help="Adam learning rate (default: exp(-9) ~ 1.2e-4, BIG-TB's)")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="weight decay; any value > 0 switches Adam -> AdamW "
                         "(default: 0 = no regularization, as in full_run)")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="dropout after each dense-head hidden layer (default: 0)")
    ap.add_argument("--hidden", type=int, default=256,
                    help="dense-head width (default: 256)")
    ap.add_argument("--per-drug-hidden", type=int, default=0,
                    help="per-output hidden branch width; no effect single-drug "
                         "(out_dim=1), kept so both entry points take the same "
                         "flags (default: 0 = off)")
    # --- transformer encoder capacity ----------------------------------------
    # Applies wherever a transformer is actually selected: per-branch under
    # late_fusion / cisfusion, per-trunk under mdcnn. Same None-default
    # discipline as the setfusion group — an unset flag stays out of the
    # override dict, so models.TRANSFORMER_DEFAULTS stays the one place the
    # defaults are written down.
    #
    # These exist because a transformer branch is NOT parameter-comparable to a
    # CNN branch at its defaults: CNNEncoder flattens (out_features = 32*L/9, so
    # ~3.1M head params on a 3.4 kb block) while TransformerEncoder mean-pools to
    # d_model=64 regardless of length — a ~30x size gap. Matching capacity means
    # raising these deliberately; see results/experiments/transformer_run/.
    tf = ap.add_argument_group(
        "transformer encoder capacity (only where --encoders/--default-encoder "
        "select 'transformer')")
    tf.add_argument("--tf-d-model", type=int, default=None,
                    help="token width, and the per-branch output width since the "
                         "encoder mean-pools (default 64)")
    tf.add_argument("--tf-nhead", type=int, default=None,
                    help="attention heads; must divide --tf-d-model (default 4)")
    tf.add_argument("--tf-layers", type=int, default=None,
                    help="TransformerEncoderLayer count (default 2)")
    tf.add_argument("--tf-dim-ff", type=int, default=None,
                    help="feed-forward width inside each layer (default 128)")
    tf.add_argument("--tf-patch", type=int, default=None,
                    help="patch-embedding kernel AND stride, so the position axis "
                         "becomes ~L/patch tokens (default 9)")
    tf.add_argument("--tf-dropout", type=float, default=None,
                    help="dropout inside the transformer layers (default 0.1)")

    # --- setfusion capacity (ignored by every other arch) ---------------------
    # Defaults are None, not the values: an unset flag stays out of the override
    # dict entirely, so models.SETFUSION_DEFAULTS remains the one place the
    # full_run/full_run_v2 configuration is written down. Swept by
    # results/experiments/setfusion_scaling — axis A is --d-model, axis B is
    # --dim-ff/--fusion-layers/--hidden, axis C is the --enc-* knobs.
    sf = ap.add_argument_group("setfusion capacity (--arch setfusion only)")
    sf.add_argument("--d-model", type=int, default=None,
                    help="token width: encoder output, modality/locus embeddings, "
                         "transformer, drug queries, head input (default: 128). "
                         "Must be divisible by --nhead.")
    sf.add_argument("--nhead", type=int, default=None,
                    help="attention heads in fusion and pooling (default: 4)")
    sf.add_argument("--fusion-layers", type=int, default=None,
                    help="transformer encoder layers over the token set (default: 2)")
    sf.add_argument("--dim-ff", type=int, default=None,
                    help="transformer feed-forward width (default: 256)")
    sf.add_argument("--fusion-dropout", type=float, default=None,
                    help="dropout INSIDE the transformer/attention (default: 0.1). "
                         "Distinct from --dropout, which is the read-out MLP's.")
    sf.add_argument("--enc-width", type=int, default=None,
                    help="SharedBlockEncoder stem/conv1 channels (default: 64). "
                         "The 12-tap conv1 at this width dominates the FLOPs, so "
                         "this is the knob that moves wall-clock most.")
    sf.add_argument("--enc-out-channels", type=int, default=None,
                    help="SharedBlockEncoder conv2/conv3 channels (default: 32)")
    sf.add_argument("--enc-depth", type=int, default=None,
                    help="how many (3-tap, 3-tap, pool-3) stages follow conv1 "
                         "(default: 1 = the BIG-TB stack). Buys receptive field.")
    sf.add_argument("--enc-bins", type=int, default=None,
                    help="pooled segments per block, mean AND max (default: 4). "
                         "This is the token's information bottleneck: a 3.4 kb "
                         "locus is squeezed to 2*out_channels*bins numbers before "
                         "the transformer ever sees it.")
    sf.add_argument("--token-norm", default=None, choices=list(TOKEN_NORMS),
                    help="standardise each token across the batch with per-"
                         "(modality, locus) statistics (default: none). Measured "
                         "motivation: only 0.14%% of an encoded token varies with "
                         "the genotype, the rest is a locus-constant, so attention "
                         "collapses to uniform and the locus embedding is "
                         "redundant. 'keyed' strips that constant, which both "
                         "exposes the genotype signal and makes locus_emb the only "
                         "carrier of identity.")
    # --- locusfusion capacity (--arch locusfusion only) -----------------------
    # Its defaults are LOCUSFUSION_DEFAULTS. Unlike the other archs, size was
    # never the binding constraint here (setfusion_scaling swept four width axes
    # across 62 arms and closed nothing) — the tokenizer knobs below matter more
    # than the width ones.
    lf = ap.add_argument_group("locusfusion capacity (--arch locusfusion only)")
    lf.add_argument("--lf-d-model", type=int, default=None,
                    help="token width, shared by both stages and the read-out "
                         "(default: 128). Must be divisible by --lf-nhead.")
    lf.add_argument("--lf-nhead", type=int, default=None,
                    help="attention heads in both stages and the pooling (default: 4)")
    lf.add_argument("--lf-enc-layers", type=int, default=None,
                    help="stage-1 layers, WITHIN one locus (default: 2)")
    lf.add_argument("--lf-enc-dim-ff", type=int, default=None,
                    help="stage-1 feed-forward width (default: 256)")
    lf.add_argument("--lf-fusion-layers", type=int, default=None,
                    help="stage-2 layers, ACROSS loci (default: 2)")
    lf.add_argument("--lf-fusion-dim-ff", type=int, default=None,
                    help="stage-2 feed-forward width (default: 256)")
    lf.add_argument("--lf-dropout", type=float, default=None,
                    help="dropout inside both transformers and the pooling "
                         "attention (default: 0.1). Distinct from --dropout, "
                         "which is the read-out MLP's.")
    lf.add_argument("--lf-max-variants", type=int, default=None,
                    help="token cap per (locus, coordinate stream) (default: 16). "
                         "The variant census puts the 99th percentile at <=7 "
                         "columns per locus for 17 of 19 loci and 26 for rrs/rrl, "
                         "so 16 covers >99%% of (isolate, locus) pairs; overflow "
                         "keeps the FIRST 16 in positional order.")
    lf.add_argument("--lf-pos-dims", type=int, default=None,
                    help="sinusoidal position-encoding width (default: 64)")
    lf.add_argument("--lf-uncovered-frac", type=float, default=None,
                    help="fraction of a locus differing from the reference above "
                         "which it is flagged UNCOVERED rather than hypervariant "
                         "(default: 0.5). 14-91 isolates per locus are all-gap.")
    lf.add_argument("--lf-locus-encoder", default=None, choices=list(LOCUS_ENCODERS),
                    help="stage-1 weight sharing: 'shared' (one encoder, identity "
                         "from locus_emb only), 'adapter' (default: shared encoder "
                         "+ a per-locus FiLM, 2*d_model params per locus), or "
                         "'per_locus' (a separate encoder per locus; 19x the "
                         "stage-1 weights and the loci can no longer be batched).")
    lf.add_argument("--lf-summary-norm", default=None, choices=list(SUMMARY_NORMS),
                    help="standardise each locus summary across the batch with "
                         "per-locus statistics before stage 2 (default: keyed). "
                         "Measured motivation: the summary is read off the [WT] "
                         "slot, whose input is identical in every isolate, so "
                         "only ~1.6%% of it varies with the genotype at init — "
                         "the same failure token_signal diagnosed in setfusion. "
                         "'none' turns it off.")
    lf.add_argument("--lf-carry-variants", type=int, default=None,
                    help="how many of each locus's own variant tokens are handed "
                         "up to stage 2 alongside its summary (default: 0). >0 "
                         "lets cross-locus attention see individual variants "
                         "(rpoB+rpoC compensatory pairs) at a larger token set.")
    # --- experimental variant-set aggregators (models/experimental_models.py) --
    # Same tokenizer as locusfusion; these six differ only in how the variant set
    # is aggregated. Knobs are checked against the SELECTED member -- passing
    # --xm-fm-rank to --arch deepsets is an error, not a silent no-op.
    xm = ap.add_argument_group(
        "experimental aggregators (--arch catalogue|additive|noisyor|"
        "gatedpool|deepsets|fm)")
    xm.add_argument("--xm-d-model", type=int, default=None,
                    help="variant embedding width (default: 128). Ignored by "
                         "--arch catalogue, which has no embedding at all.")
    xm.add_argument("--xm-max-variants", type=int, default=None,
                    help="token cap per block (default: 16). The variant census "
                         "puts the 99th percentile at <=7 columns per locus for "
                         "17 of 19 loci and 26 for rrs/rrl.")
    xm.add_argument("--xm-pos-dims", type=int, default=None,
                    help="sinusoidal position-encoding width (default: 64)")
    xm.add_argument("--xm-uncovered-frac", type=float, default=None,
                    help="fraction of a block differing from the reference above "
                         "which it is flagged UNCOVERED rather than hypervariant "
                         "(default: 0.5)")
    xm.add_argument("--xm-dropout", type=float, default=None,
                    help="dropout on the variant embedding (default: 0.1)")
    xm.add_argument("--xm-fm-rank", type=int, default=None,
                    help="--arch fm only: factorization rank, i.e. how many "
                         "dimensions the pairwise-interaction term gets "
                         "(default: 8). All pairs cost O(T*rank), not O(T^2).")
    xm.add_argument("--xm-residual-catalogue", action="store_true",
                    help="--arch additive only: add the exact-identity weight "
                         "table on top of the featurised one, so the model "
                         "memorises variants it has seen and featurises the ones "
                         "it has not. Off by default so the clean measurement "
                         "(featurisation alone vs --arch catalogue) comes first.")
    # --- input encoding (all archs) -------------------------------------------
    ap.add_argument("--delta", action="store_true",
                    help="reference-difference input encoding: zero every column "
                         "matching the H37Rv reference, in every modality. Shape "
                         "and alphabet unchanged; what goes is the ~99.9%% of each "
                         "sequence identical across a clonal cohort, which carries "
                         "no discriminative signal but dominates the CNN's input.")
    # --- LR schedule (all archs) ----------------------------------------------
    ap.add_argument("--lr-schedule", default="none", choices=["none", "cosine"],
                    help="per-epoch LR schedule: 'none' (flat, what every "
                         "recorded run used) or 'cosine' — linear warmup over "
                         "--warmup-epochs then cosine decay over the epoch cap. "
                         "The multiplier never exceeds 1, so this can only lower "
                         "the LR relative to the flat-LR control.")
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="linear LR warmup length for --lr-schedule cosine "
                         "(default: 0). Unrelated to --min-epochs, which is the "
                         "early-stopping warmup.")
    ap.add_argument("--mdcnn-trunk-per-modality", action="store_true",
                    help="--arch mdcnn: group blocks into trunks by (modality, "
                         "channels) rather than channels alone, so 5-channel "
                         "regulatory windows stop being padded to the longest CDS "
                         "inside the DNA trunk. No effect on other archs.")
    ap.add_argument("--out-bias", default="none", metavar="auto|none|FLOAT",
                    help="output-bias init: 'auto' (train log-odds), 'none' "
                         "(PyTorch default), or a float (default: none)")
    ap.add_argument("--arch", default="late_fusion", choices=list(ARCHITECTURES),
                    help="network topology: 'late_fusion' (one encoder per block, "
                         "concatenated), 'mdcnn' (BIG-TB's own: loci stacked as "
                         "channels, 12-bp conv across all of them from layer 1), "
                         "'setfusion', 'cisfusion', or 'locusfusion' (one token "
                         "per VARIANT, fused within a locus then across loci). "
                         "All but late_fusion imply per-locus branches and ignore "
                         "--encoders; 'locusfusion' additionally implies --delta.")
    ap.add_argument("--per-locus-branches", action="store_true",
                    help="one branch per gene/region instead of the default one "
                         "branch per MODALITY (loci concatenated). Also splits DNA "
                         "into one block per locus.")
    ap.add_argument("--save-weights", default="best", choices=list(SAVE_CHOICES),
                    help="which CV fold weights to persist: 'best' (default — the "
                         "fold scored on TEST, i.e. the one the reported numbers "
                         "come from), 'all' (~5x the bytes), or 'none'. Written "
                         f"with a rebuild config to {MODEL_WEIGHTS_DIR}/"
                         "{run-name}/. Nothing was checkpointed before this "
                         "existed, so full_run's models are unrecoverable.")
    ap.add_argument("--weights-dir", default=None,
                    help=f"override the weights root (default: {MODEL_WEIGHTS_DIR}, "
                         "or $ABR_MODEL_WEIGHTS_DIR)")
    ap.add_argument("--tb", action="store_true",
                    help="log to TensorBoard under the run folder's tb/ dir")
    ap.add_argument("--run-name", default=None,
                    help="run folder name under results/experiments/ (default: timestamp)")
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    if args.out_bias == "auto":
        out_bias = "auto"
    elif args.out_bias.lower() in ("none", "null"):
        out_bias = None
    else:
        try:
            out_bias = float(args.out_bias)
        except ValueError:
            ap.error(f"--out-bias must be 'auto', 'none', or a float; got {args.out_bias!r}")

    setfusion = {k: v for k, v in {
        "d_model": args.d_model, "nhead": args.nhead, "layers": args.fusion_layers,
        "dim_ff": args.dim_ff, "dropout": args.fusion_dropout,
        "enc_width": args.enc_width, "enc_out_channels": args.enc_out_channels,
        "enc_depth": args.enc_depth, "bins": args.enc_bins,
        "token_norm": args.token_norm,
    }.items() if v is not None}
    if setfusion and args.arch != "setfusion":
        # silently ignoring them would make a sweep arm look like it ran when it
        # was really the control with a different folder name
        ap.error(f"setfusion capacity flags {sorted(setfusion)} require "
                 f"--arch setfusion (got --arch {args.arch})")

    locusfusion = {k: v for k, v in {
        "d_model": args.lf_d_model, "nhead": args.lf_nhead,
        "enc_layers": args.lf_enc_layers, "enc_dim_ff": args.lf_enc_dim_ff,
        "fusion_layers": args.lf_fusion_layers, "fusion_dim_ff": args.lf_fusion_dim_ff,
        "dropout": args.lf_dropout, "max_variants": args.lf_max_variants,
        "pos_dims": args.lf_pos_dims, "uncovered_frac": args.lf_uncovered_frac,
        "locus_encoder": args.lf_locus_encoder,
        "summary_norm": args.lf_summary_norm,
        "carry_variants": args.lf_carry_variants,
    }.items() if v is not None}
    experimental = {k: v for k, v in {
        "d_model": args.xm_d_model, "max_variants": args.xm_max_variants,
        "pos_dims": args.xm_pos_dims, "uncovered_frac": args.xm_uncovered_frac,
        "dropout": args.xm_dropout, "fm_rank": args.xm_fm_rank,
        "residual_catalogue": args.xm_residual_catalogue or None,
    }.items() if v is not None}
    if experimental and args.arch not in EXPERIMENTAL_MODELS:
        ap.error(f"experimental aggregator flags "
                 f"{sorted('--xm-' + k.replace('_', '-') for k in experimental)} "
                 f"require one of --arch {'|'.join(sorted(EXPERIMENTAL_MODELS))} "
                 f"(got --arch {args.arch})")

    if locusfusion and args.arch != "locusfusion":
        ap.error(f"locusfusion capacity flags {sorted('--lf-' + k.replace('_', '-') for k in locusfusion)} "
                 f"require --arch locusfusion (got --arch {args.arch})")

    modalities = _resolve_choices(args.modalities, ALL_MODALITIES, "modality", ap)
    drugs = _resolve_choices(args.drugs or ["ISONIAZID", "RIFAMPICIN"],
                             ALL_DRUGS, "drug", ap)
    branch_models = _parse_encoders(args.encoders, ap)

    transformer = {k: v for k, v in {
        "d_model": args.tf_d_model, "nhead": args.tf_nhead,
        "layers": args.tf_layers, "dim_ff": args.tf_dim_ff,
        "patch": args.tf_patch, "dropout": args.tf_dropout,
    }.items() if v is not None}
    if transformer and "transformer" not in set(
            list(branch_models.values()) + [args.default_encoder]):
        # same rule as the setfusion flags: silently ignoring them would make a
        # sweep arm look like it ran when it was really the control
        ap.error(f"transformer capacity flags {sorted('--tf-'+k.replace('_','-') for k in transformer)} "
                 "require a transformer encoder — pass --default-encoder transformer "
                 "or --encoders MODALITY=transformer")

    epochs = args.epochs if args.epochs is not None else (60 if args.real else 5)
    # the mdcnn topology stacks LOCI as channels, so it needs one block per
    # locus; the per-modality default (loci concatenated end-to-end) would
    # collapse the whole modality into a single channel plane.
    per_locus = args.per_locus_branches or args.arch in PER_LOCUS_ARCHS
    if args.arch in PER_LOCUS_ARCHS and not args.per_locus_branches:
        print(f"--arch {args.arch}: loading per-locus branches (implied)")
    # locusfusion tokenizes the columns that DIFFER from H37Rv, so on a plain
    # one-hot every column differs from nothing, the cap keeps the first
    # max_variants columns of each block, and the model silently degenerates
    # while still producing plausible numbers. Imply it rather than trust a flag.
    delta = args.delta or args.arch in DELTA_ARCHS
    if args.arch in DELTA_ARCHS and not args.delta:
        print(f"--arch {args.arch}: reference-difference input encoding (--delta, implied)")
    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = (run_dir / "tb") if args.tb else None

    print(f"Run '{run_name}' -> {run_dir}")
    print(f"mode={'real' if args.real else 'synthetic'} modalities={modalities} "
          f"drugs={drugs}")
    print(f"arch={args.arch} branches={'per-locus' if per_locus else 'per-modality'}"
          f"{' extra_loci=on' if args.extra_loci else ''}")
    # mdcnn takes ONE encoder for all its trunks (see _build_model), so say which
    # one -- printing "n/a" was right only while the conv trunk was the sole option
    print(f"encoders: {branch_models or '{}'} (default: {args.default_encoder})"
          if args.arch != "mdcnn"
          else f"encoders: {args.default_encoder} for every mdcnn trunk")
    print(f"device={args.device} epochs={epochs} batch_size={args.batch_size} "
          f"n_splits={args.n_splits}")
    print(f"lr={args.lr if args.lr is not None else 'exp(-9)'} "
          f"lr_schedule={args.lr_schedule}"
          f"{f'(warmup {args.warmup_epochs})' if args.lr_schedule != 'none' else ''} "
          f"weight_decay={args.weight_decay} dropout={args.dropout} "
          f"hidden={args.hidden} setfusion={setfusion or 'defaults'}")
    if args.tb:
        print(f"TensorBoard: tensorboard --logdir {RESULTS_DIR}\n")

    # resolve data source (real paths, or a synthetic fixture set built once)
    import contextlib
    import tempfile
    with contextlib.ExitStack() as stack:
        if args.real:
            genotype_dir, phenotype_csv, regulatory_dir = (
                REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR)
        else:
            tmp = stack.enter_context(tempfile.TemporaryDirectory())
            genotype_dir, phenotype_csv = _make_synthetic(
                tmp, drugs, modalities, args.loci, args.regulatory_loci)
            regulatory_dir = genotype_dir

        for drug in drugs:
            try:
                data = load_dataset(drug, modalities, genotype_dir,
                                    phenotype_csv, regulatory_dir=regulatory_dir,
                                    loci=args.loci, regulatory_loci=args.regulatory_loci,
                                    per_modality_branch=not per_locus,
                                    extra_loci=args.extra_loci,
                                    all_regulatory=args.all_regulatory,
                                    delta=delta)
                result = run_modal_cv(data, epochs=epochs, n_splits=args.n_splits,
                                      batch_size=args.batch_size, device=args.device,
                                      tb_dir=tb_dir, seed=args.seed,
                                      branch_models=branch_models,
                                      default_encoder=args.default_encoder,
                                      monitor=args.monitor, patience=args.patience,
                                      min_delta=args.min_delta, out_bias=out_bias,
                                      arch=args.arch, min_epochs=args.min_epochs,
                                      lr=args.lr, weight_decay=args.weight_decay,
                                      hidden=args.hidden, dropout=args.dropout,
                                      per_drug_hidden=args.per_drug_hidden,
                                      mdcnn_trunk_per_modality=args.mdcnn_trunk_per_modality,
                                      setfusion=setfusion,
                                      transformer=transformer,
                                      locusfusion=locusfusion,
                                      experimental=experimental,
                                      lr_schedule=args.lr_schedule,
                                      warmup_epochs=args.warmup_epochs,
                                      run_name=run_name,
                                      save_weights=args.save_weights,
                                      weights_dir=args.weights_dir,
                                      data_config={
                                          "genotype_dir": str(genotype_dir),
                                          "phenotype_csv": str(phenotype_csv),
                                          "regulatory_dir": str(regulatory_dir),
                                          "real": args.real,
                                          "per_modality_branch": not per_locus,
                                          "all_regulatory": args.all_regulatory,
                                          "extra_loci": args.extra_loci,
                                          "delta": args.delta,
                                          "loci_override": args.loci,
                                          "regulatory_loci_override": args.regulatory_loci,
                                      })
                out = run_dir / f"{drug}__{result['tag']}.json"
                out.write_text(json.dumps(result, indent=2))
                write_pointer(run_dir, f"{drug}__{result['tag']}",
                              result.get("weights_dir"))
                # per-fold loss / val-metric curves — shows whether `epochs` was enough
                save_curves(result["cv_folds"], out.with_name(f"{out.stem}_curves.png"),
                            title=f"{drug} / {result['tag']}  (epoch cap {epochs}, "
                                  f"monitor={args.monitor}, patience={args.patience}"
                                  + (f", warmup={args.min_epochs}" if args.min_epochs
                                     else "") + ")")
                _write_summary(run_dir)
            except Exception as e:  # one bad drug shouldn't sink the run
                print(f"[{drug}] FAILED: {type(e).__name__}: {e}", flush=True)
                import traceback
                traceback.print_exc()

    _write_summary(run_dir)
    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
