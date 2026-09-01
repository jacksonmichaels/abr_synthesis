"""
Multi-drug pipeline entry point.

Predicts ALL drugs simultaneously with one ``models.MultiDrugNet`` over the
union of every drug's loci (one branch per locus by default). Mirrors
run_experiment.py, but a single run covers every drug at once instead of one
drug per run.

    results/experiments/{run_name}/multidrug__{modality-tag}.json
    results/experiments/{run_name}/multidrug_summary.csv   (one row per drug + MACRO)

Modes:
  --real       (default) real BIG-TB data on Unity (bigtb_ref.REAL_*).
  --synthetic  small synthetic fixtures — proves the wiring, numbers meaningless.

Examples (run from the project root):
    python scripts/run_multidrug.py --modalities dna --device cuda
    python scripts/run_multidrug.py --drugs ISONIAZID RIFAMPICIN MOXIFLOXACIN --epochs 40
    python scripts/run_multidrug.py --synthetic --modalities dna
"""
import argparse
import contextlib
import json
import sys
import tempfile
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
from training.checkpoint import SAVE_CHOICES, write_pointer  # noqa: E402
from datasets import (ALL_DRUGS, MODALITIES, loci_on_disk,  # noqa: E402
                      load_multidrug_dataset, union_loci, union_regulatory)
from datasets.fixtures import build_fixture_dataset  # noqa: E402
from models import (ARCHITECTURES, DELTA_ARCHS, ENCODERS,  # noqa: E402
                    EXPERIMENTAL_MODELS, LOCUS_ENCODERS,
                    PER_LOCUS_ARCHS, SUMMARY_NORMS,
                    VARIANT_TOKEN_ARCHS)
from training.curves import save_curves  # noqa: E402
from training.multidrug import run_multidrug_cv  # noqa: E402

RESULTS_DIR = PROJECT_DIR / "results" / "experiments"
ALL_MODALITIES = list(MODALITIES)


def _resolve_drugs(requested, ap):
    if not requested or any(x.lower() == "all" for x in requested):
        return list(ALL_DRUGS)
    picked = [x.upper() for x in requested]
    unknown = [x for x in picked if x not in ALL_DRUGS]
    if unknown:
        ap.error(f"unknown drug(s) {unknown}; choose from {ALL_DRUGS} or 'all'")
    return picked


def _resolve_modalities(requested, ap):
    if any(x.lower() == "all" for x in requested):
        return list(ALL_MODALITIES)
    picked = [x.lower() for x in requested]
    unknown = [x for x in picked if x not in ALL_MODALITIES]
    if unknown:
        ap.error(f"unknown modalit(ies) {unknown}; choose from {ALL_MODALITIES} or 'all'")
    return picked


def _write_summary(run_dir, result):
    rows = []
    for d in result["drugs"]:
        tp = result["test_per_drug"][d]
        rows.append({
            "drug": d, "cv_auc": result["cv_per_drug_auc"][d],
            "test_auc": tp["auc"], "test_auc_pr": tp["auc_pr"],
            "test_sens": tp["sens"], "test_spec": tp["spec"],
            "n_R": tp["n_R"], "n_S": tp["n_S"],
        })
    rows.append({
        "drug": "MACRO", "cv_auc": result["cv_macro_auc_mean"],
        "test_auc": result["test_macro_auc"], "test_auc_pr": result["test_macro_auc_pr"],
        "test_sens": float("nan"), "test_spec": float("nan"),
        "n_R": sum(r["n_R"] for r in rows), "n_S": sum(r["n_S"] for r in rows),
    })
    pd.DataFrame(rows).round(4).to_csv(run_dir / "multidrug_summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modalities", nargs="+", default=["dna"], metavar="M",
                    help=f"any subset of {ALL_MODALITIES}, or 'all' (default: dna)")
    ap.add_argument("--drugs", nargs="+", default=None, metavar="DRUG",
                    help="drugs to predict jointly, or 'all' (default: all 11)")
    ap.add_argument("--loci", nargs="+", default=None, metavar="GENE",
                    help="gene loci to load (default: every curated locus FASTA "
                         "in the genotype dir — MD-CNN's own drug-independent "
                         "rule; see --per-drug-loci)")
    ap.add_argument("--per-drug-loci", action="store_true",
                    help="use the UNION of the selected drugs' DRUG_TO_LOCI "
                         "instead of every locus on disk (the older behaviour; "
                         "18 loci, no fabG1)")
    ap.add_argument("--all-regulatory", action="store_true",
                    help="keep the FULL WHO region set. By default regulatory "
                         "regions are intersected with the loaded loci, so a run "
                         "never has more promoter windows than coding loci "
                         "(KANAMYCIN then keeps rrs only and loses the eis "
                         "promoter). Ignored when --regulatory-loci is given.")
    ap.add_argument("--extra-loci", action="store_true",
                    help="with --per-drug-loci, add the EXTRA_LOCI overlay "
                         "(fabG1 for INH/ETO). No effect on the all-loci default, "
                         "which already includes every curated locus.")
    ap.add_argument("--arch", default="late_fusion", choices=list(ARCHITECTURES),
                    help="network topology: 'late_fusion' (one encoder per block, "
                         "concatenated) or 'mdcnn' (BIG-TB's own: loci stacked as "
                         "channels, 12-bp conv across all of them from layer 1). "
                         "'mdcnn' implies per-locus branches and ignores --encoders.")
    ap.add_argument("--per-locus-branches", action="store_true",
                    help="one branch per locus/region (~100 branches for all "
                         "modalities). Default: one branch per MODALITY (loci "
                         "concatenated). Implied by --arch mdcnn.")
    ap.add_argument("--encoders", nargs="+", default=None, metavar="MODALITY=TYPE",
                    help=f"per-modality encoder, e.g. dna=cnn. Types: {list(ENCODERS)}")
    ap.add_argument("--default-encoder", default="cnn", choices=list(ENCODERS))
    real = ap.add_mutually_exclusive_group()
    real.add_argument("--real", dest="real", action="store_true", default=True)
    real.add_argument("--synthetic", dest="real", action="store_false")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--monitor", default="auc", choices=["auc", "loss"])
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--min-epochs", type=int, default=0,
                    help="warmup: hold early stopping off for this many epochs "
                         "(default: 0 = off). See run_experiment.py --min-epochs; "
                         "the joint setfusion folds are the case it exists for.")
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--monitor-min-n", type=int, default=0,
                    help="drop drugs with fewer than this many labelled TRAIN "
                         "isolates from the early-stopping metric (default: 0 = "
                         "keep all, the full_run behaviour). The monitor is an "
                         "unweighted mean over 11 drugs, so LEVOFLOXACIN (n=269, "
                         "~15 resistant per val fold) contributes a ninth of the "
                         "signal and mostly noise. Excluded drugs are still "
                         "trained on and still reported — this only changes WHEN "
                         "training stops. 500 excludes LEVOFLOXACIN alone.")
    # --- optimizer / capacity (all default to the full_run values) ------------
    ap.add_argument("--lr", type=float, default=None,
                    help="Adam learning rate (default: exp(-9) ~ 1.2e-4, BIG-TB's)")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="weight decay; any value > 0 switches Adam -> AdamW "
                         "(default: 0 = no regularization, as in full_run)")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="dropout after each dense-head hidden layer (default: 0)")
    ap.add_argument("--hidden", type=int, default=256,
                    help="dense-head width (default: 256). Joint models read all "
                         "11 drugs off this one vector.")
    ap.add_argument("--per-drug-hidden", type=int, default=0,
                    help="give each drug its own 'hidden -> k -> 1' branch off "
                         "the shared trunk instead of one shared output linear "
                         "(default: 0 = off). The joint models have no per-drug "
                         "capacity at all without this.")
    # --- Learn-to-Branch head (late_fusion / mdcnn / cisfusion) ---------------
    # Luo et al. 2024 (Sci Rep 14:6631), subject -> drug. The joint head shares
    # everything between the 11 drugs and the single-drug grid shares nothing;
    # this learns WHICH drugs share, via G group nodes the drugs route to with
    # an annealed Gumbel-softmax. See models.BranchedHead.
    br = ap.add_argument_group("branched head (Learn-to-Branch)")
    br.add_argument("--head", choices=("dense", "branched"), default="dense",
                    help="read-out head (default: dense = the current joint head)")
    br.add_argument("--branch-groups", type=int, default=None,
                    help="G, the number of group nodes drugs can route to "
                         "(default: 4). G=1 is the routing-free control.")
    br.add_argument("--branch-per-drug-hidden", type=int, default=None,
                    help="width of each drug's own MLP off its group node "
                         "(default: 64). Match --per-drug-hidden to compare "
                         "against the capacity control.")
    br.add_argument("--branch-tau-start", type=float, default=None,
                    help="Gumbel-softmax temperature at epoch 0 (default: 5.0)")
    br.add_argument("--branch-tau-end", type=float, default=None,
                    help="...and at the final epoch (default: 0.5). Annealed "
                         "linearly; low tau drives theta toward one-hot.")
    br.add_argument("--branch-generic-weight", type=float, default=None,
                    help="lambda on the generic head's loss (default: 0.3). The "
                         "generic head is the cold-start path; 0 disables it.")
    br.add_argument("--branch-group-dropout", type=float, default=None,
                    help="dropout inside each group node (default: 0)")

    # --- setfusion capacity (ignored by every other arch) ---------------------
    # --- transformer encoder capacity ----------------------------------------
    # Applies wherever a transformer is actually selected: per-branch under
    # late_fusion / cisfusion, per-trunk under mdcnn. Same None-default
    # discipline as the setfusion group, so models.TRANSFORMER_DEFAULTS stays the
    # single place the defaults live. A transformer branch is not
    # parameter-comparable to a CNN branch at those defaults (CNNEncoder
    # flattens, TransformerEncoder mean-pools to d_model), so matching capacity
    # means raising these deliberately — see results/experiments/transformer_run/.
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
                         "channels) rather than channels alone, so the 5-channel "
                         "regulatory windows stop being padded to the longest CDS "
                         "inside the DNA trunk. No effect on other archs.")
    ap.add_argument("--out-bias", default="none", metavar="none|FLOAT",
                    help="output-bias init: 'none' (default) or a float")
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
    ap.add_argument("--tb", action="store_true")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    drugs = _resolve_drugs(args.drugs, ap)
    modalities = _resolve_modalities(args.modalities, ap)
    branch_models = {}

    for tok in (args.encoders or []):
        if "=" not in tok:
            ap.error(f"--encoders expects MODALITY=TYPE, got {tok!r}")
        m, e = (s.strip().lower() for s in tok.split("=", 1))
        branch_models[m] = e

    transformer = {k: v for k, v in {
        "d_model": args.tf_d_model, "nhead": args.tf_nhead,
        "layers": args.tf_layers, "dim_ff": args.tf_dim_ff,
        "patch": args.tf_patch, "dropout": args.tf_dropout,
    }.items() if v is not None}
    if transformer and "transformer" not in set(
            list(branch_models.values()) + [args.default_encoder]):
        ap.error(f"transformer capacity flags {sorted('--tf-'+k.replace('_','-') for k in transformer)} "
                 "require a transformer encoder — pass --default-encoder transformer "
                 "or --encoders MODALITY=transformer")
    out_bias = None if args.out_bias.lower() in ("none", "null") else float(args.out_bias)
    setfusion = {k: v for k, v in {
        "d_model": args.d_model, "nhead": args.nhead, "layers": args.fusion_layers,
        "dim_ff": args.dim_ff, "dropout": args.fusion_dropout,
        "enc_width": args.enc_width, "enc_out_channels": args.enc_out_channels,
        "enc_depth": args.enc_depth, "bins": args.enc_bins,
    }.items() if v is not None}
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

    branched = {k: v for k, v in {
        "n_groups": args.branch_groups,
        "per_drug_hidden": args.branch_per_drug_hidden,
        "tau_start": args.branch_tau_start, "tau_end": args.branch_tau_end,
        "generic_weight": args.branch_generic_weight,
        "group_dropout": args.branch_group_dropout,
    }.items() if v is not None}
    if branched and args.head != "branched":
        ap.error(f"branch flags {sorted('--branch-' + k.replace('_', '-') for k in branched)} "
                 "require --head branched")
    branched = (branched or {}) if args.head == "branched" else None

    if setfusion and args.arch != "setfusion":
        # silently ignoring them would make a sweep arm look like it ran when it
        # was really the control with a different folder name
        ap.error(f"setfusion capacity flags {sorted(setfusion)} require "
                 f"--arch setfusion (got --arch {args.arch})")
    epochs = args.epochs if args.epochs is not None else (60 if args.real else 5)
    # mdcnn stacks LOCI as channels -> needs one block per locus (see above)
    per_locus = args.per_locus_branches or args.arch in PER_LOCUS_ARCHS
    if args.arch in PER_LOCUS_ARCHS and not args.per_locus_branches:
        print(f"--arch {args.arch}: loading per-locus branches (implied)")
    # see scripts/run_experiment.py: locusfusion tokenizes the columns that
    # DIFFER from H37Rv, so a plain one-hot degenerates it silently.
    delta = args.delta or args.arch in DELTA_ARCHS
    if args.arch in DELTA_ARCHS and not args.delta:
        print(f"--arch {args.arch}: reference-difference input encoding (--delta, implied)")
    # ...and locusfusion goes further: its blocks are one SYMBOL ID per column
    # plus the per-column H37Rv reference, coordinate and codon phase
    # (datasets/tokens.py). Implied, never a flag — the model rejects a one-hot
    # block outright rather than silently reading the head of it.
    variant_tokens = args.arch in VARIANT_TOKEN_ARCHS
    if variant_tokens:
        print(f"--arch {args.arch}: symbol-id variant tokens (implied)")
    run_name = args.run_name or time.strftime("multidrug_%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = (run_dir / "tb") if args.tb else None

    print(f"Run '{run_name}' -> {run_dir}")
    print(f"mode={'real' if args.real else 'synthetic'} modalities={modalities} "
          f"drugs={len(drugs)} loci={'union' if not args.loci else args.loci}")
    branch_mode = "per-locus" if per_locus else "per-modality"
    print(f"device={args.device} epochs={epochs} arch={args.arch} branches={branch_mode} "
          f"monitor={args.monitor} patience={args.patience} out_bias={out_bias}")
    print(f"lr={args.lr if args.lr is not None else 'exp(-9)'} "
          f"lr_schedule={args.lr_schedule}"
          f"{f'(warmup {args.warmup_epochs})' if args.lr_schedule != 'none' else ''} "
          f"weight_decay={args.weight_decay} dropout={args.dropout} "
          f"hidden={args.hidden} per_drug_hidden={args.per_drug_hidden} "
          f"setfusion={setfusion or 'defaults'} "
          f"monitor_min_n={args.monitor_min_n} "
          f"all_regulatory={args.all_regulatory} "
          f"mdcnn_trunk_per_modality={args.mdcnn_trunk_per_modality}")

    with contextlib.ExitStack() as stack:
        if args.real:
            geno, pheno, reg = REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR
        else:
            tmp = stack.enter_context(tempfile.TemporaryDirectory())
            genes = args.loci or union_loci(drugs)
            regions = union_regulatory(drugs) if "regulatory" in modalities else []
            regions = [r for r in regions if r not in set(genes)]
            geno, pheno = build_fixture_dataset(
                tmp, genes=sorted(genes), drugs=drugs, n_isolates=200,
                n_codons=30, seed=0, regulatory_regions=regions)
            reg = geno

        # --- locus set (B): MD-CNN feeds every curated locus to every drug,
        # regardless of which drugs are being predicted. The per-drug union is
        # SD-CNN's rule and misses fabG1, which has a FASTA but is named by no
        # drug. Fixtures are excluded: that directory also holds the synthetic
        # regulatory windows, which would be read as coding loci.
        loci = args.loci
        if loci is None and args.real and not args.per_drug_loci:
            loci = loci_on_disk(geno)
            print(f"loci: all {len(loci)} curated on disk (MD-CNN rule): {loci}")
        elif loci is None:
            loci = union_loci(drugs, extra=args.extra_loci)
            print(f"loci: per-drug union ({len(loci)}"
                  f"{', +EXTRA_LOCI' if args.extra_loci else ''}): {loci}")
        data = load_multidrug_dataset(drugs, modalities, geno, pheno,
                                      regulatory_dir=reg, loci=loci,
                                      per_modality_branch=not per_locus,
                                      all_regulatory=args.all_regulatory,
                                      delta=delta,
                                      variant_tokens=variant_tokens)
        result = run_multidrug_cv(data, epochs=epochs, n_splits=args.n_splits,
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
                                  monitor_min_n=args.monitor_min_n,
                                  setfusion=setfusion,
                                  branched=branched,
                                  transformer=transformer,
                                  locusfusion=locusfusion,
                                  experimental=experimental,
                                  lr_schedule=args.lr_schedule,
                                  warmup_epochs=args.warmup_epochs,
                                  run_name=run_name, save_weights=args.save_weights,
                                  weights_dir=args.weights_dir,
                                  data_config={
                                      "genotype_dir": str(geno),
                                      "phenotype_csv": str(pheno),
                                      "regulatory_dir": str(reg),
                                      "real": args.real,
                                      "drugs": drugs,
                                      "per_modality_branch": not per_locus,
                                      "all_regulatory": args.all_regulatory,
                                      "extra_loci": args.extra_loci,
                                      "delta": args.delta,
                                      "variant_tokens": variant_tokens,
                                      "per_drug_loci": args.per_drug_loci,
                                      # the resolved region list is recoverable
                                      # from the 'regulatory:*' block names
                                  })

    (run_dir / f"multidrug__{result['tag']}.json").write_text(json.dumps(result, indent=2))
    _write_summary(run_dir, result)
    # breadcrumb: the run folder records where its weights went
    write_pointer(run_dir, f"multidrug__{result['tag']}", result.get("weights_dir"))
    # per-fold loss / macro-val-AUC curves — shows whether `epochs` was enough
    save_curves(result["cv_folds"], run_dir / f"multidrug__{result['tag']}_curves.png",
                title=f"multidrug / {result['tag']}  (epoch cap {epochs}, "
                      f"monitor={args.monitor}, patience={args.patience}"
                      + (f", warmup={args.min_epochs}" if args.min_epochs else "") + ")")
    print(f"\nWrote {run_dir}/multidrug__{result['tag']}.json + multidrug_summary.csv")
    print("All done.", flush=True)


if __name__ == "__main__":
    main()
