#!/usr/bin/env python3
"""
setfusion_scaling — does a BIGGER SetFusionNet close the gap, and where?

`setfusion` is by far the smallest architecture in the grid (0.46 M parameters
joint-DNA, against 45.9 M for late_fusion and 48.0 M for cisfusion) and it is
also the weakest joint model in full_run_v2 (macro CV 0.7939 vs 0.9184 /
0.9228). This sweep asks whether that is a capacity problem, and if so which
capacity, by scaling one thing at a time along four axes:

    A  token width          --d-model                        (128 -> 512)
    B  post-encoder width   --dim-ff / --fusion-layers / --hidden
    C  encoder block        --enc-width+--enc-out-channels / --enc-depth /
                            --enc-bins
    D  per-drug read-out    --per-drug-hidden                (joint only)

plus two TRAINING-only arms (R) at baseline capacity, and a sparse set of
crosses (X). Every arm changes ONE thing against the full_run_v2 control except
the X arms, which are labelled as bundles and are only interpretable against the
single-knob arms.

The control is `results/experiments/full_run_v2/{,multidrug_}{mods}__setfusion`
— already run, so no control jobs are submitted here. Every arm reuses that
run's exact protocol (300 epochs, patience 30, warmup 50, lr exp(-9), batch 128,
seed 0, all curated loci) so the ONLY difference is the flags in the table below.

    # what would be submitted, and the model each arm builds (no cluster needed)
    python scripts/sweep_setfusion_scaling.py --params
    python scripts/sweep_setfusion_scaling.py --dry-run

    # stage it: joint first (cheap, 62 jobs), single-drug after
    python scripts/sweep_setfusion_scaling.py --scope joint
    python scripts/sweep_setfusion_scaling.py --scope single --axes A C

Submission is delegated to scripts/sbatch_all_runs.py, so every job still lands
in slurm_logs/submitted_*.json with the exact command behind it.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

RUN_PREFIX = "setfusion_scaling"
EXP = PROJECT / "results" / "experiments"
CONTROL_RUN = "full_run_v2"

# The full_run_v2 protocol, reproduced exactly. Changing anything here breaks
# the control comparison, which is the whole design.
PROTOCOL = ["--epochs", "300", "--patience", "30", "--min-epochs", "50",
            "--save-weights", "best"]

MODALITY_SETS = {"dna": ["dna"], "dna_protein": ["dna", "protein"]}


class Arm:
    """One sweep cell: a name, the axis it belongs to, and the flags that make it.

    ``scope`` is 'both' or 'joint' — axis D and the per-drug cross are no-ops for
    a single-drug model (one output, nothing to separate), so they are not
    submitted single-drug rather than being submitted as duplicate controls.

    ``heavy`` marks the arms whose activations, not whose parameters, are the
    problem: --enc-width scales the 12-tap conv1 over the full un-pooled length
    quadratically, which is both the wall-clock and the VRAM risk. Those get a
    longer limit and a card with room.
    """

    def __init__(self, name, axis, flags, note, scope="both", heavy=False):
        self.name, self.axis, self.flags = name, axis, flags
        self.note, self.scope, self.heavy = note, scope, heavy

    def __repr__(self):
        return f"Arm({self.name})"


ENC_W96 = ["--enc-width", "96", "--enc-out-channels", "48"]
ENC_W128 = ["--enc-width", "128", "--enc-out-channels", "64"]
COSINE = ["--lr-schedule", "cosine", "--warmup-epochs", "20"]
REG = ["--dropout", "0.3", "--weight-decay", "1e-4"]

ARMS = [
    # --- A: token width. Everything downstream is d_model-wide, so this is the
    # one knob that scales the encoder output, both embeddings, the transformer,
    # the drug queries AND the head input at once. nhead stays 4 throughout.
    Arm("a1_d192", "A", ["--d-model", "192"], "d_model 128 -> 192"),
    Arm("a2_d256", "A", ["--d-model", "256"], "d_model 128 -> 256"),
    Arm("a3_d384", "A", ["--d-model", "384"], "d_model 128 -> 384"),
    Arm("a4_d512", "A", ["--d-model", "512"], "d_model 128 -> 512"),

    # --- B: everything AFTER the per-block encoders. The fusion transformer is
    # 57.5% of the control's parameters, so this is where the model already
    # spends most of its capacity.
    Arm("b1_ff512", "B", ["--dim-ff", "512"], "transformer FF 256 -> 512"),
    Arm("b2_ff1024", "B", ["--dim-ff", "1024"], "transformer FF 256 -> 1024"),
    Arm("b3_ff2048", "B", ["--dim-ff", "2048"], "transformer FF 256 -> 2048"),
    Arm("b4_layers3", "B", ["--fusion-layers", "3"], "fusion depth 2 -> 3"),
    Arm("b5_layers4", "B", ["--fusion-layers", "4"], "fusion depth 2 -> 4"),
    Arm("b6_layers6", "B", ["--fusion-layers", "6"], "fusion depth 2 -> 6"),
    Arm("b7_hidden512", "B", ["--hidden", "512"], "read-out MLP 256 -> 512"),
    Arm("b8_hidden1024", "B", ["--hidden", "1024"], "read-out MLP 256 -> 1024"),

    # --- C: inside one SharedBlockEncoder. `bins` is the token's information
    # bottleneck — a 3,423 bp locus is reduced to 2*out_channels*bins numbers
    # before the transformer ever sees it, and only bins reopens position
    # resolution. width/out_channels move together in the architecture's own 2:1
    # ratio, so C-channels is one knob, not two.
    Arm("c1_enc96", "C", ENC_W96, "encoder channels x1.5 (64/32 -> 96/48)", heavy=True),
    Arm("c2_enc128", "C", ENC_W128, "encoder channels x2 (64/32 -> 128/64)", heavy=True),
    Arm("c3_depth2", "C", ["--enc-depth", "2"], "one extra conv stage per encoder"),
    Arm("c4_depth3", "C", ["--enc-depth", "3"], "two extra conv stages per encoder"),
    Arm("c5_bins8", "C", ["--enc-bins", "8"], "pooled segments 4 -> 8"),
    Arm("c6_bins16", "C", ["--enc-bins", "16"], "pooled segments 4 -> 16"),
    Arm("c7_bins32", "C", ["--enc-bins", "32"], "pooled segments 4 -> 32"),
    Arm("c8_bins64", "C", ["--enc-bins", "64"], "pooled segments 4 -> 64"),

    # --- D: per-drug read-out. The control's ONLY per-drug parameters are one
    # d_model query each (1,408 of 460,417 — 0.3%); everything after the queries
    # is shared across all 11 drugs. This is joint_capacity's b2 question asked
    # of the arch where the shared head is most extreme.
    Arm("d1_perdrug64", "D", ["--per-drug-hidden", "64"],
        "each drug gets hidden -> 64 -> 1", scope="joint"),
    Arm("d2_perdrug128", "D", ["--per-drug-hidden", "128"],
        "each drug gets hidden -> 128 -> 1", scope="joint"),

    # --- R: training regime at BASELINE capacity. These exist so a null result
    # at the top of the ladder is readable: without them, "the big model did not
    # help" and "the big model was not trained / overfit" are the same
    # observation. The base LR is NOT swept — joint_convergence already measured
    # lr 1e-3 as catastrophic here (macro CV 0.911 -> 0.735, -> 0.583 with
    # regularization), so cosine only ever scales the recorded LR down.
    Arm("r1_cosine", "R", COSINE, "cosine LR decay, 20-epoch warmup, same base LR"),
    Arm("r2_reg", "R", REG, "head dropout 0.3 + weight decay 1e-4 (joint_capacity b3 values)"),

    # --- X: sparse crosses. Each PAIR of axes once, then the triple, then the
    # top of the ladder with and without the regime change. Bundles, not
    # single-knob arms — read them only against A/B/C/D/R.
    Arm("x1_AB", "X", ["--d-model", "256", "--dim-ff", "1024", "--fusion-layers", "4"],
        "A+B mid"),
    Arm("x2_AC", "X", ["--d-model", "256", "--enc-bins", "16"] + ENC_W128,
        "A+C mid", heavy=True),
    Arm("x3_BC", "X", ["--dim-ff", "1024", "--fusion-layers", "4", "--enc-bins", "16"] + ENC_W128,
        "B+C mid", heavy=True),
    Arm("x4_mid", "X", ["--d-model", "256", "--dim-ff", "1024", "--fusion-layers", "4",
                        "--hidden", "512", "--enc-depth", "2", "--enc-bins", "16"] + ENC_W128,
        "A+B+C mid", heavy=True),
    Arm("x5_big", "X", ["--d-model", "384", "--dim-ff", "1536", "--fusion-layers", "6",
                        "--hidden", "512", "--enc-depth", "2", "--enc-bins", "32"] + ENC_W128,
        "top of every ladder, control regime", heavy=True),
    Arm("x6_big_tuned", "X", ["--d-model", "384", "--dim-ff", "1536", "--fusion-layers", "6",
                              "--hidden", "512", "--enc-depth", "2", "--enc-bins", "32"]
        + ENC_W128 + COSINE + REG,
        "x5 + cosine + regularization (= x5 + r1 + r2)", heavy=True),
    Arm("x7_big_perdrug", "X", ["--d-model", "384", "--dim-ff", "1536", "--fusion-layers", "6",
                                "--hidden", "512", "--enc-depth", "2", "--enc-bins", "32",
                                "--per-drug-hidden", "64"] + ENC_W128 + COSINE + REG,
        "x6 + per-drug heads (= x6 + d1)", scope="joint", heavy=True),
]

AXES = ("A", "B", "C", "D", "R", "X")


def select(axes, scope, mods):
    """(arm, modality-set, scope) triples this invocation covers."""
    out = []
    for arm in ARMS:
        if arm.axis not in axes:
            continue
        for m in mods:
            for sc in scope:
                if sc == "single" and arm.scope == "joint":
                    continue
                out.append((arm, m, sc))
    return out


def resources(arm, sc):
    """SLURM ask for one job of this arm.

    Joint defaults match full_run_v2's joint submission (64G / 6 cpus / 48h);
    single-drug matches its single-drug one (48G / 4 cpus / 16h). Heavy arms get
    double the time and a >=23 GB card: at --enc-width 128 the conv1 activations
    for 38 blocks at batch 128 are roughly 4x the control's, and the control
    already needed expandable_segments to survive fragmentation on 11 GB cards.
    """
    joint = sc == "joint"
    args = ["--mem", "64G" if joint else "48G", "--cpus", "6" if joint else "4",
            "--gpus", "1", "--time", ("96:00:00" if arm.heavy else "48:00:00") if joint
            else ("24:00:00" if arm.heavy else "16:00:00")]
    if arm.heavy:
        args += ["--constraint", "vram23"]
    return args


def command(arm, mods, sc, args):
    cmd = ["python", "scripts/sbatch_all_runs.py",
           "--experiments", f"{mods}__setfusion",
           "--run-prefix", f"{RUN_PREFIX}/{arm.name}_",
           *PROTOCOL, *resources(arm, sc), "--delay", str(args.delay)]
    cmd += ["--multidrug"] if sc == "joint" else ["--drugs", *args.drugs]
    cmd += arm.flags
    if args.weights_dir:
        cmd += ["--weights-dir", args.weights_dir]
    return cmd


# ---------------------------------------------------------------------------
# parameter table — what each arm actually builds, before a single GPU-hour
# ---------------------------------------------------------------------------

def _control_blocks(mods, scope):
    """(keys, specs, drugs) off a finished full_run_v2 cell, so the table is
    built on the REAL block shapes rather than a guess."""
    from models.net import parse_block_key
    tag = "+".join(MODALITY_SETS[mods])
    if scope == "joint":
        p = EXP / CONTROL_RUN / f"multidrug_{mods}__setfusion" / f"multidrug__{tag}.json"
    else:
        p = EXP / CONTROL_RUN / f"{mods}__setfusion" / f"ISONIAZID__{tag}.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    drugs = j["drugs"] if scope == "joint" else [j["drug"]]
    return ([parse_block_key(b) for b in j["blocks"]],
            [tuple(s) for s in j["branch_specs"]], drugs, j["n_params"])


def _flags_to_kwargs(flags):
    """The arm's CLI flags -> SetFusionNet kwargs (the same mapping the runners
    do). Training-only flags are ignored here — they build the same model."""
    cli = {"--d-model": "d_model", "--nhead": "nhead", "--fusion-layers": "layers",
           "--dim-ff": "dim_ff", "--fusion-dropout": "dropout",
           "--enc-width": "enc_width", "--enc-out-channels": "enc_out_channels",
           "--enc-depth": "enc_depth", "--enc-bins": "bins",
           "--hidden": "hidden", "--per-drug-hidden": "per_drug_hidden"}
    kw, it = {}, iter(flags)
    for tok in it:
        if tok in cli:
            val = next(it)
            kw[cli[tok]] = float(val) if "." in val or "e-" in val else int(val)
    return kw


def params_table(args):
    import warnings
    warnings.filterwarnings("ignore")
    from models import SetFusionNet

    rows = []
    for mods in args.mods:
        for scope in args.scope:
            ctl = _control_blocks(mods, scope)
            if ctl is None:
                print(f"  (no {CONTROL_RUN} control for {mods}/{scope} — skipped)")
                continue
            keys, specs, drugs, recorded = ctl
            base = sum(p.numel() for p in SetFusionNet(
                keys, specs, drug_names=drugs if len(drugs) > 1 else None,
                n_drugs=len(drugs)).parameters())
            status = "OK" if base == recorded else f"!= recorded {recorded:,}"
            print(f"\n{mods} / {scope}-drug — {len(specs)} blocks, "
                  f"{len(drugs)} drug(s); control {base:,} params ({status})")
            print(f"  {'arm':16s} {'axis':5s} {'params':>13s} {'xctl':>7s}  note")
            print(f"  {'(control)':16s} {'-':5s} {base:>13,} {1.0:>7.2f}  full_run_v2")
            for arm in ARMS:
                if scope == "single" and arm.scope == "joint":
                    continue
                if arm.axis not in args.axes:
                    continue
                kw = _flags_to_kwargs(arm.flags)
                n = sum(p.numel() for p in SetFusionNet(
                    keys, specs, drug_names=drugs if len(drugs) > 1 else None,
                    n_drugs=len(drugs), **kw).parameters())
                print(f"  {arm.name:16s} {arm.axis:5s} {n:>13,} {n/base:>7.2f}  {arm.note}")
                rows.append({"modality_set": mods, "scope": scope, "arm": arm.name,
                             "axis": arm.axis, "n_params": n,
                             "ratio_vs_control": round(n / base, 3),
                             "note": arm.note, "flags": " ".join(arm.flags)})
    if args.write_csv and rows:
        import csv
        out = EXP / RUN_PREFIX
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "arm_params.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}/arm_params.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axes", nargs="+", default=list(AXES), metavar="AXIS",
                    help=f"which axes to cover (default: all of {list(AXES)})")
    ap.add_argument("--scope", nargs="+", default=["joint", "single"],
                    choices=["joint", "single"],
                    help="joint = one job per arm; single = one job per (arm x drug)")
    ap.add_argument("--mods", nargs="+", default=list(MODALITY_SETS),
                    choices=list(MODALITY_SETS),
                    help="modality set(s) to hold fixed (default: both)")
    ap.add_argument("--drugs", nargs="+", default=["all"],
                    help="single-drug scope only (default: all 11)")
    ap.add_argument("--arms", nargs="+", default=None, metavar="NAME",
                    help="restrict to these arm names (default: every arm on the "
                         "selected axes)")
    ap.add_argument("--weights-dir", default=None,
                    help="override the shared weights root (default: "
                         "bigtb_ref.MODEL_WEIGHTS_DIR)")
    ap.add_argument("--delay", type=float, default=0.5,
                    help="seconds between individual sbatch calls (default: 0.5)")
    ap.add_argument("--params", action="store_true",
                    help="print the parameter count each arm builds and exit "
                         "(no cluster, no submission)")
    ap.add_argument("--write-csv", action="store_true",
                    help="with --params, also write arm_params.csv into the run folder")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the sbatch_all_runs.py invocations, submit nothing")
    args = ap.parse_args()

    bad = [a for a in args.axes if a not in AXES]
    if bad:
        sys.exit(f"unknown axis/axes {bad}; choose from {list(AXES)}")
    if args.arms:
        known = {a.name for a in ARMS}
        unknown = [a for a in args.arms if a not in known]
        if unknown:
            sys.exit(f"unknown arm(s) {unknown}; choose from {sorted(known)}")

    if args.params:
        params_table(args)
        return

    todo = [(arm, m, sc) for arm, m, sc in select(args.axes, args.scope, args.mods)
            if not args.arms or arm.name in args.arms]
    if not todo:
        sys.exit("nothing selected")

    n_drugs = 11 if args.drugs == ["all"] else len(args.drugs)
    jobs = sum(1 if sc == "joint" else n_drugs for _, _, sc in todo)
    print(f"{RUN_PREFIX}: {len({a.name for a, _, _ in todo})} arms x "
          f"{len(args.mods)} modality set(s) x {len(args.scope)} scope(s) "
          f"= {len(todo)} cells, {jobs} SLURM jobs")
    print(f"control (not submitted): results/experiments/{CONTROL_RUN}/"
          "{,multidrug_}{mods}__setfusion")
    print(f"protocol: {' '.join(PROTOCOL)}\n")
    if args.dry_run:
        for arm, m, sc in todo:
            print(f"# {arm.name} / {m} / {sc}-drug — {arm.note}")
            print(" ".join(command(arm, m, sc, args)) + "\n")
        print("DRY RUN — nothing submitted.")
        return

    failed = []
    for i, (arm, m, sc) in enumerate(todo, 1):
        cmd = command(arm, m, sc, args)
        print(f"[{i}/{len(todo)}] {arm.name} / {m} / {sc}: {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, cwd=PROJECT)
        if r.returncode != 0:
            failed.append(f"{arm.name}/{m}/{sc}")
            print(f"  !! sbatch_all_runs.py exited {r.returncode}", flush=True)
        time.sleep(args.delay)
    print(f"\nsubmitted {len(todo) - len(failed)}/{len(todo)} cells"
          + (f"; FAILED: {failed}" if failed else ""))
    print("Monitor:  squeue -u $USER   |   cancel all:  scancel -u $USER")


if __name__ == "__main__":
    main()
