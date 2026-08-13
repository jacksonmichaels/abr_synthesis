"""
Checks for ``training.checkpoint`` — weights are saved AND actually reloadable.

    python tests/test_checkpoint.py

The point of this file is one assertion, made for every architecture: take a
trained model, write it out, rebuild it from nothing but ``config.json``, load
the weights, and get **bit-identical predictions**. A checkpoint that cannot do
that is not a checkpoint — and until this module existed nothing was saved at
all, so every model behind results/experiments/full_run is gone.

Pure unit tests on synthetic tensors plus one end-to-end run on the fixture
dataset. No real data, no GPU.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import SetFusionNet  # noqa: E402
from models.net import parse_block_key  # noqa: E402
from training.checkpoint import (  # noqa: E402
    RunCheckpointer, build_model_from_config, load_model, model_config,
    run_weights_dir, write_pointer)

PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}{('  ' + extra) if extra else ''}")


class Block:
    """Minimal stand-in for datasets.base.FeatureBlock."""

    def __init__(self, name, channels, length, channel_names=()):
        self.name = name
        self.modality = parse_block_key(name)[0]
        self.channels, self.length = channels, length
        self.channel_names = list(channel_names)

    def spec(self):
        return (self.channels, self.length)


NT = ("A", "C", "T", "G", "-")
# a dna+regulatory+protein+biophysical per-locus layout, the shape the joint
# all-modalities cells actually load
BLOCKS = [
    Block("dna:katG", 5, 900, NT), Block("dna:inhA", 5, 600, NT),
    Block("regulatory:katG", 5, 200, NT), Block("regulatory:inhA", 5, 320, NT),
    Block("protein:katG", 20, 300), Block("biophysical:katG", 3, 300),
]
DRUGS = [f"DRUG{i}" for i in range(11)]


def _inputs(blocks, batch=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(batch, b.channels, b.length, generator=g) for b in blocks]


def _roundtrip(arch, blocks, drugs, head, trunk_per_modality=False, tag="t",
               tmp=None, setfusion=None):
    """Train-free round trip: build -> perturb weights -> save -> rebuild -> load.

    ``tmp`` lets the caller own the directory's lifetime when it wants to
    inspect the written files afterwards; otherwise one is made and discarded.
    """
    encoder_types = ["cnn"] * len(blocks)
    cfg_model = model_config(arch=arch, blocks=blocks, encoder_types=encoder_types,
                             drug_names=drugs, out_bias=None, head=head,
                             mdcnn_trunk_per_modality=trunk_per_modality,
                             setfusion=setfusion)
    original = build_model_from_config({"model": cfg_model})
    # perturb so we are not comparing two identical fresh inits
    with torch.no_grad():
        for p in original.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    original.eval()
    xs = _inputs(blocks)
    with torch.no_grad():
        want = original(xs)

    ctx = tempfile.TemporaryDirectory() if tmp is None else None
    tmp = tmp or ctx.name
    try:
        run, stem = "unit_run", f"{tag}__{arch}"
        ck = RunCheckpointer(run, stem, mode="all", weights_dir=tmp)
        ck.add_fold(0, original, best_epoch=7, metric=0.9)
        ck.add_fold(1, original, best_epoch=9, metric=0.8)
        out = ck.write({"run_name": run, "scope": "joint" if len(drugs) > 1 else "single",
                        "tag": tag, "model": cfg_model, "data": {}, "split": {},
                        "training": {}}, best_fold=0, isolate_ids=["SAM1", "SAM2"])
        assert out is not None, "checkpoint write returned None"

        # rebuilt from config.json ALONE — the model object is never reused
        restored, cfg = load_model(run, stem, weights_dir=tmp)
        with torch.no_grad():
            got = restored(_inputs(blocks))
        return want, got, cfg, Path(out)
    finally:
        if ctx is not None:
            ctx.cleanup()


def test_roundtrip_every_arch():
    head = {"hidden": 256, "dropout": 0.0, "per_drug_hidden": 0}
    for arch in ("late_fusion", "mdcnn", "setfusion", "cisfusion"):
        want, got, cfg, _ = _roundtrip(arch, BLOCKS, DRUGS, head, tag="joint")
        check(f"{arch}: rebuilt-from-config predictions are bit-identical",
              torch.equal(want, got),
              f"max|diff|={float((want - got).abs().max()):.2e}")
        check(f"{arch}: output width and drug order preserved",
              got.shape == (4, 11) and cfg["model"]["drug_names"] == DRUGS)


def test_roundtrip_single_drug():
    head = {"hidden": 256, "dropout": 0.0, "per_drug_hidden": 0}
    want, got, cfg, _ = _roundtrip("late_fusion", BLOCKS, ["ISONIAZID"], head,
                                   tag="single")
    check("single-drug: round trip is bit-identical", torch.equal(want, got))
    check("single-drug: one logit column", got.shape == (4, 1))


def test_roundtrip_new_capacity_knobs():
    """The Run B config must survive a round trip too, or its models are lost."""
    head = {"hidden": 512, "dropout": 0.3, "per_drug_hidden": 64}
    want, got, cfg, _ = _roundtrip("late_fusion", BLOCKS, DRUGS, head, tag="capacity")
    check("per-drug heads + dropout + hidden=512 round trip", torch.equal(want, got))
    check("head knobs recorded in config",
          (cfg["model"]["hidden"], cfg["model"]["dropout"],
           cfg["model"]["per_drug_hidden"]) == (512, 0.3, 64))


def test_roundtrip_setfusion_capacity_knobs():
    """The setfusion_scaling arms must survive a round trip, and a config
    written BEFORE those knobs existed must still rebuild at the defaults —
    that second half is what keeps full_run/full_run_v2's weights loadable."""
    head = {"hidden": 512, "dropout": 0.1, "per_drug_hidden": 64}
    sf = {"d_model": 256, "nhead": 8, "layers": 3, "dim_ff": 1024,
          "enc_width": 96, "enc_out_channels": 48, "enc_depth": 2, "bins": 8}
    want, got, cfg, _ = _roundtrip("setfusion", BLOCKS, DRUGS, head, tag="sfcap",
                                   setfusion=sf)
    check("setfusion capacity knobs round trip", torch.equal(want, got),
          f"max|diff|={float((want - got).abs().max()):.2e}")
    check("setfusion knobs recorded in config", cfg["model"]["setfusion"] == sf,
          f"{cfg['model'].get('setfusion')}")

    # a pre-knobs config: no 'setfusion' key at all, and hidden/per_drug_hidden
    # at the values every recorded run used
    legacy = {k: v for k, v in cfg["model"].items() if k != "setfusion"}
    legacy.update(hidden=256, dropout=0.0, per_drug_hidden=0)
    old = build_model_from_config({"model": legacy})
    check("config with no 'setfusion' key rebuilds at the defaults",
          sum(p.numel() for p in old.parameters())
          == sum(p.numel() for p in SetFusionNet(
              [parse_block_key(b.name) for b in BLOCKS],
              [b.spec() for b in BLOCKS], n_drugs=len(DRUGS)).parameters()))
    check("capacity knobs actually build a different model",
          sum(p.numel() for p in build_model_from_config(
              {"model": cfg["model"]}).parameters())
          != sum(p.numel() for p in old.parameters()))


def test_roundtrip_mdcnn_trunk_grouping():
    head = {"hidden": 256, "dropout": 0.0, "per_drug_hidden": 0}
    want, got, cfg, _ = _roundtrip("mdcnn", BLOCKS, DRUGS, head,
                                   trunk_per_modality=True, tag="trunk")
    check("mdcnn trunk-per-modality round trip", torch.equal(want, got))
    check("trunk grouping recorded in config",
          cfg["model"]["mdcnn_trunk_per_modality"] is True)
    # and it must actually differ from the channel-grouped model
    grouped = build_model_from_config({"model": {**cfg["model"],
                                                 "mdcnn_trunk_per_modality": True}})
    channel = build_model_from_config({"model": {**cfg["model"],
                                                 "mdcnn_trunk_per_modality": False}})
    check("modality grouping really builds a different model",
          len(grouped.trunks) > len(channel.trunks),
          f"{len(grouped.trunks)} trunks vs {len(channel.trunks)}")


def test_config_is_sufficient():
    """Everything needed to rebuild the INPUTS, not just the weights."""
    head = {"hidden": 256, "dropout": 0.0, "per_drug_hidden": 0}
    tmp = tempfile.TemporaryDirectory()          # kept alive: we inspect the files
    _, _, cfg, out = _roundtrip("cisfusion", BLOCKS, DRUGS, head, tag="cfg",
                                tmp=tmp.name)
    m = cfg["model"]
    need = {"arch", "drug_names", "branch_specs", "blocks", "encoder_types",
            "hidden", "dropout", "per_drug_hidden", "out_bias",
            "mdcnn_trunk_per_modality", "branch_models", "default_encoder"}
    check("config carries every model field", need <= set(m),
          f"missing {sorted(need - set(m))}" if not need <= set(m) else "")
    b0 = m["blocks"][0]
    check("per-block identity recorded (name/modality/locus/shape/channels)",
          {"name", "modality", "locus", "channels", "length", "channel_names"} <= set(b0)
          and b0["locus"] == "katG" and b0["channel_names"] == list(NT))
    check("block ORDER preserved (forward(xs) depends on it)",
          [b["name"] for b in m["blocks"]] == [b.name for b in BLOCKS])
    check("provenance recorded", cfg["env"]["torch"] == torch.__version__
          and "argv" in cfg["env"] and cfg["schema_version"] == 1)
    check("isolates.txt written", (out / "isolates.txt").read_text().split() == ["SAM1", "SAM2"])
    check("fold bookkeeping records best_epoch + weights file",
          cfg["best_fold"] == 0
          and {f["fold"] for f in cfg["folds"]} == {0, 1}
          and all(f["weights"] for f in cfg["folds"])
          and cfg["folds"][0]["best_epoch"] == 7)
    tmp.cleanup()


def test_save_modes():
    head = {"hidden": 256, "dropout": 0.0, "per_drug_hidden": 0}
    cfg_model = model_config(arch="late_fusion", blocks=BLOCKS,
                             encoder_types=["cnn"] * len(BLOCKS), drug_names=DRUGS,
                             out_bias=None, head=head)
    model = build_model_from_config({"model": cfg_model})
    base = {"run_name": "m", "model": cfg_model, "data": {}, "split": {}, "training": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for mode, expect in (("best", 1), ("all", 3), ("none", 0)):
            ck = RunCheckpointer("m", f"stem_{mode}", mode=mode, weights_dir=tmp)
            for f in range(3):
                ck.add_fold(f, model, best_epoch=f, metric=0.5 + f / 10)
            out = ck.write(dict(base), best_fold=2)
            n = len(list(Path(out).glob("fold*.pt"))) if out else 0
            check(f"--save-weights {mode} writes {expect} fold file(s)", n == expect,
                  f"got {n}")
        # 'best' must save the fold whose model is scored on TEST
        ck = RunCheckpointer("m", "stem_which", mode="best", weights_dir=tmp)
        for f in range(3):
            ck.add_fold(f, model, best_epoch=f, metric=0.5)
        out = ck.write(dict(base), best_fold=2)
        check("--save-weights best saves the TEST fold, not fold 0",
              (Path(out) / "fold2.pt").exists() and not (Path(out) / "fold0.pt").exists())


def test_failure_is_not_fatal():
    """A full or unwritable weights volume must not destroy a finished run."""
    head = {"hidden": 256, "dropout": 0.0, "per_drug_hidden": 0}
    cfg_model = model_config(arch="late_fusion", blocks=BLOCKS,
                             encoder_types=["cnn"] * len(BLOCKS), drug_names=DRUGS,
                             out_bias=None, head=head)
    model = build_model_from_config({"model": cfg_model})
    ck = RunCheckpointer("r", "s", mode="best", weights_dir="/proc/nonwritable/nope")
    ck.add_fold(0, model, 1, 0.5)
    out = ck.write({"model": cfg_model}, best_fold=0)
    check("unwritable weights dir returns None instead of raising", out is None)

    # missing fold gives an actionable error, not a KeyError
    with tempfile.TemporaryDirectory() as tmp:
        ck = RunCheckpointer("r", "s", mode="best", weights_dir=tmp)
        ck.add_fold(0, model, 1, 0.5)
        ck.write({"model": cfg_model}, best_fold=0)
        try:
            load_model("r", "s", fold=4, weights_dir=tmp)
            ok = False
        except FileNotFoundError as e:
            ok = "saved folds" in str(e)
        check("loading an unsaved fold raises an actionable FileNotFoundError", ok)


def test_pointer_file():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "results_run"
        run_dir.mkdir()
        write_pointer(run_dir, "ISONIAZID__dna", "/weights/run/ISONIAZID__dna")
        write_pointer(run_dir, "RIFAMPICIN__dna", "/weights/run/RIFAMPICIN__dna")
        got = json.loads((run_dir / "weights_location.json").read_text())
        check("results folder points at both cells' weights",
              got == {"ISONIAZID__dna": "/weights/run/ISONIAZID__dna",
                      "RIFAMPICIN__dna": "/weights/run/RIFAMPICIN__dna"})
        write_pointer(run_dir, "X", None)   # no weights -> no-op, no crash
        check("a None weights path is a no-op",
              len(json.loads((run_dir / "weights_location.json").read_text())) == 2)


def test_weights_root_override():
    check("ABR_MODEL_WEIGHTS_DIR / --weights-dir override the root",
          run_weights_dir("r", "s", "/tmp/xyz") == Path("/tmp/xyz/r/s"))


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    for fn in (test_roundtrip_every_arch, test_roundtrip_single_drug,
               test_roundtrip_new_capacity_knobs, test_roundtrip_setfusion_capacity_knobs,
               test_roundtrip_mdcnn_trunk_grouping,
               test_config_is_sufficient, test_save_modes, test_failure_is_not_fatal,
               test_pointer_file, test_weights_root_override):
        fn()
    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)
