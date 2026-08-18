"""
Checks for the transformer encoder path — the tunable branch encoder and the
MD-CNN transformer trunk added for results/experiments/transformer_run.

Pure unit tests on synthetic tensors (no data, no training), seconds on CPU:

    python tests/test_transformer_encoder.py

The two things worth guarding, because both are silent when they break:

  * **Nothing changes by default.** Every model must be parameter-identical with
    no `transformer` dict, with an empty one, and with one holding exactly
    TRANSFORMER_DEFAULTS. That is what keeps the full_run / full_run_v2
    checkpoints loadable and their numbers reproducible.
  * **A requested transformer is actually built.** `--arch mdcnn` used to ignore
    the encoder choice entirely and print "n/a (mdcnn)". If it silently fell back
    to the conv trunk, a transformer sweep arm would produce the CNN control's
    numbers under a transformer folder name — the exact failure the setfusion
    capacity flags are guarded against.

Also asserted: the trunk keeps MD-CNN's defining property (layer 1 spans every
locus at once, so every locus reaches every output), the knobs move the parameter
count in the right direction, and a config round-trips through the checkpoint
layer with weights transferring.
"""
import json
import sys
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from models import (  # noqa: E402
    MDCNN_TRUNKS,
    MDCNNNet,
    MDCNNTransformerTrunk,
    MDCNNTrunk,
    MultiDrugNet,
    MultiModalNet,
    TRANSFORMER_DEFAULTS,
    make_encoder,
)
from training.checkpoint import build_model_from_config, model_config  # noqa: E402
from training.multimodal import _build_model  # noqa: E402

_RESULTS = []
SPECS = [(5, 900), (5, 2223), (20, 300), (3, 300)]
TF = {"d_model": 96, "nhead": 4, "layers": 3, "dim_ff": 384}
HEAD = {"hidden": 256, "dropout": 0.0, "per_drug_hidden": 0}


class _Block:
    """Minimal stand-in for a loader FeatureBlock."""

    def __init__(self, name, modality, channels, length):
        self.name, self.modality = name, modality
        self.channels, self.length = channels, length

    def spec(self):
        return (self.channels, self.length)


BLOCKS = [_Block("dna:inhA", "dna", 5, 900), _Block("dna:katG", "dna", 5, 2223),
          _Block("protein:inhA", "protein", 20, 300),
          _Block("biophysical:inhA", "biophysical", 3, 300)]


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
        _RESULTS.append(True)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        _RESULTS.append(False)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


# --- nothing changes by default ---------------------------------------------

def _test_defaults_are_inert():
    """None / {} / exactly-the-defaults must all give the identical model."""
    for build in (
        lambda tf: MDCNNNet(SPECS, encoder="cnn", transformer=tf),
        lambda tf: MultiModalNet(SPECS, transformer=tf),
        lambda tf: MultiDrugNet(SPECS, ["A", "B"], transformer=tf),
    ):
        base = nparams(build(None))
        assert nparams(build({})) == base, "empty dict changed the model"
        assert nparams(build(dict(TRANSFORMER_DEFAULTS))) == base, \
            "passing TRANSFORMER_DEFAULTS explicitly changed the model"


def _test_cnn_trunk_is_still_the_mdcnn_default():
    assert nparams(MDCNNNet(SPECS)) == nparams(MDCNNNet(SPECS, encoder="cnn"))
    net = MDCNNNet(SPECS)
    assert all(isinstance(t, MDCNNTrunk) for t in net.trunks), \
        "mdcnn default must remain the conv trunk"


# --- a requested transformer is actually built -------------------------------

def _test_mdcnn_transformer_is_not_silently_ignored():
    net = MDCNNNet(SPECS, encoder="transformer", transformer=TF)
    assert all(isinstance(t, MDCNNTransformerTrunk) for t in net.trunks), \
        "encoder='transformer' did not build transformer trunks"
    assert nparams(net) != nparams(MDCNNNet(SPECS)), \
        "transformer mdcnn has the same size as the conv one — likely ignored"
    # every trunk mean-pools to d_model, so the head sees n_groups * d_model
    assert all(t.out_features == TF["d_model"] for t in net.trunks)


def _test_mdcnn_encoder_mix_is_refused():
    """A per-modality mix has no meaning when a trunk can span modalities."""
    try:
        _build_model("mdcnn", BLOCKS, SPECS, ["cnn", "transformer", "cnn", "cnn"],
                     1, None, head=HEAD, transformer=TF)
    except ValueError as e:
        assert "ONE encoder" in str(e), f"wrong error: {e}"
        return
    raise AssertionError("a mixed encoder request under mdcnn should raise")


def _test_unknown_encoder_is_refused():
    for fn in (lambda: MDCNNNet(SPECS, encoder="bogus"),
               lambda: make_encoder("bogus", 5, 100)):
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError("an unknown encoder key should raise")


def _test_cnn_encoder_rejects_transformer_kwargs():
    """make_encoder must not quietly swallow knobs the CNN cannot use."""
    try:
        make_encoder("cnn", 5, 100, {"d_model": 96})
    except TypeError:
        raise AssertionError("make_encoder should not forward knobs to CNNEncoder")
    enc = make_encoder("cnn", 5, 100, {"d_model": 96})
    assert enc.out_features == make_encoder("cnn", 5, 100).out_features


# --- the trunk keeps MD-CNN's defining property ------------------------------

def _test_trunk_shapes_and_registry():
    assert set(MDCNN_TRUNKS) == {"cnn", "transformer"}
    trunk = MDCNNTransformerTrunk(3, 5, 900, **TF)
    out = trunk(torch.zeros(2, 3, 5, 900))
    assert out.shape == (2, TF["d_model"]), out.shape
    # patch embedding: kernel == stride == patch, so ~L/patch tokens
    assert trunk.pos.shape[1] == (900 - 9) // 9 + 1, trunk.pos.shape


def _test_every_locus_reaches_the_output():
    """Layer 1 spans all loci at once — that is what makes this MD-CNN rather
    than late fusion, so perturbing ANY locus plane must move the logits."""
    torch.manual_seed(0)
    net = MDCNNNet([(5, 300), (5, 300), (5, 300)], encoder="transformer",
                   transformer=TF).eval()
    xs = [torch.zeros(1, 5, 300) for _ in range(3)]
    with torch.no_grad():
        base = net(xs).clone()
    for i in range(3):
        bumped = [x.clone() for x in xs]
        bumped[i][0, 0, :] = 1.0
        with torch.no_grad():
            got = net(bumped)
        assert not torch.allclose(base, got, atol=1e-7), \
            f"locus {i} does not reach the output — loci are not being mixed"


def _test_backward_reaches_every_parameter():
    net = MDCNNNet([(5, 300), (5, 300)], encoder="transformer", transformer=TF)
    net(([torch.randn(2, 5, 300), torch.randn(2, 5, 300)])).sum().backward()
    missing = [n for n, p in net.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing[:5]}"


# --- the knobs do what they say ----------------------------------------------

def _test_knobs_move_the_parameter_count():
    def size(**over):
        return nparams(MDCNNNet(SPECS, encoder="transformer",
                                transformer={**TF, **over}))
    base = size()
    assert size(d_model=192) > base, "d_model did not increase size"
    assert size(layers=6) > base, "layers did not increase size"
    assert size(dim_ff=768) > base, "dim_ff did not increase size"
    # patch is the one knob that SHRINKS the model as it grows: fewer tokens
    # means a smaller positional embedding (and the tokenize kernel widens),
    # so just assert it changes the model rather than a direction.
    assert size(patch=27) != base, "patch did not change the model"


def _test_transformer_is_length_agnostic_where_cnn_is_not():
    """The whole reason the capacity knobs exist: a CNN branch's width scales
    with sequence length, a mean-pooled transformer's does not."""
    short = make_encoder("transformer", 5, 500, TF).out_features
    long = make_encoder("transformer", 5, 5000, TF).out_features
    assert short == long == TF["d_model"], (short, long)
    assert make_encoder("cnn", 5, 5000).out_features > \
        make_encoder("cnn", 5, 500).out_features


# --- checkpoint round-trip ---------------------------------------------------

def _test_config_round_trip():
    for arch in ("mdcnn", "late_fusion", "cisfusion"):
        et = ["transformer"] * len(BLOCKS)
        orig = _build_model(arch, BLOCKS, SPECS, et, 1, None,
                            branch_models={m: "transformer" for m in
                                           ("dna", "protein", "biophysical")},
                            default_encoder="transformer", head=HEAD,
                            transformer=TF)
        cfg = {"model": model_config(
            arch=arch, blocks=BLOCKS, encoder_types=et, drug_names=["D"],
            out_bias=None, head=HEAD, default_encoder="transformer",
            branch_models={m: "transformer" for m in
                           ("dna", "protein", "biophysical")},
            n_params=nparams(orig), transformer=TF)}
        cfg = json.loads(json.dumps(cfg))       # a real serialize / parse cycle
        rebuilt = build_model_from_config(cfg)
        assert nparams(rebuilt) == nparams(orig), \
            f"{arch}: {nparams(rebuilt)} != {nparams(orig)}"
        rebuilt.load_state_dict(orig.state_dict())   # shapes must match exactly


def _test_config_without_the_key_still_loads():
    """Configs written before these knobs existed carry no 'transformer' key."""
    et = ["cnn"] * len(BLOCKS)
    orig = _build_model("mdcnn", BLOCKS, SPECS, et, 1, None, head=HEAD)
    cfg = {"model": model_config(arch="mdcnn", blocks=BLOCKS, encoder_types=et,
                                 drug_names=["D"], out_bias=None, head=HEAD,
                                 n_params=nparams(orig))}
    cfg["model"].pop("transformer")             # simulate the older schema
    rebuilt = build_model_from_config(json.loads(json.dumps(cfg)))
    assert nparams(rebuilt) == nparams(orig)
    rebuilt.load_state_dict(orig.state_dict())


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    check("defaults are inert (None / {} / explicit defaults)", _test_defaults_are_inert)
    check("mdcnn still defaults to the conv trunk", _test_cnn_trunk_is_still_the_mdcnn_default)
    check("mdcnn transformer is not silently ignored", _test_mdcnn_transformer_is_not_silently_ignored)
    check("mdcnn refuses a per-modality encoder mix", _test_mdcnn_encoder_mix_is_refused)
    check("unknown encoder key rejected", _test_unknown_encoder_is_refused)
    check("CNN encoder never sees transformer kwargs", _test_cnn_encoder_rejects_transformer_kwargs)
    check("trunk shapes + registry", _test_trunk_shapes_and_registry)
    check("every locus reaches the output", _test_every_locus_reaches_the_output)
    check("backward reaches every parameter", _test_backward_reaches_every_parameter)
    check("knobs move the parameter count", _test_knobs_move_the_parameter_count)
    check("transformer is length-agnostic, CNN is not", _test_transformer_is_length_agnostic_where_cnn_is_not)
    check("config round-trips through the checkpoint layer", _test_config_round_trip)
    check("pre-knob config still loads", _test_config_without_the_key_still_loads)
    print(f"\n{sum(_RESULTS)}/{len(_RESULTS)} checks passed")
    sys.exit(0 if all(_RESULTS) else 1)
