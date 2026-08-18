"""
Model persistence: weights + a config that fully rebuilds the model.

Until this module existed nothing was ever checkpointed. ``EarlyStopper``
snapshotted the best epoch into an in-memory dict, ``restore()`` loaded it back,
and the process then exited — so every TEST metric in
``results/experiments/full_run`` was produced by a model that no longer exists.
240 SLURM jobs of compute left json, png and csv and nothing you could re-score,
calibrate, inspect, or deploy.

What gets written
-----------------
Weights go to ``bigtb_ref.MODEL_WEIGHTS_DIR`` (the lab's shared large-storage
volume), NOT to ``results/`` — results/ is git-ignored workspace storage and a
full grid of checkpoints is tens of GB::

    {MODEL_WEIGHTS_DIR}/{run_name}/{stem}/
        config.json     schema v1 — everything needed to rebuild (below)
        isolates.txt    the exact isolate row order the model was fit on
        fold{k}.pt      state_dict for each saved CV fold

``stem`` is ``{DRUG}__{tag}`` for single-drug runs and ``multidrug__{tag}`` for
joint ones, matching the json filenames in the run folder. The run folder also
gets a ``weights_location.json`` pointer so a result and its weights can always
be walked between.

"Fully rebuild" means what it says
----------------------------------
``config.json`` carries the four things a state_dict cannot:

1. **model** — arch, per-block (name, modality, locus, channels, length), the
   encoder assignment, the head shape (hidden / dropout / per_drug_hidden /
   out_bias) and ``drug_names`` in label-column order, so ``logits[:, i]`` is
   attributable. A bare ``.pt`` is unusable without these: block ORDER alone
   decides what ``forward(xs)`` means.
2. **data** — the directories, the resolved locus and regulatory-region lists in
   load order, and the loader switches (per_modality_branch, all_regulatory,
   extra_loci). This is what lets you rebuild the *inputs* the weights expect.
3. **split** — the test isolate IDs verbatim, not just "seed 42", so a
   re-evaluation is immune to scikit-learn changing its shuffling.
4. **env** — git commit, torch version, argv. What actually ran.

Round-tripping is enforced by ``tests/test_checkpoint.py``: rebuild from config,
load the weights, and assert the predictions match the original model bit for
bit.

    from training.checkpoint import load_model
    model, cfg = load_model("full_run_v2", "multidrug__dna+regulatory")
    cfg["model"]["drug_names"]      # which column is which drug
"""
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

import torch

from bigtb_ref import MODEL_WEIGHTS_DIR
from models import (SETFUSION_DEFAULTS, CisFusionNet, MDCNNNet, MultiDrugNet,
                    MultiModalNet, SetFusionNet)
from models.net import parse_block_key

SCHEMA_VERSION = 1
SAVE_CHOICES = ("best", "all", "none")


def weights_root(weights_dir=None):
    return Path(weights_dir or os.environ.get("ABR_MODEL_WEIGHTS_DIR")
                or MODEL_WEIGHTS_DIR)


def run_weights_dir(run_name, stem, weights_dir=None):
    """The directory this run+cell's artifacts belong in (not created here)."""
    return weights_root(weights_dir) / run_name / stem


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10,
                             cwd=Path(__file__).resolve().parent.parent)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _env_block():
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "git_commit": _git_commit(),
        "hostname": socket.gethostname(),
        "argv": sys.argv,
    }


# ---------------------------------------------------------------------------
# building the config
# ---------------------------------------------------------------------------

def block_records(blocks):
    """Per-block identity: what forward(xs) expects, in order."""
    out = []
    for b in blocks:
        modality, locus = parse_block_key(b.name)
        c, l = b.spec()
        out.append({"name": b.name, "modality": modality, "locus": locus,
                    "channels": int(c), "length": int(l),
                    "channel_names": list(getattr(b, "channel_names", []) or [])})
    return out


def model_config(*, arch, blocks, encoder_types, drug_names, out_bias, head,
                 mdcnn_trunk_per_modality=False, branch_models=None,
                 default_encoder="cnn", n_params=None, setfusion=None,
                 transformer=None):
    """The `model` section — sufficient on its own for build_model_from_config.

    ``setfusion`` and ``transformer`` record only the capacity knobs that DIFFER
    from their defaults, so a config written before those knobs existed (key
    absent) rebuilds to exactly the model it always did — which is what keeps
    the full_run / full_run_v2 checkpoints loadable."""
    return {
        "arch": arch,
        "setfusion": dict(setfusion or {}),
        "transformer": dict(transformer or {}),
        "drug_names": list(drug_names),          # logits[:, i] == drug_names[i]
        "n_drugs": len(drug_names),
        "branch_specs": [[int(c), int(l)] for c, l in (b.spec() for b in blocks)],
        "blocks": block_records(blocks),
        "encoder_types": list(encoder_types),
        "branch_models": dict(branch_models or {}),
        "default_encoder": default_encoder,
        "out_bias": out_bias,
        "hidden": head["hidden"],
        "dropout": head["dropout"],
        "per_drug_hidden": head["per_drug_hidden"],
        "mdcnn_trunk_per_modality": bool(mdcnn_trunk_per_modality),
        "n_params": n_params,
    }


def build_model_from_config(cfg):
    """Rebuild an untrained model matching `cfg['model']` (or a bare model dict).

    Mirrors the construction in training.multimodal._build_model /
    training.multidrug.run_multidrug_cv. Keep the two in step: this is the
    function that decides whether a saved checkpoint is loadable a year later."""
    m = cfg.get("model", cfg)
    arch = m["arch"]
    specs = [tuple(s) for s in m["branch_specs"]]
    keys = [parse_block_key(b["name"]) for b in m["blocks"]]
    modalities = [b["modality"] for b in m["blocks"]]
    drugs = m["drug_names"]
    joint = len(drugs) > 1
    head = {"hidden": m["hidden"], "dropout": m["dropout"],
            "per_drug_hidden": m["per_drug_hidden"]}
    out_bias = m["out_bias"]

    # .get, not [...]: configs written before these knobs existed carry no key
    # and must still rebuild at the defaults.
    transformer = m.get("transformer") or None
    enc_kinds = sorted(set(m["encoder_types"]))

    if arch == "mdcnn":
        return MDCNNNet(specs, n_drugs=len(drugs),
                        drug_names=drugs if joint else None, out_bias=out_bias,
                        block_modalities=(modalities if m["mdcnn_trunk_per_modality"]
                                          else None),
                        encoder=(enc_kinds[0] if len(enc_kinds) == 1 else "cnn"),
                        transformer=transformer, **head)
    if arch == "setfusion":
        # .get, not [...]: configs written before the capacity knobs existed have
        # no 'setfusion' key at all and must still rebuild at the defaults
        return SetFusionNet(keys, specs, n_drugs=len(drugs),
                            drug_names=drugs if joint else None,
                            out_bias=out_bias, hidden=m["hidden"],
                            head_dropout=m["dropout"],
                            per_drug_hidden=m["per_drug_hidden"],
                            **{**SETFUSION_DEFAULTS, **(m.get("setfusion") or {})})
    if arch == "cisfusion":
        return CisFusionNet(keys, specs, n_drugs=len(drugs),
                            drug_names=drugs if joint else None,
                            out_bias=out_bias, branch_models=m["branch_models"],
                            default_encoder=m["default_encoder"],
                            transformer=transformer, **head)
    if joint:
        return MultiDrugNet(specs, drugs, m["encoder_types"], out_bias=out_bias,
                            transformer=transformer, **head)
    return MultiModalNet(specs, m["encoder_types"], n_drugs=1, out_bias=out_bias,
                         transformer=transformer, **head)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

class RunCheckpointer:
    """Collects fold weights during a CV loop and writes them out at the end.

    ``mode`` is 'best' (only the fold whose model is scored on TEST — the one
    the reported numbers come from), 'all' (every fold, ~5x the bytes), or
    'none'. Fold states are held on CPU; the largest model here is ~60M params
    = 240 MB, so 'all' peaks around 1.2 GB of host RAM against the 48-64 GB the
    jobs request."""

    def __init__(self, run_name, stem, mode="best", weights_dir=None, enabled=True):
        self.mode = mode if mode in SAVE_CHOICES else "best"
        self.enabled = bool(enabled) and self.mode != "none" and run_name is not None
        self.dir = run_weights_dir(run_name, stem, weights_dir) if self.enabled else None
        self.states, self.folds = {}, []

    def add_fold(self, fold, model, best_epoch, metric):
        """Snapshot a fold's (already best-weight-restored) state_dict."""
        self.folds.append({"fold": int(fold), "best_epoch": best_epoch,
                           "val_metric": metric})
        if self.enabled:
            self.states[int(fold)] = {k: v.detach().cpu().clone()
                                      for k, v in model.state_dict().items()}

    def write(self, config, best_fold, isolate_ids=None):
        """Write config.json, isolates.txt and the selected fold weights.

        Returns the directory written to, or None when saving is off. Never
        raises into the training loop: a full or unwritable shared volume must
        not destroy a finished 13-hour run — it logs and returns None."""
        if not self.enabled:
            return None
        keep = ([best_fold] if self.mode == "best" else sorted(self.states))
        keep = [f for f in keep if f in self.states]
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            saved = []
            for f in keep:
                name = f"fold{f}.pt"
                torch.save(self.states[f], self.dir / name)
                saved.append({"fold": f, "weights": name})
            by_fold = {s["fold"]: s["weights"] for s in saved}
            for rec in self.folds:
                rec["weights"] = by_fold.get(rec["fold"])
            config["folds"] = self.folds
            config["best_fold"] = best_fold
            config["save_weights"] = self.mode
            config["schema_version"] = SCHEMA_VERSION
            config["env"] = _env_block()
            (self.dir / "config.json").write_text(json.dumps(config, indent=2))
            if isolate_ids is not None:
                (self.dir / "isolates.txt").write_text("\n".join(map(str, isolate_ids)))
            total = sum((self.dir / s["weights"]).stat().st_size for s in saved)
            print(f"  [checkpoint] wrote {len(saved)} fold(s) + config.json to "
                  f"{self.dir} ({total / 1e6:.0f} MB)", flush=True)
            return self.dir
        except Exception as e:                      # noqa: BLE001 — see docstring
            print(f"  [checkpoint] WARNING: could not save weights to {self.dir}: "
                  f"{type(e).__name__}: {e}", flush=True)
            return None


def write_pointer(run_dir, stem, weights_path):
    """Leave a breadcrumb in the results folder so a run finds its weights."""
    if weights_path is None or run_dir is None:
        return
    p = Path(run_dir) / "weights_location.json"
    try:
        existing = json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        existing = {}
    existing[stem] = str(weights_path)
    p.write_text(json.dumps(existing, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def load_model(run_name, stem, fold=None, weights_dir=None, map_location="cpu",
               strict=True):
    """Rebuild a saved model and load its weights. Returns ``(model, config)``.

    ``fold`` defaults to the config's ``best_fold`` (the model whose TEST
    metrics were reported). The returned model is in eval() mode."""
    d = run_weights_dir(run_name, stem, weights_dir)
    cfg = json.loads((d / "config.json").read_text())
    if fold is None:
        fold = cfg.get("best_fold")
    available = {f["fold"]: f.get("weights") for f in cfg.get("folds", [])
                 if f.get("weights")}
    if fold not in available:
        raise FileNotFoundError(
            f"fold {fold} has no weights in {d}; saved folds: {sorted(available)} "
            f"(this run used --save-weights {cfg.get('save_weights')})")
    model = build_model_from_config(cfg)
    model.load_state_dict(torch.load(d / available[fold], map_location=map_location),
                          strict=strict)
    model.eval()
    return model, cfg
