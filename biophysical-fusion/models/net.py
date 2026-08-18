"""
The live model architecture.

  - ConvBranch / DenseHead  : shared building blocks
  - ENCODERS registry       : per-branch encoder choice (cnn / transformer)
  - MultiModalNet           : generic late fusion, one branch per feature block
  - MultiDrugNet            : same, with one sigmoid logit per drug (MD-CNN style)
  - SetFusionNet            : weight-shared per-modality encoders + locus-keyed
                              set fusion (block COUNT and ORDER are free)
  - CisFusionNet            : each promoter concatenated onto ITS OWN CDS in
                              transcription order, so one kernel spans both

`ConvBranch`'s stem+conv+pool stack mirrors BIG-TB's existing
ProteinCNN1x1 (protein-tasks/one_hot_encoded/cnn_model.py) almost exactly,
just split into a feature extractor + separate dense head so the same
branch can be reused across model variants instead of duplicated.

The earlier fusion variants (LateFusionCNN / EarlyFusionCNN /
CrossAttentionFusionCNN / DNAOnlyCNN) live in ``models/legacy.py`` and are
re-exported from ``models`` — nothing in the training path builds them.
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBranch(nn.Module):
    """1D conv feature extractor, shape mirrors BIG-TB's ProteinCNN1x1."""

    def __init__(self, in_channels, stem_out=64):
        super().__init__()
        self.stem = nn.Conv1d(in_channels, stem_out, 1)
        self.conv1 = nn.Conv1d(stem_out, 64, 12, padding=6)
        self.pool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool1d(3)
        self.out_channels = 32

    def forward(self, x):
        x = F.relu(self.stem(x))
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        return x  # (B, out_channels, L')

    def out_len(self, in_len):
        with torch.no_grad():
            return self.forward(torch.zeros(1, self.stem.in_channels, in_len)).shape[-1]


class DenseHead(nn.Module):
    """Shared dense head: fc1 -> fc2 -> one logit per output.

    ``dropout`` and ``per_drug_hidden`` are both OFF by default, so the
    parameter count and forward pass are byte-identical to the version that
    produced results/experiments/full_run. Both exist for the joint models,
    where the default shape is the binding constraint:

    * **dropout** — the stack has no regularization at all (no dropout, no
      norm, no weight decay), and a joint `late_fusion` net is ~46M parameters
      fit on 17.9k isolates. Applied after each hidden ReLU.
    * **per_drug_hidden** — with ``out_dim=11`` every drug is read off ONE
      256-d vector by one shared linear layer, so eleven tasks share all the
      capacity there is; single-drug models get the same 256 units for one
      task. Setting it to k>0 gives each drug its own ``hidden -> k -> 1``
      branch off the shared trunk (11 x (256*k + k) params — ~180k at k=64,
      against 46M for the trunk). Only takes effect when out_dim > 1.
    """

    def __init__(self, in_features, out_dim=1, hidden=256, out_bias=None,
                 dropout=0.0, per_drug_hidden=0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.out_dim = int(out_dim)
        self.per_drug_hidden = int(per_drug_hidden) if out_dim > 1 else 0
        if self.per_drug_hidden:
            k = self.per_drug_hidden
            self.drug_hidden = nn.ModuleList(nn.Linear(hidden, k) for _ in range(out_dim))
            self.drug_out = nn.ModuleList(nn.Linear(k, 1) for _ in range(out_dim))
            self.fc_out = None
        else:
            self.drug_hidden = self.drug_out = None
            self.fc_out = nn.Linear(hidden, out_dim)
        # Optional output-bias initialization to the class log-odds (TODO #6).
        # sigmoid(bias) then starts at the positive-class base rate, matching the
        # "carefully initialize the output bias" trick for imbalanced data. A
        # constant bias shift cannot change AUC ranking — it only calibrates the
        # initial loss / operating point. out_bias=None keeps PyTorch's default
        # Linear bias (small uniform); baseline Keras uses a zero bias, but since
        # neither affects ranking this difference is immaterial to AUC.
        if out_bias is not None:
            with torch.no_grad():
                for layer in (self.drug_out or [self.fc_out]):
                    layer.bias.fill_(float(out_bias))

    def forward(self, x):
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        if self.fc_out is not None:
            return self.fc_out(x)  # (B, out_dim) logits
        # one small branch per drug off the shared trunk
        return torch.cat([out(F.relu(h(x)))
                          for h, out in zip(self.drug_hidden, self.drug_out)], dim=1)


# ---------------------------------------------------------------------------
# Per-branch encoders. Each maps one (B, C, L) block to a flat (B, F) feature
# vector and exposes `out_features`. The registry lets each modality pick its
# own architecture (see MultiModalNet / training.multimodal / scripts.run_experiment).
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """1D-CNN branch encoder: the ConvBranch conv stack, flattened. This is the
    original per-branch model (what every branch used before encoders existed).
    Strong local motif detector — a good default for DNA/regulatory/biophysical."""

    kind = "cnn"

    def __init__(self, in_channels, length):
        super().__init__()
        self.branch = ConvBranch(in_channels)
        self.out_features = self.branch.out_channels * self.branch.out_len(length)

    def forward(self, x):
        return torch.flatten(self.branch(x), 1)


class TransformerEncoder(nn.Module):
    """Patch-embedding Transformer branch encoder. A strided conv chunks the
    (B, C, L) input into ~L/patch tokens (ViT-style patch embedding, which keeps
    self-attention tractable on long genomic sequences), adds a learned
    positional embedding, runs a small Transformer encoder, and mean-pools the
    tokens. Models long-range interactions between distant positions (e.g.
    epistatic residue pairs) that a local CNN can miss — a candidate for the
    protein branch."""

    kind = "transformer"

    def __init__(self, in_channels, length, d_model=64, nhead=4, layers=2,
                 dim_ff=128, patch=9, dropout=0.1):
        super().__init__()
        patch = max(1, min(patch, length))
        self.tokenize = nn.Conv1d(in_channels, d_model, kernel_size=patch, stride=patch)
        n_tokens = max(1, (length - patch) // patch + 1)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout=dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.out_features = d_model

    def forward(self, x):
        t = self.tokenize(x).transpose(1, 2)       # (B, n_tokens, d_model)
        t = t + self.pos[:, :t.shape[1], :]
        t = self.encoder(t)
        return t.mean(dim=1)                        # (B, d_model)


ENCODERS = {"cnn": CNNEncoder, "transformer": TransformerEncoder}

# Capacity knobs for the transformer branch encoder AND the transformer MD-CNN
# trunk. These are TransformerEncoder's own signature defaults, so leaving them
# alone reproduces every run made before they were tunable.
#
# They exist because a transformer branch is not remotely parameter-comparable to
# a CNN branch at its defaults: CNNEncoder flattens, so its out_features scale
# with sequence length (32 * L/9 — 12,064 for a 3.4 kb DNA block, which puts
# ~3.1M params in the dense head alone), while TransformerEncoder mean-pools to
# exactly d_model=64 no matter how long the input is. All-CNN and all-transformer
# versions of the same cell therefore differ by ~30x in size unless d_model,
# layers and dim_ff are raised deliberately. See results/experiments/
# transformer_run/README.md for how the sweep picks them per architecture.
TRANSFORMER_DEFAULTS = {
    "d_model": 64, "nhead": 4, "layers": 2, "dim_ff": 128, "patch": 9,
    "dropout": 0.1,
}


def make_encoder(kind, in_channels, length, transformer=None):
    """Build one branch encoder. ``transformer`` overrides reach only the
    transformer encoder — CNNEncoder does not accept them, and passing them to
    it should be a loud TypeError in tests rather than a silent no-op here."""
    if kind not in ENCODERS:
        raise ValueError(f"unknown encoder {kind!r}; available: {list(ENCODERS)}")
    if kind == "transformer" and transformer:
        return ENCODERS[kind](in_channels, length, **transformer)
    return ENCODERS[kind](in_channels, length)


class MultiModalNet(nn.Module):
    """Generic late-fusion network with a PER-BRANCH choice of encoder.

    Takes one ``(in_channels, length)`` spec per feature block — DNA (5ch),
    per-gene protein (20ch), per-gene biophysical (3ch), per-region regulatory
    (5ch), in any combination — and one encoder key per block (`encoder_types`,
    from ENCODERS). Each block is encoded independently, all feature vectors are
    concatenated, and a shared dense head maps the fused vector to the drug
    logits. This is how 'DNA uses a CNN, protein uses a Transformer' is
    expressed: the trainer resolves a per-modality choice into this per-block
    list. All-CNN reproduces the previous behavior exactly."""

    bio_input = "blocks"  # forward takes the list of block tensors as-is

    def __init__(self, branch_specs, encoder_types=None, n_drugs=1, out_bias=None,
                 hidden=256, dropout=0.0, per_drug_hidden=0, transformer=None):
        super().__init__()
        if not branch_specs:
            raise ValueError("MultiModalNet needs at least one branch spec")
        if encoder_types is None:
            encoder_types = ["cnn"] * len(branch_specs)
        if len(encoder_types) != len(branch_specs):
            raise ValueError("encoder_types must match branch_specs length")
        unknown = [t for t in encoder_types if t not in ENCODERS]
        if unknown:
            raise ValueError(f"unknown encoder(s) {unknown}; available: {list(ENCODERS)}")
        self.encoders = nn.ModuleList(
            make_encoder(t, c, l, transformer)
            for t, (c, l) in zip(encoder_types, branch_specs))
        self.encoder_types = list(encoder_types)
        self.transformer = dict(transformer) if transformer else None
        total = sum(e.out_features for e in self.encoders)
        self.head = DenseHead(total, out_dim=n_drugs, out_bias=out_bias,
                              hidden=hidden, dropout=dropout,
                              per_drug_hidden=per_drug_hidden)

    def forward(self, xs):
        """xs: list of (B, C_i, L_i) tensors, same order as branch_specs."""
        feats = [enc(x) for enc, x in zip(self.encoders, xs)]
        return self.head(torch.cat(feats, dim=1) if len(feats) > 1 else feats[0])


class MultiModalCNN(MultiModalNet):
    """All-CNN MultiModalNet (one ConvBranch per block) — the previous default,
    kept for backward compatibility."""

    def __init__(self, branch_specs, n_drugs=1):
        super().__init__(branch_specs, ["cnn"] * len(branch_specs), n_drugs)


class MultiDrugNet(MultiModalNet):
    """Multi-drug ("MD-CNN"-style) late-fusion network.

    Same block-per-branch wiring as ``MultiModalNet`` — here typically one CNN
    branch per *locus* (the union of every drug's loci) — but the shared dense
    head emits **one sigmoid logit per drug** instead of one. Every drug is
    predicted simultaneously from the same fused genotype representation, which
    lets loci that inform several drugs (e.g. rrs → KAN/AMK/CAP, gyrA →
    LFX/MFX) share features across tasks. Trained with the masked multi-drug
    weighted BCE so each isolate contributes only its non-missing drug labels.

    ``drug_names`` sets the output width (n_drugs = len) and is stored so the
    output columns are self-describing (logits[:, i] == drug_names[i]). Mirrors
    the BIG-TB MD-CNN's ``Dense(n_drugs, sigmoid)`` head."""

    def __init__(self, branch_specs, drug_names, encoder_types=None, out_bias=None,
                 hidden=256, dropout=0.0, per_drug_hidden=0, transformer=None):
        if not drug_names:
            raise ValueError("MultiDrugNet needs at least one drug name")
        super().__init__(branch_specs, encoder_types,
                         n_drugs=len(drug_names), out_bias=out_bias,
                         hidden=hidden, dropout=dropout,
                         per_drug_hidden=per_drug_hidden,
                         transformer=transformer)
        self.drug_names = list(drug_names)


# ---------------------------------------------------------------------------
# BIG-TB's own topology (the "architecture debt" fix, TODO.md 2026-08-04).
#
# MultiModalNet above is the ProteinCNN1x1 branch applied per block: a 1x1 stem
# then convs, with each locus in its own branch, so no convolution ever mixes
# loci — only the flatten does. BIG-TB's SD-CNN / MD-CNN do the opposite: every
# locus is a CHANNEL on one shared, zero-padded position axis, and layer 1 is a
# (n_channels x 12) conv over all of them at once, so loci mix from the first
# layer. Reference (verbatim, both scripts' get_conv_nn, filter_size=12):
#
#   Conv2D(64, (5, 12), channels_last)  # input (5, L_longest, n_loci)
#   Conv1D(64, 12) -> MaxPool1D(3) -> Conv1D(32, 3) -> Conv1D(32, 3) -> MaxPool1D(3)
#   Flatten -> Dense(256) -> Dense(256) -> Dense(n_drugs, sigmoid)
#
# Keras channels_last means the trailing locus axis is the conv's input-channel
# axis and the (5, 12) kernel spans the whole one-hot height, so its output
# height is 1 (SD-CNN keeps that axis and uses (1, k) convs after; MD-CNN
# squeezes it — the two are the same computation). In torch NCHW that is
# Conv2d(n_loci, 64, (C, 12)) on an (N, n_loci, C, L) input, then 1D convs.
# All padding is 'valid', matching Keras' default.
# ---------------------------------------------------------------------------

class MDCNNTrunk(nn.Module):
    """BIG-TB's conv stack over locus-as-channel input.

    Takes (B, n_loci, C, L) — one zero-padded channel plane per locus — and
    returns the flattened conv features. ``out_features`` is the flatten width
    (14,336 on BIG-TB's 19-locus DNA input, vs 137,952 for the per-locus
    concatenation MultiModalNet was doing)."""

    def __init__(self, n_loci, channels, length, filter_size=12):
        super().__init__()
        self.conv_in = nn.Conv2d(n_loci, 64, (channels, filter_size))
        self.conv1 = nn.Conv1d(64, 64, 12)
        self.pool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(64, 32, 3)
        self.conv3 = nn.Conv1d(32, 32, 3)
        self.pool2 = nn.MaxPool1d(3)
        out_len = (((length - filter_size + 1) - 11) // 3 - 4) // 3
        if out_len < 1:
            raise ValueError(
                f"MD-CNN trunk needs a longer position axis: L={length} collapses to "
                f"{out_len} after the valid-padded conv/pool stack (need >= 1). "
                "Real loci are hundreds of bp; check the block lengths.")
        self.out_features = 32 * out_len

    def forward(self, x):
        """x: (B, n_loci, C, L) -> (B, out_features)."""
        x = F.relu(self.conv_in(x)).squeeze(2)   # (B, 64, L') — height collapses to 1
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        return torch.flatten(x, 1)


class MDCNNTransformerTrunk(nn.Module):
    """Transformer counterpart to MDCNNTrunk, over the same locus-as-channel input.

    Keeps the property that makes this the MD-CNN topology rather than late
    fusion — **layer 1 spans every locus and the full channel height at once**,
    so loci mix before anything else happens — and swaps only the sequence model
    that follows it.

    The trick is that MD-CNN's layer 1 is already a strided-in-nothing Conv2d
    over (n_loci, channels, L); making its stride equal its kernel width along
    the position axis turns the very same op into a ViT-style patch embedding.
    So one `Conv2d(n_loci, d_model, (channels, patch), stride=(1, patch))` both
    mixes the loci exactly as the reference does AND chunks the position axis
    into ~L/patch tokens. A learned positional embedding, a small Transformer
    encoder and a mean-pool then stand in for the `Conv1d/pool` stack and the
    flatten.

    Takes (B, n_loci, C, L) -> (B, d_model), the same contract as MDCNNTrunk, so
    MDCNNNet's grouping, padding and head are untouched.

    The cost is the same one SetFusionNet pays and worth stating: the flatten is
    gone, so absolute position is coarsened to a pooled summary over tokens.
    MDCNNTrunk's 14,336-wide flatten keeps "this motif fired at column 315";
    this keeps "it fired in this ~9 bp token, somewhere in the sequence".
    """

    def __init__(self, n_loci, channels, length, d_model=64, nhead=4, layers=2,
                 dim_ff=128, patch=9, dropout=0.1):
        super().__init__()
        patch = max(1, min(int(patch), length))
        self.tokenize = nn.Conv2d(n_loci, d_model, (channels, patch),
                                  stride=(1, patch))
        n_tokens = max(1, (length - patch) // patch + 1)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout=dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.out_features = d_model

    def forward(self, x):
        """x: (B, n_loci, C, L) -> (B, d_model)."""
        t = self.tokenize(x)                    # (B, d_model, 1, n_tokens)
        t = t.squeeze(2).transpose(1, 2)        # (B, n_tokens, d_model)
        t = t + self.pos[:, :t.shape[1], :]
        return self.encoder(t).mean(dim=1)


MDCNN_TRUNKS = {"cnn": MDCNNTrunk, "transformer": MDCNNTransformerTrunk}


class MDCNNNet(nn.Module):
    """BIG-TB SD-CNN / MD-CNN topology on our block interface.

    Same ``forward(xs)`` contract as MultiModalNet — a list of (B, C_i, L_i)
    block tensors, one per LOCUS (load with per-locus branches; concatenated
    per-modality blocks would defeat the point) — but instead of encoding each
    block separately it zero-pads them to a common length and stacks them into
    the channel axis of one ``MDCNNTrunk``, so layer 1 convolves across every
    locus at once. n_drugs=1 reproduces SD-CNN, n_drugs=len(drugs) MD-CNN.

    Blocks are grouped by channel count, one trunk per group, features
    concatenated before the shared head. With DNA only (every block 5-channel)
    that is exactly one trunk = the reference model; mixed modalities (protein
    20ch, biophysical 3ch, …) get one trunk each rather than an error, since a
    single conv kernel cannot span different channel heights.

    ``block_modalities`` opts into grouping by **(modality, channels)** instead
    of channels alone. Channel-only grouping is the reference rule and stays the
    default, but it silently merges two modalities that happen to share an
    alphabet: `dna` and `regulatory` are both 5-channel (A,C,T,G,gap), so a
    dna+regulatory run puts all 16 promoter windows in the DNA trunk, where the
    group's position axis is the longest CDS (rpoC, 4,066 bp). An 87–2,060 bp
    promoter window is then 95%+ zero padding, and it occupies a whole input
    channel of the layer-1 conv. That is why `dna_regulatory__mdcnn` scored
    BELOW `dna__mdcnn` in full_run. Pass the modality per block to give the
    promoters their own trunk, padded to their own 2,060 bp maximum."""

    bio_input = "blocks"

    def __init__(self, branch_specs, n_drugs=1, out_bias=None, filter_size=12,
                 drug_names=None, block_modalities=None, hidden=256,
                 dropout=0.0, per_drug_hidden=0, encoder="cnn", transformer=None):
        super().__init__()
        if not branch_specs:
            raise ValueError("MDCNNNet needs at least one branch spec")
        if drug_names:
            n_drugs = len(drug_names)
        if block_modalities is not None and len(block_modalities) != len(branch_specs):
            raise ValueError("block_modalities must match branch_specs length "
                             f"({len(block_modalities)} vs {len(branch_specs)})")
        # group block indices by channel count (reference rule), or by
        # (modality, channel count) when the modalities are supplied
        groups = {}
        for i, (c, l) in enumerate(branch_specs):
            key = (c,) if block_modalities is None else (c, block_modalities[i])
            groups.setdefault(key, []).append(i)
        self.group_keys = sorted(groups)
        self.group_idx = [groups[k] for k in self.group_keys]
        self.group_len = [max(branch_specs[i][1] for i in idxs)
                          for idxs in self.group_idx]
        if encoder not in MDCNN_TRUNKS:
            raise ValueError(f"unknown mdcnn encoder {encoder!r}; "
                             f"available: {list(MDCNN_TRUNKS)}")
        self.encoder = encoder
        self.transformer = dict(transformer) if transformer else None
        # One trunk per channel-height group either way; only the trunk's inside
        # changes. `filter_size` is the conv trunk's kernel width and has no
        # meaning for the transformer trunk, which takes `patch` instead.
        if encoder == "transformer":
            self.trunks = nn.ModuleList(
                MDCNNTransformerTrunk(len(idxs), branch_specs[idxs[0]][0], length,
                                      **(transformer or {}))
                for idxs, length in zip(self.group_idx, self.group_len))
        else:
            self.trunks = nn.ModuleList(
                MDCNNTrunk(len(idxs), branch_specs[idxs[0]][0], length, filter_size)
                for idxs, length in zip(self.group_idx, self.group_len))
        total = sum(t.out_features for t in self.trunks)
        self.head = DenseHead(total, out_dim=n_drugs, out_bias=out_bias,
                              hidden=hidden, dropout=dropout,
                              per_drug_hidden=per_drug_hidden)
        self.drug_names = list(drug_names) if drug_names else None

    @classmethod
    def from_blocks(cls, blocks, trunk_per_modality=False, **kwargs):
        """Build from loader FeatureBlocks, optionally one trunk per modality."""
        return cls([b.spec() for b in blocks],
                   block_modalities=([b.modality for b in blocks]
                                     if trunk_per_modality else None),
                   **kwargs)

    def forward(self, xs):
        feats = []
        for idxs, length, trunk in zip(self.group_idx, self.group_len, self.trunks):
            # zero-pad each locus to the group's longest, then stack as channels:
            # X[:, locus] = one_hot, zeros elsewhere — the reference's create_X
            planes = [F.pad(xs[i], (0, length - xs[i].shape[-1])) for i in idxs]
            feats.append(trunk(torch.stack(planes, dim=1)))
        return self.head(torch.cat(feats, dim=1) if len(feats) > 1 else feats[0])


# ---------------------------------------------------------------------------
# Locus-keyed set fusion.
#
# MultiModalNet/MDCNNNet both bind a block to the model by POSITION: branch i
# owns encoder i, and after the flatten every position is a fixed feature index.
# That silently assumes the block list is a fixed-length, fixed-order vector,
# which is where the regulatory modality breaks down — a drug has e.g. 2 coding
# loci but 12 WHO promoter windows (only 2 sharing a gene name, 10 with no CDS
# loaded at all), and both counts change per drug. Slot k of the DNA branch and
# slot k of the regulatory branch are then unrelated genes.
#
# SetFusionNet binds a block by KEY instead: (modality, locus) parsed off the
# FeatureBlock name. One encoder per MODALITY, shared by every locus of that
# modality; the token it emits carries a learned modality embedding + a learned
# locus embedding, so `regulatory:katG` and `dna:katG` arrive at the fusion
# transformer carrying the same locus vector and can be matched by attention
# rather than by position. Block count/order stop mattering, which is what makes
# a varying number of promoter windows expressible at all.
#
# The cost, stated plainly: the shared encoder must be length-agnostic, so it
# ends in a pooled summary instead of MultiModalNet's flatten. Absolute position
# within a locus is coarsened to `bins` relative segments — the pooling keeps
# "this motif fired in the first quarter of katG", not "at column 315". That is
# a real loss of resolution traded for weight sharing and count-independence.
# ---------------------------------------------------------------------------

NO_LOCUS = "<none>"     # locus key for a merged per-modality block ('dna')


def parse_block_key(name):
    """``FeatureBlock.name`` -> ``(modality, locus)``.

    'protein:katG' -> ('protein', 'katG'); a merged per-modality block, which is
    named for the modality alone ('dna'), -> ('dna', None)."""
    modality, _, locus = str(name).partition(":")
    return modality, (locus or None)


class SharedBlockEncoder(nn.Module):
    """One (B, C, L) block -> one (B, d_model) token, LENGTH-AGNOSTIC.

    Same stem+conv+pool shape as ``ConvBranch`` (so the feature extractor is the
    familiar BIG-TB ProteinCNN1x1 stack), but it ends in adaptive pooling rather
    than a flatten: ``out_features`` depends only on ``d_model``, never on L.
    That is the whole point — it lets a SINGLE instance encode every locus of a
    modality (inhA 269 aa and katG 432 aa hit the same weights), which is both
    the parameter saving and the reason a new locus needs no new branch.

    Pooling is mean AND max over ``bins`` equal segments: max answers "did this
    motif occur", mean "how much of the segment matches", and the segmentation
    keeps coarse relative position that a single global pool would flatten away.

    ``width``/``out_channels``/``depth`` are the block's own capacity, separate
    from ``d_model`` (the token width everything downstream sees). ``depth`` is
    how many ``(3-tap, 3-tap, pool-3)`` stages follow ``conv1``: depth=1 is the
    BIG-TB stack verbatim, and each extra stage adds two 3-tap convs and another
    3x downsample, so it buys receptive field rather than channels. The extra
    stages live in their own ``ModuleList``, which is empty at depth=1 and
    therefore contributes no ``state_dict`` keys — a depth-1 encoder is
    byte-identical to the pre-``depth`` one and old checkpoints still load.
    """

    def __init__(self, in_channels, d_model=128, width=64, out_channels=32, bins=4,
                 depth=1):
        super().__init__()
        if depth < 1:
            raise ValueError(f"SharedBlockEncoder depth must be >= 1, got {depth}")
        self.stem = nn.Conv1d(in_channels, width, 1)
        self.conv1 = nn.Conv1d(width, width, 12, padding=6)
        # ceil_mode so a short block (a small promoter window) cannot collapse
        # the position axis to zero the way a valid-padded stack would.
        self.pool1 = nn.MaxPool1d(3, ceil_mode=True)
        self.conv2 = nn.Conv1d(width, out_channels, 3, padding=1)
        self.conv3 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.pool2 = nn.MaxPool1d(3, ceil_mode=True)
        self.extra_stages = nn.ModuleList(
            nn.Sequential(nn.Conv1d(out_channels, out_channels, 3, padding=1),
                          nn.ReLU(),
                          nn.Conv1d(out_channels, out_channels, 3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool1d(3, ceil_mode=True))
            for _ in range(depth - 1))
        self.avg_pool = nn.AdaptiveAvgPool1d(bins)
        self.max_pool = nn.AdaptiveMaxPool1d(bins)
        self.proj = nn.Linear(2 * out_channels * bins, d_model)
        self.in_channels = in_channels
        self.depth = depth
        self.out_features = d_model

    def forward(self, x):
        x = F.relu(self.stem(x))
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        for stage in self.extra_stages:
            x = stage(x)
        pooled = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        return self.proj(torch.flatten(pooled, 1))       # (B, d_model)


class KeyedTokenNorm(nn.Module):
    """Standardise each token across the BATCH, with statistics kept per
    ``(modality, locus)`` key.

    Why this exists, measured on full_run_v2/all_modalities__setfusion
    (MOXIFLOXACIN, 2,868 isolates): an encoded token has mean norm 0.6654 and a
    per-isolate spread of 0.00090. **0.14% of a token varies with the genotype;
    the other 99.86% is a constant that only says which locus this is.** The
    fusion transformer and the drug query therefore see almost the same vector
    for every isolate, and the measured consequence is attention pinned at
    exactly uniform (0.1250 = 1/8 over eight blocks) and a locus embedding that
    changes held-out AUC by -0.009 [-0.040, +0.021] when deleted outright.

    Subtracting the per-key mean deletes that constant and dividing by the
    per-key standard deviation puts the genotype signal at unit scale. Two
    things follow, and they are the point:

    * the transformer starts receiving input that actually differs between
      isolates, so attention *can* stop being uniform;
    * locus identity is no longer carried redundantly by the constant, so
      ``locus_emb`` becomes the only thing that says which locus a token is —
      load-bearing instead of decorative.

    Statistics are keyed, NOT positional. A plain ``BatchNorm1d`` over the
    flattened token axis would work here but would tie the module to a fixed
    block count and order, which is precisely the property SetFusionNet exists
    to avoid; keying preserves ``forward(xs, keys=...)`` with a subset or a
    reordering. Unseen keys fall back to the running statistics they were
    initialised with (mean 0, var 1), i.e. a no-op, rather than erroring.

    The affine term is per key as well, so a locus can learn its own scale back
    if that turns out to matter.
    """

    def __init__(self, n_keys, d_model, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.eps, self.momentum, self.affine = eps, momentum, affine
        self.register_buffer("running_mean", torch.zeros(n_keys, d_model))
        self.register_buffer("running_var", torch.ones(n_keys, d_model))
        if affine:
            self.weight = nn.Parameter(torch.ones(n_keys, d_model))
            self.bias = nn.Parameter(torch.zeros(n_keys, d_model))

    def forward(self, tokens, key_ids):
        """tokens: (B, n_blocks, d) — key_ids: (n_blocks,) long, one per block."""
        out = []
        for i, k in enumerate(key_ids.tolist()):
            x = tokens[:, i]                                    # (B, d)
            if self.training and x.shape[0] > 1:
                mean = x.mean(0)
                var = x.var(0, unbiased=False)
                with torch.no_grad():
                    self.running_mean[k].mul_(1 - self.momentum).add_(
                        self.momentum * mean.detach())
                    self.running_var[k].mul_(1 - self.momentum).add_(
                        self.momentum * var.detach())
            else:
                # eval, or a batch of 1 where a batch variance is undefined
                mean, var = self.running_mean[k], self.running_var[k]
            xn = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                xn = xn * self.weight[k] + self.bias[k]
            out.append(xn)
        return torch.stack(out, dim=1)


TOKEN_NORMS = ("none", "keyed")

# The capacity knobs the runners can override, with the values that produced
# results/experiments/full_run and full_run_v2. Every one is a SetFusionNet
# constructor kwarg, so the runners pass this dict straight through with **;
# leaving it untouched reproduces those runs exactly. Grouped by what they
# scale, which is also how results/experiments/setfusion_scaling sweeps them:
#   A  d_model            the token width — encoder output, both embeddings,
#                         the transformer, the drug queries, the head input
#   B  layers/dim_ff      the fusion transformer, i.e. everything AFTER the
#      (+ hidden)         per-block encoders (`hidden` is the head's own knob
#                         and stays where the other archs keep it)
#   C  enc_*/bins         inside one SharedBlockEncoder: channels, conv stages,
#                         and how many pooled segments each token is built from
#   D  token_norm         'keyed' standardises each token across the batch with
#                         per-(modality, locus) statistics. Default 'none' keeps
#                         every existing run bit-identical (no module is created,
#                         so no state_dict keys are added and old checkpoints
#                         still load).
SETFUSION_DEFAULTS = {
    "d_model": 128, "nhead": 4, "layers": 2, "dim_ff": 256, "dropout": 0.1,
    "enc_width": 64, "enc_out_channels": 32, "enc_depth": 1, "bins": 4,
    "token_norm": "none",
}


class SetFusionNet(nn.Module):
    """Weight-shared per-modality encoders + locus-keyed set fusion.

    Pipeline, per isolate::

        block (B, C_i, L_i)  --SharedBlockEncoder[modality_i]-->  (B, d_model)
                             + modality_emb[modality_i] + locus_emb[locus_i]
        all tokens (B, n_blocks, d_model)  --TransformerEncoder-->  contextual
        drug query q_j  --cross-attention over tokens-->  (B, n_drugs, d_model)
        shared MLP  ->  (B, n_drugs) logits

    Same ``forward(xs)`` contract as MultiModalNet — a list of (B, C_i, L_i)
    block tensors in ``branch_specs`` order — so the existing trainers can drive
    it unchanged. Three things differ:

    * **Encoders are shared per modality, not per block.** All 12 regulatory
      windows go through one promoter encoder. Adding a 13th costs no weights.
    * **Identity is carried, not implied.** Every token gets ``locus_emb[gene]``,
      so `dna:katG` and `regulatory:katG` are matchable by attention regardless
      of where they sit in the list. That is the pairing the positional layouts
      cannot express when the two modalities have different block counts.
    * **The head is one query per drug.** ``logits[:, j]`` comes from drug j
      attending over the locus tokens itself, instead of every drug reading the
      same flattened concatenation. ``forward(xs, return_attn=True)`` returns
      those weights — a (B, n_drugs, n_blocks) map of which locus each drug
      read, which is directly inspectable.

    Build it with PER-LOCUS blocks (``per_modality_branch=False`` on either
    loader). With the merged per-modality layout every block parses to locus
    None, they all share one embedding, and the locus keying degenerates to a
    no-op — the constructor warns when that happens.

    ``block_keys`` are ``(modality, locus)`` pairs; use ``from_blocks`` to take
    them straight off the loader's FeatureBlocks.
    """

    bio_input = "blocks"  # forward takes the list of block tensors as-is

    def __init__(self, block_keys, branch_specs, n_drugs=1, drug_names=None,
                 d_model=128, nhead=4, layers=2, dim_ff=256, dropout=0.1,
                 hidden=256, bins=4, out_bias=None, head_dropout=0.0,
                 enc_width=64, enc_out_channels=32, enc_depth=1,
                 per_drug_hidden=0, token_norm="none"):
        super().__init__()
        if not branch_specs:
            raise ValueError("SetFusionNet needs at least one branch spec")
        if len(block_keys) != len(branch_specs):
            raise ValueError("block_keys must match branch_specs length "
                             f"({len(block_keys)} vs {len(branch_specs)})")
        # caught here rather than inside nn.MultiheadAttention, where it surfaces
        # 100 lines deep as an assert with no mention of which knob is wrong
        if d_model % nhead:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead "
                             f"({nhead})")
        if drug_names:
            n_drugs = len(drug_names)
        block_keys = [(m, l or NO_LOCUS) for m, l in block_keys]

        # --- one encoder per modality, shared across that modality's loci ------
        chans = {}
        for (modality, _), (c, _l) in zip(block_keys, branch_specs):
            if chans.setdefault(modality, c) != c:
                raise ValueError(
                    f"modality {modality!r} has blocks with different channel "
                    f"counts ({chans[modality]} and {c}); one shared encoder "
                    "cannot span both")
        self.modalities = list(dict.fromkeys(m for m, _ in block_keys))
        self.encoders = nn.ModuleDict({
            m: SharedBlockEncoder(chans[m], d_model=d_model, bins=bins,
                                  width=enc_width, out_channels=enc_out_channels,
                                  depth=enc_depth)
            for m in self.modalities})

        # --- identity embeddings: what makes the pairing learnable -------------
        self.locus_vocab = [NO_LOCUS] + [l for l in dict.fromkeys(l for _, l in block_keys)
                                         if l != NO_LOCUS]
        self._locus_ix = {l: i for i, l in enumerate(self.locus_vocab)}
        self._mod_ix = {m: i for i, m in enumerate(self.modalities)}
        self.modality_emb = nn.Embedding(len(self.modalities), d_model)
        self.locus_emb = nn.Embedding(len(self.locus_vocab), d_model)
        nn.init.trunc_normal_(self.modality_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.locus_emb.weight, std=0.02)
        if len(self.locus_vocab) == 1:
            warnings.warn(
                "SetFusionNet: every block keyed to <none> — you passed the "
                "merged per-modality blocks, so the locus embedding is a "
                "no-op. Load with per_modality_branch=False for per-locus "
                "blocks if you want the locus pairing.", stacklevel=2)

        self.block_keys = block_keys
        self.register_buffer("_default_ids", self._key_ids(block_keys),
                             persistent=False)

        # --- optional per-key standardisation of the encoder output -----------
        # Applied BEFORE the embeddings are added: the point is to strip the
        # locus-constant part of the encoder output so the embedding is the only
        # carrier of identity, rather than a small correction on top of a
        # constant that already encodes it.
        if token_norm not in TOKEN_NORMS:
            raise ValueError(f"unknown token_norm {token_norm!r}; "
                             f"choose from {list(TOKEN_NORMS)}")
        self.token_norm = token_norm
        self.n_locus = len(self.locus_vocab)
        self.tok_norm = (KeyedTokenNorm(len(self.modalities) * self.n_locus, d_model)
                         if token_norm == "keyed" else None)

        # --- fusion over the SET of tokens (permutation-invariant, any count) --
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, layers)

        # --- one learned query per drug, then a shared per-drug MLP ------------
        self.drug_queries = nn.Parameter(torch.zeros(n_drugs, d_model))
        nn.init.trunc_normal_(self.drug_queries, std=0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        # separate from `dropout` above, which is the transformer/attention
        # dropout. This one regularizes the read-out MLP, so it is the knob the
        # runners' --dropout maps to (matching DenseHead's meaning).
        self.head_drop = nn.Dropout(float(head_dropout)) if head_dropout else nn.Identity()
        # per-drug read-out capacity. The queries ARE per-drug (one vector each),
        # but everything after them is shared: with per_drug_hidden=0 all n_drugs
        # pooled vectors go through the same fc1 -> fc_out, so the drug-specific
        # parameters are n_drugs*d_model out of the whole model — 1,408 of 460,417
        # on the joint DNA cell, 0.3%. That is the same shape of constraint
        # DenseHead.per_drug_hidden exists for, and it bites harder here: the
        # fused tokens are near-collinear at init (see TODO.md), so every query
        # pools nearly the same vector and a shared MLP then maps them to nearly
        # the same logit. k>0 gives each drug its own `hidden -> k -> 1` branch
        # off the shared trunk, for n_drugs*(hidden*k + k + k + 1) parameters
        # (~180k at k=64, 11 drugs, hidden 256). Only takes effect when n_drugs>1
        # — single-drug runs have one output and nothing to separate.
        self.per_drug_hidden = int(per_drug_hidden) if n_drugs > 1 else 0
        if self.per_drug_hidden:
            k = self.per_drug_hidden
            self.drug_hidden = nn.ModuleList(nn.Linear(hidden, k) for _ in range(n_drugs))
            self.drug_out = nn.ModuleList(nn.Linear(k, 1) for _ in range(n_drugs))
            self.fc_out = None
        else:
            self.drug_hidden = self.drug_out = None
            self.fc_out = nn.Linear(hidden, 1)
        # same convention as DenseHead: init the output bias to the class
        # log-odds so sigmoid(bias) starts at the base rate. Shared across drugs
        # when the head is, per-drug when it is not.
        if out_bias is not None:
            with torch.no_grad():
                for layer in (self.drug_out or [self.fc_out]):
                    layer.bias.fill_(float(out_bias))
        self.drug_names = list(drug_names) if drug_names else None
        self.n_drugs = n_drugs

    @classmethod
    def from_blocks(cls, blocks, **kwargs):
        """Build straight from loader FeatureBlocks (``data.blocks``)."""
        return cls([parse_block_key(b.name) for b in blocks],
                   [b.spec() for b in blocks], **kwargs)

    def _key_ids(self, keys):
        """(modality, locus) pairs -> a (n, 2) long tensor of embedding rows."""
        ids = []
        for modality, locus in keys:
            locus = locus or NO_LOCUS
            if modality not in self._mod_ix:
                raise ValueError(f"unknown modality {modality!r}; this model was "
                                 f"built for {self.modalities}")
            if locus not in self._locus_ix:
                raise ValueError(f"unknown locus {locus!r}; this model was built "
                                 f"for {self.locus_vocab[1:]}")
            ids.append((self._mod_ix[modality], self._locus_ix[locus]))
        return torch.tensor(ids, dtype=torch.long)

    def forward(self, xs, keys=None, return_attn=False):
        """xs: list of (B, C_i, L_i) block tensors.

        ``keys`` overrides the construction-time (modality, locus) list — the
        set formulation means you may pass a SUBSET, a reordering, or a repeat
        of the blocks seen at init, as long as every key is in the vocabulary.
        Leave it None for the plain trainer contract."""
        if keys is not None and len(keys) != len(xs):
            raise ValueError("keys must match the number of blocks passed")
        ids = (self._default_ids if keys is None
               else self._key_ids(keys).to(self.drug_queries.device))
        if len(xs) != len(ids):
            raise ValueError(f"expected {len(ids)} blocks, got {len(xs)}; pass "
                             "keys= to run on a different block set")

        mod_ids, locus_ids = ids[:, 0], ids[:, 1]
        names = [self.modalities[i] for i in mod_ids.tolist()]
        tokens = torch.stack([self.encoders[m](x) for m, x in zip(names, xs)], dim=1)
        if self.tok_norm is not None:
            tokens = self.tok_norm(tokens, mod_ids * self.n_locus + locus_ids)
        tokens = tokens + self.modality_emb(mod_ids) + self.locus_emb(locus_ids)

        z = self.fusion(tokens)                                  # (B, n, d)
        q = self.drug_queries.unsqueeze(0).expand(z.shape[0], -1, -1)
        pooled, attn = self.pool_attn(q, z, z, need_weights=return_attn,
                                      average_attn_weights=True)
        h = self.head_drop(F.relu(self.fc1(self.norm(pooled))))  # (B, n_drugs, hidden)
        if self.fc_out is not None:
            logits = self.fc_out(h).squeeze(-1)                  # (B, n_drugs)
        else:
            # one small branch per drug off the shared trunk; h[:, j] is drug j's
            # own pooled read-out, so branch j only ever sees drug j
            logits = torch.cat(
                [out(F.relu(hid(h[:, j]))) for j, (hid, out)
                 in enumerate(zip(self.drug_hidden, self.drug_out))], dim=1)
        return (logits, attn) if return_attn else logits


# ---------------------------------------------------------------------------
# Cis fusion: put the promoter back next to the gene it regulates.
#
# SetFusionNet makes the promoter/CDS correspondence LEARNABLE (a shared locus
# embedding two tokens can be matched on). This model makes it STRUCTURAL, and
# by the cheapest possible route: a WHO Table-22 window is the DNA immediately
# upstream of its gene, so `regulatory:katG` and `dna:katG` are neighbouring
# stretches of one chromosome that the block layout happened to tear apart.
# Concatenating them back together on the length axis — promoter first, then the
# CDS, transcription order — means a single conv kernel near the junction sees
# both, with no new architecture at all. "cis" is the biology: a promoter acts
# only on the gene physically adjacent to it on the same DNA molecule.
#
# vs the alternatives:
#   MultiModalNet  the two live in different branches, meet only after flatten
#   SetFusionNet   one pooled token each, matched by a learned locus key
#   CisFusionNet   one branch, one receptive field, full positional resolution
#
# The trade is the mirror image of SetFusionNet's. Branches are positional
# again, so the branch count is fixed at construction and encoders are not
# shared — but nothing is pooled away, so "column 315 of katG" survives to the
# flatten exactly as it does in MultiModalNet. Use this when the locus set is
# fixed (one drug, one model); use SetFusionNet when it varies.
#
# Only the two NUCLEOTIDE modalities can merge: dna and regulatory share the
# (A,C,T,G,-) alphabet, so the concatenation is over one consistent channel
# space. Protein (20ch) and biophysical (3ch) cannot be spliced onto a
# nucleotide axis and stay their own branches, unchanged.
# ---------------------------------------------------------------------------

CIS_MODALITIES = ("regulatory", "dna")   # concatenation order = transcription order
NT_CHANNELS = 5                          # A, C, T, G, gap — datasets.sequences.NT_CHANNELS
CIS_CHANNELS = NT_CHANNELS + 1           # ...plus the promoter/CDS segment marker


class CisFusionNet(nn.Module):
    """Late fusion over cis-units: promoter ⊕ CDS per locus, then per-branch encoders.

    Rebuilds the block list into one branch per **locus** before encoding::

        regulatory:katG (5, 844)  ┐
                                  ├─> cis:katG (6, 844+L_gap+2223) -> encoder
        dna:katG        (5, 2223) ┘
        protein:katG    (20, 432) ────────────────────────────────> encoder
        biophysical:katG (3, 432) ────────────────────────────────> encoder

    A 6th channel is appended to every nucleotide branch marking **which segment
    each column belongs to** — 0 promoter, 1 CDS. Without it the junction is
    invisible and the model cannot tell an upstream −15C>T from a synonymous
    change at the same offset into the CDS. The three column patterns are
    mutually distinguishable: promoter (one-hot set, flag 0), CDS (one-hot set,
    flag 1), spacer/padding (all zero).

    Three unit types come out of the regrouping, and all three are expected:

    * **paired** — promoter and CDS both present (`inhA`, `katG` for INH),
    * **promoter-only** — a WHO window whose gene is not in ``DRUG_TO_LOCI``
      (10 of INH's 12: `ahpC`, `mshA`, `ndh`, efflux loci …). Flag all 0.
    * **CDS-only** — a coding locus with no Table-22 window. Flag all 1.

    ``spacer`` inserts N all-zero columns between the two segments. The WHO
    window (upstream coords + 30 bp flank) is not guaranteed to end exactly at
    the start codon, so this asserts *order and adjacency*, not exact genomic
    contiguity; the zero columns are a "gap here" token rather than a claim
    about its true width. Default 0 leaves them flush.

    Same ``forward(xs)`` contract as MultiModalNet — the block list in
    ``branch_specs`` order — so the trainers can drive it unchanged. Build with
    PER-LOCUS blocks (``per_modality_branch=False``); a merged ``dna`` block has
    no locus to pair on and the constructor warns.
    """

    bio_input = "blocks"  # forward takes the list of block tensors as-is

    def __init__(self, block_keys, branch_specs, n_drugs=1, drug_names=None,
                 branch_models=None, default_encoder="cnn", spacer=0,
                 out_bias=None, hidden=256, dropout=0.0, per_drug_hidden=0,
                 transformer=None):
        super().__init__()
        if not branch_specs:
            raise ValueError("CisFusionNet needs at least one branch spec")
        if len(block_keys) != len(branch_specs):
            raise ValueError("block_keys must match branch_specs length "
                             f"({len(block_keys)} vs {len(branch_specs)})")
        if drug_names:
            n_drugs = len(drug_names)
        branch_models = branch_models or {}
        self.spacer = int(spacer)
        block_keys = [(m, l or NO_LOCUS) for m, l in block_keys]

        for (modality, _), (c, _l) in zip(block_keys, branch_specs):
            if modality in CIS_MODALITIES and c != NT_CHANNELS:
                raise ValueError(
                    f"{modality!r} block has {c} channels; cis concatenation "
                    f"needs the {NT_CHANNELS}-channel nucleotide alphabet "
                    "(A,C,T,G,gap) shared by dna and regulatory")

        # --- regroup blocks into cis-units ------------------------------------
        # a locus is a unit iff it has a nucleotide block; everything else
        # (protein, biophysical) passes through as its own branch, as before.
        nt_at = {}      # locus -> {modality: block index}
        for i, (modality, locus) in enumerate(block_keys):
            if modality in CIS_MODALITIES and locus != NO_LOCUS:
                nt_at.setdefault(locus, {})[modality] = i

        # every unit is (regulatory_idx, dna_idx, passthrough_idx); the two that
        # do not apply are None. A paired unit has both nucleotide indices set.
        self.units, self.unit_names, self.unit_kinds = [], [], []
        specs = []
        emitted = set()
        for i, (modality, locus) in enumerate(block_keys):
            length = branch_specs[i][1]
            if modality not in CIS_MODALITIES:                    # protein / biophysical
                unit, kind = (None, None, i), modality
                name = modality if locus == NO_LOCUS else f"{modality}:{locus}"
                spec = branch_specs[i]
            elif locus == NO_LOCUS:
                # a merged nucleotide block (all loci concatenated) has no gene
                # to pair on. It still gets the segment channel so every
                # nucleotide branch has one consistent layout.
                unit = ((i, None, None) if modality == "regulatory" else (None, i, None))
                kind, name, spec = modality, modality, (CIS_CHANNELS, length)
            else:
                if locus in emitted:
                    continue                       # already merged into its unit
                emitted.add(locus)
                at = nt_at[locus]
                reg_i, dna_i = at.get("regulatory"), at.get("dna")
                unit = (reg_i, dna_i, None)
                if reg_i is not None and dna_i is not None:       # paired
                    kind, name = "cis", f"cis:{locus}"
                    length = (branch_specs[reg_i][1] + self.spacer
                              + branch_specs[dna_i][1])
                else:                                             # orphan either way
                    kind = "regulatory" if reg_i is not None else "dna"
                    name = f"{kind}:{locus}"
                    length = branch_specs[reg_i if reg_i is not None else dna_i][1]
                spec = (CIS_CHANNELS, length)
            self.units.append(unit)
            self.unit_kinds.append(kind)
            self.unit_names.append(name)
            specs.append(spec)

        if "cis" not in self.unit_kinds:
            warnings.warn(
                "CisFusionNet: no locus had BOTH a regulatory and a dna block, "
                "so nothing was cis-paired and this is MultiModalNet with an "
                "extra channel. Load per-locus blocks (per_modality_branch="
                "False) covering both modalities.", stacklevel=2)

        self.unit_specs = specs
        encoder_types = [branch_models.get(k, default_encoder) for k in self.unit_kinds]
        unknown = [t for t in encoder_types if t not in ENCODERS]
        if unknown:
            raise ValueError(f"unknown encoder(s) {unknown}; available: {list(ENCODERS)}")
        self.encoders = nn.ModuleList(
            make_encoder(t, c, l, transformer)
            for t, (c, l) in zip(encoder_types, specs))
        self.encoder_types = encoder_types
        self.transformer = dict(transformer) if transformer else None
        self.head = DenseHead(sum(e.out_features for e in self.encoders),
                              out_dim=n_drugs, out_bias=out_bias, hidden=hidden,
                              dropout=dropout, per_drug_hidden=per_drug_hidden)
        self.drug_names = list(drug_names) if drug_names else None
        self.n_drugs = n_drugs

    @classmethod
    def from_blocks(cls, blocks, **kwargs):
        """Build straight from loader FeatureBlocks (``data.blocks``)."""
        return cls([parse_block_key(b.name) for b in blocks],
                   [b.spec() for b in blocks], **kwargs)

    def _segment(self, x, flag):
        """Append the promoter/CDS marker channel: (B, 5, L) -> (B, 6, L)."""
        return torch.cat([x, x.new_full((x.shape[0], 1, x.shape[-1]), float(flag))],
                         dim=1)

    def cis_inputs(self, xs):
        """The actual branch inputs, ``[(name, tensor), ...]``.

        Exposed rather than buried in ``forward`` so the concatenation is
        inspectable — plot one and you can see the promoter, the junction and
        the CDS in a single matrix."""
        out = []
        for name, (reg_i, dna_i, pass_i) in zip(self.unit_names, self.units):
            if pass_i is not None:                            # protein / biophysical
                out.append((name, xs[pass_i]))
                continue
            parts = []
            if reg_i is not None:
                parts.append(self._segment(xs[reg_i], 0))     # upstream first...
            if reg_i is not None and dna_i is not None and self.spacer:
                ref = parts[0]                                # ...gap marker...
                parts.append(ref.new_zeros((ref.shape[0], ref.shape[1], self.spacer)))
            if dna_i is not None:
                parts.append(self._segment(xs[dna_i], 1))     # ...then the gene
            out.append((name, torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]))
        return out

    def forward(self, xs):
        """xs: list of (B, C_i, L_i) block tensors, in ``block_keys`` order."""
        feats = [enc(x) for enc, (_n, x) in zip(self.encoders, self.cis_inputs(xs))]
        return self.head(torch.cat(feats, dim=1) if len(feats) > 1 else feats[0])
