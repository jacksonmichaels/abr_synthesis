"""
Phase 3 — fusion model architectures.

All three variants share the same two building blocks (`ConvBranch`,
`DenseHead`) instead of each being written from scratch:
  - LateFusionCNN          : Kulkarni's "independent convolutions" (default)
  - EarlyFusionCNN          : channel-stack ablation
  - CrossAttentionFusionCNN : experimental_plan.pdf's "Asymmetrical
    Cross-attention" (DNA branch as Query, biophysical branch as Key/Value)

`ConvBranch`'s stem+conv+pool stack mirrors BIG-TB's existing
ProteinCNN1x1 (protein-tasks/one_hot_encoded/cnn_model.py) almost exactly,
just split into a feature extractor + separate dense head so the same
branch can be reused across model variants instead of duplicated.
"""
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
    def __init__(self, in_features, out_dim=1, hidden=256, out_bias=None):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
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
                self.fc_out.bias.fill_(float(out_bias))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)  # (B, out_dim) logits


class DNAOnlyCNN(nn.Module):
    """DNA-only 1D CNN baseline — no biophysical branch. This is BIG-TB's
    SD-CNN in spirit (one-hot DNA -> conv stack -> dense head) and the
    control the biophysical-fusion models are measured against (TODO.md
    Phase 5). Shares ConvBranch/DenseHead with the fusion models so the only
    difference from LateFusionCNN is the absence of the bio branches.

    Accepts (and ignores) bio_lens / bio_xs so it is a drop-in for the same
    training harness as the fusion models."""

    bio_input = "none"

    def __init__(self, dna_len, bio_lens=None, n_drugs=1, dna_channels=5):
        super().__init__()
        self.dna_branch = ConvBranch(dna_channels)
        dna_flat = self.dna_branch.out_channels * self.dna_branch.out_len(dna_len)
        self.head = DenseHead(dna_flat, out_dim=n_drugs)

    def forward(self, dna_x, bio_xs=None):
        d = torch.flatten(self.dna_branch(dna_x), 1)
        return self.head(d)


# ---------------------------------------------------------------------------
# Per-branch encoders. Each maps one (B, C, L) block to a flat (B, F) feature
# vector and exposes `out_features`. The registry lets each modality pick its
# own architecture (see MultiModalNet / train_multimodal / run_experiment).
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

    def __init__(self, branch_specs, encoder_types=None, n_drugs=1, out_bias=None):
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
            ENCODERS[t](c, l) for t, (c, l) in zip(encoder_types, branch_specs))
        self.encoder_types = list(encoder_types)
        total = sum(e.out_features for e in self.encoders)
        self.head = DenseHead(total, out_dim=n_drugs, out_bias=out_bias)

    def forward(self, xs):
        """xs: list of (B, C_i, L_i) tensors, same order as branch_specs."""
        feats = [enc(x) for enc, x in zip(self.encoders, xs)]
        return self.head(torch.cat(feats, dim=1) if len(feats) > 1 else feats[0])


class MultiModalCNN(MultiModalNet):
    """All-CNN MultiModalNet (one ConvBranch per block) — the previous default,
    kept for backward compatibility."""

    def __init__(self, branch_specs, n_drugs=1):
        super().__init__(branch_specs, ["cnn"] * len(branch_specs), n_drugs)


class LateFusionCNN(nn.Module):
    """Default fusion (decision #2), matches Kulkarni et al. 2026's Model
    design and training Methods exactly: DNA branch and one biophysical
    branch PER GENE/PROTEIN (separate channels — see TODO.md #7) are each
    pooled and flattened independently, then concatenated before the dense
    head."""

    bio_input = "per_gene"

    def __init__(self, dna_len, bio_lens, n_drugs=1, dna_channels=5, bio_channels=3):
        super().__init__()
        self.dna_branch = ConvBranch(dna_channels)
        self.bio_branches = nn.ModuleList(ConvBranch(bio_channels) for _ in bio_lens)
        dna_flat = self.dna_branch.out_channels * self.dna_branch.out_len(dna_len)
        bio_flat = sum(b.out_channels * b.out_len(l) for b, l in zip(self.bio_branches, bio_lens))
        self.head = DenseHead(dna_flat + bio_flat, out_dim=n_drugs)

    def forward(self, dna_x, bio_xs):
        """bio_xs: list of (B, 3, K_g) tensors, one per gene, same order as bio_lens."""
        d = torch.flatten(self.dna_branch(dna_x), 1)
        bs = [torch.flatten(branch(x), 1) for branch, x in zip(self.bio_branches, bio_xs)]
        return self.head(torch.cat([d, *bs], dim=1))


class EarlyFusionCNN(nn.Module):
    """Ablation variant (decision #2, not from the paper): the per-gene
    biophysical arrays are concatenated and upsampled to nucleotide
    resolution (data.concat_upsampled) and stacked onto the one-hot
    channels before a single shared conv stack."""

    bio_input = "upsampled_concat"

    def __init__(self, dna_len, bio_lens=None, n_drugs=1, dna_channels=5, bio_channels=3):
        super().__init__()
        self.branch = ConvBranch(dna_channels + bio_channels)
        flat = self.branch.out_channels * self.branch.out_len(dna_len)
        self.head = DenseHead(flat, out_dim=n_drugs)

    def forward(self, dna_x, bio_x_upsampled):
        x = torch.cat([dna_x, bio_x_upsampled], dim=1)  # channel dim
        return self.head(torch.flatten(self.branch(x), 1))


class CrossAttentionFusionCNN(nn.Module):
    """experimental_plan.pdf's 'Asymmetrical Cross-attention': the DNA
    branch's feature map is the attention Query; the per-gene biophysical
    branches' feature maps are concatenated along the sequence axis into
    one Key/Value set. The idea (per the notes): let the 1D-CNN's own read
    of the DNA sequence pull in biophysical context only where attention
    weights say it's useful, instead of always concatenating it."""

    bio_input = "per_gene"

    def __init__(self, dna_len, bio_lens, n_drugs=1, dna_channels=5, bio_channels=3, n_heads=4):
        super().__init__()
        self.dna_branch = ConvBranch(dna_channels)
        self.bio_branches = nn.ModuleList(ConvBranch(bio_channels) for _ in bio_lens)
        d_model = self.dna_branch.out_channels
        self.bio_proj = nn.Linear(self.bio_branches[0].out_channels, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        dna_out_len = self.dna_branch.out_len(dna_len)
        self.head = DenseHead(dna_out_len * d_model, out_dim=n_drugs)

    def forward(self, dna_x, bio_xs):
        q = self.dna_branch(dna_x).transpose(1, 2)  # (B, L', d_model) Query
        kv = torch.cat(
            [self.bio_proj(branch(x).transpose(1, 2)) for branch, x in zip(self.bio_branches, bio_xs)],
            dim=1,
        )  # (B, sum(K_g'), d_model) Key/Value
        attended, _ = self.attn(q, kv, kv)
        return self.head(torch.flatten(attended, 1))
