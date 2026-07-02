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
    def __init__(self, in_features, out_dim=1, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)  # (B, out_dim) logits


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
