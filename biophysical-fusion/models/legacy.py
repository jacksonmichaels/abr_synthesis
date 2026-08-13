"""
Earlier model variants, kept out of the live path.

None of these are wired into the current engines (``training.multimodal`` /
``training.multidrug`` build ``MultiModalNet``/``MultiDrugNet`` from per-block
specs instead), but they document the fusion designs the project started from
and stay importable — ``models.CrossAttentionFusionCNN`` in particular is the
"Asymmetrical Cross-attention" model from experimental_plan.pdf, still an open
experiment.

They share ``ConvBranch`` / ``DenseHead`` with the live models, so a revived
variant trains with the same building blocks. Their signature is the OLD
two-argument ``forward(dna_x, bio_xs)`` (a DNA tensor plus per-gene biophysical
tensors), not the block-list interface ``MultiModalNet`` takes; the ``bio_input``
class attribute records which biophysical layout each one expects.
"""
import torch
import torch.nn as nn

from .net import ConvBranch, DenseHead


class DNAOnlyCNN(nn.Module):
    """DNA-only 1D CNN baseline — no biophysical branch. This is BIG-TB's
    SD-CNN in spirit (one-hot DNA -> conv stack -> dense head) and the original
    control the fusion models were measured against; the current equivalent is
    ``scripts/run_experiment.py --modalities dna``, which runs the corrected protocol
    (missing-phenotype filter, stratified split, train-only alpha).

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
    biophysical arrays are concatenated and upsampled to nucleotide resolution
    (``datasets.biochem.upsample_to_nt``) and stacked onto the one-hot channels
    before a single shared conv stack."""

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
