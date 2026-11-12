"""
The stacked CNN: BIG-TB's SD-CNN / MD-CNN topology on our block interface.

BIG-TB's reference (Big-TB-benchmark/dna-tasks/{SD,MD}-CNN, Keras):

  Conv2D(64, (5, 12), channels_last)  # input (5, L_longest, n_loci)
  Conv1D(64, 12) -> MaxPool1D(3) -> Conv1D(32, 3) -> Conv1D(32, 3) -> MaxPool1D(3)
  Flatten -> Dense(256) -> Dense(256) -> Dense(n_drugs, sigmoid)

Keras channels_last means the trailing locus axis is the conv's input-channel
axis and the (5, 12) kernel spans the whole one-hot height, so its output
height is 1 (SD-CNN keeps that axis and uses (1, k) convs after; MD-CNN
squeezes it — the two are the same computation). In torch NCHW that is
Conv2d(n_loci, 64, (C, 12)) on an (N, n_loci, C, L) input, then 1D convs.
All padding is 'valid', matching Keras' default.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_block_key(name):
    """``FeatureBlock.name`` -> ``(modality, locus)``.

    'protein:katG' -> ('protein', 'katG'); a name with no locus part -> (name, None)."""
    modality, _, locus = str(name).partition(":")
    return modality, (locus or None)


class DenseHead(nn.Module):
    """fc1 -> fc2 -> one logit per output (PyTorch default init throughout)."""

    def __init__(self, in_features, out_dim=1, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)  # (B, out_dim) logits


class MDCNNTrunk(nn.Module):
    """BIG-TB's conv stack over locus-as-channel input.

    Takes (B, n_loci, C, L) — one zero-padded channel plane per locus — and
    returns the flattened conv features. ``out_features`` is the flatten width
    (14,336 on BIG-TB's 19-locus DNA input)."""

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


class MDCNNNet(nn.Module):
    """The stacked CNN.

    ``forward(xs)`` takes a list of (B, C_i, L_i) block tensors, one per LOCUS.
    Instead of encoding each block separately it zero-pads them to a common
    length and stacks them into the channel axis of one ``MDCNNTrunk``, so layer
    1 convolves across every locus at once. n_drugs=1 reproduces SD-CNN,
    n_drugs=len(drugs) MD-CNN.

    Blocks are grouped by channel count, one trunk per group (sorted by channel
    count), features concatenated before the shared head. With DNA only (every
    block 5-channel) that is exactly one trunk = the reference model; protein
    (20ch) and biophysical (3ch) get a trunk each, since a single conv kernel
    cannot span different channel heights. dna and regulatory are both 5-channel,
    so the promoter windows share the DNA trunk."""

    def __init__(self, branch_specs, n_drugs=1, filter_size=12, drug_names=None,
                 hidden=256):
        super().__init__()
        if not branch_specs:
            raise ValueError("MDCNNNet needs at least one branch spec")
        if drug_names:
            n_drugs = len(drug_names)
        groups = {}
        for i, (c, l) in enumerate(branch_specs):
            groups.setdefault((c,), []).append(i)
        self.group_keys = sorted(groups)
        self.group_idx = [groups[k] for k in self.group_keys]
        self.group_len = [max(branch_specs[i][1] for i in idxs)
                          for idxs in self.group_idx]
        self.trunks = nn.ModuleList(
            MDCNNTrunk(len(idxs), branch_specs[idxs[0]][0], length, filter_size)
            for idxs, length in zip(self.group_idx, self.group_len))
        total = sum(t.out_features for t in self.trunks)
        self.head = DenseHead(total, out_dim=n_drugs, hidden=hidden)
        self.n_drugs = n_drugs
        self.drug_names = list(drug_names) if drug_names else None

    @classmethod
    def from_blocks(cls, blocks, **kwargs):
        """Build from loader FeatureBlocks."""
        return cls([b.spec() for b in blocks], **kwargs)

    def forward(self, xs):
        feats = []
        for idxs, length, trunk in zip(self.group_idx, self.group_len, self.trunks):
            # zero-pad each locus to the group's longest, then stack as channels:
            # X[:, locus] = one_hot, zeros elsewhere — the reference's create_X
            planes = [F.pad(xs[i], (0, length - xs[i].shape[-1])) for i in idxs]
            feats.append(trunk(torch.stack(planes, dim=1)))
        return self.head(torch.cat(feats, dim=1) if len(feats) > 1 else feats[0])
