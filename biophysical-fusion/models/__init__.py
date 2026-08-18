"""
Model package. ``net`` holds the live architecture — the shared building blocks
(ConvBranch / DenseHead), the per-branch encoder registry (ENCODERS), and the
late-fusion nets the training engines build (MultiModalNet / MultiDrugNet).
``legacy`` holds the earlier fusion variants, nothing in the live path.

Everything is re-exported here, so ``from models import MultiModalNet`` works
regardless of which file a class lives in.
"""
from .legacy import (  # noqa: F401
    CrossAttentionFusionCNN,
    DNAOnlyCNN,
    EarlyFusionCNN,
    LateFusionCNN,
)
from .net import (  # noqa: F401
    ENCODERS,
    MDCNN_TRUNKS,
    SETFUSION_DEFAULTS,
    TOKEN_NORMS,
    TRANSFORMER_DEFAULTS,
    CisFusionNet,
    CNNEncoder,
    ConvBranch,
    DenseHead,
    MDCNNNet,
    MDCNNTransformerTrunk,
    MDCNNTrunk,
    MultiDrugNet,
    make_encoder,
    MultiModalCNN,
    MultiModalNet,
    KeyedTokenNorm,
    SetFusionNet,
    SharedBlockEncoder,
    TransformerEncoder,
    parse_block_key,
)

# Architecture registry: what --arch selects on the runners.
#   late_fusion : our per-block encoder net (MultiModalNet / MultiDrugNet)
#   mdcnn       : BIG-TB's own locus-as-channel topology (MDCNNNet)
#   setfusion   : weight-shared per-modality encoders + locus-keyed set fusion
#   cisfusion   : promoter (+) CDS concatenated per locus, then per-branch encoders
# training/{multimodal,multidrug}.py branch on all four; the last three build
# from data.blocks (their block NAMES carry the (modality, locus) keys) and want
# per-locus blocks, which the runners imply.
ARCHITECTURES = ("late_fusion", "mdcnn", "setfusion", "cisfusion")
