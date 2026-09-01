"""
Model package. ``net`` holds the live architecture — the shared building blocks
(ConvBranch / DenseHead), the per-branch encoder registry (ENCODERS), and the
late-fusion nets the training engines build (MultiModalNet / MultiDrugNet).
``locusfusion`` holds the two-stage variant-token transformer,
``experimental_models`` the six variant-set AGGREGATORS (same tokenizer, six
different ways of combining sparse evidence), and ``legacy`` the earlier fusion
variants, nothing in the live path.

Everything is re-exported here, so ``from models import MultiModalNet`` works
regardless of which file a class lives in.
"""
from .legacy import (  # noqa: F401
    CrossAttentionFusionCNN,
    DNAOnlyCNN,
    EarlyFusionCNN,
    LateFusionCNN,
)
from .experimental_models import (  # noqa: F401
    EXPERIMENTAL_DEFAULTS,
    EXPERIMENTAL_MODELS,
    AdditiveVariantNet,
    CatalogueNet,
    DeepSetsVariantNet,
    FactorizedInteractionNet,
    GatedPoolNet,
    NoisyOrVariantNet,
    VariantEmbedding,
    VariantSet,
    make_experimental,
    variant_design_matrix,
)
from .locusfusion import (  # noqa: F401
    LOCUS_ENCODERS,
    LOCUSFUSION_DEFAULTS,
    SUMMARY_NORMS,
    LocusFusionNet,
)
from .net import (  # noqa: F401
    BRANCHED_DEFAULTS,
    BranchedHead,
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
    make_head,
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
#   locusfusion : one token per VARIANT, fused within a locus then across loci
#
# ...plus the experimental family in models/experimental_models.py, which shares
# locusfusion's variant tokenizer and differs ONLY in how the variant set is
# aggregated -- catalogue / additive / noisyor / gatedpool / deepsets / fm. They
# exist because the measured problem is sparse evidence aggregation rather than
# sequence encoding; see that module's docstring.
#
# training/{multimodal,multidrug}.py branch on all of these; everything except
# late_fusion builds from data.blocks (their block NAMES carry the (modality,
# locus) keys) and wants per-locus blocks, which the runners imply. The
# variant-token archs additionally need reference-difference input, so the
# runners imply --delta for them too.
ARCHITECTURES = (("late_fusion", "mdcnn", "setfusion", "cisfusion", "locusfusion")
                 + tuple(sorted(EXPERIMENTAL_MODELS)))

# Architectures that are built from data.blocks rather than branch_specs alone,
# and therefore need one block per LOCUS (per_modality_branch=False).
PER_LOCUS_ARCHS = (("mdcnn", "setfusion", "cisfusion", "locusfusion")
                   + tuple(sorted(EXPERIMENTAL_MODELS)))
# ...and those whose input representation is reference-difference encoded.
DELTA_ARCHS = ("locusfusion",) + tuple(sorted(EXPERIMENTAL_MODELS))
