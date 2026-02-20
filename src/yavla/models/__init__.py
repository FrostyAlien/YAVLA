"""YAVLA models — import triggers all registry registrations."""

# Registry registrations (side-effect imports)
from yavla.models.encoders import vision_registry, proprio_registry  # noqa: F401
from yavla.models.heads import head_registry  # noqa: F401
from yavla.models.merger import merger_registry  # noqa: F401
from yavla.models.backbone import backbone_registry  # noqa: F401
from yavla.models.decoder import decoder_registry  # noqa: F401

from yavla.models.config import PolicyConfig
from yavla.models.policy import VLAPolicy, build_policy
from yavla.models.types import (
    ActionChunk,
    ActionPrediction,
    ActionSpaceSpec,
    BackboneOutput,
    FreezeConfig,
    LossDict,
    ObservationBatch,
    ProprioSpec,
    TokenBatch,
    TrainingBatch,
)
from yavla.models.protocols import (
    ActionDecoderProto,
    ActionHeadProto,
    BackboneCapabilities,
    BackboneProto,
    IntegrationMode,
    ProprioEncoderProto,
    TokenMergerProto,
    VisionEncoderProto,
    validate_integration,
)
