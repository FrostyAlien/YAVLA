"""YAVLA models — import triggers all registry registrations."""

# Registry registrations (side-effect imports)
import yavla.models.backbones  # noqa: F401  # triggers vlm_registry registration
from yavla.models.backbone import backbone_registry  # noqa: F401
from yavla.models.config import PolicyConfig as PolicyConfig
from yavla.models.decoder import decoder_registry  # noqa: F401
from yavla.models.encoders import proprio_registry as proprio_registry
from yavla.models.encoders import vision_registry as vision_registry
from yavla.models.heads import head_registry as head_registry
from yavla.models.merger import merger_registry  # noqa: F401
from yavla.models.policy import VLAPolicy as VLAPolicy
from yavla.models.policy import build_policy as build_policy
from yavla.models.protocols import (
    ActionDecoderProto as ActionDecoderProto,
)
from yavla.models.protocols import (
    ActionHeadProto as ActionHeadProto,
)
from yavla.models.protocols import (
    BackboneCapabilities as BackboneCapabilities,
)
from yavla.models.protocols import (
    BackboneProto as BackboneProto,
)
from yavla.models.protocols import (
    IntegrationMode as IntegrationMode,
)
from yavla.models.protocols import (
    PolicyBase as PolicyBase,
)
from yavla.models.protocols import (
    ProprioEncoderProto as ProprioEncoderProto,
)
from yavla.models.protocols import (
    TokenMergerBase as TokenMergerBase,
)
from yavla.models.protocols import (
    TokenMergerProto as TokenMergerProto,
)
from yavla.models.protocols import (
    VisionEncoderProto as VisionEncoderProto,
)
from yavla.models.protocols import (
    validate_integration as validate_integration,
)
from yavla.models.types import (
    ActionChunk as ActionChunk,
)
from yavla.models.types import (
    ActionPrediction as ActionPrediction,
)
from yavla.models.types import (
    ActionSpaceSpec as ActionSpaceSpec,
)
from yavla.models.types import (
    BackboneOutput as BackboneOutput,
)
from yavla.models.types import (
    FreezeConfig as FreezeConfig,
)
from yavla.models.types import (
    LossDict as LossDict,
)
from yavla.models.types import (
    ObservationBatch as ObservationBatch,
)
from yavla.models.types import (
    ProprioSpec as ProprioSpec,
)
from yavla.models.types import (
    TokenBatch as TokenBatch,
)
from yavla.models.types import (
    TrainingBatch as TrainingBatch,
)
