"""Action head subpackage — import to trigger registry registrations."""

from yavla.models.heads.mlp import (
    MLPHeadConfig as MLPHeadConfig,
)
from yavla.models.heads.mlp import (
    MLPRegressionHead as MLPRegressionHead,
)
from yavla.models.heads.mlp import (
    ResidualMLP as ResidualMLP,
)
from yavla.models.heads.mlp import (
    head_registry as head_registry,
)
