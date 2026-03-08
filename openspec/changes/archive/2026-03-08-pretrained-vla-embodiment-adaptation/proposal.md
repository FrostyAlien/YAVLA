## Why

YAVLA's current MVP assumes exact equality between dataset action/proprio dimensions and model dimensions. That is the correct constraint for the first complete training test, but it is too rigid for a pretrained multi-embodiment VLA where a single checkpoint must serve robots with smaller action/state spaces than the base model width.

## What Changes

- Add a pretrained-VLA embodiment adaptation path that allows model-side maximum action/proprio dimensions to be larger than a specific dataset embodiment.
- Introduce explicit padding and masking rules for smaller embodiments so extra model dimensions do not affect loss or inference outputs.
- Require embodiment-aware checkpoint metadata and load-time validation so checkpoints can distinguish base-model width from active robot embodiment width.
- Remove compatibility shims for legacy flat training YAML and legacy embodiment-less checkpoint/config formats.
- Add config and training-guide documentation for the difference between exact-dimension MVP training and pretrained multi-embodiment training.
- Add tests covering padding, masked losses, inference slicing, and checkpoint compatibility for smaller embodiments.

## Capabilities

### New Capabilities
- `embodiment-adaptation`: Model-side max-dimension support, padding/masking rules, and embodiment-aware inference/checkpoint behavior for pretrained VLAs.
- `pretrained-vla-documentation`: Documentation and examples for configuring, training, and loading pretrained multi-embodiment VLA checkpoints.

### Modified Capabilities
- `dataset-layer-documentation`: Documentation requirements expand to explain pretrained-VLA embodiment adaptation and when exact-dimension versus padded-dimension training modes should be used.
- `training-entry`: Training entrypoint requirements now require nested `TrainConfig` YAML and reject the legacy flat train-config format.

## Impact

- Affected code: `src/yavla/models/config.py`, `src/yavla/models/types.py`, `src/yavla/models/encoders/proprio.py`, `src/yavla/models/heads/mlp.py`, `src/yavla/models/policy.py`, `scripts/train.py`, and related tests.
- Affected docs: `docs/training-guide.md` and architecture/design guidance for embodiment handling.
- API impact: policy/training config grows explicit pretrained-VLA embodiment fields; flat train YAML and legacy embodiment-less checkpoints/configs are no longer supported.
