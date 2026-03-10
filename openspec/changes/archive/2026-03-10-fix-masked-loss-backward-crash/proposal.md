## Why

Training can currently crash when action masking removes every supervised element in a batch, because the MLP action head returns a detached zero scalar that cannot be used in `backward()`. This needs to be fixed now because the repository already supports padded action chunks and embodiment-dimension masking, and those normal training paths can legitimately produce fully masked losses.

## What Changes

- Strengthen masked-loss behavior for the MLP regression head so fully masked batches produce a zero-valued loss that remains safe to backpropagate through distributed and Accelerate-backed training loops.
- Clarify the contract for combined timestep and inactive-dimension masking in pretrained-VLA embodiment adaptation so zero-valid batches do not terminate training.
- Add regression coverage for fully masked loss paths to ensure they remain finite and backward-safe, not just numerically non-NaN.

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `mvp-action-head`: Tighten the masked L1 loss requirement so fully padded chunks return a differentiable zero loss that does not break backward passes.
- `embodiment-adaptation`: Tighten the combined timestep-and-dimension masking requirement so fully masked pretrained-VLA batches return a backward-safe zero loss during training.

## Impact

- Affected code: `src/yavla/models/heads/mlp.py` and related masked-loss tests.
- Affected behavior: exact-dimension and pretrained-VLA training runs that encounter fully masked action supervision.
- Affected systems: training stability under Hugging Face Accelerate and any distributed runtime consuming `LossDict.total`.
