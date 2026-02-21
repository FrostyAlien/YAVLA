## Why

YAVLA’s current MVP MLP policy stack has multiple mismatched IO contracts (action normalization range, vision
pixel preprocessing, checkpoint serialization) that already cause failing tests and are likely to cause silent
correctness/performance failures in real training/inference. This change aligns and specifies these contracts
end-to-end (dataset → model → decoder → checkpoint) so policies are correct, reproducible, and modular as
intended.

## What Changes

- Define and enforce a single **default** action contract for the MVP MLP policy:
  - The policy predicts actions in a normalized **[-1, 1]** space.
  - The action decoder linearly unnormalizes via `ActionSpaceSpec.limits` (LeRobot/SB3-style scaling).
- Add configurable action normalization modes (persisted in config + checkpoint metadata) so different policy
  families can declare what they expect (e.g., bounded [-1, 1] vs mean/std normalization used by some diffusion /
  flow-style policies).
- Align dataset outputs with policy expectations:
  - Training batches SHALL provide actions in the policy-declared normalized space.
  - Add explicit validation/guardrails so “wrong normalization” becomes a fast, obvious error instead of a silent
    behavior bug.
- Establish a vision preprocessing contract:
  - Images passed into the vision encoder are **pixel_values** already preprocessed by the correct HuggingFace
    image processor (SigLIP/PaliGemma), with well-defined dtype/range/shape.
  - Prevent common failure modes like double-rescaling (0–1 input rescaled again by 1/255) or mismatched
    normalization statistics.
- Fix checkpointing so it is correct and supports both workflows:
  - **Default**: a self-contained checkpoint that can be loaded without requiring a base-model download.
  - **Optional**: adapter-only artifacts for LoRA/PEFT workflows (small artifacts intended for sharing), alongside
    separate non-VLM weights as needed.
  - Ensure safetensors serialization is robust to **shared tensors** (common in Transformer models).
  - Ensure embodiment metadata is complete (covers both `ActionSpaceSpec` and `ProprioSpec`).
- Bring implementation back in line with the intended modular composition model:
  - `build_policy()` is config-driven and uses registries for module selection (vision encoder, backbone, merger,
    head, decoder) instead of hardcoded wiring.
- Address correctness/performance footguns surfaced in review (batching semantics for language, AMP dtype
  promotion due to mask/readout tensor creation, action masking support in loss where applicable, shape
  validation for action limits).

## Capabilities

### New Capabilities

- `policy-action-normalization`: Define policy-declared action normalization modes (default bounds [-1, 1];
  optional mean/std), required metadata, and decoder unnormalization behavior.
- `policy-vision-preprocessing`: Define the image **pixel_values** preprocessing contract (HF processor
  compatibility), expected dtype/range/shape, and validation rules at module boundaries.
- `policy-checkpointing`: Define save/load artifacts (self-contained vs adapter-only), safetensors shared-tensor
  handling, and embodiment metadata requirements.
- `policy-factory-composition`: Define registry-driven, config-based module composition requirements for
  `build_policy()` and build-time integration validation.

### Modified Capabilities

- `data-transforms`: Extend normalization requirements to support an explicit bounds-to-[-1, 1] mode suitable for
  action normalization (while preserving existing z-score behavior).

## Impact

- Affected code (expected): `src/yavla/models/*` (policy, decoder, merger, heads, backbone, types), and likely
  `src/yavla/data/*` (normalization and image preprocessing wiring) plus tests under `tests/models/*`.
- Checkpoints/configs: introduce explicit normalization/preprocessing metadata and an additional adapter-only
  saving/loading path for LoRA workflows.
- Performance: prefer doing preprocessing in dataloaders and validation at initialization / once-per-batch (not
  per-token). Keep inference-time overhead negligible (linear scaling only).

### References (background + prior art)

- LeRobot normalization modes (bounds ↔ [-1, 1], mean/std) in its processor pipeline:
  - https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/processor/normalize_processor.py
- Stable-Baselines3 action rescaling utilities ([-1, 1] ↔ [low, high]):
  - https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/policies.html
- HuggingFace SigLIP image processor contract (expects 0–255 input; warns about double-rescaling; normalizes):
  - https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/siglip/image_processing_siglip.py
- HuggingFace PaliGemma processor generates `pixel_values` via its `image_processor`:
  - https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/paligemma/processing_paligemma.py
- Safetensors guidance for shared tensors (recommends `save_model` / `load_model` vs `save_file(state_dict)`):
  - https://huggingface.co/docs/safetensors/torch_shared_tensors
- Transformers PEFT overview (adapters are smaller artifacts intended for sharing/loading on top of a base model):
  - https://huggingface.co/docs/transformers/v4.33.2/en/peft
- PEFT discussion: saving full model vs adapters:
  - https://github.com/huggingface/peft/issues/636

