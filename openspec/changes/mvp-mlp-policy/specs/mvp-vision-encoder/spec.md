## ADDED Requirements

<!-- Ref: PaliGemma SigLIP + projector (scaling via 1/sqrt): https://github.com/huggingface/transformers/blob/556312cd/src/transformers/models/paligemma/modeling_paligemma.py#L92-L100
     Ref: π0 reuses PaliGemma's SigLIP: https://github.com/Physical-Intelligence/openpi/blob/981483dc/src/openpi/models/pi0.py#L108-L130 -->

### Requirement: PaliGemma vision encoder wrapper
`PaliGemmaVisionEncoder(VisionEncoderBase)` SHALL be a thin wrapper that holds a reference to the backbone's unwrapped `PaliGemmaForConditionalGeneration` base model (NOT the PeftModel wrapper, NOT its own copy) and calls `base_model.get_image_features(pixel_values)` to obtain image tokens already projected and scaled into the Gemma embedding space as a `Tensor[B, num_patches, D]` (returned directly, NOT as a dict). It does NOT load a separate SigLIP model. It does NOT rescale — `get_image_features` applies `1/sqrt(hidden_size)` internally. When LoRA is applied, the vision encoder MUST use the unwrapped base model (via `backbone.base_model`) because `PeftModel` may not reliably proxy `get_image_features()`.

#### Scenario: Encode single-camera batch
- **WHEN** `encode_images({"cam0": tensor[B, 3, 224, 224]})` is called (encoder already holds PaliGemma ref)
- **THEN** it SHALL return embeddings shape `[B, 256, backbone_dim]` (256 = (224/14)²)

#### Scenario: Reject multi-camera input in MVP
- **WHEN** `encode_images(images)` is called with more than one camera key
- **THEN** it SHALL raise `ValueError` indicating MVP encoder supports single-camera input only

#### Scenario: Weights NOT frozen by default
- **WHEN** the encoder is initialized with default `FreezeConfig()`
- **THEN** all SigLIP and projector parameters SHALL have `requires_grad=True`

#### Scenario: Frozen via FreezeConfig
- **WHEN** `FreezeConfig(freeze_modules=["vision_tower", "multi_modal_projector"])` is applied by `build_policy`
- **THEN** SigLIP and projector parameters SHALL have `requires_grad=False`

### Requirement: Vision encoder config
`VisionEncoderConfig` SHALL be a `@dataclass` with `type: str = "paligemma_siglip"`, compatible with tyro CLI. Freeze/LoRA is handled by `FreezeConfig` at the policy level, NOT per-encoder.

#### Scenario: Default config
- **WHEN** `VisionEncoderConfig()` is constructed with defaults
- **THEN** `type` SHALL be `"paligemma_siglip"`
