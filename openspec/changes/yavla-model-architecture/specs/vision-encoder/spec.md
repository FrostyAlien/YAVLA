## ADDED Requirements

### Requirement: SigLIP vision encoder
`SigLIPEncoder(VisionEncoderBase)` SHALL wrap PaliGemma's built-in SigLIP vision tower and linear projector via a reference to the backbone's `PaliGemmaForConditionalGeneration` instance (NOT its own copy), calling `paligemma.get_image_features(pixel_values)` to return image tokens already projected into the Gemma embedding space. For standalone use (non-PaliGemma backbones), it MAY wrap `google/siglip-so400m-patch14-384` directly with a 2-layer MLP projector. Supports frozen, LoRA, or full fine-tune via `VisionTuningConfig`.

<!-- Ref: PaliGemma built-in SigLIP + projector: https://github.com/huggingface/transformers/blob/556312cd/src/transformers/models/paligemma/modeling_paligemma.py#L92-L100
     Ref: π0 reuses PaliGemma's SigLIP: https://github.com/Physical-Intelligence/openpi/blob/981483dc/src/openpi/models/pi0.py#L108-L130 -->

#### Scenario: LoRA adaptation on last K blocks
- **WHEN** `VisionTuningConfig(mode="lora_last_k", target_blocks=4, rank=16)` is set
- **THEN** LoRA adapters SHALL be applied to the last 4 ViT blocks only; earlier blocks remain frozen

#### Scenario: Merge LoRA for deployment
- **WHEN** `encoder.merge_peft()` is called
- **THEN** LoRA weights SHALL merge into base weights, producing a fully frozen encoder with zero adapter overhead

### Requirement: DINOv2 vision encoder
`DINOv2Encoder(VisionEncoderBase)` SHALL wrap `facebook/dinov2-large` via HuggingFace with the same interface as SigLIP (configurable layer extraction, MLP projection, tuning config).

#### Scenario: Encode images
- **WHEN** `encode_images({"cam0": tensor[B, 3, 224, 224]})` is called
- **THEN** it SHALL return `TokenBatch` with `tokens` shape `[B, 256, output_dim]`

### Requirement: Dual (Prismatic) vision encoder
`DualEncoder(VisionEncoderBase)` SHALL run SigLIP and DINOv2 in parallel, concatenate their token sequences along the token dimension, and project the combined features.

#### Scenario: Dual encoding
- **WHEN** `encode_images(images)` is called
- **THEN** it SHALL return `TokenBatch` with `tokens` shape `[B, 2*num_patches, output_dim]`

### Requirement: VisionEncoderConfig with tuning
`VisionEncoderConfig` SHALL include `type: str`, `model_name: str`, `extract_layer: int = -2`, and `tuning: VisionTuningConfig`. `VisionTuningConfig` SHALL have `mode: str = "frozen"`, `target_blocks: int = 4`, `rank: int = 16`.

#### Scenario: Default config
- **WHEN** `VisionEncoderConfig()` is constructed
- **THEN** `tuning.mode` SHALL be `"frozen"`
