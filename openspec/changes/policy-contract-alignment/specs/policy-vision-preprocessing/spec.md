## ADDED Requirements

<!--
Refs (canonical processor contracts):
- SigLIP image processor expects 0–255 input unless do_rescale=False; warns on double-rescaling:
  https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/siglip/image_processing_siglip.py
- PaliGemma processor produces `pixel_values` via its `image_processor` and returns it in BatchFeature:
  https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/paligemma/processing_paligemma.py
-->

### Requirement: The model consumes preprocessed `pixel_values`
The vision preprocessing contract for VLA policies SHALL be:

- `ObservationBatch.images` contains **preprocessed** tensors compatible with the configured VLM’s image tower.
- These tensors SHALL be named/treated as **`pixel_values`** (HuggingFace convention).
- Model code (vision encoder + backbone) SHALL NOT apply additional rescaling/normalization on top of `pixel_values`.

#### Scenario: `pixel_values` are passed through unchanged
- **WHEN** the vision encoder is called with `ObservationBatch.images={"cam": pixel_values}`
- **THEN** the vision encoder SHALL forward `pixel_values` into the underlying VLM image feature path without applying extra preprocessing transforms

### Requirement: `pixel_values` originate from the correct HuggingFace processor
The data pipeline SHALL generate `pixel_values` using the HuggingFace processor/image-processor corresponding to the
configured VLM checkpoint (e.g., via `AutoProcessor.from_pretrained(vlm_name)`), ensuring the resize/rescale/normalize
steps match the model’s expected conventions.

#### Scenario: Processor output is used as model input
- **WHEN** a sample contains a raw image frame and a data pipeline step runs the VLM’s HuggingFace processor
- **THEN** the resulting `pixel_values` tensor from that processor SHALL be placed into `ObservationBatch.images` for model consumption

### Requirement: Double-rescaling is prevented
The preprocessing pipeline MUST avoid the common “double-rescaling” failure mode:

- If raw images are provided as floats in `[0, 1]`, the processor MUST be configured with `do_rescale=False`.
- If raw images are provided in `[0, 255]` (uint8 or float), the processor MAY use its default `do_rescale=True`.

This requirement exists because the SigLIP processor explicitly warns when it detects already-scaled images while
`do_rescale=True`.

#### Scenario: Float `[0, 1]` images do not get rescaled again
- **WHEN** the pipeline receives float images with values in `[0, 1]`
- **THEN** it SHALL configure the processor such that it does not apply an additional `1/255` rescale

### Requirement: `pixel_values` tensor shape, dtype, and device placement
`pixel_values` SHALL have shape `[B, C, H, W]` with `C == 3` (RGB) and dtype compatible with the model forward path
(typically `float32`, optionally cast to `float16/bfloat16` in mixed precision).

Device placement rules:
- Dataloader/transform code MAY output `pixel_values` on CPU.
- Model code SHALL move `pixel_values` to the model/device as part of the usual batch-to-device transfer (outside the processor).

#### Scenario: Batch shape is channels-first
- **WHEN** a processed batch is formed
- **THEN** `pixel_values` SHALL be channels-first (`[B, 3, H, W]`) as expected by HuggingFace vision backbones by default

### Requirement: MVP single-camera constraint is explicit
For the MVP policy family, the vision encoder SHALL support a single camera key only.

#### Scenario: Multi-camera images are rejected in MVP
- **WHEN** `ObservationBatch.images` contains more than one key
- **THEN** the MVP vision encoder SHALL raise a `ValueError` describing that multi-camera support is not enabled

