## Context

The MVP MLP policy stack (`src/yavla/models/*`) is now present and mostly wired end-to-end, but several *implicit* IO
contracts are currently inconsistent across layers (dataset → model → decoder → checkpoint). These mismatches already
manifest as a unit test failure and are likely to become silent correctness or performance issues once we train with a
real HuggingFace VLM (PaliGemma).

Key symptoms / evidence in the current repo state:

- **Action contract mismatch**:
  - The action decoder (`SimpleActionDecoder`) unnormalizes assuming **normalized actions in `[-1, 1]`**.
  - The dataset normalization transform (`NormalizeTransform(mode="min-max")`) currently maps to **`[0, 1]`** per the
    existing `data-transforms` spec, which is incompatible with the decoder’s `[-1, 1]` assumption.
  - The test suite currently fails because `ActionSpaceSpec` is used with a non-existent kwarg
    `clip_unnormalized` (`tests/models/test_policy.py:63`).
- **Vision preprocessing contract mismatch**:
  - The vision encoder calls `base_model.get_image_features(pixel_values)` and assumes its input is already the
    model-specific **`pixel_values`** tensor.
  - HuggingFace image processors (e.g., SigLIP) are explicit about expected input ranges and double-rescaling pitfalls:
    they expect pixel values in **`[0, 255]`** unless `do_rescale=False`.  
    Ref: `SiglipImageProcessor.preprocess()` docstring and `warning_once` about rescaling already-rescaled images
    (Transformers). https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/siglip/image_processing_siglip.py
  - `PaliGemmaProcessor`’s output includes `pixel_values` derived from its `image_processor`.  
    Ref: `PaliGemmaProcessor.__call__()` producing `return_data = {**inputs, "pixel_values": pixel_values}`.
    https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/paligemma/processing_paligemma.py
- **Checkpoint serialization risk (shared tensors)**:
  - `VLAPolicy.save_pretrained()` currently uses `safetensors.torch.save_file(dict(self.state_dict()), ...)`.
  - Safetensors explicitly documents that shared tensors (e.g., tied `embeddings` and `lm_head` in Transformers) are not
    handled by raw `save_file(state_dict)` and recommends `save_model` / `load_model` instead.  
    Ref: safetensors “Torch shared tensors” TL;DR + explanation.  
    https://raw.githubusercontent.com/huggingface/safetensors/main/docs/source/torch_shared_tensors.mdx
- **Performance footguns** (not necessarily failing tests, but high risk for real training):
  - `VLMBackbone.forward()` forces `output_hidden_states=True` and then uses `outputs.hidden_states[-1]`. For large
    models, returning *all* layer hidden states is a major memory/time cost. Transformers’ output types confirm that
    `hidden_states` is returned only when `output_hidden_states=True`.  
    Ref: `BaseModelOutput.hidden_states` docs. https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/modeling_outputs.py
  - The merger currently creates masks/readouts in default `float32`, which can silently upcast and defeat AMP/bfloat16
    paths by dtype promotion during concatenation.

User constraints / decisions captured so far (from discussion):

- **Default action normalization**: choose bounded **`[-1, 1]`** (aligns with current decoder + LeRobot familiarity).
- **Vision preprocessing**: preprocessing happens in the **data pipeline**; model consumes precomputed `pixel_values`.
- **Checkpointing**: default **self-contained** checkpoints; **adapter-only** as an optional export path.
- **Normalization flexibility**: must support *both* bounds-based and mean/std-style normalization, since different
  policy families (e.g., π0-style) rely on different conventions.

## Goals / Non-Goals

**Goals:**

- Make the MVP policy’s **action contract explicit and end-to-end consistent**:
  - Default: policy predicts normalized actions in `[-1, 1]`.
  - Decoder unnormalizes via `ActionSpaceSpec.limits` (linear scaling).
  - Dataset/training batches provide actions in the policy-declared normalized space.
- Add **switchable action normalization strategies** (bounds vs mean/std vs quantile-bounds) and persist the choice in
  config + checkpoint metadata so loading is unambiguous.
  - Prior art: Octo supports `NORMAL` (mean/std) and `BOUNDS` (normalized to `[-1, 1]`).  
    Ref: `NormalizationType` and bounds normalization logic. https://raw.githubusercontent.com/octo-models/octo/main/octo/data/utils/data_utils.py
  - Prior art: OpenPI (π0) normalizes actions/proprio using dataset stats (mean/std), and also supports quantile-based
    normalization mapping to `[-1, 1]`.  
    Ref: Normalize logic and quantile normalization mapping. https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/transforms.py  
    Ref: rationale for storing/reloading norm stats. https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/docs/norm_stats.md
- Define a **vision preprocessing contract**: the model expects `pixel_values` created by the correct HF processor for
  the selected VLM, and the data pipeline is responsible for producing it.
- Fix checkpointing so it is **correct for real Transformer models**:
  - Shared-tensor-safe serialization.
  - Default self-contained (no base-model download required).
  - Optional adapter-only artifacts for PEFT sharing.
  - Complete embodiment metadata (action + proprio specs, control frequency, normalization mode).
- Reduce avoidable perf/memory overheads in the hot path (train/infer) without changing the high-level architecture.

**Non-Goals:**

- Changing the MVP head family (no diffusion/flow heads in this change).
- Multi-camera / multi-view support beyond current MVP constraints.
- Redesigning training infrastructure, distributed training, or evaluation benchmarks.
- Introducing per-step runtime checks that materially slow inference; guardrails should be “init-time” or “once per
  batch” unless explicitly in debug mode.

## Decisions

### D1: Default action space = normalized `[-1, 1]` + linear unscale with `ActionSpaceSpec.limits`

**Decision:** Keep and formalize the current decoder contract: the action head outputs normalized actions in `[-1, 1]`
by default, and the decoder performs:

`unnormalized = (a + 1) / 2 * (hi - lo) + lo`

**Why:** This is a widely used convention for continuous control:

- SB3 commonly works with normalized actions in `[-1, 1]` and rescales to env bounds.  
  (Reference used in proposal; see SB3 policies module.) https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/policies.html
- RL implementations like Spinning Up squash actions with `tanh` then scale by an action limit to enforce bounds.  
  Ref: `SquashedGaussianMLPActor` applies `torch.tanh` then multiplies by `act_limit`.
  https://raw.githubusercontent.com/openai/spinningup/master/spinup/algos/pytorch/sac/core.py

**Alternatives considered:**

- **Mean/std (“normal”) target space**: matches π0/OpenPI’s default practice (dataset-statistics normalization).
  Rejected as default because it does not align with the current decoder and can be ambiguous without stored stats.
  Still supported as an explicit mode (see D2).
- **Hard clamp in decoder**: would prevent out-of-range actions but adds overhead and can hide upstream bugs. We keep the
  decoder unclamped by default; clipping can be done in env or as an optional postprocess in non-latency-critical paths.

### D2: Action normalization is a first-class, switchable contract (policy-declared)

**Decision:** Introduce an explicit normalization mode configuration that is:

- Declared by the policy/config (what the head predicts).
- Enforced by the data pipeline (what `TrainingBatch.actions` contains).
- Recorded in checkpoints (so `from_pretrained` is unambiguous).

Supported modes (initial set):

1. `bounds` (default): map physical action values ↔ normalized `[-1, 1]` using `ActionSpaceSpec.limits`.
2. `mean_std`: z-score `(x - mean) / (std + eps)` using stored action stats.
3. `quantile_bounds`: map quantiles to `[-1, 1]` using `(q_low, q_high)` with optional clipping (Octo/OpenPI style).

**Why:** Real VLA ecosystems need both:

- Octo explicitly supports both mean/std and bounds normalization for action+proprio.  
  Ref: `NormalizationType.NORMAL` vs `NormalizationType.BOUNDS`. https://raw.githubusercontent.com/octo-models/octo/main/octo/data/utils/data_utils.py
- OpenPI documents storing/reloading normalization stats as part of the checkpointed “assets,” and its transforms
  implement both mean/std and quantile-to-`[-1, 1]`.  
  Ref: https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/docs/norm_stats.md  
  Ref: https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/transforms.py
- LeRobot’s processor pipeline similarly treats normalization stats + modes as explicit pipeline state and supports
  multiple normalization modes.  
  Ref: `_NormalizationMixin` and step registry. https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/processor/normalize_processor.py

**Implementation sketch (contract-level):**

- `PolicyConfig` (or a sub-config) declares `action_normalization.mode`.
- Data pipeline uses the same mode for action targets.
- Decoder uses mode metadata to choose the correct unnormalization behavior during inference.

### D3: Extend dataset transforms with a `[-1, 1]` bounds mode (do not repurpose existing `min-max`)

**Decision:** Update the `data-transforms` capability to add a new explicit mode that maps to `[-1, 1]` (e.g., `bounds`
or `min-max-[-1,1]`) instead of changing the semantics of existing `min-max` (currently `[0, 1]` by spec).

**Why:** The existing `data-transforms` spec explicitly defines min-max normalization to `[0, 1]`. Mutating that meaning
would silently change behavior for any users relying on it. Adding a new mode is explicit and reviewable.

**Alternatives considered:**

- Replace `min-max` with `[-1, 1]`: rejected due to backward-compat and surprise factor.
- Only normalize actions in-model: rejected (user decision is data-pipeline preprocessing + performance).

### D4: Vision preprocessing happens in the data pipeline; model consumes `pixel_values`

**Decision:** The model/vision encoder expects precomputed `pixel_values` as produced by the correct HuggingFace
processor for the configured VLM (e.g., SigLIP/PaliGemma). Data transforms are responsible for converting dataset image
frames into `pixel_values` with correct dtype/range/shape, and for avoiding double-rescaling.

**Why:**

- HF processors are the canonical source of truth for resizing/rescaling/normalization conventions. SigLIP explicitly
  warns about double-rescaling if the caller provides `[0, 1]` inputs while leaving `do_rescale=True`.  
  Ref: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/siglip/image_processing_siglip.py
- PaliGemma’s processor produces `pixel_values` and is the intended integration point for multimodal inputs.  
  Ref: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/paligemma/processing_paligemma.py

**Performance note:** Do preprocessing in dataloader workers (CPU) and keep the model path “pure tensor plumbing.”
Avoid per-token/per-layer validation in the hot path; validate at transform construction time and (optionally) once per
batch (debug).

**Alternatives considered:**

- Preprocess inside `VisionEncoder.encode_images`: rejected (would mix IO concerns into model and duplicate work across
  callers).
- Torchvision-only preprocessing: potentially faster, but high risk of drifting from HF’s exact conventions; can be an
  optional “fast path” later if needed.

### D5: Backbone should return last hidden state without `output_hidden_states=True`

**Decision:** Avoid always requesting `output_hidden_states=True`. Instead, call the underlying base model forward to
obtain the last hidden state directly, then extract readout tokens from that last hidden state.

**Why:** `output_hidden_states=True` returns a tuple of all layer activations; this is expensive for large models and
unnecessary for the MLP head, which only needs the final states. Transformers output docs confirm that `hidden_states`
is only included when requested.  
Ref: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/modeling_outputs.py

**Supporting evidence (PaliGemma forward structure):**

In Transformers’ PaliGemma implementation, `PaliGemmaForConditionalGeneration.forward()` calls `self.model(...)` and
uses `outputs[0]` as the last hidden state for computing logits; the full per-layer `outputs.hidden_states` is only
populated when `output_hidden_states=True`.  
Ref: https://raw.githubusercontent.com/huggingface/transformers/556312cd/src/transformers/models/paligemma/modeling_paligemma.py

**Alternatives considered:**

- Keep `output_hidden_states=True`: rejected due to large memory/time overhead.
- Use logits or LM head outputs: rejected (wastes compute and introduces additional coupling).

### D6: AMP-safe dtype/device discipline in merger + masks

**Decision:** Ensure that tensors created in the merger (readout tokens, attention masks) match:

- Device: same as `vision_tokens.device`.
- Dtype: readout tokens match `vision_tokens.dtype` to avoid AMP upcasts; masks use an agreed dtype (`torch.long` or
  `torch.bool`) compatible with the HF model’s expectations.

**Why:** Silent dtype promotion can defeat mixed precision and increase memory. Keeping dtype consistent is a low-cost,
high-impact performance fix.

### D7: Action masking is honored in loss when provided

**Decision:** If `TrainingBatch.action_mask` is present (e.g., variable-length chunks/padding), loss functions that
operate over `[B, T, A]` must mask out invalid timesteps.

**Why:** Ignoring masks produces incorrect gradients on padded timesteps; this can silently harm training or make
variable-length batching impossible.

### D8: Checkpoint format: shared-tensor-safe, self-contained by default; adapter-only optional

**Decision:** Redesign `save_pretrained/from_pretrained` so that:

- The default artifact is **self-contained** and loads without base-model download.
- Serialization is **robust to shared tensors** by using recommended APIs.
- Adapter-only export remains possible (small artifact), but is not the default.

**Why (shared tensors):** Safetensors documents that raw `save_file(state_dict)` does not support shared tensors and
recommends `save_model/load_model`.  
Ref: https://raw.githubusercontent.com/huggingface/safetensors/main/docs/source/torch_shared_tensors.mdx

**Concrete layout (recommended):**

- `config.json`: full `PolicyConfig` including normalization + vision preprocessing metadata.
- `embodiment.json`: `ActionSpaceSpec` + `ProprioSpec` + control frequency (`dt_hz`) + control_mode/frame.
- `action_stats.json`: only required for `mean_std` / `quantile_bounds` modes (can be empty for pure bounds mode).
- `vlm/`: HuggingFace base model saved via `save_pretrained(..., safe_serialization=True)` (ensures self-contained).
- `policy_modules.safetensors`: non-VLM weights (merger/readout params, proprio encoder, action head, decoder state if
  any) saved via safetensors.
- Optional `adapter/`: PEFT adapter via `PeftModel.save_pretrained()` (for sharing / adapter-only workflows).

**Alternatives considered:**

- Single-file `model.safetensors` for entire `VLAPolicy`: attractive simplicity, but fragile if PEFT wrappers change
  `state_dict()` semantics and difficult to guarantee self-contained behavior for adapter setups.
- Adapter-only as default: rejected (user wants self-contained default for reproducibility and offline use).

### D9: Registry-driven `build_policy` composition (no hardcoded wiring)

**Decision:** `build_policy()` should instantiate module families via their registries based on config `type` fields:
vision encoder, proprio encoder, merger, backbone, head, decoder.

**Why:** This is the core architectural promise of YAVLA’s 7-module pipeline: swappable components through config, not
manual rewiring. Keeping `build_policy` hardcoded undermines modularity and makes future heads/backbones costly.

### D10: Validation strategy: cheap guardrails, no per-token overhead

**Decision:** Add explicit, low-cost validations at boundaries:

- `ActionSpaceSpec.limits` shape `[action_dim, 2]` when provided.
- Language batching: `ObservationBatch.language` supports broadcasting a single string to batch size `B` (and validates
  list length when a list is provided).
- Vision inputs: enforce `pixel_values` dtype/range expectations in the data pipeline; the model assumes correctness.

Validation should run at construction time and/or once per batch in debug mode, not inside per-token loops.

## Risks / Trade-offs

- **[More configuration surface area]** → Mitigation: provide safe defaults (`bounds` + `pixel_values` preprocessing),
  and store normalization/processor metadata in checkpoints so loading is deterministic.
- **[Action limits may be missing or unreliable for some datasets]** → Mitigation: support `mean_std` and
  `quantile_bounds` modes that rely on dataset stats instead of physical bounds; require explicit selection.
- **[HF processor preprocessing could be a dataloader bottleneck]** → Mitigation: keep it in worker processes; allow an
  optional torchvision-equivalent fast path later; encourage caching/preprocessing offline for large datasets.
- **[Breaking change for existing training scripts expecting raw images]** → Mitigation: provide a clear error message
  when raw images are fed to a model expecting `pixel_values`, and document migration steps; optionally keep a temporary
  compatibility mode behind a config flag.
- **[Checkpoint size increases for self-contained default]** → Mitigation: keep adapter-only export for sharing; provide
  documentation for expected sizes and artifact choices.

## Migration Plan

1. **Introduce config + metadata fields** (backward compatible):
   - Add `action_normalization` mode + params and `vision_preprocessing` contract fields with defaults matching MVP.
   - Bump `config_version` and add tolerant parsing in `from_pretrained` (missing fields → defaults).
2. **Update data transforms**:
   - Add explicit `[-1, 1]` normalization mode in `NormalizeTransform`.
   - Wire action normalization selection into the dataloader pipeline for `action` specifically.
3. **Update model hot path**:
   - Fix language broadcasting semantics.
   - Make merger allocations dtype-safe for AMP.
   - Update backbone forward to avoid `output_hidden_states=True` in the common path.
4. **Update checkpointing**:
   - Implement new self-contained checkpoint layout while keeping a loader path for legacy checkpoints.
   - Add adapter-only export option.
5. **Tests + documentation**:
   - Fix the failing decoder test kwarg (`clip_unnormalized`) to match the actual contract.
   - Add contract tests that assert action normalization roundtrips and pixel_values dtype/shape assumptions.

Rollback strategy: keep legacy load paths and config defaults so previously saved checkpoints remain loadable. If new
checkpoint layout causes issues, disable new save path via a temporary feature flag and continue using legacy artifacts
until resolved.

## Open Questions

- **Tanh-squash vs “learned bounded”**: should the MLP head optionally apply `tanh` to enforce `[-1, 1]` at inference?
  (Common in RL; see SpinningUp SAC actor. https://raw.githubusercontent.com/openai/spinningup/master/spinup/algos/pytorch/sac/core.py)
- **Which quantiles for `quantile_bounds`**: p01/p99 (Octo) vs q10/q90 (LeRobot appears to support q10/q90) vs
  q01/q99 (OpenPI). We should standardize naming and store the chosen stats keys explicitly.
- **Where to host the HF processor**: data pipeline transform vs a small “preprocessor” module that can be reused by
  evaluation/inference codepaths outside training.
- **Multi-embodiment metadata**: should `embodiment.json` fully embed `ActionSpaceSpec`/`ProprioSpec` (including limits)
  or reference external robot definitions?
