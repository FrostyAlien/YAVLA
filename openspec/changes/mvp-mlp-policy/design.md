## Context

YAVLA has a complete dataset layer (`src/yavla/data/`) that produces `DataLoader`s of LeRobot-format samples. The model layer (`src/yavla/models/`) is empty. The architecture research doc defines 7 core modules with Protocol-based interfaces, but implementing everything at once is too risky. This design covers the MVP: the simplest end-to-end policy that exercises all 7 module slots using an MLP regression action head, modeled after OpenVLA-OFT's architecture but wrapped in our modular structure.

Reference implementations (with verified source URLs):
- **OpenVLA-OFT** (`moojink/openvla-oft`, SHA `e4287e94`): Separate SigLIP+DINOv2 → MLP projector → Llama-2 7B via `inputs_embeds` → extract hidden states at action positions → L1RegressionActionHead (MLPResNet). Key: uses `input_ids=None, inputs_embeds=multimodal_embeddings` to bypass LLM embedding table. [modeling_prismatic.py#L571-L638](https://github.com/moojink/openvla-oft/blob/e4287e94/prismatic/extern/hf/modeling_prismatic.py#L571-L638)
- **π0 / OpenPI** (`Physical-Intelligence/openpi`, SHA `981483dc`): Uses PaliGemma's built-in SigLIP (`self.PaliGemma.img()`), action tokens via separate linear projection, dual-expert attention. Key: PaliGemma's SigLIP is reused, NOT bypassed. [pi0.py#L108-L215](https://github.com/Physical-Intelligence/openpi/blob/981483dc/src/openpi/models/pi0.py#L108-L215)
- **Octo** (`octo-models/octo`, SHA `241fb351`): Readout tokens = zeros + learned positional embedding `N(0, 0.02)`, bidirectional transformer with custom attention rules (readouts attend to obs/task, obs/task cannot attend to readouts). [octo_module.py#L248-L262](https://github.com/octo-models/octo/blob/241fb351/octo/model/octo_module.py#L248-L262)
- **LeRobot** (`huggingface/lerobot`, SHA `5f152322`): `PreTrainedPolicy(nn.Module, HubMixin, ABC)` base, draccus `ChoiceRegistry` configs, `@register_subclass` pattern, `configuration_*.py` / `modeling_*.py` naming. [pretrained.py#L45](https://github.com/huggingface/lerobot/blob/5f152322/src/lerobot/policies/pretrained.py#L45)

**Critical architectural insight (from review):** PaliGemma is decoder-only (SigLIP → linear projector → Gemma LM), NOT encoder-decoder. Its `forward()` accepts `inputs_embeds` to bypass the built-in vision pipeline. `token_type_ids` controls bidirectional (image, type=0 prefix) vs causal (text, type=1) attention — HF PaliGemma un-masks prefix tokens where `token_type_ids == 0`. [modeling_paligemma.py#L304-L387](https://github.com/huggingface/transformers/blob/556312cd/src/transformers/models/paligemma/modeling_paligemma.py#L304-L387)

## Goals / Non-Goals

**Goals:**
- All 7 module slots implemented in their simplest form, proving the interfaces work
- End-to-end forward pass: images + proprio + language → action chunk
- Training: L1 loss on continuous actions, single-GPU
- Checkpoint: save_pretrained / from_pretrained with safetensors
- Contract tests for every module Protocol
- Readout mode only (no joint-token mode)

**Non-Goals:**
- Flow matching, diffusion, CVAE, or any non-MLP action head
- Perceiver resampler (use simple concat)
- Dual-expert backbone (joint-token mode)
- Multi-embodiment support (single robot config)
- Inference optimization (no KV cache, no deployment profiles)
- Distributed training (single-GPU only)
- Temporal ensembling in action decoder
- Integration with wandb or evaluation benchmarks

## Decisions

### D1: Module boundary contracts use typed dataclasses, with tensor fast-path where needed

**Choice:** Core module boundaries use typed `@dataclass` containers (`ObservationBatch`, `TokenBatch`, `BackboneOutput`, `ActionPrediction`, `ActionChunk`, `LossDict`, `TrainingBatch`). For latency-sensitive merger/backbone wiring, Protocols use direct tensor tuples (`inputs_embeds`, `attention_mask`, `token_type_ids`) as a deliberate fast-path.

**Why not raw tensors:** Silent shape/mask bugs are the #1 debugging time sink in ML code. Typed containers catch mismatches at construction time and make the data flow self-documenting. The overhead is negligible (dataclass construction is ~100ns).

**Why not NamedTuple:** Dataclasses support `None` defaults, mutable fields, and `__post_init__` validation. NamedTuples are immutable and can't have optional fields cleanly.

### D2: Dual-layer interface — Protocols for contracts, ABC base classes for implementation

**Choice:** `@runtime_checkable` Protocols define the public API (what a module must do). `*Base(nn.Module, ABC)` classes provide the blessed implementation path with shared utilities. The factory accepts either.

**Why both:** Protocols enable third-party implementations without inheritance. ABC base classes provide `state_dict` lifecycle, device management, and shared helpers that every implementation needs. This matches the architecture doc's unanimous architect consensus.

### D3: Use PaliGemma's built-in SigLIP, configurable freeze + LoRA via `peft`

**Choice:** Reuse PaliGemma's built-in SigLIP vision tower and linear projector rather than loading a separate SigLIP model. Our `VisionEncoder` module is a thin wrapper that calls `paligemma_model.get_image_features(pixel_values, return_dict=True).pooler_output` to get projected image tokens already in the Gemma embedding space.

**Why reuse PaliGemma's SigLIP (not separate):** PaliGemma's SigLIP and projector are pretrained together — the projector maps SigLIP features into Gemma's embedding space. Loading a separate SigLIP would require training a new projector from scratch. π0 confirms this pattern: it calls `self.PaliGemma.img(images)` to get vision tokens. See [pi0.py#L108-L130](https://github.com/Physical-Intelligence/openpi/blob/981483dc/src/openpi/models/pi0.py#L108-L130).

**Freeze/LoRA policy (NOT frozen by default):** The VLM is NOT frozen by default. Instead, `FreezeConfig` specifies which module groups to freeze (e.g. `["vision_tower", "multi_modal_projector"]`) and which to apply LoRA to (e.g. `["language_model.model.layers"]`). LoRA is applied via the `peft` library (`peft.get_peft_model` + `LoraConfig`), NOT a custom implementation. This matches how OpenVLA-OFT fine-tunes with LoRA on the LLM while freezing vision. See [train.py#L89-L120](https://github.com/moojink/openvla-oft/blob/e4287e94/prismatic/training/train.py#L89-L120).

**Projector details:** PaliGemma uses a single `nn.Linear(vision_hidden_size, projection_dim)` + scaling by `1/sqrt(hidden_size)` (NOT L2 normalization). The wrapper calls `get_image_features(..., return_dict=True).pooler_output` which returns already-projected-and-scaled tokens — do NOT rescale again. See [modeling_paligemma.py#L92-L100](https://github.com/huggingface/transformers/blob/556312cd/src/transformers/models/paligemma/modeling_paligemma.py#L92-L100).

### D4: Token merger builds `inputs_embeds` with `token_type_ids`

**Choice:** The merger constructs a single `inputs_embeds` tensor by concatenating pre-embedded tokens: `[image_tokens | proprio_token | language_tokens | readout_tokens]`. It also builds `token_type_ids` (0=image/bidirectional prefix, 1=text+proprio+readout/causal) and `attention_mask`. The backbone receives these tensors directly — it does NOT receive raw images or text.

**Why `inputs_embeds` (not `input_ids`):** PaliGemma's `forward()` accepts either `input_ids` or `inputs_embeds`, but not both. Since we inject custom tokens (proprio, readout) that have no vocabulary entry, we must use `inputs_embeds` and set `input_ids=None`. This is the same pattern used by OpenVLA-OFT ([modeling_prismatic.py#L571-L638](https://github.com/moojink/openvla-oft/blob/e4287e94/prismatic/extern/hf/modeling_prismatic.py#L571-L638)) and π0 ([pi0.py#L200-L215](https://github.com/Physical-Intelligence/openpi/blob/981483dc/src/openpi/models/pi0.py#L200-L215)).

**`token_type_ids` semantics:** PaliGemma uses `token_type_ids` to control attention: image tokens are passed as type=0 (bidirectional prefix), text/other tokens as type=1 (causal). Proprio and readout tokens use type=1. Internally, HF may transform token-type conventions while constructing the hybrid mask; our contract is the call-site convention `0=image`, `1=non-image`. See [modeling_paligemma.py#L134-L138](https://github.com/huggingface/transformers/blob/556312cd/src/transformers/models/paligemma/modeling_paligemma.py#L134-L138).

**Token ordering:** `[image_tokens | proprio_token | language_tokens | readout_tokens]`. Readout tokens MUST be at the END so causal attention naturally gives them visibility into all prior tokens while preventing prior tokens from attending to readouts (Octo's isolation property for free). See Octo analysis in [octo_module.py#L257-L262](https://github.com/octo-models/octo/blob/241fb351/octo/model/octo_module.py#L257-L262).

**Why not Perceiver:** Same as before — MVP uses single camera, sequence length is manageable (~341 tokens).

### D5: PaliGemma's Gemma LM as backbone, readout tokens at sequence end

**Choice:** Wrap `PaliGemmaForConditionalGeneration` from HuggingFace. The backbone receives `inputs_embeds` (from the merger), `token_type_ids`, and `attention_mask`. It forwards through PaliGemma's Gemma decoder layers and returns `BackboneOutput` with `readout_states` extracted from the last `N_readout` positions of the final hidden layer.

**Token injection pattern:** Pass `input_ids=None, inputs_embeds=merged_embeds, pixel_values=None` to skip PaliGemma's internal vision pipeline. The hybrid causal mask (bidirectional for image tokens, causal for the rest) is controlled by `token_type_ids` via `create_causal_mask_mapping(..., is_training=self.training)` — NOT by `labels`. Do NOT pass dummy `labels`; they only trigger unnecessary LM loss computation. See [modeling_paligemma.py#L304-L387](https://github.com/huggingface/transformers/blob/556312cd/src/transformers/models/paligemma/modeling_paligemma.py#L304-L387).

**Readout token initialization:** Zeros + learned positional embedding, shape `(1, N_readout, D_backbone)`, initialized with `normal(0, 0.02)`. Follows Octo's pattern. See [octo_module.py#L315-L333](https://github.com/octo-models/octo/blob/241fb351/octo/model/octo_module.py#L315-L333).

**Readout extraction convention:** Readout tokens are always the final `N_readout` tokens in the sequence. Extraction is `hidden_states[:, -N_readout:, :]` — no variable-length indexing needed. This is simpler than OpenVLA-OFT's action-position extraction which depends on variable prompt length.

**Why readout at END:** In a causal LM, tokens can only attend to earlier positions. Readout tokens at the end see ALL prior tokens (image, proprio, language) but prior tokens cannot see readouts. This naturally replicates Octo's isolation property without custom attention masks.

**Why PaliGemma:** It's the VLM used by π0 (our eventual target). Starting with it means the backbone doesn't change when we add the dual-expert in post-MVP. 3B parameters is manageable on a single GPU with gradient checkpointing.

### D6: MLP regression action head with L1 loss

**Choice:** `MLPRegressionHead(ActionHeadBase)` — a 2-block MLPResNet that takes `readout_states` and predicts continuous action chunks. Architecture: `readout_states → mean_pool(N_readout, D) → LayerNorm → Linear → ReLU → [2× (LayerNorm → Linear → ReLU + residual)] → LayerNorm → Linear → reshape(chunk_len, action_dim)`. Loss: `F.l1_loss(predicted, ground_truth)`. Follows OpenVLA-OFT's `L1RegressionActionHead`. See [action_heads.py](https://github.com/moojink/openvla-oft/blob/e4287e94/prismatic/models/action_heads.py).

**Input:** `readout_states` shape `(B, N_readout, D)` from `BackboneOutput`. Mean-pooled to `(B, D)` before the MLP (following Octo's readout pooling pattern, [action_heads.py#L157-L165](https://github.com/octo-models/octo/blob/241fb351/octo/model/components/action_heads.py#L157-L165)).

**Why mean pool (not flatten):** Flattening `(B, 64, 2048)` → `(B, 131072)` creates a massive first linear layer. Mean pooling to `(B, 2048)` is parameter-efficient and matches Octo's approach.

**Why L1 (not MSE):** L1 is more robust to outliers in action data. OpenVLA-OFT uses L1.

**Chunk prediction:** Output shape `(B, chunk_len * action_dim)`, reshaped to `(B, chunk_len, action_dim)`.

### D7: Explicit ProprioEncoder module

**Choice:** `ProprioEncoder(nn.Module)` with a single `nn.Linear(proprio_dim, backbone_dim)` that projects proprioceptive state into a single token in the backbone's embedding space. This keeps modality-specific projection logic out of the merger.

**Why explicit module (not inline in merger):** The Oracle review identified that leaking embodiment-specific shapes into the merger/backbone is a design violation. The merger should only concatenate `TokenBatch` objects, never own projection logic. This also makes it trivial to swap proprio encoders for different robots in post-MVP multi-embodiment.

### D8: Registry pattern — explicit decorator registration

**Choice:** Generic `Registry[ConfigT, ModuleT]` class with `@register` decorator. Each module family (vision encoders, backbones, action heads, token mergers) gets its own registry instance.

```python
vision_registry = Registry[VisionEncoderConfig, VisionEncoderBase]("vision_encoder")

@vision_registry.register("siglip")
class SigLIPEncoder(VisionEncoderBase): ...
```

**Why not `__init_subclass__`:** Explicit registration is more visible — you can see all registered modules by inspecting the registry. `__init_subclass__` is implicit and can cause import-order issues.

**Why not entry points for MVP:** Entry points add complexity (pyproject.toml wiring, lazy import). MVP uses in-code registration only. Entry points are post-MVP.

### D9: Checkpoint format — safetensors + JSON config + embodiment metadata

**Choice:** `save_pretrained(path)` writes: `config.json` (full PolicyConfig with `config_version`), `model.safetensors` (weights), `action_stats.json` (normalization stats), `embodiment.json` (ActionSpaceSpec + ProprioSpec). `from_pretrained(path)` loads and validates.

**Why safetensors:** No pickle, no arbitrary code execution. Industry standard.

**Why embodiment metadata:** Prevents loading a checkpoint trained on robot A onto robot B without explicit opt-in (`strict=False`).

### D10: Configurable freeze + LoRA via `peft` library — NOT frozen by default

**Choice:** A `FreezeConfig` dataclass controls which VLM module groups to freeze and which to apply LoRA to. The VLM is NOT frozen by default — all parameters are trainable unless explicitly frozen. LoRA is applied via `peft.get_peft_model()` with `peft.LoraConfig`, NOT a custom implementation.

```python
@dataclass
class FreezeConfig:
    freeze_modules: list[str] = field(default_factory=list)  # e.g. ["vision_tower", "multi_modal_projector"]
    lora_target_modules: list[str] = field(default_factory=list)  # peft target_modules, e.g. ["q_proj", "v_proj"]
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
```

**`lora_target_modules` semantics:** These are peft `LoraConfig.target_modules` module-name suffixes (e.g. `["q_proj", "v_proj"]` by default for Gemma-family; broader sets like `["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` are intentional overrides), NOT full module paths. peft matches them against module names (exact or suffix match) across the model. This is the standard peft convention. See [peft LoraConfig docs](https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraConfig).

**Why `peft` library (not custom LoRA):** `peft` is the HuggingFace standard for parameter-efficient fine-tuning. It handles weight merging for deployment, adapter saving/loading, and is battle-tested across thousands of models. Building custom LoRA is unnecessary complexity.

**Why not frozen by default:** Freezing the entire VLM is overly conservative for MVP. Users should be able to fine-tune the full model, freeze selectively, or apply LoRA — all via config. OpenVLA-OFT demonstrates that LoRA on the LLM + frozen vision achieves strong results. See [train.py#L89-L120](https://github.com/moojink/openvla-oft/blob/e4287e94/prismatic/training/train.py#L89-L120).

**CRITICAL — PEFT freezes ALL base params in the wrapped model:** When `peft.get_peft_model()` wraps PaliGemma, it freezes ALL base parameters inside the wrapper (via `mark_only_lora_as_trainable`) — not just the targeted modules. Only LoRA adapter weights are trainable. Non-VLM modules (action head, proprio encoder, merger readout tokens) are separate `nn.Module`s in `VLAPolicy`, NOT inside the PeftModel wrapper, so they remain fully trainable. If LoRA is not enabled (`lora_target_modules` is empty), `peft` is not applied and `freeze_modules` controls trainability directly.

**PeftModel unwrap pattern:** After `peft.get_peft_model()` wraps PaliGemma, direct method access (e.g. `get_image_features()`, `get_input_embeddings()`) may not be reliably proxied. `build_policy` SHALL store a reference to the unwrapped base model BEFORE applying peft. Vision encoder and language embedding paths use this base reference. The backbone's `forward()` goes through the PeftModel wrapper (so LoRA adapters are active).

**Freeze + LoRA application order in `build_policy()`:**
1. Load PaliGemma → `base_model`
2. Store `base_model` reference (for vision encoder + language embeddings)
3. Freeze specified modules via `param.requires_grad_(False)` on `base_model`
4. If `lora_target_modules` non-empty: `peft_model = peft.get_peft_model(base_model, LoraConfig(target_modules=..., r=..., lora_alpha=..., lora_dropout=...))`
5. If peft applied: call `peft_model.enable_input_require_grads()` (required for gradients to flow through frozen base to LoRA adapters under gradient checkpointing)
6. If `config.backbone.gradient_checkpointing` is `True`: set `base_model.config.use_cache = False` (required for gradient checkpointing; MVP does not use KV cache)
7. If `config.backbone.gradient_checkpointing` is `True`: enable gradient checkpointing via `base_model.gradient_checkpointing_enable()`
8. Construct all modules and compose `VLAPolicy`

### D11: Prefer established libraries over custom implementations

**Choice:** For all core model components, prefer using established libraries directly over building from scratch:
- **LoRA/PEFT**: Use `peft` library (NOT custom LoRA layers)
- **VLM backbone**: Use `transformers.PaliGemmaForConditionalGeneration` directly (NOT reimplemented attention/FFN)
- **Vision encoder**: Use PaliGemma's built-in SigLIP via `get_image_features()` (NOT a separate SigLIP model)
- **Tokenization**: Use PaliGemma's tokenizer via `AutoTokenizer` (NOT custom tokenization)
- **Checkpointing**: Use `safetensors` for weights, `peft` for adapter save/load
- **Gradient checkpointing**: Use `model.gradient_checkpointing_enable()` from `transformers`

**Why:** Custom implementations of well-solved problems introduce bugs and maintenance burden. Our value-add is the modular composition (7-module architecture), not reimplementing attention or LoRA. Every module we build from scratch is a module we must debug and maintain.

## Risks / Trade-offs

**[PaliGemma 3B may be too large for single-GPU training without optimization]** → Use gradient checkpointing (`model.gradient_checkpointing_enable()`). Use `FreezeConfig` to freeze vision + projector and apply LoRA on LLM layers — this dramatically reduces trainable params. Full fine-tuning is possible but requires more VRAM.

**[Readout tokens may not extract enough information for precise manipulation]** → Octo demonstrates readout tokens work for diffusion heads. For MLP regression, the information bottleneck may be tighter. Mitigation: make `num_readout_tokens` configurable (default 64, can increase). If readout mode proves insufficient, joint-token mode is the post-MVP upgrade.

**[No Perceiver resampler means long sequences with multi-view cameras]** → MVP uses single camera at 224×224 = 256 vision tokens. With proprio (1 token) + language (~20 tokens) + readout (64 tokens) = ~341 total tokens. PaliGemma handles this easily. Multi-view is post-MVP.

**[MLP regression cannot capture multimodal action distributions]** → Known limitation. MLP averages modes, producing suboptimal actions when multiple valid solutions exist. Acceptable for MVP — flow matching (post-MVP) solves this.

**[No temporal ensembling means chunk boundaries may cause jerky execution]** → MVP action decoder simply returns the predicted chunk. Temporal ensembling (overlapping chunk averaging) is post-MVP. For initial testing, this is acceptable.

**[Tight coupling to PaliGemma via HuggingFace transformers + peft]** → The backbone wraps `PaliGemmaForConditionalGeneration` and LoRA uses `peft`. If either library's API changes, our wrappers break. Mitigation: pin transformers and peft versions in pyproject.toml and treat mask/forward assumptions as pinned to the validated PaliGemma source snapshot (commit `556312cd`) until an explicit revalidation pass is completed. The Protocol interface means we can swap to a different VLM without changing downstream modules.

**[`token_type_ids` and causal mask must be constructed correctly]** → PaliGemma uses `token_type_ids` to control bidirectional (image, type=0) vs causal (text, type=1) attention. The hybrid mask is constructed by `create_causal_mask_mapping(..., is_training=self.training)` which uses `token_type_ids` to un-mask image prefix tokens. In the HF source (commit `556312cd`), `is_training` only controls a validation check (requiring `token_type_ids` during training). The actual mask behavior is driven by `is_first_iteration` (inferred from `past_key_values is None`). Since MVP does not use KV cache (`past_key_values=None`), the hybrid mask works correctly in both `train()` and `eval()` modes. `position_ids` are NOT passed — PaliGemma computes them internally. Getting `token_type_ids` polarity wrong produces silent attention bugs. Mitigation: unit test that verifies attention mask shape and values for a known input.

**[Language tokens must come from PaliGemma's tokenizer]** → The backbone owns text embedding lookup. Language instructions are tokenized via PaliGemma's tokenizer, embedded via `model.get_input_embeddings()(token_ids)`, then concatenated with other modality embeddings. Mixing tokenizers or embedding tables would break the pretrained representations.

**[LoRA + freeze interaction complexity]** → When `peft.get_peft_model()` is applied, it freezes ALL base parameters in the wrapped model (not just targeted modules) and makes only LoRA adapters trainable. `freeze_modules` is still useful for the no-LoRA case (full fine-tuning with selective freeze). When LoRA IS enabled, `freeze_modules` has no additional effect since peft already freezes everything. Non-VLM modules (action head, proprio encoder, merger) are outside the PeftModel wrapper and remain trainable. Mitigation: store base model reference before peft wrapping; call `enable_input_require_grads()` for gradient flow through frozen base to LoRA adapters; set `use_cache=False` for gradient checkpointing compatibility.
