## Context

YAVLA's 7-module pipeline (`VLAPolicy`) is VLM-agnostic in theory — `BackboneBase`, `VisionEncoderBase`, etc. define clean ABCs. In practice, three coupling points hardcode PaliGemma:

1. **`build_policy()`** (policy.py:254-341) — inline PaliGemma loading via `AutoModelForVision2Seq`, LoRA wrapping, freezing, gradient checkpointing, and manual wiring of the raw HF model to vision encoder / backbone / merger
2. **`VLAPolicy.encode_observations()`** (policy.py:69-89) — calls `self.backbone.tokenizer(...)` and `self.backbone.base_model.get_input_embeddings()` directly, bypassing the backbone abstraction
3. **`PaliGemmaVisionEncoder`** (vision.py:37-46) — calls `self._base_model.get_image_features()`, a PaliGemma-specific method

Adding SmolVLM2 or Qwen3-VL would require modifying all three sites. The goal is to refactor once so new VLMs are purely additive.

## Goals / Non-Goals

**Goals:**
- Make adding a new VLM backbone a matter of implementing classes + registering them (no changes to `VLAPolicy`, merger, action head, or trainer)
- Preserve exact behavior for existing PaliGemma users (same config, same weights, same results)
- Enable a training integration test with a lightweight model (SmolVLM2-256M)

**Non-Goals:**
- Implementing SmolVLM2 or Qwen3-VL backends in this change (backlog — may need separate research)
- Changing the 7-module pipeline structure or the `VLAPolicy` forward/predict flow
- Supporting non-VLM backbones (e.g., pure transformer without vision tower)
- Multi-model ensemble or model switching at runtime

## Decisions

### D1: Add `embed_language()` to `BackboneBase`

**Choice**: Add `embed_language(texts: list[str]) -> tuple[Tensor, Tensor]` as an abstract method on `BackboneBase`, returning `(embeddings [B, T, D], attention_mask [B, T])`.

**Why**: `VLAPolicy.encode_observations()` currently reaches into `backbone.tokenizer` and `backbone.base_model.get_input_embeddings()` — both PaliGemma-specific. Moving tokenization + embedding into the backbone makes the policy VLM-agnostic. Each backbone knows its own tokenizer and embedding layer.

**Alternative considered**: Keep `tokenizer` property on `BackboneBase` and add a generic `embed_tokens(input_ids)` method. Rejected because different VLMs tokenize differently (Qwen3-VL uses chat templates, SmolVLM2 uses special image tokens in the tokenizer) — the backbone should own the full text→embedding path.

**Alternative considered**: A separate `LanguageEncoder` module (8th module). Rejected as over-engineering — language embedding is tightly coupled to the backbone's vocabulary and embedding weights. It's not independently swappable.

### D2: Backbone builder protocol via registry

**Choice**: Each VLM type registers a builder that returns a `(VisionEncoderBase, BackboneBase)` pair. `build_policy()` calls `vlm_registry.build(config.backbone)` to get both components, then builds the remaining modules (proprio encoder, merger, action head, decoder) generically.

**Why**: Vision encoder and backbone are co-dependent — they share the underlying HF model (PaliGemma's vision encoder calls `base_model.get_image_features()`; SmolVLM2's would use pixel shuffle from the same model). Bundling them in a single builder avoids passing raw HF model objects across module boundaries.

**Shape**: A new `vlm_registry` (separate from the existing `backbone_registry`) maps `backbone.type` → builder function:
```python
def build_vlm(config: BackboneConfig) -> tuple[VisionEncoderBase, BackboneBase]:
    ...
```

The existing `backbone_registry` is unused in `build_policy()` today (it's registered but `build_policy()` constructs everything inline). The new `vlm_registry` replaces that inline code.

**Alternative considered**: Have `build_policy()` switch on `config.backbone.type` with if/elif. Rejected — violates open/closed principle, requires modifying `build_policy()` for each new VLM.

### D3: Move PaliGemma code into `backbones/paligemma.py`

**Choice**: Extract all PaliGemma-specific code from `build_policy()` into a new `src/yavla/models/backbones/paligemma.py` module containing:
- `PaliGemmaBackbone(BackboneBase)` — renamed from `VLMBackbone`, implements `embed_language()`
- `PaliGemmaVisionEncoder` — moved from `encoders/vision.py`
- `build_paligemma_vlm()` — the builder function, registered as `vlm_registry.register("paligemma")`

**Why**: Groups all PaliGemma-specific code in one place. The current `backbone.py` has a generic name (`VLMBackbone`) for a PaliGemma-specific class.

### D4: `BackboneConfig` uses a union/discriminated type field

**Choice**: Keep the existing `BackboneConfig` with `type: str` as the discriminator. VLM-specific fields (like `vlm_name`) stay in `BackboneConfig` for now. Future VLM types that need different config fields can subclass or use a config union.

**Why**: Minimal change. PaliGemma's `vlm_name` field is harmless when unused by other backends. Avoids premature config hierarchy. If SmolVLM2 needs different fields, we add a `SmolVLMBackboneConfig` at that point.

### D5: `BackboneBase.tokenizer` property becomes optional

**Choice**: Keep `tokenizer` as a property on `BackboneBase` but remove it from the abstract contract (make it concrete with a default that raises `NotImplementedError`). The policy no longer calls it directly — `embed_language()` is the interface.

**Why**: Some backbones may not expose a tokenizer (or may use a processor instead). The property is still useful for serialization (`save_pretrained` / `from_pretrained`) but shouldn't be required by the pipeline.

### D6: Integration test uses PaliGemma initially, SmolVLM2 as follow-up

**Choice**: The first integration test uses `google/paligemma-3b-pt-224` (the existing backbone) with `@pytest.mark.integration`. SmolVLM2-256M integration test is added when that backend is implemented.

**Why**: The refactor itself doesn't add SmolVLM2 support. Testing the refactored pipeline with PaliGemma proves the abstraction works without introducing a new VLM simultaneously. The test exercises: synthetic data → `TrainingCollate` → `build_policy()` → `policy.forward()` → loss → backward → optimizer step → verify parameters changed.

## Risks / Trade-offs

**[Risk] Serialization compatibility** — `save_pretrained()` / `from_pretrained()` currently reference `VLMBackbone` and PaliGemma-specific paths. Renaming to `PaliGemmaBackbone` could break loading existing checkpoints.
→ Mitigation: Keep the state dict key prefix as `backbone.` (unchanged). The class name doesn't affect `state_dict()` keys. Add a `from_pretrained()` path that reads `config.backbone.type` to dispatch to the right builder.

**[Risk] `base_model` / `model` properties on `BackboneBase`** — currently used for LoRA save/load and `get_input_embeddings()`. After refactor, `get_input_embeddings()` moves inside `embed_language()`, but LoRA save/load still needs `model` / `base_model`.
→ Mitigation: Keep `base_model` and `model` as optional properties on `BackboneBase` (current default raises `NotImplementedError`). PaliGemma implements them; other backends implement them if they use LoRA.

**[Risk] Existing test breakage** — tests that mock `BackboneBase` will need to add `embed_language()` to their stubs.
→ Mitigation: Small, mechanical change. The `_StubBackbone` in `test_trainer.py` doesn't use language encoding, so a no-op implementation suffices.

**[Trade-off] Vision encoder bundled with backbone** — the builder returns `(VisionEncoder, Backbone)` as a pair, meaning you can't mix PaliGemma's vision encoder with SmolVLM2's backbone. This is intentional — these components share weights and aren't independently swappable in practice.

## Open Questions

- **SmolVLM2 `token_type_ids`**: PaliGemma uses `token_type_ids` to distinguish image vs text tokens. SmolVLM2 (Idefics3-based) may not use this. The `BackboneBase.forward()` signature currently requires it. Should it become optional (`token_type_ids: Tensor | None`)? Needs research when implementing SmolVLM2.
- **Qwen3-VL dynamic resolution**: Qwen3-VL processes images at variable resolution with dynamic token counts. This may require changes to `VisionEncoderBase.encode_images()` return shape or the merger. Needs research.
