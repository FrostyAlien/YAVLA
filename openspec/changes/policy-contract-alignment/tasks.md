## 1. Stabilize Current Tests (Baseline Contract Alignment)

- [ ] 1.1 Fix the current failing unit test by removing unsupported `clip_unnormalized` usage in `tests/models/test_policy.py` (align with `ActionSpaceSpec` contract).
- [ ] 1.2 Add/adjust decoder tests to explicitly assert the “no implicit clamp” behavior for out-of-range normalized actions (e.g., `2.0 → 15.0` under `[0, 10]` limits).
- [ ] 1.3 Run the full unit test suite (`tests/`) in the dev env and confirm a clean baseline for subsequent work (note: on macOS, `pixi run` may crash; use `.pixi/envs/dev/bin/pytest` as a fallback).

## 2. Implement Policy-Declared Action Normalization (Config + Validation)

- [ ] 2.1 Add an explicit action normalization config to the policy config tree (default `bounds`, optional `z-score`) and ensure it roundtrips through `config.json`.
- [ ] 2.2 Add strict validation for `bounds` mode: `ActionSpaceSpec.limits` MUST be present and shaped `[action_dim, 2]`; fail fast with a clear error if missing/mismatched.
- [ ] 2.3 Add strict validation for `z-score` mode: required stats (`mean`, `std`) MUST be present in checkpoint assets/metadata; fail fast if missing.
- [ ] 2.4 Define and implement the `std == 0` behavior in `z-score` mode (no division-by-zero; stable mapping).
- [ ] 2.5 Ensure the decoder continues to avoid implicit clamping in the default path (latency/perf constraint) while still enabling optional downstream clipping at env/adapter level.

## 3. Extend Dataset Transforms for `[-1, 1]` Bounds Mode (Delta Spec: `data-transforms`)

- [ ] 3.1 Update `NormalizeTransform` to support `mode="bounds"` mapping to `[-1, 1]` using `min/max` stats, including a zero-range guard.
- [ ] 3.2 Update `UnnormalizeTransform` to support `mode="bounds"` (inverse mapping), including a zero-range guard.
- [ ] 3.3 Update `DataConfig.normalize_mode` typing to include `"bounds"` and ensure dataset factory wiring remains backward compatible.
- [ ] 3.4 Add unit tests in `tests/data/test_transforms.py` that cover `bounds` normalize/unnormalize roundtrip and the `max == min` edge case.

## 4. Wire Action Normalization into the Training Data Path (No Silent Contract Drift)

- [ ] 4.1 Decide the source of truth for bounds normalization of *actions* during training:
  - (A) normalize using `ActionSpaceSpec.limits` (physical contract), or
  - (B) normalize using dataset stats min/max (distribution contract),
  and document the decision in code/docstrings to prevent ambiguous usage.
- [ ] 4.2 Implement action-target normalization in the dataloader pipeline such that `TrainingBatch.actions` matches the policy-declared normalization mode (default: `bounds` → `[-1, 1]`).
- [ ] 4.3 Add a guardrail that makes “wrong normalization for the active policy” a fast, obvious error (prefer init-time / once-per-batch checks; avoid per-token overhead).

## 5. Vision Preprocessing Contract: Data Pipeline Produces `pixel_values`

- [ ] 5.1 Add a data transform that produces HuggingFace-compatible `pixel_values` using the processor for the configured VLM (e.g., `AutoProcessor.from_pretrained(vlm_name)`), and stores it in the sample under the configured camera key(s).
- [ ] 5.2 Implement double-rescaling prevention: if raw images arrive as float `[0, 1]`, configure the processor with `do_rescale=False`; if they arrive in `[0, 255]`, ensure the configuration is consistent with processor expectations.
- [ ] 5.3 Ensure `pixel_values` are channels-first `[B, 3, H, W]` and have a dtype compatible with the model forward path; keep processing in dataloader workers (CPU) for performance.
- [ ] 5.4 Add unit tests that validate `pixel_values` shape/dtype and that MVP multi-camera inputs are rejected with a clear error message.

## 6. Model Hot-Path Correctness + AMP Safety

- [ ] 6.1 Fix language batching semantics in `VLAPolicy.encode_observations`: broadcast a single string to batch size `B`, and validate list length when a list is provided.
- [ ] 6.2 Make `ConcatMerger` AMP-safe by creating readout tokens and masks on the correct device and with dtypes that do not upcast the mixed-precision path (readout tokens should follow `vision_tokens.dtype`).
- [ ] 6.3 Update `MLPRegressionHead.compute_loss` to respect `TrainingBatch.action_mask` when provided (mask padded timesteps so gradients are correct).
- [ ] 6.4 Reduce avoidable backbone overhead by eliminating always-on `output_hidden_states=True` if the final hidden state can be obtained without material extra memory/time (keep behavior correct for PaliGemma’s forward contract).
- [ ] 6.5 Add unit tests for language broadcast, masked loss behavior, and AMP dtype discipline (at minimum: no unintended dtype promotion of `inputs_embeds` when vision tokens are fp16/bf16).

## 7. Registry-Driven `build_policy()` Composition (Restore Modularity)

- [ ] 7.1 Refactor `build_policy()` to construct all module families via their registries based on config `type` fields (vision encoder, proprio encoder, merger, backbone, head, decoder).
- [ ] 7.2 Add build-time errors for unknown registry types that list available options (improves debuggability for config-driven workflows).
- [ ] 7.3 Ensure `validate_integration(backbone, head)` runs before returning the composed policy, and that it fails early without leaving partially constructed modules around.
- [ ] 7.4 Ensure exactly one VLM base model instance is loaded and shared across backbone forward, vision features, and language embedding lookup (avoid duplicate module registration that bloats `state_dict` / breaks serialization).
- [ ] 7.5 Add a unit test for “unknown type” errors and for “single base model reference” invariants using a lightweight/dummy backbone (no model downloads).

## 8. Checkpointing: Self-Contained Default + Shared-Tensor-Safe Serialization

- [ ] 8.1 Redesign `save_pretrained/from_pretrained` to produce a self-contained checkpoint by default (no base-model download required on load).
- [ ] 8.2 Replace raw `save_file(state_dict)` for VLM weights with a shared-tensor-safe approach (e.g., `safetensors.torch.save_model/load_model` or HF `save_pretrained(safe_serialization=True)`), and update loading accordingly.
- [ ] 8.3 Update checkpoint metadata to persist action normalization mode, required stats identifiers, and the vision preprocessing contract so load behavior is unambiguous.
- [ ] 8.4 Expand `embodiment.json` to include complete `ActionSpaceSpec` and `ProprioSpec` fields plus control frequency metadata (`dt_hz`) and any required chunk metadata.
- [ ] 8.5 Implement an optional adapter-only export path for LoRA/PEFT workflows, without replacing the self-contained default; ensure loader behavior is explicit when only adapters are present.
- [ ] 8.6 Update `tests/models/test_policy_serialization.py` to match the new checkpoint format and remove any assumptions that depend on shared-tensor-unsafe key equality (`set(file_keys) == set(state_dict_keys)`).
- [ ] 8.7 Add a regression test that exercises shared-tensor behavior with a small synthetic `nn.Module` that ties weights, ensuring the chosen serialization path roundtrips correctly.

## 9. Documentation + Final Verification

- [ ] 9.1 Document the action normalization modes and the default `[-1, 1]` policy contract (including where normalization happens: dataset vs model vs env).
- [ ] 9.2 Document the `pixel_values` preprocessing contract and expected input ranges to avoid double-rescaling.
- [ ] 9.3 Re-run the full unit test suite and any configured lint/typecheck commands; confirm no regressions and no obvious performance regressions (e.g., removal of always-on hidden-states collection).
