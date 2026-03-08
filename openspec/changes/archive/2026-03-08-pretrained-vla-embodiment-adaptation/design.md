## Context

YAVLA's current MVP shape contract is exact-width: dataset action and proprio tensors are expected to match the model's configured action and proprio widths exactly. That is the correct rule for the first complete training test, but it is insufficient for a pretrained multi-embodiment VLA where one checkpoint must serve robots whose active action and proprio spaces are narrower than the base model width.

The recent MVP training fixes deliberately tighten fail-fast validation around exact dimensions. This new change does not replace that work. Instead, it adds a second, explicit mode for pretrained-VLA operation so YAVLA can support smaller embodiments without relying on unsafe defaults or silent slicing.

The external design space is split:
- Dataset-first stacks such as LeRobot ACT and Diffusion derive effective dimensions from dataset feature specs and assume embodiment-specific training runs.
- Pretrained multi-embodiment stacks such as SmolVLA, openpi, and Octo use model-side maximum widths plus padding and masking.

YAVLA should keep the strict MVP exact-dimension path and add a separate pretrained-VLA adaptation path modeled on the second pattern.

## Goals / Non-Goals

**Goals:**
- Preserve exact-dimension MVP behavior as the default and safest training path.
- Add an explicit pretrained-VLA mode that supports `max_action_dim` and `max_proprio_dim` larger than the active embodiment dimensions.
- Keep dataset tensors embodiment-exact and perform adaptation only inside model, training, and checkpoint flows.
- Ensure loss, inference, and checkpoint loading all agree on active-versus-max dimension semantics.
- Update documentation so users can choose between exact-dimension and pretrained-VLA modes without guessing.

**Non-Goals:**
- Automatically infer embodiment semantics from dataset shape alone.
- Learn arbitrary projections between incompatible action-space semantics.
- Generalize every future policy family in this change; the first implementation may target the current readout + MLP policy path while keeping the config and checkpoint contract reusable.
- Relax the MVP fail-fast rules for exact-dimension training.

## Decisions

### 1. Add an explicit embodiment adaptation config layer

Decision:
- Introduce a dedicated embodiment adaptation config in `PolicyConfig` as the single source of truth for active and maximum dimensions.
- Represent two explicit modes:
  - `exact`: active and maximum dimensions are equal.
  - `max_padded`: active dimensions may be smaller than model maximum dimensions.
- Legacy width overrides on module sub-configs are not supported as a public configuration path.

Rationale:
- Today, action and proprio widths live in separate module configs, which makes mismatch bugs easy to introduce.
- A shared config layer lets `build_policy()`, checkpoint save/load, and docs all speak the same contract.

Alternatives considered:
- Keep dimensions only on module-specific configs: rejected because it duplicates cross-cutting state and preserves mismatch risk.
- Infer mode from whether dimensions differ: rejected because checkpoint loading and docs become ambiguous.

### 2. Keep dataset boundaries exact; adapt inside the model path

Decision:
- `TrainingBatch` and `ObservationBatch` remain embodiment-exact at the data boundary.
- Pretrained-VLA adaptation zero-pads proprio tensors only after batches enter the policy path.
- Action targets remain exact-width at the batch boundary and receive an internal active-dimension mask before loss computation.

Rationale:
- This keeps the dataset layer neutral and avoids teaching every backend, transform, and collate path about model-specific maximum widths.
- It matches the user's mental model: datasets describe the robot, while the policy describes the pretrained model width.

Alternatives considered:
- Pad data inside the dataset layer: rejected because it leaks model-specific behavior into a shared data abstraction and makes dataset debugging harder.
- Require users to pre-pad datasets offline: rejected because it is brittle and easy to get wrong.

### 3. Use zero-padding for inputs and combined masks for losses

Decision:
- Pad proprio with zeros up to `max_proprio_dim`.
- Construct an active action-dimension mask from embodiment config and combine it with any timestep padding mask already present on the batch.
- Compute loss only over the combined valid elements.
- If the combined mask has zero valid elements, return a finite zero loss.

Rationale:
- Zero-padding is deterministic and simple for linear input projections.
- Combined masking makes embodiment adaptation compose cleanly with episode-boundary padding.
- The zero-valid-element rule prevents `NaN` failures on fully padded chunks.

Alternatives considered:
- Learn projection adapters between active and max dimensions: rejected for this change because it adds semantic assumptions and training instability.
- Ignore inactive dimensions without an explicit mask: rejected because it silently trains on meaningless targets.

### 4. Slice predictions at the policy/decode boundary

Decision:
- Internal heads may emit `max_action_dim` predictions.
- `VLAPolicy.predict()` and `ActionDecoder` expose only the active embodiment slice.
- The active slice is defined by embodiment metadata, not by ad hoc tensor slicing at call sites.

Rationale:
- This keeps downstream consumers robot-facing rather than model-facing.
- Centralizing slicing avoids repeated, inconsistent shape handling in evaluation and deployment code.

Alternatives considered:
- Return max-width actions everywhere and require every caller to slice: rejected because it leaks pretrained-model internals into robot control code.

### 5. Bump checkpoint semantics to store both active and max dimensions

Decision:
- Extend checkpoint metadata to store:
  - embodiment mode
  - active action/proprio dimensions
  - max action/proprio dimensions
  - active action/proprio names or equivalent embodiment descriptors
- `from_pretrained(strict=True)` validates exact max-width compatibility.
- `from_pretrained(strict=False)` may rebind to a smaller embodiment only when the target config is explicit and does not exceed checkpoint max widths.
- Older checkpoints or configs without embodiment-aware metadata are rejected as unsupported.

Rationale:
- Checkpoints need to say both what the model was built to support and what embodiment they were last bound to.
- Rejecting older formats keeps the contract explicit and avoids carrying migration logic in the runtime path.

Alternatives considered:
- Reuse only `embodiment.json` with active dimensions: rejected because it cannot describe a wider pretrained model that is temporarily bound to a smaller robot.
- Keep backward-compatible migration from embodiment-less checkpoints: rejected because it preserves a second implicit config contract.
- Allow non-strict rebinding without explicit target metadata: rejected because semantic mismatches become impossible to audit.

### 6. Document two supported training stories

Decision:
- Update `docs/training-guide.md` and dataset-layer docs to describe two supported paths:
  - exact-dimension MVP training
  - pretrained-VLA embodiment adaptation
- Include explicit YAML examples and checkpoint-loading examples for the pretrained path.
- Document that flat train YAML and embodiment-less older checkpoints are unsupported.

Rationale:
- The dimension contract is subtle enough that users will otherwise guess.
- Documentation is part of the safety mechanism here, not an afterthought.

Alternatives considered:
- Keep the behavior discoverable only from config fields or code: rejected because the failure modes are expensive and confusing.

## Risks / Trade-offs

- [Cross-cutting config change] -> Centralize embodiment dimensions in one config object and derive module widths from it to avoid duplicated state.
- [Breaking checkpoint/config cleanup] -> Fail fast on embodiment-less configs/checkpoints and document the new required format clearly.
- [Semantic mismatch between robots] -> Require explicit embodiment metadata and keep non-strict loading opt-in.
- [Policy-family scope creep] -> Limit the first implementation to the current MLP/readout path while keeping the public config and checkpoint contract reusable.
- [More complex loss masking] -> Add focused unit tests for mixed timestep padding and inactive-dimension masking, including the all-masked case.

## Adoption Plan

1. Add the embodiment adaptation config and wire `exact` mode as the explicit default.
2. Update policy build, loss, prediction, and checkpoint save/load paths to respect active-versus-max dimensions.
3. Reject flat training YAML and embodiment-less checkpoint/config formats with clear errors.
4. Update training and dataset documentation with exact-mode versus pretrained-VLA guidance and the supported checkpoint/config format.
5. Add tests for config validation, padding/masking, inference slicing, and current-format checkpoint validation/rebinding.

## Open Questions

- The initial implementation should target the current readout + MLP policy path. If future policy families need different embodiment adaptation mechanics, they should extend the shared embodiment contract rather than invent new checkpoint metadata.
