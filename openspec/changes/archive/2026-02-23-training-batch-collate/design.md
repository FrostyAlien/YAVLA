## Context

`VLAPolicy.forward()` expects `TrainingBatch` (containing nested `ObservationBatch`), but all three dataset backends produce raw `dict[str, Tensor]` with LeRobot-convention keys like `observation.state`, `observation.images.<cam>`, `action`, `task`. The `create_dataloader()` factory already accepts a `collate_fn` parameter (currently unused by training). PyTorch's `default_collate` handles tensor stacking but not restructuring into dataclasses.

The existing `RepackTransform` normalizes dataset-specific key names into canonical keys before collation, so the collate function only needs to understand one convention.

## Goals / Non-Goals

**Goals:**
- Bridge the dict→`TrainingBatch` gap so `Trainer.run()` works end-to-end
- Keep the collate function simple — convention-based, no config needed for standard LeRobot datasets
- Maintain the dict as the interface contract between data layer and training layer (data layer stays generic)

**Non-Goals:**
- Handling non-dict data sources (that's a dataset backend concern)
- Adding new transforms or key remapping (already handled by `RepackTransform`)
- Modifying `TrainingBatch` or `ObservationBatch` dataclass definitions
- Supporting multi-step / trajectory-level batching (future work for sequence models)

## Decisions

### 1. Collate function in `training/data.py`, not a Transform

**Decision**: Implement as a `collate_fn` passed to `DataLoader`, not as a per-sample transform.

**Rationale**: Transforms operate per-sample (dict→dict). Collation operates on `list[dict]` → single structured object. The restructuring into `TrainingBatch` is inherently a batch-level operation (collecting language strings into a list, grouping image tensors by camera name). Putting it in the collate also means it runs in DataLoader worker processes for free.

**Alternative considered**: Per-sample transform that converts dict→`TrainingBatch` with B=1, then a custom collate to merge. Rejected — adds complexity, breaks the clean dict interface between data and training layers.

### 2. Convention-based key detection with hardcoded defaults

**Decision**: Use hardcoded LeRobot key conventions:
- `observation.images.*` → `ObservationBatch.images` (dict keyed by camera name)
- `observation.state` → `ObservationBatch.proprio`
- `task` → `ObservationBatch.language`
- `action` → `TrainingBatch.actions`
- `action_is_pad` → `TrainingBatch.action_mask` (if present)

**Rationale**: All LeRobot datasets follow this convention. Non-standard datasets use `RepackTransform` to normalize keys before collation. Adding a config layer for key mapping would duplicate what `RepackTransform` already does.

**Alternative considered**: Configurable key mapping in the collate. Rejected — `RepackTransform` + `DataConfig.repack_keys` already solves this at the right layer.

### 3. Manual per-key collation (no `default_collate`)

**Decision**: Iterate the union of keys across all samples and collate each key individually:
- Tensor values → `torch.stack`
- String values → collect into `list[str]`
- Unmapped / unknown keys → silently ignored

**Rationale**: `default_collate` crashes with `KeyError` when optional keys (`task`, `action_is_pad`) are present in some samples but absent in others. Manual per-key collation handles the union of keys naturally — optional keys are only stacked from the samples that contain them, and missing keys simply produce `None` in the output. The per-key logic is trivial (stack or collect) so there is no real benefit to delegating to `default_collate`.

**Alternative considered**: `default_collate` with pre-pass to inject missing keys. Rejected — adds complexity for no benefit; manual stacking is equally simple and avoids the fragility.

### 4. `dt_hz` and `chunk_len` as explicit parameters, not config fields

**Decision**: `create_training_dataloader(config, *, dt_hz: float, chunk_len: int)` receives `dt_hz` and `chunk_len` as explicit keyword arguments. The caller (e.g. `scripts/train.py`) sources them from `PolicyConfig.dt_hz` and the action head's `chunk_len`. No new config fields are added to `TrainingConfig` or `DataConfig`.

**Rationale**: These are training-time constants owned by the policy, not the data layer. Making them explicit parameters keeps the data layer decoupled from policy config structure and makes the dependency obvious at the call site.

### 5. Graceful handling of missing optional fields

**Decision**: `language` defaults to `None` if `task` key is absent. `action_mask` defaults to `None` if `action_is_pad` is absent. `timestamps` and `masks` default to `None`.

**Rationale**: Not all datasets have language instructions or padded action chunks. The policy pipeline already handles `None` language (falls back to empty string in `encode_observations`).

### 6. Action shape validation — reject 2D actions

**Decision**: After stacking, if `actions` has shape `[B, action_dim]` (2D), raise `ValueError` directing the user to set `action_chunk_size` in their data config. The pipeline requires 3D actions `[B, chunk_len, action_dim]`.

**Rationale**: Without `action_chunk_size`, single-step datasets produce 1D action vectors per sample, which stack to `[B, action_dim]`. The policy expects `[B, chunk_len, action_dim]`. A clear error at collation time is far easier to debug than a shape mismatch deep inside the action head.

### 7. Proprio presence validation — reject missing `observation.state`

**Decision**: If no sample contains `observation.state`, raise `ValueError`. `ObservationBatch.proprio` is non-optional, so the collate must not silently produce an invalid batch.

**Rationale**: Unlike `language` and `action_mask` which are `Optional`, `proprio` is required by the dataclass contract. Failing early with a clear message prevents confusing downstream errors.

## Risks / Trade-offs

**[Risk] Image key detection relies on prefix convention** → If a dataset has image keys not starting with `observation.images.`, they won't be detected. Mitigation: `RepackTransform` can rename keys before collation. Document the convention.

**[Risk] `torch.stack` fails on variable-length sequences** → If samples have different action chunk lengths or image sizes, stacking fails. Mitigation: This is already enforced by the dataset layer (fixed chunk size, fixed image resolution). Not a new risk.

**[Trade-off] No config for key mapping** → Simpler code, but requires `RepackTransform` for non-standard datasets. Acceptable because `RepackTransform` already exists and is the documented way to handle key differences.
