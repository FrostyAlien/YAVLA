## 1. Core Collate Function

- [x] 1.1 Implement `TrainingCollate` callable class in `src/yavla/training/data.py` that accepts `dt_hz` and `chunk_len` as constructor params
- [x] 1.2 Implement manual per-key collation: iterate union of keys across samples, `torch.stack` tensors, collect strings into `list[str]`, silently ignore unmapped keys; restructure into `TrainingBatch` with nested `ObservationBatch`
- [x] 1.3 Implement image key detection via `observation.images.` prefix, extracting camera name suffix
- [x] 1.4 Handle optional fields: `task` → language (None if absent), `action_is_pad` → action_mask (None if absent)
- [x] 1.5 Validate action shape: raise `ValueError` if stacked actions are 2D `[B, action_dim]`, directing user to set `action_chunk_size`
- [x] 1.6 Validate proprio presence: raise `ValueError` if no sample contains `observation.state`

## 2. Wiring

- [x] 2.1 Update `create_training_dataloader(config, *, dt_hz: float, chunk_len: int)` to construct `TrainingCollate` with explicit `dt_hz` and `chunk_len` params and pass it as `collate_fn` to `create_dataloader()`
- [x] 2.2 Update call site in `scripts/train.py` to pass `dt_hz` from `PolicyConfig.dt_hz` and `chunk_len` from action head config

## 3. Tests

- [x] 3.1 Test standard collation: dict samples with images, proprio, language, actions → correct `TrainingBatch` shapes and types
- [x] 3.2 Test multiple cameras: two image keys → both present in `observations.images`
- [x] 3.3 Test missing optional fields: no `task` key → `language is None`; no `action_is_pad` → `action_mask is None`
- [x] 3.4 Test `dt_hz` and `chunk_len` passthrough from constructor to output
- [x] 3.5 Test extra/unknown keys (e.g. `timestamp`, `episode_index`) are silently ignored
- [x] 3.6 Test `create_training_dataloader()` yields `TrainingBatch` instances (with synthetic dataset)
- [x] 3.7 Test 2D actions `[B, action_dim]` raise `ValueError` mentioning `action_chunk_size`
- [x] 3.8 Test missing `observation.state` raises `ValueError`

## 4. Validation

- [x] 4.1 Run `pixi run -e dev lint` and `pixi run -e dev typecheck` — all pass
- [x] 4.2 Run full test suite `pixi run -e dev test` — no regressions
