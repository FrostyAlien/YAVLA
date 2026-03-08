## 1. Fix YAML Config Parsing (`scripts/train.py`)

- [x] 1.1 Rewrite `_pop_config_flag()` to construct full `TrainConfig` (both `training:` and `policy:`) from YAML
  - Detect nested vs flat format: if YAML has `training:` or `policy:` top-level keys → nested `TrainConfig`; otherwise → flat `TrainingConfig` only (backward compat, emit deprecation warning)
  - Use recursive dataclass construction for `PolicyConfig` sub-tree (handle `BackboneConfig`, `MLPHeadConfig`, `FreezeConfig`, `ProprioEncoderConfig`, `TokenMergerConfig`, `ActionSpaceSpec`, `ProprioSpec`)
  - Preserve existing `OptimizerConfig.betas` tuple conversion
  - Log effective merged config at INFO level for debugging

- [x] 1.2 Add `set_seed()` call at top of `main()` using `accelerate.utils.set_seed()`

## 2. Fix Action Mask in Loss (`src/yavla/models/heads/mlp.py`)

- [x] 2.1 Modify `MLPRegressionHead.compute_loss()` to use `batch.action_mask` when present
  - If `action_mask is None` or all-False → unchanged L1 (backward compat)
  - If mask has True values → compute per-element L1, zero out padded positions, divide by number of valid elements
  - If all compared timesteps are masked → return a finite zero-valued loss instead of `NaN`
  - Mask shape must exactly match the configured chunk length; do not silently truncate it

## 3. Add Dimension Validation (`scripts/train.py`)

- [x] 3.1 Add startup dimension validation in `main()` after dataloader creation:
  - Assert `cfg.training.dataset.action_chunk_size` equals `cfg.policy.action_head.chunk_len` (with clear error message)
  - Peek first batch from dataloader and assert `actions.shape[1] == cfg.policy.action_head.chunk_len`
  - Peek first batch from dataloader and assert `actions.shape[-1] == cfg.policy.action_head.action_dim`
  - Assert `proprio.shape[-1] == cfg.policy.proprio_encoder.proprio_dim`
  - If any assertion fails, `sys.exit()` with a message showing expected vs actual values and which config to fix

- [x] 3.2 Convert silent slicing in `compute_loss()` to assert + exact index:
  - Add `assert batch.actions.shape[1] == self._config.chunk_len` and `assert batch.actions.shape[2] == self._config.action_dim`
  - Remove permissive target slicing for both temporal and action dimensions; use exact indexing only after validation

## 4. Ship Smoke Config (`configs/train_smoke.yaml`)

- [x] 4.1 Create `configs/train_smoke.yaml` with:
  - `policy:` block with `action_dim: 14`, `proprio_dim: 14` (matching `aloha_sim`)
  - `freeze:` with `freeze_modules: ["vision_tower", "multi_modal_projector"]` and `lora_target_modules: ["q_proj", "v_proj"]`
  - `dataset:` with `batch_size: 2`, `action_chunk_size: 5`, `num_workers: 0`
  - `num_steps: 10`, `log_freq: 1`, `save_freq: 10`
  - `drop_last: true`
  - Comment headers explaining this is a smoke test config

- [x] 4.2 Update `configs/train.yaml` to nested format with `policy:` section and correct `aloha_sim` dims

## 5. Tests

- [x] 5.1 Add unit test for `_pop_config_flag()`: verify `policy:` block is loaded from YAML correctly (both nested and flat format)
- [x] 5.2 Add unit test for masked L1 loss: verify padded timesteps don't contribute to loss
- [x] 5.3 Add unit test for fully masked L1 loss: verify a fully padded chunk returns finite zero loss
- [x] 5.4 Add unit test for dimension validation: verify `sys.exit()` on chunk length, action-dimension, and proprio-dimension mismatch
