## Why

YAVLA's MVP model and training pipeline are architecturally complete (7-module policy, 3 data backends, Accelerate trainer), but **four critical bugs and gaps** prevent a successful first training run. These were identified by two independent code reviews and verified line-by-line:

1. **YAML config does not load `policy:` settings**: `_pop_config_flag()` in `scripts/train.py` only hydrates `TrainingConfig` and silently drops any `policy:` YAML block. This means `--config` cannot carry backbone choice, embodiment dims, LoRA/freeze config, or any other policy settings.

2. **`action_mask` ignored in loss**: `MLPRegressionHead.compute_loss()` calculates L1 over *all* action timesteps including garbage-padded episode tails. LeRobot's `action_is_pad` is correctly propagated through `TrainingCollate` as `batch.action_mask`, but the loss never reads it.

3. **Dimension mismatches silently produce garbage**: `action_dim` and `proprio_dim` default to 7, but `lerobot/aloha_sim` (the default dataset) has 14-dim actions. `compute_loss()` silently slices `batch.actions[:, :chunk_len, :action_dim]` instead of asserting exact match. No cross-validation between `DataConfig.action_chunk_size` and `MLPHeadConfig.chunk_len`.

4. **No workable smoke config**: The shipped `train.yaml` has no `policy:` section (which wouldn't work even if present, per bug #1), no LoRA/freeze defaults (meaning full 3B unfrozen), batch_size=32, and num_steps=100K — not a viable first-run setup.

Reference frameworks consulted:
- **OpenVLA-OFT**: Uses draccus dataclass configs; explicit `action_dim` matching via assert; `action_is_pad` mask in loss via `_process_action_masks`
- **LeRobot**: `action_is_pad` boolean mask field; loss masking in policy-specific code (e.g., pi0 model)
- **π0 / openpi**: Full combined config (model + training); explicit shape validation at init time

## What Changes

- **Fix `_pop_config_flag()`** in `scripts/train.py`: Parse full `TrainConfig` (both `training:` and `policy:` blocks) from YAML. Support both flat (current) and nested YAML formats for backward compatibility.
- **Add `action_mask` to loss** in `MLPRegressionHead.compute_loss()`: Mask out padded actions so only valid timesteps contribute to L1.
- **Add dimension validation**: Fail-fast assertions in `train.py` or `build_policy()` to ensure `action_dim`, `proprio_dim`, and `chunk_len` match between model config and data config.
- **Ship `configs/train_smoke.yaml`**: Complete config with policy block, LoRA/freeze defaults, small batch_size, short num_steps — a safe first-run target.
- **Add seed setup**: Call `accelerate.utils.set_seed()` in `train.py` for reproducibility.

## Capabilities

### Modified Capabilities

- `training-entry`: `_pop_config_flag()` now loads full `TrainConfig` including `policy:` from YAML
- `mvp-action-head`: `MLPRegressionHead.compute_loss()` now respects `action_mask` for masked L1 loss
- `training-config`: dimension cross-validation between model and data config at train start

### New Capabilities

- `training-smoke`: `configs/train_smoke.yaml` providing a safe, low-resource first-run configuration

## Impact

- **Modified**: `scripts/train.py`, `src/yavla/models/heads/mlp.py`, `configs/train.yaml`
- **New**: `configs/train_smoke.yaml`
- **No new dependencies**
- **Risk**: Low — all changes are additive validations and bug fixes; backward-compatible. The YAML parsing change supports both old flat format and new nested format.
