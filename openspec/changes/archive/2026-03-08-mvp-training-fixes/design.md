## Architecture Decision: YAML Config Parsing Fix

### Problem
`_pop_config_flag()` manually destructures YAML into `TrainingConfig` fields only, silently dropping `policy:`. The root cause: it was written when `train.yaml` only contained training-layer settings, before `TrainConfig` composed `training:` + `policy:`.

### Approach: Recursive dataclass deserialization
Instead of manually mapping YAML keys → constructor kwargs, use a generic helper that recursively walks dataclass fields and constructs nested sub-configs. This handles `TrainConfig.policy.action_head.action_dim` without per-field mapping code.

**Why not use tyro's YAML support directly?** tyro has experimental `from_yaml`/`to_yaml` but it's marked unstable. The current `_pop_config_flag()` + `tyro.cli(default=...)` pattern is actually correct in design — the YAML provides defaults, tyro applies CLI overrides. The bug is only in `_pop_config_flag()` not constructing the full `TrainConfig`.

### Backward compatibility
The current `train.yaml` is a **flat format** (keys are `TrainingConfig` fields directly at the top level). A new format should support a **nested format** (`training:` / `policy:` top-level keys). The fix must handle both:

```yaml
# NEW nested format (preferred)
training:
  dataset:
    repo_id: "lerobot/aloha_sim"
  num_steps: 10
policy:
  action_head:
    action_dim: 14

# OLD flat format (backward compatible — treated as training-only)
dataset:
  repo_id: "lerobot/aloha_sim"
num_steps: 10
```

Detection: if the top-level YAML contains `training:` or `policy:` keys, treat as nested `TrainConfig`; otherwise treat as flat `TrainingConfig` with warning.

---

## Architecture Decision: Masked L1 Loss

### Problem
`MLPRegressionHead.compute_loss()` computes L1 over all timesteps including `action_is_pad=True` garbage.

### Approach
Standard VLA practice (confirmed in OpenVLA-OFT, LeRobot pi0): apply boolean mask before loss reduction.

```python
# If mask is provided and has any padded timesteps:
if mask is not None and mask.any():
    valid = ~mask[:, :chunk_len]     # True = valid action
    per_elem_loss = F.l1_loss(predicted, target, reduction="none")
    masked_loss = (per_elem_loss * valid.unsqueeze(-1)).sum() / (valid.sum() * action_dim)
```

Key design choice: the mask is **optional** — if `action_mask is None` or all-False, the loss is unchanged (full L1). This preserves backward compatibility and avoids breaking tests that don't produce masks.

Edge case: if every compared timestep is masked, the implementation must return a finite zero loss instead of dividing by zero.

---

## Architecture Decision: Dimension Validation

### Problem
Silent shape slicing means misconfiguration trains on wrong data without errors.

### Approach: Validation at two points

1. **At train startup** (`scripts/train.py`): After creating the dataloader but before training, fetch one batch and assert shapes match config:
   ```python
   assert batch.actions.shape[1] == cfg.policy.action_head.chunk_len
   assert batch.actions.shape[-1] == cfg.policy.action_head.action_dim
   assert batch.observations.proprio.shape[-1] == cfg.policy.proprio_encoder.proprio_dim
   assert cfg.training.dataset.action_chunk_size == cfg.policy.action_head.chunk_len
   ```

2. **In `compute_loss()`**: Change target slicing from permissive `[:chunk_len, :action_dim]` to exact match assertion for both `chunk_len` and `action_dim`, then index exactly (or just use `target = batch.actions` when shapes are validated upstream).

The startup validation is preferred because it gives a clear error message with config values, rather than a cryptic tensor shape error mid-training.

---

## Architecture Decision: Smoke Config

### Need
A minimal config that:
- Uses the correct dims for a real dataset
- Uses LoRA + freeze to minimize VRAM (< 16GB)
- Runs just 10 steps for smoke-testing
- Sets `drop_last: true` and seeds everything

Target: `configs/train_smoke.yaml`

---

## Files Modified

| File                            | Change                                                                                    |
| ------------------------------- | ----------------------------------------------------------------------------------------- |
| `scripts/train.py`              | Rewrite `_pop_config_flag()` for full `TrainConfig`; add dim validation; add `set_seed()` |
| `src/yavla/models/heads/mlp.py` | `compute_loss()` respects `action_mask`                                                   |
| `configs/train.yaml`            | Update to nested `TrainConfig` format with `policy:` section                              |
| `configs/train_smoke.yaml`      | New — minimal smoke test config                                                           |
