## Context

YAVLA’s training collation path (`TrainingCollate`) requires chunked action targets with shape `[B, chunk_len, action_dim]`. The dataset layer exposes chunking via `DataConfig.action_chunk_size`, which is currently implemented only by the `lazy` backend (`LazyLeRobotDataset`). The `default` backend (`LeRobotDataset`) currently rejects `action_chunk_size`, forcing users to switch backends even for small datasets and blocking “default-config” end-to-end training.

LeRobot’s `LeRobotDataset` already supports temporal queries via `delta_timestamps`: it converts requested time deltas (multiples of `1/fps`) to frame-index deltas, clamps queries to episode boundaries, and returns both the stacked values and a `{key}_is_pad` boolean mask. This mechanism can be reused to implement action chunking for the default backend without introducing a new indexing implementation.

Upstream LeRobot already demonstrates future-action chunking through this exact mechanism (e.g., configuring `delta_timestamps["action"] = [t / dataset.fps for t in range(K)]`). This change makes that upstream capability available through YAVLA’s higher-level `action_chunk_size` knob when using the default backend.

Current lazy-backend semantics for action chunking are:
- `action` is stacked over contiguous future frames starting at the current frame: indices `idx + step` for `step=0..(K-1)`
- `action_is_pad` marks positions that fall past episode end and clamps the queried index to the episode’s last frame (duplicating the final action)

The default-backend implementation MUST match these semantics so that model/training code does not need backend-specific branching.

## Goals / Non-Goals

**Goals:**
- Allow `DataConfig.action_chunk_size` with `backend="default"` (no `ValueError`).
- Implement default-backend action chunking by composing an `action` entry in `delta_timestamps` and delegating to LeRobot’s temporal query logic.
- Ensure `default` and `lazy` backends produce consistent `action` and `action_is_pad` shapes/semantics when action chunking is enabled.
- Add integration coverage that exercises action chunking on the default backend with a real LeRobot dataset.
- Keep streaming backend guardrails: `backend="streaming"` remains incompatible with temporal features.

**Non-Goals:**
- Implement temporal features for `streaming` backend.
- Change the training API or automatically infer `action_chunk_size` from the action head’s `chunk_len` (still configured explicitly).
- Add new dataset formats or refactor dataset-layer architecture beyond what is needed to support default-backend action chunking.

## Decisions

### D1: Implement default-backend action chunking by mapping `action_chunk_size` → `delta_timestamps["action"]`

**Choice:** For `backend="default"` and `action_chunk_size=K`, derive:

- `fps = metadata.fps`
- `delta_timestamps["action"] = [step / fps for step in range(K)]`

Then call `LeRobotDataset(..., delta_timestamps=<merged>)`.

**Why:** `LeRobotDataset` already:
- rounds `delta_timestamps * fps` to integer frame deltas,
- clamps queries to `[episode_start, episode_end - 1]`,
- returns `{key}_is_pad` masks using the same episode-boundary predicate we want.

This yields `action` of shape `[K, action_dim]` per-sample and `action_is_pad` of shape `[K]`, matching the lazy backend’s contract after collation.

**Alternatives considered:**
- Implement a custom `ActionChunkTransform` wrapper around `LeRobotDataset` that re-indexes into the dataset for each sample. Rejected: transforms in YAVLA are sample-only (no index context), and re-indexing would duplicate LeRobot’s episode-boundary logic.
- Require users to switch to `lazy` backend. Rejected: prevents the first end-to-end run on small datasets and is unnecessary given LeRobot’s existing temporal query support.

### D2: Define chunk semantics in upstream LeRobot terms

**Choice:** Define “correct” chunk semantics to match upstream LeRobot `delta_timestamps` behavior for forward action deltas, and ensure the lazy backend matches it. For `action_chunk_size=K`, this means:

- includes current action at step 0
- pads only at episode end (future steps beyond episode end are pad)
- clamps padded steps to the last valid frame (duplicating the final action)

**Why:** The default backend is a thin wrapper around upstream `LeRobotDataset`, so upstream behavior is the most direct source of truth. The lazy backend already implements the same semantics for contiguous forward chunks; keeping them aligned ensures training code remains backend-agnostic.

**Alternatives considered:**
- Treat the lazy backend’s current implementation as the source of truth. Rejected: the goal is “default backend == upstream LeRobot,” so semantics should be expressed in those upstream terms and then matched by lazy.

### D3: Resolve configuration conflicts explicitly (`action_chunk_size` vs `delta_timestamps["action"]`)

**Choice:** If `action_chunk_size` is set and the user also provides `delta_timestamps` containing an `"action"` entry, raise a `ValueError` with a clear message (choose one mechanism).

**Why:** Both mechanisms attempt to define the shape/semantics of the `action` key. Allowing both silently risks mismatched shapes vs the model’s `chunk_len` and inconsistent behavior between backends (lazy currently overwrites `action` in its action-chunk augmentation step).

Users who need custom/non-contiguous action deltas should configure `delta_timestamps["action"]` directly and leave `action_chunk_size` unset.

**Alternatives considered:**
- Let explicit `delta_timestamps["action"]` override `action_chunk_size`. Rejected: would make `action_chunk_size` misleading and make training failures harder to diagnose.
- Merge lists (e.g., union of deltas). Rejected: changes `action` length in a way that is likely incompatible with the action head’s fixed `chunk_len`.

### D4: Update dataset-factory guardrails: streaming still rejects temporal features; default no longer does

**Choice:** Adjust factory validation so:
- `backend="streaming"` rejects `delta_timestamps` and `action_chunk_size` (unchanged)
- `backend="default"` accepts `action_chunk_size` and implements it via D1

**Why:** Streaming datasets are iterable and shard-interleaved; per-sample temporal lookups are not currently supported and would likely require a different design.

## Risks / Trade-offs

**[Risk] Default-backend performance regression for action chunking** → `LeRobotDataset` will perform additional index queries per sample when `delta_timestamps` is set.
→ Mitigation: This change targets small datasets where `backend="default"` is desirable; for large-scale runs, `lazy` remains the recommended backend.

**[Risk] Shape mismatch between `action_chunk_size` and model `chunk_len`** → Training will fail if the action head expects `chunk_len != action_chunk_size`.
→ Mitigation: Keep error messages actionable (training collate already explains the need for chunking); optionally add a future validation step in training config (out of scope here).

**[Risk] Float rounding / delta validation** → LeRobot validates deltas are multiples of `1/fps` within tolerance.
→ Mitigation: derive deltas as exact `step / fps` based on metadata `fps`, which should pass LeRobot’s check.

## Migration Plan

1. Implement D1–D4 in `src/yavla/data/factory.py` (compose effective `delta_timestamps` for default backend when `action_chunk_size` is set, and relax backend guardrails).
2. Update unit tests that currently expect default backend to reject `action_chunk_size`.
3. Add an integration test using a real dataset (e.g., `lerobot/pusht`) to assert:
   - `create_dataloader(... backend="default", action_chunk_size=K)` yields samples with `action.ndim == 2` and `action_is_pad` present
   - near episode end, `action_is_pad` contains at least one `True`
4. Update docs/examples to advertise “default backend + action_chunk_size” as a supported path for the first training run (including dataset-layer docs and the training guide).

Rollback: revert the factory change and restore the `ValueError` guard for `backend="default"` + `action_chunk_size`.

## Open Questions

- Should we add a first-class training-time check that `dataset.action_chunk_size == policy.action_head.chunk_len` to fail fast with a clearer error?
- Do we want to extend streaming backend to support temporal features via a windowed/sharded buffering strategy, or keep “streaming is temporal-feature-incompatible” as a permanent constraint?
