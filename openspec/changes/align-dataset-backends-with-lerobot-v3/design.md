## Context

YAVLA currently exposes three dataset backends through `DataConfig.backend` and `create_dataloader()`:

- `default` wraps upstream `LeRobotDataset`
- `lazy` uses YAVLA's `LazyLeRobotDataset`
- `streaming` uses YAVLA's `ShardInterleavedDataset`

Deeper research narrowed the practical question:

- `default` is already LeRobot-native in practice because the factory directly instantiates `LeRobotDataset`
- the pinned LeRobot revision already includes `StreamingLeRobotDataset`
- `LazyLeRobotDataset` remains the clearest YAVLA-specific value because it preserves map-style random access, works cleanly with samplers, and gives YAVLA tighter decoder/cache control

The most important additional conclusion is that true random access remains a map-style concern. Both YAVLA's current streaming backend and upstream `StreamingLeRobotDataset` are iterable paths, not primary random-access training substrates. That makes it premature to force an immediate streaming replacement just for ownership symmetry.

The proposal still captures the longer-term simplification direction. This design intentionally narrows near-term execution scope without revising the proposal text:

- treat `default` as the already-aligned upstream baseline
- treat `lazy` as the supported large-dataset random-access backend
- treat `streaming` as experimental only
- defer any replacement of the experimental streaming backend with upstream `StreamingLeRobotDataset` to backlog follow-up work

Three constraints shape the design:

1. YAVLA's current docs and specs overstate streaming as a peer training backend rather than an experimental iterable path
2. YAVLA is not currently using `backend="streaming"` in training configs, so there is no immediate pressure to force a replacement
3. Models and training flows that need true random access are better served by `default` or `lazy` than by any current streaming path

## Goals / Non-Goals

**Goals:**
- Document `default` as the already-aligned LeRobot-native baseline
- Document `lazy` as the supported YAVLA-owned random-access backend for larger datasets
- Mark `streaming` as experimental and outside the primary training contract
- Record possible replacement of the experimental streaming backend with upstream `StreamingLeRobotDataset` as backlog work
- Plan related documentation and spec edits without forcing an implementation change now

**Non-Goals:**
- Replace `ShardInterleavedDataset` with upstream `StreamingLeRobotDataset` in this change
- Remove `src/yavla/data/streaming.py` in this change
- Expand streaming guarantees to match random-access training needs
- Redesign `LazyLeRobotDataset`
- Change the default backend away from `LeRobotDataset`
- Unpin or upgrade LeRobot as part of this change

## Decisions

### D1: Treat `default` as an already-aligned upstream path

`backend="default"` remains a thin factory path to `LeRobotDataset`. The work here is framing, not implementation replacement.

Why:
- this path is already upstream-backed
- treating it as a migration target adds noise
- keeping it stable protects the primary training path

### D2: Treat `streaming` as experimental only

YAVLA should stop presenting `backend="streaming"` as a peer to `default` and `lazy` for general training. It remains available, but it is experimental and outside the primary random-access training contract.

This means the public guidance should be:

- use `default` for the upstream-native baseline
- use `lazy` when large local datasets still need map-style access or sampler semantics
- use `streaming` only as an experimental bounded-memory iterable path

Why:
- it matches the actual capabilities of the backends
- it avoids overpromising support for workloads that need random access
- it narrows the support surface without forcing an unnecessary implementation swap

Alternative considered:
- keep describing streaming as a fully supported peer backend
  - Rejected because it overstates what iterable backends guarantee for training workloads

### D3: Defer upstream streaming replacement to backlog

Replacing YAVLA's experimental streaming backend with upstream `StreamingLeRobotDataset` remains a reasonable future direction, but it is not the execution scope of this change.

The backlog item should explicitly evaluate:

- action-window compatibility
- distributed/process behavior
- epoch control and reproducibility
- decoder behavior and `video_backend` expectations

Why:
- upstream streaming is not a drop-in substitute for random-access training needs
- current YAVLA configs do not depend on streaming
- deferring the swap avoids turning an unused backend into a forced contract migration

Alternative considered:
- replace streaming now because upstream overlap exists
  - Rejected because the ownership simplification alone does not justify immediate contract churn

### D4: Keep `LazyLeRobotDataset` as the supported YAVLA-owned random-access backend

`LazyLeRobotDataset` remains the meaningful custom backend because it still occupies a distinct niche that neither upstream default loading nor upstream streaming replaces cleanly.

Its retained value is:

- map-style random access
- compatibility with sampler-driven uniform shuffling
- tighter control over decoder/cache behavior

Why:
- this is the backend that matches model and training flows needing real random access
- it gives YAVLA a justified custom surface rather than duplicate ownership for its own sake

### D5: Immediate work is documentation and contract positioning

This change should focus on aligning docs, specs, and task planning with the narrower support story:

- `default` is already upstream-native
- `lazy` is the supported random-access custom backend
- `streaming` is experimental
- possible future replacement with upstream `StreamingLeRobotDataset` is backlog, not current implementation scope

Why:
- it fixes the public story without forcing a backend rewrite
- it creates room for a future streaming decision only if a real use case appears

## Risks / Trade-offs

- [The proposal still describes the broader simplification direction] -> Acceptable for now; the design and task plan define the narrower near-term scope without rewriting the proposal text
- [Docs may continue to overpromise streaming until updated] -> Make docs the immediate planned work
- [YAVLA still carries a custom experimental streaming backend] -> Accept temporarily because it is unused and explicitly downgraded in support status
- [Future upstream replacement still needs real evaluation] -> Track it as backlog and revisit only when a concrete streaming use case appears

## Migration Plan

1. Update the change-local design, specs, and task plan so streaming is framed as experimental and upstream replacement is deferred to backlog
2. Edit related dataset-layer docs to describe backend roles accurately and recommend `default` or `lazy` for workloads needing random access
3. If a real streaming use case appears later, start a follow-up change to evaluate replacing the experimental backend with upstream `StreamingLeRobotDataset`

Rollback:
- no code rollback is needed because this narrowed plan does not require a backend implementation swap

## Open Questions

- Should the docs use only the term "experimental", or also explicitly say "not recommended for workloads requiring random access"?
- Should the docs mention upstream `StreamingLeRobotDataset` by name as the most likely future replacement, or keep that phrasing more generic?
- If streaming starts seeing real usage, do we want a runtime warning in addition to documentation guidance?
