## Context

`lerobot` v3 metadata loads `meta.episodes` as a HuggingFace `datasets.Dataset`, where `to_dict(orient="records")` is unsupported. Real `lerobot/pusht` data also stores camera/video references in episode-level metadata while data parquet rows often omit direct media columns. Current YAVLA lazy/streaming implementations already support these realities, but the behavior is not yet encoded in OpenSpec deltas for dataset capabilities.

## Goals / Non-Goals

**Goals:**
- Define normative requirements for metadata ingestion compatibility across HF Dataset, pandas DataFrame, and list-like records.
- Define normative requirements for dual-path media resolution in lazy and streaming backends:
  - row payload decode when media payload is present
  - canonical metadata decode via `video_path` + `videos/{key}/*` fields when row payload is absent
- Re-align integration acceptance tests with strict media tensor assertions already described by existing integration requirement wording.
- Document this behavior in targeted dataset-layer caveats.

**Non-Goals:**
- No constructor/API redesign for dataset backends.
- No OOM tuning/performance optimization work beyond preserving existing no-eager-frame-load behavior.
- No broad architecture-doc rewrite outside dataset-layer docs.

## Decisions

1. **Additive follow-up change, not retroactive rewrite**
- Keep `dataset-integration-tests` as historical record.
- Add a new change that formalizes runtime compatibility behavior and test strictness alignment.

2. **Spec-level strict media assertions remain authoritative**
- Integration tests will enforce decoded 3-D tensor media outputs when baseline decode is available.
- Test environment variability remains handled via explicit baseline-driven skip.

3. **Behavior remains internal and API-stable**
- Requirements describe internal decode-source resolution and metadata normalization semantics.
- No new user-facing constructor arguments.

4. **Memory intent preserved**
- Clarify that compatibility behavior does not introduce eager frame parquet loads at `__init__` time.

## Risks / Trade-offs

- **[Risk] Environment-dependent video decoding can cause flaky expectations**
  - **Mitigation:** keep baseline gate against upstream `LeRobotDataset`; skip with explicit reason when baseline decode is unavailable.
- **[Risk] Requirement overlap with existing lazy/streaming specs could duplicate language**
  - **Mitigation:** add narrowly scoped ADDED requirements focused on compatibility and media-source resolution only.
- **[Risk] Future upstream schema changes (new metadata containers/fields)**
  - **Mitigation:** keep unit regression around HF `Dataset` metadata normalization and continue using integration tests against real data.
