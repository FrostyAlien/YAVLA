## 1. Default Backend Action Chunking (Factory)

- [x] 1.1 Update `select_backend()` in `src/yavla/data/factory.py` to allow `backend="default"` with `action_chunk_size` (remove the current `ValueError`), while keeping `backend="streaming"` temporal-feature guardrails.
- [x] 1.2 Implement conflict validation: if `action_chunk_size` is set and `delta_timestamps` is set with an explicit `"action"` entry, raise a `ValueError` explaining the ambiguity.
- [x] 1.3 Update the default-backend `LeRobotDataset(...)` construction in `create_dataloader()` to derive `delta_timestamps["action"] = [step / fps for step in range(action_chunk_size)]` using `metadata.fps`, merge it with any existing non-action `delta_timestamps`, and pass the merged dict to LeRobot so samples include `action_is_pad`.

## 2. Unit Tests (Factory)

- [x] 2.1 Update `tests/data/test_factory.py` to remove expectations that default backend rejects `action_chunk_size` (adjust both `select_backend()` and `create_dataloader()` tests accordingly).
- [x] 2.2 Add a unit test that monkeypatches `LeRobotDataset` to capture constructor kwargs, then asserts `create_dataloader(DataConfig(backend=\"default\", action_chunk_size=K, ...))` passes a `delta_timestamps` containing an `"action"` entry with length `K` and values derived as `step / metadata.fps`.
- [x] 2.3 Add a unit test that asserts the conflict case (`action_chunk_size` + `delta_timestamps[\"action\"]`) raises a `ValueError` with an actionable message.

## 3. Integration Tests (Real Data)

- [x] 3.1 Add an integration test in `tests/integration/test_lerobot_pusht.py` that calls `create_dataloader(DataConfig(repo_id=\"lerobot/pusht\", backend=\"default\", action_chunk_size=4, batch_size=2, num_workers=0, normalize=False))`.
- [x] 3.2 In that test, compute the final absolute frame index of episode 0 from `LeRobotDatasetMetadata.episodes`, access `sample = dataloader.dataset[last_idx]`, and assert `sample[\"action\"].shape[0] == 4` and `sample[\"action_is_pad\"]` is boolean with at least one `True` (end-of-episode padding). Also assert the step-0 entry is not padded.

## 4. Docs / Example Config

- [x] 4.1 Update `docs/training-guide.md` to document “default backend with action chunking” as a supported recipe (and keep `lazy` as the recommended option for large datasets).
- [x] 4.2 Update `docs/dataset-layer/backend-guide.md` to reflect that `action_chunk_size` is supported on `backend=\"default\"` (decision flow + feature compatibility table).
- [x] 4.3 Update `docs/dataset-layer/architecture.md` to mention that the default backend supports `action_chunk_size` by delegating to upstream LeRobot `delta_timestamps[\"action\"]`.
- [x] 4.4 Update `docs/dataset-layer/usage.md` to reflect that `action_chunk_size` is supported on `backend=\"default\"`, document the conflict rule with `delta_timestamps[\"action\"]`, and note that custom/non-contiguous action deltas should be expressed via `delta_timestamps[\"action\"]` (leaving `action_chunk_size` unset).
- [x] 4.5 Update `configs/train.yaml` to include a sensible default `dataset.action_chunk_size` that matches the default action head `chunk_len` so an end-to-end training run does not fail in `TrainingCollate`.

## 5. Verification

- [x] 5.1 Run `pixi run -e dev pytest tests/data/test_factory.py -v`.
- [x] 5.2 Run `pixi run -e dev pytest tests/integration/test_lerobot_pusht.py -v -m integration`.
- [x] 5.3 Run `pixi run -e dev lint` and `pixi run -e dev typecheck`.

## 6. Sync Main Specs

- [x] 6.1 Sync delta specs into main specs: update `openspec/specs/dataset-factory/spec.md` and `openspec/specs/dataset-integration-testing/spec.md` to match the change (default backend supports action chunking, conflict validation, and integration coverage).
- [x] 6.2 Run `openspec validate dataset-factory --type spec --strict` and `openspec validate dataset-integration-testing --type spec --strict`.
