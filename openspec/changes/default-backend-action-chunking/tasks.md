## 1. Default Backend Action Chunking (Factory)

- [ ] 1.1 Update `select_backend()` in `src/yavla/data/factory.py` to allow `backend="default"` with `action_chunk_size` (remove the current `ValueError`), while keeping `backend="streaming"` temporal-feature guardrails.
- [ ] 1.2 Implement conflict validation: if `action_chunk_size` is set and `delta_timestamps` is set with an explicit `"action"` entry, raise a `ValueError` explaining the ambiguity.
- [ ] 1.3 Update the default-backend `LeRobotDataset(...)` construction in `create_dataloader()` to derive `delta_timestamps["action"] = [step / fps for step in range(action_chunk_size)]` using `metadata.fps`, merge it with any existing non-action `delta_timestamps`, and pass the merged dict to LeRobot so samples include `action_is_pad`.

## 2. Unit Tests (Factory)

- [ ] 2.1 Update `tests/data/test_factory.py` to remove expectations that default backend rejects `action_chunk_size` (adjust both `select_backend()` and `create_dataloader()` tests accordingly).
- [ ] 2.2 Add a unit test that monkeypatches `LeRobotDataset` to capture constructor kwargs, then asserts `create_dataloader(DataConfig(backend=\"default\", action_chunk_size=K, ...))` passes a `delta_timestamps` containing an `"action"` entry with length `K`.
- [ ] 2.3 Add a unit test that asserts the conflict case (`action_chunk_size` + `delta_timestamps[\"action\"]`) raises a `ValueError` with an actionable message.

## 3. Integration Tests (Real Data)

- [ ] 3.1 Add an integration test in `tests/integration/test_lerobot_pusht.py` that calls `create_dataloader(DataConfig(repo_id=\"lerobot/pusht\", backend=\"default\", action_chunk_size=4, batch_size=2, num_workers=0, normalize=False))`.
- [ ] 3.2 In that test, compute the final absolute frame index of episode 0 from `LeRobotDatasetMetadata.episodes`, access `sample = dataloader.dataset[last_idx]`, and assert `sample[\"action\"].shape[0] == 4` and `sample[\"action_is_pad\"]` is boolean with at least one `True` (end-of-episode padding).

## 4. Docs / Example Config

- [ ] 4.1 Update `docs/training-guide.md` to document “default backend with action chunking” as a supported recipe (and keep `lazy` as the recommended option for large datasets).
- [ ] 4.2 Update `docs/dataset-layer/usage.md` to reflect that `action_chunk_size` is supported on `backend=\"default\"`, and document the conflict rule with `delta_timestamps[\"action\"]`.
- [ ] 4.3 Update `configs/train.yaml` to include a sensible default `dataset.action_chunk_size` that matches the default action head `chunk_len` so an end-to-end training run does not fail in `TrainingCollate`.

## 5. Verification

- [ ] 5.1 Run `pixi run -e dev pytest tests/data/test_factory.py -v`.
- [ ] 5.2 Run `pixi run -e dev pytest tests/integration/test_lerobot_pusht.py -v -m integration`.
- [ ] 5.3 Run `pixi run -e dev lint` and `pixi run -e dev typecheck`.

