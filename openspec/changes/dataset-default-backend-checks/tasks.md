## 1. OpenSpec Artifacts

- [x] 1.1 Add `proposal.md` for default-backend integration parity scope
- [x] 1.2 Add `design.md` describing strict default decode checks and gating behavior
- [x] 1.3 Add `dataset-integration-testing` delta spec for default backend decode + factory default dataloader checks

## 2. Integration Test Changes

- [x] 2.1 Update `tests/integration/test_lerobot_pusht.py` with strict default backend media decode test (`LeRobotDataset`)
- [x] 2.2 Add integration test for `create_dataloader(..., backend="default")` real-data batch validation and batched media decode assertion
- [x] 2.3 Keep decode checks baseline-gated with explicit skip reasons for environment decode unavailability

## 3. Docs

- [x] 3.1 Update `docs/dataset-layer/caveats.md` with a brief note about default-backend integration parity checks and decode environment variability

## 4. Validation

- [x] 4.1 Run `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$PWD/.cache/hf-home HF_DATASETS_CACHE=$PWD/.cache/hf-home/datasets pixi run -e dev pytest -m integration tests/integration/test_lerobot_pusht.py -vv`
- [x] 4.2 Run `pixi run -e dev pytest tests/data/test_factory.py -vv`
- [x] 4.3 Mark completed tasks as `- [x]` only after validation succeeds
