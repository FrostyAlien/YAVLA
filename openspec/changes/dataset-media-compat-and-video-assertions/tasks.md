## 1. OpenSpec Artifacts

- [x] 1.1 Create follow-up `proposal.md` describing runtime/spec drift and alignment goals
- [x] 1.2 Create `design.md` documenting compatibility decisions and test strictness alignment
- [x] 1.3 Add `lazy-dataset` delta spec for metadata container compatibility and dual-path media resolution
- [x] 1.4 Add `streaming-dataset` delta spec for metadata container compatibility and dual-path media resolution

## 2. Test Alignment

- [x] 2.1 Tighten integration media assertions in `tests/integration/test_lerobot_pusht.py` to require 3-D tensor outputs and dtype in `{torch.float32, torch.uint8}`
- [x] 2.2 Preserve baseline-driven skip behavior when upstream `LeRobotDataset` cannot decode media in the environment
- [x] 2.3 Add a focused unit regression test for HF `datasets.Dataset` metadata record normalization to prevent `to_dict(orient=...)` regression

## 3. Dataset-Layer Docs

- [x] 3.1 Update `docs/dataset-layer/caveats.md` with a “Media Source Resolution” section covering dual-path behavior and timestamp/from_timestamp rules
- [x] 3.2 Add a small pointer in `docs/dataset-layer/README.md` to the new caveat content

## 4. Validation

- [x] 4.1 Run `pixi run -e dev pytest tests/data/test_lazy_dataset.py tests/data/test_streaming_dataset.py -vv`
- [x] 4.2 Run `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$PWD/.cache/hf-home HF_DATASETS_CACHE=$PWD/.cache/hf-home/datasets pixi run -e dev pytest -m integration tests/integration/test_lerobot_pusht.py -vv`
- [x] 4.3 Run `pixi run -e dev pytest`
- [x] 4.4 Mark all completed tasks as `- [x]` with checklist reflecting actual results
