## Why

YAVLA’s training pipeline is intended to run end-to-end on real LeRobot datasets (e.g., `lerobot/aloha_sim`), which commonly provide multiple camera views. Today our PaliGemma vision encoder hard-fails when more than one camera image is present, blocking the first full training run and preventing multi-view policies.

## What Changes

- Add first-class multi-camera support to the VLA observation → vision-encoding path (accept `ObservationBatch.images` with 1+ cameras).
- Define deterministic camera ordering and camera identity handling so multi-camera tokenization is stable across runs.
- Update the PaliGemma vision encoder implementation to encode multiple cameras (no single-camera limitation).
- Add tests covering multi-camera batches flowing through `TrainingCollate` → `VLAPolicy.encode_observations()` → merger → backbone → head.
- Update docs/config examples to show multi-camera training usage and constraints (e.g., expected shapes).

## Capabilities

### New Capabilities
- `multi-camera-vision-encoding`: Encode and merge tokens from multiple camera images in a single observation batch with deterministic ordering and stable shapes.

### Modified Capabilities

<!-- None -->

## Impact

- Model code: `src/yavla/models/backbones/paligemma.py`, potentially merger/collate ordering in `src/yavla/models/merger.py` and `src/yavla/training/data.py`.
- Tests: new/updated unit and integration coverage for multi-camera training batches.
- Docs/config: training guide and example configs for multi-camera datasets.
