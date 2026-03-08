## 1. Config And Contract

- [x] 1.1 Add an explicit embodiment adaptation config to `PolicyConfig` with `exact` and pretrained-VLA (`max_padded`) modes, active dimensions, maximum dimensions, and fail-fast validation.
- [x] 1.2 Refactor policy construction so module widths are derived from the shared embodiment config instead of duplicated action/proprio dimension settings.
- [x] 1.3 Remove legacy flat train-config fallback plus legacy config/checkpoint migration, and require explicit embodiment-aware config/checkpoint metadata.

## 2. Model And Checkpoint Behavior

- [x] 2.1 Implement pretrained-VLA proprio padding and active action-dimension mask construction inside the policy/training path while keeping dataset batches embodiment-exact.
- [x] 2.2 Update action-head loss computation to combine timestep padding with active-dimension masking and return finite zero loss for fully masked batches.
- [x] 2.3 Update prediction and decoding paths so internal max-width predictions are sliced back to the active embodiment width before being returned.
- [x] 2.4 Extend `save_pretrained()` / `from_pretrained()` metadata and validation to record max-width plus active-embodiment information, reject embodiment-less legacy formats, and support strict vs non-strict rebinding rules.

## 3. Documentation And Examples

- [x] 3.1 Update `docs/training-guide.md` to explain exact-dimension MVP training versus pretrained-VLA embodiment adaptation, including YAML examples and unsupported legacy formats.
- [x] 3.2 Update dataset-layer documentation to state that action/proprio tensors remain embodiment-exact and are not manually padded in the dataset layer.
- [x] 3.3 Add or update example training configs that demonstrate pretrained-VLA embodiment adaptation fields and checkpoint loading expectations.

## 4. Verification

- [x] 4.1 Add unit tests for embodiment config validation, proprio padding, combined timestep/dimension masking, and the all-masked zero-loss case.
- [x] 4.2 Add prediction tests that verify active-width slicing from a wider internal action head.
- [x] 4.3 Add checkpoint tests covering rejection of legacy embodiment-less formats, strict max-width validation, and non-strict loading of a smaller compatible embodiment.
