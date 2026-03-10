## 1. Masked loss implementation

- [x] 1.1 Replace the detached zero-loss branch in `src/yavla/models/heads/mlp.py` with a unified masked L1 reduction that stays connected to autograd when the combined mask has zero valid elements.
- [x] 1.2 Preserve the existing no-mask fast path and masked reduction semantics for padded timesteps, inactive action dimensions, and their combination.

## 2. Regression coverage

- [x] 2.1 Update `tests/models/test_heads.py` so the fully masked timestep-loss tests assert the returned zero loss remains finite, requires gradients when predictions do, and can be passed to `backward()`.
- [x] 2.2 Add equivalent backward-safety assertions for fully dimension-masked and combined-mask edge cases in the MLP head tests.

## 3. Verification

- [x] 3.1 Run targeted pytest coverage for `tests/models/test_heads.py` and confirm the masked-loss scenarios pass with the new reduction.
- [x] 3.2 Re-run or otherwise validate the previously failing training path enough to confirm fully masked batches no longer crash `accelerate.backward(...)`.
