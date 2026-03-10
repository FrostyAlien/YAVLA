## Context

`MLPRegressionHead.compute_loss()` already supports two masking inputs:

- `TrainingBatch.action_mask` for padded or invalid timesteps
- `TrainingBatch.action_dim_mask` for inactive action dimensions in pretrained-VLA embodiment adaptation

The current implementation builds a combined boolean validity mask and reduces L1 loss over the selected elements. When the combined mask contains zero valid elements, it returns `predicted.new_zeros(())` directly. That scalar is finite, but it is detached from autograd, so `LossDict.total.backward()` fails in the training loop once a fully masked batch reaches `accelerate.backward(...)`.

This is a head-level contract problem, not a trainer-level one. The trainer assumes every returned `LossDict.total` is safe to backpropagate, including the zero-loss edge case. Existing tests already cover finite zero loss for fully masked chunks, but they do not verify that the zero-valued loss still participates in autograd.

Three constraints shape the design:

1. Mask polarity and public batch structure must remain unchanged: `True` still means padded or inactive.
2. Unmasked and partially masked batches must preserve current scalar loss semantics.
3. The fix must remain safe under Accelerate and distributed training without introducing rank-local batch skipping.

## Goals / Non-Goals

**Goals:**
- Guarantee that fully masked action losses return a zero-valued scalar that remains connected to autograd
- Preserve current L1 normalization over valid action elements for partially masked batches
- Keep the masking contract identical across timestep masks, action-dimension masks, and their combination
- Add regression tests that prove fully masked losses are backward-safe, not merely finite

**Non-Goals:**
- Change dataset behavior or require LeRobot batches to avoid fully masked edge cases
- Add trainer-level special handling for detached losses or batch skipping
- Refactor other action heads beyond the affected MLP regression path
- Add new logging, metrics, or observability for masked batches in this change

## Decisions

### D1: Replace the detached zero branch with a unified masked reduction

The masked path should compute unreduced elementwise L1 loss first, multiply it by a float validity mask, and divide by the number of valid elements clamped to a minimum of one.

Conceptually:

```python
per_elem_l1 = F.l1_loss(predicted, target, reduction="none")
valid_f = valid.to(per_elem_l1.dtype)
l1 = (per_elem_l1 * valid_f).sum() / valid_f.sum().clamp_min(1)
```

Why:
- the numerator stays connected to `predicted`, so the fully masked case becomes a graph-connected zero rather than a detached scalar
- partially masked batches still normalize by the count of valid action elements
- the implementation no longer needs a separate `valid_count == 0` return path

Alternative considered:
- keep the current branch and return `predicted.sum() * 0.0` when `valid_count == 0`
  - Rejected because it preserves split reduction logic and makes the autograd-safe behavior easier to regress later

### D2: Keep the no-mask fast path unchanged

If both `action_mask` and `action_dim_mask` are `None`, the head should continue to call `F.l1_loss(predicted, target)` directly.

Why:
- it preserves existing behavior for the common case
- it avoids adding mask construction overhead where no masking is required
- it keeps the change focused on the broken masked-loss path

Alternative considered:
- always build a full validity mask, even when no masks are provided
  - Rejected because it adds churn without improving correctness for the unmasked path

### D3: Treat backward-safe zero loss as the action-head contract

The trainer and `Accelerate` integration should remain unchanged. The action head is responsible for returning a scalar loss that is always valid for `backward()`, including zero-valued edge cases.

Why:
- the failure originates at the head boundary where `LossDict.total` is created
- trainer-level skipping is more error-prone under distributed training because different ranks may observe different masking patterns
- fixing the contract locally keeps the behavior consistent for all existing training entry points

Alternative considered:
- detect detached losses in the trainer and skip backward for that step
  - Rejected because it hides the underlying contract violation and complicates distributed synchronization semantics

### D4: Extend unit tests at the head boundary instead of adding trainer integration coverage first

Regression coverage should expand the existing fully masked tests in `tests/models/test_heads.py` to assert that:

- the returned zero loss is finite and numerically zero
- `loss.total.requires_grad` remains true when predictions require gradients
- `loss.total.backward()` succeeds for fully masked timestep, dimension, and combined-mask cases

Why:
- the bug is deterministic and local to `MLPRegressionHead.compute_loss()`
- unit tests are cheaper and more stable than reproducing the failure through a full Accelerate training harness
- the existing tests already exercise the relevant mask combinations, so this change can strengthen them rather than introduce a new testing surface

Alternative considered:
- add only a trainer-level regression
  - Rejected because it would be slower, more brittle, and less precise about the ownership of the loss contract

## Risks / Trade-offs

- [Fully masked batches still consume a forward/backward step with zero gradients] -> Accept in this change; preserving training stability is more important than trying to skip steps safely across ranks
- [Clamping the denominator to one could hide unexpectedly frequent fully masked batches] -> Mitigate with targeted tests now; add logging in a follow-up change only if the frequency becomes operationally significant
- [The same detached-zero pattern could exist in other heads] -> Limit scope to the proven failing MLP path and treat broader auditing as separate follow-up work
- [Mask semantics could drift between specs and implementation] -> Update the affected specs in this change so fully masked cases explicitly require backward-safe zero loss

## Migration Plan

1. Update the `mvp-action-head` and `embodiment-adaptation` change specs so fully masked losses must remain differentiable and backward-safe.
2. Replace the detached zero branch in `src/yavla/models/heads/mlp.py` with the unified masked reduction.
3. Strengthen `tests/models/test_heads.py` to assert autograd participation for fully masked loss paths.
4. Run targeted test coverage for the MLP head and any affected training-path tests.

Rollback:
- Revert the masked-reduction change and the new regression assertions if needed. This would restore the current crash behavior, so rollback should only happen alongside a replacement fix.

## Open Questions

- Do we want to expose a metric for zero-valid batches in the trainer once the crash is fixed, or keep that as separate observability work?
- After this fix lands, should other action heads be audited for detached zero-loss branches under masking edge cases?
