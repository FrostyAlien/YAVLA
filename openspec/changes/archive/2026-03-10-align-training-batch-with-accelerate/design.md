## Context

YAVLA already uses typed dataclasses at model and training boundaries: `ObservationBatch` and `TrainingBatch` carry structured tensors plus non-tensor metadata such as language strings and timing fields. This is a deliberate design choice in the model architecture, and the training stack is already committed to HuggingFace Accelerate for mixed precision, gradient accumulation, clipping, checkpointing, and distributed execution.

The current failure happens at the seam between those two decisions. `Accelerate` prepares the model onto the accelerator device, but its automatic dataloader device placement is built around nested tensor containers it already understands. Plain dataclasses do not participate in that contract, so a real smoke run reaches `policy.forward(batch)` with model weights on the accelerator and nested batch tensors still on CPU.

The current test suite also under-covers this boundary:
- direct policy integration tests bypass `Trainer` and `Accelerate`
- trainer tests use stubs whose `forward()` ignores batch contents
- no test currently exercises a real typed batch through `Trainer` onto an accelerator-backed policy path

This change therefore needs a cross-cutting design, not a local patch in one function. The solution must preserve typed batch boundaries, keep Accelerate as the runtime foundation, and make the transport/device contract for typed batches explicit.

## Goals / Non-Goals

**Goals:**
- Keep `TrainingBatch` and `ObservationBatch` as the training/model boundary types.
- Keep HuggingFace Accelerate as the training runtime foundation.
- Define a runtime tensor-transport contract for typed training batches so their tensors can move devices without losing structured fields.
- Make trainer ownership of batch movement explicit instead of depending on implicit dataloader auto-placement for custom objects.
- Add regression coverage for the real `Trainer` + `Accelerate` + typed batch path so the first optimizer step succeeds on supported devices.

**Non-Goals:**
- Replacing typed batches with plain dicts, tuples, or tokenizer-specific container classes across the whole training stack.
- Removing or reducing YAVLA's use of Accelerate.
- Redesigning language payloads or moving tokenization to the dataloader boundary.
- Changing dataset backend strategy, dataset IDs, or unrelated smoke-config issues in this change.
- Solving every possible future `Accelerate` batch-dispatch mode for arbitrary iterable datasets beyond the current trainer contract.

## Decisions

### D1: Keep typed dataclasses and promote them into runtime tensor containers

**Choice:** `ObservationBatch` and `TrainingBatch` remain dataclasses, but they stop being passive DTOs only. Training-facing typed batches will gain a small runtime transport contract so tensor leaves can be transformed while preserving non-tensor fields.

This contract should support at least:
- device movement for nested tensors
- preservation of strings, floats, and other non-tensor metadata
- structured return types of the same batch class

The implementation should be centered on a single recursive tensor-tree primitive (for example, "apply a function to every tensor leaf"), with convenience methods such as device movement built on top of it.

**Why this choice:** The typed batch design is already deeply integrated with the policy interface and collate boundary. Replacing it with plain dict transport would weaken the explicit module contracts the repository has been building toward. Adding runtime tensor semantics preserves that design while fixing the missing operational behavior.

**Alternatives considered:**
- **Use plain dicts as transport batches:** more framework-native, but it weakens the typed boundary and leaks raw schema details deeper into model code.
- **Convert to namedtuple-style containers:** improves some framework interoperability, but still leaves awkward handling for non-tensor leaves like language strings and does not fit current dataclass-based code as naturally.
- **Keep dataclasses passive and teach only the trainer how to rebuild them:** works locally, but spreads tensor-transport logic away from the types that own the structure.

### D2: Trainer owns the device boundary explicitly

**Choice:** `Trainer` will stop assuming that `accelerator.prepare(train_dataloader)` is sufficient to make custom typed batches device-correct. The trainer runtime will treat batch movement as an explicit YAVLA concern at the step boundary.

The intended flow is:

```text
raw DataLoader batch
  -> typed TrainingBatch on host
  -> Trainer moves typed batch to accelerator.device
  -> policy(batch_on_device)
  -> accelerator.backward / clip / step
```

This design means `Accelerate` continues to own model/runtime orchestration, while YAVLA owns the transport semantics of its custom batch types.

**Why this choice:** It avoids relying on `Accelerate`'s implicit dataloader placement heuristics for custom objects. It also keeps the device boundary visible and debuggable in YAVLA's trainer code.

**Alternatives considered:**
- **Rely on `Accelerate` dataloader device placement alone:** this is exactly the behavior that failed for plain dataclasses.
- **Move tensors inside `policy.forward(...)`:** hides runtime transport inside model code, makes the policy responsible for host/device concerns, and complicates inference/training symmetry.

### D3: Keep non-tensor payloads in the batch and leave them host-safe

**Choice:** Non-tensor fields in typed batches, especially language strings, remain part of the training batch contract and are not forced into tensorized transport just to satisfy framework helpers.

The transport contract must therefore be selective:
- tensor leaves move
- non-tensor fields pass through unchanged

**Why this choice:** Language strings are a valid boundary type for the current policy API, and tokenization is already delegated to the backbone. Pulling tokenization forward into the dataloader would be a wider architectural change with caching, padding, and checkpoint-compatibility consequences.

**Alternatives considered:**
- **Tokenize language in the dataloader to eliminate strings from transport:** may help some framework batch utilities, but it changes the current policy contract and couples dataloading more tightly to backbone-specific tokenization.
- **Drop language from typed batches and fetch it elsewhere:** breaks locality and would make batch semantics less coherent.

### D4: Cover the real trainer path in tests, not only isolated pieces

**Choice:** The change will add regression coverage that exercises the actual `Trainer` + `Accelerate` path with a typed batch whose tensors are read by the policy. The key assertion is not only "no exception", but that a real optimizer step is reachable with typed batches under Accelerate-managed execution.

Test coverage should include:
- unit coverage for typed batch transport behavior
- trainer/runtime coverage where the batch contents are actually consumed
- at least one smoke-style path that would have caught the original device mismatch

**Why this choice:** The existing tests proved the policy math and the trainer loop separately, but not the custom-batch runtime seam between them.

**Alternatives considered:**
- **Rely on direct policy integration tests only:** misses the trainer/device boundary.
- **Rely on a full downloaded real-model smoke run only:** too expensive and too brittle as the primary regression detector.

### D5: Do not depend on advanced Accelerate batch dispatch semantics for typed batches

**Choice:** The design explicitly targets the ordinary YAVLA training path where each process/worker obtains its own batch and the trainer moves it. It does not require typed batches to participate in all possible `Accelerate` concatenation/dispatch behaviors for iterable dataloaders.

This is especially important because future or experimental iterable backends can interact poorly with generic batch concatenation, and raw string fields make that even harder.

**Why this choice:** YAVLA's current dataloaders already own backend-specific sampling/sharding behavior. Requiring typed batches to satisfy every advanced framework batch-manipulation path would over-constrain the design for little immediate benefit.

**Alternatives considered:**
- **Make typed batches fully compatible with generic framework concatenation/splitting:** possible, but it is broader than the concrete failure we are fixing and would likely force wider compromises in batch representation.

## Risks / Trade-offs

- **[Typed batch classes gain runtime behavior]** → Mitigation: keep the added API narrowly scoped to tensor transport, document it as a training-boundary contract, and avoid turning these classes into general utility objects.
- **[Explicit trainer-side batch movement overlaps with a framework feature]** → Mitigation: accept this duplication intentionally because YAVLA uses custom batch objects that sit outside `Accelerate`'s most reliable auto-placement path.
- **[Future iterable/distributed batch dispatch may still have edge cases]** → Mitigation: define the supported contract around ordinary trainer-side movement and defer broader dispatch/concatenation guarantees until a real use case requires them.
- **[More realistic tests can become heavier or more device-sensitive]** → Mitigation: keep core regression coverage stub-based but ensure the stub policy actually consumes batch tensors; reserve full real-model smoke runs for optional or higher-level validation.

## Migration Plan

1. Define the new typed-batch transport and runtime expectations in specs.
2. Implement batch transport support and trainer-side batch movement together so the codebase never sits in a half-migrated state.
3. Add regression tests for the typed batch transport behavior and the real trainer/device seam.
4. Re-run the one-step smoke path to confirm the first optimizer step completes under Accelerate.

Rollback is straightforward because the change is localized to typed batch transport semantics and the trainer boundary. If the approach proves problematic, the implementation can be reverted without changing the broader model or dataset abstractions.

## Open Questions

- Should the tensor-tree helper be a public method on the batch dataclasses, or an internal utility that powers only a narrow public `.to(...)` / transport API?
- Should `pin_memory` support be included in the same change, or treated as follow-up after the device-movement path is stable?
- Should other tensor-bearing boundary types eventually adopt the same transport protocol for consistency, or should this change stay limited to training-facing batches?
