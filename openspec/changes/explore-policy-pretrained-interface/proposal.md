## Why

`VLAPolicy.save_pretrained()` and `VLAPolicy.from_pretrained()` currently own a large amount of packaging logic directly inside `src/yavla/models/policy.py`, including config serialization, embodiment metadata, LoRA adapter handling, and checkpoint rebinding. That surface is already complex, partially backbone-specific, and not yet clearly aligned with either a stable YAVLA contract or the pretrained-policy conventions provided by the surrounding ecosystem, so it needs focused exploration before further implementation hardens it.

## What Changes

- Capture this as a backlog exploration change rather than implementation-ready work.
- Explore and document the desired long-term pretrained policy interface for YAVLA before making more checkpoint-format changes.
- Evaluate whether YAVLA should keep a fully custom `save_pretrained()` / `from_pretrained()` contract, adopt a Hugging Face Hub or LeRobot-style mixin pattern, or split responsibilities between policy-level and backbone-level serialization.
- Define the required compatibility guarantees for config loading, embodiment-aware rebinding, LoRA adapter persistence, and backbone-specific checkpoint behavior.
- Identify which parts of the current implementation are true YAVLA-owned requirements versus generic pretrained-model packaging that should be delegated to stable third-party libraries.
- Defer implementation until the design phase resolves the ownership boundary and migration strategy.

## Capabilities

### New Capabilities
- `policy-pretrained-interface`: defines the required save/load contract for YAVLA policies, including checkpoint contents, embodiment-aware validation, adapter handling, and library integration boundaries.

### Modified Capabilities
- _(none)_

## Impact

- **Status**: backlog exploration only; no implementation work is committed by this change yet.
- **Primary code under review**: `src/yavla/models/policy.py` and any future backbone-specific checkpoint hooks.
- **Potential follow-on changes**: checkpoint file layout, adapter save/load ownership, config migration rules, and user-facing checkpoint loading APIs.
- **Dependencies under evaluation**: Hugging Face Hub pretrained-model mixins, PEFT adapter save/load conventions, and LeRobot pretrained-policy patterns.
- **Risk**: medium to high, because this interface affects checkpoint compatibility and should not be refactored further without a clearer contract.
