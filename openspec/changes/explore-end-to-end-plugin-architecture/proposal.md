## Why

YAVLA already carries registries, protocol abstractions, and a config-driven architecture story, but the current `build_policy()` path still hardcodes several concrete modules instead of honoring those registries end-to-end. That mismatch makes the system more complex than a simple fixed pipeline while still falling short of the actual plugin architecture the project claims to expose.

## What Changes

- Capture this as a backlog exploration change rather than implementation-ready work.
- Explore and define what YAVLA's plugin architecture should guarantee across the full policy assembly path.
- Evaluate the gap between the existing registry/protocol surface and the modules that are still instantiated directly inside `build_policy()`.
- Decide which module families are truly intended to be swappable through config and registries, and which should remain intentionally fixed in the MVP.
- Clarify how policy construction, serialization, validation, and default config generation should behave when every module is registry-driven.
- Defer implementation until the design phase resolves whether YAVLA should fully honor the plugin contract or deliberately narrow the public extensibility surface.

## Capabilities

### New Capabilities
- `end-to-end-plugin-architecture`: defines the expected end-to-end plugin contract for policy assembly, including which module families are registry-built, how defaults are selected, and what extensibility guarantees are public.

### Modified Capabilities
- `vision-encoder-registry`: the registry story will be revisited in the context of a broader end-to-end plugin contract, so its guarantees must align with the final policy-factory design instead of remaining a one-off exception.

## Impact

- **Status**: backlog exploration only; no implementation work is committed by this change yet.
- **Primary code under review**: `src/yavla/models/policy.py`, `src/yavla/models/registry.py`, `src/yavla/models/vlm_registry.py`, and the module families under `src/yavla/models/`.
- **Potential follow-on changes**: policy factory construction rules, config dataclass shape, registry APIs, checkpoint/config reconstruction, and tests that currently assume direct instantiation of specific module classes.
- **Architectural scope**: high, because this touches the project’s public extensibility model rather than a single implementation detail.
- **Risk**: medium to high, since overcommitting to a plugin system has maintenance cost, but keeping a half-real plugin surface also creates confusion and drift.
