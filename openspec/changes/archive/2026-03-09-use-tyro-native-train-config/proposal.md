## Why

`scripts/train.py` already presents itself as a tyro-driven entry point, but the actual configuration path is only partially tyro-native. It still relies on a handwritten YAML-to-dataclass hydration layer around `_resolve_config_dataclass()`, `_coerce_config_value()`, `_build_dataclass()`, and `_pop_config_flag()`.

That custom loader is now carrying too much architectural weight:

- it duplicates parsing and type-resolution behavior outside the config type system
- it hides polymorphic config behavior behind ad hoc string dispatch instead of typed contracts
- it makes `TrainConfig` evolution harder because new config shapes often require loader work, not just type changes

For future development, the right goal is not merely to delete helper functions while preserving every current config quirk. The better direction is to make the training config model genuinely tyro-aligned: typed defaults, typed polymorphism, and a thin file-loading path that feeds tyro instances instead of re-implementing recursive config semantics in the entry script.

## What Changes

- Redesign the train config loading path so tyro is the source of truth for config structure, defaults, and polymorphic behavior.
- Remove the generic recursive hydration helpers from `scripts/train.py` and replace them with a thin file-loading step that produces typed defaults for tyro rather than re-implementing config semantics.
- Treat the nested `TrainConfig` schema as the only supported training config format, with latitude to further refine field shapes so they map cleanly onto tyro-supported types.
- Prefer tyro-native typing patterns for future polymorphic config fields, especially unions of concrete config dataclasses instead of string-dispatched base config placeholders.
- Update training config examples, docs, and CLI tests to validate the new typed contract and its file-defaults plus CLI-overrides workflow.
- Clarify that future config evolution should happen in the dataclass/type layer first, not by expanding bespoke loader logic in entry scripts.

## Capabilities

### New Capabilities
- _(none)_

### Modified Capabilities
- `training-entry`: the train CLI config-loading path will be re-centered on tyro-native typed parsing and defaults, with config schema cleanup allowed where needed to support a sustainable long-term contract.

## Impact

- **Modified code**: `scripts/train.py`, training config dataclass/type definitions as needed, config-related CLI tests, and training config examples/docs.
- **Likely removals**: `_resolve_config_dataclass()`, `_coerce_config_value()`, and `_build_dataclass()` from the train entry script.
- **Likely schema changes**: config fields that currently depend on string-based subtype reconstruction may be retyped to explicit unions or other tyro-friendly structures.
- **Dependencies**: no new configuration framework is expected; the goal is to reduce ownership of custom config semantics, not add a second config system.
- **Risk**: medium, because this change intentionally optimizes for a better long-term config model rather than preserving every current config shape.
