## 1. Refactor train-facing config types

- [ ] 1.1 Introduce explicit concrete config dataclasses for the train-facing polymorphic variants that currently depend on loader-side subtype reconstruction, starting with `policy.vision_encoder` and its default `from_backbone` branch.
- [ ] 1.2 Update `PolicyConfig` and any directly related annotations to use explicit typed variants for the train path instead of a placeholder base config plus loader-side string dispatch.
- [ ] 1.3 Ensure nested polymorphic leaves needed by the train path, such as multi-tower vision encoder sub-configs, use the same typed-variant pattern and preserve stable runtime `type` values for registry builders.

## 2. Replace the train entry config-loading path

- [ ] 2.1 Remove the generic recursive hydration helpers from `scripts/train.py` and replace them with a thin `--config` loading path that builds a typed `TrainConfig` default object.
- [ ] 2.2 Keep tyro as the source of truth for applying CLI overrides by passing the typed default object through `tyro.cli(..., default=...)`.
- [ ] 2.3 Reject unsupported config shapes with clear errors instead of coercing them through loader-specific fallback behavior.

## 3. Align CLI and config examples with the new contract

- [ ] 3.1 Update training config examples under `configs/` to use the new typed variant schema for any train-facing polymorphic fields touched by this change.
- [ ] 3.2 Update the training guide and any other train-entry documentation to describe the new YAML schema and the tyro-native CLI selection/override behavior for typed variants.
- [ ] 3.3 Verify that variant names are stable and aligned across YAML examples, CLI help, and runtime `type` values.

## 4. Validate behavior with contract tests

- [ ] 4.1 Replace loader-internals-focused train CLI tests with contract tests that verify nested `TrainConfig` YAML defaults are loaded as typed defaults before tyro overrides.
- [ ] 4.2 Add tests that cover the default train-facing polymorphic variant without generic subtype reconstruction logic.
- [ ] 4.3 Add tests that cover selecting a non-default polymorphic variant from YAML and overriding variant-specific fields through the CLI.
- [ ] 4.4 Remove obsolete helper-specific assertions and confirm the remaining train-entry tests reflect the new schema and behavior.
