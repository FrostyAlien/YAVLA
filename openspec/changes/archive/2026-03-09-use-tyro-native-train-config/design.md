## Context

`scripts/train.py` currently mixes two separate responsibilities:

1. loading defaults from a human-authored YAML file
2. parsing CLI overrides with tyro

The second part is already tyro-native. The first part is implemented with a generic recursive hydrator that reconstructs nested dataclasses, tuples, unions, lists, dicts, and `VisionEncoderConfig` subtypes from raw YAML mappings.

That design has two problems:

- the train entrypoint owns configuration semantics that should live in the type layer
- polymorphic config behavior is encoded as string dispatch against base config placeholders rather than explicit typed variants

This change is intentionally future-facing. We do not need to preserve every current config shape. The goal is to make the training config model easier to extend over time, even if that requires cleaning up field types and CLI behavior now.

There is one important library boundary to respect: tyro is strong at typed CLI parsing and at applying CLI overrides to an already-constructed typed default object, but it is not intended to be a full configuration-file framework. That means a small file-loading step will still exist, but it should become a narrow adapter that produces typed defaults rather than a generic reflection-based deserializer.

## Goals / Non-Goals

**Goals:**

- Make the type layer the source of truth for training configuration structure and polymorphism.
- Remove generic recursive config hydration logic from `scripts/train.py`.
- Keep the `--config` plus CLI-override workflow for training runs.
- Make polymorphic config fields explicit and statically visible through concrete config dataclasses and union types.
- Ensure future config changes primarily require type updates, docs updates, and tests, not new entry-script deserialization rules.

**Non-Goals:**

- Preserve legacy flat config formats or every current YAML quirk.
- Introduce a new configuration framework beyond tyro.
- Convert every registry-backed config in the repository to unions in one step if the field is not relevant to the train config path.
- Design a generic serialization layer for all dataclasses in the repo.

## Decisions

### 1. Keep file loading separate from tyro, but shrink it to a thin typed-defaults adapter

The train CLI will continue to accept `--config <path>`, but the path-loading step will no longer try to generically emulate tyro's parsing behavior. Instead, it will load raw YAML and convert only the supported `TrainConfig` structure into a typed default object that is then passed to `tyro.cli(..., default=...)`.

Rationale:

- This matches tyro's intended design boundary: typed CLI parsing plus defaults loaded from elsewhere.
- It avoids relying on deprecated tyro YAML serialization helpers.
- It removes the need for a generic reflection-heavy hydrator in the entry script.

Alternatives considered:

- Use `tyro.extras.from_yaml()` directly. Rejected because it is deprecated and designed for tyro-generated tagged YAML rather than the repo's normal config files.
- Keep the current generic `_build_dataclass()` path and only simplify it. Rejected because it preserves the same architectural problem: config semantics remain hidden in the entry script.

### 2. Represent polymorphic config fields as explicit unions of concrete dataclasses

Polymorphic fields inside `TrainConfig` should be typed as unions of concrete config dataclasses instead of a base dataclass plus a `type: str` field that is later reinterpreted by custom loader logic.

For the immediate design target, this applies most clearly to `policy.vision_encoder`. The future-facing model is:

- introduce an explicit concrete config for the default branch (`from_backbone`)
- keep explicit concrete configs for `simple_patch`, `multi_tower`, and any future registry-backed variants
- annotate the field with a union over those concrete types
- use tyro subcommand metadata to keep CLI branch names aligned with config `type` values

Rationale:

- The set of supported variants becomes visible in the type system.
- tyro can reason about the field natively instead of depending on a custom subtype resolver.
- Nested polymorphism such as `multi_tower.towers` can follow the same pattern recursively.
- Registry-backed construction remains compatible because runtime config objects still carry a stable `type` key.

Alternatives considered:

- Keep the base-class-plus-string-dispatch model. Rejected because it requires custom hydration and obscures supported variants.
- Hide polymorphism behind tyro custom constructors. Rejected because it would still move subtype logic into parsing glue rather than making variants first-class types.

### 3. Use explicit branch naming for tyro unions and accept CLI cleanup as part of the new contract

When polymorphic fields are converted to unions, we will use tyro subcommand annotations to give each branch a stable, human-authored name that matches the config `type` string. This keeps YAML and CLI terminology aligned.

We will not optimize for preserving today's ad hoc CLI shape for polymorphic selection. If union-typed fields introduce a different tyro-native selection syntax, that is an acceptable part of the new contract.

Rationale:

- Stable branch names reduce ambiguity between YAML examples, docs, and CLI help output.
- Future extension becomes straightforward: adding a new branch means adding a new dataclass and union member, not updating hidden loader rules.
- Avoiding compatibility shims keeps the design simpler while usage is still limited.

Alternatives considered:

- Add tyro markers such as `AvoidSubcommands` globally to mimic the current no-subcommand feel. Rejected because it reduces expressiveness for switching variants and hides the union structure we are explicitly trying to surface.
- Preserve `--policy.vision-encoder.type ...` style selection indefinitely. Rejected because it requires special parsing behavior that tyro does not provide naturally.

### 4. Scope the refactor around the training config path, but establish a reusable pattern

The implementation should focus on the config tree reachable from `TrainConfig` and on fields that currently require custom reconstruction. It does not need to convert unrelated registry-driven configs everywhere in the repository in the same change.

Rationale:

- This keeps the first refactor bounded and testable.
- It still establishes the repository pattern for future config work: concrete dataclasses, explicit unions where polymorphism is real, and thin adapters at file boundaries.

Alternatives considered:

- Convert all config registries in one pass. Rejected because it increases risk and dilutes the change.
- Limit the refactor strictly to `scripts/train.py` without touching field types. Rejected because it would not solve the underlying design problem.

## Risks / Trade-offs

- **[CLI syntax changes for polymorphic fields]** → Mitigation: update docs, config examples, and CLI tests in the same change so the new contract is explicit and discoverable.
- **[Partial refactor leaves mixed config patterns in the repo]** → Mitigation: document the new pattern in this change and apply it first to the train path, where the maintenance pain already exists.
- **[A thin YAML adapter can grow back into a generic deserializer]** → Mitigation: keep loader code explicitly scoped to `TrainConfig` and supported polymorphic leaves; avoid recursive generic helpers.
- **[Nested union fields such as multi-tower vision configs are easy to get wrong]** → Mitigation: add golden tests for YAML defaults and CLI overrides that exercise nested variant selection and nested field overrides.
- **[Shared model config types affect code outside the train script]** → Mitigation: preserve stable runtime `type` values and update builders/registries only at typed boundaries, not by changing registry semantics.

## Migration Plan

1. Introduce the new typed config shapes for train-relevant polymorphic fields, including explicit concrete configs for default branches that are currently implicit.
2. Update `PolicyConfig` and related config annotations to use union-typed variants where the train path needs them.
3. Replace the current generic YAML hydration flow in `scripts/train.py` with a thin loader that builds a typed `TrainConfig` default object.
4. Update train config examples and training docs to the new schema and CLI contract.
5. Replace loader-focused tests with contract tests that validate:
   - YAML defaults load into the intended typed config
   - CLI overrides still apply through tyro
   - nested polymorphic configs behave correctly
6. Remove the obsolete generic helper functions from the train entry script.

Rollback strategy:

- This is a source-level CLI/config refactor rather than a deployed service migration.
- If the new contract proves unworkable during implementation, rollback is a normal code revert of the new typed config shapes and train loader path.

## Open Questions

- Should this change convert only `vision_encoder`, or also any other train-reachable config fields that currently rely on similar hidden subtype reconstruction?
- Do we want the initial implementation to keep tyro's default nested subcommand UX, or should we add targeted ergonomic markers after the contract tests are in place?
