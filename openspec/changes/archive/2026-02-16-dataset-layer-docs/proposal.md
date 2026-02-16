## Why

The dataset-layer v1 implementation is complete, but practical onboarding still requires reading multiple specs and source files to understand backend selection, constraints, and usage. We need a single authoritative documentation surface so contributors and training users can apply the dataset layer correctly without reverse-engineering design artifacts.

## What Changes

- Add a dedicated dataset-layer documentation package that explains the most important design decisions, implementation shape, and usage patterns for `default`, `lazy`, and `streaming` backends.
- Document factory behavior (`auto` selection, explicit backend behavior, `SC-001`, and streaming temporal-feature guardrails) with concrete examples and decision guidance.
- Add a “must know” section covering operational constraints and pitfalls (shuffle quality tradeoffs, unsupported streaming features, decoder/cache implications, and distributed nuances).
- Add traceable links from docs to OpenSpec specs/design so readers can move from high-level guidance to normative requirements.
- Update documentation navigation/index so this dataset-layer docs package is discoverable from the docs surface.

## Capabilities

### New Capabilities
- `dataset-layer-documentation`: User-facing and contributor-facing documentation for dataset-layer architecture, backend behavior, configuration, usage examples, operational caveats, and links to normative specs.

### Modified Capabilities
- _(none)_

## Impact

- **Documentation**: New docs pages for dataset-layer overview, backend decision guide, configuration reference, usage examples, and “must know” caveats.
- **Navigation**: Docs navigation/index updates to point to dataset-layer docs and related OpenSpec specs.
- **No runtime behavior changes**: No changes to dataset APIs, training behavior, dependencies, or data-processing code paths.
