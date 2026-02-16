## ADDED Requirements

### Requirement: Dataset-layer documentation is discoverable
The repository SHALL provide a dataset-layer documentation entrypoint that is discoverable from documentation navigation/index pages.

#### Scenario: Docs navigation links to dataset-layer docs
- **WHEN** a user starts from documentation navigation (for example a docs index/landing page)
- **THEN** they SHALL find a direct link to the dataset-layer documentation without needing to inspect OpenSpec change artifacts

#### Scenario: Top-level README stays non-authoritative for dataset-layer behavior docs
- **WHEN** dataset-layer documentation is updated
- **THEN** behavior-level details SHALL live in docs pages (for example under `docs/`) rather than requiring a dedicated implementation guide section in top-level `README.md`

### Requirement: Documentation explains architecture and key decisions
Dataset-layer documentation SHALL explain the core architecture and the most important design decisions behind backend behavior.

#### Scenario: Architecture section covers backend model
- **WHEN** a reader opens the dataset-layer architecture section
- **THEN** it SHALL describe the roles of `default`, `lazy`, and `streaming` backends and how they fit into `create_dataloader()`

#### Scenario: Decision section captures operationally important choices
- **WHEN** a reader reviews design decisions in the docs
- **THEN** it SHALL include rationale for explicit backend selection, right-biased shard index lookup semantics, and streaming temporal-feature guardrails

### Requirement: Documentation provides actionable usage guidance
Dataset-layer documentation SHALL include concrete usage guidance for configuration and training-loop integration.

#### Scenario: Configuration examples are provided
- **WHEN** a reader needs to configure data loading
- **THEN** docs SHALL include example `DataConfig`/YAML patterns for at least `default`, `lazy`, and `streaming` usage

#### Scenario: Epoch handoff guidance is provided
- **WHEN** a reader integrates dataset-layer with training epochs
- **THEN** docs SHALL explain how and when to call epoch propagation hooks (for samplers and streaming datasets)

### Requirement: Documentation includes must-know constraints and caveats
Dataset-layer documentation SHALL include a dedicated “must know” section with behavior that can affect correctness or performance.

#### Scenario: Streaming limitations are explicit
- **WHEN** a reader evaluates streaming backend suitability
- **THEN** docs SHALL explicitly state that streaming does not support `delta_timestamps` or `action_chunk_size`

#### Scenario: Randomness and resource caveats are explicit
- **WHEN** a reader reviews operational caveats
- **THEN** docs SHALL explain approximate shuffle behavior in streaming mode and include decoder/cache-related caveats relevant to video-heavy datasets

### Requirement: Documentation links to normative specs
Dataset-layer documentation SHALL reference normative OpenSpec specs as source-of-truth requirements.

#### Scenario: Normative references are provided
- **WHEN** a reader needs requirement-level detail
- **THEN** docs SHALL provide links to `openspec/specs/lazy-dataset/spec.md`, `openspec/specs/streaming-dataset/spec.md`, `openspec/specs/dataset-factory/spec.md`, and `openspec/specs/data-transforms/spec.md`
