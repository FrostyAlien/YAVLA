# Dataset Layer

The dataset layer provides a unified data-loading interface for YAVLA training. It selects between three backends — `default`, `lazy`, and `streaming` — based on dataset size, locality, and feature requirements, then wires up transforms, sampling, and DataLoader configuration automatically.

Docs index: [Documentation Home](../README.md)

## Audience

- **Training users** who need to configure data loading for new datasets or adjust backend behavior.
- **Contributors** who need to understand architecture decisions, constraints, and implementation structure before modifying data-loading code.

## Pages

| Page | What it covers |
|------|---------------|
| [Architecture](architecture.md) | Backend model, `create_dataloader()` flow, and how `default` / `lazy` / `streaming` fit together. |
| [Backend Decision Guide](backend-guide.md) | How to choose a backend, auto-selection logic, `SC-001`, and streaming guardrails. |
| [Usage and Configuration](usage.md) | `DataConfig` fields, YAML examples, epoch handoff, and common recipes. |
| [Must-Know Caveats](caveats.md) | Streaming shuffle quality, unsupported temporal features, decoder/cache behavior, and DDP nuances. |

## Normative References

The documentation in this folder is explanatory. The normative (source-of-truth) requirements live in OpenSpec specs:

- [`openspec/specs/dataset-factory/spec.md`](../../openspec/specs/dataset-factory/spec.md) — factory, `DataConfig`, auto-selection, transform wiring
- [`openspec/specs/lazy-dataset/spec.md`](../../openspec/specs/lazy-dataset/spec.md) — lazy backend requirements
- [`openspec/specs/streaming-dataset/spec.md`](../../openspec/specs/streaming-dataset/spec.md) — streaming backend requirements
- [`openspec/specs/data-transforms/spec.md`](../../openspec/specs/data-transforms/spec.md) — transform pipeline protocol and built-in transforms

When docs and specs disagree, specs are authoritative.
