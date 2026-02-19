# Dataset Layer

The dataset layer provides a unified data-loading interface for YAVLA training. It supports three backends — `default`, `lazy`, and `streaming` — and wires up transforms, sampling, and DataLoader configuration consistently across them.

Docs index: [Documentation Home](../README.md)

## Audience

- **Training users** who need to configure data loading for new datasets or adjust backend behavior.
- **Contributors** who need to understand architecture decisions, constraints, and implementation structure before modifying data-loading code.

## Pages

| Page | What it covers |
|------|---------------|
| [Architecture](architecture.md) | Backend model, `create_dataloader()` flow, and how `default` / `lazy` / `streaming` fit together. |
| [Backend Decision Guide](backend-guide.md) | How to choose a backend explicitly and understand backend guardrails. |
| [Usage and Configuration](usage.md) | `DataConfig` fields, YAML examples, epoch handoff, and common recipes. |
| [Must-Know Caveats](caveats.md) | Streaming shuffle quality, unsupported temporal features, media source resolution, decoder/cache behavior, and DDP nuances. |

## Normative References

The documentation in this folder is explanatory. The normative (source-of-truth) requirements live in OpenSpec specs:

- [`openspec/specs/dataset-factory/spec.md`](../../openspec/specs/dataset-factory/spec.md) — factory, `DataConfig`, backend guardrails, transform wiring
- [`openspec/specs/lazy-dataset/spec.md`](../../openspec/specs/lazy-dataset/spec.md) — lazy backend requirements
- [`openspec/specs/streaming-dataset/spec.md`](../../openspec/specs/streaming-dataset/spec.md) — streaming backend requirements
- [`openspec/specs/data-transforms/spec.md`](../../openspec/specs/data-transforms/spec.md) — transform pipeline protocol and built-in transforms
- [`openspec/specs/dataloader-benchmark/spec.md`](../../openspec/specs/dataloader-benchmark/spec.md) — dataloader performance benchmark

When docs and specs disagree, specs are authoritative.
