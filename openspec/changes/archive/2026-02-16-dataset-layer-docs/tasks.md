## 1. Documentation Structure

- [x] 1.1 Create a dataset-layer docs folder with an entry page that introduces scope and audience.
- [x] 1.2 Add dedicated pages for architecture overview, backend decision guide, and usage/how-to content.

## 2. Core Content Authoring

- [x] 2.1 Document dataset-layer architecture and backend roles (`default`, `lazy`, `streaming`) within `create_dataloader()`.
- [x] 2.2 Document key decisions and rationale, including auto backend selection, `SC-001`, right-biased frame-index lookup semantics, and streaming temporal-feature guardrails.
- [x] 2.3 Add actionable configuration and integration examples covering `auto`, `lazy`, and `streaming`, plus epoch handoff guidance.
- [x] 2.4 Add a dedicated "must know" section for streaming limitations, approximate shuffle behavior, and decoder/cache caveats.

## 3. Discoverability And Spec Traceability

- [x] 3.1 Add direct navigation from docs index/landing pages to the dataset-layer docs entrypoint.
- [x] 3.2 Add a "Normative references" section linking to `openspec/specs/lazy-dataset/spec.md`, `openspec/specs/streaming-dataset/spec.md`, `openspec/specs/dataset-factory/spec.md`, and `openspec/specs/data-transforms/spec.md`.

## 4. Consistency Review

- [x] 4.1 Review new docs for terminology consistency with existing v1 behavior and archived dataset-layer design artifacts.
- [x] 4.2 Verify all newly added internal links resolve correctly.
