## Context

`dataset-layer` v1 is implemented and validated, but practical usage still depends on reading scattered sources: OpenSpec specs, archived change artifacts, and code-level behavior. This increases onboarding cost, creates repeated clarification requests, and makes policy-level constraints (such as `SC-001` and streaming temporal-feature limits) easy to miss.

This change introduces documentation only. Runtime data loading behavior, APIs, and training semantics are already defined by existing specs (`lazy-dataset`, `streaming-dataset`, `dataset-factory`, `data-transforms`) and current implementation.

## Goals / Non-Goals

**Goals:**
- Provide a discoverable, end-to-end dataset-layer docs surface for both users and contributors.
- Explain the key design decisions behind backend selection and constraints in practical language.
- Provide clear usage guidance (what to configure, when to choose each backend, and what to avoid).
- Preserve a clear separation between explanatory docs and normative requirements by linking back to specs.
- Reduce support overhead by documenting “must know” operational caveats.

**Non-Goals:**
- Changing runtime backend behavior, selection logic, or data processing semantics.
- Replacing OpenSpec specs as the normative source of requirements.
- Introducing a new docs framework/tooling dependency as part of this change.
- Exhaustive API docs generation for every internal helper.

## Decisions

### D1: Create a dedicated docs hub for dataset-layer
**Choice:** Add a focused docs section (for example under `docs/dataset-layer/`) with a small set of intentionally scoped pages:
- Overview and architecture map
- Backend decision guide (`default` vs `lazy` vs `streaming`)
- Configuration and factory usage (`DataConfig`, `create_dataloader`, epoch handoff)
- Operational “must know” caveats and troubleshooting

**Why:** A single hub minimizes context switching and avoids forcing readers into specs/code for first-pass understanding.

**Alternatives considered:**
- Expanding a single top-level landing page: too dense and hard to maintain.
- Relying only on specs: good for requirements, poor for onboarding and usage narrative.

### D2: Keep normative/explanatory boundaries explicit
**Choice:** Each docs page includes a short “Normative references” section linking to relevant specs and (where useful) design decisions.

**Why:** Prevents docs from becoming a second, divergent specification while still being practical.

**Alternatives considered:**
- Copying requirements into docs directly: higher risk of drift.
- Avoiding links and forcing readers to search specs: poor usability.

### D3: Document backend selection as a decision flow + examples
**Choice:** Present backend behavior as a decision flow and worked examples covering:
- `auto` selection behavior
- `SC-001` distributed override
- explicit `streaming` guardrails for `delta_timestamps`/`action_chunk_size`

**Why:** Most integration mistakes happen at backend-selection time, not inside dataset internals.

**Alternatives considered:**
- Narrative-only explanation: less actionable and harder to debug.

### D4: Separate “How to use” from “How it works”
**Choice:** Structure docs into two tracks:
- User path: configuration + loader usage + common recipes
- Contributor path: architecture decisions, constraints, and implementation map

**Why:** Different audiences need different entry points; mixing them degrades clarity for both.

**Alternatives considered:**
- Single blended page: shorter initially, harder to navigate as the feature evolves.

### D5: Include explicit “must know” constraints and failure modes
**Choice:** Add a dedicated section for operational caveats, including:
- streaming randomness is approximate
- streaming does not support temporal/action features
- decoder/cache behavior and backend implications
- DDP and epoch-handshake expectations

**Why:** These are correctness-affecting concerns that users often miss when skimming.

**Alternatives considered:**
- Inline caveats only: too easy to overlook.

### D6: No runtime changes; docs-only integration points
**Choice:** This change updates documentation and docs navigation/index only.

**Why:** Keeps scope controlled and avoids conflating docs work with behavioral changes already delivered in v1.

**Alternatives considered:**
- Bundling refactors with docs: increases risk and review complexity.

## Risks / Trade-offs

- **[Docs drift from implementation/specs]** → Mitigation: add explicit spec links per section and reference constraints by identifier (e.g., `SC-001`).
- **[Too much depth overwhelms quick users]** → Mitigation: layered structure (quick-start path first, deep-dive sections later).
- **[Too little depth for contributors]** → Mitigation: include architecture map + implementation pointers + caveats page.
- **[Link rot as files move]** → Mitigation: prefer stable repo-relative links and include docs update checks in review checklist.
- **[Ambiguity around normative source]** → Mitigation: recurring “Normative references” callouts and explicit statement that specs are source of truth.

## Migration Plan

1. Add docs pages and write content aligned with current v1 behavior.
2. Add docs navigation/index links to the dataset-layer docs entrypoint.
3. Verify links to specs and existing config examples.
4. Review for language consistency with current code and guardrails.
5. Land as docs-only change.

Rollback strategy: remove docs pages and navigation links; no runtime rollback required.

## Open Questions

- Should this repo adopt a formal docs site generator in a separate change, or stay Markdown-first?
- Do we want a lightweight link-check/doc-lint step in CI for long-term docs health?
- Should future changes require a docs impact checklist when any dataset-layer behavior changes?
