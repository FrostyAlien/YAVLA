# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YAVLA (Yet Another VLA) is a modular Vision-Language-Action framework for robotic manipulation. It supports multiple VLA policy types through a composable 7-module pipeline with swappable components.

## Commands

Package manager: [pixi](https://pixi.sh) (integrates uv). Dev tools live in the `dev` environment.

```bash
# Setup (skip Git LFS — lerobot bundles large test artifacts we don't need)
GIT_LFS_SKIP_SMUDGE=1 pixi install -e dev

# Test / lint / format / typecheck
pixi run -e dev test                          # pytest tests/ -v (excludes integration by default)
pixi run -e dev lint                          # ruff check src/
pixi run -e dev format                        # ruff format src/ tests/
pixi run -e dev typecheck                     # mypy src/yavla/ (strict mode)

# Run a single test
pixi run -e dev pytest tests/test_foo.py::test_bar -v

# Include integration tests (require external data)
pixi run -e dev pytest tests/ -v -m ""

# Train / evaluate
pixi run train
pixi run evaluate
```

## Workflow

OpenSpec is the default workflow for all changes: research → plan → implement → review. Use the `/opsx:*` or `/openspec-*` slash commands to drive the workflow. Specs and change artifacts live in `openspec/`.

## Architecture

### 7-Module Pipeline

`VLAPolicy` (src/yavla/models/policy.py) composes modules in a linear pipeline with overridable steps:

```
ObservationBatch (images, proprio, language)
  → encode_observations()  — VisionEncoder + ProprioEncoder
  → merge_tokens()         — TokenMerger (vision + proprio + language + readout tokens)
  → run_backbone()         — VLM Backbone (+ optional LoRA)
  → compute_loss()         — ActionHead (training) / decode_prediction() (inference)
  → LossDict / ActionChunk
```

Subclasses can override pipeline steps for different VLA paradigms: For example, AR-token policies override `merge_tokens` + `compute_loss`; flow-matching policies override `forward` for multi-step denoising; diffusion policies add noise scheduling, etc.

`validate_integration()` checks backbone↔head compatibility at construction time using capability/requirement negotiation.

### Key Abstractions (src/yavla/models/)

- **protocols.py** — ABC base classes (`PolicyBase`, `VisionEncoderBase`, `BackboneBase`, `ActionHeadBase`, `TokenMergerBase`, `ActionDecoderBase`) with `BackboneCapabilities` / `ActionHeadRequirements` for integration negotiation
- **registry.py** — Generic `Registry[ConfigT, ModuleT]` for config-driven component instantiation; new components register by name, built via `registry.build(config)`
- **vlm_registry.py** — `VLMRegistry` mapping `BackboneConfig.type` → builder returning `(VisionEncoderBase, BackboneBase)` pairs; dispatches by architecture, not variant
- **backbones/** — VLM-specific backbone + vision encoder implementations (e.g., `backbones/paligemma.py` contains `PaliGemmaBackbone`, `PaliGemmaVisionEncoder`, and the `build_paligemma_vlm` builder)
- **config.py** — `PolicyConfig` dataclass tree composing sub-configs; forward-compatible deserialization drops unknown keys
- **types.py** — Typed dataclasses at module boundaries: `ObservationBatch`, `BackboneOutput`, `ActionPrediction`, `ActionChunk`, `TrainingBatch`, `LossDict`, `ActionSpaceSpec`, `ProprioSpec`
- **policy.py** — `VLAPolicy`, `build_policy()` factory, `save_pretrained()`/`from_pretrained()` serialization

### Adding New VLM Backbones

1. Create `backbones/<vlm_name>.py` implementing `BackboneBase` and `VisionEncoderBase`
2. Implement a builder function returning `(VisionEncoderBase, BackboneBase)`
3. Register with `@vlm_registry.register("<type>")`
4. Model variants are selected via `BackboneConfig.vlm_name` (HF model ID) — no extra code per variant

### Adding Other Components

1. Implement the appropriate base class from protocols.py
2. Define a config dataclass with a `type` field
3. Register with the corresponding registry (e.g., `@head_registry.register("flow_matching")`)
4. `build_policy()` factory automatically composes and validates

### Other Packages

- **src/yavla/data/** — dataset loading (LeRobot/HuggingFace datasets), transforms, streaming, schema
- **src/yavla/training/** — training data utilities
- **scripts/** — entry points: train.py, evaluate.py, bench_dataloader.py, browse_dataset.py
- **docs/architecture/** — research docs covering the VLA design space (action paradigms, vision encoders, fusion strategies)
- **configs/** — YAML training configs (used with tyro CLI overrides)

## Conventions

- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
- Type hints on all public APIs; Google-style docstrings
- Config-driven training: YAML configs + CLI overrides via tyro
- Reproducible experiments: seed everything, log all hyperparams to wandb
- Checkpoints use safetensors (no pickle), with embodiment specs stored alongside weights

## Code Style

- Line length: 120
- Ruff rules: E, F, I, N, W, UP
- mypy strict mode — all code must pass strict type checking
- Pyright standard mode also configured (pyrightconfig.json)
- Test marker: `@pytest.mark.integration` for tests requiring external data

## Available MCP Servers

- **codex** — GPT-5.3-codex model for code generation/analysis and peer review, have build-in grok-search MCP support
- **grok-search** — web search + page scraping (see Grok Search policy below)
- **pdf-reader** — PDF text/table extraction (useful for reading papers)

## Grok Search Policy

### Activation & Routing

- Use **native web tools first** for simple and quick lookups.
- **Escalate to Grok Search** when any of these conditions apply:
  - Deep research, lateral comparison, or long-form content extraction is needed
  - Native web results are conflicting or insufficient
  - Platform-specific search is required (GitHub / Reddit / X / etc.)
  - High-stakes conclusions need multi-source cross-verification
- **Final verification**: critical conclusions must always be traced back to official/primary sources.

### Model Selection (mandatory)

Before every `web_search` or `web_fetch` call, run `switch_model` to select the appropriate model. Default to `grok-4.20-beta` when the task type is unclear.

| Scenario              | Model                 | Triggering Conditions                                                                                  |
| --------------------- | --------------------- | ------------------------------------------------------------------------------------------------------ |
| Quick lookup          | `grok-4.1-fast`       | Simple fact queries, single questions, batch searches, real-time news, **speed priority**              |
| Daily search          | `grok-4.20-beta`      | General technical searches, Chinese information, solution comparison, information collection, daily dev |
| Large document fetch  | `grok-4.1-expert`     | Fetch large pages, complete content extraction, **structured reports**, strict instruction compliance  |
| Deep research         | `grok-4.1-thinking`   | Academic research, complex reasoning, **multi-source cross-verification**, opinion synthesis, highest credibility |

### Execution Strategy

- **Query construction**: `web_search` for breadth, `web_fetch` for depth; set `platform` parameter for platform-specific searches.
- **Search execution**: start with summaries → fetch full content for key URLs → if results are insufficient, refine the query and retry (never give up after a single attempt).
- **Result integration**: cross-verify across sources + **mandatory source citation** `[Title](URL)` + annotate dates for time-sensitive information.

### Error Recovery

- Connection failure → run `get_config_info` to diagnose
- No results → broaden or rephrase the query
- Timeout → search alternative sources

### Core Constraints

- All search output **must include source citations** — no unsourced claims
- Never abandon a search after a single failed attempt — retry with adjusted queries
- Never present unverified assumptions as facts
- When results conflict, assess **source credibility** (official docs > peer-reviewed papers > blog posts > forum answers) and flag unresolved discrepancies to the user

## AI Development Rules

- Read existing code before proposing changes; understand the module boundaries and typed contracts
- Respect the registry pattern — never hardcode component choices; all components must be config-driven and registry-registered
- Maintain protocol/ABC contracts — new components must implement the correct base class and pass `validate_integration()`
- Keep tensor shapes documented in docstrings: `[B, T, D]` format
- Use the typed dataclasses (`ObservationBatch`, `BackboneOutput`, etc.) at module boundaries, not raw tensor dicts
- When adding a new action head or encoder, also add its config dataclass and registry entry
- Run `pixi run -e dev lint` and `pixi run -e dev typecheck` before committing — strict mypy must pass
- Prefer minimal, focused changes; don't refactor surrounding code unless asked
- Research docs in docs/architecture/ are the source of truth for design decisions — consult them before making architectural choices. If conflicts with existing code or your deep research result, ask for clarification.
