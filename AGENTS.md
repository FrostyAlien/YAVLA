# Repository Guidelines

## Project Overview

YAVLA (Yet Another VLA) is a modular Vision-Language-Action framework for robotic manipulation. It supports multiple VLA policy types through a composable 7-module pipeline with swappable components.

## Project Structure & Module Organization

- `src/yavla/`: main Python package.
  - `models/`: modular VLA policy pipeline (`VLAPolicy`), registries, protocols, typed boundary dataclasses.
  - `data/`: dataset loading/transforms (LeRobot + HuggingFace datasets).
  - `training/`: training utilities.
- `scripts/`: runnable entry points (`train.py`, `evaluate.py`, dataset/benchmark helpers).
- `configs/`: YAML configs for training runs (used with `tyro` CLI overrides).
- `tests/`: `pytest` suite; integration tests live in `tests/integration/`.
- `docs/`: architecture + usage docs; `openspec/`: specs and change artifacts (preferred workflow for non-trivial changes).

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

### Adding New Components

1. Implement the appropriate base class from protocols.py
2. Define a config dataclass with a `type` field
3. Register with the corresponding registry (e.g., `@head_registry.register("flow_matching")`)
4. `build_policy()` factory automatically composes and validates

## Build, Test, and Development Commands

Prereq: `pixi` (manages Python 3.12 + deps). If you don’t need Git LFS artifacts:

```bash
GIT_LFS_SKIP_SMUDGE=1 pixi install -e dev
pixi run -e dev test       # pytest tests/ -v (integration excluded by default)
pixi run -e dev lint       # ruff check src/
pixi run -e dev format     # ruff format src/ tests/
pixi run -e dev typecheck  # mypy src/yavla/ (strict)
pixi run train             # python scripts/train.py
pixi run evaluate          # python scripts/evaluate.py
```

## Coding Style & Naming Conventions

- Python: 4-space indentation; line length 120.
- Formatting/linting: `ruff` (including import sorting); prefer `ruff format` over manual formatting.
- Typing: `mypy` runs in strict mode; keep type hints on public APIs and use the typed dataclasses in `src/yavla/models/types.py` at module boundaries.

## Testing Guidelines

- Framework: `pytest` (tests live under `tests/`).
- Fast path: `pixi run -e dev test` runs unit tests; integration tests are marked `@pytest.mark.integration` and excluded by default.
- Run specific tests locally:

```bash
pixi run -e dev pytest tests/models/test_policy.py -v
pixi run -e dev pytest tests/ -v -m ""  # include integration tests (requires external data)
```

## Commit & Pull Request Guidelines

- Commits follow Conventional Commits in this repo: `feat: ...`, `fix: ...`, `docs(scope): ...`, `refactor: ...`, `perf: ...`, `test: ...`, `chore: ...`.
- PRs should include: a short problem statement, the commands you ran (e.g., `pixi run -e dev test`), and any config changes (new/updated files under `configs/`). Add screenshots only for user-facing docs/tools.

## Configuration & Security Tips

- Don’t commit credentials (e.g., `WANDB_API_KEY`, HuggingFace tokens); use environment variables.
- Avoid committing large binary artifacts (datasets, checkpoints); prefer reproducible configs under `configs/` and documented download steps.

## Available MCP Servers

- **kindly-web-search** — web search + page scraping for looking up docs, issues, and references
- **pdf-reader** — PDF text/table extraction (useful for reading papers)

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
