# Visualization

The visualization package provides tools for browsing and inspecting YAVLA datasets outside the training loop. Currently implements FiftyOne-based dataset browsing with subsampled frame export.

Docs index: [Documentation Home](../README.md)

## Audience

- **Data users** who want to visually inspect dataset quality, frame distribution, and episode structure before training.
- **Contributors** who need to understand the visualization architecture before adding attention overlays, Rerun logging, or new export formats.

## Pages

| Page | What it covers |
|------|---------------|
| [Architecture](architecture.md) | Package structure, data flow, and how FiftyOne loader converts LeRobot episodes to browsable samples. |
| [Usage and Configuration](usage.md) | `VizConfig` fields, `browse_dataset.py` CLI, Python API, and common recipes. |
| [Must-Know Caveats](caveats.md) | FiftyOne compatibility issues, Python version constraints, and known workarounds. |

## Current Scope

Only the FiftyOne dataset loader is implemented. The following modules are planned but deferred until model/training/eval code exists:

| Module | Status | Depends on |
|--------|--------|------------|
| `config.py` | Implemented | — |
| `fiftyone_loader.py` | Implemented | — |
| `attention.py` | Deferred | Model forward pass |
| `snapshot.py` | Deferred | Training loop |
| `rerun_logger.py` | Deferred | Eval pipeline |

## Normative References

- [`openspec/changes/add-visualization-toolkit/`](../../openspec/changes/add-visualization-toolkit/) — Full change artifacts (proposal, design, specs, tasks)
