"""Visualization configuration for YAVLA."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class VizConfig:
    """Central configuration for all visualization features.

    All features default to disabled — zero overhead when off.
    """

    # Training snapshots
    snapshot_enabled: bool = False
    snapshot_interval_steps: int = 10_000
    snapshot_num_samples: int = 4
    snapshot_methods: list[str] = field(default_factory=lambda: ["attention_rollout", "grad_cam"])
    snapshot_layers: list[str] = field(default_factory=lambda: ["last"])
    snapshot_seed: int = 42

    # Rerun (eval only)
    rerun_enabled: bool = False
    rerun_log_images: bool = True
    rerun_log_actions: bool = True
    rerun_log_attention: bool = False
    rerun_output_dir: str = "./rerun_logs"

    # FiftyOne (offline)
    fiftyone_subsample_rate: int = 10
    fiftyone_umap_pca_dims: int = 50
