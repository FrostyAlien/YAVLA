"""FiftyOne dataset loader for LeRobot datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import fiftyone as fo  # type: ignore[import-untyped]
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-untyped]
from PIL import Image

LOGGER = logging.getLogger(__name__)

_BATCH_SIZE = 2000


def _build_task_lookup(meta: Any) -> dict[int, str]:
    return {v: k for k, v in meta.tasks["task_index"].items()}


def _save_jpeg(image_tensor: Any, path: Path) -> None:
    """Save (C,H,W) float [0,1] tensor as JPEG, skip if exists with matching dims."""
    if path.exists():
        try:
            with Image.open(path) as existing:
                h, w = image_tensor.shape[1], image_tensor.shape[2]
                if existing.size == (w, h):
                    return
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path, quality=95)


def load_lerobot_to_fiftyone(
    repo_id: str,
    *,
    root: str | Path | None = None,
    subsample_rate: int = 10,
    output_dir: str | Path = "/tmp/yavla_fiftyone",
    dataset_name: str | None = None,
    persistent: bool = False,
) -> fo.Dataset:
    """Load subsampled LeRobot frames into a FiftyOne dataset.

    Args:
        repo_id: HuggingFace repo ID for the LeRobot dataset.
        root: Local root directory for cached dataset files.
        subsample_rate: Take every Nth frame per episode.
        output_dir: Directory for saved JPEG frames.
        dataset_name: FiftyOne dataset name. Defaults to repo_id slug.
        persistent: Whether the FiftyOne dataset persists in MongoDB.

    Returns:
        FiftyOne dataset with typed metadata fields.
    """
    lr = LeRobotDataset(repo_id, root=root)
    meta = lr.meta
    task_lookup = _build_task_lookup(meta)
    camera_keys = list(meta.camera_keys) if hasattr(meta, "camera_keys") else []

    name = dataset_name or repo_id.replace("/", "_")
    output = Path(output_dir) / name / "images"

    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    dataset = fo.Dataset(name=name, persistent=persistent)
    dataset.add_sample_field("episode_index", fo.IntField)
    dataset.add_sample_field("frame_index", fo.IntField)
    dataset.add_sample_field("timestamp", fo.FloatField)
    dataset.add_sample_field("task", fo.StringField)
    dataset.add_sample_field("action", fo.ListField, subfield=fo.FloatField)
    dataset.add_sample_field("camera_key", fo.StringField)

    samples: list[fo.Sample] = []
    total_episodes = meta.total_episodes if hasattr(meta, "total_episodes") else lr.num_episodes

    for ep_idx in range(total_episodes):
        from_idx = int(meta.episodes["dataset_from_index"][ep_idx])
        to_idx = int(meta.episodes["dataset_to_index"][ep_idx])

        for abs_idx in range(from_idx, to_idx, subsample_rate):
            frame = lr[abs_idx]
            cam_key = camera_keys[0] if camera_keys else ""
            frame_idx = int(frame.get("index", abs_idx))

            img_path = output / f"ep{ep_idx:06d}_frame{frame_idx:08d}.jpg"
            if cam_key and cam_key in frame:
                _save_jpeg(frame[cam_key], img_path)
            else:
                img_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (224, 224)).save(img_path, quality=95)

            task_idx = int(frame.get("task_index", 0))
            action = frame["action"].tolist() if "action" in frame else []

            samples.append(
                fo.Sample(
                    filepath=str(img_path),
                    episode_index=ep_idx,
                    frame_index=frame_idx,
                    timestamp=float(frame.get("timestamp", 0.0)),
                    task=task_lookup.get(task_idx, ""),
                    action=action,
                    camera_key=cam_key,
                )
            )

            if len(samples) >= _BATCH_SIZE:
                dataset.add_samples(samples)
                samples = []

        LOGGER.info("Loaded episode %d/%d", ep_idx + 1, total_episodes)

    if samples:
        dataset.add_samples(samples)

    LOGGER.info("FiftyOne dataset '%s' created with %d samples", name, len(dataset))
    return dataset


def add_embeddings_to_dataset(
    dataset: fo.Dataset,
    embeddings: np.ndarray | Any,
    brain_key: str = "default_vis",
    method: str = "umap",
    pca_dims: int | None = 50,
    cache_dir: str | Path | None = None,
) -> None:
    """Add precomputed embeddings and run dimensionality reduction visualization.

    Args:
        dataset: FiftyOne dataset to add embeddings to.
        embeddings: Array of shape (N, D) where N == len(dataset).
        brain_key: Key for the FiftyOne Brain visualization run.
        method: Reduction method ('umap' or 'tsne').
        pca_dims: If set, apply PCA pre-reduction before UMAP/t-SNE.
        cache_dir: Directory for embedding cache files. If provided, saves to
            ``{cache_dir}/embeddings_{brain_key}.npy`` and loads from cache
            when shape matches.
    """
    import fiftyone.brain as fob  # type: ignore[import-untyped]

    if hasattr(embeddings, "cpu") and callable(embeddings.cpu):
        embeddings = embeddings.cpu().numpy()
    embeddings = np.asarray(embeddings)

    if len(embeddings) != int(len(dataset)):
        raise ValueError(f"Embedding count {len(embeddings)} != dataset sample count {len(dataset)}")

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"embeddings_{brain_key}.npy"
        if cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape == embeddings.shape:
                LOGGER.info("Loaded cached embeddings from %s", cache_path)
                embeddings = cached
            else:
                LOGGER.info("Cache shape mismatch (%s vs %s), recomputing", cached.shape, embeddings.shape)

    if pca_dims is not None and embeddings.shape[1] > pca_dims:
        try:
            from sklearn.decomposition import PCA  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "scikit-learn is required for PCA pre-reduction. Install with: pip install yavla[viz]"
            ) from None
        LOGGER.info("PCA: %d -> %d dims", embeddings.shape[1], pca_dims)
        embeddings = PCA(n_components=pca_dims).fit_transform(embeddings)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        LOGGER.info("Saved embeddings to %s", cache_path)

    fob.compute_visualization(
        dataset,
        embeddings=embeddings,
        method=method,
        brain_key=brain_key,
        num_dims=2,
    )
    LOGGER.info("Brain visualization '%s' computed via %s", brain_key, method)
