## ADDED Requirements

### Requirement: Load subsampled LeRobot frames into FiftyOne
The system SHALL provide a `load_lerobot_to_fiftyone` function that reads a LeRobot dataset, takes every Nth frame per episode (uniform temporal subsampling), saves decoded images to disk as JPEG, and creates a FiftyOne dataset with typed metadata fields.

#### Scenario: Basic dataset loading with explicit schema
- **WHEN** `load_lerobot_to_fiftyone(repo_id="lerobot/pusht", subsample_rate=10)` is called
- **THEN** the function SHALL create a FiftyOne dataset where each sample has:
  - `filepath`: `str` — path to saved JPEG (e.g., `{output_dir}/{dataset_name}/images/ep000001_frame00000010.jpg`)
  - `episode_index`: `fo.IntField` — episode number from LeRobot
  - `frame_index`: `fo.IntField` — original frame index within the episode
  - `timestamp`: `fo.FloatField` — timestamp in seconds from LeRobot metadata
  - `task`: `fo.StringField` — task description string
  - `action`: `fo.VectorField` — action array as float list
  - `camera_key`: `fo.StringField` — which camera produced this frame (e.g., `"observation.image"`)

#### Scenario: Uniform per-episode subsampling
- **WHEN** a dataset with 10 episodes of 1,000 frames each is loaded with `subsample_rate=10`
- **THEN** the function SHALL take frames at indices 0, 10, 20, ... from EACH episode, resulting in 100 samples per episode, 1,000 total — preserving temporal coverage across all episodes

#### Scenario: JPEG saving details
- **WHEN** frames are saved to disk
- **THEN** the function SHALL:
  - Save to `{output_dir}/{dataset_name}/images/ep{episode_index:06d}_frame{frame_index:08d}.jpg`
  - Use JPEG quality 95 via PIL (RGB, no BGR conversion)
  - Skip re-encoding if file already exists with matching dimensions (idempotent re-runs)
  - At subsample_rate=10 on 1M frames, expect ~100K JPEGs ≈ 5-10GB disk

#### Scenario: Ephemeral dataset for debugging
- **WHEN** `persistent=False` is passed
- **THEN** the FiftyOne dataset SHALL NOT persist in MongoDB after the Python session ends

### Requirement: Add precomputed embeddings for UMAP visualization
The system SHALL provide an `add_embeddings_to_dataset` function that takes a FiftyOne dataset and a numpy array of embeddings, optionally applies PCA pre-reduction, and calls `fob.compute_visualization` to enable the Embeddings panel in the FiftyOne App.

#### Scenario: UMAP with PCA pre-reduction
- **WHEN** `add_embeddings_to_dataset(dataset, embeddings, pca_dims=50, method="umap")` is called with embeddings of shape `(N, 768)`
- **THEN** the function SHALL reduce to 50 dims via `sklearn.decomposition.PCA` before computing UMAP, and the FiftyOne App SHALL display an interactive scatter plot

#### Scenario: Embedding source and shape
- **WHEN** embeddings are computed for the dataset
- **THEN** they SHALL be **pooled** (mean over patch tokens) from the vision encoder target layer, shape `(N_samples, hidden_dim)` — not per-patch. Computed in batches of 64 via `extract_layer_embeddings()` then mean-pooled.

#### Scenario: Embeddings must be numpy arrays
- **WHEN** a torch.Tensor is passed as embeddings
- **THEN** the function SHALL convert it to numpy via `.cpu().numpy()` before passing to FiftyOne Brain

#### Scenario: Embedding count matches sample count
- **WHEN** the embeddings array length does not match the dataset sample count
- **THEN** the function SHALL raise `ValueError` with the mismatched counts

#### Scenario: Embedding caching
- **WHEN** embeddings are computed
- **THEN** they SHALL be saved to `{output_dir}/{dataset_name}/embeddings_{brain_key}.npy` for reuse without re-computing. If the cache file exists and has matching shape, load from cache instead.

### Requirement: Bulk sample addition for performance
The FiftyOne loader SHALL use `dataset.add_samples(list)` for bulk insertion rather than per-sample `add_sample()` calls.

#### Scenario: Large dataset loading performance
- **WHEN** loading 100,000 subsampled frames
- **THEN** the function SHALL add all samples in a single `add_samples` call (or batched calls of 10K), not one-by-one

### Requirement: Testability
Unit tests SHALL mock `fiftyone` and test subsampling logic, file naming, and schema construction as pure functions. Integration tests requiring FiftyOne (MongoDB) SHALL be marked `@pytest.mark.integration` and skipped in minimal CI.
