## ADDED Requirements

### Requirement: Episode metadata container compatibility for lazy indexing
`LazyLeRobotDataset` SHALL normalize `meta.episodes` records from supported container types without relying on container-specific `to_dict(orient=...)` behavior.

#### Scenario: HF Dataset metadata records are supported
- **WHEN** `meta.episodes` is provided as a HuggingFace `datasets.Dataset`
- **THEN** lazy index construction SHALL consume episode records successfully without calling `Dataset.to_dict(orient="records")`

#### Scenario: Pandas and list-backed metadata records are supported
- **WHEN** `meta.episodes` is provided as a pandas DataFrame or list-like record collection
- **THEN** lazy index construction SHALL normalize those records and build episode/file indexes correctly

### Requirement: Dual-path media decode source resolution in lazy backend
For media keys, `LazyLeRobotDataset` SHALL resolve frames from row payloads when present, and SHALL fall back to canonical LeRobot v3 episode metadata when row payloads are absent.

#### Scenario: Row payload media path uses timestamp fallback order
- **WHEN** a row contains a media payload path but omits an embedded payload timestamp
- **THEN** lazy decoding SHALL derive timestamp in this order: row payload timestamp, sample timestamp, sample `frame_index / fps`

#### Scenario: Canonical episode metadata media path is used when row payload is absent
- **WHEN** a row does not include a media key payload and episode metadata includes `videos/{media_key}/chunk_index`, `videos/{media_key}/file_index`, and `videos/{media_key}/from_timestamp`
- **THEN** lazy decoding SHALL resolve media path via `video_path` template and decode at `from_timestamp + sample_timestamp`

### Requirement: Compatibility behavior preserves lazy initialization memory goal
Metadata compatibility and media-path fallback behavior SHALL NOT require eager frame-level parquet loading during lazy dataset initialization.

#### Scenario: Initialization remains metadata-driven
- **WHEN** `LazyLeRobotDataset` is constructed for a valid LeRobot v3 dataset
- **THEN** initialization SHALL complete using metadata/index construction only, with frame parquet rows loaded on sample access
