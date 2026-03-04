## ADDED Requirements

### Requirement: Episode metadata container compatibility for shard discovery
`ShardInterleavedDataset` SHALL normalize `meta.episodes` records from supported container types without relying on container-specific `to_dict(orient=...)` behavior.

#### Scenario: HF Dataset metadata records are supported
- **WHEN** `meta.episodes` is a HuggingFace `datasets.Dataset`
- **THEN** shard discovery SHALL build shard paths and episode media references without calling `Dataset.to_dict(orient="records")`

#### Scenario: Pandas and list-backed metadata records are supported
- **WHEN** `meta.episodes` is a pandas DataFrame or list-like record collection
- **THEN** shard discovery SHALL normalize those records and discover the correct shard path set

### Requirement: Dual-path media decode source resolution in streaming backend
For media keys, `ShardInterleavedDataset` SHALL decode from row payloads when present and SHALL fall back to canonical LeRobot v3 episode metadata when row payloads are absent.

#### Scenario: Row payload media path uses timestamp fallback order
- **WHEN** a streaming row contains a media payload path but omits an embedded payload timestamp
- **THEN** streaming decode SHALL derive timestamp in this order: row payload timestamp, sample timestamp, sample `frame_index / fps`

#### Scenario: Canonical episode metadata media path is used when row payload is absent
- **WHEN** a streaming row does not include a media key payload and episode metadata includes `videos/{media_key}/chunk_index`, `videos/{media_key}/file_index`, and `videos/{media_key}/from_timestamp`
- **THEN** streaming decode SHALL resolve media path via `video_path` template and decode at `from_timestamp + sample_timestamp`
