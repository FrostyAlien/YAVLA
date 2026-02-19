## MODIFIED Requirements

### Requirement: Per-column Arrow batch conversion
`_read_shard_rows` SHALL convert Arrow batches using per-column conversion instead of `batch.to_pylist()`, selecting the optimal conversion method per column type.

#### Scenario: Primitive numeric columns use numpy conversion
- **WHEN** a batch contains primitive numeric columns (int64, float64)
- **THEN** each column SHALL be converted via `column.to_numpy(zero_copy_only=False)` and per-row values SHALL be numpy scalars obtained by indexing

#### Scenario: List/array columns use per-column to_pylist
- **WHEN** a batch contains list-type columns (e.g., action vectors stored as Arrow list arrays)
- **THEN** each such column SHALL be converted via `column.to_pylist()` at the column level, yielding Python lists per row

#### Scenario: String columns use per-column to_pylist
- **WHEN** a batch contains string columns (e.g., task names)
- **THEN** each such column SHALL be converted via `column.to_pylist()` at the column level

#### Scenario: Yielded dicts have same key set as before
- **WHEN** `_read_shard_rows` yields a sample dict
- **THEN** the dict SHALL contain the same keys as the previous `to_pylist()` implementation, with no keys added or removed

#### Scenario: Metadata columns produce numpy scalars
- **WHEN** a yielded sample contains metadata columns (episode_index, frame_index, timestamp, index, task_index)
- **THEN** those values SHALL be numpy scalars (`np.int64`, `np.float64`) which are accepted by `_to_tensor`, schema validation, and PyTorch's default collate
