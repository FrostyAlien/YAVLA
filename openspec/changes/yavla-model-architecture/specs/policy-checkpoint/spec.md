## ADDED Requirements

### Requirement: save_pretrained
`VLAPolicy.save_pretrained(path)` SHALL write `config.json` (with `config_version`), `model.safetensors`, `action_stats.json`, and `embodiment.json`.

#### Scenario: Round-trip serialization
- **WHEN** `policy.save_pretrained(tmp)` then `VLAPolicy.from_pretrained(tmp)`
- **THEN** loaded policy SHALL produce identical output (within float tolerance)

#### Scenario: PEFT checkpoint
- **WHEN** policy has LoRA adapters and `save_pretrained` is called
- **THEN** adapter weights SHALL be saved separately in `adapter_model.safetensors` alongside base weights

### Requirement: from_pretrained with validation
`VLAPolicy.from_pretrained(path)` SHALL load config, validate embodiment compatibility, reconstruct modules via registry, and load weights.

#### Scenario: Embodiment mismatch
- **WHEN** checkpoint has `action_dim=7` but current config has `action_dim=6`
- **THEN** `ValueError` SHALL be raised unless `strict=False`

### Requirement: Config version migration
When `config.json` has an older `config_version`, `from_pretrained` SHALL apply registered migration functions to upgrade the config to the current version.

#### Scenario: Version upgrade
- **WHEN** checkpoint has `config_version="1.0"` and current code expects `"2.0"`
- **THEN** migration `v1_to_v2` SHALL be applied automatically
