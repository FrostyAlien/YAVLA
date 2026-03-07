## ADDED Requirements

### Requirement: Dataset-layer docs explain embodiment-exact tensors for pretrained VLAs
Dataset-layer documentation SHALL explain that datasets remain embodiment-exact even when a policy uses pretrained-VLA embodiment adaptation. The docs SHALL state that action and proprio tensors MUST NOT be manually padded to model maximum width in the dataset layer, and SHALL clarify that padding and dimension masking happen inside the model and training pipeline.

#### Scenario: Reader prepares a smaller-embodiment dataset
- **WHEN** a user reads dataset-layer docs for a robot with 14 active action dimensions and a pretrained model width of 32
- **THEN** the docs SHALL tell them to keep dataset actions at shape `[B, chunk_len, 14]` and proprio at shape `[B, 14]`

#### Scenario: Reader compares exact and pretrained-VLA paths
- **WHEN** a user reviews dataset-layer behavior for training configuration choices
- **THEN** the docs SHALL explain that exact-dimension MVP training expects dataset and model dimensions to match, while pretrained-VLA adaptation pads and masks only after the dataset boundary
