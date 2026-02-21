## ADDED Requirements

<!--
Refs (serialization correctness + shared tensors):
- Safetensors “Torch shared tensors” explains why save_file(state_dict) is unsafe for shared/tied tensors and recommends save_model/load_model:
  https://raw.githubusercontent.com/huggingface/safetensors/main/docs/source/torch_shared_tensors.mdx
- OpenPI stores normalization stats alongside checkpoints and discusses reloading stats for finetuning:
  https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/docs/norm_stats.md
-->

### Requirement: Safe, self-contained checkpoint is the default
`VLAPolicy.save_pretrained(path)` SHALL write a **self-contained** checkpoint by default: loading the checkpoint SHALL
NOT require downloading a base model from the internet.

The checkpoint format SHALL be “safe serialization”:
- Model weights serialized with `safetensors` (no pickle).
- Metadata serialized as JSON.

#### Scenario: Offline load works
- **WHEN** a checkpoint is created in an environment with network access
- **THEN** `VLAPolicy.from_pretrained(path)` SHALL succeed in an offline environment (no base-model fetch required)

### Requirement: Shared tensors are handled correctly
Checkpoint saving/loading MUST be robust to models with shared/tied tensors (common in Transformers, e.g., tied token
embeddings and LM head).

Implementations SHALL use `safetensors.torch.save_model(model, ...)` / `safetensors.torch.load_model(model, ...)` (or an
equivalent shared-tensor-safe approach) rather than `save_file(model.state_dict(), ...)`.

#### Scenario: Shared-tensor-safe save is used
- **WHEN** `save_pretrained` writes model weights
- **THEN** it SHALL use a shared-tensor-safe method (e.g., `save_model`) rather than raw `save_file(state_dict)`

#### Scenario: Loader does not require exact key equality with state_dict
- **WHEN** a checkpoint is saved for a model that contains shared tensors
- **THEN** the on-disk tensor names MAY omit duplicate/shared keys, and loading SHALL still succeed without requiring `set(file_keys) == set(state_dict_keys)`

<!-- Caveat is documented in safetensors torch_shared_tensors.md. -->

### Requirement: Checkpoint metadata includes normalization + preprocessing contracts
Checkpoints SHALL persist enough metadata to reconstruct the policy’s IO contracts unambiguously:

- Action normalization mode and any required stats identifier(s).
- Vision preprocessing contract identifier(s) sufficient to reproduce `pixel_values` processing for the configured VLM.
- Embodiment metadata (see below).

#### Scenario: Metadata roundtrip is deterministic
- **WHEN** a policy is saved and later loaded
- **THEN** the loaded policy SHALL expose the same action normalization mode and vision preprocessing contract as the saved policy

### Requirement: Embodiment metadata is complete
The checkpoint SHALL include an embodiment metadata file (e.g., `embodiment.json`) that contains, at minimum:

- `ActionSpaceSpec` fields: `names`, `units`, `limits` (or explicit absence), plus `frame` and `control_mode`.
- `ProprioSpec` fields: `names`, `units`, and `limits` (or explicit absence).
- Control frequency metadata (`dt_hz`) and action chunk metadata if required for inference compatibility.

#### Scenario: Loading validates embodiment compatibility
- **WHEN** `from_pretrained(path, strict=True)` is called
- **THEN** it SHALL validate that checkpoint embodiment constraints (e.g., action_dim) are compatible with the loaded config and raise a clear error on mismatch

### Requirement: Adapter-only artifacts are optional exports (not the default)
When LoRA/PEFT is active, checkpointing SHALL support an optional adapter-only export path intended for sharing small
artifacts. This SHALL NOT replace the default self-contained checkpoint.

The adapter-only export SHALL include:
- The adapter weights in the PEFT-native format under an `adapter/` directory (or equivalent).
- Any non-VLM module weights required to reproduce the full policy behavior (e.g., proprio encoder, merger, action head).
- Metadata describing which base model is required to apply the adapter.

#### Scenario: Self-contained and adapter artifacts can coexist
- **WHEN** LoRA is active and saving is requested
- **THEN** the checkpoint directory MAY include both a self-contained full-model artifact and an adapter-only artifact set

#### Scenario: Adapter-only load requires base model
- **WHEN** only adapter-only artifacts are present
- **THEN** loading SHALL either (a) require an explicit base-model path/name or (b) raise a clear error explaining that the base model is required

