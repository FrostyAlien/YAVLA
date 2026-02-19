## Phase 1: MVP (separate change: mvp-mlp-policy)

- [ ] 1.1 See `openspec/changes/mvp-mlp-policy/tasks.md` for full breakdown
- [ ] 1.2 Validate end-to-end forward pass: images + proprio + language → action chunk
- [ ] 1.3 Validate save_pretrained / from_pretrained round-trip

## Phase 2: Flow Matching Head

- [ ] 2.1 Implement `FlowMatchingConfig` dataclass with `num_denoise_steps`, `sigma_min`, `ot_method`
- [ ] 2.2 Implement rectified flow noise schedule (linear interpolation `x_t = (1-t)*x_0 + t*x_1`)
- [ ] 2.3 Implement `FlowMatchingHead(ActionHeadBase)` — velocity prediction network with cross-attention conditioning on `readout_states`
- [ ] 2.4 Implement `compute_loss`: sample `t ~ U(0,1)`, interpolate, predict velocity, MSE loss
- [ ] 2.5 Implement `predict`: iterative denoising from `x_0 ~ N(0,1)` with configurable step count
- [ ] 2.6 Register `FlowMatchingHead` with `head_registry`
- [ ] 2.7 Add `SamplingConfig` support (num_samples, temperature, num_denoise_steps)

## Phase 3: PEFT Wrappers

- [ ] 3.1 Implement `PEFTConfig` dataclass hierarchy (`LoRAConfig`, `IA3Config`) with target module patterns
- [ ] 3.2 Implement `apply_peft(module, config)` utility wrapping HuggingFace `peft` library
- [ ] 3.3 Implement `merge_peft(module)` for deployment (merge LoRA into base weights)
- [ ] 3.4 Add `PEFTWrappable` protocol and implement on `VLMBackbone` and `SigLIPEncoder`
- [ ] 3.5 Implement `VisionTuningConfig` with `mode` (frozen/lora_last_k/full) and wire into `SigLIPEncoder`
- [ ] 3.6 Update `build_policy` to auto-apply PEFT when `config.backbone.peft` is set
- [ ] 3.7 Update `save_pretrained` to save adapter weights separately

## Phase 4: Dual-Expert Backbone

- [ ] 4.1 Implement `DualExpertConfig` dataclass with action expert dimensions and cross-attention config
- [ ] 4.2 Implement action expert transformer stack (~300M params, randomly initialized)
- [ ] 4.3 Implement cross-attention layers between VLM expert and action expert
- [ ] 4.4 Implement `DualExpertBackbone(BackboneBase)` with `supported_modes={READOUT, JOINT_TOKENS}`
- [ ] 4.5 Implement `Pi0FlowHead(ActionHeadBase)` — joint-token flow matching with in-place action denoising
- [ ] 4.6 Register both with their respective registries
- [ ] 4.7 Validate dual-expert + pi0 flow head end-to-end

## Phase 5: Additional Action Heads

- [ ] 5.1 Implement `DiffusionHead(ActionHeadBase)` — DDPM training, DDIM inference, noise schedules
- [ ] 5.2 Implement `CVAEHead(ActionHeadBase)` — encoder/decoder, KL + reconstruction loss
- [ ] 5.3 Implement `VQBeTHead(ActionHeadBase)` — VQ-VAE codebook, discrete code prediction
- [ ] 5.4 Register all heads with `head_registry`

## Phase 6: Advanced Vision & Token Merger

- [ ] 6.1 Implement `DINOv2Encoder(VisionEncoderBase)` with same interface as SigLIP
- [ ] 6.2 Implement `DualEncoder(VisionEncoderBase)` — parallel SigLIP+DINOv2, concatenated output
- [ ] 6.3 Register vision encoders with `vision_registry`
- [ ] 6.4 Implement `PerceiverMerger` — cross-attention with learned queries, configurable `token_budget`
- [ ] 6.5 Register merger with `merger_registry`
- [ ] 6.6 Add token budget warning when ConcatMerger exceeds 256 vision tokens

## Phase 7: Multi-Embodiment

- [ ] 7.1 Implement `ObservationAdapter` base class and per-robot adapters
- [ ] 7.2 Implement `embodiment_registry` mapping robot names to specs + adapter configs
- [ ] 7.3 Implement per-robot `ActionDecoder` with robot-specific `ActionSpaceSpec`
- [ ] 7.4 Update `from_pretrained` to handle cross-embodiment loading with `strict=False`
- [ ] 7.5 Add embodiment metadata to checkpoint format

## Phase 8: Inference Optimization

- [ ] 8.1 Implement KV-cache support in `VLMBackbone` and `DualExpertBackbone`
- [ ] 8.2 Implement vision feature caching (skip ViT when image unchanged)
- [ ] 8.3 Implement `DeploymentProfile` enum and profile presets (research/server/edge)
- [ ] 8.4 Implement LoRA merge + quantization prep for edge profile
- [ ] 8.5 Implement `EnsemblingDecoder` with temporal averaging and contact-aware disable
- [ ] 8.6 Implement receding horizon control in action decoder
- [ ] 8.7 Implement config version migration for `from_pretrained`
