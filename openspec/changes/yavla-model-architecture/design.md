## Context

YAVLA's architecture research doc (`docs/architecture/vla-design-space-research.md`) defines 7 core modules with Protocol-based interfaces, a two-level integration strategy (readout + joint-token), and support for 8+ action head paradigms. The MVP change (`mvp-mlp-policy`) implements all 7 module slots in their simplest form. This design covers the full roadmap: every module variant, integration mode, and optimization needed to reach a production-quality VLA framework.

The architecture is delivered in phases. Each phase builds on the previous, never breaking existing interfaces. The MVP proves the interfaces; subsequent phases add capability without restructuring.

Reference implementations:
- **OpenVLA-OFT** (`moojink/openvla-oft`, SHA `e4287e94`): MLP regression, readout mode, `inputs_embeds` bypass pattern. [modeling_prismatic.py#L571-L638](https://github.com/moojink/openvla-oft/blob/e4287e94/prismatic/extern/hf/modeling_prismatic.py#L571-L638)
- **Octo** (`octo-models/octo`, SHA `241fb351`): Readout token pattern (zeros + learned pos_embed), mean-pool extraction. [octo_module.py#L248-L262](https://github.com/octo-models/octo/blob/241fb351/octo/model/octo_module.py#L248-L262)
- **LeRobot** (`huggingface/lerobot`, SHA `5f152322`): Policy composition, config patterns, normalization. [pretrained.py#L45](https://github.com/huggingface/lerobot/blob/5f152322/src/lerobot/policies/pretrained.py#L45)
- **π0 / OpenPI** (`Physical-Intelligence/openpi`, SHA `981483dc`): Dual-expert backbone, flow matching, reuses PaliGemma's built-in SigLIP. [pi0.py#L108-L215](https://github.com/Physical-Intelligence/openpi/blob/981483dc/src/openpi/models/pi0.py#L108-L215)

**Critical architectural insight (CORRECTED per mvp-mlp-policy review):** PaliGemma is decoder-only (SigLIP → linear projector → Gemma LM). Its `forward()` accepts `inputs_embeds` to bypass the built-in vision pipeline. `token_type_ids` controls bidirectional (image, type=0 prefix) vs causal (text, type=1) attention. The hybrid causal mask is constructed by `create_causal_mask_mapping(..., is_training=self.training)` which uses `token_type_ids` to un-mask image prefix tokens. `is_training` only controls a validation check (requiring `token_type_ids` during training); the actual mask behavior is driven by `is_first_iteration` (inferred from `past_key_values is None`). Do NOT pass dummy `labels` — they only trigger unnecessary LM loss computation. [modeling_paligemma.py#L304-L387](https://github.com/huggingface/transformers/blob/556312cd/src/transformers/models/paligemma/modeling_paligemma.py#L304-L387)

## Goals / Non-Goals

**Goals:**
- All module variants from the architecture doc implemented and registered
- Both integration modes (readout + joint-token) working
- Multiple action heads: Flow Matching, Diffusion, CVAE, VQ-BeT (in addition to MVP's MLP)
- Multiple vision encoders: DINOv2, Dual/Prismatic (in addition to MVP's SigLIP)
- Perceiver resampler token merger
- PEFT wrappers (LoRA/IA³) for vision encoder and backbone
- Dual-expert backbone for π0-style joint-token mode
- Multi-embodiment support with per-robot stems/heads
- Inference optimization: KV-cache, vision caching, deployment profiles
- Temporal ensembling in action decoder
- Evaluation framework integration

**Non-Goals:**
- Training infrastructure (distributed training, wandb, schedulers) — separate change
- Dataset layer modifications — already complete
- Energy-based / world model heads (research-only, low priority)
- Custom CUDA kernels or FlashAttention integration

> **Note:** Autoregressive token head (uses LM head directly) was previously a non-goal. It has been reclassified as a low-priority phase — the `PolicyBase` + overridable `VLAPolicy` pipeline can accommodate it via `IntegrationMode.LM_HEAD` with minimal new code.

## Decisions

### D1: Phased delivery — each phase is independently shippable

**Choice:** 8 phases, each adding one capability layer. Every phase produces a working system. No phase depends on a future phase.

**Phase order:**
1. MVP (separate change: `mvp-mlp-policy`) — MLP head, readout mode, frozen SigLIP, concat merger
2. Flow Matching head — standalone flow matching in readout mode
3. PEFT wrappers — LoRA/IA³ for backbone and vision encoder
4. Dual-Expert backbone — joint-token mode, π0-style flow matching
5. Additional heads — Diffusion, CVAE, VQ-BeT
6. Advanced vision — DINOv2, Dual encoder, Perceiver resampler
7. Multi-embodiment — per-robot adapters, ActionSpaceSpec routing
8. Inference optimization — KV-cache, vision caching, deployment profiles

**Why this order:** Flow matching is the highest-value upgrade (captures multimodal distributions). PEFT is needed before dual-expert (memory). Dual-expert unlocks π0-style. Additional heads and vision variants are independent. Multi-embodiment and inference are polish.

### D2: Two-level integration — readout mode default, joint-token opt-in

**Choice:** Readout mode is the universal default (7 of 8 heads). Joint-token mode is opt-in, requiring `DualExpertBackbone` + a head that declares `required_mode=JOINT_TOKENS`. The `validate_integration()` function enforces compatibility at build time.

**Why two levels:** Readout mode provides a clean information bottleneck — heads are trivially swappable. Joint-token mode sacrifices modularity for performance (π0 demonstrates this is necessary for flow matching with action-conditioned denoising). Supporting both means YAVLA can reproduce any published VLA architecture.

### D2b: PolicyBase ABC with overridable VLAPolicy pipeline

**Choice (implemented):** A `PolicyBase(nn.Module, ABC)` base class defines the minimal policy contract (`forward`, `predict`, `reset`, `get_optim_params`). `VLAPolicy(PolicyBase)` implements the 7-module pipeline as 5 overridable step methods: `encode_observations`, `merge_tokens`, `run_backbone`, `compute_loss`, `decode_prediction`. Future policy types (AR token, flow matching, dual-expert) subclass `VLAPolicy` and override 1-2 steps while reusing the rest.

`__init_subclass__` enforces that every concrete subclass defines `name: str` and `config_class: type` (following LeRobot's `PreTrainedPolicy` pattern).

**Why overridable steps (not separate classes):** Most VLA architectures share 80%+ of the pipeline (vision encoder, proprio encoder, backbone). Only the merger behavior, loss computation, and sometimes the forward loop differ. Overridable steps maximize module reuse while allowing radical differences when needed.

### D3: PEFT as the default backbone adaptation strategy

**Choice:** Post-MVP, the backbone defaults to LoRA on attention+MLP projections (rank 16). Full fine-tune is opt-in. Vision encoder defaults to frozen + LoRA on last 4 ViT blocks.

**Why LoRA default:** Full fine-tuning a 3B backbone requires 4× memory for optimizer states. LoRA reduces trainable params to ~0.5% while achieving comparable performance (demonstrated by OpenVLA-OFT). For deployment, LoRA weights merge into the base model for zero overhead.

### D4: Flow matching head uses continuous-time formulation

**Choice:** Implement flow matching with the rectified flow / optimal transport formulation (linear interpolation `x_t = (1-t)x_0 + t*x_1`, velocity prediction). Support both standalone (readout mode, conditioning via cross-attention) and π0-style (joint-token mode, action tokens denoised in-place).

**Why not DDPM:** Flow matching has straighter trajectories → fewer denoising steps (10 vs 50-100 for DDPM). π0 demonstrates 10-step inference is sufficient. The continuous-time formulation is simpler to implement than discrete diffusion schedules.

### D5: Perceiver resampler as the default post-MVP token merger

**Choice:** Cross-attention with N learned queries (default 64) that attend to vision tokens. Reduces sequence length from ~729 (384×384 / 14²) to 64 tokens regardless of input resolution or number of cameras.

**Why Perceiver over concat:** Multi-view setups (2-3 cameras) produce 1500+ vision tokens with concat. PaliGemma's context window and attention cost scale quadratically. Perceiver resampler provides a fixed token budget. The concat merger remains available for single-camera research configs.

### D6: Dual-expert backbone follows π0's architecture

**Choice:** Two transformer stacks sharing cross-attention: a frozen VLM expert (PaliGemma) processes vision+language, an action expert (randomly initialized, ~300M params) processes action tokens. Cross-attention layers allow bidirectional information flow. The action expert's hidden states are the output for the flow matching head.

**Why not single backbone with joint tokens:** A single backbone forces action tokens through all VLM layers, which are pretrained for language modeling. The dual-expert lets the action expert specialize for motor control while the VLM expert retains its pretrained representations. π0 demonstrates this outperforms single-backbone approaches.

### D7: Multi-embodiment via per-robot observation adapter + action decoder

**Choice:** Each robot type gets its own `ObservationAdapter` (sensor normalization, camera config) and `ActionDecoder` (unnormalization, action space mapping). The shared modules (vision encoder, backbone, action head) are robot-agnostic. `ActionSpaceSpec` and `ProprioSpec` dataclasses define the robot-specific contract. Checkpoint metadata includes embodiment info for load-time validation.

**Why adapter-level (not head-level):** Most of the robot-specific logic is in preprocessing (sensor layouts, camera intrinsics) and postprocessing (action space limits, control modes). The core model is shared. This matches CrossFormer and Octo's multi-embodiment strategy.

### D8: Inference optimization via deployment profiles

**Choice:** Three profiles: `research` (no optimization, full flexibility), `server` (KV-cache, vision caching, batch inference), `edge` (merged LoRA, quantized, static shapes). Each profile is a config preset that enables/disables optimizations.

**Why profiles (not flags):** Individual optimization flags create a combinatorial explosion of configurations. Profiles bundle compatible optimizations and are tested as units. Users pick a profile; advanced users override individual settings.

## Risks / Trade-offs

**[Dual-expert backbone is complex and may have training instability]** → Start with frozen VLM expert + trained action expert (π0's approach). Cross-attention initialization follows π0's recipe. If unstable, fall back to single backbone with joint tokens.

**[Flow matching with 10 denoising steps may be too slow for real-time control]** → 10 steps at ~50ms each = 500ms per action chunk. With chunk_len=50 at 50Hz, this is 1 second of actions per inference. Acceptable for most manipulation tasks. For faster control, reduce to 5 steps or use MLP head.

**[Perceiver resampler may lose spatial information needed for precise manipulation]** → Mitigated by making token budget configurable and keeping concat merger as fallback. SpatialVLA demonstrates that spatial-aware resamplers can preserve 3D information.

**[Multi-embodiment weight sharing may cause negative transfer]** → Per-robot adapter layers isolate robot-specific parameters. The shared backbone sees a canonical representation. If negative transfer occurs, per-robot backbone LoRA adapters can specialize further.

**[8 phases is a long roadmap — priorities may shift]** → Each phase is independently shippable. Phases can be reordered or skipped based on evaluation results. The MVP is usable immediately.

**[LoRA checkpointing must use peft's native format]** → `save_pretrained` saves LoRA adapters separately via `peft.save_pretrained()` + `checkpoint_meta.json`. `from_pretrained` reads metadata to select adapter-only or full state dict loading. Backward-compatible with pre-metadata checkpoints.
