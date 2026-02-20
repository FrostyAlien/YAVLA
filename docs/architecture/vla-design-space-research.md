# YAVLA Architecture Research: The VLA Design Space

> Deep research into Vision-Language-Action model architectures, modular design patterns, and a proposed architecture for YAVLA.
>
> **Date**: February 2026
> **Scope**: Research only — module-level design, no implementation code
> **Sources**: 7 parallel research agents + Oracle architectural synthesis

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Action Prediction Heads — The Full Design Space](#2-action-prediction-heads)
   - 2.1 Diffusion (DDPM/DDIM)
   - 2.2 Flow Matching
   - 2.3 Autoregressive Tokens
   - 2.4 CVAE (ACT-style)
   - 2.5 VQ-BeT (Discrete Latent)
   - 2.6 MLP / Direct Regression
   - 2.7 Energy-Based Models
   - 2.8 World Model + Inverse Dynamics
   - 2.9 Hybrid (Coarse-to-Fine)
3. [Vision Encoders & VL Fusion](#3-vision-encoders--vl-fusion)
   - 3.1 Vision Encoder Landscape
   - 3.2 Vision-Language Fusion Strategies
   - 3.3 Multi-View & Temporal Handling
4. [Cutting-Edge VLA Models (2024–2025)](#4-cutting-edge-vla-models-20242025)
   - 4.1 π0 (Physical Intelligence)
   - 4.2 HPT (Heterogeneous Pre-trained Transformers)
   - 4.3 GR-2 (ByteDance)
   - 4.4 CrossFormer / Open X-Embodiment
   - 4.5 Gato (DeepMind)
   - 4.6 SpatialVLA (3D-Aware)
   - 4.7 OpenVLA-OFT
5. [Modular Framework Patterns](#5-modular-framework-patterns)
   - 5.1 PEFT (Adapter Registry)
   - 5.2 MMEngine (Hierarchical Registry)
   - 5.3 timm (Flat Function Registry)
   - 5.4 LeRobot (Most Directly Relevant)
   - 5.5 MoE as Modularity
6. [The Modularity–Performance Tension](#6-the-modularityperformance-tension)
7. [Proposed YAVLA Architecture](#7-proposed-yavla-architecture)
   - 7.1 The 7 Core Modules
   - 7.2 Interface Contracts
   - 7.3 Action Head Mapping
   - 7.4 Config-Driven Composition
   - 7.5 Directory Structure
   - 7.6 Checkpoint & Serialization
   - 7.7 Inference & Deployment
   - 7.8 Testing Strategy
8. [Design Decisions & Tradeoffs](#8-design-decisions--tradeoffs)
9. [References](#9-references)

---

## 1. Executive Summary

The VLA (Vision-Language-Action) design space has exploded in 2024–2025. We surveyed **20+ models** across **9 distinct action prediction paradigms**, cataloged vision encoder and VL fusion strategies, studied modular ML framework patterns from 6 production codebases, and synthesized a proposed modular architecture for YAVLA.

### Key Findings

- **9 known action prediction paradigms** exist, from simple MLP regression to world-model-based planning. The field is converging on **flow matching** (π0) and **parallel continuous decoding** (OpenVLA-OFT) as the most promising directions.
- **The fundamental tension** is between modularity (Octo's readout tokens — clean but lossy) and performance (π0's joint attention — tight coupling but best results). Our architecture resolves this with a **two-level integration strategy**: readout mode (default, max swap) + joint-token mode (opt-in, max performance).
- **No existing framework** provides a fully modular VLA with swappable action heads. LeRobot comes closest with its `PreTrainedPolicy` abstraction, but it doesn't address the backbone–head coupling problem.
- **The YAVLA architecture** defines 7 core modules with Protocol-based interfaces, config-driven instantiation, and explicit capability negotiation between backbone and action head.

### The Design Space at a Glance

| Dimension        | Options                                                                   | YAVLA Default                                         |
| ---------------- | ------------------------------------------------------------------------- | ----------------------------------------------------- |
| Action head      | Diffusion, Flow Matching, AR, CVAE, VQ-BeT, MLP, EBM, World Model, Hybrid | Flow Matching (π0-style)                              |
| Vision encoder   | SigLIP, DINOv2, Dual (SigLIP+DINO), Custom ViT, 3D-aware                  | SigLIP So400m/14 (frozen + late-block LoRA)           |
| VL backbone      | Direct VLM, Linear projection, Dual-stream, Cross-attention               | Dual-stream (PaliGemma + Action Expert, PEFT default) |
| VL fusion        | Tokens-in-sequence, ConditionBundle, FiLM, Readout tokens                 | Tokens-in-sequence                                    |
| Action space     | Continuous, Discretized bins, VQ codebook, DCT frequency                  | Continuous (flow matching)                            |
| Multi-embodiment | Shared trunk, Per-robot stems/heads (HPT), Per-robot action heads         | Per-robot stems/heads                                 |

---

## 2. Action Prediction Heads

The action head is the most consequential architectural choice in a VLA. It determines inference latency, action quality (smoothness, multimodality), training stability, and how tightly coupled the head must be to the backbone. We catalog 9 distinct paradigms.

### Overview Comparison

| Paradigm                | Multimodal? | Inference Steps   | Latency   | Smoothness     | Key Models             |
| ----------------------- | ----------- | ----------------- | --------- | -------------- | ---------------------- |
| MLP / Direct Regression | ❌ No        | 1                 | ~1ms      | Low            | BC-Z, RT-1             |
| CVAE (ACT)              | ✅ Yes       | 1 (+ sampling z)  | ~2ms      | High (chunked) | ACT, ACT++             |
| Diffusion (DDPM/DDIM)   | ✅ Yes       | 10–100            | 50–500ms  | High           | Diffusion Policy, Octo |
| Flow Matching           | ✅ Yes       | 1–10              | 5–50ms    | High           | π0, ManiFlow           |
| Autoregressive Tokens   | ❌ No*       | D×T sequential    | 10–200ms  | Medium         | RT-2, OpenVLA, FAST    |
| VQ-BeT                  | ✅ Yes       | 1 (+ codebook)    | ~5ms      | Medium         | VQ-BeT, QueST          |
| Energy-Based (IBC)      | ✅ Yes       | 50–200 (Langevin) | 100–500ms | Medium         | Implicit BC            |
| World Model + IDM       | ✅ Yes       | Video gen + IDM   | 1–10s     | High           | UniPi, SuSIE           |
| Hybrid (Coarse-to-Fine) | ✅ Yes       | AR + Diffusion    | 20–100ms  | High           | DiVLA, PIVOT-R         |

\* AR can represent multimodality through sampling temperature, but each sample is unimodal.

### 2.1 Diffusion (DDPM/DDIM)

**Core idea**: Model the action distribution p(a|o) by learning to reverse a noise-corruption process. At inference, start from Gaussian noise and iteratively denoise to produce an action chunk.

**Training**: Given a clean action chunk a₀, sample timestep t ~ U[0,T], add noise ε ~ N(0,I) to get aₜ, train network to predict ε (noise prediction) or a₀ (sample prediction). Loss: MSE between predicted and true noise.

**Inference**: Start from aₜ ~ N(0,I), run T denoising steps. DDIM accelerates this to ~10 steps with deterministic sampling.

**Two denoiser architectures exist in practice**:

1. **1D Temporal CNN (FiLM conditioning)** — Used in the original Diffusion Policy. The observation embedding conditions the denoiser via FiLM (Feature-wise Linear Modulation) layers: γ,β = MLP(obs); h = γ⊙h + β at each residual block. The CNN operates over the time dimension of the action chunk. Fast, lightweight, but limited receptive field.

2. **Transformer Decoder (cross-attention)** — Also from Diffusion Policy. Noisy action tokens attend to observation tokens via cross-attention. More expressive, handles variable-length conditioning naturally, but heavier.

**Key models**:

- **Diffusion Policy** (Chi et al., 2023): Demonstrated both CNN and Transformer variants. CNN variant: 1D temporal convolutions with FiLM conditioning, 256-dim features, ~100 DDPM training steps (10 DDIM steps at inference). Transformer variant: causal transformer decoder, noisy actions as queries cross-attending to observation keys. Both predict action chunks of 8–16 steps.

- **Octo** (Octo Model Team, 2024): Uses a Transformer backbone with **readout tokens** — special learned tokens appended to the input sequence whose output embeddings are fed to a diffusion head. The diffusion head itself is a small MLP that takes readout_states + noisy_action + timestep_embedding → predicted noise. This is the **cleanest modular interface** we found: the readout tokens act as an information bottleneck between backbone and head.

- **3D Diffuser Actor** (Ke et al., 2024): Operates in 3D space — denoises 3D positions and rotations directly using point cloud observations. Uses 3D relative position attention between noisy keypose tokens and visual tokens from point clouds.

**Interface to backbone**: Receives a conditioning tensor (observation embedding). In Octo, this is the readout token output. In Diffusion Policy, it's the observation encoder output. The head is self-contained — it runs its own iterative loop.

**Strengths**: Captures multimodal action distributions naturally (multiple denoising paths from different noise samples). Produces smooth, temporally coherent action chunks. Well-studied theory.

**Weaknesses**: Slow inference (10–100 forward passes per action). Sensitive to noise schedule tuning. Training can be unstable without careful hyperparameter selection.

**Modularity verdict**: ✅ Excellent. The head only needs a conditioning vector. Readout tokens (Octo-style) make this fully swappable.

### 2.2 Flow Matching

**Core idea**: Instead of learning to reverse a noise process (diffusion), learn a **velocity field** v(aₜ, t) that transports samples from noise distribution to data distribution along straight(er) paths. Mathematically: daₜ/dt = v(aₜ, t), integrated from t=0 (noise) to t=1 (data).

**Training**: Given clean action a₁ and noise a₀ ~ N(0,I), the interpolant is aₜ = (1-t)·a₀ + t·a₁ (linear). The target velocity is simply v* = a₁ - a₀. Train network to predict v(aₜ, t) with MSE loss against v*. Simpler than diffusion — no noise schedule, no variance terms.

**Inference**: Start from a₀ ~ N(0,I), integrate ODE using Euler steps: aₜ₊ₐₜ = aₜ + Δt · v(aₜ, t). Typically 5–10 Euler steps suffice (vs 10–100 for diffusion DDIM).

**Key models**:

- **π0** (Black et al., 2024, Physical Intelligence): The flagship flow matching VLA. Architecture: PaliGemma 3B VLM + 300M action expert transformer (3.3B total) sharing attention via block-causal joint attention. Both streams are concatenated into a single sequence — the VLM prefix (vision + text) attends bidirectionally among itself but cannot see action tokens; the action expert suffix attends bidirectionally among action tokens AND attends to all VLM prefix tokens. Separate FFN weights per stream, shared K/V in the same attention operation. Training uses the flow matching objective on 50-step action chunks. Inference: 10 Euler steps at 50Hz control frequency. This is the **highest-performing VLA** as of early 2025.

- **ManiFlow** (2024): Applies flow matching to manipulation with a focus on efficient few-step inference. Demonstrates that 1–5 Euler steps can match diffusion quality with 10× fewer forward passes.

**Interface to backbone**: In π0, the action expert is **tightly coupled** — it participates in block-causal joint attention with the VLM backbone. Both streams are concatenated into a single sequence with separate FFN weights but shared K/V computation. The VLM prefix cannot see action tokens, but action tokens can attend to all VLM tokens. This yields best performance but worst modularity.

**Strengths**: Faster inference than diffusion (fewer steps, straighter paths). Simpler training (no noise schedule). Captures multimodality. State-of-the-art results.

**Weaknesses**: Tight coupling to backbone in best-performing variant (π0). Relatively new — less community tooling. ODE integration can still be slow for real-time control if many steps needed.

**Modularity verdict**: ⚠️ Mixed. A standalone flow matching head (ManiFlow-style, receiving conditioning vector) is fully swappable. But the best variant (π0-style dual-expert) requires deep backbone integration. YAVLA should support both modes.

### 2.3 Autoregressive Tokens

**Core idea**: Discretize continuous actions into token IDs and predict them sequentially using the language model's next-token prediction machinery. The action becomes "just more text" in the LLM's vocabulary.

**Discretization schemes**:

1. **Uniform binning** (RT-2, OpenVLA): Each action dimension is independently discretized into 256 uniform bins spanning the observed data range. Each bin maps to a token ID. RT-2 maps bins to string representations ("128", "255"); OpenVLA maps to the 256 least-used tokens in the LLM vocabulary (`vocab[-256:]`), avoiding collision with real language tokens.

2. **DCT frequency tokenization** (FAST, Pertsch et al. 2025): Instead of tokenizing raw actions, apply Discrete Cosine Transform to an action chunk (e.g., 50 steps), quantize the DCT coefficients, and encode them via a learned codebook. This compresses a 50-step chunk to ~32 tokens — a 10× reduction in sequence length vs per-timestep tokenization. Dramatically faster inference.

**Training**: Standard cross-entropy loss on next-token prediction, identical to language modeling. The action tokens are simply appended to the text/vision token sequence.

**Inference**: Autoregressively sample one action token at a time. For 7-DoF robot with per-step tokenization: 7 sequential forward passes per timestep. For FAST with chunk compression: ~32 forward passes per chunk of 50 steps.

**Key models**:

- **RT-2** (Brohan et al., 2023): 55B PaLI-X or 562B PaLM-E backbone. Actions tokenized as text strings in the output sequence. 256 bins per dimension. Demonstrated that VLMs can output robot actions without architectural changes — just fine-tune on action data.

- **OpenVLA** (Kim et al., 2024): 7B Llama-2 backbone with dual DINOv2 + SigLIP vision encoder (Prismatic). Maps 256 action bins to least-frequent vocabulary tokens. Extracts features from the **second-to-last** transformer layer (not final) for better spatial grounding. Single-step prediction (no chunking).

- **FAST** (Pertsch et al., 2025): DCT-based chunk tokenization. Compresses 50-step action chunks to ~32 tokens. Achieves 5–8× faster inference than per-step AR while maintaining quality. Compatible with any LLM backbone.

**Interface to backbone**: Trivial — action tokens are part of the vocabulary. The LLM's existing next-token prediction head handles everything. No separate action head module needed (the LM head IS the action head).

**Strengths**: Zero architectural modification to the LLM. Leverages pre-trained language modeling capabilities. Simple training pipeline. Benefits from LLM scaling.

**Weaknesses**: Sequential decoding is slow (D tokens per step). Discretization loses precision. Per-step tokenization doesn't capture temporal correlations. Unimodal per-sample (though temperature sampling adds diversity).

**Modularity verdict**: ✅ Excellent (trivially). The "head" is just the LM head — nothing to swap. But this means the action representation is locked to the tokenization scheme, which is a different kind of coupling.

### 2.4 CVAE (ACT-style)

**Core idea**: Use a Conditional Variational Autoencoder to model the action distribution. An encoder network compresses a demonstration action chunk into a latent code z ~ N(μ, σ²). A decoder network reconstructs the action chunk from z conditioned on observations. At inference, sample z ~ N(0, I) and decode.

**Architecture (ACT — Action Chunking with Transformers, Tony Zhao et al. 2023)**:

- **Encoder** (training only): A transformer encoder that takes the ground-truth action chunk + observation as input and outputs μ, σ for the latent z (typically 32-dim). This is discarded at inference.
- **Decoder**: A transformer decoder with learned action queries (one per timestep in the chunk). Cross-attends to observation tokens + z token. Outputs a full action chunk (e.g., 100 steps) in a single forward pass.
- **Chunk size**: ACT uses chunk_size=100 (predicting 100 future steps at once). Temporal ensembling averages overlapping predictions from consecutive chunks for smoother execution.

**Training**: ELBO loss = reconstruction MSE + KL divergence D_KL(q(z|a,o) || p(z)). The KL term regularizes the latent space. β-VAE weighting can control the trade-off.

**Inference**: Single forward pass — sample z ~ N(0,I), decode to full action chunk. Extremely fast.

**Strengths**: Single-step inference (fastest of all multimodal methods). Chunk prediction captures temporal coherence. Well-understood VAE theory. Temporal ensembling provides smooth execution.

**Weaknesses**: Mode coverage limited by Gaussian prior (may miss rare modes). KL collapse can reduce expressiveness. Chunk boundaries can cause discontinuities without ensembling.

**Modularity verdict**: ✅ Excellent. The decoder only needs observation embeddings + sampled z. Fully self-contained head.

### 2.5 VQ-BeT (Discrete Latent)

**Core idea**: Use a learned vector-quantized codebook to discretize action chunks into a small set of discrete codes, then predict those codes with a transformer. Combines the multimodality of discrete representations with the expressiveness of learned (not uniform) quantization.

**Architecture**:

- **Stage 1 — VQ-VAE training**: Train a VQ-VAE on action chunks. The encoder maps an action chunk to a sequence of discrete codebook indices. The decoder reconstructs the chunk from these indices. Codebook size typically 512–2048 entries.
- **Stage 2 — Prior training**: Train a transformer to predict the codebook indices conditioned on observations. At inference, predict codes → decode to continuous actions via the VQ-VAE decoder.

**Key models**:

- **VQ-BeT** (Lee et al., 2024): Hierarchical VQ with coarse and fine codebooks. The transformer first predicts a coarse code (selecting a cluster), then a fine code (selecting within the cluster). This two-level hierarchy improves mode coverage. Reported 5× faster than diffusion at inference.

- **QueST** (2024): Extends VQ-BeT with improved codebook utilization and commitment loss scheduling. Demonstrates better coverage of rare action modes.

**Strengths**: Fast inference (single forward pass for code prediction + one VQ decode). Naturally multimodal (different codes = different modes). Learned discretization adapts to data distribution (unlike uniform bins).

**Weaknesses**: Two-stage training pipeline adds complexity. Codebook collapse (unused entries) requires careful training. Reconstruction quality bounded by codebook expressiveness.

**Modularity verdict**: ✅ Good. The prior model needs observation embeddings; the VQ decoder is self-contained. The codebook is action-head-specific and doesn't leak into the backbone.

### 2.6 MLP / Direct Regression

**Core idea**: Directly regress continuous action values from observation embeddings via an MLP. The simplest possible action head.

**Architecture**: Observation embedding → MLP (2–4 layers, ReLU/GELU) → continuous action vector. Loss: MSE or L1 between predicted and ground-truth actions.

**Key models**:

- **BC-Z** (Jang et al., 2022): ResNet-18 encoder → MLP action head. Baseline behavioral cloning.
- **RT-1** (Brohan et al., 2023): EfficientNet + FiLM-conditioned TokenLearner → per-dimension classification heads (256 bins each, technically a discretized MLP). Each action dimension has its own linear head predicting a softmax over 256 bins — decoded in parallel (not autoregressive).
- **RoboFlamingo** (Li et al., 2024): Flamingo VLM backbone → MLP or LSTM action head. The MLP variant is a simple 2-layer network on the last hidden state.

**Strengths**: Fastest inference (single forward pass, no iteration). Simplest to implement and debug. Stable training.

**Weaknesses**: Cannot represent multimodal distributions (averages modes → poor performance on tasks with multiple valid solutions). No temporal coherence without explicit chunking. Poor on contact-rich manipulation.

**Modularity verdict**: ✅ Trivially modular. An MLP head is the simplest possible module — takes a vector, outputs a vector.

### 2.7 Energy-Based Models (Implicit BC)

**Core idea**: Learn an energy function E(a, o) over action-observation pairs. Low energy = good action. At inference, find the action that minimizes energy via gradient-based optimization (Langevin dynamics) or sampling (CEM — Cross-Entropy Method).

**Training**: Contrastive learning — push energy down for ground-truth actions, up for negative samples. InfoNCE-style loss with negative actions sampled from a proposal distribution.

**Inference**: Initialize action candidates randomly, iteratively refine via ∇ₐE(a, o) (Langevin) or evolutionary sampling (CEM). Requires 50–200 iterations.

**Key model — Implicit BC** (Florence et al., 2021): Demonstrated that EBMs can capture multimodal action distributions better than MSE regression. Used coordinate-wise energy with Langevin MCMC at inference.

**Strengths**: Naturally multimodal (energy landscape has multiple basins). No mode averaging. Flexible — any differentiable architecture works as the energy function.

**Weaknesses**: Very slow inference (iterative optimization). Sensitive to initialization. Energy landscape can have spurious local minima. Difficult to scale to high-dimensional action spaces.

**Modularity verdict**: ✅ Good. The energy function only needs (action, observation) pairs. Self-contained, but the iterative inference loop adds latency.

### 2.8 World Model + Inverse Dynamics

**Core idea**: Instead of directly predicting actions, first generate a **future visual plan** (video or image sequence), then extract actions from the plan using an inverse dynamics model (IDM). Decouples "what to do" (planning) from "how to move" (action extraction).

**Architecture**:

1. **World model**: Given current observation + language goal, generate a sequence of future images (video prediction). Can use diffusion-based video generation.
2. **Inverse dynamics model (IDM)**: Given two consecutive frames (oₜ, oₜ₊₁), predict the action aₜ that transitions between them. Typically a small CNN or MLP.

**Key models**:

- **UniPi** (Du et al., 2023): Text-conditioned video diffusion model generates future video plan. A separate inverse dynamics model extracts actions frame-by-frame. Demonstrated zero-shot generalization to new tasks via language-guided video planning.

- **SuSIE** (Black et al., 2023): Image-editing diffusion model generates a single subgoal image. A low-level policy (trained separately) reaches the subgoal. Hierarchical: high-level visual planner + low-level controller.

**Strengths**: Leverages powerful pre-trained video generation models. Visual plans are interpretable. Can generalize to novel tasks via language-conditioned video generation.

**Weaknesses**: Extremely slow (video generation + IDM). IDM errors compound. Generated videos may be physically implausible. Two-stage pipeline is complex.

**Modularity verdict**: ✅ Excellent (by design). The world model and IDM are completely separate modules with a clean image-sequence interface between them.

### 2.9 Hybrid (Coarse-to-Fine)

**Core idea**: Combine two paradigms — use a fast/discrete method for coarse action planning, then refine with a continuous method. Typically: AR tokens for high-level intent → diffusion/flow for precise continuous actions.

**Key models**:

- **DiVLA** (2024): LLM backbone autoregressively generates coarse discrete action tokens (waypoints or subgoals). A diffusion head then refines these into smooth continuous trajectories. The AR stage handles semantic reasoning ("move to the cup"), the diffusion stage handles motor precision.

- **PIVOT-R** (2024): VLM generates candidate action proposals as text/tokens. A value function or refinement network selects and refines the best proposal into precise continuous actions.

**Strengths**: Best of both worlds — AR handles discrete decisions and language grounding, diffusion/flow handles continuous precision. Natural decomposition of reasoning vs. control.

**Weaknesses**: Two-stage inference adds latency and complexity. Interface between coarse and fine stages must be carefully designed. Error propagation between stages.

**Modularity verdict**: ✅ Good. The two stages are naturally separate modules. The interface (coarse action representation) must be standardized.

---

## 3. Vision Encoders & VL Fusion

### 3.1 Vision Encoder Landscape

| Encoder                     | Pre-training          | Resolution | Patch Size | Output Dim    | Used By              |
| --------------------------- | --------------------- | ---------- | ---------- | ------------- | -------------------- |
| SigLIP So400m/14            | Contrastive (sigmoid) | 224–384    | 14×14      | 1152          | π0, OpenVLA          |
| DINOv2 ViT-L/14             | Self-supervised       | 224–518    | 14×14      | 1024          | OpenVLA (dual), Octo |
| SigLIP + DINOv2 (Prismatic) | Both                  | 224        | 14×14      | 2176 (concat) | OpenVLA v2           |
| SmallStem16 CNN             | From scratch          | 256        | 16×16      | 512           | Octo                 |
| ViT-22B                     | Contrastive           | 224        | 14×14      | 6144          | RT-2                 |
| EfficientNet-B3             | ImageNet              | 300        | N/A        | 1536          | RT-1                 |

**Key observations from source code analysis**:

- **SigLIP vs DINO**: SigLIP excels at semantic/language-aligned features (good for instruction following). DINOv2 excels at spatial/geometric features (good for precise manipulation). The dual-encoder approach (Prismatic) channel-concatenates both, getting the best of both worlds at 2× token cost.

- **Feature extraction layer matters**: OpenVLA extracts from the **second-to-last** transformer layer, not the final layer. The final layer is too specialized for the contrastive objective; earlier layers retain richer spatial information needed for action prediction.

- **Frozen vs fine-tuned**: Most VLAs freeze the vision encoder during action fine-tuning (π0, OpenVLA). RT-2 fine-tunes end-to-end but uses a massive 22B ViT. Freezing is preferred for smaller encoders to preserve pre-trained representations.

### 3.2 Vision-Language Fusion Strategies

How vision tokens meet language tokens is a critical design axis. Four main strategies exist:

**1. Tokens-in-sequence (most common)**

Vision patch tokens are projected to the LLM's embedding dimension via a linear layer (or small MLP) and prepended to the text token sequence. The LLM's self-attention handles all cross-modal interaction.

- Used by: π0, OpenVLA, RT-2, LLaVA-family
- Projection: Linear(vision_dim, llm_dim) or 2-layer MLP
- Pros: Simplest. Leverages LLM's pre-trained attention. No new modules.
- Cons: Quadratic attention cost with many vision tokens. Vision tokens compete with text tokens for attention bandwidth.

**2. Cross-attention (Flamingo-style)**

Dedicated cross-attention layers are interleaved between LLM self-attention layers. Vision tokens serve as keys/values; text tokens serve as queries. The LLM layers are frozen; only cross-attention layers are trained.

- Used by: Flamingo, RoboFlamingo, Otter
- Pros: Keeps LLM weights frozen. Efficient — vision tokens don't consume sequence length.
- Cons: Additional parameters. Cross-attention layers must be trained from scratch.

**3. FiLM conditioning**

Vision features modulate intermediate activations via learned affine transforms: h = γ(v)⊙h + β(v). No attention over vision tokens — just global conditioning.

- Used by: RT-1 (EfficientNet → FiLM → TokenLearner), Diffusion Policy CNN variant
- Pros: Very lightweight. No sequence length increase.
- Cons: Loses spatial structure of vision (global pooling). Cannot attend to specific image regions.

**4. Readout tokens (Octo-style)**

Learned special tokens are appended to the input sequence. After the backbone processes everything, the output embeddings at readout token positions are extracted and passed to the action head. The readout tokens learn to aggregate task-relevant information.

- Used by: Octo
- Pros: Clean information bottleneck. Backbone and head are fully decoupled.
- Cons: Information loss through the bottleneck. Readout tokens must learn what to extract.

### 3.3 Multi-View & Temporal Handling

**Multi-view cameras**: All surveyed models process each camera view independently through the vision encoder, then concatenate the resulting token sequences. No model uses cross-view attention at the encoder level — fusion happens in the backbone.

```
cam_1 → ViT → [tokens_1]  ─┐
cam_2 → ViT → [tokens_2]  ─┼─→ concat → [all_vision_tokens] → backbone
cam_3 → ViT → [tokens_3]  ─┘
```

Camera identity is encoded via:
- Learned camera-ID embeddings added to patch tokens (Octo, π0)
- Separate projection layers per camera (less common)
- Position encoding that includes camera index

**Temporal handling** (observation history):

- **Frame stacking**: Concatenate N recent frames as additional token sequences. Simple but expensive (N× vision tokens). Used by Octo (2-frame history).
- **Temporal embedding**: Add a learned time-step embedding to each frame's tokens. Allows the backbone to distinguish "current" from "1 step ago".
- **No history**: Many VLAs use single-frame input and rely on action chunking for temporal coherence (π0, OpenVLA). The action chunk implicitly encodes temporal context.

**Proprioception encoding**: Joint positions/velocities are typically projected via a small MLP to the backbone's embedding dimension and appended as additional tokens (Octo, π0) or concatenated to the observation embedding (Diffusion Policy).

---

## 4. Cutting-Edge VLA Models (2024–2025)

This section provides architectural deep-dives into the most influential recent VLAs, with a focus on interface boundaries relevant to modular design.

### 4.1 π0 (Physical Intelligence, 2024)

**The current state-of-the-art VLA.** π0 achieves the best reported manipulation performance by tightly coupling a pre-trained VLM with a dedicated action expert via shared attention.

**Architecture**:

```
Images → SigLIP So400m/14 (frozen) → Linear proj → vision tokens
Text   → Gemma tokenizer → text tokens

[vision_tokens, text_tokens] → PaliGemma 3B (VLM stream, prefix)
[noisy_action_tokens + robot_state + timestep_emb] → Action Expert 300M (action stream, suffix)

Block-causal joint attention at every transformer layer:
  Prefix (vision + text): bidirectional attention among themselves, CANNOT see suffix
  Suffix (actions + state): bidirectional among themselves + attends to ALL prefix tokens
  Separate FFN weights per stream

Action Expert output → Linear → velocity prediction v(aₜ, t)
```

**Key design choices**:
- **Block-causal joint attention**: The VLM (prefix) and action expert (suffix) are separate parameter sets with shared attention keys/values. The prefix tokens attend bidirectionally among themselves but cannot see suffix tokens. The suffix tokens attend bidirectionally among themselves AND to all prefix tokens. This asymmetric (block-causal) design lets the action expert read VLM context without contaminating VLM representations.
- **Flow matching objective**: Predicts velocity field v(aₜ, t) = a₁ - a₀. 10 Euler steps at inference.
- **Action chunk**: 50 steps at 50Hz = 1 second of action per inference.
- **Full fine-tuning by default**: PaliGemma 3B (Gemma 2B LLM + SigLIP = 3B total) is fully fine-tuned alongside the 300M action expert (3.3B total). LoRA is available as an optional efficiency mode but is not the default.

**Modularity analysis**: The block-causal joint attention architecture is the **least modular** design we found. You cannot swap the action expert without understanding the shared attention mechanism. However, the vision encoder (SigLIP) IS modular — it's a standard frozen encoder with linear projection.

### 4.2 HPT (Heterogeneous Pre-trained Transformers, NeurIPS 2024 Spotlight)

**The most modular VLA architecture.** HPT explicitly decomposes the model into per-embodiment stems, a shared trunk, and per-embodiment heads.

**Architecture**:

```
Observation (varies per robot)
  → Embodiment-specific Stem (learned projection per robot type)
    → Shared Transformer Trunk (pre-trained, frozen or fine-tuned)
      → Embodiment-specific Head (per-robot action decoder)
        → Actions (robot-specific dim)
```

**Key design choices**:
- **Stems**: Each robot type has its own stem that projects heterogeneous observations (different camera counts, proprioception dims) into a fixed-size token sequence for the trunk. Stems are small MLPs or lightweight transformers.
- **Trunk**: A single shared transformer pre-trained on data from many robots. Learns embodiment-agnostic manipulation representations.
- **Heads**: Per-robot action decoders. Can be MLP, diffusion, or any paradigm — the trunk doesn't care.

**Modularity analysis**: ✅ **Best-in-class modularity.** Adding a new robot = adding a new stem + head. The trunk is shared and frozen. This is the architecture YAVLA should emulate for multi-embodiment support.

### 4.3 GR-2 (ByteDance, 2024)

**A video-generation approach to robot learning.** GR-2 (Cheang et al.) pre-trains a large video prediction model on internet video, then fine-tunes it to jointly predict future frames and robot actions.

**Architecture**:
- **Stage 1**: Pre-train a video prediction transformer on large-scale internet video (no robot data). The model learns physics, object permanence, and motion priors.
- **Stage 2**: Fine-tune on robot data with a dual objective: (1) predict next video frame, (2) predict robot action. Actions are predicted by an MLP head on the same hidden states used for video prediction.

**Key insight**: Video prediction as a pre-training task gives the model a world model "for free." The action head piggybacks on representations that already understand physical dynamics.

**Modularity analysis**: ⚠️ Medium. The video prediction backbone is tightly coupled to the action head (shared hidden states). But the action MLP head itself is simple and swappable.

### 4.4 CrossFormer / Open X-Embodiment (2024)

**A multi-robot generalist trained on the Open X-Embodiment dataset** (1M+ episodes from 22 robot types).

**Architecture**:
- Transformer backbone processes tokenized observations from any robot type
- Per-embodiment action heads handle different action spaces (different DoF, gripper types)
- Shared observation tokenization with robot-type conditioning

**Key insight**: Cross-embodiment transfer works — a model trained on many robots outperforms single-robot specialists on some tasks. The key is per-robot action heads with a shared backbone.

**Modularity analysis**: ✅ Good. Per-embodiment heads are naturally modular. Similar to HPT's philosophy but with a single shared tokenization scheme.

### 4.5 Gato (DeepMind, 2022)

**The original "one model for everything" — a single transformer for Atari, robotics, captioning, and dialogue.**

**Architecture**:
- All modalities tokenized into a flat sequence: images → ViT patches, text → BPE tokens, actions → discretized bins, proprioception → discretized bins
- Single decoder-only transformer processes everything autoregressively
- 1.2B parameters, 1024 token context length

**Key insight**: Unified tokenization enables a single model to handle radically different tasks. But performance on each task is worse than specialists.

**Modularity analysis**: ❌ Anti-modular by design. Everything is flattened into one sequence. No separable components. Interesting as a research direction but not a practical architecture for YAVLA.

### 4.6 SpatialVLA (3D-Aware, 2025)

**Adds explicit 3D spatial reasoning to VLAs** via Ego3D Position Encoding — encoding camera extrinsics and depth into the vision tokens.

**Architecture**:
- Standard VLM backbone (PaliGemma-family)
- Vision tokens are augmented with 3D positional encodings derived from camera intrinsics/extrinsics and estimated depth
- The model "knows where" each patch is in 3D space, improving spatial manipulation tasks

**Key insight**: Standard 2D vision tokens lose depth and camera-relative position information. Ego3D PE recovers this, significantly improving tasks requiring precise spatial reasoning (e.g., stacking, insertion).

**Modularity analysis**: ✅ Good. The 3D PE is an additive module on top of standard vision encoding. Can be toggled on/off without changing the backbone or action head.

### 4.7 OpenVLA-OFT (Feb 2025)

**The fastest open-source VLA.** OpenVLA-OFT replaces autoregressive token decoding with parallel continuous decoding, achieving 26× speedup over the original OpenVLA.

**Architecture changes from OpenVLA**:
- **Parallel decoding**: Instead of predicting 7 action tokens sequentially, predict all dimensions simultaneously via a single linear head on the last hidden state. Eliminates the sequential bottleneck.
- **L1 regression loss**: Replaces cross-entropy over discrete bins with direct L1 loss on continuous actions. Simpler, faster, and surprisingly competitive.
- **Chunked prediction**: Predicts multiple future steps (chunk_size=8) in one forward pass.

**Key insight**: The autoregressive tokenization in the original OpenVLA was the primary bottleneck — both for speed (sequential decoding) and quality (discretization error). Parallel continuous decoding fixes both.

**Modularity analysis**: ✅ Excellent. The action head is a simple linear layer — trivially swappable. The backbone (Llama-2 7B + DINOv2+SigLIP via Prismatic) is unchanged from OpenVLA.

---

## 5. Modular Framework Patterns

We studied 5 production ML frameworks to understand how they achieve component swappability. These patterns directly inform YAVLA's registry and plugin architecture.

### 5.1 PEFT (Adapter Registry)

**Pattern**: Four parallel dictionaries keyed by `PeftType` enum.

```python
# Four registries, one per concern
PEFT_TYPE_TO_CONFIG_MAPPING: dict[PeftType, type[PeftConfig]] = {}
PEFT_TYPE_TO_TUNER_MAPPING: dict[PeftType, type[BaseTuner]] = {}
PEFT_TYPE_TO_MIXED_MODEL_MAPPING: dict[PeftType, type[BaseTuner]] = {}
PEFT_TYPE_TO_PREFIX_MAPPING: dict[PeftType, str] = {}

# Registration at import time
def register_peft_method(peft_type, config_cls, tuner_cls, model_cls, prefix):
    PEFT_TYPE_TO_CONFIG_MAPPING[peft_type] = config_cls
    # ... etc
```

**Strengths**: Simple, explicit, type-safe. Easy to see all registered methods at a glance.
**Weaknesses**: Manual registration. Adding a new method requires touching the registry file.
**Relevance to YAVLA**: Good pattern for action heads — each head type maps to (config, module, loss) tuple.

### 5.2 MMEngine (Hierarchical Registry)

**Pattern**: Hierarchical `Registry` with scope, auto-import locations, and decorator-based registration.

```python
MODELS = Registry('models', scope='mmdet', locations=['mmdet.models'])

@MODELS.register_module()
class FasterRCNN(BaseDetector):
    ...

# Build from config dict
model = MODELS.build(dict(type='FasterRCNN', backbone=dict(type='ResNet', depth=50)))
```

**Key features**:
- **Scoped registries**: `mmdet.MODELS` vs `mmseg.MODELS` — same name, different scope. Child registries can fall back to parent.
- **Auto-import via locations**: Registry accepts `locations` — module paths that are auto-imported on first `build()` call. This provides lazy loading without an explicit `lazy_init` flag.
- **Config-driven construction**: `build(cfg_dict)` recursively instantiates nested components. The `type` key selects the class; remaining keys become constructor args.

**Strengths**: Scales to huge ecosystems (OpenMMLab has 20+ repos sharing registries). Config-driven composition is powerful.
**Weaknesses**: Magic strings (`type='FasterRCNN'`). Runtime errors if type not registered. Complex inheritance hierarchy.
**Relevance to YAVLA**: The config-driven `build()` pattern is directly applicable. But we prefer dataclass configs over raw dicts for type safety.

### 5.3 timm (Flat Function Registry)

**Pattern**: Simple flat registry with `register_model` decorator and `create_model()` factory.

```python
@register_model
def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        load_pretrained(model, ...)
    return model

# Usage
model = timm.create_model('resnet50', pretrained=True, num_classes=10)
```

**Strengths**: Dead simple. One function per model. Easy to add new models.
**Weaknesses**: No structured config. kwargs are untyped. No composition (each factory function builds the full model).
**Relevance to YAVLA**: Too flat for our needs — we need composable sub-components, not monolithic model factories.

### 5.4 LeRobot (Most Directly Relevant)

**Pattern**: `draccus.ChoiceRegistry` + `PreTrainedPolicy` base class + convention-based plugin discovery.

```python
# Config registration via draccus
@PreTrainedConfig.register_subclass("act")
@dataclass
class ACTConfig(PreTrainedConfig):
    chunk_size: int = 100
    ...

# Policy base class with contract enforcement
class PreTrainedPolicy(nn.Module):
    config_class: type[PreTrainedConfig]  # Must be set by subclass

    def __init_subclass__(cls, **kwargs):
        # Enforces that every subclass defines config_class
        ...

# Convention-based discovery
# configuration_act.py → modeling_act.py
# The config file name implies the modeling file name
```

**Key features**:
- **Dataclass configs with discriminated unions**: `draccus.ChoiceRegistry` enables `PreTrainedConfig` to be a union type — the `type` field selects which subclass to instantiate. This is exactly what tyro needs for CLI composition.
- **Contract enforcement via `__init_subclass__`**: Every policy subclass MUST define `config_class`, `select_action()`, and `forward()`. Violations are caught at class definition time, not runtime.
- **Convention-based discovery**: `configuration_X.py` implies `modeling_X.py`. No explicit registration needed — just follow the naming convention.

**Strengths**: Type-safe configs. Contract enforcement. Convention over configuration. Directly compatible with YAVLA's tyro-based config system.
**Weaknesses**: Convention-based discovery is implicit — new contributors may not know the naming rules.
**Relevance to YAVLA**: **This is our primary reference.** YAVLA should adopt LeRobot's `PreTrainedPolicy` pattern with dataclass configs and `__init_subclass__` contracts.

### 5.5 MoE as Modularity

**Pattern**: Mixture-of-Experts routers as a modularity mechanism — dynamically selecting which "expert" (sub-network) processes each input.

```python
class MoELayer(nn.Module):
    def __init__(self, experts: list[nn.Module], router: nn.Module):
        self.experts = nn.ModuleList(experts)
        self.router = router  # Produces routing weights

    def forward(self, x):
        weights = self.router(x)  # [batch, num_experts]
        # Top-k expert selection + weighted combination
        ...
```

**Relevance to YAVLA**: MoE is relevant for multi-embodiment scenarios — different robots could activate different experts within a shared trunk. Not a primary pattern for YAVLA v1, but worth keeping in mind for scaling.

---

## 6. The Modularity–Performance Tension

This is the central design challenge for YAVLA. The best-performing VLA (π0) achieves its results through tight coupling between backbone and action head. The most modular VLA (Octo) uses readout tokens that create a clean interface but lose information.

### The Spectrum

```
← More Modular                                    More Performant →

Octo (readout tokens)  →  OpenVLA-OFT (linear head)  →  π0 (dual-expert shared attention)
     ✅ Swap any head         ✅ Swap head easily          ❌ Head is fused with backbone
     ⚠️ Info bottleneck       ⚠️ No multimodality          ✅ Best manipulation results
```

### Why π0 is Hard to Modularize

In π0, the action expert's tokens participate in a **block-causal joint attention** with the VLM's tokens. At every transformer layer:

1. Prefix tokens (vision + text) attend bidirectionally among themselves but **cannot see** suffix tokens
2. Suffix tokens (actions + state) attend bidirectionally among themselves **and** to all prefix tokens
3. Both streams have **separate FFN weights** but **shared attention keys/values**

This asymmetric flow means the action expert reads VLM context at every layer. While the VLM prefix is technically independent (it can't see actions), the action expert's architecture is deeply entangled with the VLM's hidden states. Remove the action expert → you lose the suffix stream entirely → need to retrain any replacement to match the prefix interface.

### The Resolution: Two Integration Levels

Oracle's key insight: **don't force a single interface. Offer two modes.**

**Level 1 — Readout Mode (default, max modularity)**:
- Backbone appends learned readout tokens to its sequence
- After processing, extracts `readout_states: Tensor[B, N_readout, D]`
- Any action head consumes `readout_states` — zero coupling to backbone internals
- Performance: ~90% of joint-attention mode (Octo demonstrates this is viable)

**Level 2 — Joint-Token Mode (opt-in, max performance)**:
- Action head requests backbone to include action tokens in its attention via `integration_request()`
- Backbone validates compatibility at build time (not all backbones support this)
- Only specific (backbone, head) pairs work in this mode — factory enforces valid combinations
- Performance: 100% (π0-level)

```python
class BackboneCapabilities:
    supports_joint_tokens: bool = False  # Can action tokens join attention?
    supports_readout: bool = True        # Can extract readout states?

class ActionHeadRequirements:
    needs_joint_tokens: bool = False     # Requires joint attention?
    accepts_readout: bool = True         # Can work with readout states?

# Factory validates at build time:
# if head.needs_joint_tokens and not backbone.supports_joint_tokens → ERROR
# if head.accepts_readout and backbone.supports_readout → OK (readout mode)
```

**Why this works**: Most action heads (diffusion, CVAE, MLP, VQ-BeT) only need a conditioning vector — readout mode is perfect. Only flow matching with dual-expert (π0-style) needs joint-token mode. By making joint-token opt-in, we get modularity by default and performance when needed.

**Honest tradeoff**: This means YAVLA needs at least **2 backbone classes** — a standard VLM backbone (readout only) and a dual-expert backbone (supports joint tokens). This is acceptable complexity for the performance gain.

---

## 7. Proposed YAVLA Architecture

> **Revision note**: This section was revised following a structured multi-agent architecture review (4 specialist architects × 2 debate rounds → unanimous consensus on 15 changes). See Appendix A for the full debate record.

### 7.1 The 7 Core Modules

```
┌──────────────────────────────────────────────────────────────────┐
│                        YAVLA Policy                              │
│                                                                  │
│  ┌───────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ 1. Observation │  │ 2. Vision    │  │ 3. Proprio Encoder     │ │
│  │    Adapter     │─▶│    Encoder   │  │    + Language Encoder   │ │
│  │                │  │              │  │                        │ │
│  │ Normalizes,    │  │ SigLIP,DINO, │  │ MLP projection (robot- │ │
│  │ timestamps,    │  │ Dual, Custom │  │ specific) + tokenizer  │ │
│  │ canonical repr │  │ + LoRA adapt │  │ contract               │ │
│  └───────────────┘  └──────┬───────┘  └───────┬────────────────┘ │
│                            │                   │                  │
│                     ┌──────▼───────────────────▼──────┐           │
│                     │ 4. Token Merger + Resampler     │           │
│                     │    Default: Perceiver → 64 tok  │           │
│                     │    + context tokens (robot/task) │           │
│                     └──────────────┬──────────────────┘           │
│                                    │                              │
│                     ┌──────────────▼──────────────────┐           │
│                     │ 5. VL Backbone (PEFT default)   │           │
│                     │    PaliGemma / Llama / DualExpert│           │
│                     │                                 │           │
│                     │    Readout mode: → readout_states│           │
│                     │    Joint mode:  → shared attn   │           │
│                     └──────────────┬──────────────────┘           │
│                                    │                              │
│                     ┌──────────────▼──────────────────┐           │
│                     │ 6. Action Head                  │           │
│                     │    FlowMatching / Diffusion /   │           │
│                     │    CVAE / AR / MLP / VQ-BeT     │           │
│                     └──────────────┬──────────────────┘           │
│                                    │                              │
│                     ┌──────────────▼──────────────────┐           │
│                     │ 7. Action Decoder               │           │
│                     │    Unnormalize, chunk→step,     │           │
│                     │    temporal ensemble (optional)  │           │
│                     └─────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────────┘
```

**Module responsibilities**:

1. **Observation Adapter**: Normalizes raw sensor data into a canonical `ObservationBatch` with explicit timestamps and masks. Handles multi-view camera stacking, image cropping/resizing, frame history assembly. Robot-specific — maps heterogeneous sensor data to a canonical schema via `ActionSpaceSpec`/`ProprioSpec`. Formal contract: `encode(raw_obs) → ObservationBatch`.
2. **Vision Encoder**: Pre-trained ViT with configurable adaptation (frozen + late-block LoRA by default). Converts images to patch token sequences. Swappable (SigLIP, DINOv2, dual). Edge deployment profile merges LoRA weights for fully frozen inference.
3. **Proprio Encoder + Language Encoder**: Proprio encoder is a small MLP projecting joint positions/velocities/gripper state to backbone embedding dim (robot-specific, maps to canonical representation). Language encoder provides a formal tokenizer contract — backbone declares whether it consumes `input_ids` vs pre-embedded tokens.
4. **Token Merger + Resampler**: Combines vision tokens, proprio tokens, language tokens, and context tokens (robot embedding, task ID, control mode, camera intrinsics) into a single sequence. **Default: Perceiver-style resampler reducing vision tokens from ~729 to 64** (configurable via `token_budget`). Identity (no reduction) available for research configs with warning when >256 tokens.
5. **VL Backbone**: The core transformer with **PEFT by default** (LoRA on attention+MLP projections; full fine-tune opt-in). Either a standard VLM (readout mode) or dual-expert (joint-token mode). Temporal positional encodings for observation sequences.
6. **Action Head**: Converts backbone output to `ActionPrediction` (mean, samples, log_prob, aux). The most swappable component — any of the 9 paradigms from Section 2. Richer loss contract: `compute_loss(backbone_out, batch, *, rng, mask, loss_cfg) → LossDict`. Accepts `SamplingConfig` for inference.
7. **Action Decoder**: Post-processes `ActionPrediction` from the head into executable `ActionChunk`. Selects the action (mean or sample), unnormalizes to physical units via `ActionSpaceSpec`, applies temporal ensembling (optional, state-aware — disabled on contact/high-force). Formal contract: `decode(pred: ActionPrediction, meta) → ActionChunk`.

### 7.2 Interface Contracts

**Dual-layer interface design** (unanimous architect consensus): `@runtime_checkable` Protocols define the public API contracts for static type checking and third-party extensibility. `*Base(nn.Module, ABC)` base classes provide the "blessed" implementation path with shared utilities, device/dtype management, and `state_dict` lifecycle. The factory accepts either via `isinstance(x, nn.Module)` + Protocol conformance; an `ensure_module()` adapter wraps Protocol-only implementations.

#### 7.2.1 Typed Data Containers

All module boundaries pass typed containers — never raw tensors. This prevents silent shape/mask bugs and enables compile-time validation.

```python
@dataclass
class ObservationBatch:
    images: dict[str, torch.Tensor]     # camera_name → [B, C, H, W]
    proprio: torch.Tensor               # [B, D_proprio]
    language: str | list[str] | None    # raw text (tokenized by LanguageEncoder)
    timestamps: torch.Tensor            # [B] observation timestamps
    masks: torch.Tensor | None          # [B] validity masks
    dt_hz: float | None = None          # sensor capture rate (e.g., 10.0 for 10 Hz cameras)
    chunk_len: int | None = None        # observation history length (if windowed)
    camera_intrinsics: dict[str, torch.Tensor] | None = None

@dataclass
class TokenBatch:
    tokens: torch.Tensor                # [B, N_total, D_backbone]
    attn_mask: torch.Tensor             # [B, N_total]
    pos_ids: torch.Tensor               # [B, N_total]
    modality_ids: torch.Tensor          # [B, N_total] — 0=vision, 1=language, 2=proprio, 3=readout
    readout_indices: torch.Tensor | None  # indices of readout tokens in sequence
    context_tokens: torch.Tensor | None   # robot/task embeddings
    dt_hz: float | None = None            # token-level temporal rate (inherited from source)
    chunk_len: int | None = None          # sequence history length

@dataclass
class BackboneOutput:
    token_states: torch.Tensor | None   # [B, N, D] — full hidden states (joint-token mode)
    readout_states: torch.Tensor | None # [B, N_readout, D] — readout mode
    pooled_state: torch.Tensor | None   # [B, D] — optional pooled representation
    attn_mask: torch.Tensor
    modality_ids: torch.Tensor
    aux: dict[str, torch.Tensor]        # head-specific extras (KV cache, etc.)

@dataclass
class ActionChunk:
    actions: torch.Tensor               # [B, chunk_len, action_dim]
    dt_hz: float                        # control frequency
    chunk_len: int                      # number of steps in chunk
    stride: int                         # execution stride
    t0: torch.Tensor | None             # start timestamp
    action_mask: torch.Tensor | None    # [B, chunk_len] validity

@dataclass
class ActionPrediction:
    mean: torch.Tensor                  # [B, chunk_len, action_dim]
    samples: torch.Tensor | None        # [B, N_samples, chunk_len, action_dim]
    log_prob: torch.Tensor | None       # [B]
    aux: dict[str, torch.Tensor]        # head-specific (e.g., flow trajectory)

@dataclass
class LossDict:
    total: torch.Tensor                 # scalar loss for backward()
    breakdown: dict[str, torch.Tensor]  # per-component losses for logging

@dataclass
class TrainingBatch:
    observations: ObservationBatch
    actions: torch.Tensor               # [B, chunk_len, action_dim] ground-truth
    action_mask: torch.Tensor | None    # [B, chunk_len] validity (variable-length episodes)
    dt_hz: float                        # control frequency for this batch
    chunk_len: int                      # action chunk length
    embodiment_id: str | None = None    # for multi-embodiment routing
```

#### 7.2.2 Module Protocols + Base Classes

```python
from typing import Protocol, runtime_checkable
from abc import ABC, abstractmethod
import torch.nn as nn

# --- Protocols (public API contracts) ---

@runtime_checkable
class VisionEncoderProto(Protocol):
    @property
    def output_dim(self) -> int: ...
    @property
    def num_patches(self) -> int: ...
    def encode_images(self, images: dict[str, torch.Tensor] | torch.Tensor) -> TokenBatch: ...

@runtime_checkable
class ObservationAdapterProto(Protocol):
    def encode(self, raw_obs: dict) -> ObservationBatch: ...

@runtime_checkable
class TokenMergerProto(Protocol):
    @property
    def token_budget(self) -> int | None: ...
    def forward(self, vision_tokens: torch.Tensor, proprio_tokens: torch.Tensor,
                language_tokens: torch.Tensor, context: dict | None) -> TokenBatch: ...

@runtime_checkable
class BackboneProto(Protocol):
    @property
    def capabilities(self) -> "BackboneCapabilities": ...
    @property
    def hidden_dim(self) -> int: ...
    def forward(self, tokens: TokenBatch,
                integration_request: "IntegrationRequest | None" = None) -> BackboneOutput: ...

@runtime_checkable
class ActionHeadProto(Protocol):
    @property
    def requirements(self) -> "ActionHeadRequirements": ...
    def compute_loss(self, backbone_output: BackboneOutput, batch: TrainingBatch, *,
                     rng: torch.Generator | None, mask: torch.Tensor | None,
                     loss_cfg: dict | None) -> LossDict: ...
    def predict(self, backbone_output: BackboneOutput,
                sampling_cfg: "SamplingConfig | None" = None) -> ActionPrediction: ...

@runtime_checkable
class ActionDecoderProto(Protocol):
    @property
    def action_space_spec(self) -> "ActionSpaceSpec": ...
    def decode(self, pred: ActionPrediction, meta: dict | None) -> ActionChunk: ...

# --- Base Classes (blessed implementation path) ---

class VisionEncoderBase(nn.Module, ABC):
    @property
    @abstractmethod
    def output_dim(self) -> int: ...
    @property
    @abstractmethod
    def num_patches(self) -> int: ...

class BackboneBase(nn.Module, ABC):
    @property
    @abstractmethod
    def capabilities(self) -> "BackboneCapabilities": ...
    @property
    @abstractmethod
    def hidden_dim(self) -> int: ...

class ActionHeadBase(nn.Module, ABC):
    @property
    @abstractmethod
    def requirements(self) -> "ActionHeadRequirements": ...

class PolicyBase(nn.Module, ABC):
    """Minimal contract for all YAVLA policies.
    Concrete subclasses MUST define: name (str) and config_class (type).
    Enforced via __init_subclass__.
    """
    @abstractmethod
    def forward(self, batch: TrainingBatch) -> LossDict: ...
    @abstractmethod
    def predict(self, obs: ObservationBatch) -> ActionChunk: ...
    def reset(self) -> None: pass
    def get_optim_params(self) -> dict: return {"params": self.parameters()}
```

#### 7.2.3 Capability Negotiation (Expanded)

Replaces the original 2-boolean system with a richer contract validated at both build time and load time.

```python
from enum import Enum

class IntegrationMode(Enum):
    READOUT = "readout"           # Head consumes readout_states
    JOINT_TOKENS = "joint_tokens" # Head tokens participate in backbone attention

@dataclass
class BackboneCapabilities:
    supported_modes: set[IntegrationMode]
    supports_kv_cache: bool = False
    supports_temporal_tokens: bool = False
    max_context_tokens: int | None = None
    supports_variable_horizon: bool = False

@dataclass
class ActionHeadRequirements:
    required_mode: IntegrationMode = IntegrationMode.READOUT
    accepts_readout: bool = True
    needs_per_token_states: bool = False
    expected_backbone_output: set[str] = field(default_factory=lambda: {"readout_states"})

@dataclass
class IntegrationRequest:
    mode: IntegrationMode
    num_readouts: int = 64
    readout_init: str = "learned"  # "learned" | "zeros"
    requires_per_token_states: bool = False

# Factory validates at build time:
def validate_integration(backbone: BackboneProto, head: ActionHeadProto) -> IntegrationMode:
    caps = backbone.capabilities
    reqs = head.requirements
    if reqs.required_mode not in caps.supported_modes:
        raise IncompatibleError(
            f"Head requires {reqs.required_mode} but backbone supports {caps.supported_modes}"
        )
    return reqs.required_mode
```

#### 7.2.4 Multi-Embodiment Contracts

```python
@dataclass
class ActionSpaceSpec:
    """Canonical action space description — stored in dataset metadata + checkpoint."""
    names: list[str]          # e.g., ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    units: list[str]          # e.g., ["m/s", "m/s", "m/s", "rad/s", ...]
    limits: torch.Tensor      # [action_dim, 2] — (min, max) per dimension
    frame: str                # "ee_delta" | "joint_delta" | "absolute_ee"
    control_mode: str         # "position" | "velocity" | "impedance"

@dataclass
class ProprioSpec:
    """Canonical proprioception description."""
    names: list[str]          # e.g., ["joint_0", ..., "gripper_pos"]
    units: list[str]
    limits: torch.Tensor
```

#### 7.2.5 Training Pipeline Contract

Canonical training step — all training loops follow this contract regardless of action head or integration mode.

```python
def train_step(policy: PolicyBase, batch: TrainingBatch, optimizer: Optimizer) -> LossDict:
    """Canonical training step — works with any PolicyBase subclass.
    
    VLAPolicy exposes overridable steps:
        encode_observations → merge_tokens → run_backbone → compute_loss / decode_prediction
    Subclasses (e.g., ARTokenPolicy, FlowMatchPolicy) override 1-2 steps.
    """
    # Policy.forward() composes the overridable steps internally
    loss_dict = policy.forward(batch)

    # Backward + optimizer step (PEFT: only trainable params have grad)
    loss_dict.total.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss_dict
```

### 7.3 Action Head Mapping

Which heads work with which integration modes:

| Action Head                | Readout Mode | Joint-Token Mode | Backbone Requirement |
| -------------------------- | ------------ | ---------------- | -------------------- |
| Flow Matching (standalone) | ✅            | ❌                | Any                  |
| Flow Matching (π0-style)   | ❌            | ✅                | DualExpert only      |
| Diffusion (DDPM/DDIM)      | ✅            | ❌                | Any                  |
| CVAE (ACT)                 | ✅            | ❌                | Any                  |
| Autoregressive             | N/A          | N/A              | LM head (built-in)   |
| VQ-BeT                     | ✅            | ❌                | Any                  |
| MLP / L1 Regression        | ✅            | ❌                | Any                  |
| Energy-Based               | ✅            | ❌                | Any                  |

Key takeaway: **7 of 8 heads work in readout mode.** Only π0-style flow matching requires joint-token mode. This validates the two-level strategy — readout is the universal default.

### 7.4 Config-Driven Composition

YAVLA uses dataclass configs with tyro CLI, following LeRobot's pattern. Each module has its own config dataclass. The top-level policy config composes them. **Key changes from original design**: PEFT default, vision tuning config, token budget, training strategy serialization.

```python
from dataclasses import dataclass, field
import tyro

@dataclass
class VisionTuningConfig:
    mode: str = "lora_last_k"      # "frozen" | "lora_last_k" | "full"
    target_blocks: int = 4         # LoRA on last N ViT blocks
    rank: int = 16                 # LoRA rank

@dataclass
class VisionEncoderConfig:
    type: str = "siglip"           # "siglip" | "dinov2" | "dual"
    model_name: str = "google/siglip-so400m-patch14-384"
    extract_layer: int = -2        # Second-to-last layer
    tuning: VisionTuningConfig = field(default_factory=VisionTuningConfig)

@dataclass
class TokenMergerConfig:
    type: str = "resampler"        # "resampler" | "concat" | "cross_attn"
    max_vision_tokens: int = 64    # Perceiver target (from ~729)
    include_context_tokens: bool = True  # robot/task embeddings

@dataclass
class ContextTokenSchema:
    """Schema for context tokens injected into the token sequence by the merger."""
    embodiment_id_dim: int = 64    # learned embedding for robot type
    task_id_dim: int = 64          # learned embedding for task
    control_mode_dim: int = 16     # one-hot or learned (position/velocity/impedance)
    camera_intrinsics_dim: int = 16  # projected intrinsics per view
    # Total context tokens = 1 (embodiment) + 1 (task) + 1 (control) + N_views (intrinsics)
    # Each projected to D_backbone before concatenation

@dataclass
class TrainingStrategyConfig:
    backbone_peft: bool = True     # LoRA on backbone attention+MLP (DEFAULT)
    backbone_lora_rank: int = 32
    full_finetune: bool = False    # Explicit opt-in
    # Serialized into checkpoint metadata for reproducibility

@dataclass
class BackboneConfig:
    type: str = "dual_expert"      # "vlm" | "dual_expert"
    vlm_name: str = "google/paligemma-3b-pt-224"
    action_expert_dim: int = 768
    action_expert_layers: int = 12
    num_readout_tokens: int = 64

@dataclass
class FlowMatchingHeadConfig:
    num_euler_steps: int = 10      # Configurable per-task; distill to 1-2 for edge
    chunk_len: int = 50            # Configurable per-robot/task
    control_hz: float = 50.0
    action_dim: int = 7

@dataclass
class PolicyConfig:
    config_version: int = 1        # For migration support
    vision: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    merger: TokenMergerConfig = field(default_factory=TokenMergerConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    head: FlowMatchingHeadConfig = field(default_factory=FlowMatchingHeadConfig)
    training: TrainingStrategyConfig = field(default_factory=TrainingStrategyConfig)

# CLI: python train.py --vision.tuning.mode frozen --merger.max-vision-tokens 128
config = tyro.cli(PolicyConfig)
```

#### 7.4.1 Registry Pattern

**Two-tier registry** (replaces `__init_subclass__` approach):

```python
# Tier 1: Explicit in-code registries (source of truth)
from yavla.registry import register_vision, register_head, register_backbone

@register_vision("siglip_so400m")
class SigLIPEncoder(VisionEncoderBase): ...

@register_head("flow_matching")
class FlowMatchingHead(ActionHeadBase): ...

# Tier 2: Optional plugin discovery via entry points
# pyproject.toml:
#   [project.entry-points."yavla.plugins"]
#   my_custom_head = "my_package:register"

# Factory loads plugins at startup (disable with --no-plugins)
```

**Registry API** (used by factory, tests, and CLI):

```python
class Registry[T]:
    def register(self, name: str) -> Callable:  ...  # @register decorator
    def build(self, config) -> T:                ...  # instantiate from config
    def list(self) -> list[str]:                 ...  # all registered names
    def get_default_config(self, name: str):     ...  # default config for name
```

### 7.5 Directory Structure

```
src/yavla/models/
├── types.py                    # ObservationBatch, TokenBatch, BackboneOutput, ActionChunk, etc.
├── protocols.py                # PolicyBase ABC, Protocols, Base classes, capability negotiation
├── registry.py                 # Explicit registries: register_vision, register_head, etc.
├── factory.py                  # build_policy(PolicyConfig, deps) → Policy + validation
├── policy.py                   # VLAPolicy(PolicyBase) — 7-module pipeline with overridable steps
│                               # Steps: encode_observations, merge_tokens, run_backbone,
│                               #        compute_loss, decode_prediction
│
├── observation/
│   ├── adapter.py              # ObservationAdapterBase — encode(raw_obs) → ObservationBatch
│   └── specs.py                # ActionSpaceSpec, ProprioSpec, canonical schemas
│
├── encoders/
│   ├── vision_siglip.py        # SigLIP encoder (+ LoRA adaptation support)
│   ├── vision_dinov2.py        # DINOv2 encoder
│   ├── vision_dual.py          # SigLIP + DINOv2 channel-concat (Prismatic)
│   ├── proprio.py              # Proprioception MLP encoder (robot-specific)
│   └── language.py             # LanguageEncoder contract + tokenizer wrappers
│
├── merger/
│   ├── concat.py               # Simple concatenation (research/server profile)
│   ├── resampler.py            # Perceiver-style token resampler (DEFAULT)
│   └── context.py              # Context token injection (robot/task embeddings)
│
├── backbones/
│   ├── vlm.py                  # Standard VLM backbone (readout mode only)
│   ├── dual_expert.py          # π0-style dual-stream (readout + joint-token modes)
│   ├── capabilities.py         # BackboneCapabilities, IntegrationRequest, IntegrationMode
│   └── peft.py                 # PEFT wrappers (LoRA/IA3 application + merge for deploy)
│
├── heads/
│   ├── flow_matching.py        # Flow matching (standalone + π0-style)
│   ├── diffusion.py            # DDPM/DDIM diffusion head
│   ├── autoregressive.py       # AR token prediction (uses LM head)
│   ├── cvae.py                 # ACT-style CVAE
│   ├── mlp.py                  # Direct regression
│   └── vq_bet.py               # VQ-BeT discrete latent
│
├── inference/
│   ├── cache.py                # KV-cache reuse, vision caching, static/dynamic split
│   ├── profiles.py             # DeployProfile: edge (quantized, merged LoRA) vs server
│   └── horizon.py              # Receding horizon controller (5-20Hz replanning)
│
└── configs/
    ├── model.py                # All config dataclasses (PolicyConfig, etc.)
    └── migrations.py           # Config version migrations: migrate_config(dict) → dict
```

### 7.6 Checkpoint & Serialization

Every YAVLA checkpoint is **self-contained and versioned** — you can load it on a different machine with zero external lookups.

**Save / Load Contract:**

```python
class VLAPolicy(PolicyBase):
    def save_pretrained(self, path: Path) -> None: ...
    @classmethod
    def from_pretrained(cls, path: Path, *, strict: bool = True) -> "VLAPolicy": ...
```

**Checkpoint Layout:**

```
checkpoint/
├── config.json              # Full PolicyConfig, includes config_version
├── model.safetensors        # Weights (safetensors format, no pickle)
├── obs_stats.json           # Observation normalization statistics
├── action_stats.json        # Action normalization statistics
├── embodiment.json          # ActionSpaceSpec + ProprioSpec for this embodiment
└── capabilities.json        # IntegrationMode, supported features, model metadata
```

**Key Design Choices:**

- **`config_version: int`** in every config — `from_pretrained` runs `migrate_config()` to upgrade old configs automatically
- **`strict=True` default** — missing or unexpected keys raise errors; opt-in `strict=False` for transfer learning
- **`safetensors` only** — no pickle, no arbitrary code execution on load
- **`push_to_hub()` helper** — uploads checkpoint + model card to HuggingFace Hub with embodiment metadata tags
- **Embodiment metadata** — `embodiment.json` records the exact action/proprio specs so mismatched loading fails fast with a clear error

**Load-Time Validation (`from_pretrained` checks):**

```python
@classmethod
def from_pretrained(cls, path: Path, *, strict: bool = True) -> "VLAPolicy":
    config = load_and_migrate_config(path / "config.json")
    caps = json.load(open(path / "capabilities.json"))
    embod = json.load(open(path / "embodiment.json"))

    # 1. Config version — migrate if needed, fail if unsupported
    if config["config_version"] > CURRENT_VERSION:
        raise VersionError(f"Checkpoint v{config['config_version']} > supported v{CURRENT_VERSION}")

    # 2. Integration mode — verify backbone supports the saved mode
    saved_mode = IntegrationMode(caps["integration_mode"])
    backbone = build_backbone(config)
    if saved_mode not in backbone.capabilities.supported_modes:
        raise IncompatibleError(f"Checkpoint requires {saved_mode}, backbone supports {backbone.capabilities.supported_modes}")

    # 3. Embodiment — warn or fail on action space mismatch
    if strict:
        validate_action_space(embod["action_space"], expected_spec)
        validate_proprio_space(embod["proprio"], expected_spec)

    # 4. Weight shape — safetensors keys must match model architecture
    policy = cls(config)
    load_safetensors(policy, path / "model.safetensors", strict=strict)
    return policy
```

### 7.7 Inference & Deployment

Real-time robot control requires careful latency management. YAVLA provides three layers of inference optimization.

**Receding Horizon Control:**

The action head predicts chunks (e.g., 50 steps at 50Hz = 1 second), but the robot replans every 50–200ms (5–20Hz). Only the first *k* actions execute before the next prediction overwrites the buffer. This provides temporal smoothness while staying reactive.

```python
@dataclass
class HorizonConfig:
    chunk_len: int = 50          # Actions per prediction
    replan_hz: float = 10.0      # How often to replan
    blend_steps: int = 4         # Overlap blending between consecutive chunks
```

**Caching Strategy (Static / Dynamic Split):**

| Component                  | Cache Policy                     | Refresh Rate          |
| -------------------------- | -------------------------------- | --------------------- |
| Vision encoder output      | Static within replan window      | 5–10 Hz (camera rate) |
| Language tokens / KV-cache | Static until instruction changes | On new instruction    |
| Action expert KV-cache     | Dynamic, rebuilt each replan     | 5–20 Hz               |

The vision encoder is the most expensive forward pass. By caching its output and only rerunning when a new camera frame arrives, inference cost drops ~40–60%.

**Deployment Profiles:**

```python
class DeployProfile(str, Enum):
    EDGE = "edge"        # INT8 quantized, LoRA merged into base, static CUDA graphs
    SERVER = "server"    # FP16/BF16, FlashAttention, dynamic batching
    RESEARCH = "research"  # FP32, no fusion, full logging, gradient checkpointing
```

Each profile configures: dtype, attention backend, whether LoRA is merged or kept separate, compilation strategy (`torch.compile` / CUDA graphs), and logging verbosity.

**Compilation Compatibility:**

- All modules use **static shapes** where possible (fixed chunk_len, fixed token count after resampler)
- `torch.compile(mode="reduce-overhead")` for server profile
- CUDA graphs for edge profile (requires fully static computation graph)
- FlashAttention-2/3 as attention backend when available

### 7.8 Testing Strategy

Four categories of automated tests ensure module contracts hold as the codebase evolves.

**1. Contract Tests** — Verify every module satisfies its Protocol:

```python
def test_vision_encoder_contract(encoder: VisionEncoderProto):
    """Any VisionEncoder must: accept image batch → return TokenBatch with correct shape."""
    imgs = torch.randn(2, 3, 224, 224)
    out = encoder.encode_images(imgs)
    assert isinstance(out, TokenBatch)
    assert out.tokens.shape[0] == 2  # batch preserved
```

**2. Serialization Round-Trip** — Save → load → compare outputs:

```python
def test_checkpoint_roundtrip(policy, tmp_path):
    policy.save_pretrained(tmp_path)
    loaded = type(policy).from_pretrained(tmp_path)
    assert_outputs_close(policy, loaded, atol=1e-6)
```

**3. Registry Discovery** — All registered modules are importable and instantiable:

```python
@pytest.mark.parametrize("name", ActionHeadRegistry.list())
def test_action_head_instantiable(name):
    cfg = ActionHeadRegistry.get_default_config(name)
    head = ActionHeadRegistry.build(cfg)
    assert isinstance(head, ActionHeadBase)
```

**4. Shape & Signature Tests** — Verify tensor shapes through the full pipeline:

```python
def test_full_forward_shapes(policy):
    obs = make_dummy_observation(batch=4, views=2)
    out = policy(obs)
    assert out.actions.shape == (4, policy.config.chunk_len, policy.config.action_dim)
```

These tests run in CI on every PR. Contract tests are **mandatory** — a new module that fails its Protocol test cannot be merged.

---

## 8. Design Decisions & Tradeoffs

Nine key decisions that shape YAVLA's architecture, with our recommendations and reasoning. Decisions 1–6 are from the original design; Decisions 7–9 were added after the multi-agent architecture review (4 specialist architects, 2 debate rounds, unanimous consensus).

### Decision 1: Default Action Head

**Options**: Flow Matching (π0-style) vs Diffusion vs CVAE vs AR tokens

**Recommendation**: Flow Matching with two variants — **standalone (readout mode)** as the default, **π0-style dual-expert (joint-token mode)** as opt-in for maximum quality.

**Reasoning**: Flow matching offers the best latency/quality tradeoff (5–10 Euler steps vs 10–100 for diffusion). The standalone variant works in readout mode (compatible with any backbone), making it the safe default. The π0-style dual-expert variant requires joint-token mode and a DualExpert backbone, but achieves state-of-the-art results — available as an explicit upgrade path when users configure `backbone.type: "dual_expert"`.

**Fallback**: Diffusion (DDIM) as the second-priority head. Well-studied, good multimodality, works in readout mode.

### Decision 2: Vision Encoder

**Options**: SigLIP only vs DINOv2 only vs Dual (SigLIP + DINOv2); Frozen vs Fine-tuned vs Frozen + LoRA

**Recommendation**: SigLIP So400m/14 as default, dual-encoder as opt-in. **Frozen trunk with late-block LoRA** (last 2–4 ViT blocks) as the default tuning strategy.

**Reasoning**: SigLIP provides language-aligned features essential for instruction following. DINOv2 adds spatial precision but doubles token count. Start with SigLIP; add DINOv2 when spatial tasks demand it. Extract from second-to-last layer (OpenVLA finding). Freezing the vision trunk preserves pretrained representations while LoRA on the last 2–4 blocks allows task-specific adaptation at minimal parameter cost. Edge deployment profiles merge LoRA weights into the base for zero overhead.

### Decision 3: VL Fusion Strategy

**Options**: Tokens-in-sequence vs Cross-attention vs FiLM vs Readout tokens

**Recommendation**: Tokens-in-sequence (default) with readout tokens for decoupled heads

**Reasoning**: Tokens-in-sequence is simplest and used by the best-performing models (π0, OpenVLA). Readout tokens are added as the backbone-to-head interface for readout mode. No need for FiLM or cross-attention — they add complexity without clear benefit for our architecture.

### Decision 4: Action Chunk Size

**Options**: Single-step (1) vs Short chunk (5–16) vs Long chunk (50–100)

**Recommendation**: 50 steps at 50Hz (1 second), matching π0

**Reasoning**: Longer chunks capture temporal coherence and reduce inference frequency. π0's 50-step chunks at 50Hz are well-validated. Shorter chunks (ACT's 100 steps) work too but require temporal ensembling. The chunk size should be configurable per-task.

### Decision 5: Registry Pattern

**Options**: PEFT-style dicts vs MMEngine hierarchical registry vs LeRobot draccus + convention vs Explicit registries + entry points

**Recommendation**: Explicit in-code registries as source of truth, with optional Python entry points for third-party plugins

**Reasoning**: Each module family (action heads, vision encoders, backbones, token mergers) gets a typed `Registry[ConfigT, ModuleT]` dict that maps string names to `(config_class, module_class)` pairs. This is the single source of truth — no magic `__init_subclass__`, no convention-based file discovery. Third-party extensions register via standard Python `entry_points` in `pyproject.toml`, which the registry discovers at import time. This gives full type safety, explicit control, and ecosystem extensibility without import-time side effects.

### Decision 6: Backbone Training Strategy

**Options**: Full freeze vs LoRA/PEFT fine-tune vs Full fine-tune

**Recommendation**: PEFT default (LoRA rank 16–64 on backbone attention + MLP layers), full fine-tune as explicit opt-in

**Reasoning**: Full fine-tuning a 3B VLM backbone is expensive and risks catastrophic forgetting of pretrained knowledge. LoRA/IA³ on attention and MLP layers provides 90%+ of full fine-tune quality at ~5% of trainable parameters. This is the default training strategy. Full fine-tune is available as an explicit opt-in (`training_strategy: "full"` in config) for users with sufficient compute who need maximum adaptation. The action expert is always trained from scratch regardless of backbone strategy.

### Decision 7: Vision Token Reduction

**Options**: No reduction (raw ViT tokens) vs Adaptive pooling vs Perceiver resampler vs Convolutional downsampling

**Recommendation**: Perceiver resampler to 64 tokens (default-on), identity bypass for research

**Reasoning**: A SigLIP So400m/14 at 384×384 (our default) produces ~729 tokens per view. With 2 views, that's ~1458 vision tokens competing for backbone context length. A Perceiver resampler (cross-attention with 64 learned queries) compresses this ~11× with minimal information loss — the learned queries attend to the most task-relevant spatial features. Default-on because the latency and memory savings are substantial. Research profile can bypass the resampler (`token_merger: "identity"`) with a warning when total vision tokens exceed 256.

### Decision 8: Temporal Contracts

**Options**: Ad-hoc timestamp handling vs First-class time semantics in all data containers

**Recommendation**: First-class temporal contracts — timestamps, temporal masks, `dt_hz`, and `chunk_len` in all data containers

**Reasoning**: Robotics data is inherently temporal. Without explicit time semantics, every module reinvents timestamp alignment, frame dropping, and chunk boundary logic. By embedding `timestamps_s: Tensor`, `temporal_mask: Tensor`, `dt_hz: float`, and `chunk_len: int` into `ObservationBatch`, `TokenBatch`, and `ActionChunk`, temporal reasoning becomes a shared contract rather than per-module ad-hoc code. This also enables proper multi-rate fusion (vision at 10Hz, proprio at 100Hz) without hacks.

### Decision 9: Checkpoint Serialization

**Options**: PyTorch `state_dict` only vs HuggingFace-style `save_pretrained` vs Custom format

**Recommendation**: Versioned, self-contained checkpoints with `save_pretrained` / `from_pretrained`

**Reasoning**: A checkpoint must be loadable on any machine without external lookups. Every checkpoint includes `config.json` (with `config_version`), `model.safetensors`, normalization stats, embodiment specs, and capability metadata. Config migrations handle version skew automatically. `safetensors` format eliminates pickle-based code execution risks. This follows the HuggingFace convention that the community already knows, while adding robotics-specific metadata (embodiment, action space) that HuggingFace doesn't provide.

---

## 9. References

### Action Heads & Core Methods

- Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion," RSS 2023
- Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control," 2024
- Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT), RSS 2023
- Lee et al., "Behavior Generation with Latent Actions" (VQ-BeT), ICML 2024
- Florence et al., "Implicit Behavioral Cloning," CoRL 2021
- Du et al., "Learning Universal Policies via Text-Guided Video Generation" (UniPi), NeurIPS 2023
- Black et al., "Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models" (SuSIE), 2023

### VLA Models

- Brohan et al., "RT-1: Robotics Transformer for Real-World Control at Scale," RSS 2023
- Brohan et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control," CoRL 2023
- Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model," CoRL 2024
- Pertsch et al., "FAST: Efficient Action Tokenization for Vision-Language-Action Models," 2025
- Octo Model Team, "Octo: An Open-Source Generalist Robot Policy," RSS 2024
- Li et al., "Vision-Language Foundation Models as Effective Robot Imitators" (RoboFlamingo), ICLR 2024

### Cutting-Edge Architectures

- Wang et al., "Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers" (HPT), NeurIPS 2024
- Cheang et al., "GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation," ByteDance 2024
- Open X-Embodiment Collaboration, "Open X-Embodiment: Robotic Learning Datasets and RT-X Models," 2024
- Doshi et al., "Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion and Aviation" (CrossFormer), CoRL 2024
- Reed et al., "A Generalist Agent" (Gato), DeepMind 2022
- Qu et al., "SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model," 2025
- Kim et al., "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success" (OpenVLA-OFT), 2025
- Ke et al., "3D Diffuser Actor: Policy Diffusion with 3D Scene Representations," CoRL 2024

### Vision Encoders & Foundations

- Zhai et al., "Sigmoid Loss for Language Image Pre-Training" (SigLIP), ICCV 2023
- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," 2023
- Karamcheti et al., "Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models," ICML 2024
- Beyer et al., "PaliGemma: A Versatile 3B VLM for Transfer," Google 2024

### Frameworks & Patterns

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022
- PEFT library, Hugging Face, https://github.com/huggingface/peft
- MMEngine, OpenMMLab, https://github.com/open-mmlab/mmengine
- timm, Ross Wightman, https://github.com/huggingface/pytorch-image-models
- LeRobot, Hugging Face, https://github.com/huggingface/lerobot

### Surveys

- Zhen et al., "A Survey on Vision-Language-Action Models for Embodied AI," 2024
- Xiao et al., "Robot Learning in the Era of Foundation Models: A Survey," 2024
