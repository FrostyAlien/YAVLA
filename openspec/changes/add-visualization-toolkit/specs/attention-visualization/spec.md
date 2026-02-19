## ADDED Requirements

### Requirement: Generate attention heatmaps from vision encoder
The system SHALL provide a `generate_attention_heatmap` function that accepts an `nn.Module` (model), a batch of images `(N, C, H, W)`, a target layer `nn.Module`, a method string, and optional `model_forward_kwargs`. It SHALL return spatial heatmaps as a `Tensor` of shape `(N, H_img, W_img)` with values in `[0, 1]`.

#### Scenario: Grad-CAM on SigLIP ViT encoder
- **WHEN** `generate_attention_heatmap(model, images, target_layer, method="grad_cam")` is called
- **THEN** the function SHALL:
  1. Lazy-import `pytorch_grad_cam.GradCAM` (raising `ImportError` with install instructions if missing)
  2. Register forward/backward hooks on the target layer via internal context manager
  3. Run a forward+backward pass with `torch.enable_grad()` and `torch.autocast(enabled=False)`
  4. Compute gradient-weighted activation maps
  5. Reshape ViT patch tokens to spatial grid: strip CLS token (index 0), reshape `(batch, num_patches, hidden_dim)` → `(batch, H_patches, W_patches, hidden_dim)` where `H_patches = W_patches = sqrt(num_patches)`, using pytorch-grad-cam's `reshape_transform`
  6. Upsample to input image resolution via bilinear interpolation
  7. Return heatmaps normalized to `[0, 1]`

#### Scenario: Attention rollout on vision transformer
- **WHEN** `generate_attention_heatmap(model, images, target_layer, method="attention_rollout", model_forward_kwargs={"output_attentions": True})` is called
- **THEN** the function SHALL:
  1. Run forward pass with `output_attentions=True` to get per-layer attention matrices
  2. For each layer, average attention across heads and add identity residual: `A_i = 0.5 * attention_i + 0.5 * I`
  3. Compute rollout: `rollout = A_1 @ A_2 @ ... @ A_L`
  4. Extract CLS-to-patch row from rollout matrix
  5. Reshape to `(H_patches, W_patches)` and upsample to image resolution via bilinear interpolation
  6. No backward pass required

#### Scenario: Invalid method name
- **WHEN** an unsupported method string is passed
- **THEN** the function SHALL raise `ValueError` listing valid methods: `["grad_cam", "attention_rollout"]`

#### Scenario: Gradient isolation during heatmap generation
- **WHEN** `generate_attention_heatmap` is called with `method="grad_cam"` while the model has accumulated training gradients
- **THEN** the function SHALL NOT modify or discard those gradients — callers (e.g., `maybe_generate_snapshot`) are responsible for the full isolation context (save/restore grads, eval mode, autocast off) as specified in design D4

### Requirement: Extract layer embeddings via Captum
The system SHALL provide an `extract_layer_embeddings` function that uses Captum's `LayerActivation` to extract intermediate representations from any specified layer. It SHALL return a `Tensor` of shape `(N, num_tokens, hidden_dim)`.

#### Scenario: Extract SigLIP patch embeddings
- **WHEN** `extract_layer_embeddings(model, images, target_layer)` is called with the last encoder layer
- **THEN** the function SHALL lazy-import `captum.attr.LayerActivation`, extract patch-level embeddings, and return them without modifying model state

#### Scenario: Hook cleanup guaranteed via context manager
- **WHEN** `extract_layer_embeddings` completes (normally or via exception)
- **THEN** no forward hooks SHALL remain registered on the target layer
- **AND** the invariant `len(module._forward_hooks)` before == after SHALL hold
- **AND** no reference cycles SHALL be created (hooks capture only the target layer, not the model)

### Requirement: Compute Integrated Gradients via Captum
The system SHALL provide a `compute_integrated_gradients` function that uses Captum's `IntegratedGradients` to compute per-pixel attribution. It SHALL return a `Tensor` of shape `(N, C, H, W)`.

#### Scenario: Post-hoc attribution analysis
- **WHEN** `compute_integrated_gradients(model, images, target_layer, n_steps=50)` is called
- **THEN** the function SHALL lazy-import `captum.attr.IntegratedGradients`, compute attribution with the specified interpolation steps, and return per-pixel attribution values
- **AND** the caller decides surfacing: save as numpy, log as `wandb.Image` with colormap, or store in FiftyOne

#### Scenario: Hook cleanup after IG computation
- **WHEN** `compute_integrated_gradients` completes (normally or via exception)
- **THEN** the same hook cleanup invariants as `extract_layer_embeddings` SHALL hold

### Requirement: Grad-CAM requires gradients
The `generate_attention_heatmap` function with `method="grad_cam"` SHALL NOT be called inside a `torch.no_grad()` context. The function SHALL temporarily enable gradients if needed and restore the prior gradient state after completion.

#### Scenario: Called during eval mode
- **WHEN** `generate_attention_heatmap` is called while the model is in `eval()` mode
- **THEN** the function SHALL still produce valid heatmaps by enabling gradients temporarily, without switching the model to train mode

### Requirement: HuggingFace model compatibility
The attention visualization functions SHALL work with HuggingFace `transformers` models (SigLIP, PaliGemma) by accepting a `model_forward_kwargs` dict parameter.

#### Scenario: SigLIP with output_attentions
- **WHEN** the model is a HuggingFace SigLIP and attention rollout is requested
- **THEN** the caller SHALL pass `model_forward_kwargs={"output_attentions": True}` and the function SHALL forward these kwargs to the model's forward method

### Requirement: Testability
Unit tests SHALL mock `pytorch_grad_cam` and `captum` imports. Core logic (reshape_transform, rollout matrix multiplication, normalization) SHALL be pure functions testable without GPU or heavy deps. Integration tests requiring actual models SHALL be marked `@pytest.mark.integration`.
