"""VLAPolicy module, build_policy factory, and serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from yavla.models.config import PolicyConfig
from yavla.models.protocols import (
    ActionDecoderBase,
    ActionHeadBase,
    BackboneBase,
    ProprioEncoderBase,
    VisionEncoderBase,
    validate_integration,
)
from yavla.models.types import (
    ActionChunk,
    LossDict,
    ObservationBatch,
    TrainingBatch,
)


class VLAPolicy(nn.Module):
    def __init__(
        self,
        vision_encoder: VisionEncoderBase,
        proprio_encoder: ProprioEncoderBase,
        merger: nn.Module,
        backbone: BackboneBase,
        action_head: ActionHeadBase,
        decoder: ActionDecoderBase,
        config: PolicyConfig,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.proprio_encoder = proprio_encoder
        self.merger = merger
        self.backbone = backbone
        self.action_head = action_head
        self.decoder = decoder
        self.config = config

    def _encode_and_merge(self, obs: ObservationBatch) -> tuple[Tensor, Tensor, Tensor]:
        image_embeds = self.vision_encoder.encode_images(obs.images)
        proprio_embeds = self.proprio_encoder.encode_proprio(obs.proprio)

        # Tokenize language via backbone's tokenizer, embed via unwrapped base_model
        lang = obs.language or ""
        if isinstance(lang, str):
            lang = [lang]
        tok_out = self.backbone.tokenizer(lang, return_tensors="pt", padding=True)
        input_ids = tok_out["input_ids"].to(image_embeds.device)
        language_attn_mask = tok_out["attention_mask"].to(image_embeds.device)
        lang_embeds = self.backbone.base_model.get_input_embeddings()(input_ids)

        return self.merger.merge(image_embeds, proprio_embeds, lang_embeds, language_attn_mask)

    def forward(self, batch: TrainingBatch) -> LossDict:
        inputs_embeds, attention_mask, token_type_ids = self._encode_and_merge(batch.observations)
        backbone_output = self.backbone(inputs_embeds, attention_mask, token_type_ids)
        return self.action_head.compute_loss(backbone_output, batch)

    @torch.no_grad()
    def predict(self, obs: ObservationBatch) -> ActionChunk:
        inputs_embeds, attention_mask, token_type_ids = self._encode_and_merge(obs)
        backbone_output = self.backbone(inputs_embeds, attention_mask, token_type_ids)
        prediction = self.action_head.predict(backbone_output)
        return self.decoder.decode(prediction)

    def _has_lora(self) -> bool:
        """Check whether the backbone model is wrapped with peft."""
        return hasattr(self.backbone.model, "peft_config")

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Config
        import dataclasses

        cfg_dict = dataclasses.asdict(self.config)
        # Convert tensors in config to lists for JSON serialization
        _tensor_to_list(cfg_dict)
        with open(path / "config.json", "w") as f:
            json.dump(cfg_dict, f, indent=2)

        # Model weights — full state dict (always saved as fallback)
        from safetensors.torch import save_file

        save_file(dict(self.state_dict()), str(path / "model.safetensors"))

        # Additionally save LoRA adapter via peft's native format
        lora_active = self._has_lora()
        if lora_active:
            adapter_dir = path / "adapter"
            adapter_dir.mkdir(exist_ok=True)
            self.backbone.model.save_pretrained(str(adapter_dir))

            # Save non-VLM module weights separately for adapter-only loading
            non_vlm_state = {}
            for name, param in self.state_dict().items():
                if not name.startswith("backbone."):
                    non_vlm_state[name] = param
            save_file(non_vlm_state, str(path / "non_vlm_weights.safetensors"))

        # Checkpoint metadata
        with open(path / "checkpoint_meta.json", "w") as f:
            json.dump({"has_lora": lora_active}, f, indent=2)

        # Action stats (placeholder — populated during training)
        with open(path / "action_stats.json", "w") as f:
            json.dump({}, f)

        # Embodiment info
        embodiment = {
            "action_dim": self.config.action_head.action_dim,
            "proprio_dim": self.config.proprio_encoder.proprio_dim,
            "action_space_names": self.config.action_space.names,
        }
        with open(path / "embodiment.json", "w") as f:
            json.dump(embodiment, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str | Path, strict: bool = True) -> VLAPolicy:
        path = Path(path)

        with open(path / "config.json") as f:
            cfg_dict = json.load(f)

        # Reconstruct config — handle ActionSpaceSpec.limits tensor
        limits = cfg_dict.get("action_space", {}).get("limits")
        if limits is not None:
            cfg_dict["action_space"]["limits"] = torch.tensor(limits)

        config = _dict_to_config(cfg_dict)

        # Check embodiment compatibility
        with open(path / "embodiment.json") as f:
            embodiment = json.load(f)
        if strict and embodiment["action_dim"] != config.action_head.action_dim:
            raise ValueError(
                f"Embodiment mismatch: checkpoint action_dim={embodiment['action_dim']}, "
                f"config action_dim={config.action_head.action_dim}"
            )

        # Read checkpoint metadata (backward-compat: missing file → no LoRA)
        meta_path = path / "checkpoint_meta.json"
        has_lora = False
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            has_lora = meta.get("has_lora", False)

        policy = build_policy(config)

        from safetensors.torch import load_file

        if has_lora and (path / "adapter").exists():
            # Load LoRA adapter via peft
            import peft

            # Replace backbone's model with adapter loaded onto base model
            policy.backbone.model = peft.PeftModel.from_pretrained(
                policy.backbone.base_model,
                str(path / "adapter"),
            )
            policy.backbone.model.enable_input_require_grads()

            # Load non-VLM module weights (action head, proprio encoder, merger)
            if (path / "non_vlm_weights.safetensors").exists():
                non_vlm_state = load_file(str(path / "non_vlm_weights.safetensors"))
                # Only load matching keys into the policy
                policy_state = policy.state_dict()
                for key, value in non_vlm_state.items():
                    if key in policy_state:
                        policy_state[key] = value
                policy.load_state_dict(policy_state, strict=False)
        else:
            # No LoRA — load full state dict
            state = load_file(str(path / "model.safetensors"))
            policy.load_state_dict(state, strict=strict)

        return policy


def build_policy(config: PolicyConfig) -> VLAPolicy:
    from transformers import AutoModelForVision2Seq, AutoProcessor

    from yavla.models.backbone import VLMBackbone
    from yavla.models.decoder import SimpleActionDecoder
    from yavla.models.encoders.proprio import ProprioEncoder, ProprioEncoderConfig
    from yavla.models.encoders.vision import PaliGemmaVisionEncoder
    from yavla.models.heads.mlp import MLPRegressionHead
    from yavla.models.merger import ConcatMerger

    # 1. Load PaliGemma
    base_model = AutoModelForVision2Seq.from_pretrained(
        config.backbone.vlm_name,
        torch_dtype=torch.float32,
    )
    tokenizer = AutoProcessor.from_pretrained(config.backbone.vlm_name).tokenizer

    # 2. Store unwrapped ref
    unwrapped = base_model

    # 3. Freeze modules
    for name in config.freeze.freeze_modules:
        for param_name, param in base_model.named_parameters():
            if param_name.startswith(name):
                param.requires_grad_(False)

    # 4. Apply LoRA if configured
    peft_model = None
    if config.freeze.lora_target_modules:
        import peft

        lora_config = peft.LoraConfig(
            target_modules=config.freeze.lora_target_modules,
            r=config.freeze.lora_r,
            lora_alpha=config.freeze.lora_alpha,
            lora_dropout=config.freeze.lora_dropout,
        )
        peft_model = peft.get_peft_model(base_model, lora_config)
        # 5. Enable input require grads for gradient flow
        peft_model.enable_input_require_grads()

    # 6-7. Gradient checkpointing
    if config.backbone.gradient_checkpointing:
        unwrapped.config.use_cache = False
        unwrapped.gradient_checkpointing_enable()

    # Build backbone
    model_for_forward = peft_model if peft_model is not None else base_model
    backbone = VLMBackbone(
        model=model_for_forward,
        tokenizer_instance=tokenizer,
        num_readout_tokens=config.merger.num_readout_tokens,
    )
    backbone.base_model = unwrapped

    # Build vision encoder with unwrapped base_model ref
    vision_encoder = PaliGemmaVisionEncoder(base_model=unwrapped, config=config.vision_encoder)

    # Build proprio encoder
    proprio_encoder = ProprioEncoder(
        config=ProprioEncoderConfig(
            type=config.proprio_encoder.type,
            proprio_dim=config.proprio_encoder.proprio_dim,
            backbone_dim=backbone.hidden_dim,
        )
    )

    # Build merger
    merger = ConcatMerger(config=config.merger, backbone_dim=backbone.hidden_dim)

    # Build action head
    action_head = MLPRegressionHead(config=config.action_head, backbone_dim=backbone.hidden_dim)

    # 10. Validate integration
    validate_integration(backbone, action_head)

    # Build decoder
    decoder = SimpleActionDecoder(action_space_spec=config.action_space, dt_hz=config.dt_hz)

    return VLAPolicy(
        vision_encoder=vision_encoder,
        proprio_encoder=proprio_encoder,
        merger=merger,
        backbone=backbone,
        action_head=action_head,
        decoder=decoder,
        config=config,
    )


def _tensor_to_list(d: dict) -> None:
    """Recursively convert Tensor values to lists for JSON serialization."""
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.tolist()
        elif isinstance(v, dict):
            _tensor_to_list(v)


def _dict_to_config(d: dict) -> PolicyConfig:
    """Reconstruct PolicyConfig from a dict (loaded from JSON)."""
    from yavla.models.backbone import BackboneConfig
    from yavla.models.encoders.proprio import ProprioEncoderConfig
    from yavla.models.encoders.vision import VisionEncoderConfig
    from yavla.models.heads.mlp import MLPHeadConfig
    from yavla.models.merger import TokenMergerConfig
    from yavla.models.types import ActionSpaceSpec, FreezeConfig, ProprioSpec

    return PolicyConfig(
        vision_encoder=VisionEncoderConfig(**d.get("vision_encoder", {})),
        proprio_encoder=ProprioEncoderConfig(**d.get("proprio_encoder", {})),
        merger=TokenMergerConfig(**d.get("merger", {})),
        backbone=BackboneConfig(**d.get("backbone", {})),
        action_head=MLPHeadConfig(**d.get("action_head", {})),
        freeze=FreezeConfig(**d.get("freeze", {})),
        action_space=ActionSpaceSpec(**d.get("action_space", {"names": [], "units": [], "limits": None})),
        proprio=ProprioSpec(**d.get("proprio", {"names": [], "units": []})),
        dt_hz=d.get("dt_hz", 10.0),
        config_version=d.get("config_version", "1.0"),
    )
