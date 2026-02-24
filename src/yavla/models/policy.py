"""VLAPolicy module, build_policy factory, and serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from yavla.models.config import PolicyConfig
from yavla.models.protocols import (
    ActionDecoderBase,
    ActionHeadBase,
    BackboneBase,
    PolicyBase,
    ProprioEncoderBase,
    TokenMergerBase,
    VisionEncoderBase,
    validate_integration,
)
from yavla.models.types import (
    ActionChunk,
    BackboneOutput,
    LossDict,
    ObservationBatch,
    TrainingBatch,
)


class VLAPolicy(PolicyBase):
    """VLM-based VLA policy composing 7 modules in a linear pipeline.

    The pipeline steps are individual methods that subclasses can override:
        encode_observations → merge_tokens → run_backbone → compute_loss / decode_prediction

    For example, an AR-token policy can override merge_tokens (no readout) and
    compute_loss (cross-entropy from logits). A flow-matching policy can override
    forward to run a denoising loop calling run_backbone multiple times.
    """

    name = "vla"
    config_class = PolicyConfig

    def __init__(
        self,
        vision_encoder: VisionEncoderBase,
        proprio_encoder: ProprioEncoderBase,
        merger: TokenMergerBase,
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

    # ------------------------------------------------------------------
    # Overridable pipeline steps
    # ------------------------------------------------------------------

    def encode_observations(
        self, obs: ObservationBatch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode raw observations into embeddings.

        Returns:
            (image_embeds, proprio_embeds, lang_embeds, lang_attn_mask)
        """
        image_embeds = self.vision_encoder.encode_images(obs.images)
        proprio_embeds = self.proprio_encoder.encode_proprio(obs.proprio)

        # Normalize language to list[str] matching batch size, then delegate to backbone
        batch_size = image_embeds.shape[0]
        lang = obs.language or ""
        if isinstance(lang, str):
            lang = [lang] * batch_size
        lang_embeds, lang_attn_mask = self.backbone.embed_language(lang)

        return image_embeds, proprio_embeds, lang_embeds, lang_attn_mask

    def merge_tokens(
        self,
        image_embeds: Tensor,
        proprio_embeds: Tensor,
        lang_embeds: Tensor,
        lang_attn_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Merge modality embeddings into backbone input.

        Returns:
            (inputs_embeds, attention_mask, token_type_ids)
        """
        return self.merger.merge(image_embeds, proprio_embeds, lang_embeds, lang_attn_mask)

    def run_backbone(
        self, inputs_embeds: Tensor, attention_mask: Tensor, token_type_ids: Tensor
    ) -> BackboneOutput:
        """Run the VLM backbone."""
        result: BackboneOutput = self.backbone(inputs_embeds, attention_mask, token_type_ids)
        return result

    def compute_loss(self, backbone_output: BackboneOutput, batch: TrainingBatch) -> LossDict:
        """Compute training loss from backbone output."""
        return self.action_head.compute_loss(backbone_output, batch)

    def decode_prediction(self, backbone_output: BackboneOutput) -> ActionChunk:
        """Predict and decode an action chunk from backbone output."""
        prediction = self.action_head.predict(backbone_output)
        return self.decoder.decode(prediction)

    # ------------------------------------------------------------------
    # Composed pipeline
    # ------------------------------------------------------------------

    def forward(self, batch: TrainingBatch) -> LossDict:
        img_e, prop_e, lang_e, lang_m = self.encode_observations(batch.observations)
        embeds, attn, ttids = self.merge_tokens(img_e, prop_e, lang_e, lang_m)
        backbone_out = self.run_backbone(embeds, attn, ttids)
        return self.compute_loss(backbone_out, batch)

    @torch.no_grad()
    def predict(self, obs: ObservationBatch) -> ActionChunk:
        img_e, prop_e, lang_e, lang_m = self.encode_observations(obs)
        embeds, attn, ttids = self.merge_tokens(img_e, prop_e, lang_e, lang_m)
        backbone_out = self.run_backbone(embeds, attn, ttids)
        return self.decode_prediction(backbone_out)

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

        # build_policy() dispatches via vlm_registry using config.backbone.type.
        # Pre-refactor checkpoints missing the "backbone" key entirely (or just the
        # "type" sub-key) default to "paligemma" through BackboneConfig's field default.
        try:
            policy = build_policy(config)
        except KeyError as e:
            raise ValueError(
                f"Cannot load checkpoint: unsupported backbone type "
                f"'{config.backbone.type}' in {path / 'config.json'}"
            ) from e

        from safetensors.torch import load_file

        if has_lora and (path / "adapter").exists():
            # NOTE: LoRA loading accesses backbone.model / backbone.base_model directly,
            # which are PaliGemma-specific properties (not on BackboneBase). Loading a
            # LoRA checkpoint with a different backbone type will raise AttributeError.
            # Generalizing this is deferred to per-backbone save/load methods.
            if not hasattr(policy.backbone, "base_model"):
                raise NotImplementedError(
                    f"LoRA checkpoint loading is not supported for backbone type "
                    f"'{config.backbone.type}' — only PaliGemma backbones currently "
                    f"support LoRA adapter loading."
                )
            import peft

            # Replace backbone's model with adapter loaded onto base model
            policy.backbone.model = peft.PeftModel.from_pretrained(
                policy.backbone.base_model,
                str(path / "adapter"),
            )
            policy.backbone.model.enable_input_require_grads()  # pyright: ignore[reportCallIssue]

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
    from yavla.models.decoder import SimpleActionDecoder
    from yavla.models.encoders.proprio import ProprioEncoder, ProprioEncoderConfig
    from yavla.models.heads.mlp import MLPRegressionHead
    from yavla.models.merger import ConcatMerger
    from yavla.models.vlm_registry import vlm_registry

    # Build vision encoder + backbone via registry (type-driven dispatch)
    vision_encoder, backbone = vlm_registry.build(
        config.backbone, config.freeze, config.merger.num_readout_tokens
    )

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

    # Validate integration
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


def _tensor_to_list(d: dict[str, Any]) -> None:
    """Recursively convert Tensor values to lists for JSON serialization."""
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.tolist()
        elif isinstance(v, dict):
            _tensor_to_list(v)


def _filter_known_fields(cls: type, d: dict[str, Any]) -> dict[str, Any]:
    """Filter dict to only keys accepted by a dataclass constructor.

    Logs a warning for any dropped keys (forward-compatibility).
    """
    import dataclasses
    import logging

    known = {f.name for f in dataclasses.fields(cls)}
    filtered = {}
    for k, v in d.items():
        if k in known:
            filtered[k] = v
        else:
            logging.warning(
                "Config key %r not recognized by %s — skipping (possibly from a newer config version)",
                k,
                cls.__name__,
            )
    return filtered


def _dict_to_config(d: dict[str, Any]) -> PolicyConfig:
    """Reconstruct PolicyConfig from a dict (loaded from JSON).

    Unknown keys in sub-configs are silently dropped with a warning,
    enabling forward-compatibility when loading checkpoints from newer versions.
    """
    from yavla.models.config import BackboneConfig
    from yavla.models.encoders.proprio import ProprioEncoderConfig
    from yavla.models.encoders.vision import VisionEncoderConfig
    from yavla.models.heads.mlp import MLPHeadConfig
    from yavla.models.merger import TokenMergerConfig
    from yavla.models.types import ActionSpaceSpec, FreezeConfig, ProprioSpec

    return PolicyConfig(
        vision_encoder=VisionEncoderConfig(**_filter_known_fields(VisionEncoderConfig, d.get("vision_encoder", {}))),
        proprio_encoder=ProprioEncoderConfig(
            **_filter_known_fields(ProprioEncoderConfig, d.get("proprio_encoder", {}))
        ),
        merger=TokenMergerConfig(**_filter_known_fields(TokenMergerConfig, d.get("merger", {}))),
        backbone=BackboneConfig(**_filter_known_fields(BackboneConfig, d.get("backbone", {}))),
        action_head=MLPHeadConfig(**_filter_known_fields(MLPHeadConfig, d.get("action_head", {}))),
        freeze=FreezeConfig(**_filter_known_fields(FreezeConfig, d.get("freeze", {}))),
        action_space=ActionSpaceSpec(
            **_filter_known_fields(ActionSpaceSpec, d.get("action_space", {"names": [], "units": [], "limits": None}))
        ),
        proprio=ProprioSpec(**_filter_known_fields(ProprioSpec, d.get("proprio", {"names": [], "units": []}))),
        dt_hz=d.get("dt_hz", 10.0),
        config_version=d.get("config_version", "1.0"),
    )

