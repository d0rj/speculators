from typing import ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.dflash.utils import get_base_indices_for_anchored_blocks
from speculators.models.dflare.config import DFlareSpeculatorConfig
from speculators.models.dflare.model_definitions import Qwen3DFlareDecoderLayer
from speculators.models.metrics import LossConfig, resolve_loss_config
from speculators.models.utils import conditional_torch_compile

__all__ = [
    "DFlareDraftModel",
]


@SpeculatorModel.register("dflare")
class DFlareDraftModel(DFlashDraftModel):
    """DFlash-family drafter with DFlare layer-wise target-state fusion."""

    config_class: ClassVar[type[DFlareSpeculatorConfig]] = DFlareSpeculatorConfig  # type: ignore[misc,assignment]
    _no_split_modules = ["Qwen3DFlareDecoderLayer"]

    def __init__(self, config: DFlareSpeculatorConfig) -> None:
        super().__init__(config=config)

        tl_config = config.transformer_layer_config
        num_draft_layers = tl_config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                Qwen3DFlareDecoderLayer(tl_config, layer_idx)  # type: ignore[arg-type]
                for layer_idx in range(num_draft_layers)
            ]
        )
        if hasattr(self, "fc"):
            del self.fc

        self.layer_fusion_weights = nn.Parameter(
            torch.empty(num_draft_layers, len(self.target_layer_ids))
        )
        self.hidden_norm = Qwen3RMSNorm(
            tl_config.hidden_size,
            eps=tl_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.post_init()
        self._init_fusion_weights()

    def _init_fusion_weights(self) -> None:
        nn.init.constant_(self.layer_fusion_weights, 0.0)
        num_target_layers = self.layer_fusion_weights.shape[1]
        num_draft_layers = self.layer_fusion_weights.shape[0]
        for draft_idx in range(num_draft_layers):
            target_idx = min(
                num_target_layers - 1,
                int((draft_idx / max(num_draft_layers, 1)) * num_target_layers),
            )
            self.layer_fusion_weights.data[draft_idx, target_idx] = 2.0

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DFlareDraftModel":
        config = DFlareSpeculatorConfig(
            **cls._build_base_config_kwargs("dflare", verifier_config, **kwargs)
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        loss_config = resolve_loss_config(kwargs["loss_fn"])
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 3072)
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "max_anchors": max_anchors,
        }
        return dict(shared), dict(shared)

    def _fuse_target_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, width = hidden_states.shape
        hidden_size = self.config.transformer_layer_config.hidden_size
        num_target_layers = len(self.target_layer_ids)
        expected_width = num_target_layers * hidden_size
        if width != expected_width:
            raise ValueError(
                "DFlare hidden_states width must equal "
                f"len(target_layer_ids) * hidden_size ({expected_width}), got {width}."
            )
        target_hidden = hidden_states.view(
            bsz,
            seq_len,
            num_target_layers,
            hidden_size,
        )
        fusion_probs = torch.softmax(self.layer_fusion_weights, dim=1).to(
            target_hidden.dtype
        )
        fused_hidden = torch.einsum("bsth,dt->bsdh", target_hidden, fusion_probs)
        return self.hidden_norm(fused_hidden)

    def _backbone_forward(
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_hidden*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        **kwargs,
    ):
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]
        num_anchors = kwargs.pop("max_anchors", 3072)

        if position_ids is None:
            position_ids = torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid = (
            self._build_attention_mask(loss_mask, num_anchors, document_ids, device)
        )

        mask_tokens_size = num_anchors * self.block_size
        mask_token_ids = torch.full(
            (1, mask_tokens_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        noise_embedding = self.embed_tokens(mask_token_ids)

        fused_target_hidden = self._fuse_target_hidden_states(hidden_states)

        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[0, anchor_positions], self.block_size
        )
        position_ids = torch.cat([position_ids, mask_position_ids.unsqueeze(0)], dim=1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )

        with torch.no_grad():
            verifier_logits = self.verifier_lm_head(
                self.verifier_norm(verifier_last_hidden_states)
            )
            softcap = getattr(
                self.config.transformer_layer_config,
                "verifier_final_logit_softcapping",
                None,
            )
            if softcap is not None:
                verifier_logits = torch.tanh(verifier_logits / softcap) * softcap
            verifier_logits = torch.roll(verifier_logits, 1, dims=1)
            targets = verifier_logits[:, anchored_block_indices]

        for layer_idx, layer in enumerate(self.layers):
            layer_target_hidden = fused_target_hidden[:, :, layer_idx, :]
            noise_embedding = layer(
                hidden_states=noise_embedding,
                target_hidden=layer_target_hidden,
                attention_mask=sliding_window_attn_mask
                if layer_idx in self.sliding_window_indices
                else full_attn_mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden = self.norm(noise_embedding)
        logits = self.lm_head(hidden)

        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )
        aligned_loss_mask[:, :: self.block_size] = 0

        return hidden, logits, targets, aligned_loss_mask, anchored_block_indices

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 3072,
        **kwargs,
    ):
        _, logits, targets, aligned_loss_mask, _ = self._backbone_forward(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            position_ids,
            max_anchors=max_anchors,
            **kwargs,
        )
        loss, metrics = compute_metrics(
            logits,
            targets,
            aligned_loss_mask,
            self.block_size,
            gamma=gamma,
            loss_config=loss_config,
        )
        draft_tokens = torch.argmax(logits, dim=-1)
        return draft_tokens, loss, metrics
