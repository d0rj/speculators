import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators.models.dflare import DFlareDraftModel, DFlareSpeculatorConfig


def _make_config() -> DFlareSpeculatorConfig:
    layer_config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        layer_types=["full_attention", "full_attention"],
    )
    layer_config._attn_implementation = "eager"  # noqa: SLF001
    return DFlareSpeculatorConfig(
        transformer_layer_config=layer_config,
        draft_vocab_size=32,
        block_size=2,
        aux_hidden_state_layer_ids=[1, 2, 3],
        mask_token_id=0,
    )


def test_dflare_forward_shapes_and_finite_loss():
    torch.manual_seed(0)
    model = DFlareDraftModel(_make_config())
    with torch.no_grad():
        model.embed_tokens.weight.normal_(0, 0.02)
        model.lm_head.weight.normal_(0, 0.02)
        model.verifier_lm_head.weight.copy_(model.lm_head.weight)

    seq_len = 8
    hidden_size = model.config.transformer_layer_config.hidden_size
    num_target_layers = len(model.target_layer_ids)
    hidden_states = torch.randn(1, seq_len, num_target_layers * hidden_size)
    verifier_last_hidden_states = torch.randn(1, seq_len, hidden_size)
    input_ids = torch.randint(1, 31, (1, seq_len))
    loss_mask = torch.ones(1, seq_len)
    document_ids = torch.zeros(1, seq_len, dtype=torch.long)

    draft_tokens, loss, metrics = model(
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        verifier_last_hidden_states=verifier_last_hidden_states,
        document_ids=document_ids,
        max_anchors=2,
    )

    assert draft_tokens.shape == (1, 4)
    assert torch.isfinite(loss)
    assert "loss_sum" in metrics
    assert model.layer_fusion_weights.shape == (2, 3)
