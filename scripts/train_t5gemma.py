"""Train a Qwen-compatible dFlash drafter for a T5Gemma2 decoder."""

import sys
from copy import deepcopy
from pathlib import Path

import transformers
from packaging import version
from transformers.models.auto.configuration_auto import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as generic_train

from speculators.models.utils import get_verifier_text_config


def create_t5gemma_transformer_layer_config(
    verifier_name_or_path: str,
    num_layers: int,
    draft_arch: str,
    hidden_act: str | None,
    sliding_window: int,
    sliding_window_indices: list[int],
):
    if draft_arch not in generic_train.DRAFT_ARCH_CONFIGS:
        raise ValueError(
            f"Unknown draft architecture: {draft_arch}. "
            f"Available: {list(generic_train.DRAFT_ARCH_CONFIGS)}"
        )

    outer_config = AutoConfig.from_pretrained(verifier_name_or_path)
    if not getattr(outer_config, "is_encoder_decoder", False):
        raise ValueError(
            f"{verifier_name_or_path} is not an encoder-decoder model. "
            "Use scripts/train.py for decoder-only verifiers."
        )
    decoder_config = get_verifier_text_config(outer_config)

    if sliding_window_indices and (
        min(sliding_window_indices) < 0 or max(sliding_window_indices) >= num_layers
    ):
        raise ValueError(
            "Sliding window indices must be valid draft layer ids "
            "in range [0, num_layers)."
        )

    layer_types = [
        "sliding_attention" if i in sliding_window_indices else "full_attention"
        for i in range(num_layers)
    ]
    config_class = generic_train.DRAFT_ARCH_CONFIGS[draft_arch]
    resolved_hidden_act = (
        hidden_act
        or getattr(decoder_config, "hidden_act", None)
        or getattr(decoder_config, "hidden_activation", None)
        or "silu"
    )
    config = config_class(
        vocab_size=decoder_config.vocab_size,
        hidden_size=decoder_config.hidden_size,
        intermediate_size=decoder_config.intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=decoder_config.num_attention_heads,
        num_key_value_heads=decoder_config.num_key_value_heads,
        hidden_act=resolved_hidden_act,
        max_position_embeddings=getattr(
            decoder_config, "max_position_embeddings", 131072
        ),
        initializer_range=getattr(decoder_config, "initializer_range", 0.02),
        rms_norm_eps=getattr(decoder_config, "rms_norm_eps", 1e-6),
        head_dim=getattr(decoder_config, "head_dim", None),
        attention_bias=getattr(decoder_config, "attention_bias", False),
        attention_dropout=getattr(decoder_config, "attention_dropout", 0.0),
        tie_word_embeddings=False,
        sliding_window=sliding_window,
        layer_types=layer_types,
    )

    rope_parameters = deepcopy(getattr(decoder_config, "rope_parameters", None))
    if rope_parameters:
        if "full_attention" in rope_parameters:
            rope_parameters = rope_parameters["full_attention"]
        rope_parameters.pop("type", None)
        config.rope_parameters = rope_parameters
        config.rope_theta = rope_parameters.get("rope_theta", 1_000_000.0)
    elif version.parse(transformers.__version__) < version.parse("5.0.0"):
        config.rope_theta = getattr(decoder_config, "rope_theta", 1_000_000.0)

    config.verifier_model_type = getattr(outer_config, "model_type", None)
    config.verifier_is_encoder_decoder = True
    config.verifier_hidden_states_are_normalized = True
    config.verifier_final_logit_softcapping = getattr(
        decoder_config, "final_logit_softcapping", None
    )
    config.architectures = getattr(outer_config, "architectures", None)
    return config


def main() -> None:
    generic_train.create_transformer_layer_config = (
        create_t5gemma_transformer_layer_config
    )
    args = generic_train.parse_args()
    generic_train.args = args
    if args.speculator_type != "dflash":
        raise ValueError(
            "scripts/train_t5gemma.py only supports --speculator-type dflash"
        )
    generic_train.main(args)


if __name__ == "__main__":
    main()
