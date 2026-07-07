from types import SimpleNamespace

from speculators.models.utils import get_verifier_text_config


def test_get_verifier_text_config_prefers_encoder_decoder_decoder():
    decoder = SimpleNamespace(model_type="t5gemma2_decoder")
    config = SimpleNamespace(
        is_encoder_decoder=True,
        decoder=decoder,
        text_config=SimpleNamespace(model_type="wrong"),
    )
    assert get_verifier_text_config(config) is decoder


def test_get_verifier_text_config_uses_multimodal_text_config():
    text_config = SimpleNamespace(model_type="qwen3")
    config = SimpleNamespace(is_encoder_decoder=False, text_config=text_config)
    assert get_verifier_text_config(config) is text_config


def test_get_verifier_text_config_returns_plain_config():
    config = SimpleNamespace(is_encoder_decoder=False)
    assert get_verifier_text_config(config) is config
