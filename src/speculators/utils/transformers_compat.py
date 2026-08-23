"""Narrow compatibility fixes for malformed upstream model configs."""

from __future__ import annotations

from typing import Any

from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
    DeepseekV3Config,
)


def patch_gigachat3_deepseek_config() -> None:
    """Accept GigaChat3's integer routed scaling factor in Transformers 5.12.

    ``ai-sage/GigaChat3-10B-A1.8B`` publishes ``routed_scaling_factor: 1``.
    Transformers 5.12 validates DeepseekV3Config dataclass fields strictly and
    requires a float. Coercing this scalar to ``1.0`` preserves its value while
    allowing AutoConfig, AutoProcessor, and vLLM to read the repository.
    """

    original = DeepseekV3Config.__init__
    if getattr(original, "_gigachat3_integer_scaling_compat", False):
        return

    def patched_init(
        self: Any,
        *args: Any,
        routed_scaling_factor: float = 2.5,
        **kwargs: Any,
    ) -> None:
        original(
            self,
            *args,
            routed_scaling_factor=float(routed_scaling_factor),
            **kwargs,
        )

    patched_init._gigachat3_integer_scaling_compat = True  # type: ignore[attr-defined]  # noqa: SLF001
    DeepseekV3Config.__init__ = patched_init


__all__ = ["patch_gigachat3_deepseek_config"]
