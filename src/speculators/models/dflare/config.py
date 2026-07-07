from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DFlareSpeculatorConfig",
]


@SpeculatorModelConfig.register("dflare")
class DFlareSpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash-family config for DFlare.

    DFlare keeps DFlash's training/data contract, but replaces the single
    shared ``fc(T * H -> H)`` target-state projection with learnable per-draft
    layer fusion weights and uses separate context/noise K/V projections.
    """

    speculators_model_type: Literal["dflare"] = "dflare"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DFlareDraftModel"],
        description="Model architectures that can load these weights",
    )
