#!/usr/bin/env python3
"""Expand a trained DSpark draft with identity-initialized decoder layers."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file

from speculators.models.dspark import DSparkDraftModel, DSparkSpeculatorConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Build and validate the expanded model without writing it.",
    )
    return parser.parse_args()


def build_expanded_model(
    source: Path, target_num_layers: int
) -> tuple[DSparkDraftModel, int, list[str]]:
    weights_path = source / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing source weights: {weights_path}")

    config = DSparkSpeculatorConfig.from_pretrained(source)
    layer_config = config.transformer_layer_config
    source_num_layers = int(layer_config.num_hidden_layers)
    if target_num_layers <= source_num_layers:
        raise ValueError(
            f"--num-layers must exceed the source layer count "
            f"({source_num_layers}), got {target_num_layers}."
        )

    old_layer_types = list(layer_config.layer_types)
    fill_type = old_layer_types[-1] if old_layer_types else "full_attention"
    layer_config.num_hidden_layers = target_num_layers
    layer_config.layer_types = old_layer_types + [fill_type] * (
        target_num_layers - len(old_layer_types)
    )

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = DSparkDraftModel(config)
    finally:
        torch.set_default_dtype(old_dtype)

    source_state = load_file(weights_path, device="cpu")
    incompatible = model.load_state_dict(source_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected source keys: {incompatible.unexpected_keys}")

    allowed_missing_prefixes = tuple(
        f"layers.{idx}." for idx in range(source_num_layers, target_num_layers)
    )
    allowed_missing_exact = {"verifier_lm_head.weight", "verifier_norm.weight"}
    bad_missing = [
        key
        for key in incompatible.missing_keys
        if key not in allowed_missing_exact
        and not key.startswith(allowed_missing_prefixes)
    ]
    if bad_missing:
        raise RuntimeError(f"Unexpected missing source keys: {bad_missing}")

    # A decoder layer is exactly residual/identity when both branch output
    # projections are zero. Other weights remain randomly initialized and can
    # start learning immediately without destroying the three-layer function.
    with torch.no_grad():
        for idx in range(source_num_layers, target_num_layers):
            layer = model.layers[idx]
            layer.self_attn.o_proj.weight.zero_()
            layer.mlp.down_proj.weight.zero_()

    for idx in range(source_num_layers, target_num_layers):
        layer = model.layers[idx]
        if torch.count_nonzero(layer.self_attn.o_proj.weight).item() != 0:
            raise RuntimeError(f"layers.{idx}.self_attn.o_proj is not zero")
        if torch.count_nonzero(layer.mlp.down_proj.weight).item() != 0:
            raise RuntimeError(f"layers.{idx}.mlp.down_proj is not zero")

    return model, source_num_layers, incompatible.missing_keys


def save_expanded_model(
    model: DSparkDraftModel,
    source: Path,
    output: Path,
    source_num_layers: int,
) -> None:
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new --output directory."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.tmp-", dir=output.parent
    ) as tmp_name:
        tmp = Path(tmp_name)
        model.save_pretrained(tmp, safe_serialization=True)
        source_config_py = source / "config.py"
        if source_config_py.is_file():
            shutil.copy2(source_config_py, tmp / "config.py")
        provenance = {
            "source": str(source.resolve()),
            "source_num_layers": source_num_layers,
            "target_num_layers": len(model.layers),
            "new_layer_initialization": "identity_zero_o_proj_and_down_proj",
        }
        (tmp / "expansion_info.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        shutil.move(str(tmp), str(output))


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    model, source_num_layers, missing_keys = build_expanded_model(
        source, args.num_layers
    )
    print(
        f"Validated DSpark expansion: {source_num_layers} -> {len(model.layers)} "
        f"layers ({len(missing_keys)} expected missing tensors)."
    )
    if args.validate_only:
        return
    if args.output is None:
        raise ValueError("--output is required unless --validate-only is used")
    save_expanded_model(model, source, args.output, source_num_layers)
    print(f"Saved expanded checkpoint to {args.output}")


if __name__ == "__main__":
    main()
