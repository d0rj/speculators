#!/usr/bin/env python3
"""Launch the trained T5Gemma 2 DFlash checkpoint with vLLM."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


DEFAULT_DRAFT = "dflash-output/t5gemma-5k/checkpoints/checkpoint_best"


def resolve_checkpoint(value: str) -> Path:
    """Resolve both dflash-output and the commonly mistyped dflash_output."""
    candidates = [Path(value)]
    if "dflash_output" in value:
        candidates.append(Path(value.replace("dflash_output", "dflash-output")))
    script_root = Path(__file__).resolve().parents[1]
    candidates.extend(script_root / candidate for candidate in list(candidates))
    candidates.extend(
        candidate / "checkpoints" / "checkpoint_best"
        for candidate in list(candidates)
        if candidate.name.startswith("t5gemma-")
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"DFlash checkpoint not found: {value}")


def validate_checkpoint(path: Path) -> None:
    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures", [])
    if "DFlashDraftModel" not in architectures:
        raise ValueError(f"{path} is not a DFlash checkpoint")
    verifier = config.get("speculators_config", {}).get("verifier", {})
    if "T5Gemma2ForConditionalGeneration" not in verifier.get("architectures", []):
        raise ValueError(f"{path} was not trained for T5Gemma 2")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Serve a T5Gemma 2 DFlash checkpoint with vLLM",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_DRAFT)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--served-model-name", default="t5gemma-2-1b-1b-dflash")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()
    os.environ.setdefault("VLLM_PLUGINS", "t5gemma2_vllm_plugin")
    checkpoint = resolve_checkpoint(args.checkpoint)
    validate_checkpoint(checkpoint)

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(checkpoint),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--served-model-name",
        args.served_model_name,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--no-enable-chunked-prefill",
        *passthrough,
    ]
    print(" ".join(cmd), flush=True)
    if not args.dry_run:
        os.execvp(cmd[0], cmd)  # noqa: S606


if __name__ == "__main__":
    main()
