#!/usr/bin/env python3
"""Benchmark GigaChat3 baseline, DSpark Base-Dev, and DSpark Large-Dev.

This is a configured front end for ``benchmark_t5gemma_suite.py``. It keeps the
same prompt snapshots, latency/throughput summaries, acceptance metrics, resume
logic, and Markdown/CSV/JSON output format while selecting GigaChat3-specific
models and prompt rendering.
"""

from __future__ import annotations

import sys

import benchmark_t5gemma_suite as suite
from gigachat3_vllm_plugin.config_compat import patch_gigachat3_deepseek_config


DEFAULTS = [
    "--base-model",
    "ai-sage/GigaChat3-10B-A1.8B",
    "--named-dspark-model",
    (
        "base_dev=/home/pc/dflash-output/"
        "dspark_gigachat3_twix_200k_8k_3layer_lk/checkpoints/0"
    ),
    "--named-dspark-model",
    (
        "large_dev=/home/pc/dflash-output/"
        "dspark_gigachat3_twix_200k_8k_5layer_lk_sdpa/checkpoints/0"
    ),
    "--vllm-plugin",
    "gigachat3_vllm_plugin",
    "--served-model-prefix",
    "gigachat3",
    "--render-user-prompts",
    "--chunked-prefill",
    "--linear-backend",
    "triton",
    "--cudagraph-mode",
    "PIECEWISE",
    "--no-disable-compile-cache",
    "--output-dir",
    "benchmark-results/gigachat3_dspark_suite",
]


def main() -> None:
    # Prompt preparation loads AutoTokenizer/AutoConfig in the benchmark process,
    # before vLLM has a chance to discover plugin entry points.
    patch_gigachat3_deepseek_config()
    # argparse uses the final occurrence for scalar options, so callers can
    # override defaults such as --output-dir, --base-model, or runtime limits.
    sys.argv[1:1] = DEFAULTS
    suite.main()


if __name__ == "__main__":
    main()
