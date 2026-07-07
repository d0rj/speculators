#!/usr/bin/env python3
"""Extract T5Gemma2 decoder hidden states into the Speculators file format."""

import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import load_from_disk
from safetensors.torch import save_file
from transformers import AutoConfig

from speculators.models.utils import get_verifier_text_config
from speculators.train.t5gemma_online import T5GemmaHiddenStateExtractor

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/t5gemma-2-1b-1b")
    parser.add_argument("--preprocessed-data", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--target-layer-ids", type=int, nargs="+")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    config = AutoConfig.from_pretrained(args.model)
    decoder_config = get_verifier_text_config(config)
    target_layer_ids = args.target_layer_ids or [
        2,
        decoder_config.num_hidden_layers // 2,
        decoder_config.num_hidden_layers - 3,
    ]
    dtype = getattr(torch, args.dtype)
    dataset = load_from_disk(args.preprocessed_data).with_format("torch")
    output = Path(args.output or Path(args.preprocessed_data) / "hidden_states")
    output.mkdir(parents=True, exist_ok=True)
    limit = min(len(dataset), args.max_samples or len(dataset))
    extractor = T5GemmaHiddenStateExtractor(args.model, target_layer_ids, dtype=dtype)

    try:
        for idx in range(limit):
            target = output / f"hs_{idx}.safetensors"
            if target.exists() and not args.overwrite:
                continue
            item = dataset[idx]
            hidden_states = extractor.extract_stacked(
                item["encoder_input_ids"], item["input_ids"]
            )
            if torch.isnan(hidden_states).any():
                raise ValueError(f"NaN hidden states in sample {idx}")
            temporary = target.with_suffix(".tmp")
            save_file(
                {
                    "hidden_states": hidden_states,
                    "token_ids": item["input_ids"].long().contiguous(),
                },
                temporary,
            )
            temporary.replace(target)
            if (idx + 1) % 10 == 0 or idx + 1 == limit:
                logger.info("Extracted %d/%d", idx + 1, limit)
    finally:
        extractor.close()

    (output / "metadata.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "target_layer_ids": target_layer_ids,
                "hidden_size": decoder_config.hidden_size,
                "dtype": args.dtype,
                "samples": limit,
                "last_hidden_state_is_normalized": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
