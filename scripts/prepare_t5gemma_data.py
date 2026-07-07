#!/usr/bin/env python3
"""Prepare encoder/decoder text pairs for T5Gemma2 dFlash training."""

import argparse
import json
import logging
import shutil
from pathlib import Path

from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer

from speculators.data_generation.preprocessing import (
    _normalize_conversation,
    load_raw_dataset,
)
from speculators.models.utils import get_verifier_text_config
from speculators.train.vocab_mapping import save_token_frequency_distribution

logger = logging.getLogger(__name__)
MIN_TARGET_TOKENS = 2


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content or "")


def _conversation_to_pair(example: dict, normalize_fn=None):
    if normalize_fn is not None:
        example = normalize_fn(example)
    if "source" in example and "target" in example:
        return str(example["source"]), str(example["target"])

    conv = example.get("conversations", example.get("messages"))
    if not conv:
        return None
    normalized = _normalize_conversation(conv)
    assistant_idx = next(
        (
            idx
            for idx in range(len(normalized) - 1, -1, -1)
            if normalized[idx]["role"] == "assistant"
            and _text(normalized[idx]["content"]).strip()
        ),
        None,
    )
    if assistant_idx is None:
        return None
    return (
        normalized[:assistant_idx],
        _text(normalized[assistant_idx]["content"]).strip(),
    )


def _format_source(tokenizer, source) -> str:
    if isinstance(source, str):
        return source
    if tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                source,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (TypeError, ValueError):
            pass
    lines = []
    for turn in source:
        role = turn["role"].capitalize()
        lines.append(f"{role}: {_text(turn['content'])}")
    lines.append("Assistant:")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/t5gemma-2-1b-1b")
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--encoder-seq-length", type=int, default=2048)
    parser.add_argument("--decoder-seq-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:  # noqa: C901
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    output = Path(args.output)
    if output.exists() and any(output.glob("*.arrow")):
        if not args.overwrite:
            raise FileExistsError(
                f"{output} already contains a dataset; pass --overwrite"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    outer_config = AutoConfig.from_pretrained(args.model)
    decoder_config = get_verifier_text_config(outer_config)
    start_id = (
        getattr(outer_config, "decoder_start_token_id", None)
        or getattr(decoder_config, "bos_token_id", None)
        or tokenizer.bos_token_id
        or tokenizer.pad_token_id
    )
    if start_id is None:
        raise ValueError("Could not resolve decoder start token id")
    eos_id = getattr(decoder_config, "eos_token_id", None) or tokenizer.eos_token_id
    if isinstance(eos_id, list):
        eos_id = eos_id[0]

    rows = []
    for data_path in args.data:
        raw, normalize_fn = load_raw_dataset(data_path)
        raw = raw.shuffle(seed=args.seed)
        for example in raw:
            pair = _conversation_to_pair(example, normalize_fn)
            if pair is None:
                continue
            source, target = pair
            source_text = _format_source(tokenizer, source)
            encoder_ids = tokenizer(
                source_text,
                add_special_tokens=True,
                truncation=True,
                max_length=args.encoder_seq_length,
            )["input_ids"]
            target_ids = tokenizer(
                target,
                add_special_tokens=False,
                truncation=True,
                max_length=args.decoder_seq_length - 1,
            )["input_ids"]
            if eos_id is not None:
                target_ids = target_ids[: args.decoder_seq_length - 1] + [eos_id]
            labels = target_ids[: args.decoder_seq_length]
            if len(labels) < MIN_TARGET_TOKENS:
                continue
            decoder_ids = [start_id, *labels[:-1]]
            loss_mask = [0, *([1] * (len(decoder_ids) - 1))]
            rows.append(
                {
                    "encoder_input_ids": encoder_ids,
                    "input_ids": decoder_ids,
                    "loss_mask": loss_mask,
                    "seq_len": len(decoder_ids),
                }
            )
            if args.max_samples is not None and len(rows) >= args.max_samples:
                break
        if args.max_samples is not None and len(rows) >= args.max_samples:
            break

    if not rows:
        raise ValueError("No usable source/target pairs were found")
    dataset = Dataset.from_list(rows)
    dataset.set_format(type="torch")
    dataset.save_to_disk(str(output))
    save_token_frequency_distribution(dataset, output / "token_freq.pt")
    (output / "t5gemma_data_config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "samples": len(rows),
                "decoder_start_token_id": start_id,
                "eos_token_id": eos_id,
            },
            indent=2,
        )
    )
    logger.info("Saved %d samples to %s", len(rows), output)


if __name__ == "__main__":
    main()
