#!/usr/bin/env python3
"""Prepare encoder/decoder text pairs for T5Gemma2 dFlash training."""

import argparse
import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoConfig, AutoTokenizer

from speculators.data_generation.preprocessing import (
    _normalize_conversation,
    load_raw_dataset,
)
from speculators.models.utils import get_verifier_text_config
from speculators.train.vocab_mapping import save_token_frequency_distribution

logger = logging.getLogger(__name__)
MIN_TARGET_TOKENS = 2


@dataclass(frozen=True)
class MixtureSource:
    name: str
    weight: float
    converter: str = "auto"


MIXTURE_PRESETS: dict[str, list[MixtureSource]] = {
    # 200k default:
    #   60k UltraChat, 20k ShareGPT, 50k code, 40k xLAM function calling,
    #   20k Glaive function calling, 10k math/reasoning.
    "t5gemma2_200k": [
        MixtureSource("ultrachat", 0.30, "auto"),
        MixtureSource("sharegpt", 0.10, "auto"),
        MixtureSource("hf:ise-uiuc/Magicoder-Evol-Instruct-110K::train", 0.25, "instruction_response"),
        MixtureSource("hf:Salesforce/xlam-function-calling-60k::train", 0.20, "xlam_function_calling"),
        MixtureSource("hf:glaiveai/glaive-function-calling-v2::train", 0.10, "glaive_function_calling"),
        MixtureSource("hf:TIGER-Lab/MathInstruct::train", 0.05, "instruction_output"),
    ],
}


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


def _instruction_response_to_pair(example: dict, *_):
    source = example.get("instruction") or example.get("prompt")
    target = example.get("response") or example.get("output") or example.get("answer")
    if not source or not target:
        return None
    return str(source), str(target)


def _instruction_output_to_pair(example: dict, *_):
    source = example.get("instruction") or example.get("problem") or example.get("question")
    target = example.get("output") or example.get("response") or example.get("answer")
    if not source or not target:
        return None
    return str(source), str(target)


def _xlam_function_calling_to_pair(example: dict, *_):
    query = example.get("query")
    tools = example.get("tools")
    answers = example.get("answers") or example.get("answer")
    if not query or not tools or not answers:
        return None
    source = (
        "You are given a user query and a list of available tools. "
        "Return the correct tool call answer as JSON.\n\n"
        f"Tools:\n{tools}\n\n"
        f"User query:\n{query}"
    )
    return source, str(answers)


def _split_last_marker(text: str, markers: tuple[str, ...]):
    best_idx = -1
    best_marker = ""
    for marker in markers:
        idx = text.rfind(marker)
        if idx > best_idx:
            best_idx = idx
            best_marker = marker
    if best_idx < 0:
        return None
    source = text[:best_idx].strip()
    target = text[best_idx + len(best_marker) :].strip()
    if not source or not target:
        return None
    return source, target


def _glaive_function_calling_to_pair(example: dict, *_):
    system = str(example.get("system") or "").strip()
    chat = str(example.get("chat") or "").strip()
    if not chat:
        return None
    split = _split_last_marker(
        chat,
        (
            "ASSISTANT:",
            "Assistant:",
            "assistant:",
            "<|assistant|>",
            "<|assistant|>\n",
        ),
    )
    if split is None:
        return None
    source, target = split
    if system:
        source = f"System:\n{system}\n\n{source}"
    return source, target


PAIR_CONVERTERS: dict[str, Callable] = {
    "auto": _conversation_to_pair,
    "instruction_response": _instruction_response_to_pair,
    "instruction_output": _instruction_output_to_pair,
    "xlam_function_calling": _xlam_function_calling_to_pair,
    "glaive_function_calling": _glaive_function_calling_to_pair,
}


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
    parser.add_argument("--data", action="append")
    parser.add_argument(
        "--mixture",
        choices=sorted(MIXTURE_PRESETS),
        default=None,
        help=(
            "Deterministic built-in dataset mixture. When set, --data is ignored. "
            "Use --max-samples to scale the fixed proportions."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--encoder-seq-length", type=int, default=2048)
    parser.add_argument("--decoder-seq-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _parse_hf_spec(spec: str) -> tuple[str, str | None, str]:
    if not spec.startswith("hf:"):
        raise ValueError(f"Expected hf: dataset spec, got {spec!r}")
    parts = spec.removeprefix("hf:").split(":")
    if len(parts) == 1:
        return parts[0], None, "train"
    if len(parts) == 2:
        return parts[0], None, parts[1] or "train"
    if len(parts) == 3:
        return parts[0], parts[1] or None, parts[2] or "train"
    raise ValueError(f"Invalid hf: spec {spec!r}. Expected hf:<id>[:<subset>:<split>].")


def _load_source(name: str):
    if name.startswith("hf:"):
        dataset_id, subset, split = _parse_hf_spec(name)
        return load_dataset(dataset_id, name=subset, split=split), None
    return load_raw_dataset(name)


def _scaled_counts(sources: list[MixtureSource], max_samples: int | None) -> list[int | None]:
    if max_samples is None:
        return [None for _ in sources]
    raw_counts = [int(max_samples * source.weight) for source in sources]
    remainder = max_samples - sum(raw_counts)
    if remainder > 0:
        # Put the remainder into the largest bucket to keep exact total while
        # preserving the intended proportions as closely as possible.
        largest = max(range(len(sources)), key=lambda idx: sources[idx].weight)
        raw_counts[largest] += remainder
    return raw_counts


def _resolve_sources(args) -> list[tuple[str, int | None, str]]:
    if args.mixture:
        sources = MIXTURE_PRESETS[args.mixture]
        counts = _scaled_counts(sources, args.max_samples)
        return [
            (source.name, count, source.converter)
            for source, count in zip(sources, counts, strict=True)
        ]
    if not args.data:
        raise ValueError("Either --mixture or at least one --data must be provided")
    return [(data_path, None, "auto") for data_path in args.data]


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
    source_stats = []
    global_limit = None if args.mixture else args.max_samples
    for source_idx, (data_path, source_limit, converter_name) in enumerate(
        _resolve_sources(args)
    ):
        raw, normalize_fn = _load_source(data_path)
        raw = raw.shuffle(seed=args.seed + source_idx)
        converter = PAIR_CONVERTERS[converter_name]
        before = len(rows)
        for example in raw:
            pair = converter(example, normalize_fn)
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
            if source_limit is not None and len(rows) - before >= source_limit:
                break
            if global_limit is not None and len(rows) >= global_limit:
                break
        kept = len(rows) - before
        source_stats.append(
            {
                "source": data_path,
                "converter": converter_name,
                "requested": source_limit,
                "kept": kept,
            }
        )
        logger.info(
            "Source %s kept %d/%s samples",
            data_path,
            kept,
            "all" if source_limit is None else source_limit,
        )
        if global_limit is not None and len(rows) >= global_limit:
            break

    if not rows:
        raise ValueError("No usable source/target pairs were found")
    dataset = Dataset.from_list(rows)
    dataset = dataset.shuffle(seed=args.seed)
    dataset.set_format(type="torch")
    dataset.save_to_disk(str(output))
    save_token_frequency_distribution(dataset, output / "token_freq.pt")
    (output / "t5gemma_data_config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "mixture": args.mixture,
                "samples": len(rows),
                "seed": args.seed,
                "sources": source_stats,
                "decoder_start_token_id": start_id,
                "eos_token_id": eos_id,
            },
            indent=2,
        )
    )
    logger.info("Saved %d samples to %s", len(rows), output)


if __name__ == "__main__":
    main()
