#!/usr/bin/env python3
"""Streaming inference against a vLLM T5Gemma 2 DFlash server."""

from __future__ import annotations

import argparse
import statistics
import time

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="Translate to German: The house is wonderful.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="t5gemma-2-1b-1b-dflash")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key="unused")
    started = time.perf_counter()
    event_times: list[float] = []
    usage = None

    stream = client.completions.create(
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].text:
            now = time.perf_counter()
            event_times.append(now)
            print(chunk.choices[0].text, end="", flush=True)

    finished = time.perf_counter()
    print()
    ttft = event_times[0] - started if event_times else float("nan")
    gaps = [b - a for a, b in zip(event_times, event_times[1:])]
    output_tokens = usage.completion_tokens if usage else len(event_times)
    tpot = (finished - event_times[0]) / max(output_tokens - 1, 1) if event_times else float("nan")
    print(
        f"latency={finished - started:.3f}s ttft={ttft * 1000:.2f}ms "
        f"mean_itl={statistics.mean(gaps) * 1000:.2f}ms "
        f"tpot={tpot * 1000:.2f}ms output_tokens={output_tokens}"
        if gaps
        else f"latency={finished - started:.3f}s ttft={ttft * 1000:.2f}ms output_tokens={output_tokens}"
    )


if __name__ == "__main__":
    main()
