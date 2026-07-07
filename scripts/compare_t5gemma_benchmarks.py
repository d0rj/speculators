#!/usr/bin/env python3
"""Render a markdown comparison table for multiple T5Gemma benchmark runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        help=(
            "Benchmark directories, summary.json files, or label=path entries. "
            "Each summary may contain multiple runs, e.g. baseline and dflash."
        ),
    )
    parser.add_argument("--output", default=None, help="Optional markdown output path.")
    parser.add_argument(
        "--sort-by",
        default="output_tps",
        choices=["name", "output_tps", "latency_p50", "ttft_p50", "acceptance_rate"],
    )
    parser.add_argument("--descending", action="store_true")
    return parser.parse_args()


def resolve_summary_path(value: str) -> tuple[str | None, Path]:
    label = None
    path_text = value
    if "=" in value:
        label, path_text = value.split("=", 1)
    path = Path(path_text)
    if path.is_dir():
        path = path / "summary.json"
    return label, path


def get_nested(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    number = as_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}{suffix}"


def fmt_pct(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return "-"
    return f"{number * 100:.2f}%"


def load_rows(entries: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        label, path = resolve_summary_path(entry)
        summary = json.loads(path.read_text(encoding="utf-8"))
        group = label or path.parent.name
        for run_name, run in summary.items():
            acc = run.get("speculative_acceptance") or {}
            totals = run.get("totals") or {}
            rows.append(
                {
                    "name": f"{group}/{run_name}",
                    "requests": totals.get("ok_requests"),
                    "output_tokens": totals.get("output_tokens"),
                    "output_tps": totals.get("output_throughput_tps"),
                    "request_rps": totals.get("request_throughput_rps"),
                    "latency_mean": get_nested(run, "latency_s", "mean"),
                    "latency_p50": get_nested(run, "latency_s", "p50"),
                    "latency_p95": get_nested(run, "latency_s", "p95"),
                    "ttft_mean": get_nested(run, "ttft_ms", "mean"),
                    "ttft_p50": get_nested(run, "ttft_ms", "p50"),
                    "ttft_p95": get_nested(run, "ttft_ms", "p95"),
                    "itl_mean": get_nested(run, "mean_itl_ms", "mean"),
                    "itl_p50": get_nested(run, "mean_itl_ms", "p50"),
                    "itl_p95": get_nested(run, "mean_itl_ms", "p95"),
                    "tpot_mean": get_nested(run, "tpot_ms", "mean"),
                    "tpot_p50": get_nested(run, "tpot_ms", "p50"),
                    "tpot_p95": get_nested(run, "tpot_ms", "p95"),
                    "acceptance_rate": acc.get("acceptance_rate"),
                    "mean_acceptance_length": acc.get(
                        "mean_acceptance_length_including_bonus"
                    ),
                    "draft_tokens": acc.get("num_draft_tokens"),
                    "accepted_tokens": acc.get("num_accepted_tokens"),
                    "path": str(path),
                }
            )
    return rows


def sort_rows(
    rows: list[dict[str, Any]],
    sort_by: str,
    descending: bool,
) -> list[dict[str, Any]]:
    if sort_by == "name":
        return sorted(rows, key=lambda row: row["name"], reverse=descending)
    return sorted(
        rows,
        key=lambda row: (
            as_float(row.get(sort_by)) is None,
            as_float(row.get(sort_by)) or 0.0,
        ),
        reverse=descending,
    )


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "run",
        "req",
        "out tok",
        "out tok/s",
        "req/s",
        "lat mean",
        "lat p50",
        "lat p95",
        "TTFT p50",
        "TTFT p95",
        "ITL p50",
        "ITL p95",
        "TPOT p50",
        "acc rate",
        "acc len",
        "accepted/draft",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        accepted = as_float(row.get("accepted_tokens"))
        draft = as_float(row.get("draft_tokens"))
        accepted_draft = (
            f"{accepted:.0f}/{draft:.0f}" if accepted is not None and draft else "-"
        )
        values = [
            row["name"],
            fmt(row.get("requests"), 0),
            fmt(row.get("output_tokens"), 0),
            fmt(row.get("output_tps"), 2),
            fmt(row.get("request_rps"), 3),
            fmt(row.get("latency_mean"), 2, "s"),
            fmt(row.get("latency_p50"), 2, "s"),
            fmt(row.get("latency_p95"), 2, "s"),
            fmt(row.get("ttft_p50"), 1, "ms"),
            fmt(row.get("ttft_p95"), 1, "ms"),
            fmt(row.get("itl_p50"), 1, "ms"),
            fmt(row.get("itl_p95"), 1, "ms"),
            fmt(row.get("tpot_p50"), 1, "ms"),
            fmt_pct(row.get("acceptance_rate")),
            fmt(row.get("mean_acceptance_length"), 2),
            accepted_draft,
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = sort_rows(load_rows(args.runs), args.sort_by, args.descending)
    markdown = markdown_table(rows)
    print(markdown, end="")
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
