#!/usr/bin/env python3
"""Print and save a baseline versus DFlash benchmark summary."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


KEYS = ["subset", "strategy", "target_rate"]
METRICS = [
    "rps_median",
    "latency_median_s",
    "ttft_median_ms",
    "itl_median_ms",
    "output_tps_median",
]


def load(path: Path) -> dict[tuple[str, ...], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {tuple(row[key] for key in KEYS): row for row in csv.DictReader(handle)}


def number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("dflash", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load(args.baseline)
    dflash = load(args.dflash)
    rows: list[dict[str, str | float]] = []
    for key in sorted(baseline.keys() & dflash.keys()):
        row: dict[str, str | float] = dict(zip(KEYS, key))
        for metric in METRICS:
            base_value = number(baseline[key].get(metric, ""))
            spec_value = number(dflash[key].get(metric, ""))
            row[f"baseline_{metric}"] = "" if base_value is None else base_value
            row[f"dflash_{metric}"] = "" if spec_value is None else spec_value
            if base_value is not None and spec_value not in (None, 0):
                if metric in {"latency_median_s", "ttft_median_ms", "itl_median_ms"}:
                    row[f"{metric}_speedup"] = base_value / spec_value
                else:
                    row[f"{metric}_speedup"] = spec_value / base_value if base_value else ""
        rows.append(row)

    if not rows:
        raise SystemExit("No matching benchmark rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'subset/rate':<28} {'TTFT b/s':>18} {'ITL b/s':>18} {'TPS b/s':>18}")
    for row in rows:
        label = f"{row['subset']}@{row['target_rate']}"
        print(
            f"{label:<28} "
            f"{row.get('baseline_ttft_median_ms', ''):>8}/{row.get('dflash_ttft_median_ms', ''):<8} "
            f"{row.get('baseline_itl_median_ms', ''):>8}/{row.get('dflash_itl_median_ms', ''):<8} "
            f"{row.get('baseline_output_tps_median', ''):>8}/{row.get('dflash_output_tps_median', ''):<8}"
        )
    print(f"Summary written to {args.output}")


if __name__ == "__main__":
    main()
