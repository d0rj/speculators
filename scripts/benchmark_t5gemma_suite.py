#!/usr/bin/env python3
"""Benchmark T5Gemma2 baseline and DFlash-family checkpoints on five suites.

The script starts one vLLM server per model configuration and runs the same
saved prompts through every server.  By default it compares the raw verifier
with DFlash, DFlare, and DSpark using 3 and 7 speculative tokens.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from benchmark_t5gemma_vllm import (
    acceptance_delta,
    fetch_metrics,
    normalize_path,
    one_completion,
    run_load,
    stop_server,
    summarize_run,
    wait_for_health,
    write_samples_csv,
    write_summary_csv,
)
from datasets import load_dataset

if TYPE_CHECKING:
    from collections.abc import Callable


DEFAULT_BASE_MODEL = "google/t5gemma-2-1b-1b"
DEFAULT_OUTPUT_DIR = "benchmark-results/t5gemma2_speculators_suite"
MAX_SPECULATIVE_TOKENS = 7


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str
    config: str | None
    split: str
    format_row: Callable[[dict[str, Any]], list[str]]


@dataclass(frozen=True)
class RunSpec:
    name: str
    checkpoint: str | None
    num_speculative_tokens: int | None


def _math_prompt(row: dict[str, Any]) -> list[str]:
    problem = row.get("problem")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("MATH-500 row has no non-empty 'problem'")
    return [f"Solve the problem step by step.\n\nProblem: {problem}\nAnswer:"]


def _gsm8k_prompt(row: dict[str, Any]) -> list[str]:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("GSM8K row has no non-empty 'question'")
    return [f"Solve the problem step by step.\n\nQuestion: {question}\nAnswer:"]


def _humaneval_prompt(row: dict[str, Any]) -> list[str]:
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("HumanEval row has no non-empty 'prompt'")
    return [prompt]


def _mbpp_prompt(row: dict[str, Any]) -> list[str]:
    problem = row.get("text")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("MBPP row has no non-empty 'text'")
    return [
        f"Write a Python solution for the following task.\n\n"
        f"{problem}\n\n```python\n"
    ]


def _mt_bench_prompts(row: dict[str, Any]) -> list[str]:
    prompts = row.get("prompt")
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(prompts, (list, tuple)):
        raise ValueError("MT-Bench row has no string/list 'prompt'")
    # For a speed benchmark, turns are independent requests. This keeps inputs
    # identical across models instead of feeding model-dependent first answers
    # into the second turn.
    result = [str(prompt) for prompt in prompts if str(prompt).strip()]
    if not result:
        raise ValueError("MT-Bench row has no non-empty turns")
    return result


DATASETS: dict[str, DatasetSpec] = {
    "math500": DatasetSpec(
        "math500", "HuggingFaceH4/MATH-500", None, "test", _math_prompt
    ),
    "gsm8k": DatasetSpec("gsm8k", "openai/gsm8k", "main", "test", _gsm8k_prompt),
    "humaneval": DatasetSpec(
        "humaneval", "openai/openai_humaneval", None, "test", _humaneval_prompt
    ),
    "mbpp": DatasetSpec(
        "mbpp", "google-research-datasets/mbpp", "full", "test", _mbpp_prompt
    ),
    "mt_bench": DatasetSpec(
        "mt_bench",
        "HuggingFaceH4/mt_bench_prompts",
        None,
        "train",
        _mt_bench_prompts,
    ),
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--dflash-model",
        default=str(
            repo_root
            / "dflash-output/dflash_t5gemma2_mixture_200k/checkpoints/5"
        ),
    )
    parser.add_argument(
        "--dflare-model",
        default=str(
            repo_root
            / "dflash-output/dflare_t5gemma2_mixture_200k/checkpoints/5"
        ),
    )
    parser.add_argument(
        "--dspark-model",
        default=str(
            repo_root
            / "dflash-output/dspark_t5gemma2_mixture_200k/checkpoints/5"
        ),
    )
    parser.add_argument(
        "--speculative-token-counts",
        type=int,
        nargs="+",
        default=[3, 7],
        metavar="K",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["dflash", "dflare", "dspark"],
        default=["dflash", "dflare", "dspark"],
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=0,
        help="Prompts per dataset; 0 runs each complete split.",
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--server-timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--include-baseline", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed run/dataset summaries.",
    )
    parser.add_argument(
        "--refresh-prompts",
        action="store_true",
        help="Reload datasets instead of reusing saved prompt snapshots.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checkpoints and print the run matrix without starting vLLM.",
    )
    return parser.parse_args()


def _prompt_snapshot_path(output_dir: Path, dataset_name: str) -> Path:
    return output_dir / "prompts" / f"{dataset_name}.jsonl"


def load_dataset_prompts(
    spec: DatasetSpec,
    *,
    output_dir: Path,
    num_prompts: int,
    refresh: bool,
) -> list[str]:
    snapshot = _prompt_snapshot_path(output_dir, spec.name)
    if snapshot.exists() and not refresh:
        prompts = [
            json.loads(line)["prompt"]
            for line in snapshot.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        kwargs: dict[str, Any] = {"split": spec.split}
        if spec.config is not None:
            dataset = load_dataset(spec.path, spec.config, **kwargs)
        else:
            dataset = load_dataset(spec.path, **kwargs)

        prompts = []
        for row in dataset:
            prompts.extend(spec.format_row(dict(row)))

        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open("w", encoding="utf-8") as handle:
            for prompt in prompts:
                handle.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")

    if num_prompts > 0:
        prompts = prompts[:num_prompts]
    if not prompts:
        raise ValueError(f"No prompts loaded for {spec.name}")
    return prompts


def prompt_hash(prompts: list[str]) -> str:
    payload = "\n\0\n".join(prompts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def benchmark_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_checkpoint(path_value: str, expected_architecture: str) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    resolved = Path(normalize_path(path_value, base=repo_root))
    config_path = resolved / "config.json"
    weights_path = resolved / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint weights: {weights_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    if expected_architecture not in architectures:
        raise ValueError(
            f"Expected {expected_architecture} in {config_path}, got {architectures}"
        )
    block_size = int(config.get("block_size", 0))
    if block_size <= 1:
        raise ValueError(f"Invalid block_size={block_size} in {config_path}")
    return str(resolved.resolve())


def build_run_specs(args: argparse.Namespace) -> list[RunSpec]:
    models = {
        "dflash": validate_checkpoint(args.dflash_model, "DFlashDraftModel"),
        "dflare": validate_checkpoint(args.dflare_model, "DFlareDraftModel"),
        "dspark": validate_checkpoint(args.dspark_model, "DSparkDraftModel"),
    }
    counts = list(dict.fromkeys(args.speculative_token_counts))
    if any(count <= 0 or count > MAX_SPECULATIVE_TOKENS for count in counts):
        raise ValueError(
            f"Speculative token counts must be between 1 and "
            f"{MAX_SPECULATIVE_TOKENS}"
        )

    runs = [RunSpec("baseline", None, None)] if args.include_baseline else []
    for method in args.methods:
        checkpoint = models[method]
        runs.extend(RunSpec(f"{method}_k{k}", checkpoint, k) for k in counts)
    return runs


def server_command(
    run: RunSpec,
    *,
    args: argparse.Namespace,
    served_model_name: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.base_model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--served-model-name",
        served_model_name,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--no-enable-chunked-prefill",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--seed",
        str(args.seed),
    ]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if run.checkpoint is not None:
        speculative_config = {
            "model": run.checkpoint,
            "method": "dflash",
            "num_speculative_tokens": run.num_speculative_tokens,
        }
        cmd.extend(["--speculative-config", json.dumps(speculative_config)])
    return cmd


def start_server(
    run: RunSpec,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    served_model_name: str,
) -> subprocess.Popen:
    cmd = server_command(run, args=args, served_model_name=served_model_name)
    env = os.environ.copy()
    env["VLLM_PLUGINS"] = "t5gemma2_vllm_plugin"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

    log_dir = output_dir / "server-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (log_dir / f"{run.name}.log").open("w", encoding="utf-8")
    print(f"[server:{run.name}] {' '.join(cmd)}", flush=True)
    return subprocess.Popen(  # noqa: S603
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
        start_new_session=True,
    )


async def run_warmup(
    *,
    run_name: str,
    served_model_name: str,
    prompts: list[str],
    base_url: str,
    args: argparse.Namespace,
) -> None:
    if args.warmup <= 0:
        return
    timeout = httpx.Timeout(args.timeout_s, connect=20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for index, prompt in enumerate(prompts[: args.warmup]):
            sample = await one_completion(
                client,
                f"{base_url}/v1/completions",
                served_model_name,
                prompt,
                -1 - index,
                run_name,
                args,
            )
            if not sample.ok:
                raise RuntimeError(f"Warmup failed for {run_name}: {sample.error}")


def run_result_dir(output_dir: Path, run_name: str, dataset_name: str) -> Path:
    return output_dir / "runs" / run_name / dataset_name


def load_completed_summary(path: Path, run_name: str) -> dict[str, Any] | None:
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        return None
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return data.get(run_name)


async def benchmark_dataset(
    *,
    run: RunSpec,
    dataset_name: str,
    prompts: list[str],
    served_model_name: str,
    base_url: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    result_dir = run_result_dir(output_dir, run.name, dataset_name)
    result_dir.mkdir(parents=True, exist_ok=True)

    await run_warmup(
        run_name=run.name,
        served_model_name=served_model_name,
        prompts=prompts,
        base_url=base_url,
        args=args,
    )
    before = await fetch_metrics(base_url)

    # Warmup is performed above and deliberately excluded from measured samples
    # and speculative-acceptance counters.
    measured_args = argparse.Namespace(**vars(args))
    measured_args.warmup = 0
    measured_args.num_prompts = 0
    samples, totals = await run_load(
        run_name=run.name,
        served_model=served_model_name,
        prompts=prompts,
        base_url=base_url,
        args=measured_args,
    )
    after = await fetch_metrics(base_url)

    (result_dir / "metrics.before.prom").write_text(before, encoding="utf-8")
    (result_dir / "metrics.after.prom").write_text(after, encoding="utf-8")
    acceptance = acceptance_delta(before, after) if run.checkpoint else None
    summary = summarize_run(samples, totals, acceptance)
    summary["metadata"] = {
        "dataset": dataset_name,
        "run": run.name,
        "checkpoint": run.checkpoint,
        "num_speculative_tokens": run.num_speculative_tokens,
        "prompt_count": len(prompts),
        "prompt_sha256": prompt_hash(prompts),
    }

    write_samples_csv(result_dir / "samples.csv", samples)
    wrapped = {run.name: summary}
    (result_dir / "summary.json").write_text(
        json.dumps(wrapped, indent=2), encoding="utf-8"
    )
    write_summary_csv(result_dir / "summary.csv", wrapped)
    return summary


def _nested(summary: dict[str, Any], metric: str, stat: str) -> Any:
    return (summary.get(metric) or {}).get(stat)


def _scaled(value: float | None, factor: float) -> float | None:
    return value * factor if value is not None else None


def comparison_rows(
    summaries: dict[tuple[str, str], dict[str, Any]],
    datasets: list[str],
    runs: list[RunSpec],
) -> list[dict[str, Any]]:
    rows = []
    for dataset_name in datasets:
        baseline = summaries.get(("baseline", dataset_name))
        baseline_tps = (
            baseline.get("totals", {}).get("output_throughput_tps")
            if baseline
            else None
        )
        for run in runs:
            summary = summaries.get((run.name, dataset_name))
            if summary is None:
                continue
            totals = summary["totals"]
            output_tps = totals["output_throughput_tps"]
            acc = summary.get("speculative_acceptance") or {}
            rows.append(
                {
                    "dataset": dataset_name,
                    "run": run.name,
                    "method": run.name.rsplit("_k", 1)[0]
                    if run.checkpoint
                    else "baseline",
                    "num_speculative_tokens": run.num_speculative_tokens,
                    "requests": totals["ok_requests"],
                    "output_tokens": totals["output_tokens"],
                    "request_throughput_rps": totals["request_throughput_rps"],
                    "output_throughput_tps": output_tps,
                    "speedup_vs_baseline": output_tps / baseline_tps
                    if baseline_tps
                    else None,
                    "latency_mean_ms": _scaled(
                        _nested(summary, "latency_s", "mean"), 1000
                    ),
                    "latency_p50_ms": _scaled(
                        _nested(summary, "latency_s", "p50"), 1000
                    ),
                    "latency_p95_ms": _scaled(
                        _nested(summary, "latency_s", "p95"), 1000
                    ),
                    "latency_p99_ms": _scaled(
                        _nested(summary, "latency_s", "p99"), 1000
                    ),
                    "ttft_mean_ms": _nested(summary, "ttft_ms", "mean"),
                    "ttft_p50_ms": _nested(summary, "ttft_ms", "p50"),
                    "ttft_p95_ms": _nested(summary, "ttft_ms", "p95"),
                    "ttft_p99_ms": _nested(summary, "ttft_ms", "p99"),
                    "itl_mean_ms": _nested(summary, "mean_itl_ms", "mean"),
                    "itl_p50_ms": _nested(summary, "mean_itl_ms", "p50"),
                    "itl_p95_ms": _nested(summary, "mean_itl_ms", "p95"),
                    "itl_p99_ms": _nested(summary, "mean_itl_ms", "p99"),
                    "tpot_mean_ms": _nested(summary, "tpot_ms", "mean"),
                    "tpot_p50_ms": _nested(summary, "tpot_ms", "p50"),
                    "tpot_p95_ms": _nested(summary, "tpot_ms", "p95"),
                    "acceptance_rate": acc.get("acceptance_rate"),
                    "mean_acceptance_length": acc.get(
                        "mean_acceptance_length_including_bonus"
                    ),
                }
            )
    return rows


def write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def write_comparison_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "Dataset",
        "Run",
        "Req",
        "Output tok/s",
        "Speedup",
        "Latency mean/p95 ms",
        "TTFT mean/p95 ms",
        "ITL mean/p95 ms",
        "TPOT mean/p95 ms",
        "Accept rate",
        "Accept len",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        speedup = row["speedup_vs_baseline"]
        acceptance = row["acceptance_rate"]
        values = [
            row["dataset"],
            row["run"],
            str(row["requests"]),
            _fmt(row["output_throughput_tps"]),
            f"{_fmt(speedup, 3)}x" if speedup is not None else "—",
            f"{_fmt(row['latency_mean_ms'])}/{_fmt(row['latency_p95_ms'])}",
            f"{_fmt(row['ttft_mean_ms'])}/{_fmt(row['ttft_p95_ms'])}",
            f"{_fmt(row['itl_mean_ms'])}/{_fmt(row['itl_p95_ms'])}",
            f"{_fmt(row['tpot_mean_ms'])}/{_fmt(row['tpot_p95_ms'])}",
            f"{_fmt(acceptance * 100)}%" if acceptance is not None else "—",
            _fmt(row["mean_acceptance_length"], 3),
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main_async() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(normalize_path(args.output_dir, base=repo_root)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = build_run_specs(args)
    print("Run matrix:", flush=True)
    for run in runs:
        print(
            f"  {run.name}: checkpoint={run.checkpoint or args.base_model}, "
            f"K={run.num_speculative_tokens or '-'}",
            flush=True,
        )
    print(f"Datasets: {', '.join(args.datasets)}", flush=True)
    if args.dry_run:
        return

    prompt_sets = {
        name: load_dataset_prompts(
            DATASETS[name],
            output_dir=output_dir,
            num_prompts=args.num_prompts,
            refresh=args.refresh_prompts,
        )
        for name in args.datasets
    }
    fingerprint_args = {
        key: value
        for key, value in vars(args).items()
        if key not in {"dry_run", "refresh_prompts", "resume", "output_dir"}
    }
    fingerprint_input = {
        "args": fingerprint_args,
        "runs": [run.__dict__ for run in runs],
        "prompt_sets": {
            name: {"count": len(prompts), "sha256": prompt_hash(prompts)}
            for name, prompts in prompt_sets.items()
        },
    }
    fingerprint = benchmark_fingerprint(fingerprint_input)
    run_config_path = output_dir / "run_config.json"
    if args.resume and run_config_path.is_file():
        previous = json.loads(run_config_path.read_text(encoding="utf-8"))
        previous_fingerprint = previous.get("benchmark_fingerprint")
        if previous_fingerprint is not None and previous_fingerprint != fingerprint:
            raise ValueError(
                "The existing output directory was created with a different "
                "benchmark setup. Use another --output-dir or pass --no-resume."
            )
    run_config_path.write_text(
        json.dumps(
            {
                **fingerprint_input,
                "benchmark_fingerprint": fingerprint,
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    base_url = f"http://{args.host}:{args.port}"
    for run in runs:
        pending = []
        for dataset_name in args.datasets:
            result_dir = run_result_dir(output_dir, run.name, dataset_name)
            completed = (
                load_completed_summary(result_dir, run.name) if args.resume else None
            )
            if completed is not None:
                summaries[(run.name, dataset_name)] = completed
                print(f"[resume] {run.name}/{dataset_name}", flush=True)
            else:
                pending.append(dataset_name)
        if not pending:
            continue

        served_model_name = f"t5gemma2-{run.name}"
        proc = start_server(
            run,
            args=args,
            output_dir=output_dir,
            served_model_name=served_model_name,
        )
        try:
            await wait_for_health(base_url, proc, args.server_timeout_s)
            for dataset_name in pending:
                print(f"[benchmark] {run.name}/{dataset_name}", flush=True)
                summary = await benchmark_dataset(
                    run=run,
                    dataset_name=dataset_name,
                    prompts=prompt_sets[dataset_name],
                    served_model_name=served_model_name,
                    base_url=base_url,
                    args=args,
                    output_dir=output_dir,
                )
                summaries[(run.name, dataset_name)] = summary
        finally:
            stop_server(proc)

    rows = comparison_rows(summaries, args.datasets, runs)
    write_comparison_csv(output_dir / "comparison.csv", rows)
    write_comparison_markdown(output_dir / "comparison.md", rows)
    (output_dir / "results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(f"Results: {output_dir / 'comparison.md'}", flush=True)


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        # Child process groups are terminated in each run's finally block.
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise


if __name__ == "__main__":
    main()
