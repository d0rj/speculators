#!/usr/bin/env python3
"""Benchmark google/t5gemma-2-1b-1b vs a trained DFlash head on vLLM.

The runner starts two vLLM servers sequentially with the same serving flags,
uses the same prompt set for both runs, records OpenAI-compatible streaming
latencies, and snapshots vLLM Prometheus counters for speculative acceptance.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_MODEL = "google/t5gemma-2-1b-1b"
DEFAULT_DFLASH = "../t5gemma-2-1b-1b.dflash-dev"
DEFAULT_OUTPUT = "benchmark-results/t5gemma2_gsm8k_vllm"
HTTP_OK = 200


@dataclass
class RequestSample:
    run: str
    index: int
    ok: bool
    latency_s: float
    ttft_ms: float | None
    mean_itl_ms: float | None
    p50_itl_ms: float | None
    p95_itl_ms: float | None
    tpot_ms: float | None
    output_tokens: int
    prompt_chars: int
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dflash-model", default=DEFAULT_DFLASH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--prompt-file", default=None, help="Optional JSONL/TXT prompt file."
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=128,
        help="Measured prompts. Use 0 or a negative value to run the whole split.",
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--server-timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-dflash", action="store_true")
    return parser.parse_args()


def normalize_path(value: str, *, base: Path) -> str:
    if re.match(r"^[A-Za-z]:[\\/]", value):
        drive = value[0].lower()
        rest = value[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{rest}"
    path = Path(value)
    if path.exists():
        return str(path.resolve())
    candidate = (base / value).resolve()
    return str(candidate) if candidate.exists() else value


def load_prompts(args: argparse.Namespace, output_dir: Path) -> list[str]:
    limit = args.num_prompts if args.num_prompts > 0 else None
    if args.prompt_file:
        path = Path(normalize_path(args.prompt_file, base=Path.cwd()))
        prompts: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if path.suffix.lower() == ".jsonl":
                obj = json.loads(line)
                prompts.append(
                    str(obj.get("prompt") or obj.get("question") or obj.get("text"))
                )
            else:
                prompts.append(line)
            if limit is not None and len(prompts) >= limit + args.warmup:
                break
        return prompts

    try:
        load_dataset = importlib.import_module("datasets").load_dataset
        ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
        prompts = []
        for row in ds:
            question = str(row.get("question") or row.get("prompt") or row.get("text"))
            prompts.append(
                f"Solve the problem step by step.\n\nQuestion: {question}\nAnswer:"
            )
            if limit is not None and len(prompts) >= limit + args.warmup:
                break
        if prompts:
            return prompts
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] failed to load {args.dataset}: {exc}", file=sys.stderr)

    fallback = [
        (
            "Solve the problem step by step.\n\n"
            "Question: If a store has 12 apples and sells 5, "
            "how many apples are left?\nAnswer:"
        ),
        (
            "Solve the problem step by step.\n\n"
            "Question: A train travels 60 miles in 2 hours. "
            "What is its average speed?\nAnswer:"
        ),
        (
            "Solve the problem step by step.\n\n"
            "Question: There are 4 boxes with 6 pencils each. "
            "How many pencils are there?\nAnswer:"
        ),
        (
            "Solve the problem step by step.\n\n"
            "Question: Maria had $20 and bought lunch for $7.50. "
            "How much money remains?\nAnswer:"
        ),
    ]
    fallback_count = max(args.num_prompts, 4) if args.num_prompts > 0 else 4
    prompts = [fallback[i % len(fallback)] for i in range(fallback_count)]
    (output_dir / "DATASET_FALLBACK_USED.txt").write_text(
        "HF dataset load failed; repeated local fallback prompts were used.\n",
        encoding="utf-8",
    )
    return prompts


def quantile(values: list[float], q: float) -> float | None:
    clean = sorted(v for v in values if v is not None and math.isfinite(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - pos) + clean[hi] * (pos - lo)


def summarize_values(values: list[float | None]) -> dict[str, float | int | None]:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(clean),
        "mean": statistics.mean(clean),
        "std": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "p50": quantile(clean, 0.50),
        "p90": quantile(clean, 0.90),
        "p95": quantile(clean, 0.95),
        "p99": quantile(clean, 0.99),
        "max": max(clean),
    }


async def wait_for_health(
    base_url: str, proc: subprocess.Popen, timeout_s: float
) -> None:
    deadline = time.perf_counter() + timeout_s
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.perf_counter() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with code {proc.returncode}")
            try:
                resp = await client.get(f"{base_url}/health")
                if resp.status_code == HTTP_OK:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2.0)
    raise TimeoutError(f"vLLM did not become healthy within {timeout_s}s")


def start_server(
    *,
    run_name: str,
    model: str,
    served_name: str,
    args: argparse.Namespace,
    output_dir: Path,
    repo_root: Path,
) -> subprocess.Popen:
    log_path = output_dir / f"{run_name}.server.log"
    env = os.environ.copy()
    env["VLLM_PLUGINS"] = "t5gemma2_vllm_plugin"
    env.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    common = [
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--served-model-name",
        served_name,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--no-enable-chunked-prefill",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--seed",
        str(args.seed),
    ]
    if args.enforce_eager:
        common.append("--enforce-eager")

    if run_name == "dflash":
        cmd = [
            sys.executable,
            str(repo_root / "speculators" / "scripts" / "serve_t5gemma_dflash.py"),
            "--checkpoint",
            model,
            *common,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            model,
            *common,
        ]

    print(f"[server:{run_name}] {' '.join(cmd)}", flush=True)
    log_f = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(  # noqa: S603
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(repo_root / "speculators"),
        start_new_session=True,
    )


def _process_group_has_live_members(pgid: int) -> bool:
    """Return whether a Linux process group contains a non-zombie process."""
    proc_root = Path("/proc")
    if proc_root.is_dir():
        for stat_path in proc_root.glob("[0-9]*/stat"):
            try:
                stat = stat_path.read_text(encoding="utf-8")
                # The command in /proc/PID/stat is parenthesized and may contain
                # spaces. Fields after it start with state, ppid, and pgrp.
                fields = stat[stat.rfind(")") + 2 :].split()
                if len(fields) >= 3 and int(fields[2]) == pgid:
                    if fields[0] != "Z":
                        return True
            except (FileNotFoundError, PermissionError, ValueError):
                continue
        return False

    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_process_group(
    proc: subprocess.Popen,
    pgid: int,
    timeout_s: float,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        proc.poll()  # Reap the frontend as soon as it exits.
        if not _process_group_has_live_members(pgid):
            return True
        time.sleep(0.2)
    proc.poll()
    return not _process_group_has_live_members(pgid)


def stop_server(
    proc: subprocess.Popen | None,
    *,
    graceful_timeout_s: float = 30.0,
    terminate_timeout_s: float = 15.0,
    kill_timeout_s: float = 15.0,
) -> bool:
    """Stop the vLLM frontend and every worker in its process group."""
    if proc is None:
        return True

    # start_server uses start_new_session=True, therefore the frontend PID is
    # also the process-group ID even if the frontend has already exited.
    pgid = proc.pid
    if not _process_group_has_live_members(pgid):
        proc.poll()
        return True

    # Let the frontend coordinate an orderly EngineCore shutdown first. Sending
    # SIGTERM to every multiprocessing child at once can wedge CUDA teardown.
    if proc.poll() is None:
        try:
            proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
    if _wait_for_process_group(proc, pgid, graceful_timeout_s):
        return True

    for sig, timeout_s in (
        (signal.SIGTERM, terminate_timeout_s),
        (signal.SIGKILL, kill_timeout_s),
    ):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            proc.poll()
            return True
        if _wait_for_process_group(proc, pgid, timeout_s):
            return True

    print(
        f"WARNING: vLLM process group {pgid} did not exit after SIGKILL. "
        "Do not start the next server until the GPU context is released; "
        "WSL may need to be restarted.",
        file=sys.stderr,
        flush=True,
    )
    return False


async def fetch_metrics(base_url: str) -> str:
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(f"{base_url}/metrics")
        resp.raise_for_status()
        return resp.text


def prometheus_counter_total(text: str, metric: str) -> float:
    total = 0.0
    pattern = re.compile(
        rf"^{re.escape(metric)}(?:_total)?(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$"
    )
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if match:
            total += float(match.group(1))
    return total


def prometheus_per_pos(text: str, metric: str) -> dict[str, float]:
    out: dict[str, float] = {}
    pattern = re.compile(
        rf"^{re.escape(metric)}(?:_total)?\{{([^}}]*)\}}\s+([-+0-9.eE]+)$"
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        labels, value = match.groups()
        pos_match = re.search(r'position="([^"]+)"', labels)
        if pos_match:
            out[pos_match.group(1)] = out.get(pos_match.group(1), 0.0) + float(value)
    return out


def acceptance_delta(before: str, after: str) -> dict[str, Any]:
    drafts = prometheus_counter_total(
        after, "vllm:spec_decode_num_drafts"
    ) - prometheus_counter_total(before, "vllm:spec_decode_num_drafts")
    draft_tokens = prometheus_counter_total(
        after, "vllm:spec_decode_num_draft_tokens"
    ) - prometheus_counter_total(before, "vllm:spec_decode_num_draft_tokens")
    accepted_tokens = prometheus_counter_total(
        after, "vllm:spec_decode_num_accepted_tokens"
    ) - prometheus_counter_total(before, "vllm:spec_decode_num_accepted_tokens")

    before_pos = prometheus_per_pos(
        before, "vllm:spec_decode_num_accepted_tokens_per_pos"
    )
    after_pos = prometheus_per_pos(
        after, "vllm:spec_decode_num_accepted_tokens_per_pos"
    )
    per_pos = {
        pos: after_pos.get(pos, 0.0) - before_pos.get(pos, 0.0)
        for pos in sorted(
            set(before_pos) | set(after_pos), key=lambda x: int(x) if x.isdigit() else x
        )
    }
    return {
        "num_drafts": drafts,
        "num_draft_tokens": draft_tokens,
        "num_accepted_tokens": accepted_tokens,
        "acceptance_rate": accepted_tokens / draft_tokens if draft_tokens > 0 else None,
        "mean_acceptance_length_including_bonus": 1 + accepted_tokens / drafts
        if drafts > 0
        else None,
        "accepted_tokens_per_position": per_pos,
        "acceptance_rate_per_position": {
            pos: value / drafts if drafts > 0 else None
            for pos, value in per_pos.items()
        },
    }


async def one_completion(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    index: int,
    run_name: str,
    args: argparse.Namespace,
) -> RequestSample:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        # Token lists let us account for multiple accepted speculative tokens
        # delivered in one SSE chunk. Without them, "ITL" measures inter-chunk
        # latency and systematically penalizes speculative decoding.
        "logprobs": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token_at: float | None = None
    chunk_times: list[float] = []
    output_tokens = 0
    saw_nonempty_text = False
    try:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    continue
                obj = json.loads(data)
                usage = obj.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    output_tokens = int(usage["completion_tokens"])
                choices = obj.get("choices") or []
                choice = choices[0] if choices else {}
                text = choice.get("text") or ""
                saw_nonempty_text = saw_nonempty_text or bool(text)
                logprobs = choice.get("logprobs") or {}
                tokens = logprobs.get("tokens") or []
                token_count = len(tokens) if tokens else (1 if text else 0)
                if token_count:
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    # Tokens in one speculative chunk become visible together,
                    # so their intra-chunk ITLs are correctly represented as 0.
                    chunk_times.extend([now] * token_count)
        finished = time.perf_counter()
        if output_tokens > 1 and not saw_nonempty_text:
            raise RuntimeError(
                "vLLM reported generated tokens but emitted no text; this usually "
                "indicates non-finite logits or repeated special tokens"
            )
        gaps = [
            (b - a) * 1000
            for a, b in zip(chunk_times, chunk_times[1:], strict=False)
        ]
        if output_tokens == 0:
            output_tokens = len(chunk_times)
        tpot = None
        if first_token_at is not None and output_tokens > 1:
            tpot = ((finished - first_token_at) / (output_tokens - 1)) * 1000
        return RequestSample(
            run=run_name,
            index=index,
            ok=True,
            latency_s=finished - started,
            ttft_ms=(first_token_at - started) * 1000 if first_token_at else None,
            mean_itl_ms=statistics.mean(gaps) if gaps else None,
            p50_itl_ms=quantile(gaps, 0.50),
            p95_itl_ms=quantile(gaps, 0.95),
            tpot_ms=tpot,
            output_tokens=output_tokens,
            prompt_chars=len(prompt),
        )
    except Exception as exc:  # noqa: BLE001
        finished = time.perf_counter()
        return RequestSample(
            run=run_name,
            index=index,
            ok=False,
            latency_s=finished - started,
            ttft_ms=None,
            mean_itl_ms=None,
            p50_itl_ms=None,
            p95_itl_ms=None,
            tpot_ms=None,
            output_tokens=0,
            prompt_chars=len(prompt),
            error=repr(exc),
        )


async def run_load(
    *,
    run_name: str,
    served_model: str,
    prompts: list[str],
    base_url: str,
    args: argparse.Namespace,
) -> tuple[list[RequestSample], dict[str, float]]:
    url = f"{base_url}/v1/completions"
    timeout = httpx.Timeout(args.timeout_s, connect=20.0)
    limits = httpx.Limits(max_connections=max(args.concurrency, 1) + 4)
    samples: list[RequestSample] = []

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        warmup_prompts = prompts[: args.warmup]
        for i, prompt in enumerate(warmup_prompts):
            sample = await one_completion(
                client, url, served_model, prompt, -1 - i, run_name, args
            )
            if not sample.ok:
                raise RuntimeError(f"warmup failed: {sample.error}")

        if args.num_prompts > 0:
            bench_prompts = prompts[args.warmup : args.warmup + args.num_prompts]
        else:
            bench_prompts = prompts[args.warmup :]
        sem = asyncio.Semaphore(args.concurrency)
        started = time.perf_counter()

        async def guarded(i: int, prompt: str) -> RequestSample:
            async with sem:
                return await one_completion(
                    client, url, served_model, prompt, i, run_name, args
                )

        tasks = [
            asyncio.create_task(guarded(i, p)) for i, p in enumerate(bench_prompts)
        ]
        for task in asyncio.as_completed(tasks):
            samples.append(await task)
        finished = time.perf_counter()

    ok = [s for s in samples if s.ok]
    totals = {
        "wall_time_s": finished - started,
        "requests": len(samples),
        "ok_requests": len(ok),
        "output_tokens": sum(s.output_tokens for s in ok),
        "request_throughput_rps": len(ok) / (finished - started)
        if finished > started
        else 0.0,
        "output_throughput_tps": sum(s.output_tokens for s in ok) / (finished - started)
        if finished > started
        else 0.0,
    }
    return sorted(samples, key=lambda s: s.index), totals


def summarize_run(
    samples: list[RequestSample],
    totals: dict[str, float],
    acceptance: dict[str, Any] | None,
) -> dict[str, Any]:
    ok = [s for s in samples if s.ok]
    per_req_tps = [
        (s.output_tokens / s.latency_s) if s.latency_s > 0 and s.output_tokens else None
        for s in ok
    ]
    return {
        "totals": totals,
        "errors": [asdict(s) for s in samples if not s.ok],
        "latency_s": summarize_values([s.latency_s for s in ok]),
        "ttft_ms": summarize_values([s.ttft_ms for s in ok]),
        "mean_itl_ms": summarize_values([s.mean_itl_ms for s in ok]),
        "p50_itl_ms": summarize_values([s.p50_itl_ms for s in ok]),
        "p95_itl_ms": summarize_values([s.p95_itl_ms for s in ok]),
        "tpot_ms": summarize_values([s.tpot_ms for s in ok]),
        "output_tokens": summarize_values([float(s.output_tokens) for s in ok]),
        "per_request_output_tps": summarize_values(per_req_tps),
        "speculative_acceptance": acceptance,
    }


def write_samples_csv(path: Path, samples: list[RequestSample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(samples[0]).keys())
            if samples
            else list(RequestSample.__dataclass_fields__.keys()),
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))


def write_summary_csv(path: Path, summaries: dict[str, Any]) -> None:
    rows = []
    for run_name, summary in summaries.items():
        for metric in [
            "latency_s",
            "ttft_ms",
            "mean_itl_ms",
            "tpot_ms",
            "output_tokens",
            "per_request_output_tps",
        ]:
            row = {"run": run_name, "metric": metric}
            row.update(summary[metric])
            rows.append(row)
        totals = summary["totals"]
        rows.append(
            {
                "run": run_name,
                "metric": "overall_output_throughput_tps",
                "mean": totals["output_throughput_tps"],
                "count": totals["ok_requests"],
            }
        )
        rows.append(
            {
                "run": run_name,
                "metric": "overall_request_throughput_rps",
                "mean": totals["request_throughput_rps"],
                "count": totals["ok_requests"],
            }
        )
    fieldnames = [
        "run",
        "metric",
        "count",
        "mean",
        "std",
        "min",
        "p50",
        "p90",
        "p95",
        "p99",
        "max",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


async def benchmark_one(
    *,
    run_name: str,
    model: str,
    served_name: str,
    prompts: list[str],
    args: argparse.Namespace,
    output_dir: Path,
    repo_root: Path,
) -> tuple[list[RequestSample], dict[str, Any]]:
    base_url = f"http://{args.host}:{args.port}"
    proc = start_server(
        run_name=run_name,
        model=model,
        served_name=served_name,
        args=args,
        output_dir=output_dir,
        repo_root=repo_root,
    )
    try:
        await wait_for_health(base_url, proc, args.server_timeout_s)
        before = await fetch_metrics(base_url)
        samples, totals = await run_load(
            run_name=run_name,
            served_model=served_name,
            prompts=prompts,
            base_url=base_url,
            args=args,
        )
        after = await fetch_metrics(base_url)
        (output_dir / f"{run_name}.metrics.before.prom").write_text(
            before, encoding="utf-8"
        )
        (output_dir / f"{run_name}.metrics.after.prom").write_text(
            after, encoding="utf-8"
        )
        acc = acceptance_delta(before, after) if run_name == "dflash" else None
        summary = summarize_run(samples, totals, acc)
        return samples, summary
    finally:
        if not stop_server(proc):
            raise RuntimeError(
                f"vLLM process group for {run_name} is still alive after "
                "SIGKILL. Restart WSL before resuming the benchmark."
            )


async def main_async() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(normalize_path(args.output_dir, base=repo_root)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dflash_model = normalize_path(args.dflash_model, base=repo_root)
    prompts = load_prompts(args, output_dir)
    if args.num_prompts > 0:
        needed = args.num_prompts + args.warmup
        if len(prompts) < needed:
            prompts = (prompts * math.ceil(needed / max(len(prompts), 1)))[:needed]
        else:
            prompts = prompts[:needed]
    elif len(prompts) <= args.warmup:
        raise ValueError(
            f"Need more than warmup={args.warmup} prompts for full-split run; "
            f"loaded {len(prompts)} prompts."
        )

    with (output_dir / "prompts.jsonl").open("w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
    (output_dir / "run_config.json").write_text(
        json.dumps({**vars(args), "dflash_model_resolved": dflash_model}, indent=2),
        encoding="utf-8",
    )

    all_samples: list[RequestSample] = []
    summaries: dict[str, Any] = {}

    if not args.skip_baseline:
        samples, summary = await benchmark_one(
            run_name="baseline",
            model=args.base_model,
            served_name="t5gemma-2-1b-1b",
            prompts=prompts,
            args=args,
            output_dir=output_dir,
            repo_root=repo_root,
        )
        all_samples.extend(samples)
        summaries["baseline"] = summary

    if not args.skip_dflash:
        samples, summary = await benchmark_one(
            run_name="dflash",
            model=dflash_model,
            served_name="t5gemma-2-1b-1b-dflash",
            prompts=prompts,
            args=args,
            output_dir=output_dir,
            repo_root=repo_root,
        )
        all_samples.extend(samples)
        summaries["dflash"] = summary

    write_samples_csv(output_dir / "samples.csv", all_samples)
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    write_summary_csv(output_dir / "summary.csv", summaries)
    print(json.dumps(summaries, indent=2), flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
