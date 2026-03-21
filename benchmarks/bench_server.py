"""Server throughput benchmarking via concurrent HTTP requests."""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from datetime import datetime, timezone

import httpx

from .bench_types import ServerBenchResult

_DEFAULT_PROMPT = "Explain the concept of machine learning in simple terms."


def _percentile(data: list[float], p: float) -> float:
    """Compute percentile of sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


async def _send_request(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    max_tokens: int,
) -> tuple[float, float, int]:
    """Send a streaming chat request. Returns (ttft_ms, total_ms, tokens)."""
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    start = time.perf_counter()
    ttft = None
    tokens = 0

    async with client.stream("POST", url, json=body, timeout=120) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            if ttft is None:
                ttft = (time.perf_counter() - start) * 1000

            try:
                chunk = json.loads(line[6:])
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content") or delta.get("reasoning_content") or ""
                if content:
                    tokens += 1  # approximate: count chunks with content
            except (json.JSONDecodeError, IndexError):
                pass

    total = (time.perf_counter() - start) * 1000
    return ttft or total, total, tokens


async def _run_concurrent(
    url: str,
    concurrency: int,
    num_requests: int,
    prompt: str,
    max_tokens: int,
) -> ServerBenchResult:
    """Run concurrent requests and collect metrics."""
    ttfts = []
    latencies = []
    total_tokens = 0
    errors = 0

    async with httpx.AsyncClient() as client:
        # Run requests in batches of `concurrency`
        for batch_start in range(0, num_requests, concurrency):
            batch_size = min(concurrency, num_requests - batch_start)
            tasks = [
                _send_request(client, url, prompt, max_tokens)
                for _ in range(batch_size)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    errors += 1
                else:
                    ttft, total, tokens = r
                    ttfts.append(ttft)
                    latencies.append(total)
                    total_tokens += tokens

    # Compute throughput
    total_time_s = sum(latencies) / 1000 if latencies else 1
    throughput = total_tokens / (total_time_s / len(latencies)) if latencies else 0

    return ServerBenchResult(
        concurrency=concurrency,
        total_requests=num_requests,
        ttft_p50_ms=round(_percentile(ttfts, 50), 1),
        ttft_p95_ms=round(_percentile(ttfts, 95), 1),
        ttft_p99_ms=round(_percentile(ttfts, 99), 1),
        gen_tok_per_sec_total=round(throughput, 1),
        latency_p50_ms=round(_percentile(latencies, 50), 1),
        latency_p95_ms=round(_percentile(latencies, 95), 1),
        latency_p99_ms=round(_percentile(latencies, 99), 1),
        errors=errors,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def run_server_bench(
    host: str = "127.0.0.1",
    port: int = 8080,
    concurrency: int = 4,
    num_requests: int = 10,
    prompt: str = _DEFAULT_PROMPT,
    max_tokens: int = 50,
) -> ServerBenchResult:
    """Run server throughput benchmark (synchronous wrapper)."""
    url = f"http://{host}:{port}/v1/chat/completions"

    # Verify server is healthy
    try:
        resp = httpx.get(f"http://{host}:{port}/health", timeout=5)
        if resp.status_code != 200:
            raise ConnectionError(f"Server unhealthy: {resp.status_code}")
    except httpx.ConnectError:
        raise ConnectionError(f"Server not reachable at {host}:{port}")

    return asyncio.run(
        _run_concurrent(url, concurrency, num_requests, prompt, max_tokens)
    )
