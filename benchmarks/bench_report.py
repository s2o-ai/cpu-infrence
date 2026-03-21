"""Benchmark report generation."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .bench_types import BenchResult, ServerBenchResult


def generate_markdown(
    result: BenchResult,
    server_results: list[ServerBenchResult] | None = None,
) -> str:
    """Generate a markdown benchmark report."""
    lines = [
        f"# Benchmark: {Path(result.model_path).stem}",
        "",
        f"- **Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        f"- **Build:** {result.build_commit}",
        f"- **CPU:** {result.cpu_info}",
        f"- **Threads:** {result.threads}",
        "",
        "## Inference Performance",
        "",
        "| Test | Tokens | Median tok/s | Avg ± StdDev |",
        "|------|--------|-------------|--------------|",
    ]

    pp = result.prompt_processing
    if pp.tokens > 0:
        lines.append(
            f"| Prompt processing | {pp.tokens} | "
            f"{pp.median_tok_per_sec:.2f} | "
            f"{pp.avg_tok_per_sec:.2f} ± {pp.stddev_tok_per_sec:.2f} |"
        )

    tg = result.text_generation
    if tg.tokens > 0:
        lines.append(
            f"| Text generation | {tg.tokens} | "
            f"{tg.median_tok_per_sec:.2f} | "
            f"{tg.avg_tok_per_sec:.2f} ± {tg.stddev_tok_per_sec:.2f} |"
        )

    if server_results:
        lines.extend([
            "",
            "## Server Throughput",
            "",
            "| Concurrency | TTFT P50 | TTFT P95 | Latency P50 | Latency P95 | Throughput | Errors |",
            "|------------|----------|----------|-------------|-------------|------------|--------|",
        ])
        for sr in server_results:
            lines.append(
                f"| {sr.concurrency} | "
                f"{sr.ttft_p50_ms:.0f}ms | {sr.ttft_p95_ms:.0f}ms | "
                f"{sr.latency_p50_ms:.0f}ms | {sr.latency_p95_ms:.0f}ms | "
                f"{sr.gen_tok_per_sec_total:.1f} tok/s | {sr.errors} |"
            )

    lines.extend(["", f"*Model: {result.model_path}*", ""])
    return "\n".join(lines)


def generate_json(
    result: BenchResult,
    server_results: list[ServerBenchResult] | None = None,
) -> str:
    """Generate a JSON benchmark report."""
    data = {
        "benchmark": asdict(result),
        "server": [asdict(sr) for sr in server_results] if server_results else [],
    }
    return json.dumps(data, indent=2)


def save_report(
    result: BenchResult,
    output_path: str,
    fmt: str = "md",
    server_results: list[ServerBenchResult] | None = None,
):
    """Save benchmark report to file."""
    if fmt == "json":
        content = generate_json(result, server_results)
    else:
        content = generate_markdown(result, server_results)

    Path(output_path).write_text(content)
