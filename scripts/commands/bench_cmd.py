"""s2o bench — Run inference benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def bench(
    model: str,
    runs: int = 5,
    prompt_tokens: int = 512,
    gen_tokens: int = 128,
    threads: int | None = None,
    output: str | None = None,
    fmt: str = "md",
    server: bool = False,
    concurrency: str = "1,4",
):
    """Run inference benchmarks."""
    from serve import find_model  # noqa: E402

    model_path = find_model(model)

    # Auto-detect threads
    if threads is None:
        try:
            from engine.detect import detect_cpu
            cpu = detect_cpu()
            threads = cpu.recommendation.threads
        except Exception:
            import os
            threads = os.cpu_count() or 4

    # Run llama-bench
    console.print(f"\n[bold blue]Benchmarking:[/] {model_path.name}")
    console.print(f"[dim]Threads: {threads} | Runs: {runs} | PP: {prompt_tokens} | TG: {gen_tokens}[/]\n")

    from benchmarks.bench_runner import run_llama_bench
    from benchmarks.bench_report import save_report, generate_markdown

    with console.status("[bold]Running llama-bench...[/]"):
        result = run_llama_bench(
            model_path=str(model_path),
            runs=runs,
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            threads=threads,
        )

    # Display results
    table = Table(title="Benchmark Results")
    table.add_column("Test", style="cyan")
    table.add_column("Tokens", justify="right")
    table.add_column("Median tok/s", justify="right", style="green")
    table.add_column("Avg ± StdDev", justify="right")

    pp = result.prompt_processing
    if pp.samples:
        table.add_row(
            "Prompt processing", str(pp.tokens),
            f"{pp.median_tok_per_sec:.2f}",
            f"{pp.avg_tok_per_sec:.2f} ± {pp.stddev_tok_per_sec:.2f}",
        )

    tg = result.text_generation
    if tg.samples:
        table.add_row(
            "Text generation", str(tg.tokens),
            f"{tg.median_tok_per_sec:.2f}",
            f"{tg.avg_tok_per_sec:.2f} ± {tg.stddev_tok_per_sec:.2f}",
        )

    console.print(table)

    # Server throughput (optional)
    server_results = None
    if server:
        from benchmarks.bench_server import run_server_bench

        levels = [int(c.strip()) for c in concurrency.split(",")]
        server_results = []

        for level in levels:
            console.print(f"\n[bold]Server bench @ concurrency={level}...[/]")
            try:
                sr = run_server_bench(concurrency=level, num_requests=max(10, level * 3))
                server_results.append(sr)
                console.print(
                    f"  TTFT P50: {sr.ttft_p50_ms:.0f}ms | "
                    f"Latency P50: {sr.latency_p50_ms:.0f}ms | "
                    f"Throughput: {sr.gen_tok_per_sec_total:.1f} tok/s | "
                    f"Errors: {sr.errors}"
                )
            except ConnectionError as e:
                console.print(f"  [red]Server not reachable: {e}[/]")
                console.print("  [dim]Start server first: python scripts/s2o.py serve <model>[/]")
                break

    # Save report
    if output:
        save_report(result, output, fmt, server_results)
        console.print(f"\n[green]Report saved to {output}[/]")

    console.print(f"\n[dim]Build: {result.build_commit} | CPU: {result.cpu_info}[/]")
