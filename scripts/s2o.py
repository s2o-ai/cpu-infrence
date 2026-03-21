#!/usr/bin/env python3
"""S2O CLI — Zero-GPU AI Inference Platform."""

import sys
from pathlib import Path

# Set up import paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import typer  # noqa: E402

app = typer.Typer(
    name="s2o",
    help="S2O Zero-GPU AI Inference Platform",
    no_args_is_help=True,
)

# Register commands
from commands.info import info  # noqa: E402
from commands.build_cmd import build  # noqa: E402
from commands.serve_cmd import serve  # noqa: E402
from commands.models_cmd import models  # noqa: E402
from commands.run_cmd import run  # noqa: E402
from commands.quantize_cmd import quantize  # noqa: E402

app.command()(info)
app.command()(build)
app.command()(serve)
app.command()(models)
app.command()(run)
app.command()(quantize)

# Bench command — lazy import to avoid loading benchmarks module unnecessarily
@app.command()
def bench(
    model: str = typer.Argument(help="Path to GGUF model or name to search in models/"),
    runs: int = typer.Option(5, help="Number of benchmark runs"),
    prompt_tokens: int = typer.Option(512, "--pp", help="Prompt tokens to process"),
    gen_tokens: int = typer.Option(128, "--tg", help="Tokens to generate"),
    threads: int = typer.Option(None, help="CPU threads (default: auto-detect)"),
    output: str = typer.Option(None, "--output", "-o", help="Output report file path"),
    fmt: str = typer.Option("md", "--format", "-f", help="Output format: md or json"),
    server: bool = typer.Option(False, "--server", help="Also benchmark running server"),
    concurrency: str = typer.Option("1,4", "--concurrency", help="Concurrency levels (comma-separated)"),
):
    """Run inference benchmarks."""
    from commands.bench_cmd import bench as do_bench  # noqa: E402
    do_bench(
        model=model, runs=runs, prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens, threads=threads, output=output,
        fmt=fmt, server=server, concurrency=concurrency,
    )


if __name__ == "__main__":
    app()
