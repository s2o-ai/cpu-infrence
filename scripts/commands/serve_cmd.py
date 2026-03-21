"""s2o serve — Start the inference server."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import typer

# Import serve function from scripts/serve.py
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
from serve import serve as do_serve  # noqa: E402


def serve(
    model: str = typer.Argument(help="Path to GGUF model or name to search in models/"),
    host: str = typer.Option("127.0.0.1", help="Bind address"),
    port: int = typer.Option(8080, help="Port"),
    ctx_size: int = typer.Option(4096, "--ctx-size", help="Context size"),
    parallel: int = typer.Option(4, help="Parallel slots"),
    threads: int = typer.Option(None, help="CPU threads (default: auto-detect)"),
    api_key: str = typer.Option(None, "--api-key", help="API key for authentication"),
):
    """Start the OpenAI-compatible inference server."""
    args = SimpleNamespace(
        model=model, host=host, port=port, ctx_size=ctx_size,
        parallel=parallel, threads=threads, api_key=api_key,
    )
    do_serve(args)
