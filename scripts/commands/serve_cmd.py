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
    kv_quant: str = typer.Option(None, "--kv-quant", help="KV cache type: f16, q8_0, q4_0"),
    speculative: bool = typer.Option(False, "--speculative", help="Enable speculative decoding"),
    draft_model: str = typer.Option(None, "--draft-model", help="Draft model path for speculation"),
    draft_k: int = typer.Option(4, "--draft-k", help="Draft tokens per speculation step (default: 4)"),
    max_concurrent: int = typer.Option(16, "--max-concurrent", help="Max concurrent requests"),
    proxy: bool = typer.Option(False, "--proxy", help="Enable S2O admission control proxy"),
):
    """Start the OpenAI-compatible inference server."""
    args = SimpleNamespace(
        model=model, host=host, port=port, ctx_size=ctx_size,
        parallel=parallel, threads=threads, api_key=api_key,
        kv_quant=kv_quant, speculative=speculative, draft_model=draft_model,
        draft_k=draft_k, max_concurrent=max_concurrent, proxy=proxy,
    )
    do_serve(args)
