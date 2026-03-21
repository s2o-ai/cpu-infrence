"""s2o models — List available models."""

from __future__ import annotations

import re
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()
ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models"

_GGUF_RE = re.compile(r"^(.+?)[-_](Q\d+_\w+|F16|F32|BF16)\.gguf$", re.IGNORECASE)


def models():
    """List downloaded models."""
    if not MODELS_DIR.exists():
        console.print("[yellow]No models/ directory found.[/]")
        raise typer.Exit()

    gguf_files = sorted(MODELS_DIR.rglob("*.gguf"))
    if not gguf_files:
        console.print("[yellow]No GGUF models found in models/[/]")
        console.print(f"Download models to: {MODELS_DIR}")
        raise typer.Exit()

    table = Table(title="Downloaded Models")
    table.add_column("Model", style="cyan")
    table.add_column("Quantization", style="green")
    table.add_column("Size", justify="right")
    table.add_column("Path", style="dim")

    total_bytes = 0
    for gguf in gguf_files:
        size = gguf.stat().st_size
        total_bytes += size

        match = _GGUF_RE.match(gguf.name)
        if match:
            name, quant = match.group(1), match.group(2)
        else:
            name, quant = gguf.stem, "?"

        if size >= 1024 ** 3:
            size_str = f"{size / (1024 ** 3):.1f} GB"
        else:
            size_str = f"{size / (1024 ** 2):.0f} MB"

        table.add_row(name, quant, size_str, str(gguf.relative_to(ROOT)))

    console.print(table)
    if total_bytes >= 1024 ** 3:
        console.print(f"\nTotal: {total_bytes / (1024 ** 3):.1f} GB")
    else:
        console.print(f"\nTotal: {total_bytes / (1024 ** 2):.0f} MB")
