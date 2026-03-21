"""s2o run — Interactive chat with a model."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

console = Console()
ROOT = Path(__file__).resolve().parent.parent.parent
BIN_DIR = ROOT / "engine" / "src" / "llama" / "build" / "bin"

if platform.system() == "Windows":
    CLI_BIN = BIN_DIR / "llama-cli.exe"
else:
    CLI_BIN = BIN_DIR / "llama-cli"


def run(
    model: str = typer.Argument(help="Path to GGUF model or name to search in models/"),
    threads: int = typer.Option(None, help="CPU threads (default: auto-detect)"),
):
    """Start interactive chat with a model."""
    if not CLI_BIN.exists():
        console.print(f"[red]ERROR: llama-cli not found at {CLI_BIN}[/]")
        console.print("Run: [bold]python scripts/s2o.py build[/]")
        raise typer.Exit(1)

    # Find model
    SCRIPTS_DIR = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(SCRIPTS_DIR))
    from serve import find_model  # noqa: E402
    model_path = find_model(model)

    # Auto-detect threads
    if threads is None:
        try:
            from engine.detect import detect_cpu
            cpu = detect_cpu()
            threads = cpu.recommendation.threads
            console.print(f"[dim]CPU: {cpu.brand} | Threads: {threads} | {cpu.recommendation.backend}[/]")
        except Exception:
            import os
            threads = os.cpu_count() or 4

    console.print(f"[bold]Model:[/] {model_path.name}")
    console.print(f"[bold]Threads:[/] {threads}")
    console.print("[dim]Type /bye to exit[/]\n")

    cmd = [
        str(CLI_BIN),
        "-m", str(model_path),
        "-t", str(threads),
        "--interactive-first",
        "-r", "User:",
        "--color",
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        console.print("\n[dim]Chat ended.[/]")
