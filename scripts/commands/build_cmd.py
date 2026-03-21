"""s2o build — Build the inference engine."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console

console = Console()

# Import build function from scripts/build.py
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
from build import build as do_build  # noqa: E402


def build(
    clean: bool = typer.Option(False, "--clean", help="Clean build directory first"),
    lut: bool = typer.Option(False, "--lut", help="Enable S2O LUT-based INT4 kernels"),
    openvino: bool = typer.Option(False, "--openvino", help="Enable Intel OpenVINO backend (iGPU/Arc/NPU)"),
):
    """Build llama-server and related binaries."""
    console.print("[bold blue]Building inference engine...[/]")
    if lut:
        console.print("[bold yellow]S2O LUT kernels enabled[/]")
    if openvino:
        console.print("[bold yellow]OpenVINO backend enabled[/]")
    do_build(clean=clean, lut=lut, openvino=openvino)
    console.print("[bold green]Build complete.[/]")
