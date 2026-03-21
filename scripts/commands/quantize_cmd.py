"""s2o quantize — Convert and quantize HuggingFace models to GGUF."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()
ROOT = Path(__file__).resolve().parent.parent.parent


def quantize(
    model: str = typer.Argument(help="HuggingFace model ID (e.g. 'Qwen/Qwen3-0.6B') or local path"),
    types: str = typer.Option("Q4_K_M,Q8_0", help="Quantization types (comma-separated)"),
    output: str = typer.Option(None, help="Output directory (default: models/{model_name}/)"),
    validate: bool = typer.Option(False, help="Run perplexity validation (slow)"),
    keep_fp16: bool = typer.Option(False, "--keep-fp16", help="Keep intermediate FP16 GGUF"),
    no_remote: bool = typer.Option(False, "--no-remote", help="Download full weights instead of streaming"),
    threads: int = typer.Option(None, help="CPU threads (default: auto-detect)"),
):
    """Convert and quantize a model to optimized GGUF format."""
    sys.path.insert(0, str(ROOT))

    quant_types = [t.strip() for t in types.split(",")]
    output_dir = Path(output) if output else None

    console.print(f"\n[bold]Model:[/]  {model}")
    console.print(f"[bold]Types:[/]  {', '.join(quant_types)}")
    if output_dir:
        console.print(f"[bold]Output:[/] {output_dir}")
    console.print()

    def on_status(stage: str, detail: str):
        icons = {
            "convert": "[blue]Converting[/]",
            "quantize": "[yellow]Quantizing[/]",
            "validate": "[cyan]Validating[/]",
            "report": "[green]Reporting[/]",
            "cleanup": "[dim]Cleanup[/]",
            "done": "[bold green]Done[/]",
        }
        prefix = icons.get(stage, f"[dim]{stage}[/]")
        console.print(f"  {prefix}  {detail}")

    try:
        from engine.quantize import quantize_model

        result = quantize_model(
            model_id_or_path=model,
            output_dir=output_dir,
            quant_types=quant_types,
            validate=validate,
            keep_fp16=keep_fp16,
            remote=not no_remote,
            threads=threads,
            on_status=on_status,
        )

        # Display results table
        console.print()
        table = Table(title=f"Quantization Results: {result.model_name}")
        table.add_column("Type", style="bold")
        table.add_column("Size", justify="right")
        table.add_column("Compression", justify="right")
        table.add_column("Perplexity", justify="right")
        table.add_column("Path")

        for v in result.variants:
            ppl = f"{v.perplexity:.2f}" if v.perplexity is not None else "—"
            table.add_row(
                v.quant_type,
                f"{v.size_mb:.1f} MB",
                f"{v.compression_ratio:.1f}x",
                ppl,
                v.output_path,
            )

        console.print(table)
        console.print(f"\n[green]Reports saved to output directory.[/]")

    except FileNotFoundError as e:
        console.print(f"\n[red]ERROR:[/] {e}")
        raise typer.Exit(1)
    except RuntimeError as e:
        console.print(f"\n[red]ERROR:[/] {e}")
        raise typer.Exit(1)
    except ImportError as e:
        console.print(f"\n[red]Missing dependencies:[/] {e}")
        console.print("\nInstall with:")
        console.print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        console.print("  pip install transformers numpy huggingface_hub safetensors sentencepiece protobuf")
        console.print("  pip install -e engine/src/llama/gguf-py/")
        raise typer.Exit(1)
