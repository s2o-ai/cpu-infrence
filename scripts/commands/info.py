"""s2o info — CPU feature detection and backend recommendation."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from engine.detect import detect_cpu

console = Console()


def info(
    output_json: bool = typer.Option(False, "--json", help="Output full JSON report"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show all feature flags"),
):
    """Show CPU features and recommended inference configuration."""
    cpu = detect_cpu()

    if output_json:
        console.print_json(json.dumps(cpu.to_dict()))
        return

    # SIMD features
    simd_present = []
    simd_absent = []
    if cpu.arch == "x86_64":
        checks = [
            ("AVX2", cpu.features.avx2), ("AVX-512F", cpu.features.avx512f),
            ("AVX-512BW", cpu.features.avx512bw), ("AVX-512VL", cpu.features.avx512vl),
            ("AVX-VNNI", cpu.features.avx_vnni), ("FMA", cpu.features.fma),
            ("F16C", cpu.features.f16c), ("BMI2", cpu.features.bmi2),
            ("AMX-TILE", cpu.features.amx_tile), ("AMX-INT8", cpu.features.amx_int8),
        ]
        for name, val in checks:
            (simd_present if val else simd_absent).append(name)
    else:
        checks = [
            ("NEON", cpu.features.neon), ("SVE", cpu.features.sve),
            ("SVE2", cpu.features.sve2), ("DOTPROD", cpu.features.dotprod),
            ("I8MM", cpu.features.i8mm), ("BF16", cpu.features.bf16),
        ]
        for name, val in checks:
            (simd_present if val else simd_absent).append(name)

    # Hardware info
    hw_lines = [
        f"[bold]CPU:[/]       {cpu.brand}",
        f"[bold]Vendor:[/]    {cpu.vendor}  (Family {cpu.family}, Model {cpu.model})",
        f"[bold]Cores:[/]     {cpu.cores_physical} physical, {cpu.cores_logical} logical",
        f"[bold]Arch:[/]      {cpu.arch}",
        "",
        f"[bold]SIMD:[/]      [green]{', '.join(simd_present)}[/]" if simd_present else "[bold]SIMD:[/]      none",
        "",
        f"[bold]Cache:[/]     L1d {cpu.cache.l1d_kb}KB | L1i {cpu.cache.l1i_kb}KB | L2 {cpu.cache.l2_kb}KB | L3 {cpu.cache.l3_kb}KB",
        f"[bold]Memory:[/]    {cpu.memory.total_gb} GB total, {cpu.memory.available_gb} GB available",
        f"[bold]NUMA:[/]      {cpu.numa.num_nodes} node{'s' if cpu.numa.num_nodes != 1 else ''}",
    ]
    console.print(Panel("\n".join(hw_lines), title="S2O CPU Detection", border_style="blue"))

    # Recommendation
    r = cpu.recommendation
    rec_lines = [
        f"[bold]Backend:[/]       {r.backend}",
        f"[bold]Reason:[/]        {r.reason}",
        f"[bold]Quantization:[/]  {r.quantization}",
        f"[bold]Threads:[/]       {r.threads} (physical cores)",
        f"[bold]Max model:[/]     {r.max_model_b:.0f}B parameters",
    ]
    if r.numa_strategy:
        rec_lines.append(f"[bold]NUMA:[/]          --numa {r.numa_strategy}")
    console.print(Panel("\n".join(rec_lines), title="Recommendation", border_style="green"))

    # Verbose: all feature flags
    if verbose:
        table = Table(title="All Feature Flags")
        table.add_column("Feature", style="cyan")
        table.add_column("Supported", justify="center")
        from dataclasses import fields
        for f in fields(cpu.features):
            val = getattr(cpu.features, f.name)
            if isinstance(val, bool):
                mark = "[green]Yes[/]" if val else "[dim]No[/]"
                table.add_row(f.name, mark)
            elif val:
                table.add_row(f.name, str(val))
        console.print(table)
