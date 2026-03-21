"""Quantization report generation — JSON and markdown."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class QuantVariant:
    quant_type: str
    output_path: str
    size_mb: float
    compression_ratio: float  # original_size / quantized_size
    perplexity: float | None = None  # None if validation skipped


@dataclass
class QuantResult:
    model_id: str
    model_name: str
    original_size_mb: float
    variants: list[QuantVariant] = field(default_factory=list)
    cpu_info: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def generate_json(result: QuantResult) -> str:
    """Generate JSON report."""
    return json.dumps(asdict(result), indent=2)


def generate_markdown(result: QuantResult) -> str:
    """Generate markdown report."""
    lines = [
        f"# Quantization Report: {result.model_name}",
        "",
        f"**Model:** {result.model_id}",
        f"**Date:** {result.timestamp}",
        f"**CPU:** {result.cpu_info}",
        f"**Original Size:** {result.original_size_mb:.1f} MB",
        "",
        "## Variants",
        "",
        "| Type | Size (MB) | Compression | Perplexity |",
        "|------|-----------|-------------|------------|",
    ]

    for v in result.variants:
        ppl = f"{v.perplexity:.2f}" if v.perplexity is not None else "—"
        lines.append(
            f"| {v.quant_type} | {v.size_mb:.1f} | {v.compression_ratio:.1f}x | {ppl} |"
        )

    lines.extend([
        "",
        "## Files",
        "",
    ])

    for v in result.variants:
        lines.append(f"- `{v.output_path}` ({v.size_mb:.1f} MB)")

    lines.append("")
    return "\n".join(lines)


def save_report(result: QuantResult, output_dir: Path) -> tuple[Path, Path]:
    """Save both JSON and markdown reports. Returns (json_path, md_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{result.model_name}_quant_report.json"
    md_path = output_dir / f"{result.model_name}_quant_report.md"

    json_path.write_text(generate_json(result), encoding="utf-8")
    md_path.write_text(generate_markdown(result), encoding="utf-8")

    return json_path, md_path
