"""Auto-quantization pipeline — convert HuggingFace models to optimized GGUF."""

from __future__ import annotations

import os
from pathlib import Path

from ._convert import convert_to_gguf, is_hf_model_id, model_name_from_id
from ._quantize import quantize_gguf
from ._validate import compute_perplexity
from ._report import QuantResult, QuantVariant, save_report

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models"


def quantize_model(
    model_id_or_path: str,
    output_dir: str | Path | None = None,
    quant_types: list[str] | None = None,
    validate: bool = False,
    keep_fp16: bool = False,
    remote: bool = True,
    threads: int | None = None,
    on_status: callable | None = None,
) -> QuantResult:
    """Run the full quantization pipeline.

    Args:
        model_id_or_path: HuggingFace model ID or local path to safetensors directory.
        output_dir: Output directory. Defaults to models/{model_name}/.
        quant_types: List of quantization types. Defaults to ["Q4_K_M", "Q8_0"].
        validate: Run perplexity validation on each variant.
        keep_fp16: Keep intermediate FP16 GGUF file.
        remote: Stream weights from HuggingFace (avoids full download).
        threads: CPU threads for quantization and validation.
        on_status: Optional callback(stage: str, detail: str) for progress updates.

    Returns:
        QuantResult with all variant information.
    """
    if quant_types is None:
        quant_types = ["Q4_K_M", "Q8_0"]

    if threads is None:
        threads = os.cpu_count() or 4

    # Determine model name and output directory
    if is_hf_model_id(model_id_or_path):
        model_name = model_name_from_id(model_id_or_path)
    else:
        model_name = Path(model_id_or_path).name

    if output_dir is None:
        output_dir = MODELS_DIR / model_name
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get CPU info for report
    cpu_info = ""
    try:
        from engine.detect import detect_cpu
        cpu = detect_cpu()
        cpu_info = cpu.brand
    except Exception:
        pass

    result = QuantResult(
        model_id=model_id_or_path,
        model_name=model_name,
        original_size_mb=0.0,
        cpu_info=cpu_info,
    )

    # Step 1: Convert to FP16 GGUF (skip if already exists)
    fp16_path = output_dir / f"{model_name}-F16.gguf"
    if fp16_path.exists():
        _status(on_status, "convert", f"FP16 GGUF already exists: {fp16_path.name}")
    else:
        _status(on_status, "convert", f"Converting {model_id_or_path} to FP16 GGUF...")
        fp16_path = convert_to_gguf(
            model_id_or_path,
            output_dir=output_dir,
            output_type="f16",
            remote=remote,
        )
    result.original_size_mb = fp16_path.stat().st_size / (1024 * 1024)
    _status(on_status, "convert", f"FP16 GGUF: {result.original_size_mb:.1f} MB")

    # Step 2: Quantize to each requested type
    for qtype in quant_types:
        _status(on_status, "quantize", f"Quantizing to {qtype}...")
        out_name = f"{model_name}-{qtype}.gguf"
        out_path = output_dir / out_name

        quantize_gguf(
            input_gguf=fp16_path,
            output_gguf=out_path,
            quant_type=qtype,
            threads=threads,
        )

        size_mb = out_path.stat().st_size / (1024 * 1024)
        ratio = result.original_size_mb / size_mb if size_mb > 0 else 0

        variant = QuantVariant(
            quant_type=qtype,
            output_path=str(out_path.relative_to(ROOT)),
            size_mb=round(size_mb, 1),
            compression_ratio=round(ratio, 1),
        )

        # Step 3: Optional perplexity validation
        if validate:
            _status(on_status, "validate", f"Computing perplexity for {qtype}...")
            try:
                ppl = compute_perplexity(out_path, threads=threads)
                variant.perplexity = round(ppl, 4)
                _status(on_status, "validate", f"{qtype} perplexity: {ppl:.4f}")
            except Exception as e:
                _status(on_status, "validate", f"Perplexity failed for {qtype}: {e}")

        result.variants.append(variant)

    # Step 4: Generate reports
    _status(on_status, "report", "Generating reports...")
    save_report(result, output_dir)

    # Step 5: Cleanup FP16 intermediate
    if not keep_fp16 and fp16_path.exists():
        _status(on_status, "cleanup", "Removing FP16 intermediate...")
        fp16_path.unlink()

    _status(on_status, "done", "Quantization complete!")
    return result


def _status(callback: callable | None, stage: str, detail: str):
    """Call status callback if provided."""
    if callback:
        callback(stage, detail)
