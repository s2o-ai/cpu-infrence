"""GGUF quantization wrapper — wraps llama-quantize.exe."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BIN_DIR = ROOT / "engine" / "src" / "llama" / "build" / "bin"

if platform.system() == "Windows":
    QUANTIZE_BIN = BIN_DIR / "llama-quantize.exe"
else:
    QUANTIZE_BIN = BIN_DIR / "llama-quantize"


def _get_env() -> dict:
    """Get environment with MSYS2 in PATH for DLL resolution."""
    env = os.environ.copy()
    msys2_bin = "C:/msys64/mingw64/bin"
    if os.path.isdir(msys2_bin) and msys2_bin not in env.get("PATH", ""):
        env["PATH"] = msys2_bin + os.pathsep + env.get("PATH", "")
    return env


def quantize_gguf(
    input_gguf: Path,
    output_gguf: Path,
    quant_type: str = "Q4_K_M",
    threads: int | None = None,
) -> Path:
    """Quantize a GGUF model file.

    Args:
        input_gguf: Path to input GGUF file (typically FP16).
        output_gguf: Path to write quantized output.
        quant_type: Quantization type (Q4_K_M, Q8_0, etc.).
        threads: CPU threads for quantization.

    Returns:
        Path to the quantized GGUF file.
    """
    if not QUANTIZE_BIN.exists():
        raise FileNotFoundError(
            f"llama-quantize not found at {QUANTIZE_BIN}\n"
            "Run: python scripts/s2o.py build"
        )

    if not input_gguf.exists():
        raise FileNotFoundError(f"Input GGUF not found: {input_gguf}")

    output_gguf.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(QUANTIZE_BIN),
        str(input_gguf),
        str(output_gguf),
        quant_type,
    ]

    if threads:
        cmd.append(str(threads))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,
        env=_get_env(),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Quantization failed (exit {result.returncode}):\n"
            f"STDERR: {result.stderr[-2000:] if result.stderr else '(empty)'}"
        )

    if not output_gguf.exists():
        raise FileNotFoundError(
            f"Quantization completed but output not found: {output_gguf}"
        )

    return output_gguf
