"""Perplexity validation wrapper — wraps llama-perplexity.exe."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BIN_DIR = ROOT / "engine" / "src" / "llama" / "build" / "bin"
DATA_DIR = Path(__file__).resolve().parent / "data"

if platform.system() == "Windows":
    PERPLEXITY_BIN = BIN_DIR / "llama-perplexity.exe"
else:
    PERPLEXITY_BIN = BIN_DIR / "llama-perplexity"


def _get_env() -> dict:
    """Get environment with MSYS2 in PATH for DLL resolution."""
    env = os.environ.copy()
    msys2_bin = "C:/msys64/mingw64/bin"
    if os.path.isdir(msys2_bin) and msys2_bin not in env.get("PATH", ""):
        env["PATH"] = msys2_bin + os.pathsep + env.get("PATH", "")
    return env


def compute_perplexity(
    model_path: Path,
    test_file: Path | None = None,
    ctx_size: int = 512,
    threads: int | None = None,
) -> float:
    """Compute perplexity of a model on a test file.

    Args:
        model_path: Path to GGUF model file.
        test_file: Path to test text file. Defaults to bundled WikiText-2 sample.
        ctx_size: Context size for evaluation.
        threads: CPU threads.

    Returns:
        Perplexity value (lower is better).
    """
    if not PERPLEXITY_BIN.exists():
        raise FileNotFoundError(
            f"llama-perplexity not found at {PERPLEXITY_BIN}\n"
            "Run: python scripts/s2o.py build"
        )

    if test_file is None:
        test_file = DATA_DIR / "wikitext2_test_sample.txt"

    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    if threads is None:
        threads = os.cpu_count() or 4

    cmd = [
        str(PERPLEXITY_BIN),
        "-m", str(model_path),
        "-f", str(test_file),
        "--ctx-size", str(ctx_size),
        "-t", str(threads),
        "-ngl", "0",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 minutes
        env=_get_env(),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Perplexity computation failed (exit {result.returncode}):\n"
            f"STDERR: {result.stderr[-2000:] if result.stderr else '(empty)'}"
        )

    # Parse perplexity from output — llama-perplexity outputs lines like:
    # "Final estimate: PPL = 12.3456 +/- 0.1234"
    output = result.stdout + result.stderr
    match = re.search(r"Final estimate: PPL\s*=\s*([\d.]+)", output)
    if match:
        return float(match.group(1))

    # Fallback: look for last "PPL" value
    matches = re.findall(r"PPL\s*=?\s*([\d.]+)", output)
    if matches:
        return float(matches[-1])

    raise ValueError(
        f"Could not parse perplexity from output:\n{output[-1000:]}"
    )
