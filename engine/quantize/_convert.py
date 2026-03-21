"""HuggingFace model to GGUF conversion wrapper."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CONVERT_SCRIPT = ROOT / "engine" / "src" / "llama" / "convert_hf_to_gguf.py"
_python_name = "python.exe" if platform.system() == "Windows" else "python"
PYTHON = ROOT / ".venv" / ("Scripts" if platform.system() == "Windows" else "bin") / _python_name


def is_hf_model_id(model_id_or_path: str) -> bool:
    """Check if input looks like a HuggingFace model ID (e.g. 'Qwen/Qwen3-0.6B').

    HF IDs have exactly one slash with no file extensions and no path separators.
    """
    if Path(model_id_or_path).exists():
        return False
    parts = model_id_or_path.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return False
    # File paths typically have extensions like .gguf, .bin, etc.
    if "." in parts[1] and parts[1].rsplit(".", 1)[1] in ("gguf", "bin", "safetensors", "pt", "pth"):
        return False
    return True


def model_name_from_id(model_id: str) -> str:
    """Extract model name from HF model ID. e.g. 'Qwen/Qwen3-0.6B' -> 'Qwen3-0.6B'."""
    return model_id.split("/")[-1]


def convert_to_gguf(
    model_id_or_path: str,
    output_dir: Path,
    output_type: str = "f16",
    remote: bool = True,
) -> Path:
    """Convert a HuggingFace model to GGUF format.

    Args:
        model_id_or_path: HuggingFace model ID (e.g. 'Qwen/Qwen3-0.6B') or local path.
        output_dir: Directory to write the GGUF file.
        output_type: Output type (f16, f32, bf16, q8_0, auto).
        remote: If True and input is HF ID, stream weights via HTTP (avoids full download).

    Returns:
        Path to the generated GGUF file.
    """
    if not CONVERT_SCRIPT.exists():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at {CONVERT_SCRIPT}\n"
            "Ensure the llama.cpp submodule is initialized: git submodule update --init"
        )

    python = str(PYTHON) if PYTHON.exists() else sys.executable

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename
    if is_hf_model_id(model_id_or_path):
        name = model_name_from_id(model_id_or_path)
    else:
        name = Path(model_id_or_path).name

    outfile = output_dir / f"{name}-F16.gguf"

    # For HF model IDs without --remote, download the model first
    model_input = model_id_or_path
    if is_hf_model_id(model_id_or_path) and not remote:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(repo_id=model_id_or_path)
        model_input = local_dir

    cmd = [
        python,
        str(CONVERT_SCRIPT),
        model_input,
        "--outfile", str(outfile),
        "--outtype", output_type,
    ]

    if remote and is_hf_model_id(model_id_or_path):
        cmd.append("--remote")

    env = os.environ.copy()
    # Pass through HF_TOKEN if set
    if "HF_TOKEN" in os.environ:
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour timeout for large models
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"GGUF conversion failed (exit {result.returncode}):\n"
            f"STDOUT: {result.stdout[-2000:] if result.stdout else '(empty)'}\n"
            f"STDERR: {result.stderr[-2000:] if result.stderr else '(empty)'}"
        )

    if not outfile.exists():
        raise FileNotFoundError(
            f"Conversion completed but output file not found: {outfile}\n"
            f"STDOUT: {result.stdout[-1000:] if result.stdout else '(empty)'}"
        )

    return outfile
