"""Benchmark runner — wraps llama-bench.exe."""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .bench_types import BenchResult, SingleBenchResult

ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT / "engine" / "src" / "llama" / "build" / "bin"

if platform.system() == "Windows":
    BENCH_BIN = BIN_DIR / "llama-bench.exe"
else:
    BENCH_BIN = BIN_DIR / "llama-bench"


def _get_env() -> dict:
    """Get environment with MSYS2 in PATH for DLL resolution."""
    env = os.environ.copy()
    msys2_bin = "C:/msys64/mingw64/bin"
    if os.path.isdir(msys2_bin) and msys2_bin not in env.get("PATH", ""):
        env["PATH"] = msys2_bin + os.pathsep + env.get("PATH", "")
    return env


def _compute_stats(samples: list[float], discard_first: bool = True) -> tuple[float, float, float]:
    """Compute avg, stddev, median after optional warmup discard."""
    if not samples:
        return 0.0, 0.0, 0.0
    data = samples[1:] if (discard_first and len(samples) > 1) else samples
    avg = statistics.mean(data)
    stddev = statistics.stdev(data) if len(data) > 1 else 0.0
    median = statistics.median(data)
    return avg, stddev, median


def run_llama_bench(
    model_path: str,
    runs: int = 5,
    prompt_tokens: int = 512,
    gen_tokens: int = 128,
    threads: int | None = None,
) -> BenchResult:
    """Run llama-bench and return parsed results."""
    if not BENCH_BIN.exists():
        raise FileNotFoundError(f"llama-bench not found at {BENCH_BIN}")

    model = Path(model_path)
    if not model.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if threads is None:
        threads = os.cpu_count() or 4

    cmd = [
        str(BENCH_BIN),
        "-m", str(model),
        "-p", str(prompt_tokens),
        "-n", str(gen_tokens),
        "-r", str(runs),
        "-t", str(threads),
        "-ngl", "0",
        "-o", "json",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, env=_get_env(),
    )

    if result.returncode != 0:
        raise RuntimeError(f"llama-bench failed:\n{result.stderr}")

    return _parse_json_output(result.stdout, model_path, threads)


def _parse_json_output(output: str, model_path: str, threads: int) -> BenchResult:
    """Parse llama-bench JSON output."""
    # llama-bench outputs a JSON array
    # Find the JSON array in the output (may have other text before/after)
    start = output.find("[")
    end = output.rfind("]") + 1
    if start < 0 or end <= 0:
        raise ValueError(f"No JSON array found in llama-bench output:\n{output[:500]}")

    entries = json.loads(output[start:end])
    if not entries:
        raise ValueError("Empty results from llama-bench")

    bench = BenchResult(
        model_path=model_path,
        timestamp=datetime.now(timezone.utc).isoformat(),
        threads=threads,
    )

    # Extract model info from first entry
    first = entries[0]
    bench.model_type = first.get("model_type", "")
    bench.model_size_gb = first.get("model_size", 0) / (1024 ** 3) if first.get("model_size") else 0
    bench.model_params_b = first.get("model_n_params", 0) / 1e9 if first.get("model_n_params") else 0
    bench.build_commit = first.get("build_commit", "")
    bench.cpu_info = first.get("cpu_info", "")

    # Separate pp (prompt processing) and tg (text generation) results
    pp_samples = []
    tg_samples = []
    pp_tokens = 0
    tg_tokens = 0

    for entry in entries:
        tok_per_sec = entry.get("avg_ts", 0)
        n_prompt = entry.get("n_prompt", 0)
        n_gen = entry.get("n_gen", 0)

        if n_prompt > 0 and n_gen == 0:
            pp_samples.append(tok_per_sec)
            pp_tokens = n_prompt
        elif n_gen > 0:
            tg_samples.append(tok_per_sec)
            tg_tokens = n_gen

    # Compute statistics (discard first sample as warmup)
    if pp_samples:
        avg, stddev, median = _compute_stats(pp_samples)
        bench.prompt_processing = SingleBenchResult(
            test_type="pp",
            tokens=pp_tokens,
            avg_tok_per_sec=round(avg, 2),
            stddev_tok_per_sec=round(stddev, 2),
            samples=pp_samples,
            median_tok_per_sec=round(median, 2),
        )

    if tg_samples:
        avg, stddev, median = _compute_stats(tg_samples)
        bench.text_generation = SingleBenchResult(
            test_type="tg",
            tokens=tg_tokens,
            avg_tok_per_sec=round(avg, 2),
            stddev_tok_per_sec=round(stddev, 2),
            samples=tg_samples,
            median_tok_per_sec=round(median, 2),
        )

    return bench
