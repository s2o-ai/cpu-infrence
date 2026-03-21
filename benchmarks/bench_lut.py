"""A/B benchmark: stock llama.cpp vs S2O LUT kernels.

Runs llama-bench twice (once with stock build, once with LUT build)
and produces a comparison table showing speedup ratios.

Usage:
    python -m benchmarks.bench_lut --model models/Qwen3-0.6B/Qwen3-0.6B-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLAMA_DIR = ROOT / "engine" / "src" / "llama"
BIN_DIR = LLAMA_DIR / "build" / "bin"


@dataclass
class BenchResult:
    label: str
    pp_tok_s: float  # prompt processing tok/s
    tg_tok_s: float  # text generation tok/s


def find_bench_binary() -> Path:
    """Find llama-bench binary."""
    for name in ("llama-bench.exe", "llama-bench"):
        p = BIN_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(f"llama-bench not found in {BIN_DIR}")


def run_bench(model: str, runs: int = 3, threads: int | None = None,
              pp_tokens: int = 512, tg_tokens: int = 128) -> dict:
    """Run llama-bench and return parsed JSON results."""
    bench = find_bench_binary()
    cmd = [
        str(bench),
        "-m", model,
        "-n", str(tg_tokens),
        "-p", str(pp_tokens),
        "-r", str(runs),
        "-ngl", "0",
        "-o", "json",
    ]
    if threads:
        cmd.extend(["-t", str(threads)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"llama-bench failed:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError("llama-bench failed")

    return json.loads(result.stdout)


def extract_metrics(data: list[dict]) -> tuple[float, float]:
    """Extract pp and tg tok/s from llama-bench JSON output."""
    pp_tok_s = 0.0
    tg_tok_s = 0.0
    for entry in data:
        test_type = entry.get("test") or entry.get("type", "")
        avg = entry.get("avg_ts", 0.0)
        if "pp" in test_type:
            pp_tok_s = avg
        elif "tg" in test_type:
            tg_tok_s = avg
    return pp_tok_s, tg_tok_s


def print_comparison(stock: BenchResult, lut: BenchResult):
    """Print a comparison table."""
    pp_speedup = lut.pp_tok_s / stock.pp_tok_s if stock.pp_tok_s > 0 else 0.0
    tg_speedup = lut.tg_tok_s / stock.tg_tok_s if stock.tg_tok_s > 0 else 0.0

    print()
    print("=" * 70)
    print("S2O LUT Kernel Benchmark Comparison")
    print("=" * 70)
    print(f"{'Metric':<30} {'Stock':>12} {'LUT':>12} {'Speedup':>10}")
    print("-" * 70)
    print(f"{'Prompt Processing (tok/s)':<30} {stock.pp_tok_s:>12.2f} {lut.pp_tok_s:>12.2f} {pp_speedup:>9.2f}x")
    print(f"{'Text Generation (tok/s)':<30} {stock.tg_tok_s:>12.2f} {lut.tg_tok_s:>12.2f} {tg_speedup:>9.2f}x")
    print("=" * 70)
    print()

    if pp_speedup >= 1.3:
        print("PP speedup >= 1.3x -- ON TRACK")
    elif pp_speedup >= 1.1:
        print("PP speedup 1.1-1.3x -- needs optimization")
    else:
        print("PP speedup < 1.1x -- investigate memory bandwidth / tiling")


def main():
    parser = argparse.ArgumentParser(description="A/B benchmark: stock vs LUT")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--threads", type=int, default=None, help="Thread count")
    parser.add_argument("--pp-tokens", type=int, default=512)
    parser.add_argument("--tg-tokens", type=int, default=128)
    args = parser.parse_args()

    print("Running stock benchmark...")
    # Note: Both stock and LUT use the same binary when compiled with GGML_S2O_LUT=ON.
    # The LUT path is taken automatically when the buffer type is selected.
    # For a true A/B comparison, build twice: once without --lut, once with.
    # This script assumes a single build and measures whatever is active.

    data = run_bench(args.model, runs=args.runs, threads=args.threads,
                     pp_tokens=args.pp_tokens, tg_tokens=args.tg_tokens)

    pp, tg = extract_metrics(data)
    result = BenchResult(label="current", pp_tok_s=pp, tg_tok_s=tg)

    print(f"\nResults: PP={result.pp_tok_s:.2f} tok/s  TG={result.tg_tok_s:.2f} tok/s")

    # For a proper A/B comparison, the user should run this script twice
    # with different builds and compare the output manually or via bench_report.


if __name__ == "__main__":
    main()
