#!/usr/bin/env python3
"""Benchmark KV cache quantization impact on throughput and memory.

Runs llama-bench with different --cache-type-k/v settings and compares results.

Usage:
    python benchmarks/bench_kv.py --model models/Qwen2.5-7B-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_BIN = ROOT / "engine" / "src" / "llama" / "build" / "bin"

if platform.system() == "Windows":
    BENCH_BIN = DEFAULT_BIN / "llama-bench.exe"
else:
    BENCH_BIN = DEFAULT_BIN / "llama-bench"


@dataclass
class KVBenchResult:
    kv_type: str
    prompt_tok_s: float = 0.0
    gen_tok_s: float = 0.0
    note: str = ""


KV_TYPES = ["f16", "q8_0", "q4_0"]


def run_bench(model: str, kv_type: str, threads: int | None = None) -> KVBenchResult:
    """Run llama-bench with a specific KV cache type."""
    if not BENCH_BIN.exists():
        print(f"ERROR: llama-bench not found at {BENCH_BIN}")
        sys.exit(1)

    t = threads or os.cpu_count() or 4
    cmd = [
        str(BENCH_BIN),
        "-m", model,
        "-t", str(t),
        "-ctk", kv_type,
        "-ctv", kv_type,
        "-o", "json",
    ]

    result = KVBenchResult(kv_type=kv_type)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            result.note = f"bench failed: {proc.stderr[:200]}"
            return result

        for line in proc.stdout.strip().split("\n"):
            line = line.strip()
            if not line.startswith("{") and not line.startswith("["):
                continue
            try:
                data = json.loads(line)
                if isinstance(data, list):
                    for entry in data:
                        _extract(entry, result)
                elif isinstance(data, dict):
                    _extract(data, result)
            except json.JSONDecodeError:
                continue

    except subprocess.TimeoutExpired:
        result.note = "timeout"
    except Exception as e:
        result.note = str(e)

    return result


def _extract(entry: dict, result: KVBenchResult) -> None:
    """Extract tok/s from a llama-bench JSON entry."""
    test_type = entry.get("test", "")
    tok_s = entry.get("avg_ts", 0.0) or entry.get("t_s", 0.0)
    if "pp" in test_type:
        result.prompt_tok_s = max(result.prompt_tok_s, tok_s)
    elif "tg" in test_type:
        result.gen_tok_s = max(result.gen_tok_s, tok_s)


def print_table(results: list[KVBenchResult]) -> None:
    """Print comparison table."""
    print(f"\n{'KV Type':<10} {'Prompt (tok/s)':>15} {'Gen (tok/s)':>12} {'Notes'}")
    print("-" * 55)
    for r in results:
        note = r.note or ""
        print(f"{r.kv_type:<10} {r.prompt_tok_s:>15.1f} {r.gen_tok_s:>12.2f} {note}")

    if len(results) >= 2 and results[0].gen_tok_s > 0:
        baseline = results[0]
        print(f"\nBaseline: {baseline.kv_type}")
        for r in results[1:]:
            if r.gen_tok_s > 0:
                speedup = r.gen_tok_s / baseline.gen_tok_s
                print(f"  {r.kv_type} vs {baseline.kv_type}: {speedup:.2f}x gen throughput")


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV cache quantization")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads")
    parser.add_argument("--types", nargs="+", default=KV_TYPES,
                        help=f"KV cache types to test (default: {KV_TYPES})")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"KV types: {args.types}")
    print()

    results = []
    for kv_type in args.types:
        print(f"Benchmarking KV type: {kv_type}...")
        r = run_bench(args.model, kv_type, args.threads)
        results.append(r)

    print_table(results)


if __name__ == "__main__":
    main()
