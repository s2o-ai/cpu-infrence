#!/usr/bin/env python3
"""Benchmark speculative decoding throughput improvement.

Compares baseline (no draft) vs speculative decoding with a draft model.

Usage:
    python benchmarks/bench_speculative.py \
        --model models/Qwen2.5-7B-Q4_K_M.gguf \
        --draft-model models/Qwen2.5-0.5B-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_BIN = ROOT / "engine" / "src" / "llama" / "build" / "bin"

if platform.system() == "Windows":
    BENCH_BIN = DEFAULT_BIN / "llama-bench.exe"
else:
    BENCH_BIN = DEFAULT_BIN / "llama-bench"


@dataclass
class SpecBenchResult:
    label: str
    prompt_tok_s: float = 0.0
    gen_tok_s: float = 0.0
    note: str = ""


def run_bench(model: str, threads: int, draft_model: str | None = None,
              draft_max: int = 8) -> SpecBenchResult:
    """Run llama-bench with optional draft model."""
    if not BENCH_BIN.exists():
        print(f"ERROR: llama-bench not found at {BENCH_BIN}")
        sys.exit(1)

    label = f"speculative (draft-max={draft_max})" if draft_model else "baseline"
    cmd = [
        str(BENCH_BIN),
        "-m", model,
        "-t", str(threads),
        "-o", "json",
    ]

    if draft_model:
        cmd.extend(["-md", draft_model, "-ndr", str(draft_max)])

    result = SpecBenchResult(label=label)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
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


def _extract(entry: dict, result: SpecBenchResult) -> None:
    test_type = entry.get("test", "")
    tok_s = entry.get("avg_ts", 0.0) or entry.get("t_s", 0.0)
    if "pp" in test_type:
        result.prompt_tok_s = max(result.prompt_tok_s, tok_s)
    elif "tg" in test_type:
        result.gen_tok_s = max(result.gen_tok_s, tok_s)


def print_comparison(baseline: SpecBenchResult, speculative: SpecBenchResult) -> None:
    """Print A/B comparison."""
    print(f"\n{'Variant':<35} {'Prompt (tok/s)':>15} {'Gen (tok/s)':>12} {'Notes'}")
    print("-" * 75)
    for r in [baseline, speculative]:
        note = r.note or ""
        print(f"{r.label:<35} {r.prompt_tok_s:>15.1f} {r.gen_tok_s:>12.2f} {note}")

    if baseline.gen_tok_s > 0 and speculative.gen_tok_s > 0:
        speedup = speculative.gen_tok_s / baseline.gen_tok_s
        print(f"\nSpeculative speedup: {speedup:.2f}x generation throughput")
        if speedup < 1.0:
            print("  NOTE: Speculation overhead exceeds benefit — consider a better draft model")


def main():
    parser = argparse.ArgumentParser(description="Benchmark speculative decoding")
    parser.add_argument("--model", required=True, help="Path to main GGUF model")
    parser.add_argument("--draft-model", required=True, help="Path to draft GGUF model")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--draft-max", type=int, default=8, help="Max draft tokens (default: 8)")
    args = parser.parse_args()

    print(f"Main model:  {args.model}")
    print(f"Draft model: {args.draft_model}")
    print(f"Threads:     {args.threads}")
    print(f"Draft max:   {args.draft_max}")
    print()

    print("Running baseline (no speculation)...")
    baseline = run_bench(args.model, args.threads)

    print("Running with speculative decoding...")
    speculative = run_bench(args.model, args.threads, args.draft_model, args.draft_max)

    print_comparison(baseline, speculative)


if __name__ == "__main__":
    main()
