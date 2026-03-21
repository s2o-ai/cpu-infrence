#!/usr/bin/env python3
"""CLI tool for CPU feature detection and backend recommendation."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.detect import detect_cpu


def format_features(info) -> str:
    """Format detected SIMD features as a compact string."""
    feat = info.features
    names = []

    if info.arch == "x86_64":
        checks = [
            ("AVX2", feat.avx2), ("AVX-512F", feat.avx512f),
            ("AVX-512BW", feat.avx512bw), ("AVX-512VL", feat.avx512vl),
            ("AVX-512-VNNI", feat.avx512_vnni), ("AVX-512-BF16", feat.avx512_bf16),
            ("AVX-VNNI", feat.avx_vnni), ("FMA", feat.fma), ("F16C", feat.f16c),
            ("BMI2", feat.bmi2), ("AMX-TILE", feat.amx_tile),
            ("AMX-INT8", feat.amx_int8), ("AMX-BF16", feat.amx_bf16),
        ]
    else:
        checks = [
            ("NEON", feat.neon), ("SVE", feat.sve), ("SVE2", feat.sve2),
            ("DOTPROD", feat.dotprod), ("I8MM", feat.i8mm),
            ("BF16", feat.bf16), ("FP16", feat.fp16), ("SME", feat.sme),
        ]

    present = [name for name, val in checks if val]
    absent_important = []

    if info.arch == "x86_64":
        if not feat.avx512f:
            absent_important.append("AVX-512")
        if not feat.amx_tile:
            absent_important.append("AMX")
    elif info.arch == "aarch64":
        if not feat.sve:
            absent_important.append("SVE")

    line = ", ".join(present)
    if absent_important:
        line += f"\n           (no {', no '.join(absent_important)})"
    return line


def print_human(info):
    """Print human-readable detection report."""
    print("S2O CPU Detection")
    print("=" * 40)
    print(f"CPU:       {info.brand}")
    print(f"Vendor:    {info.vendor}  (Family {info.family}, Model {info.model})")
    print(f"Cores:     {info.cores_physical} physical, {info.cores_logical} logical")
    print(f"Arch:      {info.arch}")
    print()

    print(f"SIMD:      {format_features(info)}")
    print()

    c = info.cache
    print(f"Cache:     L1d {c.l1d_kb}KB | L1i {c.l1i_kb}KB | L2 {c.l2_kb}KB | L3 {c.l3_kb}KB")
    print(f"Memory:    {info.memory.total_gb} GB total, {info.memory.available_gb} GB available")
    print(f"NUMA:      {info.numa.num_nodes} node{'s' if info.numa.num_nodes != 1 else ''}")
    print()

    r = info.recommendation
    print("Recommendation")
    print("-" * 40)
    print(f"Backend:       {r.backend} ({r.reason})")
    print(f"Quantization:  {r.quantization}")
    print(f"Threads:       {r.threads} (physical cores)")
    print(f"Max model:     {r.max_model_b:.0f}B parameters")
    if r.numa_strategy:
        print(f"NUMA:          --numa {r.numa_strategy}")


def print_verbose(info):
    """Print verbose detection report with all features."""
    print_human(info)
    print()
    print("All Features")
    print("-" * 40)
    feat = info.features
    from dataclasses import fields
    for f in fields(feat):
        val = getattr(feat, f.name)
        if isinstance(val, bool):
            mark = "+" if val else "-"
            print(f"  {mark} {f.name}")
        elif val:
            print(f"    {f.name} = {val}")


def main():
    parser = argparse.ArgumentParser(description="S2O CPU feature detection")
    parser.add_argument("--json", action="store_true", help="Output full JSON report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all feature flags")
    args = parser.parse_args()

    info = detect_cpu()

    if args.json:
        print(json.dumps(info.to_dict(), indent=2))
    elif args.verbose:
        print_verbose(info)
    else:
        print_human(info)


if __name__ == "__main__":
    main()
