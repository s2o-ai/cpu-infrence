"""CPU cache size detection."""

from __future__ import annotations

import platform
from pathlib import Path

from ._types import CacheInfo


def _detect_x86_cpuid() -> CacheInfo:
    """Detect cache sizes via CPUID leaf 4 (deterministic cache parameters)."""
    from ._cpuid_x86 import CpuidExecutor

    info = CacheInfo()
    cpuid = CpuidExecutor()
    try:
        for subleaf in range(32):  # max reasonable cache entries
            eax, ebx, ecx, _ = cpuid(4, subleaf)
            cache_type = eax & 0x1F
            if cache_type == 0:
                break

            level = (eax >> 5) & 0x7
            line_size = (ebx & 0xFFF) + 1
            partitions = ((ebx >> 12) & 0x3FF) + 1
            ways = ((ebx >> 22) & 0x3FF) + 1
            sets = ecx + 1
            size_kb = (ways * partitions * line_size * sets) // 1024

            # type: 1=Data, 2=Instruction, 3=Unified
            if level == 1 and cache_type == 1:
                info.l1d_kb = size_kb
            elif level == 1 and cache_type == 2:
                info.l1i_kb = size_kb
            elif level == 2:
                info.l2_kb = size_kb
            elif level == 3:
                info.l3_kb = size_kb
    finally:
        cpuid.close()
    return info


def _detect_linux_sysfs() -> CacheInfo:
    """Detect cache sizes from Linux sysfs."""
    info = CacheInfo()
    base = Path("/sys/devices/system/cpu/cpu0/cache")
    if not base.exists():
        return info

    for idx in range(8):
        index_dir = base / f"index{idx}"
        if not index_dir.exists():
            break

        try:
            level = int((index_dir / "level").read_text().strip())
            cache_type = (index_dir / "type").read_text().strip()
            size_str = (index_dir / "size").read_text().strip()

            # Parse size like "48K" or "1280K" or "24M"
            if size_str.endswith("K"):
                size_kb = int(size_str[:-1])
            elif size_str.endswith("M"):
                size_kb = int(size_str[:-1]) * 1024
            else:
                size_kb = int(size_str) // 1024

            if level == 1 and cache_type == "Data":
                info.l1d_kb = size_kb
            elif level == 1 and cache_type == "Instruction":
                info.l1i_kb = size_kb
            elif level == 2:
                info.l2_kb = size_kb
            elif level == 3:
                info.l3_kb = size_kb
        except (ValueError, OSError):
            continue
    return info


def _detect_macos_sysctl() -> CacheInfo:
    """Detect cache sizes via sysctl on macOS."""
    import subprocess

    info = CacheInfo()
    mapping = {
        "hw.l1dcachesize": "l1d_kb",
        "hw.l1icachesize": "l1i_kb",
        "hw.l2cachesize": "l2_kb",
        "hw.l3cachesize": "l3_kb",
    }
    for key, attr in mapping.items():
        try:
            result = subprocess.run(
                ["sysctl", "-n", key],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                setattr(info, attr, int(result.stdout.strip()) // 1024)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            continue
    return info


def detect_cache() -> CacheInfo:
    """Detect CPU cache sizes for the current platform."""
    system = platform.system()
    machine = platform.machine().lower()

    if machine in ("amd64", "x86_64", "x86"):
        if system == "Windows":
            return _detect_x86_cpuid()
        elif system == "Linux":
            # Prefer sysfs, fall back to CPUID
            info = _detect_linux_sysfs()
            if info.l1d_kb > 0:
                return info
            return _detect_x86_cpuid()
        elif system == "Darwin":
            return _detect_macos_sysctl()
    elif machine in ("aarch64", "arm64"):
        if system == "Linux":
            return _detect_linux_sysfs()
        elif system == "Darwin":
            return _detect_macos_sysctl()

    return CacheInfo()
