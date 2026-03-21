"""S2O CPU detection module.

Detects CPU features, cache topology, memory, NUMA, and recommends
the optimal inference backend and quantization format.

Usage:
    from engine.detect import detect_cpu, detect_cpu_json

    info = detect_cpu()
    print(info.vendor, info.features.avx2)
    print(info.recommendation.backend)
"""

from __future__ import annotations

import os
import platform

from ._types import CpuInfo, CpuFeatures, CacheInfo, MemoryInfo, NumaInfo, Recommendation
from ._cache import detect_cache
from ._memory import detect_memory
from ._numa import detect_numa
from ._recommend import recommend

__all__ = ["detect_cpu", "detect_cpu_json", "CpuInfo"]


def detect_cpu() -> CpuInfo:
    """Detect CPU capabilities, memory, and recommend backend configuration."""
    info = CpuInfo()
    machine = platform.machine().lower()

    if machine in ("amd64", "x86_64", "x86"):
        info.arch = "x86_64"
        from ._cpuid_x86 import detect_x86
        features, vendor, brand, family, model, stepping = detect_x86()
        info.features = features
        info.vendor = vendor
        info.brand = brand
        info.family = family
        info.model = model
        info.stepping = stepping
    elif machine in ("aarch64", "arm64"):
        info.arch = "aarch64"
        info.vendor = "ARM"
        from ._hwcap_arm import detect_arm
        info.features = detect_arm()
        info.brand = platform.processor() or "ARM"
    else:
        info.arch = machine
        info.vendor = "unknown"
        info.brand = platform.processor() or "unknown"

    # Core counts
    info.cores_logical = os.cpu_count() or 1
    info.cores_physical = _get_physical_cores(info.cores_logical)

    # Cache, memory, NUMA
    info.cache = detect_cache()
    info.memory = detect_memory()
    info.numa = detect_numa()

    # Recommendation
    info.recommendation = recommend(
        vendor=info.vendor,
        arch=info.arch,
        features=info.features,
        cache=info.cache,
        memory=info.memory,
        numa=info.numa,
        cores_physical=info.cores_physical,
    )

    return info


def detect_cpu_json() -> dict:
    """Return detection results as a JSON-serializable dictionary."""
    return detect_cpu().to_dict()


def _get_physical_cores(logical: int) -> int:
    """Get physical core count (best effort)."""
    system = platform.system()

    if system == "Windows":
        # NUMBER_OF_PROCESSORS gives logical; use wmic for physical
        try:
            import subprocess
            result = subprocess.run(
                ["wmic", "cpu", "get", "NumberOfCores", "/value"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().split("\n"):
                if line.startswith("NumberOfCores="):
                    return int(line.split("=")[1].strip())
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                core_ids = set()
                phys_ids = set()
                cur_phys = cur_core = None
                for line in f:
                    if line.startswith("physical id"):
                        cur_phys = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        cur_core = line.split(":")[1].strip()
                    elif line.strip() == "" and cur_phys is not None and cur_core is not None:
                        core_ids.add((cur_phys, cur_core))
                        cur_phys = cur_core = None
                if cur_phys is not None and cur_core is not None:
                    core_ids.add((cur_phys, cur_core))
                if core_ids:
                    return len(core_ids)
        except OSError:
            pass
    elif system == "Darwin":
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass

    # Fallback: assume half of logical cores are physical
    return max(1, logical // 2)
