"""System memory detection."""

from __future__ import annotations

import ctypes
import platform

from ._types import MemoryInfo


def _detect_windows() -> MemoryInfo:
    """Detect memory via GlobalMemoryStatusEx on Windows."""

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_uint32),
            ("dwMemoryLoad", ctypes.c_uint32),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return MemoryInfo(
            total_gb=round(stat.ullTotalPhys / (1024 ** 3), 1),
            available_gb=round(stat.ullAvailPhys / (1024 ** 3), 1),
        )
    return MemoryInfo()


def _detect_linux() -> MemoryInfo:
    """Detect memory from /proc/meminfo."""
    info = MemoryInfo()
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    info.total_gb = round(kb / (1024 ** 2), 1)
                elif line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    info.available_gb = round(kb / (1024 ** 2), 1)
    except OSError:
        pass
    return info


def _detect_macos() -> MemoryInfo:
    """Detect memory via sysctl on macOS."""
    import subprocess

    info = MemoryInfo()
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info.total_gb = round(int(result.stdout.strip()) / (1024 ** 3), 1)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass
    return info


def detect_memory() -> MemoryInfo:
    """Detect system memory for the current platform."""
    system = platform.system()
    if system == "Windows":
        return _detect_windows()
    elif system == "Linux":
        return _detect_linux()
    elif system == "Darwin":
        return _detect_macos()
    return MemoryInfo()
