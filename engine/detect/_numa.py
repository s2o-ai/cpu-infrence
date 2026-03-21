"""NUMA topology detection."""

from __future__ import annotations

import ctypes
import platform
from pathlib import Path

from ._types import NumaInfo


def _parse_cpulist(text: str) -> list[int]:
    """Parse a Linux cpulist string like '0-3,8-11' into a list of CPU IDs."""
    cpus = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return cpus


def _detect_linux() -> NumaInfo:
    """Detect NUMA topology from Linux sysfs."""
    node_base = Path("/sys/devices/system/node")
    if not node_base.exists():
        return NumaInfo()

    nodes = sorted(
        p for p in node_base.iterdir()
        if p.is_dir() and p.name.startswith("node")
    )
    if not nodes:
        return NumaInfo()

    cpus_per_node = []
    for node_dir in nodes:
        cpulist_file = node_dir / "cpulist"
        try:
            cpus = _parse_cpulist(cpulist_file.read_text())
            cpus_per_node.append(cpus)
        except OSError:
            cpus_per_node.append([])

    return NumaInfo(num_nodes=len(nodes), cpus_per_node=cpus_per_node)


def _detect_windows() -> NumaInfo:
    """Detect NUMA topology on Windows."""
    try:
        highest = ctypes.c_ulong(0)
        if not ctypes.windll.kernel32.GetNumaHighestNodeNumber(ctypes.byref(highest)):
            return NumaInfo()

        num_nodes = highest.value + 1
        if num_nodes <= 1:
            return NumaInfo(num_nodes=1)

        cpus_per_node = []
        for node in range(num_nodes):
            mask = ctypes.c_uint64(0)
            ctypes.windll.kernel32.GetNumaNodeProcessorMask(
                ctypes.c_uchar(node), ctypes.byref(mask)
            )
            cpus = [i for i in range(64) if mask.value & (1 << i)]
            cpus_per_node.append(cpus)

        return NumaInfo(num_nodes=num_nodes, cpus_per_node=cpus_per_node)
    except (OSError, AttributeError):
        return NumaInfo()


def detect_numa() -> NumaInfo:
    """Detect NUMA topology for the current platform."""
    system = platform.system()
    if system == "Linux":
        return _detect_linux()
    elif system == "Windows":
        return _detect_windows()
    # macOS: no NUMA on Apple Silicon
    return NumaInfo()
