"""Data types for CPU detection results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CpuFeatures:
    """CPU SIMD feature flags."""

    # x86 SIMD
    sse3: bool = False
    ssse3: bool = False
    sse4_1: bool = False
    sse4_2: bool = False
    avx: bool = False
    avx2: bool = False
    avx_vnni: bool = False
    fma: bool = False
    f16c: bool = False
    bmi2: bool = False
    avx512f: bool = False
    avx512bw: bool = False
    avx512vl: bool = False
    avx512_vnni: bool = False
    avx512_bf16: bool = False
    avx512_vbmi: bool = False
    amx_tile: bool = False
    amx_int8: bool = False
    amx_bf16: bool = False

    # ARM
    neon: bool = False
    sve: bool = False
    sve2: bool = False
    dotprod: bool = False
    i8mm: bool = False
    bf16: bool = False
    fp16: bool = False
    sme: bool = False
    sve_vector_length: int = 0  # bytes


@dataclass
class CacheInfo:
    """CPU cache sizes in KB."""

    l1d_kb: int = 0
    l1i_kb: int = 0
    l2_kb: int = 0
    l3_kb: int = 0


@dataclass
class MemoryInfo:
    """System memory information."""

    total_gb: float = 0.0
    available_gb: float = 0.0


@dataclass
class NumaInfo:
    """NUMA topology."""

    num_nodes: int = 1
    cpus_per_node: list[list[int]] = field(default_factory=list)


@dataclass
class Recommendation:
    """Backend and quantization recommendation."""

    backend: str = "llama_cpp"
    quantization: str = "Q4_K_M"
    reason: str = ""
    fallback_backend: str = "llama_cpp"
    threads: int = 0
    max_model_b: float = 0.0  # max model size in billions of params
    numa_strategy: str = ""  # "" or "distribute"


@dataclass
class CpuInfo:
    """Complete CPU detection result."""

    vendor: str = ""
    brand: str = ""
    arch: str = ""
    family: int = 0
    model: int = 0
    stepping: int = 0
    cores_physical: int = 0
    cores_logical: int = 0
    features: CpuFeatures = field(default_factory=CpuFeatures)
    cache: CacheInfo = field(default_factory=CacheInfo)
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    numa: NumaInfo = field(default_factory=NumaInfo)
    recommendation: Recommendation = field(default_factory=Recommendation)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        from dataclasses import asdict
        return asdict(self)
