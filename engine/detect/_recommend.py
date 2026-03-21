"""Backend and quantization recommendation engine."""

from __future__ import annotations

from ._types import CpuFeatures, CacheInfo, MemoryInfo, NumaInfo, Recommendation


def recommend(
    vendor: str,
    arch: str,
    features: CpuFeatures,
    cache: CacheInfo,
    memory: MemoryInfo,
    numa: NumaInfo,
    cores_physical: int,
) -> Recommendation:
    """Generate backend and quantization recommendation based on detected hardware."""

    rec = Recommendation()
    rec.threads = cores_physical if cores_physical > 0 else 4

    # Backend selection
    if arch in ("x86_64", "AMD64"):
        rec = _recommend_x86(vendor, features, rec)
    elif arch in ("aarch64", "arm64"):
        rec = _recommend_arm(features, rec)
    else:
        rec.backend = "llama_cpp"
        rec.quantization = "Q8_0"
        rec.reason = f"Unknown architecture ({arch}) - using safe defaults"

    # Adjust quantization and max model size based on RAM
    rec = _adjust_for_memory(memory, rec)

    # NUMA strategy
    if numa.num_nodes > 1:
        rec.numa_strategy = "distribute"

    rec.fallback_backend = "llama_cpp"
    return rec


def _recommend_x86(vendor: str, features: CpuFeatures, rec: Recommendation) -> Recommendation:
    """Recommendation logic for x86 CPUs."""

    if vendor == "GenuineIntel":
        if features.amx_int8 and features.amx_tile:
            rec.backend = "openvino"
            rec.quantization = "Q4_K_M"
            rec.reason = "Intel AMX detected - OpenVINO can exploit AMX for INT8 acceleration"
        elif features.avx512f and features.avx512bw and features.avx512vl:
            rec.backend = "llama_cpp_lut"
            rec.quantization = "Q4_K_M"
            rec.reason = "AVX-512 detected - optimal for LUT-based INT4 matmul"
        elif features.avx2:
            rec.backend = "llama_cpp"
            rec.quantization = "Q4_K_M"
            rec.reason = "AVX2 detected - good baseline performance"
        else:
            rec.backend = "llama_cpp"
            rec.quantization = "Q8_0"
            rec.reason = "Basic x86 - limited SIMD support"

    elif vendor == "AuthenticAMD":
        if features.avx512f and features.avx512bw and features.avx512vl:
            rec.backend = "llama_cpp_lut"
            rec.quantization = "Q4_K_M"
            rec.reason = "AMD EPYC with AVX-512 - optimal for LUT kernels"
        elif features.avx2:
            rec.backend = "llama_cpp"
            rec.quantization = "Q4_K_M"
            rec.reason = "AVX2 detected - good baseline performance"
        else:
            rec.backend = "llama_cpp"
            rec.quantization = "Q8_0"
            rec.reason = "Basic x86 - limited SIMD support"

    else:
        # Unknown x86 vendor
        if features.avx2:
            rec.backend = "llama_cpp"
            rec.quantization = "Q4_K_M"
            rec.reason = "AVX2 detected"
        else:
            rec.backend = "llama_cpp"
            rec.quantization = "Q8_0"
            rec.reason = "Basic x86"

    return rec


def _recommend_arm(features: CpuFeatures, rec: Recommendation) -> Recommendation:
    """Recommendation logic for ARM CPUs."""

    if features.sve2 or features.sve:
        rec.backend = "lut_neon"
        rec.quantization = "Q4_K_M"
        rec.reason = "ARM SVE/SVE2 detected - optimal for NEON/SVE LUT kernels"
    elif features.dotprod and features.neon:
        rec.backend = "lut_neon"
        rec.quantization = "Q4_K_M"
        rec.reason = "ARM NEON + DOTPROD detected - good for LUT kernels"
    elif features.neon:
        rec.backend = "llama_cpp"
        rec.quantization = "Q4_K_M"
        rec.reason = "ARM NEON detected - baseline performance"
    else:
        rec.backend = "llama_cpp"
        rec.quantization = "Q8_0"
        rec.reason = "Basic ARM - limited SIMD"

    return rec


def _adjust_for_memory(memory: MemoryInfo, rec: Recommendation) -> Recommendation:
    """Adjust recommendation based on available system memory."""

    total = memory.total_gb
    if total >= 60:
        rec.max_model_b = 13.0
    elif total >= 30:
        rec.max_model_b = 7.0
    elif total >= 14:
        rec.max_model_b = 7.0
        if rec.quantization == "Q4_K_M":
            rec.quantization = "Q4_K_S"
    elif total >= 8:
        rec.max_model_b = 3.0
        rec.quantization = "Q4_0"
    else:
        rec.max_model_b = 1.0
        rec.quantization = "Q4_0"

    return rec
