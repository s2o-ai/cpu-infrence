"""Tests for the CPU detection module."""

import json
import platform
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.detect import detect_cpu, detect_cpu_json
from engine.detect._types import CpuFeatures, CpuInfo, Recommendation
from engine.detect._recommend import recommend, _adjust_for_memory


class TestDetectCpu:
    """Integration tests - run on the current machine."""

    def test_detect_returns_cpuinfo(self):
        info = detect_cpu()
        assert isinstance(info, CpuInfo)
        assert info.arch in ("x86_64", "aarch64", platform.machine().lower())

    def test_vendor_not_empty(self):
        info = detect_cpu()
        assert len(info.vendor) > 0

    def test_brand_not_empty(self):
        info = detect_cpu()
        assert len(info.brand) > 0

    def test_cores_positive(self):
        info = detect_cpu()
        assert info.cores_logical > 0
        assert info.cores_physical > 0
        assert info.cores_physical <= info.cores_logical

    def test_memory_detected(self):
        info = detect_cpu()
        assert info.memory.total_gb > 0

    def test_cache_detected(self):
        info = detect_cpu()
        # At minimum L1 data cache should be detected
        assert info.cache.l1d_kb > 0

    def test_recommendation_populated(self):
        info = detect_cpu()
        assert info.recommendation.backend != ""
        assert info.recommendation.quantization != ""
        assert info.recommendation.threads > 0

    def test_to_dict(self):
        info = detect_cpu()
        d = info.to_dict()
        assert isinstance(d, dict)
        assert "vendor" in d
        assert "features" in d
        assert "cache" in d
        assert "memory" in d
        assert "recommendation" in d

    def test_json_output(self):
        d = detect_cpu_json()
        # Should be JSON-serializable
        s = json.dumps(d)
        parsed = json.loads(s)
        assert parsed["vendor"] == d["vendor"]


class TestRecommendation:
    """Unit tests for recommendation logic."""

    def _make_features(self, **kwargs) -> CpuFeatures:
        return CpuFeatures(**kwargs)

    def test_intel_amx(self):
        from engine.detect._types import CacheInfo, MemoryInfo, NumaInfo
        feat = self._make_features(avx2=True, avx512f=True, avx512bw=True,
                                    avx512vl=True, amx_tile=True, amx_int8=True)
        rec = recommend("GenuineIntel", "x86_64", feat,
                        CacheInfo(), MemoryInfo(total_gb=64), NumaInfo(), 16)
        assert rec.backend == "openvino"

    def test_intel_avx512(self):
        from engine.detect._types import CacheInfo, MemoryInfo, NumaInfo
        feat = self._make_features(avx2=True, avx512f=True, avx512bw=True, avx512vl=True)
        rec = recommend("GenuineIntel", "x86_64", feat,
                        CacheInfo(), MemoryInfo(total_gb=64), NumaInfo(), 16)
        assert rec.backend == "llama_cpp_lut"

    def test_intel_avx2(self):
        from engine.detect._types import CacheInfo, MemoryInfo, NumaInfo
        feat = self._make_features(avx2=True)
        rec = recommend("GenuineIntel", "x86_64", feat,
                        CacheInfo(), MemoryInfo(total_gb=32), NumaInfo(), 8)
        assert rec.backend == "llama_cpp"
        assert rec.quantization == "Q4_K_M"

    def test_amd_avx512(self):
        from engine.detect._types import CacheInfo, MemoryInfo, NumaInfo
        feat = self._make_features(avx2=True, avx512f=True, avx512bw=True, avx512vl=True)
        rec = recommend("AuthenticAMD", "x86_64", feat,
                        CacheInfo(), MemoryInfo(total_gb=64), NumaInfo(), 32)
        assert rec.backend == "llama_cpp_lut"

    def test_arm_sve(self):
        from engine.detect._types import CacheInfo, MemoryInfo, NumaInfo
        feat = self._make_features(neon=True, sve=True, dotprod=True)
        rec = recommend("ARM", "aarch64", feat,
                        CacheInfo(), MemoryInfo(total_gb=32), NumaInfo(), 8)
        assert rec.backend == "lut_neon"

    def test_arm_neon_dotprod(self):
        from engine.detect._types import CacheInfo, MemoryInfo, NumaInfo
        feat = self._make_features(neon=True, dotprod=True)
        rec = recommend("ARM", "aarch64", feat,
                        CacheInfo(), MemoryInfo(total_gb=32), NumaInfo(), 8)
        assert rec.backend == "lut_neon"

    def test_low_memory_adjusts_quantization(self):
        from engine.detect._types import CacheInfo, MemoryInfo, NumaInfo
        feat = self._make_features(avx2=True)
        rec = recommend("GenuineIntel", "x86_64", feat,
                        CacheInfo(), MemoryInfo(total_gb=8), NumaInfo(), 4)
        assert rec.quantization == "Q4_0"
        assert rec.max_model_b <= 3.0

    def test_numa_strategy(self):
        from engine.detect._types import CacheInfo, MemoryInfo, NumaInfo
        feat = self._make_features(avx2=True)
        numa = NumaInfo(num_nodes=2, cpus_per_node=[[0, 1], [2, 3]])
        rec = recommend("GenuineIntel", "x86_64", feat,
                        CacheInfo(), MemoryInfo(total_gb=64), numa, 4)
        assert rec.numa_strategy == "distribute"

    def test_thread_count_matches_physical(self):
        from engine.detect._types import CacheInfo, MemoryInfo, NumaInfo
        feat = self._make_features(avx2=True)
        rec = recommend("GenuineIntel", "x86_64", feat,
                        CacheInfo(), MemoryInfo(total_gb=32), NumaInfo(), 14)
        assert rec.threads == 14


class TestCliScript:
    """Test the CLI script produces output."""

    def test_cli_runs(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "detect.py")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "S2O CPU Detection" in result.stdout

    def test_cli_json(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "detect.py"), "--json"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "vendor" in data
        assert "recommendation" in data
