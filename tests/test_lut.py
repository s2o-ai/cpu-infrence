"""Tests for S2O LUT kernel integration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
ENGINE_DIR = ROOT / "engine"
LLAMA_DIR = ENGINE_DIR / "src" / "llama"
LUT_DIR = LLAMA_DIR / "ggml" / "src" / "ggml-cpu" / "s2o-lut"


# --------------------------------------------------------------------------
# File existence tests
# --------------------------------------------------------------------------

class TestLutFilesExist:
    """Verify all S2O LUT source files are present."""

    def test_header_exists(self):
        assert (LUT_DIR / "s2o-lut.h").exists()

    def test_common_header_exists(self):
        assert (LUT_DIR / "lut-common.h").exists()

    def test_integration_cpp_exists(self):
        assert (LUT_DIR / "s2o-lut.cpp").exists()

    def test_avx2_kernel_exists(self):
        assert (LUT_DIR / "lut-x86-avx2.cpp").exists()

    def test_avx512_kernel_exists(self):
        assert (LUT_DIR / "lut-x86-avx512.cpp").exists()

    def test_correctness_test_exists(self):
        assert (LUT_DIR / "test_lut.cpp").exists()


# --------------------------------------------------------------------------
# CMake configuration tests
# --------------------------------------------------------------------------

class TestLutCMakeIntegration:
    """Verify CMake integration for S2O LUT."""

    def test_cmake_option_defined(self):
        cmake_file = LLAMA_DIR / "ggml" / "CMakeLists.txt"
        content = cmake_file.read_text()
        assert "GGML_S2O_LUT" in content, "GGML_S2O_LUT option not found in CMakeLists.txt"

    def test_cmake_source_registration(self):
        cmake_file = LLAMA_DIR / "ggml" / "src" / "ggml-cpu" / "CMakeLists.txt"
        content = cmake_file.read_text()
        assert "s2o-lut/s2o-lut.cpp" in content, "s2o-lut sources not registered"

    def test_cmake_guard_x86(self):
        cmake_file = LLAMA_DIR / "ggml" / "src" / "ggml-cpu" / "CMakeLists.txt"
        content = cmake_file.read_text()
        assert 'GGML_S2O_LUT AND GGML_SYSTEM_ARCH STREQUAL "x86"' in content


# --------------------------------------------------------------------------
# Build script tests
# --------------------------------------------------------------------------

class TestLutBuildScript:
    """Verify build.py --lut flag integration."""

    def test_build_script_has_lut_param(self):
        build_py = SCRIPTS_DIR / "build.py"
        content = build_py.read_text()
        assert "lut" in content
        assert "GGML_S2O_LUT" in content

    def test_build_cmd_has_lut_option(self):
        build_cmd = SCRIPTS_DIR / "commands" / "build_cmd.py"
        content = build_cmd.read_text()
        assert "--lut" in content


# --------------------------------------------------------------------------
# Kernel source code sanity tests
# --------------------------------------------------------------------------

class TestLutKernelSanity:
    """Basic sanity checks on kernel source files."""

    def test_avx2_exports_kernel_table(self):
        content = (LUT_DIR / "lut-x86-avx2.cpp").read_text()
        assert "s2o_lut_kernels_avx2" in content
        assert "s2o_lut_gemv_q4_0_avx2" in content

    def test_avx512_exports_kernel_table(self):
        content = (LUT_DIR / "lut-x86-avx512.cpp").read_text()
        assert "s2o_lut_kernels_avx512" in content
        assert "s2o_lut_gemv_q4_0_avx512" in content

    def test_integration_has_buffer_type(self):
        content = (LUT_DIR / "s2o-lut.cpp").read_text()
        assert "ggml_backend_cpu_s2o_lut_buffer_type" in content
        assert "extra_buffer_type" in content
        assert "tensor_traits" in content

    def test_common_header_has_kernel_signatures(self):
        content = (LUT_DIR / "lut-common.h").read_text()
        assert "s2o_lut_gemv_fn" in content
        assert "s2o_lut_gemm_fn" in content
        assert "s2o_lut_kernels" in content

    def test_public_header_declares_buffer_type(self):
        content = (LUT_DIR / "s2o-lut.h").read_text()
        assert "ggml_backend_cpu_s2o_lut_buffer_type" in content


# --------------------------------------------------------------------------
# ARM NEON kernel tests
# --------------------------------------------------------------------------

class TestLutArmKernel:
    """Verify ARM NEON LUT kernel files and integration."""

    def test_arm_neon_kernel_exists(self):
        assert (LUT_DIR / "lut-arm-neon.cpp").exists()

    def test_arm_neon_kernel_has_dispatch_table(self):
        content = (LUT_DIR / "lut-arm-neon.cpp").read_text()
        assert "s2o_lut_kernels_neon" in content
        assert "s2o_lut_kernels_neon_dotprod" in content

    def test_arm_neon_kernel_has_gemv(self):
        content = (LUT_DIR / "lut-arm-neon.cpp").read_text()
        assert "s2o_lut_gemv_q4_0_neon" in content

    def test_arm_neon_kernel_has_dotprod_variant(self):
        content = (LUT_DIR / "lut-arm-neon.cpp").read_text()
        assert "vdotq_s32" in content

    def test_arm_neon_kernel_has_quantization(self):
        content = (LUT_DIR / "lut-arm-neon.cpp").read_text()
        assert "s2o_quantize_block_f32_to_i8_neon" in content

    def test_cmake_guard_arm(self):
        cmake_file = LLAMA_DIR / "ggml" / "src" / "ggml-cpu" / "CMakeLists.txt"
        content = cmake_file.read_text()
        assert 'GGML_S2O_LUT AND GGML_SYSTEM_ARCH STREQUAL "arm"' in content

    def test_arm_externs_in_common_header(self):
        content = (LUT_DIR / "lut-common.h").read_text()
        assert "s2o_lut_kernels_neon" in content
        assert "s2o_lut_kernels_neon_dotprod" in content

    def test_arm_dispatch_in_selector(self):
        content = (LUT_DIR / "lut-common.h").read_text()
        assert "__ARM_NEON" in content

    def test_integration_guard_includes_arm(self):
        content = (LUT_DIR / "s2o-lut.cpp").read_text()
        assert "__ARM_NEON" in content


# --------------------------------------------------------------------------
# Benchmark script tests
# --------------------------------------------------------------------------

class TestLutBenchmark:
    """Verify benchmark script exists and is importable."""

    def test_bench_lut_exists(self):
        assert (ROOT / "benchmarks" / "bench_lut.py").exists()

    def test_bench_lut_importable(self):
        sys.path.insert(0, str(ROOT))
        import benchmarks.bench_lut as bl
        assert hasattr(bl, "BenchResult")
        assert hasattr(bl, "print_comparison")
