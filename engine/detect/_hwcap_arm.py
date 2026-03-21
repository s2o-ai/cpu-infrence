"""ARM CPU feature detection."""

from __future__ import annotations

import platform

from ._types import CpuFeatures

# Linux HWCAP bit definitions (from asm/hwcap.h)
_HWCAP_NEON = 1 << 12       # aarch32
_HWCAP_FP = 1 << 0          # aarch64
_HWCAP_ASIMD = 1 << 1       # aarch64 (NEON)
_HWCAP_FPHP = 1 << 9
_HWCAP_ASIMDHP = 1 << 10    # FP16
_HWCAP_ASIMDDP = 1 << 20    # DOTPROD
_HWCAP_SVE = 1 << 22

# HWCAP2 bits
_HWCAP2_SVE2 = 1 << 1
_HWCAP2_I8MM = 1 << 13
_HWCAP2_BF16 = 1 << 14
_HWCAP2_SME = 1 << 23

_AT_HWCAP = 16
_AT_HWCAP2 = 26


def _detect_linux() -> CpuFeatures:
    """Detect ARM features via getauxval on Linux."""
    import ctypes

    features = CpuFeatures()
    try:
        libc = ctypes.CDLL(None)
        libc.getauxval.restype = ctypes.c_ulong
        libc.getauxval.argtypes = [ctypes.c_ulong]

        hwcap = libc.getauxval(_AT_HWCAP)
        hwcap2 = libc.getauxval(_AT_HWCAP2)

        features.neon = bool(hwcap & _HWCAP_ASIMD)
        features.fp16 = bool(hwcap & _HWCAP_ASIMDHP)
        features.dotprod = bool(hwcap & _HWCAP_ASIMDDP)
        features.sve = bool(hwcap & _HWCAP_SVE)
        features.sve2 = bool(hwcap2 & _HWCAP2_SVE2)
        features.i8mm = bool(hwcap2 & _HWCAP2_I8MM)
        features.bf16 = bool(hwcap2 & _HWCAP2_BF16)
        features.sme = bool(hwcap2 & _HWCAP2_SME)

        # SVE vector length via prctl
        if features.sve:
            try:
                PR_SVE_GET_VL = 51
                libc.prctl.restype = ctypes.c_int
                libc.prctl.argtypes = [ctypes.c_int]
                vl = libc.prctl(PR_SVE_GET_VL)
                if vl > 0:
                    features.sve_vector_length = vl & 0xFFFF  # lower 16 bits
            except (OSError, AttributeError):
                pass
    except OSError:
        pass
    return features


def _detect_macos() -> CpuFeatures:
    """Detect ARM features via sysctl on macOS."""
    import subprocess

    features = CpuFeatures()
    features.neon = True  # Always present on Apple Silicon

    checks = {
        "hw.optional.arm.FEAT_DotProd": "dotprod",
        "hw.optional.arm.FEAT_FP16": "fp16",
        "hw.optional.arm.FEAT_I8MM": "i8mm",
        "hw.optional.arm.FEAT_BF16": "bf16",
        "hw.optional.arm.FEAT_SME": "sme",
    }
    for key, attr in checks.items():
        try:
            result = subprocess.run(
                ["sysctl", "-n", key],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip() == "1":
                setattr(features, attr, True)
        except (subprocess.TimeoutExpired, OSError):
            continue
    return features


def _detect_windows() -> CpuFeatures:
    """Detect ARM features via IsProcessorFeaturePresent on Windows."""
    import ctypes

    features = CpuFeatures()
    try:
        PF_ARM_NEON = 19
        if ctypes.windll.kernel32.IsProcessorFeaturePresent(PF_ARM_NEON):
            features.neon = True
    except (OSError, AttributeError):
        pass
    return features


def detect_arm() -> CpuFeatures:
    """Detect ARM CPU features for the current platform."""
    system = platform.system()
    if system == "Linux":
        return _detect_linux()
    elif system == "Darwin":
        return _detect_macos()
    elif system == "Windows":
        return _detect_windows()
    return CpuFeatures()
