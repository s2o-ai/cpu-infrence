"""x86 CPU feature detection via CPUID instruction."""

from __future__ import annotations

import ctypes
import platform
import struct
from typing import Tuple

from ._types import CpuFeatures

# Windows x64 calling convention: rcx=arg1, rdx=arg2, r8=arg3
# We pass: leaf (rcx), subleaf (rdx), output_ptr (r8)
_SHELLCODE_WIN64 = bytes([
    0x53,                       # push rbx          (rbx is callee-saved)
    0x4D, 0x89, 0xC1,           # mov r9, r8        (save output ptr before cpuid clobbers)
    0x89, 0xC8,                 # mov eax, ecx      (leaf)
    0x89, 0xD1,                 # mov ecx, edx      (subleaf)
    0x0F, 0xA2,                 # cpuid
    0x41, 0x89, 0x01,           # mov [r9], eax
    0x41, 0x89, 0x59, 0x04,     # mov [r9+4], ebx
    0x41, 0x89, 0x49, 0x08,     # mov [r9+8], ecx
    0x41, 0x89, 0x51, 0x0C,     # mov [r9+12], edx
    0x5B,                       # pop rbx
    0xC3,                       # ret
])

# System V AMD64 ABI: rdi=arg1, rsi=arg2, rdx=arg3
_SHELLCODE_LINUX64 = bytes([
    0x53,                       # push rbx
    0x49, 0x89, 0xD0,           # mov r8, rdx       (save output ptr)
    0x89, 0xF8,                 # mov eax, edi      (leaf)
    0x89, 0xF1,                 # mov ecx, esi      (subleaf)
    0x0F, 0xA2,                 # cpuid
    0x41, 0x89, 0x00,           # mov [r8], eax
    0x41, 0x89, 0x58, 0x04,     # mov [r8+4], ebx
    0x41, 0x89, 0x48, 0x08,     # mov [r8+8], ecx
    0x41, 0x89, 0x50, 0x0C,     # mov [r8+12], edx
    0x5B,                       # pop rbx
    0xC3,                       # ret
])


class CpuidExecutor:
    """Execute CPUID instruction via shellcode in executable memory."""

    def __init__(self):
        self._addr = None
        self._size = 0
        system = platform.system()

        if system == "Windows":
            self._shellcode = _SHELLCODE_WIN64
            self._alloc_windows()
        else:
            self._shellcode = _SHELLCODE_LINUX64
            self._alloc_posix()

    def _alloc_windows(self):
        kernel32 = ctypes.windll.kernel32
        MEM_COMMIT = 0x1000
        MEM_RESERVE = 0x2000
        PAGE_EXECUTE_READWRITE = 0x40

        # Must set argtypes/restype for 64-bit pointer correctness
        kernel32.VirtualAlloc.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32
        ]
        kernel32.VirtualAlloc.restype = ctypes.c_void_p

        self._size = len(self._shellcode)
        self._addr = kernel32.VirtualAlloc(
            None, self._size, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE
        )
        if not self._addr:
            raise OSError("VirtualAlloc failed")
        ctypes.memmove(self._addr, self._shellcode, self._size)

    def _alloc_posix(self):
        import mmap
        self._size = len(self._shellcode)
        self._buf = mmap.mmap(
            -1, self._size,
            prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
        )
        self._buf.write(self._shellcode)
        self._addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buf))

    def __call__(self, leaf: int, subleaf: int = 0) -> Tuple[int, int, int, int]:
        """Execute CPUID and return (eax, ebx, ecx, edx)."""
        output = (ctypes.c_uint32 * 4)()
        func_type = ctypes.CFUNCTYPE(
            None,
            ctypes.c_uint32,                    # leaf
            ctypes.c_uint32,                    # subleaf
            ctypes.POINTER(ctypes.c_uint32),    # output
        )
        func = func_type(self._addr)
        func(ctypes.c_uint32(leaf), ctypes.c_uint32(subleaf), output)
        return output[0], output[1], output[2], output[3]

    def close(self):
        if self._addr and platform.system() == "Windows":
            MEM_RELEASE = 0x8000
            kernel32 = ctypes.windll.kernel32
            kernel32.VirtualFree.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32
            ]
            kernel32.VirtualFree.restype = ctypes.c_int
            kernel32.VirtualFree(self._addr, 0, MEM_RELEASE)
            self._addr = None
        elif hasattr(self, "_buf"):
            self._buf.close()
            self._addr = None

    def __del__(self):
        self.close()


def _regs_to_str(ebx: int, edx: int, ecx: int) -> str:
    """Convert register values to vendor string (leaf 0 order: EBX+EDX+ECX)."""
    return (
        struct.pack("<I", ebx).decode("ascii", errors="replace")
        + struct.pack("<I", edx).decode("ascii", errors="replace")
        + struct.pack("<I", ecx).decode("ascii", errors="replace")
    )


def _brand_string(cpuid: CpuidExecutor) -> str:
    """Get CPU brand string from extended CPUID leaves 0x80000002-4."""
    eax, _, _, _ = cpuid(0x80000000)
    if eax < 0x80000004:
        return ""
    parts = []
    for leaf in (0x80000002, 0x80000003, 0x80000004):
        a, b, c, d = cpuid(leaf)
        parts.append(struct.pack("<IIII", a, b, c, d))
    return b"".join(parts).decode("ascii", errors="replace").strip().rstrip("\x00")


def detect_x86() -> Tuple[CpuFeatures, str, str, int, int, int]:
    """Detect x86 CPU features via CPUID.

    Returns: (features, vendor, brand, family, model, stepping)
    """
    cpuid = CpuidExecutor()
    try:
        features = CpuFeatures()

        # Leaf 0: vendor
        eax, ebx, ecx, edx = cpuid(0)
        max_leaf = eax
        vendor = _regs_to_str(ebx, edx, ecx)

        # Brand string
        brand = _brand_string(cpuid)

        # Leaf 1: family/model/stepping + basic features
        family = model = stepping = 0
        if max_leaf >= 1:
            eax, ebx, ecx, edx = cpuid(1)
            stepping = eax & 0xF
            base_model = (eax >> 4) & 0xF
            base_family = (eax >> 8) & 0xF
            ext_model = (eax >> 16) & 0xF
            ext_family = (eax >> 20) & 0xFF

            if base_family == 0xF:
                family = base_family + ext_family
            else:
                family = base_family
            if base_family in (0x6, 0xF):
                model = (ext_model << 4) | base_model
            else:
                model = base_model

            features.sse3 = bool(ecx & (1 << 0))
            features.ssse3 = bool(ecx & (1 << 9))
            features.sse4_1 = bool(ecx & (1 << 19))
            features.sse4_2 = bool(ecx & (1 << 20))
            features.fma = bool(ecx & (1 << 12))
            features.avx = bool(ecx & (1 << 28))
            features.f16c = bool(ecx & (1 << 29))

        # Leaf 7, subleaf 0: extended features
        if max_leaf >= 7:
            eax, ebx, ecx, edx = cpuid(7, 0)
            features.bmi2 = bool(ebx & (1 << 8))
            features.avx2 = bool(ebx & (1 << 5))
            features.avx512f = bool(ebx & (1 << 16))
            features.avx512bw = bool(ebx & (1 << 30))
            features.avx512vl = bool(ebx & (1 << 31))
            features.avx512_vbmi = bool(ecx & (1 << 1))
            features.avx512_vnni = bool(ecx & (1 << 11))
            features.amx_bf16 = bool(edx & (1 << 22))
            features.amx_tile = bool(edx & (1 << 24))
            features.amx_int8 = bool(edx & (1 << 25))

            # Leaf 7, subleaf 1
            max_subleaf = eax
            if max_subleaf >= 1:
                eax1, _, _, edx1 = cpuid(7, 1)
                features.avx_vnni = bool(eax1 & (1 << 4))
                features.avx512_bf16 = bool(eax1 & (1 << 5))

        return features, vendor, brand, family, model, stepping
    finally:
        cpuid.close()
