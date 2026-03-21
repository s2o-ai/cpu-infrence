#!/usr/bin/env python3
"""Build llama-server and related binaries from the llama.cpp submodule."""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLAMA_DIR = ROOT / "engine" / "src" / "llama"
BUILD_DIR = LLAMA_DIR / "build"
BIN_DIR = BUILD_DIR / "bin"


def find_compiler():
    """Find GCC or Clang compiler."""
    for cc, cxx in [("gcc", "g++"), ("clang", "clang++")]:
        if shutil.which(cc):
            return cc, cxx
    # Check MSYS2 path on Windows
    msys2_gcc = Path("C:/msys64/mingw64/bin/gcc.exe")
    if msys2_gcc.exists():
        return str(msys2_gcc), str(msys2_gcc.parent / "g++.exe")
    print("ERROR: No C/C++ compiler found. Install GCC or Clang.")
    sys.exit(1)


def find_cmake():
    """Find CMake binary."""
    cmake = shutil.which("cmake")
    if cmake:
        return cmake
    msys2_cmake = Path("C:/msys64/mingw64/bin/cmake.exe")
    if msys2_cmake.exists():
        return str(msys2_cmake)
    print("ERROR: CMake not found. Install CMake 3.20+.")
    sys.exit(1)


def find_generator():
    """Find Ninja or fallback to Make."""
    if shutil.which("ninja"):
        return "Ninja"
    msys2_ninja = Path("C:/msys64/mingw64/bin/ninja.exe")
    if msys2_ninja.exists():
        return "Ninja"
    if shutil.which("make") or shutil.which("mingw32-make"):
        return "MinGW Makefiles" if platform.system() == "Windows" else "Unix Makefiles"
    return "Ninja"


def get_nproc():
    """Get number of CPU cores."""
    return os.cpu_count() or 4


def build(clean=False, lut=False):
    """Configure and build llama.cpp."""
    if not LLAMA_DIR.exists():
        print(f"ERROR: llama.cpp not found at {LLAMA_DIR}")
        print("Run: git submodule update --init")
        sys.exit(1)

    if clean and BUILD_DIR.exists():
        print("Cleaning build directory...")
        shutil.rmtree(BUILD_DIR)

    # Ensure MSYS2 is on PATH for Windows builds
    msys2_bin = Path("C:/msys64/mingw64/bin")
    if msys2_bin.exists() and str(msys2_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(msys2_bin) + os.pathsep + os.environ.get("PATH", "")

    cc, cxx = find_compiler()
    cmake = find_cmake()
    generator = find_generator()

    print(f"Compiler: {cc} / {cxx}")
    print(f"CMake: {cmake}")
    print(f"Generator: {generator}")
    print(f"Source: {LLAMA_DIR}")
    print(f"Build: {BUILD_DIR}")
    print()

    # CMake configure
    cmake_args = [
        cmake, "-B", str(BUILD_DIR),
        "-G", generator,
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_C_COMPILER={cc}",
        f"-DCMAKE_CXX_COMPILER={cxx}",
        "-DGGML_NATIVE=ON",
        "-DGGML_OPENMP=ON",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_OPENSSL=OFF",
        "-DBUILD_SHARED_LIBS=OFF",
    ]

    # S2O LUT kernels
    if lut:
        cmake_args.append("-DGGML_S2O_LUT=ON")
        print("S2O LUT kernels: ENABLED")

    # Windows 10+ target for MinGW
    if platform.system() == "Windows":
        cmake_args.extend([
            "-DCMAKE_C_FLAGS=-D_WIN32_WINNT=0x0A00 -DNTDDI_VERSION=0x0A000000",
            "-DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00 -DNTDDI_VERSION=0x0A000000",
        ])

    print("Configuring...")
    subprocess.run(cmake_args, cwd=str(LLAMA_DIR), check=True)

    # Build
    nproc = get_nproc()
    print(f"\nBuilding with {nproc} threads...")
    subprocess.run(
        [cmake, "--build", str(BUILD_DIR), "--config", "Release", f"-j{nproc}"],
        cwd=str(LLAMA_DIR),
        check=True,
    )

    # Report built binaries
    print("\nBuild complete. Binaries:")
    for exe in sorted(BIN_DIR.glob("*.exe" if platform.system() == "Windows" else "llama-*")):
        size_mb = exe.stat().st_size / (1024 * 1024)
        print(f"  {exe.name:40s} {size_mb:6.1f} MB")


if __name__ == "__main__":
    clean = "--clean" in sys.argv
    lut = "--lut" in sys.argv
    build(clean=clean, lut=lut)
