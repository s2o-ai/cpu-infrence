#!/usr/bin/env python3
"""Start and manage the llama-server inference process."""

import argparse
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    httpx = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEFAULT_BIN = ROOT / "engine" / "src" / "llama" / "build" / "bin"

if platform.system() == "Windows":
    SERVER_BIN = DEFAULT_BIN / "llama-server.exe"
else:
    SERVER_BIN = DEFAULT_BIN / "llama-server"


def find_model(model_path: str) -> Path:
    """Resolve model path, checking models/ directory."""
    p = Path(model_path)
    if p.exists():
        return p
    # Check in models/ directory
    models_dir = ROOT / "models"
    for gguf in models_dir.rglob("*.gguf"):
        if model_path in str(gguf):
            return gguf
    print(f"ERROR: Model not found: {model_path}")
    print(f"Available models in {models_dir}:")
    for gguf in models_dir.rglob("*.gguf"):
        print(f"  {gguf.relative_to(ROOT)}")
    sys.exit(1)


def wait_for_health(host: str, port: int, timeout: int = 30) -> bool:
    """Wait for server to become healthy."""
    url = f"http://{host}:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            if httpx:
                r = httpx.get(url, timeout=2)
                if r.status_code == 200:
                    return True
            else:
                import urllib.request
                req = urllib.request.urlopen(url, timeout=2)
                if req.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def serve(args):
    """Start llama-server."""
    if not SERVER_BIN.exists():
        print(f"ERROR: llama-server not found at {SERVER_BIN}")
        print("Run: python scripts/build.py")
        sys.exit(1)

    model = find_model(args.model)

    # Detect CPU and use recommended thread count
    try:
        from engine.detect import detect_cpu
        cpu = detect_cpu()
        recommended_threads = cpu.recommendation.threads
        print(f"CPU:      {cpu.brand}")
        print(f"SIMD:     ", end="")
        simd = []
        if cpu.features.avx512f: simd.append("AVX-512")
        elif cpu.features.avx2: simd.append("AVX2")
        if cpu.features.avx_vnni: simd.append("AVX-VNNI")
        if cpu.features.fma: simd.append("FMA")
        if cpu.features.amx_int8: simd.append("AMX")
        if cpu.features.neon: simd.append("NEON")
        if cpu.features.sve2: simd.append("SVE2")
        elif cpu.features.sve: simd.append("SVE")
        print(", ".join(simd) if simd else "baseline")
        print(f"Memory:   {cpu.memory.total_gb} GB")
        print(f"Backend:  {cpu.recommendation.backend} ({cpu.recommendation.reason})")
        print()
    except Exception:
        recommended_threads = os.cpu_count() or 4

    threads = args.threads or recommended_threads

    cmd = [
        str(SERVER_BIN),
        "-m", str(model),
        "--host", args.host,
        "--port", str(args.port),
        "--ctx-size", str(args.ctx_size),
        "--parallel", str(args.parallel),
        "--cont-batching",
        "--metrics",
        "-t", str(threads),
    ]

    if args.api_key:
        cmd.extend(["--api-key", args.api_key])

    print(f"Model:    {model}")
    print(f"Server:   http://{args.host}:{args.port}")
    print(f"Threads:  {threads}")
    print(f"Context:  {args.ctx_size}")
    print(f"Parallel: {args.parallel} slots")
    print()
    print("Endpoints:")
    print(f"  Chat:    POST http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  Models:  GET  http://{args.host}:{args.port}/v1/models")
    print(f"  Health:  GET  http://{args.host}:{args.port}/health")
    print(f"  Metrics: GET  http://{args.host}:{args.port}/metrics")
    print(f"  Web UI:  http://{args.host}:{args.port}")
    print()
    print("Starting server...")

    proc = subprocess.Popen(cmd)

    def shutdown(sig, frame):
        print("\nShutting down...")
        proc.terminate()
        proc.wait(timeout=10)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Wait for health
    if wait_for_health(args.host, args.port):
        print("Server is ready!")
    else:
        print("WARNING: Server did not become healthy within 30s")

    proc.wait()


def main():
    parser = argparse.ArgumentParser(description="Start S2O inference server")
    parser.add_argument("model", help="Path to GGUF model file or name to search in models/")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--ctx-size", type=int, default=4096, help="Context size (default: 4096)")
    parser.add_argument("--parallel", type=int, default=4, help="Parallel slots (default: 4)")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads (default: all)")
    parser.add_argument("--api-key", default=None, help="API key for authentication")
    args = parser.parse_args()
    serve(args)


if __name__ == "__main__":
    main()
