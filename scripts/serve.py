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

    # KV cache quantization
    kv_quant = getattr(args, "kv_quant", None)
    if kv_quant and kv_quant != "f16":
        cmd.extend(["--cache-type-k", kv_quant, "--cache-type-v", kv_quant])

    # Speculative decoding
    speculative = getattr(args, "speculative", False)
    draft_model = getattr(args, "draft_model", None)
    if speculative and draft_model:
        draft_path = find_model(draft_model)
        cmd.extend(["--model-draft", str(draft_path)])
    elif speculative and not draft_model:
        # Auto-select draft model
        try:
            from engine.serving.speculative import suggest_draft_model
            models_dir = ROOT / "models"
            draft = suggest_draft_model(str(model), models_dir)
            if draft:
                cmd.extend(["--model-draft", draft])
                print(f"Draft:    {Path(draft).name} (auto-selected)")
            else:
                print("WARNING: --speculative set but no draft model found. Falling back to ngram.")
        except ImportError:
            print("WARNING: Could not import speculative module")

    print(f"Model:    {model}")
    print(f"Server:   http://{args.host}:{args.port}")
    print(f"Threads:  {threads}")
    print(f"Context:  {args.ctx_size}")
    print(f"Parallel: {args.parallel} slots")
    if kv_quant and kv_quant != "f16":
        print(f"KV Cache: {kv_quant}")
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
    health_port = args.port
    if wait_for_health(args.host, health_port):
        print("Server is ready!")
    else:
        print("WARNING: Server did not become healthy within 30s")

    # Start proxy if requested
    use_proxy = getattr(args, "proxy", False)
    max_concurrent = getattr(args, "max_concurrent", 16)
    if use_proxy:
        try:
            from engine.serving.config import ServingConfig
            from engine.serving.proxy import ServingProxy

            proxy_port = args.port + 1
            proxy_config = ServingConfig(
                max_concurrent=max_concurrent,
                kv_cache_type=kv_quant or "f16",
            )
            upstream = f"http://{args.host}:{args.port}"
            proxy = ServingProxy(upstream, proxy_config)
            print(f"\nProxy:    http://{args.host}:{proxy_port} (max {max_concurrent} concurrent)")
            proxy.run(host=args.host, port=proxy_port)
        except ImportError as e:
            print(f"WARNING: Could not start proxy: {e}")
            proc.wait()
    else:
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
    parser.add_argument("--kv-quant", default=None, choices=["f16", "q8_0", "q4_0"],
                        help="KV cache quantization type (default: f16)")
    parser.add_argument("--speculative", action="store_true", help="Enable speculative decoding")
    parser.add_argument("--draft-model", default=None, help="Path to draft model for speculation")
    parser.add_argument("--max-concurrent", type=int, default=16, help="Max concurrent requests (default: 16)")
    parser.add_argument("--proxy", action="store_true", help="Enable S2O admission control proxy")
    args = parser.parse_args()
    serve(args)


if __name__ == "__main__":
    main()
