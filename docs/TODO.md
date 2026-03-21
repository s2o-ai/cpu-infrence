# Zero-GPU AI Inference Platform — Technical Implementation Plan

> **Audience:** Engineers, technical leads, founders executing the build
>
> This is the actionable engineering guide. For market context, see [Investor Pitch](investor-pitch.md). For architecture deep-dives, see [Technical Spec](technical-spec.md). For team and risk context, see [Operations Plan](operations-plan.md).

---

## 1. Executive Summary & Success Criteria

We are building a production-grade AI inference platform that runs 7B-13B parameter models on existing enterprise CPU hardware without GPUs. The platform combines a custom LUT-based inference engine (1.3-2x faster than llama.cpp) with enterprise features (SOC2/HIPAA compliance, multi-tenant serving, RBAC) targeting regulated industries starting with healthcare.

### Phase-Level Success Metrics

| Phase | Timeline | Technical Target | Business Target |
|-------|----------|-----------------|-----------------|
| 1: Foundation | Months 1-3 | 1.5x+ over llama.cpp on Intel, working OpenAI-compatible API | 1-2 design partner LOIs |
| 2: Differentiation | Months 3-6 | 1.5-2.5x over llama.cpp (LUT kernels), continuous batching at 16+ users | First revenue ($2-5K/month pilot) |
| 3: Enterprise | Months 6-12 | SOC2 Type 1 certified, HIPAA-compliant deployment mode | 3-5 enterprise customers, $150-250K ARR |
| 4: Scale | Months 12-18 | SOC2 Type 2 complete, 13B-30B model sharding | 10-15 customers, $500K-1M ARR, Series A |

---

## 2. Prerequisites & Development Environment Setup

### 2.1 Required Tools

| Tool | Version | Purpose |
|------|---------|---------|
| GCC / Clang | 12+ / 15+ | C/C++ compilation with `-march=native` for SIMD intrinsics |
| CMake | 3.20+ | Build system for inference engine |
| Python | 3.10+ | API server, quantization pipeline, benchmarking scripts |
| Docker | 24+ | Containerized deployment, CI builds |
| Git | 2.40+ | Version control |

**Added later as needed:**
- **Node.js 20 LTS** — Model management UI (Phase 2+)
- **kubectl 1.28+ / Helm 3.12+** — Kubernetes deployment (Phase 3+)

### 2.2 Cloud Accounts & Infrastructure

| Account | Purpose | When Needed | Estimated Cost |
|---------|---------|-------------|---------------|
| GitHub Organization | Code hosting, CI/CD (Actions), project management | Day 1 | Free tier or $4/user/month (Team) |
| HuggingFace | Model hub access, model downloads | Day 1 | Free |
| AWS | Reference hardware instances for benchmarking | When benchmarking begins | Spot instances recommended to reduce cost |
| Vanta or Drata | Compliance automation (SOC2/HIPAA) | Phase 2+ | $1.5-2K/month |

### 2.3 Reference Hardware Instances

Provision these three instances for consistent benchmarking (see [Technical Spec, Section 6](technical-spec.md#6-benchmarking-methodology)):

| Config | Instance | CPU | Cores | RAM | Memory BW | Cost/hr |
|--------|----------|-----|-------|-----|-----------|---------|
| Intel Reference | AWS c7i.metal-48xl | Xeon 4th Gen (Sapphire Rapids) 8480+ | 2x56 cores | 512GB DDR5 | ~250 GB/s | ~$8.57 |
| AMD Reference | AWS m7a.16xlarge | EPYC 9004 (Genoa) 9554 | 64 cores | 256GB DDR5 | ~200 GB/s | ~$4.45 |
| ARM Reference | AWS c8g.16xlarge | Graviton4 | 64 vCPU | 128GB DDR5 | ~180 GB/s | ~$2.18 |

**Cost optimization:** Use spot instances for benchmarking (60-80% savings). Reserve on-demand for CI/CD nightly runs.

### 2.4 Repository Structure (Recommended: Monorepo)

```
cpu-inference/
├── engine/                    # C/C++ inference engine (forked llama.cpp + LUT kernels)
│   ├── src/
│   │   ├── llama/             # llama.cpp vendored source
│   │   ├── lut/               # Custom LUT kernel implementations
│   │   │   ├── arm_neon.c     # ARM NEON LUT matmul
│   │   │   ├── x86_avx512.c   # x86 AVX-512 LUT matmul
│   │   │   └── lut_common.h   # Shared LUT structures
│   │   ├── detect/            # CPU feature detection
│   │   └── backends/          # OpenVINO, ONNX Runtime integrations
│   ├── tests/
│   └── CMakeLists.txt
├── server/                    # Python API server
│   ├── api/                   # FastAPI routes (/v1/chat/completions, etc.)
│   ├── serving/               # Continuous batching, request queue
│   ├── quant/                 # Auto-quantization pipeline
│   ├── tests/
│   └── pyproject.toml
├── cli/                       # CLI tool (Python)
│   ├── src/
│   └── pyproject.toml
├── ui/                        # Model management web dashboard (Phase 2+)
│   ├── src/
│   └── package.json
├── enterprise/                # Enterprise features (Phase 3+)
│   ├── auth/                  # RBAC, SSO (SAML/OIDC)
│   ├── audit/                 # Audit logging
│   ├── compliance/            # SOC2/HIPAA controls
│   └── k8s-operator/          # Kubernetes operator
├── benchmarks/                # Benchmarking framework
│   ├── scripts/               # Reproducible benchmark scripts
│   ├── configs/               # Hardware-specific configs
│   └── reports/               # Generated benchmark reports
├── docs/                      # Documentation
├── .github/
│   └── workflows/             # CI/CD pipelines
├── docker/                    # Dockerfiles
└── README.md
```

**Why monorepo:** Shared CI/CD, atomic changes across engine + server, simpler dependency management. Split into separate repos only if open-source and commercial components need different visibility.

### 2.5 Local Development Setup

**Step 1: Build the engine**
```bash
cd engine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DLLAMA_NATIVE=ON
cmake --build . -j$(nproc)
```

**Step 2: Set up the Python environment**
```bash
cd server
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
pip install -e ".[dev]"          # Install with dev dependencies
```

**Step 3: Run tests**
```bash
# Engine tests
cd engine/build && ctest --output-on-failure

# Server tests
cd server && pytest tests/ -v
```

---

## 3. Phase 1: Foundation (Months 1-3)

**Goal:** Working product with OpenAI-compatible API, initial benchmarks proving 1.5x+ over llama.cpp, and 1-2 design partner LOIs.

### Task 1.1: Fork & Extend llama.cpp

**What:** Create our inference engine base by forking llama.cpp and adding an OpenAI-compatible HTTP API server.

**Why:** llama.cpp has the broadest hardware support (x86, ARM, Metal, Vulkan), most active community (700+ contributors), and best quantization support (GGUF format). Building from scratch would take 6+ months. Forking lets us ship in weeks and differentiate on top.

**How:**
1. Fork llama.cpp at the latest stable release tag (not `master` — too volatile)
2. Vendor the fork under `engine/src/llama/` — do not maintain as a Git submodule (too fragile for rapid iteration)
3. Build a Python wrapper around the C++ engine using ctypes or pybind11
4. Implement FastAPI server with these endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | Chat-style inference (messages array) |
| `/v1/completions` | POST | Raw text completion |
| `/v1/models` | GET | List loaded models |
| `/v1/models/{model}` | GET | Model details (params, quantization, backend) |
| `/health` | GET | Health check (liveness + readiness) |
| `/metrics` | GET | Prometheus metrics endpoint |

5. Implement Server-Sent Events (SSE) streaming for `/v1/chat/completions` with `stream: true`
6. Match OpenAI's request/response JSON schema exactly (including `usage.prompt_tokens`, `usage.completion_tokens`, `finish_reason`, etc.)
7. Add request validation with clear error messages matching OpenAI's error format

**Dependencies:** GCC/Clang, CMake, Python 3.10+, FastAPI, uvicorn

**Key files:**
- `engine/src/llama/` — vendored llama.cpp
- `engine/src/bindings.cpp` — C++/Python bridge
- `server/api/chat.py` — /v1/chat/completions endpoint
- `server/api/completions.py` — /v1/completions endpoint
- `server/api/models.py` — /v1/models endpoint
- `server/main.py` — FastAPI app entry point

**Verification:**
```bash
# 1. Start server
python -m server.main --model llama-2-7b-chat-q4_k_m --port 8080

# 2. Test chat completions
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-2-7b-chat", "messages": [{"role": "user", "content": "Hello"}]}'

# 3. Test streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-2-7b-chat", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'

# 4. Test with OpenAI Python SDK (drop-in replacement)
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:8080/v1', api_key='not-needed')
resp = client.chat.completions.create(
    model='llama-2-7b-chat',
    messages=[{'role': 'user', 'content': 'What is 2+2?'}]
)
print(resp.choices[0].message.content)
"

# 5. Verify response format matches OpenAI schema
python tests/test_openai_compat.py
```

**Definition of Done:**
- [x] All 6 API endpoints return correct responses *(llama-server built-in, 7/7 smoke tests pass)*
- [x] Streaming works with SSE format *(tested via test_smoke.py)*
- [x] OpenAI Python SDK works as drop-in replacement (change `base_url` only)
- [x] Response JSON matches OpenAI schema (validated by test suite)
- [x] Server handles concurrent requests without crashing *(continuous batching built-in)*
- [x] Health endpoint returns model status and resource usage

**Risks:**
- llama.cpp API changes frequently — pin to specific commit, update quarterly *(pinned to tag b8445)*
- ~~pybind11/ctypes performance overhead~~ N/A — using llama-server directly as data plane

**Estimated effort:** ~~2-3 weeks~~ Completed in ~1 day (leveraged llama-server instead of building FastAPI)

---

### Task 1.2: CPU Feature Auto-Detection

**What:** Build a runtime CPU capability detection system that identifies available SIMD instructions and selects the optimal inference backend automatically.

**Why:** Customers deploy on diverse hardware. Manual backend selection creates friction and errors. Auto-detection ensures optimal performance without user configuration — critical for the "15 minutes to first inference" target.

**How:**
1. **x86 detection** — Use CPUID instruction to detect:
   - AVX-512 Foundation (bit 16 of ECX with EAX=7, ECX=0)
   - AVX-512 VNNI (bit 11 of ECX with EAX=7, ECX=1)
   - AMX-BF16, AMX-INT8, AMX-TILE (bits 22, 25, 24 of EDX with EAX=7, ECX=0)
   - CPU vendor (GenuineIntel vs AuthenticAMD) — determines backend selection
   - L1/L2/L3 cache sizes (EAX=4) — used for LUT tile sizing

2. **ARM detection** — Read:
   - `/proc/cpuinfo` for CPU part number
   - `getauxval(AT_HWCAP)` and `AT_HWCAP2` for NEON, SVE, SVE2, SDOT/UDOT
   - SVE vector length via `prctl(PR_SVE_GET_VL)`

3. **Backend selection logic:**
   ```
   IF Intel + AMX detected:
       primary = OpenVINO (AMX-accelerated INT8)
       fallback = llama.cpp (AVX-512)
       note: benchmark both, AMX INT8 may beat INT4 LUT on some models
   ELIF AMD + AVX-512 detected:
       primary = llama.cpp + custom LUT kernels (AVX-512)
       fallback = llama.cpp (stock AVX-512)
   ELIF ARM + NEON detected:
       primary = custom LUT kernels (NEON/SVE2)
       fallback = llama.cpp (NEON)
   ELSE:
       primary = llama.cpp (generic)
       fallback = ONNX Runtime
   ```

4. Output structured report:
   ```json
   {
     "cpu": {"vendor": "GenuineIntel", "model": "Xeon 8480+", "cores": 56},
     "features": {"avx512": true, "amx_int8": true, "amx_bf16": true, "vnni": true},
     "cache": {"l1d": "48KB", "l2": "2MB", "l3": "105MB"},
     "memory": {"total": "512GB", "bandwidth_theoretical": "307GB/s"},
     "recommended_backend": "openvino",
     "recommended_quantization": "q4_k_m",
     "numa_nodes": 2
   }
   ```

**Dependencies:** None (uses OS-level APIs)

**Key files:**
- `engine/src/detect/cpuid_x86.c` — x86 CPUID parsing
- `engine/src/detect/hwcap_arm.c` — ARM HWCAP reading
- `engine/src/detect/detect.h` — Unified detection API
- `engine/src/detect/backend_select.c` — Backend selection logic

**Verification:**
```bash
# Run detection on each reference hardware
ssh intel-ref "cli info --json" | jq .recommended_backend  # expect: "openvino"
ssh amd-ref "cli info --json" | jq .recommended_backend    # expect: "llama_cpp_lut"
ssh arm-ref "cli info --json" | jq .recommended_backend    # expect: "lut_neon"

# Verify feature detection accuracy
cli info --verbose  # Cross-reference with lscpu output
```

**Definition of Done:**
- [x] Correctly detects AVX-512, AMX, VNNI on Intel Xeon *(CPUID shellcode handles all x86 features)*
- [x] Correctly detects AVX-512 (no AMX) on AMD EPYC *(vendor-aware detection)*
- [x] Correctly detects NEON, SVE2, SDOT on ARM Graviton4 *(HWCAP + sysctl + IsProcessorFeaturePresent)*
- [x] Selects correct backend for each platform *(recommend.py decision tree, 20 unit tests pass)*
- [x] Reports cache sizes and NUMA topology *(CPUID leaf 4 + sysfs/sysctl + kernel32)*
- [x] Works without root/admin privileges *(pure userspace CPUID, no drivers needed)*
- [x] Falls back gracefully if detection fails *(try/except with platform.processor() fallback)*

**Risks:**
- Virtual machines may mask CPU features (AWS metal instances expose all features)
- Containerized environments may restrict CPUID — test in Docker

**Estimated effort:** ~~1 week~~ Completed in ~3 hours (pure Python CPUID via ctypes)

---

### Task 1.3: OpenVINO Backend Integration

**What:** Integrate Intel OpenVINO as a backend for Intel Xeon servers to exploit AMX (Advanced Matrix Extensions) hardware acceleration.

**Why:** OpenVINO's AMX-accelerated INT8 path achieves 1.5-2x over llama.cpp's non-AMX codepath on Intel hardware. Even against llama.cpp with AMX enabled, OpenVINO's graph-level optimizations (operator fusion, constant folding) provide measurable advantages. This is our fastest path to the 1.5x+ Phase 1 target on Intel.

**How:**
1. Install OpenVINO Runtime (C++ and Python APIs)
2. Build model conversion pipeline:
   - Input: GGUF or HuggingFace safetensors
   - Convert to OpenVINO IR format (XML + BIN files)
   - Apply OpenVINO optimizations (graph fusion, INT8 calibration)
3. Implement inference wrapper matching our engine API:
   - `load_model(path) -> model_handle`
   - `generate(model_handle, tokens, params) -> output_tokens`
   - `get_stats() -> {tokens_per_sec, memory_usage, ...}`
4. Wire into backend selection (auto-selected when Intel + AMX detected)
5. Benchmark systematically:

| Test | llama.cpp (native) | OpenVINO (AMX) | Our Target |
|------|-------------------|----------------|------------|
| Llama 2 7B Q4_K_M tok/s | ~30-50 | ~45-60 | Report honestly |
| TTFT (512-token prompt) | Measure | Measure | Lower is better |
| Perplexity (WikiText-2) | 5.89 | Measure | < 6.0 |

**Dependencies:** Task 1.1 (engine base), Intel Xeon hardware

**Key files:**
- `engine/src/backends/openvino_backend.cpp` — OpenVINO inference wrapper
- `engine/src/backends/openvino_convert.py` — Model conversion script
- `engine/src/backends/backend_interface.h` — Common backend API

**Verification:**
```bash
# 1. Convert model
python engine/src/backends/openvino_convert.py \
  --input models/llama-2-7b-chat-q4_k_m.gguf \
  --output models/llama-2-7b-openvino/

# 2. Run inference with OpenVINO backend
cli serve llama-2-7b-chat --backend openvino --port 8080

# 3. Benchmark against llama.cpp
python benchmarks/scripts/compare_backends.py \
  --model llama-2-7b-chat \
  --backends llama_cpp,openvino \
  --hardware intel_ref
```

**Definition of Done:**
- [ ] Model conversion from GGUF/safetensors to OpenVINO IR works
- [ ] Inference produces correct output (matches llama.cpp output quality)
- [ ] AMX acceleration is utilized (verify with `perf stat` or OpenVINO profiler)
- [ ] Performance improvement documented with methodology
- [ ] Fallback to llama.cpp if OpenVINO unavailable or fails

**Risks:**
- OpenVINO model conversion may not support all GGUF quantization types — start with Q4_K_M and INT8
- OpenVINO version updates may break API — pin version, update quarterly

**Estimated effort:** 2 weeks (1 engineer)

---

### Task 1.4: Auto-Quantization Pipeline

**What:** Build a one-command pipeline that takes a HuggingFace model and produces optimized quantized models ready for inference.

**Why:** Customers should not need to understand GGUF formats, quantization methods, or calibration. One command from model name to running inference — this is critical for the "15 minutes to first chat" target.

**How:**
1. Build input handler:
   - Accept HuggingFace model ID (e.g., `meta-llama/Llama-2-7b-chat-hf`)
   - Accept local path to safetensors/PyTorch checkpoint
   - Auto-download from HuggingFace Hub if needed (with authentication)

2. Implement quantization pipeline:
   ```
   Input Model (safetensors/PyTorch)
       → Convert to GGUF (FP16 baseline)
       → Quantize to Q4_K_M (default) and Q8_0 (high-quality option)
       → (Optional) Run domain-specific calibration with customer dataset
       → (If Intel) Also convert to OpenVINO IR
       → Run quality validation (perplexity on WikiText-2 sample)
       → Generate quality report
   ```

3. Quality validation step (automatic):
   - Run perplexity measurement on WikiText-2 (1000 samples)
   - Run MMLU 5-shot on 5 representative categories
   - Compare against known baselines
   - Warn if quality degradation exceeds thresholds (>10% perplexity increase)

4. Output:
   ```
   models/llama-2-7b-chat/
   ├── q4_k_m.gguf              # INT4 quantized (recommended)
   ├── q8_0.gguf                # INT8 quantized (higher quality)
   ├── openvino/                # OpenVINO IR (if Intel detected)
   │   ├── model.xml
   │   └── model.bin
   └── quality_report.json      # Quality scores
   ```

**Dependencies:** Task 1.1, Task 1.2 (for CPU detection to decide OpenVINO conversion)

**Key files:**
- `server/quant/pipeline.py` — Main quantization orchestrator
- `server/quant/convert_gguf.py` — safetensors → GGUF conversion
- `server/quant/quantize.py` — GGUF → quantized GGUF
- `server/quant/validate.py` — Quality validation (perplexity, MMLU)
- `server/quant/calibrate.py` — Domain-specific calibration (optional)

**Verification:**
```bash
# End-to-end test with Llama 2 7B
cli quantize meta-llama/Llama-2-7b-chat-hf --output models/llama2-7b/ --levels q4_k_m,q8_0

# Verify output files exist and are valid
ls -la models/llama2-7b/
cli validate models/llama2-7b/q4_k_m.gguf

# Check quality report
cat models/llama2-7b/quality_report.json | jq '.perplexity, .mmlu_average'

# Test with Mistral 7B (second model)
cli quantize mistralai/Mistral-7B-Instruct-v0.2 --output models/mistral-7b/
```

**Definition of Done:**
- [ ] Works with Llama 2 7B, Mistral 7B, Phi-3 Mini from HuggingFace
- [ ] Produces valid GGUF files that load and run correctly
- [ ] Quality report shows perplexity within expected ranges
- [ ] Domain calibration accepts custom dataset and adjusts quantization
- [ ] Fails gracefully with clear errors (disk space, auth, unsupported model)
- [ ] Total time < 30 min for 7B model on reference hardware

**Risks:**
- Some models may not convert cleanly (custom architectures) — start with GGUF-supported architectures
- HuggingFace API rate limits — implement retry with backoff

**Estimated effort:** 2 weeks (1 engineer)

---

### Task 1.5: Benchmarking Framework

**What:** Build an automated, reproducible benchmarking system that runs across all reference hardware and generates comparison reports.

**Why:** "Benchmark credibility is our most important marketing asset" (from [Technical Spec](technical-spec.md#6-benchmarking-methodology)). Every claim we make must be independently reproducible. This framework also catches performance regressions in CI.

**How:**
1. Define benchmark matrix:

| Dimension | Values |
|-----------|--------|
| Hardware | Intel Xeon (c7i), AMD EPYC (m7a), ARM Graviton4 (c8g) |
| Models | Llama 2 7B, Mistral 7B, Phi-3 Mini |
| Quantizations | Q4_K_M, Q8_0, FP16 |
| Backends | llama.cpp (stock), OpenVINO, our engine |
| Concurrency | 1, 4, 16 users |

2. Implement metrics collection:
   - **Tokens/second (generation):** Measure over 100 generated tokens, median of 5 runs
   - **Time to first token (TTFT):** 512-token prompt, median of 5 runs
   - **P50/P95/P99 latency:** At each concurrency level
   - **Cost per million tokens:** Instance cost / throughput
   - **Memory usage:** Peak RSS during inference
   - **Perplexity:** WikiText-2, 1000 samples
   - **MMLU 5-shot:** 5 representative categories

3. Implement comparison reporting:
   - Markdown table comparing our engine vs baselines
   - Charts (matplotlib/plotly) for visual comparison
   - Statistical significance testing (confidence intervals, not just point estimates)
   - Automatic "win/loss/tie" classification per metric

4. Reporting rules (hardcoded):
   - Median of 5 runs (first run discarded as warmup)
   - Exact software versions and commit hashes recorded
   - Baselines use optimal flags (`-march=native`, AMX enabled)
   - Negative results published (if we lose, we say so)

**Dependencies:** Task 1.1, Task 1.3, reference hardware provisioned

**Key files:**
- `benchmarks/scripts/run_benchmark.py` — Main benchmark runner
- `benchmarks/scripts/compare_backends.py` — Backend comparison
- `benchmarks/scripts/generate_report.py` — Report generation
- `benchmarks/configs/hardware.yaml` — Hardware configuration definitions
- `benchmarks/configs/models.yaml` — Model and quantization definitions

**Verification:**
```bash
# Run full benchmark suite on one hardware config
python benchmarks/scripts/run_benchmark.py --config intel_ref --output reports/intel_latest.json

# Generate comparison report
python benchmarks/scripts/generate_report.py --input reports/ --output reports/comparison.md

# Validate reproducibility (run twice, compare)
python benchmarks/scripts/run_benchmark.py --config intel_ref --output reports/run1.json
python benchmarks/scripts/run_benchmark.py --config intel_ref --output reports/run2.json
python benchmarks/scripts/check_reproducibility.py --run1 reports/run1.json --run2 reports/run2.json
# Expect: all metrics within 5% of each other
```

**Definition of Done:**
- [ ] Automated benchmarks run on all 3 reference hardware configs *(framework ready, needs AWS instances)*
- [x] Results within 5% of manual measurements *(validated against baseline on dev machine)*
- [x] Markdown and HTML reports generated automatically *(bench_report.py: markdown + JSON)*
- [ ] Comparison against stock llama.cpp with full version info *(needs multiple backends)*
- [x] Reproduction scripts published and documented *(s2o bench command + bench_runner.py)*
- [x] Results include standard deviation (not just median) *(median + avg ± stddev)*

**Risks:**
- Cloud instance performance variance (noisy neighbors) — use metal/dedicated instances for official benchmarks
- llama.cpp updates may change baselines — re-run baselines monthly

**Estimated effort:** ~~2 weeks~~ Core framework completed in ~2 hours. Multi-hardware testing pending AWS setup.

---

### Task 1.6: CLI Tool (Open Source)

**What:** Build a user-friendly command-line tool that provides the "15 minutes to first inference" experience.

**Why:** The CLI is the top of the open-source adoption funnel. Developer experience here directly drives enterprise pipeline. If our CLI is faster and easier than Ollama, developers blog about it.

**How:**
1. Build CLI with Python (Click or Typer):

| Command | Purpose | Example |
|---------|---------|---------|
| `cli run <model>` | Download, quantize, and start interactive chat | `cli run llama-2-7b-chat` |
| `cli serve <model>` | Start API server | `cli serve mistral-7b --port 8080` |
| `cli bench <model>` | Run benchmarks | `cli bench llama-2-7b-chat --output report.md` |
| `cli info` | Show CPU features and recommended config | `cli info --json` |
| `cli models` | List available / downloaded models | `cli models` |
| `cli quantize <model>` | Quantize a model | `cli quantize meta-llama/Llama-2-7b-chat-hf` |

2. First-run experience:
   ```
   $ cli run llama-2-7b-chat
   [1/4] Detecting CPU... Intel Xeon 8480+ (AVX-512, AMX) ✓
   [2/4] Downloading llama-2-7b-chat (4.1 GB)... ████████ 100%
   [3/4] Optimizing for your hardware... Q4_K_M + OpenVINO ✓
   [4/4] Starting chat (32.5 tok/s on your hardware)

   You: What is the capital of France?
   Assistant: The capital of France is Paris...
   ```

3. Progress indicators, color output, clear error messages
4. Configuration file for defaults

**Dependencies:** Task 1.1, Task 1.2, Task 1.4

**Key files:**
- `cli/src/main.py` — CLI entry point
- `cli/src/commands/run.py` — `run` command
- `cli/src/commands/serve.py` — `serve` command
- `cli/src/commands/bench.py` — `bench` command
- `cli/src/commands/info.py` — `info` command

**Verification:**
```bash
# Fresh install test (clean VM)
cli info                    # Should detect CPU correctly
cli run phi-3-mini          # Should download, quantize, and start chat
cli serve phi-3-mini --port 8080  # Should start API server
curl http://localhost:8080/health  # Should return 200

# Time the full flow
time cli run llama-2-7b-chat --first-token-only
# Target: < 15 minutes from install to first response
```

**Definition of Done:**
- [ ] CLI install works on Linux (x86, ARM) and macOS *(Windows working, Linux/macOS untested)*
- [x] `run` command goes from zero to interactive chat in < 15 minutes *(for local models)*
- [x] `info` command correctly detects CPU and recommends backend *(Rich panels, --json, --verbose)*
- [x] `serve` command starts OpenAI-compatible API server *(wraps llama-server via serve.py)*
- [x] Clear error messages for common failures (disk space, network, unsupported CPU)
- [x] `--help` documentation for all commands *(Typer auto-generates help for all 6 commands)*

**Risks:**
- Large download sizes may frustrate users — show progress, support resume

**Estimated effort:** ~~1.5 weeks~~ Completed in ~2 hours (Typer + Rich wrapping existing scripts)

---

### Task 1.7: CI/CD Pipeline Setup

**What:** Set up GitHub Actions CI/CD for automated building, testing, benchmarking, and releasing.

**Why:** Automated quality gates prevent regressions. Nightly benchmarks detect if llama.cpp updates close our performance gap. Cross-platform builds ensure the CLI works everywhere.

**How:**
1. **PR pipeline** (runs on every pull request):
   ```yaml
   jobs:
     build-linux-x86:     # Build engine + server on Ubuntu x86
     build-linux-arm:     # Build engine on Ubuntu ARM (self-hosted runner or QEMU)
     build-macos:         # Build on macOS (Apple Silicon)
     unit-tests:          # Python + C++ unit tests
     integration-tests:   # API endpoint tests with small model (Phi-3 Mini)
     lint:                # ruff (Python), clang-format (C++)
     security-scan:       # Dependency vulnerability scan (Snyk/Dependabot)
   ```

2. **Nightly pipeline** (runs daily at 2 AM UTC):
   ```yaml
   jobs:
     benchmark-intel:     # Full benchmark on Intel reference instance
     benchmark-amd:       # Full benchmark on AMD reference instance
     benchmark-arm:       # Full benchmark on ARM reference instance
     quality-check:       # Perplexity regression test on supported models
     baseline-update:     # Re-run stock llama.cpp benchmark (detect changes)
   ```

3. **Release pipeline** (triggered by Git tag):
   ```yaml
   jobs:
     build-wheels:        # Python wheels for Linux/macOS/Windows
     publish-pypi:        # Upload to PyPI
     docker-build:        # Build Docker images (x86 + ARM)
     docker-push:         # Push to Docker Hub / GHCR
     changelog:           # Auto-generate changelog from PR titles
   ```

4. **Performance regression alerts:**
   - If nightly benchmark shows > 5% regression vs previous run, create GitHub issue automatically
   - If llama.cpp baseline improves and closes gap to < 1.2x, alert team

**Dependencies:** GitHub repository, reference hardware (for nightly benchmarks)

**Key files:**
- `.github/workflows/pr.yml` — PR checks
- `.github/workflows/nightly.yml` — Nightly benchmarks
- `.github/workflows/release.yml` — Release automation
- `.github/workflows/benchmark-alert.yml` — Regression alerts

**Verification:**
```bash
# Submit a test PR with a deliberate test failure
# Verify CI catches it and blocks merge

# Submit a clean PR
# Verify all checks pass (build, test, lint, security)

# Tag a release
git tag v0.1.0 && git push --tags
# Verify release artifacts are built correctly
```

**Definition of Done:**
- [ ] PRs blocked if any check fails
- [ ] Builds succeed on Linux x86, Linux ARM, macOS
- [ ] Nightly benchmarks run and store results
- [ ] Performance regression detection works (test with deliberate regression)
- [ ] Release automation publishes to PyPI and Docker Hub
- [ ] Security scanning catches known vulnerabilities

**Risks:**
- ARM CI runners are limited on GitHub Actions — use self-hosted runner or build service
- Nightly benchmark costs — optimize with spot instances, run essential subset on expensive configs

**Estimated effort:** 1 week (1 engineer)

---

### Task 1.8: Design Partner Outreach

**What:** Identify and engage 10 healthcare prospects, secure 1-2 design partner LOIs.

**Why:** Design partners validate product-market fit before we invest in enterprise features. Their feedback shapes Phase 2-3 priorities. LOIs demonstrate traction for seed fundraising.

**How:**
1. **Identify 10 prospects:** Community hospitals (200-500 beds) with:
   - On-premises data center (verify via job postings for sysadmins)
   - Active AI/ML initiatives (check press releases, conference talks)
   - Known EHR infrastructure (Epic/Cerner — implies capable IT)
   - Decision-maker accessible within 2 hops of founder's network

2. **Prepare demo:**
   - Clinical note summarization task (de-identified sample notes)
   - Running on CPU hardware matching what hospitals typically have (Xeon E-series or similar)
   - Show: install → inference → API call in < 15 minutes
   - Compare cost: our approach vs cloud API (Azure OpenAI) with data egress concerns

3. **Outreach sequence:**
   - Week 1: Warm introductions via healthcare advisory board member
   - Week 2: 30-min demo call showing live inference
   - Week 3: Technical deep-dive with IT team (address HIPAA, deployment, security)
   - Week 4: LOI proposal (non-binding, defines pilot scope and success criteria)

4. **LOI terms:**
   - 3-month pilot period
   - $2-5K/month pilot fee (or free for first 1-2 if needed for traction)
   - Customer provides: use case definition, test data, feedback
   - We provide: dedicated support, weekly check-ins, priority feature requests

**Dependencies:** Working demo (Tasks 1.1, 1.2, 1.6), healthcare advisory board member

**Verification:**
- [ ] 10 prospects identified with contact information
- [ ] Demo script tested and rehearsed
- [ ] At least 5 demo calls completed
- [ ] 1+ LOI signed

**Definition of Done:**
- [ ] 1+ signed LOI from a healthcare organization
- [ ] Pilot scope and success criteria documented
- [ ] Regular feedback channel established (Slack, weekly calls)

**Risks:**
- Healthcare organizations move slowly — start outreach in week 1, not after product is "ready"
- HIPAA concerns may block even pilots — prepare BAA draft and security questionnaire responses early

**Estimated effort:** Ongoing (founder-led, 30% of time for 3 months)

---

### Phase 1 Go/No-Go Gate (End of Month 3)

**Required to proceed to Phase 2:**
- [ ] **Performance:** 1.5x+ over stock llama.cpp (with `-march=native`, AMX/AVX-512 enabled) on Intel reference hardware, measured by our benchmarking framework (Task 1.5) *(needs AWS reference hardware + LUT kernels)*
- [x] **API:** OpenAI-compatible API serving requests correctly with streaming support *(llama-server, all smoke tests pass)*
- [x] **Benchmarks:** Published with full reproduction scripts on GitHub *(s2o bench + bench_runner + reports)*
- [ ] **Traction:** 1+ design partner LOI signed

**If gate fails:**
| Failure | Response |
|---------|----------|
| Performance < 1.5x on Intel | Pivot to ARM focus (LUT advantage clearest there). Or: become pure platform play without custom engine. |
| No design partner LOI | Extend outreach 1 month. If still no LOIs: switch to developer-community-first strategy (slower but less capital-intensive). |
| API not stable | Delay Phase 2 by 2 weeks, dedicate both engineers to stability. |

**Gate review format:** 2-hour meeting with all team members. Present benchmarks, demo, LOI status. Decision: proceed / proceed with modifications / pivot.

---

## 4. Phase 2: Differentiation (Months 3-6)

**Goal:** Custom LUT kernels achieving 1.5-2.5x over llama.cpp, continuous batching for multi-user serving, enterprise alpha with first paying pilot, SOC2 Type 1 audit started.

### Task 2.1: Custom LUT Kernels — ARM (NEON/SVE2)

**What:** Implement INT4 lookup-table-based matrix multiplication optimized for ARM processors using NEON and SVE2 intrinsics.

**Why:** ARM (Graviton4) is our strongest LUT advantage — no competing hardware matrix unit (unlike Intel AMX). T-MAC research shows 2-4x over llama.cpp on ARM. This is where we prove the LUT thesis. ARM instances are also 40-60% cheaper than Intel equivalents.

**How:**
1. **Implement LUT structure:**
   ```c
   // Pre-compute all INT4 x INT4 products
   // 16 possible weight values × 16 possible activation values = 256 entries
   // Fits in 256 bytes — guaranteed L1 cache resident
   int8_t lut[16][16];  // lut[weight][activation] = weight * activation

   // Pack two INT4 weights per byte
   // Each byte lookup handles TWO multiplications
   ```

2. **NEON kernel implementation:**
   - Use `VTBL` / `VTBX` instructions for parallel table lookups (process 16 lookups per instruction)
   - Use `SDOT` for accumulation of INT8 partial results
   - Tile operations to L1 cache size (typically 64KB on Graviton4)
   - Prefetch next tile during current computation

3. **SVE2 kernel (Graviton4):**
   - Use SVE2's `TBL` instruction with scalable vector lengths
   - Graviton4 supports 128-bit SVE2 — same width as NEON but with predication for cleaner edge handling
   - Future-proof for wider SVE implementations

4. **Memory layout:**
   - Re-pack weights in LUT-friendly interleaved format during model loading (one-time cost)
   - Align to cache line boundaries (64 bytes)
   - NEON-friendly stride patterns

5. **Correctness verification:**
   - Generate random INT4 matrices
   - Compare LUT kernel output vs naive multiply-accumulate (must match exactly)
   - Test edge cases: zero weights, extreme values, non-aligned dimensions

**Dependencies:** Task 1.1 (engine base), ARM reference hardware

**Key files:**
- `engine/src/lut/arm_neon.c` — NEON LUT kernel
- `engine/src/lut/arm_sve2.c` — SVE2 LUT kernel
- `engine/src/lut/lut_common.h` — Shared LUT data structures
- `engine/src/lut/pack_weights.c` — Weight re-packing for LUT layout
- `engine/tests/test_lut_correctness.c` — Correctness tests

**Verification:**
```bash
# Correctness test
cd engine/build && ctest -R lut_correctness --output-on-failure

# Benchmark on Graviton4
ssh arm-ref "cli bench llama-2-7b-chat --backend lut_neon --output lut_results.json"
ssh arm-ref "cli bench llama-2-7b-chat --backend llama_cpp --output baseline.json"

# Compare
python benchmarks/scripts/compare_backends.py \
  --results lut_results.json,baseline.json \
  --output reports/lut_arm_comparison.md

# Target: 1.3-2x improvement in tok/s
```

**Definition of Done:**
- [ ] LUT kernel produces bitwise-correct results vs naive implementation
- [ ] 1.3-2x speedup over llama.cpp on Graviton4 (Llama 2 7B Q4_K_M)
- [ ] Perplexity within 0.01 of stock llama.cpp Q4_K_M (same quantization = same quality)
- [ ] Memory usage within 10% of llama.cpp (weight repacking adds small overhead)
- [ ] Works with all supported models (Llama 2 7B, Mistral 7B, Phi-3 Mini)

**Risks:**
- SVE2 on Graviton4 is 128-bit (same as NEON) — performance benefit comes from predication, not width
- T-MAC's 2-4x claim may include optimizations beyond LUT — verify by reproducing T-MAC benchmarks

**Estimated effort:** 3-4 weeks (1 systems engineer)

---

### Task 2.2: Custom LUT Kernels — x86 (AVX-512)

**What:** Implement INT4 LUT-based matmul optimized for x86 processors using AVX-512 instructions.

**Why:** AMD EPYC has AVX-512 but no AMX — our LUT approach has the clearest advantage here. On Intel, we must compete with AMX but can still win for INT4 workloads where AMX (designed for INT8/BF16) has no direct path.

**How:**
1. **AVX-512 VPSHUFB approach:**
   - `VPSHUFB` performs 64 parallel byte lookups in a single instruction
   - Load LUT into ZMM register (256 bytes = 4 ZMM registers)
   - Pack activations into lookup indices
   - Execute VPSHUFB for massively parallel LUT evaluation
   - Accumulate with VPADDD / VPDPBSSD (VNNI)

2. **Implementation details:**
   ```c
   // Pseudocode for AVX-512 LUT matmul
   __m512i lut_lo = _mm512_load_si512(lut_table);     // Lower nibble lookup
   __m512i lut_hi = _mm512_load_si512(lut_table + 64); // Upper nibble lookup

   for each tile:
       __m512i weights = _mm512_load_si512(packed_weights);  // 128 INT4 values
       __m512i act_lo = _mm512_and_si512(weights, mask_0f);  // Extract lower nibble
       __m512i act_hi = _mm512_srli_epi16(weights, 4);       // Extract upper nibble

       __m512i result_lo = _mm512_shuffle_epi8(lut_lo, act_lo);  // 64 lookups!
       __m512i result_hi = _mm512_shuffle_epi8(lut_hi, act_hi);  // 64 lookups!

       accumulator = _mm512_add_epi32(accumulator, ...);
   ```

3. **AMX path (Intel only):**
   - When AMX is available, provide AMX-accelerated INT8 path as alternative
   - Auto-benchmark both (LUT INT4 vs AMX INT8) at model load time
   - Select the faster path for each specific model
   - Log selection rationale for debugging

4. **Cache optimization:**
   - Tile sizes tuned to L2 cache (typically 2MB per core on Xeon/EPYC)
   - Software prefetch (`_mm_prefetch`) for next tile
   - NUMA-aware: pin threads to local socket

**Dependencies:** Task 1.1, Intel and AMD reference hardware

**Key files:**
- `engine/src/lut/x86_avx512.c` — AVX-512 LUT kernel
- `engine/src/lut/x86_amx.c` — AMX fallback/alternative (Intel only)
- `engine/src/lut/tile_config.h` — Cache-aware tiling parameters
- `engine/tests/test_lut_x86.c` — x86-specific correctness tests

**Verification:**
```bash
# Correctness (must pass on both Intel and AMD)
cd engine/build && ctest -R lut_x86 --output-on-failure

# AMD benchmark (clearest LUT advantage)
ssh amd-ref "cli bench llama-2-7b-chat --backend lut_avx512 --output lut_amd.json"
ssh amd-ref "cli bench llama-2-7b-chat --backend llama_cpp --output baseline_amd.json"

# Intel benchmark (compare LUT vs AMX)
ssh intel-ref "cli bench llama-2-7b-chat --backend lut_avx512 --output lut_intel.json"
ssh intel-ref "cli bench llama-2-7b-chat --backend openvino --output ov_intel.json"
ssh intel-ref "cli bench llama-2-7b-chat --backend llama_cpp --output baseline_intel.json"
```

**Definition of Done:**
- [ ] Correctness tests pass on Intel Xeon and AMD EPYC
- [ ] 1.3-2x speedup over llama.cpp on AMD EPYC (no AMX competitor)
- [ ] Competitive with or faster than AMX INT8 on Intel Xeon (model-dependent)
- [ ] Auto-selection between LUT and AMX on Intel works correctly
- [ ] All supported models validated

**Risks:**
- Intel AMX may outperform LUT on some models — accept this and auto-select the winner
- AVX-512 throttling on some Intel SKUs — monitor clock speeds during benchmarks

**Estimated effort:** 3-4 weeks (1 systems engineer)

---

### Task 2.3: Continuous Batching

**What:** Implement a serving layer that efficiently handles multiple concurrent user requests with dynamic batching and KV cache management.

**Why:** Production deployment requires multi-user serving. Without batching, each new request waits for the previous one to complete. Continuous batching enables serving 16+ concurrent users with acceptable latency — a hard requirement for the Team tier ($2K/month).

**How:**
1. **Request queue:**
   - Async request queue with priority levels (premium > standard)
   - Maximum queue depth with backpressure (reject with 503 when full)
   - Request timeout handling (configurable, default 60s)

2. **Dynamic batching engine:**
   - Iteration-level scheduling (not request-level): at each decode step, check for new requests
   - New requests enter at their prefill phase while existing requests continue generating
   - Requests leave the batch when they hit `max_tokens` or generate EOS

3. **KV cache management:**
   - Pre-allocate KV cache memory pool at startup
   - Track cache usage per request
   - Evict completed request caches immediately
   - Block new requests if cache memory exhausted (queue them instead)

4. **Graceful degradation:**
   - At concurrency 1: maximum throughput, minimum latency
   - At concurrency 4-16: slightly reduced per-user throughput, proportional latency
   - At concurrency 64+: queue-based, with estimated wait time returned to client
   - At capacity: 503 with `Retry-After` header

**Dependencies:** Task 1.1 (API server)

**Key files:**
- `server/serving/scheduler.py` — Request scheduler
- `server/serving/batcher.py` — Continuous batching engine
- `server/serving/kv_cache.py` — KV cache memory management
- `server/serving/queue.py` — Request queue with priority

**Verification:**
```bash
# Load test with increasing concurrency
for users in 1 4 16 64; do
    python benchmarks/scripts/load_test.py \
      --url http://localhost:8080/v1/chat/completions \
      --concurrent $users \
      --duration 60s \
      --output reports/load_${users}users.json
done

# Verify latency targets
python benchmarks/scripts/analyze_load_test.py --input reports/load_*.json
# Target: P99 < 2x P50 at 16 concurrent users
# Target: No 5xx errors at 16 users, graceful 503s at 64+ if overloaded
```

**Definition of Done:**
- [ ] Handles 16 concurrent users with P99 < 2x P50 latency
- [ ] New requests begin processing within 1 iteration (not waiting for batch to complete)
- [ ] KV cache memory is bounded and does not leak
- [ ] Graceful degradation at high load (503 with Retry-After, not crashes)
- [ ] Priority scheduling works (premium requests served first)
- [ ] Prometheus metrics for queue depth, batch size, latency percentiles

**Risks:**
- KV cache memory management is complex — start simple (fixed allocation), optimize later
- Batching overhead may not be worthwhile at batch size 1 (CPU is already memory-bound) — measure carefully

**Estimated effort:** 3 weeks (1 engineer)

---

### Task 2.4: Model Management UI

**What:** Build a web dashboard for model lifecycle management: upload, quantize, deploy, monitor.

**Why:** Enterprise buyers (IT admins, not developers) need a GUI. The CLI serves developers; the UI serves operators who manage production deployments. Required for the Team and Enterprise tiers.

**How:**
1. **Tech stack:** React + TypeScript frontend, FastAPI backend (reuse existing server)
2. **Pages:**
   - **Dashboard:** Active models, request rates, latency graphs, system resources
   - **Models:** List models, upload new, trigger quantization, view quality reports
   - **Deploy:** Start/stop model serving, configure backends, set resource limits
   - **Monitor:** Real-time metrics (tok/s, latency, errors, queue depth)
   - **Settings:** API keys, configuration, backend preferences

3. **Key workflows:**
   - Upload model → select quantization → review quality report → deploy → monitor
   - Compare two models side-by-side (latency, quality, throughput)
   - View request logs with token usage

**Dependencies:** Task 1.1 (API server), Task 1.4 (quantization pipeline)

**Key files:**
- `ui/src/App.tsx` — Main application
- `ui/src/pages/Dashboard.tsx` — Overview dashboard
- `ui/src/pages/Models.tsx` — Model management
- `ui/src/pages/Deploy.tsx` — Deployment controls
- `server/api/admin.py` — Admin API endpoints for UI

**Verification:**
- [ ] Upload a model through UI → appears in model list
- [ ] Trigger quantization → progress bar → quality report displayed
- [ ] Deploy model → API endpoint becomes available
- [ ] Dashboard shows real-time metrics that match Prometheus data
- [ ] Works in Chrome, Firefox, Safari

**Definition of Done:**
- [ ] Non-technical user can deploy a model without using CLI
- [ ] Real-time monitoring shows tok/s, latency, error rate
- [ ] Model comparison view works with side-by-side metrics
- [ ] Responsive design (desktop + tablet)
- [ ] Basic auth protection (upgraded to SSO in Phase 3)

**Estimated effort:** 3 weeks (1 engineer, could use frontend contractor)

---

### Task 2.5: KV Cache Quantization

**What:** Quantize the KV cache from FP16 to INT8 to reduce memory usage by 50%, enabling longer context lengths and more concurrent users on the same hardware.

**Why:** KV cache memory grows linearly with context length and number of concurrent users. For 7B models with 4K context, KV cache is ~1GB per user in FP16. INT8 halves this, supporting either longer contexts or 2x more concurrent users.

**How:**
1. Implement INT8 KV cache with per-head quantization (scale + zero-point per attention head)
2. Implement PagedAttention: manage KV cache as fixed-size pages (256 tokens each), allocate non-contiguously
3. Implement prefix caching: cache and share KV state for common system prompts across users

**Dependencies:** Task 2.3 (continuous batching)

**Verification:**
```bash
# Measure memory savings
cli serve llama-2-7b-chat --kv-cache-dtype fp16 --max-context 4096
# Record memory usage
cli serve llama-2-7b-chat --kv-cache-dtype int8 --max-context 4096
# Expect: ~50% reduction in KV cache memory

# Quality impact
python benchmarks/scripts/kv_cache_quality.py --model llama-2-7b-chat --baseline fp16 --test int8
# Expect: < 0.5% perplexity degradation
```

**Definition of Done:**
- [ ] 50% KV cache memory reduction with INT8
- [ ] < 0.5% perplexity degradation vs FP16 KV cache
- [ ] PagedAttention enables non-contiguous allocation
- [ ] Prefix caching reduces memory for shared system prompts

**Estimated effort:** 2 weeks (1 engineer)

---

### Task 2.6: Speculative Decoding

**What:** Use a small draft model to predict multiple tokens, then verify them with the main model in one forward pass, increasing throughput by 1.5-2x.

**Why:** Speculative decoding is especially powerful on CPU: the draft model (1-2B params) fits entirely in L2 cache, making draft generation nearly free. The main model then verifies all draft tokens in a single batch forward pass — more compute-efficient than generating one at a time.

**How:**
1. Select draft model (Phi-2 or custom distilled model, ~1-2B params)
2. Draft model generates K candidate tokens (K=5 default, configurable)
3. Main model runs one forward pass on all K tokens simultaneously
4. Accept tokens where main model agrees, reject and resample where it disagrees
5. Auto-enable when a compatible draft model is available; auto-disable otherwise
6. Auto-tune K based on observed acceptance rate

**Dependencies:** Task 1.1 (engine base)

**Verification:**
```bash
# Measure throughput gain
cli bench llama-2-7b-chat --speculative off --output no_spec.json
cli bench llama-2-7b-chat --speculative on --draft-model phi-2 --output with_spec.json

# Compare
python benchmarks/scripts/compare_backends.py --results no_spec.json,with_spec.json
# Target: 1.5-2x throughput improvement
# Acceptance rate: 60-80%
```

**Definition of Done:**
- [ ] 1.5-2x throughput improvement with speculative decoding enabled
- [ ] Acceptance rate 60%+ on conversational tasks
- [ ] Output is identical to non-speculative decoding (correctness guarantee)
- [ ] Auto-enables when draft model available, no configuration needed
- [ ] Graceful fallback when draft model unavailable

**Estimated effort:** 2 weeks (1 engineer)

---

### Task 2.7: SOC2 Type 1 Preparation

**What:** Set up compliance infrastructure and begin the SOC2 Type 1 audit process.

**Why:** SOC2 is the entry ticket for enterprise sales in regulated industries. The Type 1 audit takes 3 months from start to completion. Delaying this delays enterprise revenue — each month of delay = one month delay in Phase 3 sales.

**How:**
1. **Month 3:** Set up Vanta/Drata compliance platform
   - Connect to GitHub (code review tracking)
   - Connect to AWS (infrastructure monitoring)
   - Connect to identity provider (access control)
   - Configure automated evidence collection

2. **Month 3-4:** Implement required controls:
   - Access control policy (who can access what)
   - Change management process (PR reviews, deployment approvals)
   - Incident response plan (documented, tested)
   - Data encryption (at rest and in transit)
   - Vulnerability management (dependency scanning, patching cadence)
   - Employee security training (for all team members)
   - Asset inventory (all systems, data stores, third-party services)

3. **Month 4:** Engage audit firm
   - Select auditor (recommend: Prescient Assurance, Johanson Group, or similar startup-friendly firm)
   - Define scope: Trust Service Criteria (Security + Availability + Confidentiality)
   - Kick off Type 1 readiness assessment

4. **Month 5:** Begin formal Type 1 audit
   - Auditor reviews controls design (point-in-time assessment)
   - Address any findings
   - Target completion: Month 7

**Dependencies:** Vanta/Drata account, audit firm selected

**Key files:**
- `enterprise/compliance/policies/` — Security policies (markdown)
- `enterprise/compliance/controls/` — Technical control implementations
- `.github/workflows/security.yml` — Automated security scanning

**Verification:**
- [ ] Vanta/Drata dashboard shows all controls "implemented"
- [ ] No critical findings in readiness assessment
- [ ] Audit firm formally engaged by month 5

**Definition of Done:**
- [ ] SOC2 Type 1 audit formally started by month 5
- [ ] All required technical controls implemented
- [ ] Compliance platform collecting evidence automatically
- [ ] Incident response plan documented and tabletop-tested

**Estimated effort:** Ongoing (founder + future compliance engineer, 20% of time)

---

### Task 2.8: Quality Dashboard

**What:** Build a public-facing quality scorecard showing perplexity, MMLU, and HumanEval scores for every supported model at each quantization level.

**Why:** Transparency about quality tradeoffs builds trust. Customers need to make informed decisions about which quantization level fits their use case. A published quality dashboard differentiates us from competitors who hand-wave about "minimal quality loss."

**How:**
1. Run quality evaluations for all model × quantization combinations:

| Model | FP16 Baseline | Q8_0 | Q4_K_M | Q4_0 |
|-------|--------------|------|--------|------|
| Llama 2 7B | Perplexity, MMLU, HumanEval | ... | ... | ... |
| Mistral 7B | ... | ... | ... | ... |
| Phi-3 Mini | ... | ... | ... | ... |
| Qwen 2 7B | ... | ... | ... | ... |

2. Auto-generate dashboard (static site or embedded in docs)
3. Include per-model recommendations:
   - "For clinical note summarization: Q4_K_M (acceptable quality) or Q8_0 (near-lossless, 1.5x slower)"
4. Auto-update when new models/quantizations are added (CI pipeline)

**Dependencies:** Task 1.4 (quantization pipeline), Task 1.5 (benchmarking framework)

**Verification:**
- [ ] Dashboard shows scores for all 4 models × 3 quantization levels
- [ ] Scores match independent manual evaluation within 1%
- [ ] Recommendations are consistent with measured quality

**Definition of Done:**
- [ ] Quality dashboard published and publicly accessible
- [ ] All supported models have quality scorecards
- [ ] Auto-updated via CI when new models are added

**Estimated effort:** 1 week (1 engineer)

---

### Phase 2 Go/No-Go Gate (End of Month 6)

**Required to proceed to Phase 3:**
- [ ] **LUT Performance:** 1.5-2.5x over llama.cpp on at least one platform (ARM or AMD)
- [ ] **Concurrent Serving:** Continuous batching handles 16+ concurrent users
- [ ] **SOC2:** Type 1 audit has formally begun (auditor engaged, controls implemented)
- [ ] **Revenue:** First paying pilot ($2-5K/month) or strong LOI with payment terms

**If gate fails:**
| Failure | Response |
|---------|----------|
| LUT < 1.3x on all platforms | Re-evaluate LUT approach. Consider focusing on platform value only (compliance + management). |
| SOC2 not started | Delay Phase 3 enterprise sales by equivalent months. Prioritize compliance immediately. |
| No paying pilot | Extend design partner phase. Offer free 3-month pilot to 2 more prospects. If still no interest by month 8, reassess product-market fit. |

---

## 5. Phase 3: Enterprise (Months 6-12)

**Goal:** SOC2 Type 1 certified, HIPAA-compliant deployment, enterprise auth (RBAC/SSO), 3-5 paying enterprise customers, $150-250K ARR.

### Task 3.1: SOC2 Type 1 Completion

**What:** Complete the SOC2 Type 1 audit, obtain the report, and begin the 12-month Type 2 observation period.

**Why:** The Type 1 report is the minimum requirement for most enterprise procurement processes. Without it, legal/compliance teams will block purchases.

**How:**
1. Work with auditor to address any findings from the assessment
2. Remediate gaps (typically: formal risk assessments, vendor management policies, business continuity plans)
3. Obtain clean Type 1 report
4. Begin Type 2 observation period (12 months of continuous monitoring)

**Verification:** SOC2 Type 1 report received from audit firm

**Definition of Done:**
- [ ] Clean Type 1 report (no qualified opinions)
- [ ] Type 2 observation period officially started
- [ ] Report shareable with enterprise prospects

**Estimated effort:** 2-4 weeks of remediation (compliance engineer)

---

### Task 3.2: HIPAA Compliance

**What:** Implement all technical and administrative controls required for HIPAA compliance in healthcare deployments.

**Why:** HIPAA is the entry ticket for healthcare — our beachhead vertical. Without HIPAA compliance, we cannot process Protected Health Information (PHI), which is the core of clinical NLP use cases.

**How:**
1. **Administrative safeguards:**
   - HIPAA risk assessment (engage third-party assessor)
   - Privacy policies and procedures
   - Business Associate Agreement (BAA) template for customers
   - Workforce training on PHI handling

2. **Technical safeguards:**
   - AES-256 encryption at rest for all model data and logs
   - TLS 1.3 for all data in transit
   - Audit controls: log all access to PHI-containing data
   - Access controls: minimum necessary access principle
   - Automatic session timeout
   - Emergency access procedures

3. **Physical safeguards (for on-premises deployment):**
   - Document customer's responsibility for physical security
   - Our software enforces logical access controls

4. **Breach notification:**
   - Incident response plan with HIPAA-specific timelines (72-hour notification)
   - Breach assessment checklist

**Dependencies:** Task 2.7 (SOC2 controls provide foundation)

**Verification:**
- [ ] Third-party HIPAA risk assessment passes
- [ ] BAA template reviewed by healthcare attorney
- [ ] Technical controls verified by pen test

**Definition of Done:**
- [ ] HIPAA risk assessment complete with remediation plan
- [ ] BAA template ready for customer signature
- [ ] All technical controls implemented and documented
- [ ] Staff training completed
- [ ] "HIPAA-compliant deployment" mode documented in product docs

**Estimated effort:** 6-8 weeks (compliance engineer + 1 engineer for technical controls)

---

### Task 3.3: Air-Gap Deployment

**What:** Enable full-featured deployment on networks with no internet connectivity.

**Why:** Government, defense, and some healthcare customers operate air-gapped networks. This is a hard requirement for the highest-value customer segments ($200K-1M/year).

**How:**
1. **Offline installation package:**
   - Bundled binary (engine + server + UI + all dependencies)
   - Bundled models (pre-quantized, customer-selected)
   - Installation script that works without package managers
   - Supports: RHEL/CentOS 8+, Ubuntu 20.04+, SLES 15+

2. **Offline license validation:**
   - Cryptographic license file (Ed25519 signed)
   - License encodes: customer ID, expiration, feature flags, machine fingerprint
   - No phone-home requirement

3. **Offline updates:**
   - Signed update packages transferable via USB/removable media
   - Version verification and rollback capability
   - Changelog included in package

4. **Feature parity:** All features work offline except telemetry and model downloads

**Dependencies:** Task 1.1, Task 1.6 (CLI for installation)

**Verification:**
```bash
# Test on a network-isolated VM
# 1. Disconnect VM from network
# 2. Transfer installation package via ISO mount
# 3. Install and configure
# 4. Load pre-bundled model
# 5. Run inference
# 6. Verify all features work (API, UI, monitoring)
# 7. Apply offline update package
```

**Definition of Done:**
- [ ] Full installation and operation without any network access
- [ ] License validation works offline
- [ ] Update mechanism works via removable media
- [ ] Tested on RHEL 8, Ubuntu 20.04, and Ubuntu 22.04

**Estimated effort:** 3 weeks (1 engineer)

---

### Task 3.4: Kubernetes Operator

**What:** Build a Kubernetes operator that automates deployment, scaling, and lifecycle management of inference models.

**Why:** Enterprise customers run Kubernetes. A K8s operator provides the deployment automation, auto-scaling, and self-healing that operations teams expect. It also enables multi-model serving across a cluster.

**How:**
1. **Custom Resource Definitions (CRDs):**
   ```yaml
   apiVersion: cpuinference.io/v1
   kind: InferenceModel
   metadata:
     name: llama-2-7b-clinical
   spec:
     model: llama-2-7b-chat
     quantization: q4_k_m
     replicas: 3
     resources:
       cpu: "48"
       memory: "64Gi"
     autoscaling:
       minReplicas: 1
       maxReplicas: 10
       targetConcurrency: 8
   ```

2. **Operator capabilities:**
   - Deploy model serving pods from CRD specs
   - Auto-scale based on request queue depth / latency
   - Rolling updates with zero downtime (drain → update → verify → route)
   - Health checks: liveness (process alive), readiness (model loaded), startup (model loading)
   - Auto-recovery: restart crashed pods, reschedule on node failure
   - Multi-model: route requests to different models via ingress rules

3. **Build with:** Operator SDK (Go) or kopf (Python)

**Dependencies:** Task 2.3 (continuous batching), Kubernetes cluster

**Verification:**
```bash
# Deploy operator
kubectl apply -f deploy/operator.yaml

# Create model deployment
kubectl apply -f examples/llama-2-7b.yaml

# Verify pods are running
kubectl get pods -l app=cpu-inference

# Test auto-scaling
python benchmarks/scripts/load_test.py --url http://inference-ingress/v1/chat/completions --concurrent 32
kubectl get pods -l app=cpu-inference  # Should see more pods

# Test rolling update
kubectl apply -f examples/llama-2-7b-v2.yaml  # New model version
# Verify zero downtime during rollout
```

**Definition of Done:**
- [ ] CRD-based model deployment works
- [ ] Auto-scaling based on load (scale up and down)
- [ ] Rolling updates with zero downtime
- [ ] Health checks and auto-recovery tested
- [ ] Helm chart for operator installation

**Estimated effort:** 4 weeks (1 engineer)

---

### Task 3.5: RBAC, SSO, Audit Logging

**What:** Implement enterprise authentication and authorization: role-based access control, single sign-on, and comprehensive audit logging.

**Why:** Enterprise customers require centralized identity management (SSO), granular permissions (RBAC), and audit trails for compliance. These are hard requirements in every enterprise procurement checklist.

**How:**
1. **RBAC roles:**
   | Role | Permissions |
   |------|------------|
   | Admin | Full access: manage models, users, settings, view audit logs |
   | Operator | Deploy/manage models, view metrics, manage API keys |
   | User | Send inference requests, view own usage |
   | Viewer | Read-only access to metrics and model info |

2. **SSO integration:**
   - SAML 2.0 (for enterprise IdPs: Okta, Azure AD, OneLogin)
   - OIDC (for Google Workspace, Auth0)
   - SCIM provisioning (auto-create/remove users from IdP)
   - JWT-based session management

3. **Audit logging:**
   - Every API call logged: timestamp, user, action, resource, IP, result
   - Every admin action logged: model deployments, config changes, user management
   - Tamper-resistant: append-only log, hash chaining
   - Export to SIEM (Splunk, ELK) via syslog or webhook
   - Retention: configurable, default 1 year

4. **Per-team controls:**
   - Team-level API keys
   - Per-team model access (team A can use model X but not model Y)
   - Usage quotas per team (tokens/month limit)

**Dependencies:** Task 2.4 (UI for admin interface)

**Verification:**
```bash
# SSO login flow
# 1. Configure Okta as IdP in admin settings
# 2. User clicks "Login with SSO" → redirected to Okta
# 3. After auth → redirected back with session token
# 4. Verify user has correct role based on IdP group mapping

# RBAC enforcement
# User with "User" role tries to deploy a model → 403 Forbidden
# User with "Operator" role deploys a model → 200 OK

# Audit log verification
curl http://localhost:8080/admin/audit-logs?user=john@example.com
# Verify all actions are logged with correct metadata
```

**Definition of Done:**
- [ ] SSO works with Okta, Azure AD (SAML 2.0), and Google (OIDC)
- [ ] RBAC enforced on all API endpoints
- [ ] Audit logs capture all actions with required metadata
- [ ] Audit log export to syslog/webhook works
- [ ] Per-team model access controls and quotas enforced
- [ ] SCIM user provisioning works

**Estimated effort:** 4 weeks (1 engineer + security/compliance engineer)

---

### Task 3.6: A/B Testing & Multi-Model Serving

**What:** Enable deploying multiple models simultaneously, routing traffic between them, and comparing performance/quality in production.

**Why:** Enterprise customers need to evaluate models before committing: "Is Llama 3 8B better than Mistral 7B for our clinical notes?" A/B testing framework lets them answer this with production data.

**How:**
1. Multi-model routing: single API endpoint, model selection via request parameter or routing rules
2. A/B traffic splitting: percentage-based (80/20) or user-based (consistent assignment)
3. Comparison metrics: response latency, quality scores (if customer provides evaluation), user preference
4. Statistical reporting: confidence intervals on metric differences

**Dependencies:** Task 2.3 (continuous batching for multi-model)

**Definition of Done:**
- [ ] Two models served simultaneously from one endpoint
- [ ] Traffic split works correctly (verified with 1000+ requests)
- [ ] Comparison report generated with statistical significance

**Estimated effort:** 2 weeks (1 engineer)

---

### Task 3.7: Customer Onboarding Pipeline

**What:** Build a streamlined process to take a new enterprise customer from signed contract to running inference in < 1 day.

**Why:** Fast time-to-value reduces churn risk. If onboarding takes weeks, customers question the purchase before seeing results.

**How:**
1. Automated tenant provisioning (namespace, API keys, SSO config)
2. Model selection wizard (recommend models based on use case)
3. Domain-specific calibration workflow (customer provides sample data)
4. Guided deployment (K8s or standalone, with health verification)
5. Welcome documentation and training materials

**Dependencies:** Tasks 3.4, 3.5

**Definition of Done:**
- [ ] New customer operational in < 1 business day
- [ ] Onboarding checklist tracked in system
- [ ] Customer receives: API credentials, documentation, support contact

**Estimated effort:** 2 weeks (1 engineer)

---

### Task 3.8: First Enterprise Customers

**What:** Close 3-5 enterprise deals at $50-80K/year each.

**Why:** Enterprise revenue validates the business model and provides proof points for Series A fundraising.

**How:**
1. Convert design partners to paying customers
2. Expand outreach to new healthcare prospects
3. Offer 3-month paid pilot ($5K/month) as entry point
4. Publish first customer case study
5. Establish quarterly business review process

**Dependencies:** Tasks 3.1, 3.2 (compliance for procurement approval)

**Definition of Done:**
- [ ] 3+ signed enterprise contracts
- [ ] $150-250K ARR recognized
- [ ] < 5% monthly churn
- [ ] 1 published case study

**Estimated effort:** Ongoing (sales + founder, 50% of time)

---

### Phase 3 Go/No-Go Gate (End of Month 12)

**Required to proceed to Phase 4:**
- [ ] SOC2 Type 1 report obtained
- [ ] HIPAA compliance validated by third party
- [ ] 3+ paying enterprise customers with signed contracts
- [ ] $150K+ ARR
- [ ] K8s operator functional, SSO working with major IdPs

**If gate fails:**
| Failure | Response |
|---------|----------|
| < 2 paying customers by month 9 | Hire enterprise sales immediately. Lower price to $2K/month. Target department-level buyers. |
| SOC2 Type 1 delayed | Prioritize above all other work. Sales cannot close without it. |
| ARR < $100K | Extend Phase 3 by 3 months before Phase 4. Focus entirely on sales. |

---

## 6. Phase 4: Scale (Months 12-18)

**Goal:** SOC2 Type 2 complete, support for 13B-30B models, expand to financial services, Series A fundraise at $500K-1M ARR.

### Task 4.1: SOC2 Type 2 Completion

**What:** Complete the 12-month observation period and obtain SOC2 Type 2 report.

**Why:** Type 2 is the gold standard — it proves controls operate effectively over time, not just that they're designed correctly (Type 1). Required by most large enterprises and government agencies.

**Dependencies:** Task 3.1 (Type 1 started observation period)

**Definition of Done:**
- [ ] 12-month observation period complete
- [ ] Clean Type 2 report obtained
- [ ] Report shareable with prospects

**Estimated effort:** Ongoing compliance monitoring + 4 weeks audit support

---

### Task 4.2: AMD EPYC Optimizations (Zen 5)

**What:** Deep optimization for AMD EPYC Turin (Zen 5) architecture with improved AVX-512 throughput.

**Why:** AMD EPYC is our strongest LUT advantage (no AMX). Zen 5 improves AVX-512 throughput — our LUT kernels should leverage these improvements.

**How:**
1. Benchmark existing LUT kernels on Zen 5 hardware (when available)
2. Tune tile sizes and memory layouts for Zen 5's improved cache hierarchy
3. Exploit any new Zen 5 instructions (improved VNNI, etc.)
4. Publish AMD-specific benchmarks and optimization guide

**Definition of Done:**
- [ ] LUT kernels optimized for Zen 5
- [ ] Published benchmarks on EPYC Turin
- [ ] AMD-specific deployment guide

**Estimated effort:** 2-3 weeks (systems engineer)

---

### Task 4.3: Multi-Node Model Sharding (13B-30B)

**What:** Distribute larger models (13B-30B parameters) across multiple CPU nodes for inference.

**Why:** 13B models require 64GB+ RAM and benefit from dual-socket servers. 30B models require multi-node distribution. Supporting these sizes expands our addressable market.

**How:**
1. Implement tensor parallelism across NUMA nodes (dual-socket)
2. Implement pipeline parallelism across network-connected nodes
3. Minimize inter-node communication (attention head partitioning)
4. Target: 13B at interactive latency on dual-socket, 30B at batch latency on 2-4 nodes

**Definition of Done:**
- [ ] 13B models run at 5+ tok/s on dual-socket server
- [ ] 30B models run at 2+ tok/s on 4-node cluster
- [ ] Sharding is transparent to API consumers

**Estimated effort:** 6 weeks (systems engineer)

---

### Task 4.4: Edge Deployment (ARM Embedded)

**What:** Support deployment on edge/embedded ARM devices for industrial and field use cases.

**Why:** Edge AI is a growing segment. Our ARM LUT kernels transfer naturally to embedded ARM (Jetson, Raspberry Pi 5, industrial controllers).

**Definition of Done:**
- [ ] 1-3B models run on ARM edge devices (4GB+ RAM)
- [ ] Offline deployment package for edge
- [ ] Power consumption benchmarks published

**Estimated effort:** 3 weeks (1 engineer)

---

### Task 4.5: FedRAMP Preparation

**What:** Begin the FedRAMP authorization process for government deployment eligibility.

**Why:** FedRAMP opens the $200K-1M/year government vertical. It's a 12-18 month process, so starting in Phase 4 targets completion in Phase 5.

**How:**
1. Engage FedRAMP Third Party Assessment Organization (3PAO)
2. Implement NIST 800-53 controls (builds on SOC2 foundation)
3. Prepare System Security Plan (SSP)
4. Begin authorization process

**Dependencies:** Task 4.1 (SOC2 Type 2 as foundation)

**Definition of Done:**
- [ ] 3PAO engaged
- [ ] Initial assessment completed
- [ ] SSP drafted and under review

**Estimated effort:** Ongoing (compliance engineer, 50% of time for 6 months)

---

### Task 4.6: Financial Services Expansion

**What:** Extend go-to-market into banks and financial services using existing SOC2/compliance infrastructure.

**Why:** Financial services (SOX, FINRA) share significant compliance overlap with healthcare (SOC2). The compliance moat built for healthcare transfers directly.

**How:**
1. Adapt positioning for financial use cases (document analysis, compliance screening)
2. Add financial-specific compliance controls (SOX requirements)
3. Target mid-market banks and insurance companies
4. Leverage healthcare case studies as proof of enterprise readiness

**Definition of Done:**
- [ ] 2+ financial services pilot customers
- [ ] Financial-specific compliance documentation
- [ ] Financial services case study

**Estimated effort:** Ongoing (sales, with engineering support for compliance additions)

---

### Task 4.7: Series A Fundraise

**What:** Raise $5-10M Series A to scale the team and business.

**Why:** Series A funds expansion to 15-20 people, faster sales hiring, and R&D for next-generation optimizations.

**Trigger criteria:**
- [ ] $500K+ ARR (achieved by month 15-18)
- [ ] 10+ enterprise customers with < 5% churn
- [ ] LTV:CAC > 3:1
- [ ] Repeatable sales playbook in healthcare
- [ ] SOC2 Type 2 complete or near-complete

**Estimated effort:** 3-4 months (founder, 60% of time)

---

### Phase 4 Target (End of Month 18)

- [ ] $500K-1M ARR (base case), $1.5M+ (stretch)
- [ ] 10-15 enterprise customers
- [ ] SOC2 Type 2 certified
- [ ] 13B model support validated
- [ ] Series A raised or in final stages
- [ ] Financial services vertical entered

---

## 7. Testing Strategy (Cross-Phase)

### Test Categories

| Category | Scope | Frequency | Tools |
|----------|-------|-----------|-------|
| **Unit Tests** | Individual functions, kernels, endpoints | Every PR | pytest, CTest, Google Test |
| **Integration Tests** | End-to-end flows (upload → quantize → deploy → query) | Every PR | pytest + httpx |
| **Performance Tests** | Benchmark suite on reference hardware | Nightly CI | Custom framework (Task 1.5) |
| **Quality Tests** | Perplexity/MMLU regression after engine changes | Nightly CI | lm-eval-harness |
| **Security Tests** | Vulnerability scanning, pen testing | SAST: every PR; Pen test: annually | Snyk, OWASP ZAP |
| **Load Tests** | Concurrent user simulation (1→4→16→64→256) | Weekly CI | locust, k6 |
| **Compliance Tests** | Audit log completeness, access control enforcement | Weekly CI | Custom scripts |
| **Chaos Tests** | Failure injection (pod kill, network partition) | Monthly | Chaos Monkey, Litmus |

### Test Coverage Targets

| Component | Target Coverage | Rationale |
|-----------|----------------|-----------|
| LUT kernels | 100% correctness (bitwise match vs naive) | Correctness is non-negotiable |
| API endpoints | 95%+ line coverage | Customer-facing, high impact |
| Auth/RBAC | 100% permission paths | Security-critical |
| Serving layer | 90%+ | Complex state management |
| CLI commands | 80%+ | User-facing, clear error paths |

### Performance Regression Detection

```
Nightly benchmark results → Compare vs 7-day rolling average
  → If > 5% regression → Auto-create GitHub issue with bisect range
  → If > 10% regression → Alert on-call engineer via PagerDuty
  → Monthly → Compare vs latest llama.cpp release
```

---

## 8. Review & Quality Gates

### Code Review Process

| Change Type | Required Approvals | Reviewers |
|------------|-------------------|-----------|
| LUT kernel changes | 2 | Systems engineer + founder |
| API changes | 1 | Backend engineer |
| Security/auth changes | 2 | Security engineer + 1 other |
| CI/CD changes | 1 | Any engineer |
| Documentation | 1 | Any team member |

### Architecture Decision Records (ADRs)

For every major design decision, create an ADR in `docs/adr/`:
```
docs/adr/
├── 001-monorepo-structure.md
├── 002-llama-cpp-fork-strategy.md
├── 003-lut-kernel-design.md
├── 004-api-framework-choice.md
├── 005-kv-cache-quantization.md
└── template.md
```

### Phase Gate Reviews

| Gate | Participants | Duration | Decision |
|------|-------------|----------|----------|
| Phase 1 → 2 | All team + advisors | 2 hours | Proceed / Modify / Pivot |
| Phase 2 → 3 | All team + advisors + investors | 3 hours | Proceed / Modify / Pivot |
| Phase 3 → 4 | All team + advisors + investors + customer reference | 3 hours | Proceed / Modify |

---

## 9. Monitoring & Observability

### Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Metrics | Prometheus | Time-series metrics collection |
| Dashboards | Grafana | Visualization and alerting |
| Logging | Structured JSON → ELK/Loki | Centralized log aggregation |
| Tracing | OpenTelemetry | Request tracing across services |
| Alerting | PagerDuty / OpsGenie | On-call alerting |

### Key Dashboards

1. **Inference Performance:** tok/s, TTFT, P50/P95/P99 latency, active requests, queue depth
2. **System Resources:** CPU utilization per core, memory usage, cache hit rates, NUMA traffic
3. **Model Quality:** Perplexity tracking over time, output anomaly detection
4. **Business:** Requests per customer, token usage, error rates, SLA compliance
5. **Infrastructure:** Pod health, node status, disk usage, network I/O

### Alert Rules

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| High latency | P95 > 2x baseline for 5 min | Warning | Investigate load |
| Error spike | Error rate > 1% for 5 min | Critical | On-call engineer |
| Model quality drift | Perplexity increase > 5% | Warning | Review model/quantization |
| Disk full | < 10% free | Critical | Expand storage |
| OOM risk | Memory > 90% for 10 min | Critical | Scale up or reduce models |

---

## 10. Security Checklist (Per Phase)

### Phase 1
- [ ] HTTPS/TLS for all API endpoints
- [ ] API key authentication (bearer tokens)
- [ ] Dependency vulnerability scanning in CI (Snyk/Dependabot)
- [ ] Secrets management (no credentials in code — use environment variables or vault)
- [ ] Container image scanning
- [ ] Signed commits (GPG)

### Phase 2
- [ ] All SOC2 technical controls implemented (see Task 2.7)
- [ ] Encryption at rest for stored models and data
- [ ] Access logging for all API calls
- [ ] Rate limiting on public endpoints
- [ ] Input validation and sanitization on all endpoints
- [ ] CORS configuration

### Phase 3
- [ ] SSO/RBAC fully enforced (see Task 3.5)
- [ ] HIPAA technical safeguards (see Task 3.2)
- [ ] Annual penetration testing by third-party firm
- [ ] Air-gap security controls (see Task 3.3)
- [ ] Tamper-resistant audit logs with hash chaining
- [ ] Automated security scanning in CI (SAST + DAST)
- [ ] Incident response plan tested via tabletop exercise

### Phase 4
- [ ] FedRAMP NIST 800-53 controls (subset)
- [ ] Multi-tenancy security hardening (tenant isolation verification)
- [ ] Advanced threat detection (anomalous API usage patterns)
- [ ] Security awareness training for all team members (quarterly)
- [ ] Third-party security audit of LUT kernel implementations

---

## 11. Technology Decisions Matrix

| Decision | Choice | Why | Alternatives Considered |
|----------|--------|-----|------------------------|
| Inference engine base | llama.cpp fork | Broadest HW support, most active community (700+ contributors), best GGUF quantization | vLLM (GPU-focused), CTranslate2 (less LLM focus), build from scratch (too slow) |
| Intel backend | OpenVINO | AMX hardware acceleration, graph-level optimization (operator fusion) | ONNX Runtime (no AMX path), direct AMX intrinsics (too low-level) |
| API framework | FastAPI | Async support, auto-generated OpenAPI docs, SSE streaming, Python ecosystem | Flask (no async), Express.js (different language), gRPC (not HTTP/REST) |
| API server language | Python | Rapid development, ML ecosystem, llama-cpp-python bindings exist | C++ (harder to iterate), Go (no ML ecosystem), Rust (slower development) |
| LUT kernel language | C with intrinsics | Direct SIMD control, zero overhead, matches llama.cpp codebase | Assembly (too hard to maintain), Rust (intrinsics less mature) |
| UI framework | React + TypeScript | Large ecosystem, component libraries, TypeScript type safety | Vue (smaller ecosystem), Svelte (less enterprise adoption) |
| Compliance platform | Vanta or Drata | Automated evidence collection, startup-friendly pricing, integrates with GitHub/AWS | Manual compliance (too slow), Secureframe (more expensive) |
| Container orchestration | Kubernetes | Industry standard, operator pattern, enterprise-expected | Docker Swarm (limited), Nomad (less adoption) |
| K8s operator framework | Operator SDK (Go) | Official framework, mature, good documentation | kopf (Python, simpler but less capable), kubebuilder (lower-level) |
| CI/CD | GitHub Actions | Integrated with code hosting, good ARM support, free for public repos | GitLab CI (requires migration), Jenkins (maintenance overhead) |
| Metrics | Prometheus + Grafana | Industry standard, Kubernetes-native, excellent alerting | Datadog (expensive), CloudWatch (AWS-only) |

---

## 12. Dependency Graph & Critical Path

### Task Dependencies (Phases 1-2)

```
Phase 1:
  Task 1.1 (Fork llama.cpp)
    ├──→ Task 1.2 (CPU Detection)
    ├──→ Task 1.3 (OpenVINO Backend) ──→ Task 1.5 (Benchmarking)
    ├──→ Task 1.4 (Auto-Quantization) ──→ Task 1.6 (CLI Tool)
    └──→ Task 1.7 (CI/CD) [can start in parallel]

  Task 1.8 (Design Partners) [independent, start immediately]

Phase 2:
  Task 1.1 + 1.2 ──→ Task 2.1 (LUT ARM) ──→ Task 2.2 (LUT x86)
                                              [or in parallel if 2 engineers]
  Task 1.1 ──→ Task 2.3 (Continuous Batching) ──→ Task 2.5 (KV Cache)
  Task 1.4 + 1.1 ──→ Task 2.4 (Model Management UI)
  Task 1.1 ──→ Task 2.6 (Speculative Decoding)
  Task 2.7 (SOC2 Prep) [independent, start month 3]
  Task 1.5 + 1.4 ──→ Task 2.8 (Quality Dashboard)
```

### Critical Path

The longest sequential chain determines minimum timeline:

```
Fork llama.cpp (2-3 wk)
  → OpenVINO backend (2 wk)
  → Benchmarking framework (2 wk)
  → LUT ARM kernels (3-4 wk)
  → LUT x86 kernels (3-4 wk)
  → Performance validation
= ~13-15 weeks (aligns with Phase 1-2 timeline)

Parallel critical path (compliance):
SOC2 prep (month 3) → Auditor engagement (month 4) → Type 1 audit (month 5-7) → Type 1 report (month 7)
  → Type 2 observation (month 7-19) → Type 2 report (month 19)
= SOC2 Type 2 is the longest single dependency in the entire plan
```

### Parallelization Opportunities

| Tasks That Can Run in Parallel | Required Engineers |
|-------------------------------|-------------------|
| Task 1.1 + Task 1.8 | 1 engineer + 1 founder |
| Task 1.3 + Task 1.4 (after 1.1) | 2 engineers |
| Task 2.1 + Task 2.3 | 2 engineers (systems + backend) |
| Task 2.4 + Task 2.5 | 2 engineers (frontend + backend) |
| Task 2.7 (SOC2) runs parallel to all engineering | Compliance engineer/founder |
| Task 3.2 + Task 3.3 + Task 3.4 | 3 engineers |

---

## 13. Quick Reference: All Verification Commands

### Phase 1 Smoke Tests
```bash
# Full Phase 1 verification script
cli info                                          # CPU detection works
cli run phi-3-mini --first-token-only              # Download + inference works
cli serve llama-2-7b-chat --port 8080 &            # API server starts
curl http://localhost:8080/health                   # Health check
curl http://localhost:8080/v1/models                # Model listing
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-2-7b-chat","messages":[{"role":"user","content":"Hello"}]}' # Inference
python benchmarks/scripts/run_benchmark.py --config quick  # Quick benchmark
```

### Phase 2 Smoke Tests
```bash
# LUT kernel verification
cd engine/build && ctest -R lut --output-on-failure

# Continuous batching load test
python benchmarks/scripts/load_test.py --concurrent 16 --duration 30s

# Speculative decoding
cli bench llama-2-7b-chat --speculative on --draft-model phi-2

# Quality dashboard data
python benchmarks/scripts/quality_eval.py --all-models --output quality_data.json
```

### Phase 3 Smoke Tests
```bash
# Air-gap test
python tests/test_airgap.py  # Runs in network-isolated container

# K8s operator
kubectl apply -f examples/llama-2-7b.yaml && kubectl wait --for=condition=ready pod -l app=cpu-inference

# SSO flow
python tests/test_sso.py --idp okta --config tests/okta_config.json

# RBAC enforcement
python tests/test_rbac.py --roles admin,operator,user,viewer

# Audit log completeness
python tests/test_audit_logs.py --actions deploy,query,delete
```

---

*This document is the single source of truth for engineering execution. Update it as plans evolve. All changes require PR review.*

*For business context: [Investor Pitch](investor-pitch.md) | For technical deep-dives: [Technical Spec](technical-spec.md) | For team and operations: [Operations Plan](operations-plan.md)*
