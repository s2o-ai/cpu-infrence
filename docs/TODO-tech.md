# Engineering TODO

> Backend, kernels, infrastructure, and serving.
> For UI/UX tasks, see [TODO-uiux.md](TODO-uiux.md). For business/compliance, see [TODO-biz.md](TODO-biz.md).

---

## Phase 1: Foundation (Months 1-3)

### Task 1.1: Fork & Extend llama.cpp — DONE
- [x] llama.cpp forked @ tag b8445, submodule at `engine/src/llama/`
- [x] llama-server.exe built (GCC 15.2, `-march=native`, AVX2)
- [x] All 6 API endpoints working (health, models, chat, completions, streaming, metrics)
- [x] 7/7 smoke tests pass (`scripts/test_smoke.py`)
- [x] OpenAI Python SDK works as drop-in replacement
- [x] Scripts: `scripts/build.py`, `scripts/serve.py`, `scripts/test_smoke.py`
- [x] Baseline: 212 tok/s prompt processing, 8.46 tok/s generation (Qwen3.5-0.8B Q4_K_M)

### Task 1.2: CPU Feature Auto-Detection — DONE
- [x] x86 CPUID detection (AVX2, AVX-512, AMX, VNNI, FMA, F16C, BMI2)
- [x] ARM HWCAP detection (NEON, SVE, SVE2, DOTPROD, I8MM, BF16)
- [x] Cache topology (L1/L2/L3 via CPUID leaf 4 + sysfs + sysctl)
- [x] Memory detection (GlobalMemoryStatusEx + /proc/meminfo)
- [x] NUMA topology (sysfs + kernel32)
- [x] Backend recommendation engine (llama_cpp / openvino / llama_cpp_lut / lut_neon)
- [x] 20 tests pass (`tests/test_detect.py`)
- [x] Module: `engine/detect/`

### Task 1.3: OpenVINO Backend Integration — ALREADY IN LLAMA.CPP
> **Audit finding:** Full OpenVINO backend exists at `ggml/src/ggml-openvino/`. Supports Intel iGPU (Xe), Arc, NPU. Graph-level SDPA fusion, matmul squeeze, ZP elimination. Build with `-DGGML_OPENVINO=ON`.

- [x] OpenVINO backend fully implemented upstream (iGPU, Arc, NPU via Level Zero)
- [x] Full op coverage + graph-level optimization passes
- [x] Wire `-DGGML_OPENVINO=ON` into `s2o build --openvino` flag
- [ ] Benchmark: llama.cpp CPU vs OpenVINO on Intel Xeon
- [ ] Fallback to llama.cpp CPU if OpenVINO unavailable
- **Blocker:** Needs Intel Xeon with AMX (AWS c7i.metal-48xl, ~$8.57/hr)

### Task 1.4: Auto-Quantization Pipeline — DONE
- [x] Accept HuggingFace model ID or local safetensors path
- [x] Convert to GGUF (FP16 baseline) — remote streaming + full download modes
- [x] Quantize to Q4_K_M + Q8_0 (any of 34+ types supported)
- [x] Quality validation (perplexity on WikiText-2) — optional via `--validate`
- [x] Generate quality report JSON + markdown
- [x] `s2o quantize` CLI command with Rich progress output
- [x] 18 tests pass (`tests/test_quantize.py`)
- [x] Module: `engine/quantize/`
- [x] Validated: Qwen/Qwen3-0.6B → Q4_K_M (462 MB, 3.1x compression)

### Task 1.5: Benchmarking Framework — MOSTLY DONE
- [x] `benchmarks/bench_runner.py` — wraps llama-bench.exe, parses JSON
- [x] `benchmarks/bench_server.py` — concurrent HTTP streaming benchmarks (TTFT, P50/P95/P99)
- [x] `benchmarks/bench_report.py` — markdown + JSON report generation
- [x] `benchmarks/bench_types.py` — dataclasses for results
- [x] Results include median + avg ± stddev
- [x] Validated against baseline on dev machine
- [x] 11 tests pass (`tests/test_bench.py`)
- [ ] Run on all 3 reference hardware configs — **BLOCKED: needs AWS instances**
- [ ] Multi-backend comparison reports — **BLOCKED: needs OpenVINO (Task 1.3)**

### Task 1.6: CLI Tool — DONE
> Full UX details in [TODO-uiux.md](TODO-uiux.md).

- [x] 8 commands: info, build, serve, models, run, bench, quantize + subcommands
- [x] Typer + Rich, --help for all commands
- [x] 5 tests pass (`tests/test_cli.py`)
- [ ] Test on Linux and macOS — **BLOCKED: needs Linux/macOS machines**

---

## Phase 2: Differentiation (Months 3-6)

### Task 2.1: Custom LUT Kernels — ARM (NEON/SVE2) — MOSTLY DONE
- [x] NEON baseline kernel: `vmull_s8` + `vpaddlq_s16` integer dot product (`lut-arm-neon.cpp`)
- [x] DOTPROD variant: `vdotq_s32` (~4x fewer instructions in inner loop)
- [x] On-the-fly FP32→INT8 quantization: `vcvtnq_s32_f32` + `vqmovn` narrowing chain
- [x] Q4_0 nibble unpacking: `vandq_u8`/`vshrq_n_u8` split + center at [-8,+7]
- [x] Dual dispatch tables: `s2o_lut_kernels_neon` and `s2o_lut_kernels_neon_dotprod`
- [x] CMake integration: ARM `elseif` block in `ggml-cpu/CMakeLists.txt`
- [x] `lut-common.h` updated: ARM externs + compile-time selector
- [x] `s2o-lut.cpp` guard widened: `#if defined(__AVX2__) || defined(__ARM_NEON)`
- [x] `build.py --arch aarch64` cross-compile flag
- [x] 9 Python integration tests pass (`tests/test_lut.py::TestLutArmKernel`)
- [x] Benchmark config: `benchmarks/configs/graviton.yaml`
- [ ] SVE2 kernel (TBL with predication) — **future optimization**
- [ ] Weight re-packing for LUT-friendly layout
- [ ] Correctness tests on actual ARM hardware — **BLOCKED: needs Graviton instance**
- [ ] Target: 1.3-2x over llama.cpp on Graviton4

### Task 2.2: Custom LUT Kernels — x86 (AVX2/AVX-512) — MOSTLY DONE
> **Note:** llama.cpp already has full AVX-512 SIMD kernels for all quant types (`ggml-cpu/arch/x86/quants.c` — VPSHUFB, VPDPBUSD, VPMADDUBSW). Also has AMX INT8 backend (`ggml-cpu/amx/`). Our LUT kernels must beat these existing paths to justify the fork overhead.

- [x] Backend infrastructure: `extra_buffer_type` + `tensor_traits` (pattern from AMX/KleidiAI)
- [x] Public header: `ggml_backend_cpu_s2o_lut_buffer_type()` in `s2o-lut/s2o-lut.h`
- [x] Integration: `s2o-lut/s2o-lut.cpp` — supports_op, compute_forward, thread partitioning
- [x] Kernel dispatch table + runtime selector in `s2o-lut/lut-common.h`
- [x] AVX2 kernel: 4-wide GEMV + VPSHUFB dequant + L2 tiled GEMM + software prefetch
- [x] AVX-512 kernel: 4-wide (2×dual-column 512-bit) + VPSHUFB dequant + L2 tiling + prefetch
- [x] CMake integration: `GGML_S2O_LUT` option, `s2o build --lut`
- [x] C++ correctness tests pass (17/17 — GEMV×7 + GEMM×3 + repack roundtrip×2 + packed GEMV×4 + summary)
- [x] Python integration tests pass (42/42)
- [x] VPSHUFB LUT path: 16-entry constant table `[-8..+7]`, replaces arithmetic nibble-unpack
- [x] Cache-aware tiling: L2-sized tiles (256KB default), `_mm_prefetch` with PREFETCH_DIST=4
- [x] Auto-benchmark at kernel init: synthetic GEMV reports GOPS to stderr
- [x] Weight repacking in `set_tensor`: column-interleaved 4-wide layout (`s2o_repack_q4_0`)
- [x] Packed GEMV/GEMM kernels (AVX2 + AVX-512) using contiguous 4-block access
- [x] Dual `tensor_traits` pattern: standard vs packed layout selected via `tensor->extra`
- [ ] Target: 1.3-2x over stock llama.cpp on AMD EPYC / Intel Xeon — **needs hardware benchmarking**

### Task 2.3: Continuous Batching — MOSTLY IN LLAMA.CPP
> **Audit finding:** Iteration-level continuous batching is fully implemented in `tools/server/server-context.cpp` (slot-based scheduling). Use `--parallel N` flag. No backpressure (503 + Retry-After) yet — that's our value-add.

- [x] Iteration-level scheduling — already in llama-server (`--parallel N`)
- [x] KV cache memory pool — already managed per-slot in llama-server
- [x] Wire `--parallel` into `s2o serve --max-concurrent N`
- [x] Priority queue proxy (`engine/serving/proxy.py` — header-based priority, `X-Priority` header)
- [x] Graceful degradation (503 + Retry-After at capacity, semaphore-based admission control)
- [x] `/metrics` monitoring endpoint (Prometheus-compatible: `s2o_active_requests`, `s2o_requests_total`, `s2o_503_total`)
- [x] `/v1/status` endpoint (active, max, pct_used, total_requests, rejected_503)
- [x] Streaming support via `httpx.AsyncClient.stream()` for SSE responses
- [x] `s2o serve --proxy --max-concurrent N` to enable proxy
- [x] 13 tests pass (`tests/test_serving.py`)
- [ ] Target: 16+ concurrent users with P99 < 2x P50 — **BLOCKED: needs load testing**

### Task 2.4: Model Management UI — MOVED TO [TODO-uiux.md](TODO-uiux.md)

### Task 2.5: KV Cache Optimization — PARTIALLY IN LLAMA.CPP
> **Audit finding:** KV cache quantization is fully implemented (`--cache-type-k/v` supports F32, F16, BF16, Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0, Q5_1). Per-slot prefix caching exists (`--cache-reuse N`). Global cross-slot prefix cache and PagedAttention are NOT in llama.cpp.

- [x] KV cache quantization — already in llama.cpp (`--cache-type-k Q8_0 --cache-type-v Q8_0`)
- [x] Per-slot prefix caching — already in llama.cpp (`--cache-reuse N`, uses `llama_kv_cache_seq_rm/shift`)
- [x] Wire `--cache-type-k/v` into `s2o serve --kv-quant` flag
- [x] Auto-recommendation: `engine/detect/_recommend.py` adds `kv_quant` for low-RAM scenarios
- [x] Benchmark script: `benchmarks/bench_kv.py` (f16 vs q8_0 vs q4_0 comparison)
- [ ] Run benchmarks on reference hardware — **BLOCKED: needs AWS instances**
- [ ] Global cross-slot prefix cache (shared KV pools for common system prompts) — **NOT in llama.cpp, must build**
- [ ] PagedAttention (vLLM-style block table / physical-virtual page mapping) — **NOT in llama.cpp, must build**
- [ ] Target: 40-50% KV cache memory reduction, < 0.5% perplexity degradation

### Task 2.6: Speculative Decoding — ALREADY IN LLAMA.CPP
> **Audit finding:** Fully implemented in `common/speculative.cpp`. 5 strategies: draft model, ngram-simple, ngram-map-k, ngram-mod, ngram-cache, plus EAGLE3 variant. Tracks `n_draft_total`/`n_draft_accepted` for acceptance rate monitoring.

- [x] K-token speculation with verification — already in llama.cpp (`--speculative`)
- [x] Draft model support — already in llama.cpp (`--model-draft`)
- [x] Multiple strategies (draft, ngram, EAGLE3) — already in llama.cpp
- [x] Acceptance rate tracking — already in llama.cpp (`n_draft_total`/`n_draft_accepted`)
- [x] Wire `--speculative` + `--model-draft` into `s2o serve --speculative --draft-model`
- [x] Draft model auto-selection (`engine/serving/speculative.py` — family+size matching from `models/`)
- [x] Benchmark script: `benchmarks/bench_speculative.py` (baseline vs draft A/B comparison)
- [x] 18 tests pass (`tests/test_speculative.py`)
- [x] Auto-tune K based on acceptance rate (`KAutoTuner` class + `--draft-k` CLI flag)
- [ ] Run benchmarks on reference hardware — **BLOCKED: needs matching model pairs**
- [ ] Target: 1.3-2x throughput improvement

### Task 2.8: Quality Dashboard — MOVED TO [TODO-uiux.md](TODO-uiux.md)

---

## Phase 3: Enterprise (Months 6-12)

### Task 1.7: CI/CD Pipeline (moved from Phase 1)
- [ ] PR pipeline: build + test + lint on Linux x86/ARM + macOS
- [ ] Nightly pipeline: benchmarks on reference hardware
- [ ] Release pipeline: Python wheels + Docker images
- [ ] Performance regression alerts (>5% regression → auto-issue)

### Task 3.3: Air-Gap Deployment
- [ ] Offline installation package (bundled binary + models)
- [ ] Offline license validation (Ed25519 signed)
- [ ] Offline updates via USB/removable media
- [ ] Test on RHEL 8, Ubuntu 20.04, Ubuntu 22.04

### Task 3.4: Kubernetes Operator
- [ ] CRDs for InferenceModel
- [ ] Auto-scaling based on request queue / latency
- [ ] Rolling updates with zero downtime
- [ ] Helm chart for operator installation

### Task 3.6: A/B Testing & Multi-Model Serving
> UI components moved to [TODO-uiux.md](TODO-uiux.md). Backend routing stays here.

- [x] Multi-model routing (single endpoint, model via request param) — `engine/serving/router.py`
- [x] Traffic splitting (weighted-random A/B) — `ModelRouter._select_route()`
- [x] Routing proxy (`engine/serving/router.py`) — 14 tests pass (`tests/test_router.py`)

---

## Phase 4: Scale (Months 12-18)

### Task 4.2: AMD EPYC Optimizations (Zen 5)
- [ ] Tune LUT kernels for Zen 5 cache hierarchy
- [ ] Exploit new Zen 5 instructions
- [ ] AMD-specific benchmarks and deployment guide

### Task 4.3: Multi-Node Model Sharding (13B-30B)
- [ ] Tensor parallelism across NUMA nodes (dual-socket)
- [ ] Pipeline parallelism across network nodes
- [ ] Target: 13B @ 5+ tok/s dual-socket, 30B @ 2+ tok/s 4-node

### Task 4.4: Edge Deployment (ARM Embedded)
- [ ] 1-3B models on ARM edge devices (4GB+ RAM)
- [ ] Offline deployment package for edge
- [ ] Power consumption benchmarks

---

## Test Suite Summary

| File | Tests | Status |
|------|-------|--------|
| `tests/test_detect.py` | 20 | All pass |
| `tests/test_bench.py` | 11 | All pass |
| `tests/test_cli.py` | 7 | All pass |
| `tests/test_quantize.py` | 18 | All pass |
| `tests/test_lut.py` | 42 | All pass (19 x86 + 9 ARM + 4 kernel opt + 2 bench + 8 repacking) |
| `tests/test_serving.py` | 13 | All pass |
| `tests/test_speculative.py` | 26 | All pass |
| `tests/test_router.py` | 14 | All pass |
| `s2o-lut/test_lut.cpp` (C++) | 17 | All pass (AVX2 GEMV×7 + GEMM×3 + repack×2 + packed GEMV×4 + summary) |
| **Total** | **168** | **All pass** |

---

## What's Free from llama.cpp (Don't Rebuild)

> Per [llama-cpp-audit.md](llama-cpp-audit.md) — these features are production-ready upstream.

| Feature | llama.cpp Location | CLI Flag |
|---------|-------------------|----------|
| Continuous Batching | `tools/server/server-context.cpp` | `--parallel N` |
| KV Cache Quantization | `src/llama-kv-cache.cpp` | `--cache-type-k/v F16\|Q8_0\|Q4_0` |
| Speculative Decoding | `common/speculative.cpp` | `--speculative` / `--model-draft` |
| OpenVINO Backend | `ggml/src/ggml-openvino/` | `-DGGML_OPENVINO=ON` |
| CPU Feature Detection | `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp` | Automatic |
| AMX INT8 Backend | `ggml/src/ggml-cpu/amx/` | Automatic |
| ARM KleidiAI Kernels | `ggml/src/ggml-cpu/kleidiai/` | Automatic |
| Per-slot Prefix Cache | `tools/server/server-context.cpp` | `--cache-reuse N` |

## What S2O Must Build (True Differentiators)

1. **LUT INT4 kernels** (x86 + ARM) — proprietary, T-MAC inspired
2. **Global cross-slot prefix cache** — llama.cpp only does per-slot
3. **PagedAttention** — vLLM-style, not in llama.cpp
4. **Enterprise wrap** — RBAC, SSO, HIPAA, K8s operator, air-gap
