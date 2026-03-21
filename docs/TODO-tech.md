# Engineering TODO

> Technical tasks only. For business/compliance tasks, see [TODO-biz.md](TODO-biz.md).
> Full plan: [TODO.md](TODO.md)

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

### Task 1.3: OpenVINO Backend Integration — NOT STARTED
- [ ] Install OpenVINO Runtime
- [ ] Model conversion pipeline (GGUF/safetensors → OpenVINO IR)
- [ ] Inference wrapper matching engine API
- [ ] Benchmark: llama.cpp vs OpenVINO on Intel Xeon
- [ ] Fallback to llama.cpp if OpenVINO unavailable
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
- [x] `s2o info` — CPU detection with Rich panels (--json, --verbose)
- [x] `s2o build` — build inference engine (--clean)
- [x] `s2o serve <model>` — start OpenAI-compatible server
- [x] `s2o models` — list downloaded GGUF models
- [x] `s2o run <model>` — interactive chat via llama-cli
- [x] `s2o bench <model>` — benchmarks with reports (--server for throughput)
- [x] Typer + Rich, --help for all commands
- [x] 5 tests pass (`tests/test_cli.py`)
- [ ] Test on Linux and macOS — **BLOCKED: needs Linux/macOS machines**

---

## Phase 2: Differentiation (Months 3-6)

### Task 2.1: Custom LUT Kernels — ARM (NEON/SVE2)
- [ ] LUT data structure (16x16 INT4 lookup table, L1-resident)
- [ ] NEON kernel (VTBL/VTBX + SDOT)
- [ ] SVE2 kernel (TBL with predication)
- [ ] Weight re-packing for LUT-friendly layout
- [ ] Correctness tests (bitwise match vs naive)
- [ ] Target: 1.3-2x over llama.cpp on Graviton4

### Task 2.2: Custom LUT Kernels — x86 (AVX-512)
- [ ] AVX-512 VPSHUFB-based parallel LUT lookups
- [ ] AMX INT8 alternative path (Intel only)
- [ ] Cache-aware tiling (L2-sized tiles, software prefetch)
- [ ] Auto-benchmark LUT INT4 vs AMX INT8 at model load
- [ ] Target: 1.3-2x over llama.cpp on AMD EPYC

### Task 2.3: Continuous Batching
- [ ] Request queue with priority levels
- [ ] Iteration-level scheduling (not request-level)
- [ ] KV cache memory pool management
- [ ] Graceful degradation (503 + Retry-After at capacity)
- [ ] Target: 16+ concurrent users with P99 < 2x P50
- **Note:** llama-server already has basic continuous batching built-in

### Task 2.4: Model Management UI
- [ ] React + TypeScript dashboard
- [ ] Pages: Dashboard, Models, Deploy, Monitor, Settings
- [ ] Upload → quantize → deploy → monitor workflow

### Task 2.5: KV Cache Quantization
- [ ] INT8 KV cache with per-head quantization
- [ ] PagedAttention (fixed-size pages, non-contiguous allocation)
- [ ] Prefix caching for shared system prompts
- [ ] Target: 50% KV cache memory reduction, < 0.5% perplexity degradation

### Task 2.6: Speculative Decoding
- [ ] Draft model selection (1-2B params, L2-cache-resident)
- [ ] K-token speculation with verification
- [ ] Auto-tune K based on acceptance rate
- [ ] Target: 1.5-2x throughput improvement

### Task 2.8: Quality Dashboard
- [ ] Perplexity + MMLU + HumanEval for all model × quantization combos
- [ ] Static site or embedded dashboard
- [ ] Auto-update via CI

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
- [ ] Multi-model routing (single endpoint, model via request param)
- [ ] Traffic splitting (percentage-based or user-based)
- [ ] Statistical comparison reporting

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
| `tests/test_cli.py` | 5 | All pass |
| `tests/test_quantize.py` | 18 | All pass |
| **Total** | **54** | **All pass** |
