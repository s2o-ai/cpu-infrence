# Phase 2: Differentiation — Implementation Plan

## Context

Phase 1 is complete: CPU detection, quantization pipeline, CLI, benchmarks, and tests are all working. The baseline is 212 tok/s prompt processing, 8.46 tok/s generation (Qwen3.5-0.8B Q4_K_M). Phase 2's goal is to differentiate S2O from stock llama.cpp through custom LUT kernels (1.5-2.5x speedup), enhanced serving (continuous batching, speculative decoding, KV cache optimization), a quality dashboard, and a model management UI. Success criteria: 1.5-2.5x over stock llama.cpp, LUT engine on both ARM and x86, first paying customer.

---

## Sub-Phase 2A: x86 LUT Kernels (Weeks 1-5) — CRITICAL PATH

The core innovation. INT4 LUT replaces multiply-accumulate with cached lookups. The 256-byte LUT fits in L1/registers. Uses VPSHUFB for parallel lookups.

### New files (inside llama.cpp fork)

| File | Purpose |
|------|---------|
| `engine/src/llama/ggml/src/ggml-cpu/s2o-lut/s2o-lut.h` | Public header: `ggml_backend_cpu_s2o_lut_buffer_type()` (pattern from `kleidiai/kleidiai.h`) |
| `engine/src/llama/ggml/src/ggml-cpu/s2o-lut/s2o-lut.cpp` | `extra_buffer_type` + `tensor_traits` impl (pattern from `kleidiai/kleidiai.cpp`, `traits.h`). Handles weight repacking at buffer alloc time, `supports_op()` for MUL_MAT on Q4_K_M/Q4_0, `compute_forward()` dispatch |
| `engine/src/llama/ggml/src/ggml-cpu/s2o-lut/lut-common.h` | LUT table generation from quant scales, tile size constants |
| `engine/src/llama/ggml/src/ggml-cpu/s2o-lut/lut-x86-avx512.cpp` | AVX-512 kernels: `s2o_lut_gemv_q4_avx512()` (single-row, VPSHUFB + VPDPBUSD), `s2o_lut_gemm_q4_avx512()` (batched for prompt). Cache-aware L2 tiling with `_mm_prefetch` |
| `engine/src/llama/ggml/src/ggml-cpu/s2o-lut/lut-x86-avx2.cpp` | AVX2 fallback (256-bit VPSHUFB, ~50% throughput of AVX-512) |
| `engine/src/llama/ggml/src/ggml-cpu/s2o-lut/test_lut.cpp` | C++ correctness: random Q4 matrices, compare LUT output vs reference, tolerance < 1e-5 |

### Modified files

| File | Change |
|------|--------|
| `engine/src/llama/ggml/src/ggml-cpu/CMakeLists.txt` | Add `GGML_S2O_LUT` option, conditionally compile `s2o-lut/*.cpp` for x86 |
| `scripts/build.py` | Add `--lut` flag → passes `-DGGML_S2O_LUT=ON` to CMake |
| `engine/detect/_recommend.py` | Refine `llama_cpp_lut` backend recommendation logic |

### New Python files

| File | Purpose |
|------|---------|
| `benchmarks/bench_lut.py` | A/B benchmark: stock vs LUT, comparison table with speedup ratio |
| `tests/test_lut.py` | Test `--lut` build flag, LUT backend selection, benchmark parsing |

### Key design decisions
- **Weight repacking at buffer alloc time** (like KleidiAI) — transparent to user, standard GGUF files work, ~1-3s load overhead
- **Additive to fork** — new `s2o-lut/` dir, guarded by `GGML_S2O_LUT`, no modifications to existing ggml files → clean upstream rebases
- **Extension via `traits.h`** — implement `ggml::cpu::extra_buffer_type` and `ggml::cpu::tensor_traits` interfaces

### Validation
- C++ test: LUT output matches reference within tolerance
- On AVX-512 (AWS AMD EPYC c6a): prompt processing speedup >= 1.3x for Qwen3.5-0.8B Q4_K_M
- On AVX2 (dev machine): no regression vs stock
- All existing 40+ tests pass

### Go/No-Go at Week 4
- < 1.1x → investigate memory bandwidth, tile sizing
- 1.1-1.3x → optimize hot loops with perf/VTune
- >= 1.3x → on track

---

## Sub-Phase 2B: ARM NEON/SVE2 LUT Kernels (Weeks 5-8)

Port LUT approach to ARM for Graviton3/4 cloud deployment.

### New files (inside llama.cpp fork)

| File | Purpose |
|------|---------|
| `engine/src/llama/ggml/src/ggml-cpu/s2o-lut/lut-arm-neon.cpp` | NEON kernel: VTBL/VTBX for LUT lookups + SDOT accumulation |
| `engine/src/llama/ggml/src/ggml-cpu/s2o-lut/lut-arm-sve2.cpp` | SVE2 kernel: `svtbl` for LUT + `svdot` accumulation |
| `engine/src/llama/ggml/src/ggml-cpu/s2o-lut/lut-selector.cpp` | Runtime micro-benchmark (100 iters small matmul) to auto-select fastest kernel variant. On x86: LUT-AVX512 vs AMX-INT8. On ARM: NEON vs SVE2 vs stock |
| `benchmarks/configs/graviton.yaml` | ARM benchmark config |

### Modified files

| File | Change |
|------|--------|
| `engine/src/llama/ggml/src/ggml-cpu/CMakeLists.txt` | Conditionally compile ARM sources when `GGML_SYSTEM_ARCH STREQUAL "ARM"` |
| `engine/src/llama/ggml/src/ggml-cpu/s2o-lut/s2o-lut.cpp` | Add ARM dispatch paths |
| `scripts/build.py` | Cross-compilation support (`--arch aarch64`) |
| `tests/test_lut.py` | ARM-specific test cases |

### Validation
- On Graviton3 (AWS c7g): >= 1.3x prompt speedup
- Auto-selector picks correct kernel per platform
- ARM correctness tests pass

---

## Sub-Phase 2C: Serving Enhancements (Weeks 5-9, parallelizable with 2B)

Leverages existing llama-server capabilities. Mostly Python orchestration work.

### 2C.1: Enhanced Batching & Monitoring (Weeks 5-6)

New `engine/serving/` module:

| File | Purpose |
|------|---------|
| `engine/serving/__init__.py` | Module init |
| `engine/serving/config.py` | `ServingConfig` dataclass: max_concurrent, priority_levels, kv_cache_type, degradation_threshold, etc. |
| `engine/serving/proxy.py` | Lightweight httpx reverse proxy: priority queuing (header-based), `/metrics` monitoring, 503 + Retry-After at capacity, `/v1/status` endpoint |
| `tests/test_serving.py` | Test proxy, config, degradation behavior |

Modified:
- `scripts/commands/serve_cmd.py` — add `--max-concurrent`, `--priority`, `--flash-attn`, `--kv-cache-type` options
- `scripts/serve.py` — pass new flags to llama-server

### 2C.2: KV Cache Optimization (Weeks 6-7)

Leverage llama.cpp's existing `type_k`/`type_v` params (marked EXPERIMENTAL in llama_context_params).

| File | Action |
|------|--------|
| `scripts/serve.py` | MODIFY — pass `--cache-type-k q8_0 --cache-type-v q8_0` when KV quant enabled |
| `scripts/commands/serve_cmd.py` | MODIFY — add `--kv-quant` flag |
| `benchmarks/bench_kv.py` | NEW — benchmark perplexity impact + memory savings at f16/q8_0/q4_0 |
| `engine/detect/_recommend.py` | MODIFY — recommend KV quant when < 32GB RAM and model >= 7B |

### 2C.3: Speculative Decoding (Weeks 7-9)

llama-server already supports `--model-draft` and tracks `n_draft_total/n_draft_accepted`.

| File | Action |
|------|--------|
| `engine/serving/speculative.py` | NEW — draft model selection (7B→0.5-1B, 13B→1-3B same family), acceptance rate monitoring, auto-tune K |
| `scripts/serve.py` | MODIFY — accept `--draft-model`, pass `--model-draft` to llama-server |
| `scripts/commands/serve_cmd.py` | MODIFY — add `--speculative`, `--draft-model` flags |
| `benchmarks/bench_speculative.py` | NEW — throughput comparison with/without speculation |
| `tests/test_speculative.py` | NEW — test draft model selection, config |

### Validation
- 16+ concurrent requests, P99 < 2x P50 (via `bench_server.py`)
- 503 + Retry-After when capacity exceeded
- KV quant (q8_0): ~40-50% memory reduction, < 0.5% perplexity degradation
- Speculative decoding: >= 1.3x throughput improvement

---

## Sub-Phase 2D: Quality Dashboard & Model Management UI (Weeks 8-12)

Lowest priority. Can be cut to 2D.1 only if time is tight.

### 2D.1: Quality Dashboard (Weeks 8-9)

| File | Purpose |
|------|---------|
| `engine/quality/__init__.py` | Module init |
| `engine/quality/evaluate.py` | Wraps perplexity (reuse `engine/quantize/_validate.py`) + MMLU via lm-eval-harness subprocess |
| `engine/quality/dashboard.py` | Jinja2 static HTML: model × quant × hardware → perplexity, MMLU, throughput |
| `engine/quality/templates/` | HTML templates |
| `scripts/commands/quality_cmd.py` | `s2o quality <model>` CLI command |
| `scripts/s2o.py` | Register quality command |
| `tests/test_quality.py` | Test evaluation and reporting |

### 2D.2: Model Management UI (Weeks 9-12) — CAN BE DEFERRED TO PHASE 3

React + TypeScript SPA with FastAPI management backend.

| File | Purpose |
|------|---------|
| `ui/` | React app (Vite + TS): Dashboard, Models, Deploy, Monitor, Settings pages |
| `engine/serving/api.py` | FastAPI management API: `/api/system`, `/api/models`, `/api/quantize`, `/api/server/start|stop|status` |
| `scripts/commands/ui_cmd.py` | `s2o ui` CLI command |
| `tests/test_api.py` | API endpoint tests |

---

## Timeline (1 engineer)

```
Week:  1    2    3    4    5    6    7    8    9   10   11   12
       |---- 2A: x86 LUT Kernels -------|
                                  |---- 2B: ARM LUT ----|
                                  |-- 2C.1 --|
                                       |- 2C.2 -|
                                            |-- 2C.3 --|
                                                 |- 2D.1 -|
                                                      |-- 2D.2 --|
```

With 2 engineers: Engineer 1 does 2A → 2B → 2D.2. Engineer 2 does 2C (weeks 5-9) → 2D.1 (weeks 9-10).

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| LUT kernels memory-bandwidth-bound | Start with micro-benchmarks before full model integration. Focus on cache tiling and prefetch. Fallback: AMX INT8 for Intel, KleidiAI for ARM |
| llama.cpp upstream breaks fork | LUT is additive (new dir, CMake guard). Monthly rebase + full test suite |
| No local ARM hardware | QEMU for correctness, Graviton instances for perf (~$10-20/session) |
| UI scope creep | UI is last priority. Cut to CLI-only quality dashboard (2D.1) if behind |

---

## Key Reference Files

- `engine/src/llama/ggml/src/ggml-cpu/traits.h` — `tensor_traits` + `extra_buffer_type` interfaces to implement
- `engine/src/llama/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` — reference pattern for custom backend (buffer type, weight repacking, compute dispatch)
- `engine/src/llama/ggml/src/ggml-cpu/amx/amx.cpp` — second reference pattern (x86-specific)
- `engine/src/llama/ggml/src/ggml-cpu/CMakeLists.txt` — where to register new backend sources
- `engine/detect/_recommend.py` — backend recommendation logic to update
- `scripts/build.py` — build orchestration to extend
- `engine/quantize/_validate.py` — perplexity validation to reuse in quality dashboard

## Verification

1. **Unit tests**: `pytest tests/` — all existing + new tests pass
2. **C++ correctness**: `test_lut.cpp` — LUT matches reference output
3. **Benchmark A/B**: `python -m benchmarks.bench_lut` — stock vs LUT comparison
4. **Serving load test**: `python -m benchmarks.bench_server` — 16 concurrent users
5. **Quality eval**: `python scripts/s2o.py quality <model>` — perplexity + MMLU scores
6. **End-to-end**: `s2o build --lut && s2o serve --model <model> --speculative` → send requests → verify speedup
