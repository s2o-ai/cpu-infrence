# llama.cpp Feature Audit

> **Purpose:** Map every major claim and task in `TODO.md` against actual implementation status in the vendored llama.cpp codebase (`engine/src/llama/`). This prevents duplicating work and focuses custom engineering on true differentiators.

> **Last Updated:** March 2026 | **Audit Scope:** llama.cpp commit b8445 (pinned in TODO.md Task 1.1)

---

## 1. ✅ ALREADY IN LLAMA.CPP — Use for Free

These features are **fully implemented and production-ready** in the llama.cpp codebase. Do NOT rebuild — instead, wire them into your CLI/server via command-line flags.

| TODO.md Task | Feature | Status | Key Location | How to Use |
|---|---|---|---|---|
| **Task 2.6** | Speculative Decoding | ✅ Fully implemented | `common/speculative.cpp`<br>`common/speculative.h` | `--speculative` / `--draft-model` flags in server |
| | | | | 5 strategies: draft model, ngram-simple, ngram-map-k, ngram-mod, ngram-cache |
| | | | | + EAGLE3 variant for extended speculative |
| **Task 2.5** | KV Cache Quantization | ✅ Fully implemented | `src/llama-kv-cache.cpp` (line 22-23)<br>`common/arg.cpp` (lines 391-401) | `--cache-type-k F16\|Q8_0\|Q4_0` + `--cache-type-v F16\|Q8_0\|Q4_0` |
| | | | | Supported: F32, F16, BF16, Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0, Q5_1 |
| **Task 2.3** | Continuous Batching | ✅ Fully implemented | `tools/server/server-context.cpp` (slot loop)<br>`tools/server/server-context.h` (server_slot struct)<br>`tools/server/server-queue.cpp` | `--parallel N` flag (number of concurrent slots) |
| | | | | Iteration-level scheduling: new requests enter at prefill while others generate |
| | | | | No `max_queue_depth` backpressure yet (open feature request) |
| **Task 1.3** | OpenVINO Backend | ✅ Fully implemented | `ggml/src/ggml-openvino/ggml-openvino.cpp`<br>`ggml/src/ggml-openvino/openvino/translate_session.cpp`<br>`ggml/src/ggml-openvino/openvino/op/*` (per-op translators)<br>`ggml/src/ggml-openvino/openvino/pass/*` (graph optimization: SDPA fusion, matmul squeeze, zp elimination) | Build with `-DGGML_OPENVINO=ON` |
| | | | | Supports: Intel iGPU (Xe), Intel Arc, Intel NPU (Level Zero)<br>Full op coverage + graph-level fusion |
| **Task 1.2** | CPU Feature Detection (CPUID) | ✅ Fully implemented | `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`<br>`ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp`<br>`ggml/src/ggml-cpu/arch/s390/cpu-feats.cpp` | Automatic at runtime via `__cpuid()` (x86) / `getauxval()` (ARM) |
| | | | | Detects: AVX-512F/DQ/BW/VL/VNNI, AMX-TILE/INT8/FP16, NEON, SVE, SVE2, etc. |
| (All) | AVX-512 SIMD Kernels | ✅ Fully implemented | `ggml-cpu/arch/x86/quants.c`<br>24 quantization types with AVX-512 paths | All quantized dot products optimized: Q4_0, Q4_K, Q8_0, IQ4_XS, etc. |
| | | | | `VPSHUFB` (64 parallel lookups), `VPDPBUSD` (VNNI accumulation), `VPMADDUBSW` |
| (All) | ARM NEON Kernels | ✅ Fully implemented | `ggml-cpu/arch/arm/quants.c`<br>`ggml-cpu/kleidiai/kleidiai.cpp` | All quantized ops optimized for NEON (`SDOT`, `SMMLA`) |
| | | | | + KleidiAI integration for enhanced kernels |
| (All) | ARM SVE2 Support | ✅ Fully implemented | `ggml-cpu/arch/arm/quants.c` (SVE2 paths)<br>`ggml-cpu/simd-mappings.h` (type mappings) | `HWCAP2_SVE2` runtime detection + `GGML_USE_SVE2` compile flag |
| | | | | Variable-length vector register support |
| **Task 1.3** | AMX Backend (Intel) | ✅ Fully implemented | `ggml-cpu/amx/mmq.cpp`<br>`ggml-cpu/amx/amx.h`<br>`ggml-cpu/arch/x86/cpu-feats.cpp` | Automatic selection if `__AMX_INT8__` and `__AVX512VNNI__` available |
| | | | | Tile management: `_tile_loadd`, `_tile_dpbssd`, `_tile_stored` |
| | | | | Supports: Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS |
| **Task 2.5** | Prefix Caching (partial) | ⚠️ Per-slot only | `tools/server/server-context.cpp` (lines 2236-2278)<br>`tools/server/server-common.h` (get_common_prefix)<br>`server-task.h` (n_cache_reuse field) | `--cache-reuse N` flag; KV cache shifted via `llama_kv_cache_seq_rm/shift` |
| | | | | **Important:** Each slot independently checks its own history. NOT a global cross-slot prefix pool. |

---

## 2. ⚠️ S2O CUSTOM — IN PROGRESS (Complete These)

Proprietary S2O LUT kernel infrastructure with integer dot product path. Both AVX2 and AVX-512 kernels are functional and pass correctness tests. Still need VPSHUFB LUT path and weight repacking for the 1.5-2x target.

| TODO.md Task | Feature | Status | Location | What's Done | What's Missing |
|---|---|---|---|---|---|
| **Task 2.2** | LUT x86 AVX2 Kernel | ✅ Integer dot product | `ggml/src/ggml-cpu/s2o-lut/lut-x86-avx2.cpp` | On-the-fly FP32→INT8 quant, VPMADDUBSW+VPMADDWD dot product, pre-quantize activations once | VPSHUFB LUT path, weight repacking, cache tiling |
| **Task 2.2** | LUT x86 AVX-512 Kernel | ✅ Dual-column dot product | `ggml/src/ggml-cpu/s2o-lut/lut-x86-avx512.cpp` | 512-bit dual-column processing (2 cols/register), AVX512F+BW only (no DQ dep), VPMOVDB packing | VPSHUFB LUT path, weight repacking, cache tiling |
| **Task 2.2** | Backend integration | ✅ Complete | `ggml/src/ggml-cpu/s2o-lut/s2o-lut.cpp` | `extra_buffer_type` + `tensor_traits`, thread partitioning, Q4_0 support | Weight repacking in `set_tensor` (passthrough) |
| **Task 2.2** | Tests | ✅ All pass | `s2o-lut/test_lut.cpp` + `tests/test_lut.py` | 10 C++ tests (GEMV+GEMM, K=32..4096) + 18 Python integration tests | Need AVX-512 hardware tests |
| **Task 2.1** | LUT ARM NEON/SVE2 | ❌ Not started | — | — | Full kernel implementation needed |

**Key Code (current AVX2 hot loop):**
```cpp
// From lut-x86-avx2.cpp — integer dot product path:
// Pre-quantize activations to INT8 once, reuse across all output columns
for (int64_t b = 0; b < nb_k; b++) {
    __m256i qw = _mm256_sub_epi8(s2o_bytes_from_nibbles_32(wr[b].qs), off);
    __m256i qa = _mm256_loadu_si256((const __m256i *)(act_q8 + b * QK4_0));
    __m256 q = s2o_mul_sum_i8_pairs_float(qw, qa);  // VPMADDUBSW + VPMADDWD
    acc = _mm256_fmadd_ps(_mm256_set1_ps(combined_d), q, acc);
}
```

---

## 3. ❌ NOT IN LLAMA.CPP — Must Build from Scratch

### Performance Features (Phase 2)

| TODO.md Task | Feature | Why Not Upstream | Impact |
|---|---|---|---|
| **Task 2.1** | LUT ARM NEON/SVE2 Kernel | T-MAC patent; not part of llama.cpp | **Your biggest differentiator** — enables 2-4x on Graviton4 (no competing VNNI path) |
| | | | No VTBL-based INT4 kernel exists anywhere in llama.cpp |
| **Task 2.2** | VPSHUFB Vectorization (x86) | Scaffolded but incomplete | Needed to hit 1.3-2x target on Sapphire Rapids / Granite Rapids |
| **Task 2.5** | PagedAttention (global) | Design choice: vLLM pattern vs llama.cpp slot model | vLLM's block table / physical-virtual page mapping for better memory fragmentation |
| **Task 2.5** | Global Prefix Cache | Per-slot only in llama.cpp; cross-slot sharing not built | Shared KV pools for common system prompts across users |

### Enterprise Features (Phase 3+)

| TODO.md Task | Feature | Note |
|---|---|---|
| **Task 3.1** | SOC2 Compliance | Not in inference engine |
| **Task 3.2** | HIPAA Compliance | Not in inference engine |
| **Task 3.3** | Air-Gap Deployment | Not in inference engine |
| **Task 3.4** | Kubernetes Operator | Not in inference engine |
| **Task 3.5** | RBAC / SSO / Audit Logging | Not in inference engine |
| **Task 1.4** | Auto-Quantization Pipeline | HuggingFace integration + quality validation (perplexity, MMLU) |

---

## 4. Strategic Implications for TODO.md Timeline

### What Gets "Free" (Reuse, don't rebuild)
- **Task 2.3 (Continuous Batching):** Already working in `tools/server` — just expose `--parallel` flag
- **Task 2.5 (KV Cache Quant):** Already working — add `--cache-type-k/v` flags to CLI
- **Task 2.6 (Speculative Decoding):** Already working — expose `--speculative` + `--draft-model` flags
- **Task 1.3 (OpenVINO):** Already built — compile with `-DGGML_OPENVINO=ON`
- **Task 1.2 (CPU Detection):** Already automatic — just document it

### What Saves Phase 1 Effort
- Task 1.1 (Fork llama.cpp): **Already done** — you have commit b8445 vendored
- Task 1.5 (Benchmarking): **Partial** — framework exists, just needs multi-hardware CI
- Task 1.6 (CLI): **Mostly done** — Typer wrapper + Rich panels

### What Requires Real Engineering (Phase 2)
1. **ARM LUT NEON kernel** (4-5 weeks) — highest ROI, proven 2-4x gains
2. **x86 AVX-512 VPSHUFB completion** (2-3 weeks) — finish what's started
3. **Auto-quant pipeline** (2 weeks) — unblocks design partner demos

### What You DON'T Need to Do
- ~~Implement speculative decoding~~ — it's already there, just wire it
- ~~Implement KV cache quantization~~ — it's already there
- ~~Implement continuous batching~~ — it's already there
- ~~Implement OpenVINO backend~~ — it's already there
- ~~Implement CPU feature detection~~ — it's already there

---

## 5. Quick Reference: All 14+ Backends in llama.cpp

If you need offloading/acceleration beyond CPU:

| Backend | Status | Path | Use Case |
|---|---|---|---|
| CPU | ✅ Complete | `ggml/src/ggml-cpu/` | Your primary focus — includes S2O LUT |
| CUDA | ✅ Complete | `ggml/src/ggml-cuda/` | NVIDIA GPU (fallback if customer has GPU) |
| Metal | ✅ Complete | `ggml/src/ggml-metal/` | Apple Silicon (macOS) |
| OpenVINO | ✅ Complete | `ggml/src/ggml-openvino/` | Intel iGPU/Arc/NPU (your Task 1.3) |
| Vulkan | ✅ Complete | `ggml/src/ggml-vulkan/` | AMD/Intel/Mobile GPUs (cross-platform) |
| SYCL | ✅ Complete | `ggml/src/ggml-sycl/` | Intel oneAPI / Arc Alchemist |
| BLAS | ✅ Complete | `ggml/src/ggml-blas/` | OpenBLAS / MKL / Apple Accelerate (FP32 only) |
| AMX | ✅ Complete | `ggml/src/ggml-cpu/amx/` | Intel AMX-INT8 (integrated in CPU backend) |
| KleidiAI | ✅ Complete | `ggml/src/ggml-cpu/kleidiai/` | ARM KleidiAI optimized kernels (integrated in CPU) |
| HIP | ✅ Complete | `ggml/src/ggml-hip/` | AMD ROCm (CUDA-compatible variant) |
| WebGPU | ✅ Complete | `ggml/src/ggml-webgpu/` | Browser/WASM GPU |
| Hexagon | ✅ Complete | `ggml/src/ggml-hexagon/` | Qualcomm Hexagon DSP |
| CANN | ✅ Complete | `ggml/src/ggml-cann/` | Huawei Ascend NPU |
| zDNN | ✅ Complete | `ggml/src/ggml-zdnn/` | IBM z/OS mainframe |
| RPC | ✅ Complete | `ggml/src/ggml-rpc/` | Remote inference server |

---

## 6. Verification Checklist

- [ ] Confirm `engine/src/llama/common/speculative.cpp` exists and has `common_speculative_draft` function
- [ ] Confirm `engine/src/llama/src/llama-kv-cache.cpp` constructor takes `type_k` and `type_v` parameters
- [ ] Confirm `engine/src/llama/tools/server/server-context.cpp` has `slot-based` batching (search for `std::vector<server_slot>`)
- [ ] Confirm `engine/src/llama/ggml/src/ggml-openvino/ggml-openvino.cpp` exists
- [ ] Confirm `engine/src/llama/ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp` has CPUID detection
- [ ] Confirm `engine/src/llama/ggml/src/ggml-cpu/s2o-lut/lut-x86-avx512.cpp` exists and is the starting point
- [ ] Cross-check TODO.md Task claims against file locations above

---

## 7. Gaps vs. Public llama.cpp

What S2O AI brings that public llama.cpp does NOT:

1. **LUT INT4 kernels** (x86 + ARM) — proprietary optimization, T-MAC based
2. **Global prefix caching** (cross-slot) — llama.cpp only does per-slot
3. **Enterprise wrap** — RBAC, SSO, HIPAA, Kubernetes operator

Everything else in your TODO.md is either already in llama.cpp or will be in open-source fairly soon.

---

*Last verified against llama.cpp commit b8445 (pinned in Task 1.1)*
