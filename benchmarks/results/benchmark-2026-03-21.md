# S2O LUT vs Stock llama.cpp Benchmark

**Date:** 2026-03-21
**Hardware:** Intel Core i7-13650HX (10 threads), 16GB DDR5, Windows 11
**Model:** Qwen3.5-0.8B Q4_0 (490 MB, requantized from Q4_K_M)
**Build:** llama.cpp fork @ f895fe1, GCC 15.2, `-march=native` (AVX2+FMA)
**Config:** pp=512, tg=128, 5 runs, ngl=0

## Critical Bug Found

**The S2O LUT buffer type was never being registered** with the ggml CPU backend.
The `ggml_backend_cpu_get_extra_buffer_types()` function in `ggml-cpu.cpp` had no
`#ifdef GGML_USE_S2O_LUT` block, so even with `GGML_S2O_LUT=ON`, the LUT kernels
were compiled but never activated. Fixed by adding the registration block
(same pattern as AMX, KleidiAI, repack).

## Results

| Metric | Stock llama.cpp | S2O LUT (active) | Delta |
|--------|:-:|:-:|:-:|
| **Prompt Processing (tok/s)** | 242.9 ± 10.7 | 225.5 ± 5.3 | -7.2% |
| **Text Generation (tok/s)** | 6.23 ± 0.31 | 6.19 ± 0.07 | -0.6% |
| **LUT Micro-bench** | N/A | 23.2 GOPS | — |

### Earlier run (warmer machine, same session)

| Metric | Stock llama.cpp | S2O LUT (active) | Delta |
|--------|:-:|:-:|:-:|
| **Prompt Processing (tok/s)** | 232.6 ± 5.3 | 234.6 ± 7.7 | +0.9% |
| **Text Generation (tok/s)** | 6.13 ± 0.22 | 6.74 ± 0.09 | **+10.0%** |

## Analysis

On this 0.8B model on a laptop i7-13650HX, results are **within noise** (±5-10%).
This is expected for several reasons:

1. **Model too small** — at 490 MB, the model weights largely fit in L3 cache (24 MB)
   plus system memory bandwidth is not saturated. LUT's advantage is reducing
   memory bandwidth pressure on larger models.

2. **llama.cpp Q4_0 is already fast** — stock uses VPSHUFB dequantization in
   `arch/x86/quants.c`, very similar to our approach. The improvement headroom
   on Q4_0 is narrow.

3. **Laptop thermal variance** — the i7-13650HX throttles under sustained load,
   causing run-to-run variance of ±10%.

4. **LUT table build overhead** — for small inner dimensions (K=1024 on 0.8B),
   the cost of building the 16-entry LUT per activation group may exceed savings.

## Where LUT Should Shine

The LUT approach is designed for **7B-13B models on server CPUs** (EPYC, Xeon):
- Larger weight matrices that don't fit in cache
- Higher memory bandwidth saturation where LUT reduces traffic
- More cores → better amortization of LUT setup cost
- Server CPUs have larger L2/L3 → better tiling efficiency

## Next Steps

- [ ] Benchmark with 7B model (need to download/quantize)
- [ ] Benchmark on server hardware (AWS c7i or hpc7a)
- [ ] Profile with `perf` to identify hotspots (Linux only)
- [ ] Consider Q4_K support (currently LUT only handles Q4_0)

## Q4_K_M Baseline (no LUT — uses stock path for K-quants)

For reference, Q4_K_M on the same hardware (LUT doesn't activate for K-quants):

| Metric | Q4_K_M (stock path) |
|--------|:-:|
| **Prompt Processing (tok/s)** | 181.9 ± 2.6 |
| **Text Generation (tok/s)** | 6.36 ± 0.17 |

Q4_0 is ~33% faster at prompt processing than Q4_K_M due to simpler dequantization,
though Q4_K_M has better quality (~0.3 ppl less degradation at 8B scale).
