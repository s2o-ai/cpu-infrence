# Zero-GPU AI Inference Platform — Technical Spec

> **Audience:** Engineers, technical evaluators, architecture reviewers
>
> For market and business context, see [Investor Pitch](investor-pitch.md). For execution and operations, see [Operations Plan](operations-plan.md).

---

## 1. Technical Foundation — How GPU/CPU Inference Works

### Why GPUs Are Fast (and Wasteful for Inference)

GPUs contain thousands of CUDA cores organized into Streaming Multiprocessors (SMs), plus specialized Tensor Cores for matrix multiplication. The NVIDIA software stack (CUDA Runtime API -> CUDA Driver API -> kernel driver -> GPU hardware) enables massive parallelism.

However, during LLM inference with batch size 1 (typical for real-time serving):
- The model generates ONE token at a time
- Each token requires loading billions of weights from HBM memory (~hundreds of nanoseconds per access)
- A tiny amount of math happens, then more weights are loaded
- GPU compute units sit idle 90%+ of the time — the bottleneck is memory bandwidth, not compute

### Why CPUs Can Compete (for the Right Workloads)

For small-medium models (7B-13B) with INT4 quantization:
- Model fits in ~4-8GB RAM (vs 14-28GB at FP16)
- CPU L3 cache (30-100MB) can hold significant working set
- Modern CPUs have specialized instructions: AVX-512 (512-bit vectors), AMX (matrix tiles), VNNI (INT8 dot products)
- CPU memory bandwidth on enterprise servers: 8-channel DDR5-4800 delivers ~307 GB/s theoretical, ~250 GB/s effective — sufficient for quantized 7B models where the bottleneck is ~30-60 GB/s of weight loading
- Cost comparison: AWS c7i.24xlarge (96 vCPU, 192GB, ~$3.06/hr) vs p4d.24xlarge (8xA100, ~$32.77/hr) = **10.7x cost difference** at the instance level. Per-token cost difference depends on model and batch size, but for batch-1 INT4 inference on 7B models, CPU is typically 3-5x cheaper per token

### The NVIDIA Lock-in (and How We Sidestep It)

NVIDIA's moat is CUDA — 20 years of software, 4 million developers, closed-source optimized libraries (cuBLAS, cuDNN). These libraries ship as pre-compiled SASS binaries locked to NVIDIA hardware.

Our approach sidesteps the CUDA ecosystem entirely, which means we have no dependency on NVIDIA's roadmap, pricing, or supply chain. The tradeoff is that we cannot leverage CUDA-optimized libraries and must build our own optimization stack:
- We run on CPUs — no CUDA dependency
- We build on open-source foundations (llama.cpp, OpenVINO, ONNX Runtime)
- Our optimization is at the CPU instruction level (AVX-512, AMX, NEON), not GPU-specific

### Model Size Ceiling — Where CPU Inference Hits Its Limits

CPU inference is not viable at all scales. Honest assessment of parameter-count boundaries:

| Model Size | CPU Viability | Hardware Required | Interactive Latency |
|-----------|--------------|-------------------|-------------------|
| 1-3B | Excellent | Any modern server | < 50ms first token |
| 7B (sweet spot) | Good | Single-socket Xeon/EPYC, 32GB+ RAM | 50-150ms first token |
| 13B | Viable | Dual-socket, 64GB+ RAM, DDR5 | 150-400ms first token |
| 30B | Marginal | Multi-node sharding, significant engineering | 500ms-2s first token |
| 70B+ | Impractical for interactive | Would require 4+ nodes, latency too high | > 5s first token |

**Our focus:** 7B-13B models, which cover the majority of enterprise NLP tasks (summarization, extraction, classification, Q&A). We do not claim to be a solution for frontier-scale models.

---

## 2. The Core Innovation — LUT-Based Matmul Optimization

### The Insight

AI inference is ~90% matrix multiplication by compute. With aggressive quantization (INT4), the multiplication table becomes tiny enough to cache:

| Precision | LUT Size | Fits In | Speedup vs. Naive Multiply |
|-----------|----------|---------|---------------------------|
| FP16 (65536 values) | 16 GB | Impossible | N/A |
| INT8 (256 values) | 256 x 256 = 64KB | L2 cache | 1.5-2x |
| INT4 (16 values) | 16 x 16 = 256 bytes | L1 cache / registers | 2-4x vs naive; **1.3-2x vs llama.cpp's optimized INT4 path** |
| INT2 (4 values) | 4 x 4 = 16 bytes | Registers | 3-5x vs naive (significant quality loss) |
| Ternary {-1,0,+1} | No LUT needed | N/A (add/sub/skip) | 5-10x vs naive (requires purpose-trained models) |

**Critical caveat:** The 2-4x speedup for INT4 LUT is measured against naive multiply-accumulate on the same hardware. llama.cpp already ships highly optimized INT4 kernels using AVX-512 VNNI. The real-world gain of our LUT approach over llama.cpp's existing INT4 codepath is **1.3-2x**, not 2-4x. This smaller but meaningful margin is our technical edge, and we must be honest about it.

### How It Works

1. **Quantize** model weights from FP16 to INT4 (16 possible values per weight)
2. **Pre-compute** all possible products: LUT[a][b] = a x b for all 16x16 combinations
3. **Replace multiplication with table lookup**: one memory read instead of an ALU operation
4. **Pack lookups**: two INT4 weights fit in one byte, so LUT[byte] handles two multiplications at once

### Prior Art (We're Not Alone)

- **T-MAC (2024)**: LUT-based matmul for LLMs on CPU — 2-4x faster than llama.cpp on ARM
- **BitNet b1.58 (Microsoft)**: Ternary weights {-1,0,+1} — multiplication becomes add/sub/skip
- **GPTQ/AWQ**: INT4 quantization used in llama.cpp — enables running 70B models on consumer hardware
- **Product Quantization (FAISS)**: Pre-computed dot products via codebook lookup — used at Meta scale
- **FPGAs**: Entire architecture built around lookup tables — Microsoft uses for Bing AI

### Where LUT Wins, Doesn't Win, and Is Uncertain

**Wins:**
- CPU inference (no tensor cores to compete with)
- ARM architectures (NEON + SDOT + LUT = very efficient)
- Low-bit quantization (INT4 and below)
- Edge/embedded devices (low power, no GPU)

**Doesn't win:**
- GPU inference (tensor cores already do 2048 multiply-adds per cycle — LUT adds memory pressure to an already memory-bound workload)
- High-precision (FP16+) — LUT too large to cache

**Uncertain / depends on implementation:**
- x86 with Intel AMX: AMX already does INT8 matrix multiply in dedicated silicon. Whether INT4 LUT outperforms AMX-accelerated INT8 depends on the quality/speed tradeoff for the specific model — INT4 uses less memory but AMX has hardware acceleration. We must benchmark both paths per-model.
- High batch sizes (>4): LUT is optimized for batch-1 (memory-bound regime). At higher batch sizes, the compute/memory ratio shifts and native multiply instructions may be faster. Our serving layer will auto-select the optimal path based on current batch size.

### LUT Kernel Defensibility

The LUT kernels themselves are open-source (Apache 2.0). This is intentional — open-source builds community trust and drives adoption. The defensibility comes from three layers built on top:

1. **Auto-tuning system:** Selects optimal LUT tile sizes, thread counts, and memory layouts per CPU microarchitecture. This requires months of profiling data across dozens of hardware configurations (Xeon 4th/5th Gen variants, EPYC Genoa/Turin, Graviton 3/4, various DDR4/DDR5 configurations). The tuning database is proprietary.
2. **Calibration dataset pipeline:** Finds quantization parameters that preserve quality for domain-specific models (e.g., clinical NLP, financial analysis). A generic INT4 quantization may lose critical domain knowledge. Our calibration pipeline uses customer-provided representative data to minimize domain-specific quality loss.
3. **Enterprise platform stack:** The LUT kernel is one component. The platform (multi-tenant serving, compliance, monitoring, auto-quantization) represents 12-18 months of engineering on top.

Any competitor can adopt the open-source kernels. Replicating the tuning infrastructure, calibration pipeline, and enterprise stack takes 12-18 months — and by then, we will have moved further ahead.

---

## 3. Inference Engine Comparison

### CPU Inference Engines Ranked

*All tok/s numbers measured on Llama 2 7B-Chat, Q4_K_M quantization, single-socket Intel Xeon 4th Gen (Sapphire Rapids, 48 cores), 256GB DDR5-4800, batch size 1. Numbers are approximate ranges across community benchmarks and may vary by specific SKU and memory configuration.*

| Engine | Best For | CPU Speed (7B Q4) | Strengths | Weaknesses |
|--------|----------|-------------------|-----------|------------|
| **llama.cpp** | Everything CPU | 15-50 tok/s | Zero deps, broadest HW support, best quant, active AVX-512/AMX optimization | No batching, no multi-tenant, no enterprise features |
| **OpenVINO** | Intel servers | 30-60 tok/s (w/ AMX) | 1.5-2x over llama.cpp non-AMX path on Intel, graph optimization | Intel-only, smaller ecosystem |
| **ONNX Runtime** | Cross-platform | 10-40 tok/s | Framework-agnostic, Microsoft backed | LLM features behind llama.cpp |
| **vLLM (CPU mode)** | Multi-user serving | 8-25 tok/s | PagedAttention, continuous batching | CPU is afterthought, slow |
| **Ollama** | Developer experience | Same as llama.cpp | One-command setup, model hub, massive developer mindshare | Adding enterprise features (Ollama Enterprise); risk to our open-source funnel |
| **T-MAC** | ARM + low-bit | 50-80 tok/s on ARM | LUT-based, 2-4x over llama.cpp on ARM | Research stage, narrow HW support |
| **CTranslate2** | Non-LLM models | Variable | Good for BERT/translation | Less LLM focus |

### Non-CPU Inference (Context for Competitive Positioning)

These are not direct competitors (they require specialized hardware we're trying to avoid), but customers will compare against them:

| Platform | Speed | Hardware Required | Our Positioning |
|----------|-------|------------------|----------------|
| **Cerebras Inference** | 900+ tok/s (70B) | Wafer-scale chip, cloud-only | We compete on data sovereignty, not raw speed |
| **Groq** | 500+ tok/s | Custom LPU silicon, cloud-only | Same — we serve customers who cannot use cloud |
| **SambaNova** | Variable | Custom dataflow hardware, on-prem option | Direct competitor for regulated on-prem. Differentiator: they require their own hardware; we run on existing servers |
| **Intel Gaudi 3** | Competitive with A100 | Intel accelerator card | Lower-cost GPU alternative. Erodes "GPUs too expensive" narrative but still requires new hardware procurement |
| **AMD ROCm (MI300X)** | Competitive with H100 | AMD GPU | ROCm is improving rapidly (6.x closes CUDA gap). Still requires GPU hardware our customers don't have |

### Recommendation for Our Stack

- **Intel servers detected** -> OpenVINO backend (exploit AMX matrix unit)
- **AMD servers detected** -> Custom llama.cpp fork with AVX-512 LUT kernels
- **ARM servers detected** -> T-MAC-inspired LUT engine with NEON/SVE2
- **Multi-user serving** -> Custom continuous batching layer on top
- **Portability fallback** -> ONNX Runtime as universal backend

**Important note on OpenVINO claims:** The "2-3x over generic" figure for OpenVINO on Intel is measured against non-AMX codepaths. When comparing against llama.cpp compiled with `-march=native` (which enables AMX on supported hardware), the gap narrows. Our benchmarks (see [Benchmarking Methodology](#6-benchmarking-methodology)) will always compare against the best available open-source option on each platform, not against strawman configurations.

---

## 4. CPU Architecture Analysis

### Intel Xeon (4th/5th Gen — Sapphire/Emerald Rapids)

**Key AI features:**
- AVX-512: 512-bit vectors, process 16 FP32 or 64 INT8 values per instruction
- **AMX (Advanced Matrix Extensions)**: Hardware matrix multiply — BF16 and INT8 matmul in dedicated silicon. This is Intel's "tensor core" equivalent for CPUs
- VNNI: Fused INT8 multiply-accumulate in one instruction
- Up to 128 cores, 2-8 sockets, up to 4TB RAM

**Cloud instances:** AWS m7i/c7i, Azure Dv5/Ev5, GCP c3
**Pricing:** $0.50-2.00/hr for 96 vCPU instance
**Best engine:** OpenVINO for AMX-accelerated INT8 path; our LUT engine for INT4 path. Must benchmark both per-model — AMX INT8 may outperform INT4 LUT on some models where INT8 quality is acceptable.

### AMD EPYC (Genoa/Turin — Zen 4/5)

**Key AI features:**
- AVX-512: Same vector width as Intel
- **No AMX** — this is the biggest gap vs Intel for AI workloads
- VNNI: INT8 acceleration similar to Intel
- Up to 192 cores — more cores than Intel
- Zen 5 (Turin) brings improved AVX-512 throughput and is worth separate benchmarking

**Cloud instances:** AWS m7a/c7a, Azure Dav5, GCP c2d
**Pricing:** $0.80-1.50/hr for 96 vCPU
**Best engine:** llama.cpp with AVX-512 optimized kernels (no AMX = our LUT approach has the clearest advantage here)

### ARM (AWS Graviton4 / Ampere Altra)

**Key AI features:**
- NEON: 128-bit vectors (always available)
- SVE2: Scalable vectors up to 2048-bit (Graviton4)
- SDOT/UDOT: Hardware INT8 dot product
- ~60% less power than x86 at similar performance

**Cloud instances:** AWS c8g (Graviton4) — ~40% cheaper than Intel equivalent
**Pricing:** $0.30-0.80/hr for 64+ vCPU
**Best engine:** T-MAC / custom LUT kernels (ARM + INT4 LUT is the sweet spot — no competing hardware matrix unit)

### Apple Silicon (M1-M4)

**Key AI features:**
- Unified memory (CPU + GPU share RAM) — up to 192GB on M4 Ultra
- Metal GPU + Neural Engine accessible alongside CPU
- NEON + AMX (Apple's own matrix extensions, not Intel AMX)

**Best for:** Local development, not server deployment
**Best engine:** MLX (Apple's framework) or llama.cpp with Metal

### Emerging Architectures (Monitor, Not Target)

Qualcomm Snapdragon X Elite and RISC-V vector extensions are emerging for edge inference. We monitor developments but do not target these in Phase 1-2. If ARM server adoption accelerates (via Graviton or Ampere), the ARM optimization work transfers naturally.

---

## 5. Optimization Techniques (Ordered by Impact)

### 1. Quantization — 2-4x speedup (HIGHEST IMPACT)

Reduce weight precision: FP16 -> INT8 -> INT4 -> INT2. Each step halves memory footprint and doubles effective bandwidth.

**Methods:**
- GPTQ: Post-training quantization, one calibration pass
- AWQ: Activation-aware, preserves important channels at higher precision
- GGUF: llama.cpp native format, supports mixed precision (Q4_K_M)

**Sweet spot:** Q4_K_M — 4-bit with important layers kept at higher precision. Quality retention varies by model (see Quality Assurance section below). On Llama 2 7B, Q4_K_M scores 5.89 perplexity on WikiText-2 vs 5.47 at FP16 (7.7% degradation). MMLU 5-shot accuracy drops 1-3 points depending on the subject area. This is acceptable for most enterprise NLP tasks but must be validated per use case.

**Our LUT advantage:** INT4 quantization produces only 16 possible weight values. The entire multiplication table (256 entries) fits in L1 cache. Replace hardware multiply with cache lookup.

### 2. SIMD Vectorization — 1.5-4x speedup

Use CPU vector instructions to process multiple values simultaneously:

- SSE (128-bit): 4 FP32 at once
- AVX2 (256-bit): 8 FP32 at once
- AVX-512 (512-bit): 16 FP32 or 64 INT8 at once
- AMX (tile-based): Full matrix multiply in hardware
- ARM NEON (128-bit) + SVE2 (scalable): Variable width vectors

**Key:** Compile with `-march=native`, hand-write critical inner loops with intrinsics.

### 3. Memory Layout Optimization — 1.3-2x speedup

- Tile matrices to fit in L1/L2 cache (32KB/1MB typical)
- Pack weights in SIMD-friendly interleaved layout
- Prefetch next tile while computing current tile (software pipelining)
- Align data to cache line boundaries (64 bytes)

### 4. KV Cache Optimization — 1.2-2x for long contexts

- Quantize KV cache: FP16 -> INT8 (50% memory savings)
- PagedAttention: Non-contiguous KV pages for efficient multi-user serving
- SnapKV: Evict low-attention keys to reduce cache size
- Prefix caching: Share KV cache for common system prompts

### 5. Thread Pinning & NUMA Awareness — 1.2-1.5x speedup

- Pin inference threads to specific physical cores
- NUMA-aware memory allocation: keep data near processing cores
- Avoid efficiency cores on Intel hybrid CPUs (3x slower)
- Isolate inference from OS scheduling jitter

### 6. Graph-Level Optimization — 1.1-1.3x speedup

- Operator fusion: Merge LayerNorm + Add + MatMul into single kernel
- Constant folding: Pre-compute static operations
- Dead code elimination: Remove unused computation paths
- OpenVINO and ONNX Runtime do this automatically

### 7. Speculative Decoding — 1.5-3x for token generation

- Small draft model (1B) predicts next 5 tokens
- Large model verifies all 5 at once (batched = more efficient)
- Especially powerful on CPU: draft model fits entirely in L2 cache
- Acceptance rate typically 60-80% = net 2-3x throughput gain

### 8. Structured Pruning — Variable

- Remove entire attention heads or FFN neurons with low contribution
- Requires fine-tuning after pruning to recover quality
- Can reduce model size by 30-50% with minimal quality loss

### Realistic Combined Impact

In theory, multiplying best-case individual speedups gives ~18x. **In practice, these optimizations interact and partially overlap:** SIMD vectorization is already used inside the quantized matmul kernel, so you don't get full multiplicative benefit. Cache tiling benefits diminish as the model gets smaller from quantization.

**Realistic combined speedup over naive FP32 baseline: 6-10x on well-optimized hardware.** This narrows the CPU-GPU gap from ~20x to roughly **2-5x for 7B models at batch size 1**, which is the regime where CPU inference is economically viable for regulated on-premises deployment.

### Quality Assurance at Each Quantization Level

Quality degradation from quantization is model-specific and task-specific. We commit to publishing quality scorecards for every supported model.

**Representative quality data (Llama 2 7B):**

| Quantization | Model Size | Perplexity (WikiText-2) | MMLU 5-shot | HumanEval pass@1 | Notes |
|-------------|-----------|------------------------|-------------|------------------|-------|
| FP16 (baseline) | 13.5 GB | 5.47 | 45.3% | 12.8% | Reference baseline |
| INT8 (Q8_0) | 7.2 GB | 5.50 | 45.0% | 12.5% | Near-lossless, 1.9x smaller |
| Q4_K_M | 4.1 GB | 5.89 | 43.5% | 11.2% | Sweet spot: good quality, 3.3x smaller |
| INT4 (Q4_0) | 3.8 GB | 6.15 | 42.1% | 10.5% | Basic INT4, more degradation |
| INT2 | ~2.0 GB | 8.5+ | 35-38% | 5-7% | Significant quality loss; only for classification/extraction |
| Ternary | ~1.5 GB | Model-specific | Model-specific | Model-specific | Requires purpose-trained models (BitNet) |

**Our commitments:**
- Publish a public quality dashboard showing perplexity and downstream task scores for every model we support at each quantization level
- Provide per-model recommendations: "For clinical note summarization with Llama 2 7B, we recommend Q4_K_M (acceptable quality) or INT8 (near-lossless, 1.5x slower)"
- Offer domain-specific calibration using customer-provided representative data to minimize quality loss on their specific tasks
- Maintain an honest "known limitations" list for models that do not quantize well (e.g., models with narrow weight distributions or specialized vocabularies)

---

## 6. Benchmarking Methodology

Benchmark credibility is our most important marketing asset. We commit to rigorous, transparent, and reproducible benchmarking.

### Hardware Test Matrix

All published benchmarks run on these three reference configurations:

| Config | CPU | Cores | RAM | Memory BW (effective) | Cloud Instance |
|--------|-----|-------|-----|----------------------|---------------|
| Intel Reference | Xeon 4th Gen (Sapphire Rapids) 8480+ | 2x 56 cores | 512GB DDR5-4800 | ~250 GB/s | AWS c7i.metal-48xl |
| AMD Reference | EPYC 9004 (Genoa) 9554 | 1x 64 cores | 256GB DDR5-4800 | ~200 GB/s | AWS m7a.16xlarge |
| ARM Reference | Graviton4 | 64 vCPU | 128GB DDR5 | ~180 GB/s | AWS c8g.16xlarge |

### Software Baselines

Every benchmark compares against:
- **llama.cpp** (latest release tag, compiled with `-march=native` to enable AMX/AVX-512/NEON on each platform)
- **OpenVINO** (latest release, for Intel configurations only)
- **ONNX Runtime** (latest release, as cross-platform reference)

We specify exact version numbers and commit hashes in every published benchmark. If llama.cpp ships a new optimization that closes our gap, we update benchmarks within 2 weeks.

### Reference Models

- Llama 2 7B-Chat (Q4_K_M and FP16) — standard benchmark model
- Mistral 7B Instruct v0.2 (Q4_K_M) — represents production instruction-following workload
- Phi-3 Mini (Q4_K_M) — represents smaller, efficient model class

### Metrics Reported

| Metric | What It Measures | Why It Matters |
|--------|-----------------|---------------|
| Tokens/second (generation) | Sustained generation rate, batch size 1 | Core throughput for interactive use |
| Time to first token (TTFT) | Prefill latency for a 512-token prompt | User-perceived responsiveness |
| "Time to 10 tok/s interactive" | From `curl install` to interactive chat | Our signature marketing metric |
| P50/P95/P99 latency | Tail latency at concurrency 1, 4, 16, 64 | SLA-critical for enterprise |
| Cost per million tokens | Instance cost / throughput | Economic viability comparison |
| Perplexity (WikiText-2) | Quality loss from quantization | Accuracy/speed tradeoff |
| MMLU 5-shot accuracy | Downstream task quality | Real-world capability retention |

### Reporting Rules

1. **Full reproduction scripts** published in our GitHub repo for every benchmark
2. **Median of 5 runs** with standard deviation reported (no cherry-picking best run)
3. **Warm cache**: First run discarded, report steady-state performance
4. **We benchmark against the best available open-source option on each platform**, not against strawman configurations. If llama.cpp with AMX enabled matches our performance, we say so.
5. **Negative results published**: If we lose on a specific configuration, we report it with explanation
6. Community members can run our benchmark suite on their own hardware and submit results
