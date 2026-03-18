# Zero-GPU AI Inference Platform — Startup Plan & Technical Analysis

## Executive Summary

Deploy production-grade AI models (7B-13B parameters) on existing enterprise CPU infrastructure without GPUs. A custom inference engine with LUT-based optimization delivers 1.5-2.5x performance over llama.cpp (latest release, compiled with `-march=native`, AMX/AVX-512 enabled) on INT4-quantized models running on enterprise-class Xeon and EPYC servers. Combined with an enterprise deployment platform that handles HIPAA/SOC2 compliance, monitoring, and multi-tenant serving.

This is not a general replacement for GPU inference at all model scales. It targets a specific, underserved niche: regulated organizations that need to run 7B-13B parameter models on existing hardware, where data cannot leave the premises.

**One-liner:** "Production AI on the servers you already own. No GPUs. No cloud. Your data never leaves your building."

**Initial launch vertical:** Healthcare (HIPAA-covered entities running clinical NLP).

---

## 1. The Market Opportunity

### The Problem

- Enterprises want AI but face blockers: GPU procurement takes 6-12 months, costs $200K+ per server
- Regulated industries (healthcare, finance, government, defense) cannot send data to cloud AI APIs
- Every enterprise already has powerful CPU servers sitting at 20-30% utilization
- Existing open-source tools (llama.cpp, Ollama) are not enterprise-ready — no compliance, no multi-tenant serving, no management

### The Insight

GPU hardware was designed for graphics and adapted for AI. During LLM inference (generating tokens), GPU cores sit idle 90%+ of the time waiting for memory — the "memory wall." For small-to-medium models (7B-13B parameters) with aggressive quantization, optimized CPU inference closes the performance gap enough to be production-viable — defined as sustaining **10+ tokens/second for interactive use** and **30+ tokens/second for batch processing** on a single-socket server, at a fraction of the cost.

### Target Customers

| Segment | Why They Can't Use Cloud GPUs | Primary Use Case | Budget |
|---------|-------------------------------|------------------|--------|
| Healthcare & Hospitals | HIPAA, patient data cannot leave premises | Clinical note summarization, diagnostic support | $50-200K/yr |
| Banks & Financial Services | Customer data regulations, FINRA/SOX compliance | Document analysis, compliance screening | $100-500K/yr |
| Government & Defense | Air-gapped networks, ITAR/FedRAMP requirements | Intelligence summarization, translation | $200K-1M/yr |
| Legal Firms | Attorney-client privilege, data sovereignty | Contract review, legal research | $50-150K/yr |
| Pharmaceutical | IP protection, FDA validation requirements | Literature review, trial data analysis | $100-300K/yr |

### Latency vs. Throughput Requirements by Use Case

| Use Case | Acceptable First-Token Latency | Min Tokens/s | Batch OK? |
|----------|-------------------------------|-------------|-----------|
| Interactive chat (clinician Q&A) | < 100ms | 10+ | No |
| Document summarization | < 5s total response | 30+ | Yes |
| Coding assistant | < 200ms | 15+ | No |
| Background classification/extraction | Not user-facing | 5+ | Yes |

### Market Sizing

- **TAM:** Global AI inference market: ~$103-106 billion in 2025, projected ~$255 billion by 2030 (Source: MarketsandMarkets, Grand View Research, 2024-2025 reports)
- **SAM:** On-premises AI inference in regulated industries: ~$15-25 billion (estimated at ~15-25% of TAM based on regulated industry IT spend share)
- **SOM:** Realistic addressable market for a startup in years 1-3: $50-100 million (enterprise CPU inference for healthcare, finance, and government in North America/EU)

On-premises AI deployment is the fastest-growing segment within enterprise AI. Regulated industries represent ~40% of enterprise IT spending.

### Why Healthcare First

Healthcare is the optimal beachhead vertical for five reasons:

1. **HIPAA is binary** — you either comply or you cannot sell. This creates a hard barrier that eliminates casual competitors
2. **Clinical NLP models are 7B-scale** — the sweet spot for CPU inference. Tasks like clinical note summarization, ICD coding assistance, and discharge letter drafting work well with current-generation 7B models
3. **Hospitals already have server infrastructure** — most health systems operate on-premises data centers for EHR (Epic, Cerner) and PACS systems
4. **The buyer (CMIO/CIO) has budget authority** — healthcare IT budgets for AI adoption are growing 15-20% annually
5. **Measurable ROI** — clinician time savings from AI-assisted documentation are directly quantifiable ($50-100K/yr per provider in time savings)

**Phase 2 expansion:** Financial services (significant compliance overlap — SOC2 work done for healthcare transfers directly to SOX/FINRA requirements).

---

## 2. Technical Foundation — How GPU/CPU Inference Works

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

## 3. The Core Innovation — LUT-Based Matmul Optimization

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

## 4. Inference Engine Comparison

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

**Important note on OpenVINO claims:** The "2-3x over generic" figure for OpenVINO on Intel is measured against non-AMX codepaths. When comparing against llama.cpp compiled with `-march=native` (which enables AMX on supported hardware), the gap narrows. Our benchmarks (see Section 11) will always compare against the best available open-source option on each platform, not against strawman configurations.

---

## 5. CPU Architecture Analysis

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

## 6. Optimization Techniques (Ordered by Impact)

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

## 7. Product Architecture

### Tech Stack

```
+---------------------------------------------------+
|  OpenAI-Compatible REST API                        |  <- Drop-in replacement
+---------------------------------------------------+
|  Serving Layer: Router + Continuous Batching        |  <- Multi-user, queue management
+---------------------------------------------------+
|  Auto-Quant Pipeline: FP16 -> Optimized INT4       |  <- Customer gives model, gets optimized binary
+---------------------------------------------------+
|  Inference Engine (multi-backend)                   |
|  +----------+ +----------+ +------------------+    |
|  | OpenVINO | | llama.cpp| | Custom LUT Engine|    |  <- Auto-detect CPU, pick best backend
|  | (Intel)  | | (AMD)    | | (ARM/AVX-512)    |    |
|  +----------+ +----------+ +------------------+    |
+---------------------------------------------------+
|  Enterprise Features                                |
|  Audit logs . RBAC . Model versioning . A/B test    |  <- What you actually sell
|  SOC2/HIPAA . Air-gap support . K8s operator        |
+---------------------------------------------------+
```

### Open-Core Model

| Component | License | Purpose |
|-----------|---------|---------|
| Local CLI tool | Open source (Apache 2.0) | Growth engine — developer adoption |
| Inference engine | Open source (Apache 2.0) | Trust building, community contributions |
| Enterprise platform | Commercial license | Revenue — $50-200K/year per customer |
| Custom LUT kernels | Open source (Apache 2.0) | Community trust and adoption; differentiation comes from the auto-tuning and calibration stack built on top (proprietary) |

### Supported Models at Launch

**Phase 1-2 supported models (validated for quality at INT4):**
- Llama 2 7B / Llama 3 8B — most widely deployed open-source LLMs, quantize well
- Mistral 7B Instruct — strong instruction-following, efficient architecture
- Phi-2 / Phi-3 Mini — Microsoft's small models, excellent quality per parameter
- Qwen 2 7B — strong multilingual support, important for international deployments

**Models requiring additional validation before support:**
- Llama 2/3 13B — viable on dual-socket servers but memory constraints on single-socket; quality validation needed per-task
- Code Llama 7B — specialized vocabulary may quantize differently; coding benchmarks required
- Domain-specific fine-tuned models — quality must be verified per-model with customer's evaluation criteria

**Models explicitly out of scope for CPU inference:**
- 70B+ parameter models at interactive latency
- Image generation models (Stable Diffusion, FLUX)
- Multimodal models with vision encoders (LLaVA, GPT-4V-class)
- Real-time speech models (Whisper is viable for batch, not real-time streaming)

### Retention & Churn Strategy — Why Customers Stay

The open-source engine is free. What prevents customers from self-hosting with the open-source components?

1. **Compliance documentation and audit trail** — SOC2/HIPAA compliance requires documented controls, audit logs, access policies, and incident response procedures. Rebuilding this costs $100K+ and 6+ months of dedicated security/compliance engineering
2. **Multi-tenant serving with RBAC and SSO** — SAML/OIDC integration, per-team model access controls, and usage quotas. Significant engineering to build correctly and securely
3. **Hardware-specific auto-tuning** — Our tuning pipeline automatically selects optimal LUT tile sizes, thread pinning, and memory layout for the customer's specific CPU model and memory configuration. Self-tuning requires deep systems knowledge
4. **Quality monitoring and alerting** — Automated detection of model quality degradation, output anomaly detection, and latency SLA monitoring. Enterprise operations, not just inference
5. **SLA-backed support** — 4-hour response time for production issues, dedicated customer success engineer, quarterly business reviews

Each of these individually could be rebuilt. Together, they represent 12-18 months of engineering and $500K+ in compliance costs. The rational build-vs-buy decision favors buying for any organization that values time-to-production over cost minimization.

---

## 8. Go-to-Market Strategy

### Positioning by Audience

**To a CTO:** "Deploy AI models on your existing servers in 30 minutes. No GPU procurement. No cloud dependency. HIPAA compliant."

**To VP Engineering:** "OpenAI-compatible API that runs on your Xeon/EPYC fleet. Your developers change one URL. Everything else stays the same."

**To Investors:** "Enterprise AI deployment without NVIDIA. We turn existing enterprise servers into AI infrastructure with optimization delivering 1.5-2.5x performance over the best open-source alternative on the same hardware. The moat is compliance certifications + customer-specific calibration, not just the engine."

### Distribution: Open Source -> Enterprise Funnel

1. **Open-source local CLI** -> developers use it at home/work
2. **Developer blogs about speed** -> word of mouth, HN/Reddit posts
3. **Developer's company wants it on their servers** -> enterprise conversation begins
4. **"Can we get compliance, multi-user, support?"** -> commercial license discussion
5. **$24-36K first-year deal** (Team tier) with expansion clause. Land with one department, prove value, expand to enterprise license over 6-12 months. Expect **6-9 month sales cycle** for first regulated enterprise deal.

### Pricing

| Tier | Price | Typical Customer | Includes |
|------|-------|-----------------|----------|
| Community | Free | Individual developers, students, researchers | CLI tool, single-user, no support, no compliance features |
| Team | $2K/month | Engineering teams (5-20 users) at startups or within departments | Multi-user API, monitoring, email support, basic auth |
| Enterprise | $8-15K/month | Regulated organizations (50+ users) with compliance requirements | Full compliance (SOC2/HIPAA), air-gap, SLA, SSO/RBAC, dedicated support |
| Custom | Negotiated | Large health systems, banks, government agencies | On-site deployment, custom model calibration, training, audit support |

### Beachhead: Healthcare Go-to-Market

**Channel strategy:** Partner with 1-2 healthcare IT consulting firms (Nordic, Tegria, or similar) for warm introductions to CIO/CMIO buyers. These firms already have trusted relationships and can position our product within larger digital transformation initiatives.

**Target profile:** Community hospitals (200-500 beds) — large enough to have on-premises IT infrastructure and AI budget, small enough to make purchasing decisions in < 6 months (vs. 12-18 months at large academic medical centers).

**Proof point needed:** One reference customer with a published case study: "Hospital X reduced clinician documentation time by Y% using on-premises AI, maintaining full HIPAA compliance, at Z cost."

### Signature Benchmark

Our primary marketing metric: **"Time from download to 10 tok/s interactive chat on your existing hardware."**

This is concrete, reproducible, and any prospect can verify it on their own servers. Target: **< 15 minutes** on any supported CPU architecture. This includes download, auto-detection of CPU features, model quantization, and first interactive response.

---

## 9. Competitive Landscape

| Competitor | What They Do | Their Strength Against Us | Our Differentiation |
|------------|-------------|--------------------------|-------------------|
| **Ollama** | Free local inference with exceptional DX | Massive developer mindshare. Adding enterprise features (Ollama Enterprise). Could ship multi-tenant + basic auth before us. | Compliance certifications (SOC2/HIPAA), hardware-specific optimization, domain calibration. Must stay ahead on enterprise features. |
| **LM Studio** | Free GUI for local models | Beautiful UX, growing user base | Not targeting server deployment. Different market entirely. |
| **vLLM** | Production GPU serving | Production-proven at massive scale. PagedAttention, continuous batching. Industry standard for GPU serving. | No GPU requirement, on-prem data sovereignty. Different hardware target. |
| **NVIDIA NIM** | GPU-optimized containers | Best performance on NVIDIA hardware, strong enterprise sales org | Requires NVIDIA GPUs customers don't have. We run on existing servers. |
| **Cerebras Inference** | Wafer-scale chip inference | 900+ tok/s on 70B — redefines speed baselines customers compare against | Cloud-only. We compete on data sovereignty, not raw speed. |
| **Groq** | Custom LPU silicon inference | 500+ tok/s, lowest latency available | Cloud-only. Cannot serve regulated on-prem customers. |
| **SambaNova** | Dataflow architecture, on-prem option | Direct competitor for regulated on-prem AI. Enterprise sales team. | They require their own proprietary hardware ($$$). We run on servers you already own. |
| **Intel Gaudi 3** | Lower-cost GPU alternative | Competitive with A100 at lower price. Intel's own enterprise sales channel. | Still requires new hardware procurement and installation. Months of lead time. |
| **AMD ROCm (MI300X)** | AMD GPU ecosystem | ROCm 6.x closing CUDA gap rapidly. MI300X competitive with H100. | Still GPU hardware. Our customers can't/won't procure accelerators. |
| **Hugging Face TGI** | Production serving framework | Well-known brand, wide model support | Entering maintenance mode (Dec 2025). GPU-focused. |
| **Anyscale / Together** | Cloud inference APIs | Easy to use, competitive pricing, wide model selection | Cloud-dependent. Data leaves premises. Non-starter for regulated customers. |
| **Customer's own engineering team** | Build internally using open-source tools | Zero licensing cost, full control, uses llama.cpp/Ollama directly | 12-18 months of engineering for enterprise features. $500K+ in compliance costs. No dedicated support or SLA. **This is our most common "competitor."** |

### Competitive Response Playbook

When a prospect says "Why not just use X?":

**"We'll just use Ollama / llama.cpp internally"** -> "You absolutely can for prototyping. When you need SOC2 audit trails, HIPAA-compliant access controls, multi-user serving with SSO, and a vendor to point auditors at — that's where we come in. We use llama.cpp under the hood. You get the same engine with enterprise operations on top."

**"Cerebras/Groq is 100x faster"** -> "They are, and if your data can live in their cloud, they're a great option. Our customers chose us because their data cannot leave the building. We're not competing on speed — we're competing on sovereignty."

**"We could just buy Intel Gaudi / AMD MI300X"** -> "You could, in 6-12 months when procurement completes. We deploy on the Xeon servers in your data center today. Start generating value now, evaluate dedicated AI hardware in parallel."

**"SambaNova does on-prem too"** -> "They do, with proprietary hardware starting at $XXX,000. We run on the servers you already own. Try our free tier on your existing infrastructure in 15 minutes."

---

## 10. Benchmarking Methodology

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

---

## 11. Funding Requirements & Runway

### Pre-Seed / Bootstrapping (Months 0-3): $150-250K

| Item | Monthly | 3-Month Total |
|------|---------|--------------|
| 2 founders (reduced salary) | $16K | $48K |
| Cloud compute (benchmarking, CI/CD) | $5K | $15K |
| SOC2 readiness tooling (Vanta/Drata) | $3K | $10K |
| Legal / incorporation | — | $5K |
| Travel (design partner meetings) | $3K | $10K |
| Equipment / software licenses | — | $10K |
| Contingency (15%) | — | $15K |
| **Total** | | **$113-150K** |

**Source:** Personal savings, angel investors, or pre-seed fund. Some costs can be deferred (cloud credits from AWS/GCP startup programs can offset $10-15K).

### Seed Round (Months 3-6): $1-2M Target

**Purpose:** Hire first 4 employees, begin SOC2 Type 1 audit, build enterprise platform, fund 12 months of operations.

| Item | Monthly Burn |
|------|-------------|
| Salaries (6 people, see Team section) | $95K |
| Benefits / payroll tax (~25%) | $24K |
| Cloud infrastructure | $8K |
| SOC2 audit + compliance tools | $8K (averaged over 12mo) |
| Office / co-working | $3K |
| Marketing / content / events | $5K |
| Legal / accounting | $3K |
| **Total monthly burn** | **~$146K** |

- $1.5M seed = **~10 months runway** at full burn
- $2.0M seed = **~14 months runway** at full burn
- Revenue from design partners (starting month 6-9) extends runway by 2-4 months

### Compliance Cost Detail

SOC2 and HIPAA compliance is not optional for regulated enterprise sales. Detailed cost breakdown:

| Item | Cost | Timeline |
|------|------|----------|
| Compliance platform (Vanta/Drata) | $15-25K/yr | Month 1 onward |
| SOC2 Type 1 auditor | $30-80K | Months 4-7 (3-month process) |
| SOC2 Type 2 auditor | $50-100K | Months 10-16 (12-month observation) |
| HIPAA risk assessment + policies | $20-40K | Months 6-9 |
| Penetration testing (annual) | $15-30K | Month 8 |
| **Total compliance (first 18 months)** | **$150-275K** | |

### Series A Trigger (Months 12-18)

Raise Series A ($5-10M) when we can demonstrate:
- $500K+ ARR (achieved by month 15-18, not month 12)
- 10+ enterprise customers with < 5% monthly churn
- Proven unit economics (LTV:CAC > 3:1)
- Repeatable sales playbook in at least one vertical (healthcare)

### Key Financial Assumption

Two founders can build the MVP engine, publish benchmarks, and close 1-2 design partner LOIs before needing the seed round. **If this assumption fails** (no design partners by month 4), the plan requires either: (a) a larger pre-seed ($300-400K) to extend runway, or (b) pivoting to a developer-community-first strategy where open-source adoption drives inbound enterprise interest (slower but less capital-intensive).

---

## 12. Execution Plan

### Phase 1: Foundation (Months 1-3)

**Goal:** Working product, 1-2 design partner LOIs

- [ ] Fork llama.cpp, add OpenAI-compatible API server
- [ ] Implement auto-detection of CPU features (AMX/AVX-512/NEON)
- [ ] Add OpenVINO backend for Intel Xeon with AMX
- [ ] Build basic auto-quantization pipeline (HuggingFace -> optimized GGUF)
- [ ] Deploy on AWS reference hardware (all 3 configs) for benchmarking
- [ ] Publish initial benchmarks using methodology from Section 10, including quality scores
- [ ] Identify 10 target prospects in healthcare vertical
- [ ] Secure 1-2 design partner LOIs (assumes founder has warm introduction path into 2-3 health systems)

**Key metric:** 1.5x+ performance over stock llama.cpp (with `-march=native`) on 7B model

**Go/No-Go gate:** If we cannot achieve 1.5x over llama.cpp (with AMX/AVX-512 enabled) on our reference Intel hardware by month 3, the technical thesis needs revisiting before proceeding to Phase 2. Options: (a) focus on ARM where LUT advantage is clearest, (b) pivot to pure platform play without custom engine, (c) investigate alternative optimization approaches.

### Phase 2: Differentiation (Months 3-6)

**Goal:** LUT engine, enterprise alpha, first revenue

- [ ] Implement custom LUT kernels for INT4 on ARM (NEON/SVE2)
- [ ] Implement custom LUT kernels for INT4 on x86 (AVX-512)
- [ ] Add continuous batching for multi-user serving
- [ ] Build model management UI (upload, quantize, deploy, monitor)
- [ ] Implement KV cache quantization (INT8) for memory efficiency
- [ ] Add speculative decoding support
- [ ] Begin SOC2 Type 1 audit process (**critical dependency for Phase 3 sales**)
- [ ] Publish quality retention data (perplexity/MMLU) for all supported model+quantization combinations
- [ ] Close first paying design partner ($2-5K/month pilot)

**Key metric:** 1.5-2.5x performance over stock llama.cpp, first revenue (even if small)

**Go/No-Go gate:** If SOC2 Type 1 process has not begun by month 5, Phase 3 enterprise sales timeline must be pushed back accordingly. Each month of delay in starting the audit = one month delay in enterprise revenue.

### Phase 3: Enterprise (Months 6-12)

**Goal:** Production enterprise product, first enterprise customers

- [ ] Complete SOC2 Type 1 certification
- [ ] HIPAA compliance documentation and controls
- [ ] Air-gap deployment mode (no internet required)
- [ ] Kubernetes operator for automated deployment
- [ ] RBAC, SSO (SAML/OIDC), audit logging
- [ ] A/B testing framework for model comparison
- [ ] Multi-model serving (route different queries to different models)
- [ ] Hire customer success (1 person)
- [ ] Close 3-5 enterprise customers ($50-80K/year each)
- [ ] Publish first customer case study

**Key metric:** $150-250K ARR by month 12, < 5% monthly churn

**Go/No-Go gate:** If fewer than 2 paying customers by month 9, reassess sales strategy. Options: (a) hire enterprise sales earlier, (b) lower price point to reduce friction, (c) expand to financial services vertical in parallel.

### Phase 4: Scale (Months 12-18)

**Goal:** Repeatable sales, expanding vertical reach

- [ ] Complete SOC2 Type 2 certification (12-month observation period completes)
- [ ] Expand to AMD EPYC-specific optimizations (Zen 5 + AVX-512)
- [ ] Support for larger models (13B-30B) with model sharding across CPU nodes
- [ ] Edge deployment support (embedded ARM, industrial)
- [ ] Begin FedRAMP preparation (12-18 month process — will complete in Phase 5)
- [ ] Partnership exploration with Dell/HPE for pre-installed bundles
- [ ] Expand to financial services vertical
- [ ] Series A fundraise based on revenue traction

**Key metric:** $500K-1M ARR (base case), $1.5M+ ARR (stretch). 10-15 enterprise customers.

**Dependency map:**
- SOC2 Type 2 requires Type 1 (Phase 2) + 12 months observation
- FedRAMP requires SOC2 Type 2 as a foundation — cannot start until Phase 4 at earliest
- Financial services expansion leverages SOC2/compliance work from healthcare vertical

---

## 13. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **GPU prices collapse**, making CPU irrelevant | Medium | High | Data sovereignty value prop survives regardless of GPU pricing. Even free GPUs don't solve "data can't leave the building." |
| **llama.cpp adds enterprise features** (multi-tenant, auth, compliance) | Medium | High | Our 6-12 month head start on SOC2/HIPAA certifications is hard to replicate quickly. Compliance audits cannot be rushed. Continue contributing upstream; position our platform as the enterprise distribution of llama.cpp. |
| **Ollama ships enterprise tier** with compliance and multi-tenant | Medium-High | High | Ollama has massive mindshare. If they ship compliance features, we must differentiate on hardware optimization, domain calibration, and deeper vertical expertise (healthcare-specific features). Monitor their enterprise roadmap closely. |
| **NVIDIA releases CPU inference product** | Low | High | Open-source trust + existing customer relationships + compliance head start. NVIDIA's incentive is to sell GPUs, not optimize CPUs. |
| **Quantization quality insufficient** for customer's specific use case | High | High | Support multiple quantization levels (INT8 through INT4) with published quality scores. Offer domain-specific calibration using customer data. Maintain honest "known limitations" list. Some use cases may require INT8 (slower but higher quality). |
| **Enterprise sales cycle too long** (> 9 months to first deal) | High | High | Pre-qualify prospects on decision-making speed. Target department-level buyers (radiology AI lead, clinical informatics) rather than organization-wide IT purchases. Offer 3-month paid pilot at $5K/month to de-risk the buyer's decision. |
| **Founder sales bottleneck** | High | Medium | Hire enterprise sales by month 4. Build repeatable sales playbook with healthcare-specific case studies by month 6. Founder should be selling, not closing — hire closers. |
| **Open-source community forks our engine** | Medium | Low | A fork validates the approach and increases ecosystem awareness. The engine alone is not valuable without auto-tuning, calibration, compliance, and platform. We contribute upstream and maintain the canonical distribution. |
| **Intel/AMD release inference-optimized CPU instructions** that make LUT unnecessary | Low-Medium | Medium | Our value is the platform, not any single optimization technique. If native instructions surpass LUT, we adopt them into our engine and maintain the platform advantage. The LUT approach is one tool, not the entire business. |
| **Key engineer departure** | Medium | High | Document all custom kernel implementations. Ensure at least 2 people understand each critical system component. Competitive compensation. |

---

## 14. Key Metrics to Track

### Technical

- **Tokens/second per CPU core** — our engine vs llama.cpp (latest, `-march=native`) vs OpenVINO
- **"Time to 10 tok/s interactive"** on reference hardware (our signature metric)
- Cost per million tokens on each CPU architecture
- P50/P95/P99 latency at concurrency levels 1, 4, 16, 64
- Model quality retention at each quantization level (perplexity, MMLU)

### Quality

- Perplexity delta (our quantized vs FP16 baseline) per supported model
- MMLU score delta per supported model
- Customer-reported quality satisfaction score (quarterly survey)
- Number of models with published quality scorecards

### Competitive

- Monthly benchmark against latest llama.cpp release (with AMX/AVX-512 path)
- Ollama enterprise feature release tracking
- Cerebras/Groq/SambaNova pricing changes
- New CPU instruction set announcements (Intel/AMD/ARM roadmaps)

### Business

- Design partner -> paid customer conversion rate
- ARR and MRR growth
- Customer acquisition cost (CAC) by vertical
- Net revenue retention (target: > 120%)
- Time to first value (deployment -> production)

### Funnel

- Open-source CLI downloads (weekly)
- CLI -> enterprise inquiry conversion rate
- Average sales cycle length by vertical
- Pilot -> full contract conversion rate

---

## 15. Team Needed (First 18 Months)

### Core Team

| Role | When | Why | Estimated Total Comp |
|------|------|-----|---------------------|
| Founder/CEO | Day 1 | Product vision, enterprise sales (initially), fundraising | $8K/mo (reduced) -> market rate post-seed |
| Co-founder / Systems Engineer | Day 1 | LUT kernels, SIMD intrinsics, benchmarking, engine architecture | $8K/mo (reduced) -> market rate post-seed |
| Backend Engineer | Month 2 | API server, serving layer, infrastructure, auto-quant pipeline | $180-220K/yr |
| Enterprise Sales | Month 4 | Outbound to healthcare systems, relationship building in regulated industries | $120-150K base + commission |
| DevRel / Community | Month 4 | Open-source adoption, content, developer trust, benchmark publications | $140-170K/yr |
| Security/Compliance Engineer | Month 6 | SOC2, HIPAA, FedRAMP preparation, security architecture | $170-210K/yr |

### Advisory Board

| Role | Why | Compensation |
|------|-----|-------------|
| Healthcare CIO/CMIO | Vertical credibility, warm introductions to health systems, validates product-market fit | 0.25-0.5% equity |
| CPU Architecture Expert (Intel/AMD alumni) | Validates technical approach, helps optimize for upcoming CPU features | 0.25-0.5% equity |
| Enterprise Sales Advisor | Sales playbook, intro to buyers, helps hire first sales rep | 0.25-0.5% equity |

### Hiring Priorities if Constrained

If seed round is smaller than expected ($1M instead of $2M), defer DevRel to month 6 and have founders handle community/content. Security/Compliance hire is non-negotiable — compliance is the moat.

---

## Summary

The opportunity is clear: regulated enterprises need AI but can't use cloud GPUs. They have powerful CPU servers sitting underutilized. We build the bridge — a production-grade inference platform that turns their existing hardware into AI infrastructure, with the compliance they require and the performance that makes it viable.

The technical edge is LUT-based optimization that delivers **1.5-2.5x over the best open-source alternative** (llama.cpp with full hardware optimization enabled) on the same CPU hardware. This is a meaningful but honest improvement — not a 10x leap, but enough to differentiate when combined with the platform.

The business moat is multi-layered and compounds over time:
- **Compliance certifications** (SOC2/HIPAA) — 12+ months and $150K+ to replicate
- **Customer-specific calibration data** — domain-tuned quantization that improves quality on their specific tasks
- **Switching costs** — re-integrating compliance, RBAC, monitoring, and deployment pipelines with a new vendor
- **Auto-tuning pipeline** — hardware-specific optimization database built from profiling across dozens of configurations

This plan targets **7B-13B parameter models**, launches in **healthcare** as the beachhead vertical, and charts a realistic **18-month path to $500K ARR**. It is not a plan to replace GPU inference broadly — it is a plan to own the regulated on-premises CPU inference niche, and expand from there.

Open-source the engine. Sell the platform. Win the enterprise.
