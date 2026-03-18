# Zero-GPU AI Inference Platform

**Production AI on the servers you already own. No GPUs. No cloud. Your data never leaves your building.**

Deploy production-grade AI models (7B-13B parameters) on existing enterprise CPU infrastructure without GPUs. A custom inference engine with LUT-based optimization delivers 1.5-2.5x performance over llama.cpp (latest release, compiled with `-march=native`, AMX/AVX-512 enabled) on INT4-quantized models running on enterprise-class Xeon and EPYC servers. Combined with an enterprise deployment platform that handles HIPAA/SOC2 compliance, monitoring, and multi-tenant serving.

This is not a general replacement for GPU inference at all model scales. It targets a specific, underserved niche: regulated organizations that need to run 7B-13B parameter models on existing hardware, where data cannot leave the premises.

**Initial launch vertical:** Healthcare (HIPAA-covered entities running clinical NLP).

---

## Documentation

| Document | Audience | Contents |
|----------|----------|----------|
| [Investor Pitch](docs/investor-pitch.md) | Investors, CTOs, board members, business buyers | Market opportunity, competitive landscape, go-to-market strategy, funding requirements, financials |
| [Technical Spec](docs/technical-spec.md) | Engineers, technical evaluators, architecture reviewers | GPU/CPU analysis, LUT innovation, CPU architectures, optimization techniques, benchmarking methodology |
| [Operations Plan](docs/operations-plan.md) | Internal team, ops leads, hiring managers | Product architecture, execution plan, risks, key metrics, team & hiring |
| [Technical Implementation Plan](docs/technical-implementation-plan.md) | Engineers, founders executing the build | Step-by-step tasks per phase, verification criteria, testing strategy, security checklists, dependency graph |

---

## Quick Links

- **Market size:** $15-25B SAM in regulated on-prem AI inference ([Investor Pitch](docs/investor-pitch.md#1-market-sizing))
- **Core innovation:** LUT-based matmul for 1.3-2x over llama.cpp's INT4 path ([Technical Spec](docs/technical-spec.md#2-the-core-innovation--lut-based-matmul-optimization))
- **Pricing:** Free -> $2K/mo -> $8-15K/mo -> Custom ([Investor Pitch](docs/investor-pitch.md#3-pricing))
- **Funding need:** $150-250K pre-seed, $1-2M seed ([Investor Pitch](docs/investor-pitch.md#4-funding-requirements--runway))
- **Execution phases:** 4 phases over 18 months ([Operations Plan](docs/operations-plan.md#2-execution-plan))
- **Benchmark methodology:** 3 hardware configs, transparent reporting ([Technical Spec](docs/technical-spec.md#6-benchmarking-methodology))
