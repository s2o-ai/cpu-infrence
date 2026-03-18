# Zero-GPU AI Inference Platform — Operations Plan

> **Audience:** Internal team, ops leads, hiring managers, execution tracking
>
> For market and business context, see [Investor Pitch](investor-pitch.md). For technical details, see [Technical Spec](technical-spec.md).

---

## 1. Product Architecture

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

## 2. Execution Plan

### Phase 1: Foundation (Months 1-3)

**Goal:** Working product, 1-2 design partner LOIs

- [ ] Fork llama.cpp, add OpenAI-compatible API server
- [ ] Implement auto-detection of CPU features (AMX/AVX-512/NEON)
- [ ] Add OpenVINO backend for Intel Xeon with AMX
- [ ] Build basic auto-quantization pipeline (HuggingFace -> optimized GGUF)
- [ ] Deploy on AWS reference hardware (all 3 configs) for benchmarking
- [ ] Publish initial benchmarks using methodology from [Technical Spec](technical-spec.md#6-benchmarking-methodology), including quality scores
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

## 3. Risks and Mitigations

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

## 4. Key Metrics to Track

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

## 5. Team Needed (First 18 Months)

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
