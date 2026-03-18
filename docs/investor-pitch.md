# Zero-GPU AI Inference Platform — Investor Pitch

> **Audience:** Investors, CTOs, board members, business buyers
>
> For technical details, see [Technical Spec](technical-spec.md). For execution and operations, see [Operations Plan](operations-plan.md).

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

## 2. Competitive Landscape

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

## 3. Go-to-Market Strategy

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

## 4. Funding Requirements & Runway

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
| Salaries (6 people, see [Team](operations-plan.md#5-team-needed)) | $95K |
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

## Summary

The opportunity is clear: regulated enterprises need AI but can't use cloud GPUs. They have powerful CPU servers sitting underutilized. We build the bridge — a production-grade inference platform that turns their existing hardware into AI infrastructure, with the compliance they require and the performance that makes it viable.

The technical edge is LUT-based optimization that delivers **1.5-2.5x over the best open-source alternative** (llama.cpp with full hardware optimization enabled) on the same CPU hardware. This is a meaningful but honest improvement — not a 10x leap, but enough to differentiate when combined with the platform. For technical details, see [Technical Spec](technical-spec.md).

The business moat is multi-layered and compounds over time:
- **Compliance certifications** (SOC2/HIPAA) — 12+ months and $150K+ to replicate
- **Customer-specific calibration data** — domain-tuned quantization that improves quality on their specific tasks
- **Switching costs** — re-integrating compliance, RBAC, monitoring, and deployment pipelines with a new vendor
- **Auto-tuning pipeline** — hardware-specific optimization database built from profiling across dozens of configurations

This plan targets **7B-13B parameter models**, launches in **healthcare** as the beachhead vertical, and charts a realistic **18-month path to $500K ARR**. It is not a plan to replace GPU inference broadly — it is a plan to own the regulated on-premises CPU inference niche, and expand from there.

Open-source the engine. Sell the platform. Win the enterprise.
