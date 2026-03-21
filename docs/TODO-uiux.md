# UI/UX TODO

> Frontend, dashboards, and user-facing experience.
> For backend engineering, see [TODO-tech.md](TODO-tech.md). For business/compliance, see [TODO-biz.md](TODO-biz.md).

---

## Phase 1: CLI Experience — DONE

### Task 1.6: CLI Tool
- [x] `s2o info` — CPU detection with Rich panels (--json, --verbose)
- [x] `s2o build` — build inference engine (--clean, --lut)
- [x] `s2o serve <model>` — start OpenAI-compatible server
- [x] `s2o models` — list downloaded GGUF models
- [x] `s2o run <model>` — interactive chat via llama-cli
- [x] `s2o bench <model>` — benchmarks with reports (--server for throughput)
- [x] Typer + Rich, --help for all commands
- [x] 5 tests pass (`tests/test_cli.py`)
- [ ] Test on Linux and macOS — **BLOCKED: needs Linux/macOS machines**

---

## Phase 2: Dashboards (Months 3-6)

### Task 2.4: Model Management UI — NOT STARTED
React + TypeScript SPA with FastAPI management backend.

**Pages:**
- [ ] **Dashboard** — system overview: running models, resource usage, request throughput
- [ ] **Models** — browse downloaded GGUFs, upload new, view quant info, delete
- [ ] **Deploy** — select model → configure (context length, threads, KV quant) → start server
- [ ] **Monitor** — live metrics: tokens/sec, TTFT, P50/P95, active connections, memory
- [ ] **Settings** — build config, default parameters, API keys

**Backend API (`engine/serving/api.py`):**
- [ ] `GET /api/system` — CPU info, memory, running processes
- [ ] `GET /api/models` — list available GGUFs with metadata
- [ ] `POST /api/quantize` — trigger quantization job (async)
- [ ] `POST /api/server/start` — launch llama-server with params
- [ ] `POST /api/server/stop` — graceful shutdown
- [ ] `GET /api/server/status` — running state, slot usage, throughput

**Infra:**
- [ ] Vite + TypeScript + React
- [ ] `s2o ui` CLI command to launch (serves built frontend + FastAPI)
- [ ] Tests: `tests/test_api.py`

### Task 2.8: Quality Dashboard — NOT STARTED
Static HTML quality report for model × quantization × hardware combinations.

- [ ] Evaluation engine (`engine/quality/evaluate.py`):
  - [ ] Perplexity (reuse `engine/quantize/_validate.py`)
  - [ ] MMLU via lm-eval-harness subprocess
  - [ ] HumanEval (optional, code models only)
- [ ] Jinja2 static HTML templates (`engine/quality/templates/`)
- [ ] Comparison matrix: rows = models, columns = quant types, cells = (perplexity, MMLU, tok/s)
- [ ] `s2o quality <model>` CLI command
- [ ] Auto-update via CI (nightly benchmark runs → regenerate dashboard)
- [ ] Tests: `tests/test_quality.py`

---

## Phase 3: Enterprise UX (Months 6-12)

### Task 3.6: A/B Testing & Multi-Model Serving
- [ ] Multi-model routing UI (single endpoint, model selector dropdown)
- [ ] Traffic splitting configuration (percentage-based or user-based)
- [ ] Side-by-side comparison view (same prompt, two models)
- [ ] Statistical comparison reporting (TTFT, throughput, quality diffs)

### Task 3.7: Customer Onboarding UX
> Backend provisioning is in [TODO-biz.md](TODO-biz.md). This covers the user-facing experience.

- [ ] Model selection wizard (guided: use case → model size → quant recommendation)
- [ ] One-click deployment (pre-filled config from wizard)
- [ ] Getting started guide (in-app, contextual)
- [ ] Welcome walkthrough for new users

---

## Design Principles

1. **CLI-first** — Every UI action must also work via `s2o` CLI. The UI is a convenience layer, not a gate.
2. **No login required for local** — Single-user local mode needs zero auth. Enterprise SSO is Phase 3.
3. **Real-time metrics** — Dashboards show live data (WebSocket or polling), not stale snapshots.
4. **Minimal JS dependencies** — Quality dashboard is static HTML (Jinja2). Management UI uses React only because it needs interactivity.
