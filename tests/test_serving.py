"""Tests for S2O serving module — config, proxy, admission control."""

from __future__ import annotations

import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.serving.config import ServingConfig


# --------------------------------------------------------------------------
# ServingConfig tests
# --------------------------------------------------------------------------

class TestServingConfig:

    def test_defaults(self):
        cfg = ServingConfig()
        assert cfg.max_concurrent == 16
        assert cfg.kv_cache_type == "f16"
        assert cfg.speculative is False
        assert cfg.draft_model is None
        assert cfg.retry_after_seconds == 5
        assert cfg.degradation_threshold == 0.90

    def test_effective_kv_type_v_defaults_to_k(self):
        cfg = ServingConfig(kv_cache_type="q8_0")
        assert cfg.effective_kv_type_v == "q8_0"

    def test_effective_kv_type_v_explicit(self):
        cfg = ServingConfig(kv_cache_type="q8_0", kv_cache_type_v="q4_0")
        assert cfg.effective_kv_type_v == "q4_0"

    def test_llama_server_args_default(self):
        cfg = ServingConfig()
        assert cfg.llama_server_args() == []

    def test_llama_server_args_kv_quant(self):
        cfg = ServingConfig(kv_cache_type="q8_0")
        args = cfg.llama_server_args()
        assert "--cache-type-k" in args
        assert "q8_0" in args
        assert "--cache-type-v" in args

    def test_llama_server_args_speculative(self):
        cfg = ServingConfig(speculative=True, draft_model="/models/draft.gguf")
        args = cfg.llama_server_args()
        assert "--model-draft" in args
        assert "/models/draft.gguf" in args

    def test_llama_server_args_speculative_no_draft(self):
        cfg = ServingConfig(speculative=True)
        args = cfg.llama_server_args()
        assert "--model-draft" not in args


# --------------------------------------------------------------------------
# Proxy tests (without running server)
# --------------------------------------------------------------------------

class TestProxyMetrics:

    def test_proxy_module_importable(self):
        """Proxy module should import even without fastapi/httpx."""
        from engine.serving import proxy
        assert hasattr(proxy, "ServingProxy")
        assert hasattr(proxy, "ProxyMetrics")

    def test_metrics_increment(self):
        import asyncio
        from engine.serving.proxy import ProxyMetrics
        m = ProxyMetrics()
        assert m.active_requests == 0
        assert m.requests_total == 0
        asyncio.run(m.inc_active())
        assert m.active_requests == 1
        assert m.requests_total == 1
        asyncio.run(m.dec_active())
        assert m.active_requests == 0

    def test_metrics_rejected(self):
        import asyncio
        from engine.serving.proxy import ProxyMetrics
        m = ProxyMetrics()
        asyncio.run(m.inc_rejected())
        assert m.rejected_503 == 1
        assert m.requests_total == 1


# --------------------------------------------------------------------------
# Proxy integration tests (requires fastapi/httpx)
# --------------------------------------------------------------------------

HAS_FASTAPI = False
try:
    import fastapi
    import httpx
    HAS_FASTAPI = True
except ImportError:
    pass


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi/httpx not installed")
class TestProxyApp:

    def _make_proxy(self):
        from engine.serving.proxy import ServingProxy
        config = ServingConfig(max_concurrent=2, retry_after_seconds=3)
        return ServingProxy("http://localhost:9999", config)

    def test_proxy_creates_app(self):
        proxy = self._make_proxy()
        assert proxy.app is not None

    def test_proxy_status_route(self):
        from fastapi.testclient import TestClient
        proxy = self._make_proxy()
        client = TestClient(proxy.app)
        resp = client.get("/v1/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "active" in data
        assert "max" in data
        assert data["max"] == 2

    def test_proxy_metrics_route(self):
        from fastapi.testclient import TestClient
        proxy = self._make_proxy()
        client = TestClient(proxy.app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.text
        assert "s2o_active_requests" in body
        assert "s2o_requests_total" in body
        assert "s2o_503_total" in body
        assert "s2o_max_concurrent" in body
