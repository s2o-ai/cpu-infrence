"""Tests for S2O multi-model router."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.serving.router import Route, ModelRouter
from engine.serving.config import RouterConfig


# --------------------------------------------------------------------------
# Route dataclass
# --------------------------------------------------------------------------

class TestRoute:

    def test_route_defaults(self):
        r = Route(name="test", upstream="http://localhost:8080")
        assert r.weight == 100

    def test_route_custom_weight(self):
        r = Route(name="test", upstream="http://localhost:8080", weight=50)
        assert r.weight == 50

    def test_route_fields(self):
        r = Route(name="llama-7b", upstream="http://localhost:8081", weight=75)
        assert r.name == "llama-7b"
        assert r.upstream == "http://localhost:8081"
        assert r.weight == 75


# --------------------------------------------------------------------------
# RouterConfig
# --------------------------------------------------------------------------

class TestRouterConfig:

    def test_router_config_defaults(self):
        cfg = RouterConfig()
        assert cfg.routes == []
        assert cfg.port == 8080

    def test_router_config_custom(self):
        routes = [{"name": "a", "upstream": "http://a:8080", "weight": 100}]
        cfg = RouterConfig(routes=routes, port=9090)
        assert len(cfg.routes) == 1
        assert cfg.port == 9090


# --------------------------------------------------------------------------
# ModelRouter
# --------------------------------------------------------------------------

class TestModelRouter:

    def test_requires_routes(self):
        with pytest.raises(ValueError, match="At least one route"):
            ModelRouter(routes=[])

    def test_exact_match_routing(self):
        routes = [
            Route(name="model-a", upstream="http://a:8080"),
            Route(name="model-b", upstream="http://b:8080"),
        ]
        router = ModelRouter(routes)
        selected = router._select_route("model-a")
        assert selected.name == "model-a"
        assert selected.upstream == "http://a:8080"

    def test_exact_match_routing_second(self):
        routes = [
            Route(name="model-a", upstream="http://a:8080"),
            Route(name="model-b", upstream="http://b:8080"),
        ]
        router = ModelRouter(routes)
        selected = router._select_route("model-b")
        assert selected.name == "model-b"

    def test_unknown_model_falls_back_to_weighted(self):
        routes = [
            Route(name="model-a", upstream="http://a:8080", weight=100),
        ]
        router = ModelRouter(routes)
        # Unknown model should still route (weighted random with 1 route)
        selected = router._select_route("nonexistent")
        assert selected.name == "model-a"

    def test_none_model_uses_weighted(self):
        routes = [
            Route(name="model-a", upstream="http://a:8080", weight=100),
        ]
        router = ModelRouter(routes)
        selected = router._select_route(None)
        assert selected.name == "model-a"

    def test_weighted_distribution(self):
        """Verify weighted random roughly follows weights over many trials."""
        routes = [
            Route(name="heavy", upstream="http://a:8080", weight=90),
            Route(name="light", upstream="http://b:8080", weight=10),
        ]
        router = ModelRouter(routes)
        counts = {"heavy": 0, "light": 0}
        for _ in range(1000):
            r = router._select_route(None)
            counts[r.name] += 1
        # heavy should get ~90% of traffic
        assert counts["heavy"] > 700  # conservative bound


# --------------------------------------------------------------------------
# FastAPI app routes
# --------------------------------------------------------------------------

class TestRouterApp:

    @pytest.fixture
    def router(self):
        routes = [
            Route(name="model-a", upstream="http://localhost:9999"),
        ]
        return ModelRouter(routes)

    def test_app_has_health(self, router):
        from fastapi.testclient import TestClient
        client = TestClient(router.app)
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_app_has_status(self, router):
        from fastapi.testclient import TestClient
        client = TestClient(router.app)
        r = client.get("/v1/status")
        assert r.status_code == 200
        data = r.json()
        assert data["router"] is True
        assert len(data["routes"]) == 1

    def test_app_has_metrics(self, router):
        from fastapi.testclient import TestClient
        client = TestClient(router.app)
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "s2o_router_routes" in r.text
