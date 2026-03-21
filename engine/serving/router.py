"""S2O multi-model router — A/B routing and model-based request dispatch.

Routes incoming requests to one of several upstream llama-server or proxy
instances based on the ``model`` field in the request body or weighted-random
selection for A/B traffic splitting.
"""

from __future__ import annotations

import json
import random
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse
    import httpx
    import uvicorn

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@dataclass
class Route:
    """A single model route."""

    name: str          # model name or alias (e.g. "llama-7b-q4")
    upstream: str      # URL of llama-server or proxy
    weight: int = 100  # traffic weight for A/B splits


class ModelRouter:
    """Routes requests to upstream model servers.

    Routing logic:
    1. Parse request body JSON for ``model`` field.
    2. If model matches a route name exactly -> route to that upstream.
    3. If model not found -> weighted-random among all routes.
    4. Proxy the full request (streaming + non-streaming) to chosen upstream.
    """

    def __init__(self, routes: list[Route]) -> None:
        if not HAS_DEPS:
            raise ImportError(
                "ModelRouter requires fastapi, httpx, and uvicorn. "
                "Install with: pip install fastapi httpx uvicorn"
            )
        if not routes:
            raise ValueError("At least one route is required")

        self.routes = routes
        self._route_map: dict[str, Route] = {r.name: r for r in routes}
        self._total_weight = sum(r.weight for r in routes)
        self.app = self._create_app()

    def _select_route(self, model_name: str | None) -> Route:
        """Select a route based on model name or weighted random."""
        if model_name and model_name in self._route_map:
            return self._route_map[model_name]
        # Weighted random selection
        r = random.randint(1, self._total_weight)
        cumulative = 0
        for route in self.routes:
            cumulative += route.weight
            if r <= cumulative:
                return route
        return self.routes[-1]  # fallback

    def _create_app(self) -> FastAPI:
        router = self

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            app.state.client = httpx.AsyncClient(timeout=120.0)
            yield
            await app.state.client.aclose()

        app = FastAPI(title="S2O Model Router", lifespan=lifespan)

        @app.get("/health")
        async def health():
            return {"status": "ok", "routes": len(router.routes)}

        @app.get("/v1/status")
        async def status():
            return {
                "router": True,
                "routes": [
                    {"name": r.name, "upstream": r.upstream, "weight": r.weight}
                    for r in router.routes
                ],
            }

        @app.get("/metrics")
        async def metrics():
            lines = [
                "# HELP s2o_router_routes Number of configured routes",
                "# TYPE s2o_router_routes gauge",
                f"s2o_router_routes {len(router.routes)}",
            ]
            for r in router.routes:
                lines.append(f's2o_route_weight{{name="{r.name}"}} {r.weight}')
            return Response(content="\n".join(lines) + "\n", media_type="text/plain")

        @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
        async def proxy_request(request: Request, path: str):
            # Try to extract model name from request body
            model_name = None
            body = await request.body()
            if body:
                try:
                    parsed = json.loads(body)
                    model_name = parsed.get("model")
                except (json.JSONDecodeError, AttributeError):
                    pass

            route = router._select_route(model_name)
            client: httpx.AsyncClient = app.state.client
            url = f"{route.upstream.rstrip('/')}/{path}"
            headers = dict(request.headers)
            headers.pop("host", None)

            accept = request.headers.get("accept", "")
            is_stream = "text/event-stream" in accept

            try:
                if is_stream:
                    req = client.build_request(
                        method=request.method, url=url, headers=headers, content=body,
                    )
                    upstream_resp = await client.send(req, stream=True)

                    async def stream_body():
                        try:
                            async for chunk in upstream_resp.aiter_bytes():
                                yield chunk
                        finally:
                            await upstream_resp.aclose()

                    return StreamingResponse(
                        stream_body(),
                        status_code=upstream_resp.status_code,
                        headers=dict(upstream_resp.headers),
                    )
                else:
                    resp = await client.request(
                        method=request.method, url=url, headers=headers, content=body,
                    )
                    return Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        headers=dict(resp.headers),
                    )
            except httpx.ConnectError:
                return JSONResponse({"error": "upstream_down", "route": route.name}, status_code=502)

        return app

    def run(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """Start the router server."""
        uvicorn.run(self.app, host=host, port=port, log_level="info")
