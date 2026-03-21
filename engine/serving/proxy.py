"""S2O admission control proxy — FastAPI reverse proxy with backpressure.

Wraps llama-server with:
- Semaphore-based admission control (503 + Retry-After when at capacity)
- Priority header support (X-Priority: 0=low, 1=normal, 2=high)
- /v1/status endpoint for monitoring
- /metrics endpoint (Prometheus-compatible)
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from .config import ServingConfig

try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse
    import httpx
    import uvicorn

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


class ProxyMetrics:
    """Thread-safe request metrics."""

    def __init__(self) -> None:
        self.requests_total: int = 0
        self.active_requests: int = 0
        self.rejected_503: int = 0
        self._lock = asyncio.Lock()

    async def inc_active(self) -> None:
        async with self._lock:
            self.active_requests += 1
            self.requests_total += 1

    async def dec_active(self) -> None:
        async with self._lock:
            self.active_requests -= 1

    async def inc_rejected(self) -> None:
        async with self._lock:
            self.rejected_503 += 1
            self.requests_total += 1


class ServingProxy:
    """Async reverse proxy with admission control for llama-server."""

    def __init__(self, upstream_url: str, config: ServingConfig | None = None) -> None:
        if not HAS_DEPS:
            raise ImportError(
                "ServingProxy requires fastapi, httpx, and uvicorn. "
                "Install with: pip install fastapi httpx uvicorn"
            )

        self.upstream_url = upstream_url.rstrip("/")
        self.config = config or ServingConfig()
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self.metrics = ProxyMetrics()
        self.app = self._create_app()

    def _priority_timeout(self, priority: int) -> float:
        """Timeout for semaphore acquire based on priority level."""
        if priority >= 2:
            return 30.0  # high priority: wait longer
        if priority >= 1:
            return 10.0  # normal: moderate wait
        return 1.0  # low: quick fail

    def _create_app(self) -> FastAPI:
        proxy = self

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            app.state.client = httpx.AsyncClient(timeout=120.0)
            yield
            await app.state.client.aclose()

        app = FastAPI(title="S2O Inference Proxy", lifespan=lifespan)

        @app.get("/v1/status")
        async def status():
            return {
                "active": proxy.metrics.active_requests,
                "max": proxy.config.max_concurrent,
                "pct_used": proxy.metrics.active_requests / max(proxy.config.max_concurrent, 1),
                "total_requests": proxy.metrics.requests_total,
                "rejected_503": proxy.metrics.rejected_503,
            }

        @app.get("/metrics")
        async def metrics():
            lines = [
                f"# HELP s2o_active_requests Current active requests",
                f"# TYPE s2o_active_requests gauge",
                f"s2o_active_requests {proxy.metrics.active_requests}",
                f"# HELP s2o_requests_total Total requests received",
                f"# TYPE s2o_requests_total counter",
                f"s2o_requests_total {proxy.metrics.requests_total}",
                f"# HELP s2o_503_total Total 503 rejections",
                f"# TYPE s2o_503_total counter",
                f"s2o_503_total {proxy.metrics.rejected_503}",
                f"# HELP s2o_max_concurrent Maximum concurrent requests",
                f"# TYPE s2o_max_concurrent gauge",
                f"s2o_max_concurrent {proxy.config.max_concurrent}",
            ]
            return Response(content="\n".join(lines) + "\n", media_type="text/plain")

        @app.get("/health")
        async def health():
            client: httpx.AsyncClient = app.state.client
            try:
                r = await client.get(f"{proxy.upstream_url}/health")
                return Response(content=r.content, status_code=r.status_code)
            except httpx.ConnectError:
                return JSONResponse({"status": "upstream_down"}, status_code=502)

        @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
        async def proxy_request(request: Request, path: str):
            priority = int(request.headers.get(proxy.config.priority_header, "1"))
            timeout = proxy._priority_timeout(priority)

            try:
                await asyncio.wait_for(proxy.semaphore.acquire(), timeout=timeout)
            except asyncio.TimeoutError:
                await proxy.metrics.inc_rejected()
                return JSONResponse(
                    {"error": "Server at capacity", "type": "capacity_exceeded"},
                    status_code=503,
                    headers={"Retry-After": str(proxy.config.retry_after_seconds)},
                )

            await proxy.metrics.inc_active()
            try:
                client: httpx.AsyncClient = app.state.client
                url = f"{proxy.upstream_url}/{path}"
                body = await request.body()
                headers = dict(request.headers)
                headers.pop("host", None)

                # Check if client wants streaming
                accept = request.headers.get("accept", "")
                is_stream = "text/event-stream" in accept

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
                return JSONResponse({"error": "upstream_down"}, status_code=502)
            finally:
                proxy.semaphore.release()
                await proxy.metrics.dec_active()

        return app

    def run(self, host: str = "127.0.0.1", port: int = 8081) -> None:
        uvicorn.run(self.app, host=host, port=port, log_level="info")
