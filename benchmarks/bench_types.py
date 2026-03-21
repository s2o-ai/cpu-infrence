"""Data types for benchmark results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SingleBenchResult:
    """Result for one benchmark test (prompt processing or text generation)."""

    test_type: str  # "pp" or "tg"
    tokens: int
    avg_tok_per_sec: float
    stddev_tok_per_sec: float
    samples: list[float] = field(default_factory=list)
    median_tok_per_sec: float = 0.0


@dataclass
class BenchResult:
    """Complete benchmark result for a model."""

    model_path: str = ""
    model_type: str = ""
    model_size_gb: float = 0.0
    model_params_b: float = 0.0
    cpu_info: str = ""
    build_commit: str = ""
    threads: int = 0
    prompt_processing: SingleBenchResult = field(
        default_factory=lambda: SingleBenchResult("pp", 0, 0, 0)
    )
    text_generation: SingleBenchResult = field(
        default_factory=lambda: SingleBenchResult("tg", 0, 0, 0)
    )
    timestamp: str = ""


@dataclass
class ServerBenchResult:
    """Server throughput benchmark result."""

    concurrency: int = 0
    total_requests: int = 0
    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    gen_tok_per_sec_total: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    errors: int = 0
    timestamp: str = ""
