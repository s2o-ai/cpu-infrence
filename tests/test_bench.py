"""Tests for the benchmarking framework."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.bench_types import BenchResult, SingleBenchResult, ServerBenchResult
from benchmarks.bench_runner import _compute_stats, _parse_json_output
from benchmarks.bench_report import generate_markdown, generate_json


# Sample llama-bench JSON output (captured from real run)
SAMPLE_JSON = json.dumps([
    {
        "build_commit": "c46583b",
        "build_number": 1,
        "cpu_info": "13th Gen Intel(R) Core(TM) i7-13650HX",
        "model_type": "qwen35 0.8B Q4_K - Medium",
        "model_size": 521555200,
        "model_n_params": 752393024,
        "n_threads": 10,
        "n_prompt": 512,
        "n_gen": 0,
        "avg_ts": 210.5,
        "stddev_ts": 2.1,
        "samples_ts": [208.0, 211.0, 212.5],
    },
    {
        "build_commit": "c46583b",
        "build_number": 1,
        "cpu_info": "13th Gen Intel(R) Core(TM) i7-13650HX",
        "model_type": "qwen35 0.8B Q4_K - Medium",
        "model_size": 521555200,
        "model_n_params": 752393024,
        "n_threads": 10,
        "n_prompt": 0,
        "n_gen": 128,
        "avg_ts": 8.4,
        "stddev_ts": 0.05,
        "samples_ts": [8.3, 8.45, 8.45],
    },
])


class TestComputeStats:
    def test_basic(self):
        avg, stddev, median = _compute_stats([10.0, 20.0, 30.0], discard_first=False)
        assert avg == 20.0
        assert median == 20.0

    def test_warmup_discard(self):
        avg, stddev, median = _compute_stats([5.0, 10.0, 20.0, 30.0], discard_first=True)
        # After discarding 5.0: [10.0, 20.0, 30.0]
        assert avg == 20.0
        assert median == 20.0

    def test_single_sample(self):
        avg, stddev, median = _compute_stats([42.0], discard_first=False)
        assert avg == 42.0
        assert stddev == 0.0
        assert median == 42.0

    def test_empty(self):
        avg, stddev, median = _compute_stats([])
        assert avg == 0.0


class TestParseJson:
    def test_parse_pp_and_tg(self):
        result = _parse_json_output(SAMPLE_JSON, "test_model.gguf", 10)
        assert isinstance(result, BenchResult)
        assert result.model_type == "qwen35 0.8B Q4_K - Medium"
        assert result.build_commit == "c46583b"

    def test_pp_parsed(self):
        result = _parse_json_output(SAMPLE_JSON, "test.gguf", 10)
        pp = result.prompt_processing
        assert pp.test_type == "pp"
        assert pp.tokens == 512
        assert len(pp.samples) == 1  # 1 entry in sample data (avg_ts, not samples_ts)

    def test_tg_parsed(self):
        result = _parse_json_output(SAMPLE_JSON, "test.gguf", 10)
        tg = result.text_generation
        assert tg.test_type == "tg"
        assert tg.tokens == 128

    def test_model_params(self):
        result = _parse_json_output(SAMPLE_JSON, "test.gguf", 10)
        assert result.model_params_b == pytest.approx(0.752, abs=0.01)
        assert result.model_size_gb == pytest.approx(0.486, abs=0.01)


class TestReport:
    def _make_result(self):
        return BenchResult(
            model_path="models/test.gguf",
            model_type="test 7B Q4_K",
            build_commit="abc123",
            cpu_info="Test CPU",
            threads=8,
            prompt_processing=SingleBenchResult(
                test_type="pp", tokens=512,
                avg_tok_per_sec=200.0, stddev_tok_per_sec=5.0,
                samples=[195.0, 200.0, 205.0], median_tok_per_sec=200.0,
            ),
            text_generation=SingleBenchResult(
                test_type="tg", tokens=128,
                avg_tok_per_sec=10.0, stddev_tok_per_sec=0.5,
                samples=[9.5, 10.0, 10.5], median_tok_per_sec=10.0,
            ),
        )

    def test_markdown_format(self):
        md = generate_markdown(self._make_result())
        assert "# Benchmark:" in md
        assert "Prompt processing" in md
        assert "Text generation" in md
        assert "200.00" in md
        assert "10.00" in md

    def test_json_format(self):
        j = generate_json(self._make_result())
        data = json.loads(j)
        assert "benchmark" in data
        assert data["benchmark"]["build_commit"] == "abc123"
        assert data["benchmark"]["prompt_processing"]["avg_tok_per_sec"] == 200.0

    def test_markdown_with_server(self):
        sr = ServerBenchResult(
            concurrency=4, total_requests=10,
            ttft_p50_ms=100, ttft_p95_ms=200, ttft_p99_ms=300,
            gen_tok_per_sec_total=30.0,
            latency_p50_ms=500, latency_p95_ms=800, latency_p99_ms=1000,
            errors=0,
        )
        md = generate_markdown(self._make_result(), server_results=[sr])
        assert "Server Throughput" in md
        assert "Concurrency" in md
