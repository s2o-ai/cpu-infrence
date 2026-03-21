#!/usr/bin/env python3
"""Smoke tests for the inference server endpoints."""

import json
import sys
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8080"
PASSED = 0
FAILED = 0


def test(name, func):
    """Run a test and report result."""
    global PASSED, FAILED
    try:
        func()
        print(f"  PASS  {name}")
        PASSED += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        FAILED += 1


def api_get(path):
    """GET request to server."""
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def api_post(path, data):
    """POST request to server."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def api_post_stream(path, data):
    """POST request with streaming response."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    chunks = []
    with urllib.request.urlopen(req, timeout=60) as resp:
        for line in resp:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))
    return chunks


def test_health():
    result = api_get("/health")
    assert result.get("status") == "ok", f"Expected status ok, got {result}"


def test_v1_health():
    result = api_get("/v1/health")
    assert result.get("status") == "ok", f"Expected status ok, got {result}"


def test_models():
    result = api_get("/v1/models")
    assert "data" in result, "Missing 'data' field"
    assert len(result["data"]) > 0, "No models listed"
    model = result["data"][0]
    assert "id" in model, "Model missing 'id'"


def test_chat_completions():
    result = api_post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 20,
    })
    assert "choices" in result, "Missing 'choices'"
    assert len(result["choices"]) > 0, "Empty choices"
    msg = result["choices"][0]["message"]
    content = msg.get("content") or msg.get("reasoning_content") or ""
    assert len(content) > 0, "Empty response (no content or reasoning_content)"
    assert "usage" in result, "Missing 'usage'"
    assert "prompt_tokens" in result["usage"], "Missing prompt_tokens"
    assert "completion_tokens" in result["usage"], "Missing completion_tokens"


def test_chat_streaming():
    chunks = api_post_stream("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Count to 3."}],
        "max_tokens": 30,
        "stream": True,
    })
    assert len(chunks) > 0, "No streaming chunks received"
    has_content = any(
        c.get("choices", [{}])[0].get("delta", {}).get("content")
        or c.get("choices", [{}])[0].get("delta", {}).get("reasoning_content")
        for c in chunks
    )
    assert has_content, "No content in streaming chunks"


def test_completions():
    result = api_post("/v1/completions", {
        "prompt": "The capital of France is",
        "max_tokens": 20,
    })
    assert "choices" in result, "Missing 'choices'"
    assert len(result["choices"]) > 0, "Empty choices"
    assert "text" in result["choices"][0], "Missing 'text' in choice"


def test_metrics():
    req = urllib.request.Request(f"{BASE_URL}/metrics")
    with urllib.request.urlopen(req, timeout=10) as resp:
        text = resp.read().decode()
    assert "llama_" in text or "# HELP" in text, "Metrics endpoint not returning Prometheus format"


def main():
    print(f"\nSmoke testing server at {BASE_URL}\n")

    # Check server is reachable
    try:
        api_get("/health")
    except Exception as e:
        print(f"ERROR: Server not reachable at {BASE_URL}: {e}")
        print("Start the server first: python scripts/serve.py <model>")
        sys.exit(1)

    test("GET /health", test_health)
    test("GET /v1/health", test_v1_health)
    test("GET /v1/models", test_models)
    test("POST /v1/chat/completions", test_chat_completions)
    test("POST /v1/chat/completions (stream)", test_chat_streaming)
    test("POST /v1/completions", test_completions)
    test("GET /metrics", test_metrics)

    print(f"\nResults: {PASSED} passed, {FAILED} failed\n")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
