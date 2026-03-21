"""Tests for speculative decoding draft model selector."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.serving.speculative import (
    suggest_draft_model,
    acceptance_rate,
    _parse_size,
    _parse_family,
    DRAFT_SIZE_MAP,
)


# --------------------------------------------------------------------------
# Size/family parsing
# --------------------------------------------------------------------------

class TestParsing:

    def test_parse_size_standard(self):
        assert _parse_size("Qwen2.5-7B-Instruct-Q4_K_M") == "7b"

    def test_parse_size_decimal(self):
        assert _parse_size("Qwen3-0.6B-Q4_K_M") == "0.6b"

    def test_parse_size_large(self):
        assert _parse_size("Llama-3.1-70B-Q4_0") == "70b"

    def test_parse_size_no_match(self):
        assert _parse_size("random-file") is None

    def test_parse_family_qwen(self):
        assert _parse_family("Qwen2.5-7B-Instruct-Q4_K_M") == "qwen2.5"

    def test_parse_family_llama(self):
        assert _parse_family("Llama-3.1-70B-Q4_0") == "llama-3.1"

    def test_parse_family_no_match(self):
        assert _parse_family("12345") is None


# --------------------------------------------------------------------------
# Draft model suggestion
# --------------------------------------------------------------------------

class TestSuggestDraftModel:

    def test_returns_none_for_no_models_dir(self, tmp_path):
        fake_dir = tmp_path / "nonexistent"
        result = suggest_draft_model("Qwen2.5-7B-Q4_K_M.gguf", fake_dir)
        assert result is None

    def test_returns_none_when_no_matching_family(self, tmp_path):
        # Create a model from a different family
        (tmp_path / "Llama-3-1B-Q4_0.gguf").touch()
        result = suggest_draft_model("Qwen2.5-7B-Q4_K_M.gguf", tmp_path)
        assert result is None

    def test_returns_none_for_smallest_model(self, tmp_path):
        # 0.5B has no draft candidates
        (tmp_path / "Qwen2.5-0.5B-Q4_K_M.gguf").touch()
        result = suggest_draft_model("Qwen2.5-0.5B-Q4_K_M.gguf", tmp_path)
        assert result is None

    def test_finds_matching_draft(self, tmp_path):
        # Create a 7B main and 1B draft from same family
        (tmp_path / "Qwen2.5-7B-Q4_K_M.gguf").touch()
        (tmp_path / "Qwen2.5-1B-Q4_K_M.gguf").touch()
        result = suggest_draft_model("Qwen2.5-7B-Q4_K_M.gguf", tmp_path)
        assert result is not None
        assert "Qwen2.5-1B" in result

    def test_prefers_smaller_draft(self, tmp_path):
        # Both 0.5B and 1B available — should prefer 0.5B for 7B main
        (tmp_path / "Qwen2.5-0.5B-Q4_K_M.gguf").touch()
        (tmp_path / "Qwen2.5-1B-Q4_K_M.gguf").touch()
        result = suggest_draft_model("Qwen2.5-7B-Q4_K_M.gguf", tmp_path)
        assert result is not None
        assert "0.5B" in result

    def test_returns_absolute_path(self, tmp_path):
        (tmp_path / "Qwen2.5-1B-Q4_K_M.gguf").touch()
        result = suggest_draft_model("Qwen2.5-7B-Q4_K_M.gguf", tmp_path)
        assert result is not None
        assert Path(result).is_absolute()


# --------------------------------------------------------------------------
# Acceptance rate
# --------------------------------------------------------------------------

class TestAcceptanceRate:

    def test_normal(self):
        assert acceptance_rate(100, 80) == pytest.approx(0.8)

    def test_zero_denominator(self):
        assert acceptance_rate(0, 0) == 0.0

    def test_perfect(self):
        assert acceptance_rate(50, 50) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Draft size map completeness
# --------------------------------------------------------------------------

class TestDraftSizeMap:

    def test_common_sizes_have_entries(self):
        for size in ["7b", "13b", "70b"]:
            assert size in DRAFT_SIZE_MAP
            assert len(DRAFT_SIZE_MAP[size]) > 0

    def test_smallest_has_no_drafts(self):
        assert DRAFT_SIZE_MAP["0.5b"] == []
