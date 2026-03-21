"""Tests for the auto-quantization pipeline."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.quantize._report import QuantResult, QuantVariant, generate_json, generate_markdown
from engine.quantize._convert import is_hf_model_id, model_name_from_id


class TestModelIdDetection:
    def test_hf_model_id(self):
        assert is_hf_model_id("Qwen/Qwen3-0.6B") is True

    def test_hf_model_id_nested(self):
        assert is_hf_model_id("meta-llama/Llama-3-8B") is True

    def test_local_path(self):
        assert is_hf_model_id("models/test.gguf") is False

    def test_local_dir(self, tmp_path):
        # If the path actually exists, it should be treated as local
        d = tmp_path / "org" / "model"
        d.mkdir(parents=True)
        assert is_hf_model_id(str(d)) is False


class TestModelNameExtraction:
    def test_simple(self):
        assert model_name_from_id("Qwen/Qwen3-0.6B") == "Qwen3-0.6B"

    def test_org_model(self):
        assert model_name_from_id("meta-llama/Llama-3-8B") == "Llama-3-8B"

    def test_nested(self):
        assert model_name_from_id("a/b/c") == "c"


class TestQuantVariant:
    def test_creation(self):
        v = QuantVariant(
            quant_type="Q4_K_M",
            output_path="models/test/test-Q4_K_M.gguf",
            size_mb=500.0,
            compression_ratio=3.2,
        )
        assert v.quant_type == "Q4_K_M"
        assert v.perplexity is None

    def test_with_perplexity(self):
        v = QuantVariant(
            quant_type="Q8_0",
            output_path="models/test/test-Q8_0.gguf",
            size_mb=800.0,
            compression_ratio=2.0,
            perplexity=6.1234,
        )
        assert v.perplexity == 6.1234


class TestQuantResult:
    def _make_result(self):
        return QuantResult(
            model_id="Qwen/Qwen3-0.6B",
            model_name="Qwen3-0.6B",
            original_size_mb=1200.0,
            cpu_info="13th Gen Intel Core i7-13650HX",
            variants=[
                QuantVariant(
                    quant_type="Q4_K_M",
                    output_path="models/Qwen3-0.6B/Qwen3-0.6B-Q4_K_M.gguf",
                    size_mb=400.0,
                    compression_ratio=3.0,
                ),
                QuantVariant(
                    quant_type="Q8_0",
                    output_path="models/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf",
                    size_mb=700.0,
                    compression_ratio=1.7,
                    perplexity=5.8,
                ),
            ],
        )

    def test_json_format(self):
        result = self._make_result()
        j = generate_json(result)
        data = json.loads(j)
        assert data["model_id"] == "Qwen/Qwen3-0.6B"
        assert data["model_name"] == "Qwen3-0.6B"
        assert data["original_size_mb"] == 1200.0
        assert len(data["variants"]) == 2
        assert data["variants"][0]["quant_type"] == "Q4_K_M"
        assert data["variants"][1]["perplexity"] == 5.8

    def test_json_roundtrip(self):
        result = self._make_result()
        j = generate_json(result)
        data = json.loads(j)
        # Should be valid JSON with all fields
        assert "timestamp" in data
        assert "cpu_info" in data

    def test_markdown_format(self):
        result = self._make_result()
        md = generate_markdown(result)
        assert "# Quantization Report: Qwen3-0.6B" in md
        assert "Q4_K_M" in md
        assert "Q8_0" in md
        assert "400.0" in md
        assert "3.0x" in md
        assert "5.80" in md  # perplexity formatted
        assert "—" in md  # missing perplexity for Q4_K_M

    def test_markdown_has_files_section(self):
        result = self._make_result()
        md = generate_markdown(result)
        assert "## Files" in md
        assert "Qwen3-0.6B-Q4_K_M.gguf" in md

    def test_empty_variants(self):
        result = QuantResult(
            model_id="test/model",
            model_name="model",
            original_size_mb=0.0,
        )
        j = generate_json(result)
        data = json.loads(j)
        assert data["variants"] == []


class TestReportSave:
    def test_save_creates_files(self, tmp_path):
        from engine.quantize._report import save_report

        result = QuantResult(
            model_id="test/model",
            model_name="test-model",
            original_size_mb=100.0,
            variants=[
                QuantVariant(
                    quant_type="Q4_K_M",
                    output_path="test-Q4_K_M.gguf",
                    size_mb=30.0,
                    compression_ratio=3.3,
                ),
            ],
        )

        json_path, md_path = save_report(result, tmp_path)
        assert json_path.exists()
        assert md_path.exists()
        assert json_path.name == "test-model_quant_report.json"
        assert md_path.name == "test-model_quant_report.md"

        # Verify JSON is valid
        data = json.loads(json_path.read_text())
        assert data["model_name"] == "test-model"


class TestQuantTypeParsing:
    """Test the CLI quant type parsing pattern."""

    def test_single_type(self):
        types = [t.strip() for t in "Q4_K_M".split(",")]
        assert types == ["Q4_K_M"]

    def test_multiple_types(self):
        types = [t.strip() for t in "Q4_K_M,Q8_0".split(",")]
        assert types == ["Q4_K_M", "Q8_0"]

    def test_with_spaces(self):
        types = [t.strip() for t in "Q4_K_M, Q8_0, IQ2_XS".split(",")]
        assert types == ["Q4_K_M", "Q8_0", "IQ2_XS"]
