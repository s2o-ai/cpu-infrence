"""Tests for the S2O CLI tool."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(ROOT / ".venv" / "Scripts" / "python")
S2O = str(ROOT / "scripts" / "s2o.py")


def run_cli(*args):
    """Run s2o CLI command and return result."""
    return subprocess.run(
        [PYTHON, S2O, *args],
        capture_output=True, text=True, timeout=30,
    )


class TestCliHelp:
    def test_help(self):
        r = run_cli("--help")
        assert r.returncode == 0
        assert "S2O Zero-GPU AI Inference Platform" in r.stdout

    def test_commands_listed(self):
        r = run_cli("--help")
        for cmd in ("info", "build", "serve", "models", "run", "bench"):
            assert cmd in r.stdout


class TestInfoCommand:
    def test_info(self):
        r = run_cli("info")
        assert r.returncode == 0
        assert "S2O CPU Detection" in r.stdout

    def test_info_json(self):
        r = run_cli("info", "--json")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "vendor" in data
        assert "features" in data
        assert "recommendation" in data


class TestBuildCommand:
    def test_build_openvino_flag_exists(self):
        build_cmd = ROOT / "scripts" / "commands" / "build_cmd.py"
        content = build_cmd.read_text()
        assert "--openvino" in content
        assert "OpenVINO" in content

    def test_build_py_has_openvino_param(self):
        build_py = ROOT / "scripts" / "build.py"
        content = build_py.read_text()
        assert "openvino" in content
        assert "GGML_OPENVINO" in content


class TestModelsCommand:
    def test_models(self):
        r = run_cli("models")
        assert r.returncode == 0
        assert "Qwen3.5-0.8B" in r.stdout or "Downloaded Models" in r.stdout
