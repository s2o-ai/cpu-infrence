"""Serving configuration for S2O admission control proxy."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ServingConfig:
    """Configuration for the S2O serving proxy."""

    max_concurrent: int = 16
    priority_header: str = "X-Priority"
    kv_cache_type: str = "f16"
    kv_cache_type_v: str | None = None
    speculative: bool = False
    draft_model: str | None = None
    degradation_threshold: float = 0.90
    retry_after_seconds: int = 5

    @property
    def effective_kv_type_v(self) -> str:
        return self.kv_cache_type_v or self.kv_cache_type

    def llama_server_args(self) -> list[str]:
        """Generate llama-server CLI args from this config."""
        args: list[str] = []

        if self.kv_cache_type != "f16":
            args.extend(["--cache-type-k", self.kv_cache_type])
            args.extend(["--cache-type-v", self.effective_kv_type_v])

        if self.speculative and self.draft_model:
            args.extend(["--model-draft", self.draft_model])

        return args
