"""Draft model auto-selection for speculative decoding."""

from __future__ import annotations

import re
from pathlib import Path


# Main model size → candidate draft sizes (in descending preference)
DRAFT_SIZE_MAP: dict[str, list[str]] = {
    "0.5b": [],
    "0.6b": [],
    "1b": [],
    "1.5b": ["0.5b", "0.6b"],
    "3b": ["0.5b", "0.6b", "1b", "1.5b"],
    "7b": ["0.5b", "0.6b", "1b", "1.5b"],
    "8b": ["0.5b", "0.6b", "1b", "1.5b", "3b"],
    "13b": ["1b", "1.5b", "3b"],
    "14b": ["1b", "1.5b", "3b"],
    "30b": ["3b", "7b"],
    "32b": ["3b", "7b", "8b"],
    "70b": ["7b", "8b", "13b", "14b"],
}

# Regex to extract model size from filename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf" → "7b"
_SIZE_RE = re.compile(r"[_-](\d+(?:\.\d+)?)[Bb][-_.]")

# Regex to extract model family: "Qwen2.5-7B-..." → "qwen2.5", "Llama-3.1-70B-..." → "llama-3.1"
_FAMILY_RE = re.compile(r"^([A-Za-z][A-Za-z0-9._-]*?)[-_]\d+(?:\.\d+)?[Bb]")


def _parse_size(filename: str) -> str | None:
    """Extract model size string from GGUF filename."""
    m = _SIZE_RE.search(filename)
    if m:
        return m.group(1).lower() + "b"
    return None


def _parse_family(filename: str) -> str | None:
    """Extract model family from GGUF filename."""
    m = _FAMILY_RE.search(filename)
    if m:
        return m.group(1).lower()
    return None


def suggest_draft_model(main_model_path: str, models_dir: Path) -> str | None:
    """Find a smaller model of the same family in models_dir.

    Returns absolute path to the best draft model GGUF, or None.
    """
    main_name = Path(main_model_path).stem
    main_size = _parse_size(main_name)
    main_family = _parse_family(main_name)

    if not main_size or not main_family:
        return None

    candidate_sizes = DRAFT_SIZE_MAP.get(main_size, [])
    if not candidate_sizes:
        return None

    if not models_dir.is_dir():
        return None

    # Find all GGUFs from the same family
    candidates: list[tuple[int, Path]] = []
    for gguf in models_dir.rglob("*.gguf"):
        name = gguf.stem
        family = _parse_family(name)
        size = _parse_size(name)
        if family == main_family and size in candidate_sizes:
            # Prefer earlier in candidate_sizes list (smaller = better draft)
            try:
                priority = candidate_sizes.index(size)
            except ValueError:
                continue
            candidates.append((priority, gguf))

    if not candidates:
        return None

    # Return the best candidate (lowest priority index = preferred size)
    candidates.sort(key=lambda x: x[0])
    return str(candidates[0][1].resolve())


def acceptance_rate(n_draft_total: int, n_draft_accepted: int) -> float:
    """Calculate speculative decoding acceptance rate."""
    return n_draft_accepted / max(n_draft_total, 1)


class KAutoTuner:
    """Adaptive draft-K controller for speculative decoding.

    Tracks acceptance rate over a sliding window and adjusts the number
    of draft tokens (K) up or down to stay near the target acceptance rate.
    """

    def __init__(
        self,
        k_min: int = 1,
        k_max: int = 8,
        target_accept: float = 0.7,
        window: int = 50,
    ) -> None:
        self.k: int = 4  # starting draft tokens
        self.k_min = k_min
        self.k_max = k_max
        self.target_accept = target_accept
        self._window = window
        self._history: list[tuple[int, int]] = []  # (draft_total_delta, draft_accepted_delta)

    def record(self, n_draft_total_delta: int, n_draft_accepted_delta: int) -> None:
        """Record a batch of draft token statistics."""
        if n_draft_total_delta <= 0:
            return
        self._history.append((n_draft_total_delta, n_draft_accepted_delta))
        if len(self._history) > self._window:
            self._history = self._history[-self._window:]

    @property
    def current_rate(self) -> float:
        """Acceptance rate over the sliding window."""
        total = sum(d for d, _ in self._history)
        accepted = sum(a for _, a in self._history)
        return acceptance_rate(total, accepted)

    def suggest_k(self) -> int:
        """Suggest a new K value based on recent acceptance rate.

        If rate > target + 0.1 → increment K (more drafts, more speedup).
        If rate < target - 0.1 → decrement K (fewer wasted drafts).
        Otherwise keep current K.
        """
        if not self._history:
            return self.k

        rate = self.current_rate
        margin = 0.1

        if rate > self.target_accept + margin:
            self.k = min(self.k + 1, self.k_max)
        elif rate < self.target_accept - margin:
            self.k = max(self.k - 1, self.k_min)

        return self.k
