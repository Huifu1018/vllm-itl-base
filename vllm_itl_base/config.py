"""Runtime configuration for the vLLM ITL_BASE proposer."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive when set.")
    return parsed


def _env_float(name: str, default: float | None) -> float | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    if value.strip().lower() in {"0", "false", "off", "none"}:
        return None
    parsed = float(value)
    if parsed == 0:
        return None
    if parsed < 0:
        raise ValueError(f"{name} must be positive when set.")
    return parsed


@dataclass(frozen=True)
class BaseVLLMConfig:
    """Configuration read from ``VLLM_ITL_BASE_*`` environment variables."""

    method: str = "auto"
    draft_model: str | None = None
    draft_device: str | None = None
    draft_device_map: str | None = None
    draft_dtype: str = "auto"
    max_draft_tokens: int | None = None
    max_context_tokens: int | None = None
    assistant_lookbehind: int = 10
    target_lookbehind: int = 10
    max_cached_requests: int = 256
    add_special_tokens: bool = False
    enable_draft_cache: bool = True
    clone_draft_cache: bool = True
    tli_min_intersection: int = 1
    strict_tli_probs: bool = True
    log_proposals: bool = False
    metrics_log_interval: float | None = 60.0

    @classmethod
    def from_env(cls) -> "BaseVLLMConfig":
        method = os.getenv("VLLM_ITL_BASE_METHOD", "auto").strip().lower()
        if method not in {"auto", "slem", "tli"}:
            raise ValueError("VLLM_ITL_BASE_METHOD must be one of: auto, slem, tli.")
        return cls(
            method=method,
            draft_model=os.getenv("VLLM_ITL_BASE_DRAFT_MODEL") or None,
            draft_device=os.getenv("VLLM_ITL_BASE_DRAFT_DEVICE") or None,
            draft_device_map=os.getenv("VLLM_ITL_BASE_DRAFT_DEVICE_MAP") or None,
            draft_dtype=os.getenv("VLLM_ITL_BASE_DRAFT_DTYPE", "auto"),
            max_draft_tokens=_env_int("VLLM_ITL_BASE_MAX_DRAFT_TOKENS", None),
            max_context_tokens=_env_int("VLLM_ITL_BASE_MAX_CONTEXT_TOKENS", None),
            assistant_lookbehind=(
                _env_int("VLLM_ITL_BASE_ASSISTANT_LOOKBEHIND", 10) or 10
            ),
            target_lookbehind=_env_int("VLLM_ITL_BASE_TARGET_LOOKBEHIND", 10) or 10,
            max_cached_requests=(
                _env_int("VLLM_ITL_BASE_MAX_CACHED_REQUESTS", 256) or 256
            ),
            add_special_tokens=_env_bool("VLLM_ITL_BASE_ADD_SPECIAL_TOKENS", False),
            enable_draft_cache=_env_bool("VLLM_ITL_BASE_ENABLE_DRAFT_CACHE", True),
            clone_draft_cache=_env_bool("VLLM_ITL_BASE_CLONE_DRAFT_CACHE", True),
            tli_min_intersection=(
                _env_int("VLLM_ITL_BASE_TLI_MIN_INTERSECTION", 1) or 1
            ),
            strict_tli_probs=_env_bool("VLLM_ITL_BASE_STRICT_TLI_PROBS", True),
            log_proposals=_env_bool("VLLM_ITL_BASE_LOG_PROPOSALS", False),
            metrics_log_interval=_env_float(
                "VLLM_ITL_BASE_METRICS_LOG_INTERVAL", 60.0
            ),
        )

    def method_for_request(self, *, is_greedy: bool) -> str:
        if self.method == "auto":
            return "slem" if is_greedy else "tli"
        return self.method
