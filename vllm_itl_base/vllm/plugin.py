"""vLLM general-plugin entry point."""

from __future__ import annotations

import os


def activate() -> None:
    """Install the ITL_BASE patch when the launcher enables it."""

    if os.getenv("VLLM_ITL_BASE_ENABLE", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    from vllm_itl_base.vllm.compat import install_patch

    install_patch()
