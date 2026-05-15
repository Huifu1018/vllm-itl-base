"""Preflight checks for vLLM ITL_BASE deployments."""

from __future__ import annotations

import argparse
import importlib.metadata
import json

from vllm_itl_base import SUPPORTED_VLLM_VERSION


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check vllm-itl-base readiness.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args(argv)

    result = {
        "vllm_installed": False,
        "vllm_version": None,
        "supported_vllm_version": SUPPORTED_VLLM_VERSION,
        "vllm_version_ok": False,
        "plugin_entrypoint_installed": False,
        "serve_command": "vllm-itl-base-serve",
    }
    try:
        version = importlib.metadata.version("vllm")
        result["vllm_installed"] = True
        result["vllm_version"] = version
        result["vllm_version_ok"] = version == SUPPORTED_VLLM_VERSION
    except importlib.metadata.PackageNotFoundError:
        pass

    eps = importlib.metadata.entry_points()
    group_eps = eps.select(group="vllm.general_plugins")
    result["plugin_entrypoint_installed"] = any(
        ep.name == "vllm_itl_base" for ep in group_eps
    )

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
