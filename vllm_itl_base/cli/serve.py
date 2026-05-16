"""Launch vLLM 0.15.1 with ITL_BASE enabled."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from vllm_itl_base.vllm.compat import install_patch


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _make_parser()
    args, vllm_args = parser.parse_known_args(argv)

    draft_model = args.itl_base_draft_model or os.getenv("VLLM_ITL_BASE_DRAFT_MODEL")
    if not draft_model:
        parser.error(
            "--itl-base-draft-model is required unless "
            "VLLM_ITL_BASE_DRAFT_MODEL is set."
        )

    _set_env_from_args(args, draft_model)
    vllm_args = _rewrite_or_add_speculative_config(
        vllm_args,
        num_speculative_tokens=args.itl_base_num_speculative_tokens,
    )
    install_patch()

    if vllm_args and vllm_args[0] == "serve":
        sys.argv = ["vllm", *vllm_args]
    else:
        sys.argv = ["vllm", "serve", *vllm_args]

    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch vLLM with first-paper ITL_BASE speculative decoding. "
            "All unrecognized arguments are forwarded to `vllm serve`."
        ),
        add_help=True,
    )
    parser.add_argument("--itl-base-draft-model", default=None)
    parser.add_argument(
        "--itl-base-method",
        choices=("auto", "slem", "tli"),
        default=os.getenv("VLLM_ITL_BASE_METHOD", "auto"),
    )
    parser.add_argument("--itl-base-num-speculative-tokens", type=int, default=5)
    parser.add_argument("--itl-base-max-draft-tokens", type=int, default=None)
    parser.add_argument("--itl-base-max-context-tokens", type=int, default=None)
    parser.add_argument("--itl-base-draft-device", default=None)
    parser.add_argument("--itl-base-draft-device-map", default=None)
    parser.add_argument("--itl-base-draft-dtype", default=None)
    parser.add_argument("--itl-base-draft-tp-rank", type=int, default=None)
    parser.add_argument("--itl-base-assistant-lookbehind", type=int, default=None)
    parser.add_argument("--itl-base-target-lookbehind", type=int, default=None)
    parser.add_argument("--itl-base-max-cached-requests", type=int, default=None)
    parser.add_argument("--itl-base-tli-min-intersection", type=int, default=None)
    parser.add_argument(
        "--itl-base-add-special-tokens",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--itl-base-draft-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--itl-base-strict-tli-probs",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--itl-base-log-proposals",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def _set_env_from_args(args: argparse.Namespace, draft_model: str) -> None:
    os.environ["VLLM_ITL_BASE_ENABLE"] = "1"
    os.environ["VLLM_ITL_BASE_DRAFT_MODEL"] = draft_model
    os.environ["VLLM_ITL_BASE_METHOD"] = args.itl_base_method
    _set_env_if_not_none("VLLM_ITL_BASE_MAX_DRAFT_TOKENS", args.itl_base_max_draft_tokens)
    _set_env_if_not_none(
        "VLLM_ITL_BASE_MAX_CONTEXT_TOKENS", args.itl_base_max_context_tokens
    )
    _set_env_if_not_none("VLLM_ITL_BASE_DRAFT_DEVICE", args.itl_base_draft_device)
    _set_env_if_not_none(
        "VLLM_ITL_BASE_DRAFT_DEVICE_MAP", args.itl_base_draft_device_map
    )
    _set_env_if_not_none("VLLM_ITL_BASE_DRAFT_DTYPE", args.itl_base_draft_dtype)
    _set_env_if_not_none("VLLM_ITL_BASE_DRAFT_TP_RANK", args.itl_base_draft_tp_rank)
    _set_env_if_not_none(
        "VLLM_ITL_BASE_ASSISTANT_LOOKBEHIND",
        args.itl_base_assistant_lookbehind,
    )
    _set_env_if_not_none(
        "VLLM_ITL_BASE_TARGET_LOOKBEHIND",
        args.itl_base_target_lookbehind,
    )
    _set_env_if_not_none(
        "VLLM_ITL_BASE_MAX_CACHED_REQUESTS",
        args.itl_base_max_cached_requests,
    )
    _set_env_if_not_none(
        "VLLM_ITL_BASE_TLI_MIN_INTERSECTION",
        args.itl_base_tli_min_intersection,
    )
    _set_env_bool_if_not_none(
        "VLLM_ITL_BASE_ADD_SPECIAL_TOKENS",
        args.itl_base_add_special_tokens,
    )
    _set_env_bool_if_not_none(
        "VLLM_ITL_BASE_ENABLE_DRAFT_CACHE",
        args.itl_base_draft_cache,
    )
    _set_env_bool_if_not_none(
        "VLLM_ITL_BASE_STRICT_TLI_PROBS",
        args.itl_base_strict_tli_probs,
    )
    _set_env_bool_if_not_none(
        "VLLM_ITL_BASE_LOG_PROPOSALS",
        args.itl_base_log_proposals,
    )
    _enable_vllm_plugin()


def _rewrite_or_add_speculative_config(
    argv: list[str],
    *,
    num_speculative_tokens: int,
) -> list[str]:
    config = {
        "method": "ngram",
        "model": "ngram",
        "num_speculative_tokens": int(num_speculative_tokens),
        "prompt_lookup_min": 1,
        "prompt_lookup_max": 1,
    }
    rewritten = list(argv)
    for index, item in enumerate(rewritten):
        if item == "--speculative-config" and index + 1 < len(rewritten):
            user_config = _parse_json_object(rewritten[index + 1])
            user_config.update(config)
            rewritten[index + 1] = json.dumps(user_config, separators=(",", ":"))
            return rewritten
        if item.startswith("--speculative-config="):
            user_config = _parse_json_object(item.split("=", 1)[1])
            user_config.update(config)
            rewritten[index] = (
                "--speculative-config="
                + json.dumps(user_config, separators=(",", ":"))
            )
            return rewritten

    rewritten.extend(
        [
            "--speculative-config",
            json.dumps(config, separators=(",", ":")),
        ]
    )
    return rewritten


def _parse_json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--speculative-config must be a JSON object.")
    return parsed


def _set_env_if_not_none(name: str, value: object | None) -> None:
    if value is not None:
        os.environ[name] = str(value)


def _set_env_bool_if_not_none(name: str, value: bool | None) -> None:
    if value is not None:
        os.environ[name] = "1" if value else "0"


def _enable_vllm_plugin() -> None:
    name = "vllm_itl_base"
    current = os.environ.get("VLLM_PLUGINS")
    if current is None:
        os.environ["VLLM_PLUGINS"] = name
        return
    plugins = [item for item in current.split(",") if item]
    if name not in plugins:
        plugins.append(name)
    os.environ["VLLM_PLUGINS"] = ",".join(plugins)


if __name__ == "__main__":
    main()
