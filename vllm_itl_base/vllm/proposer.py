"""vLLM-compatible proposer for first-paper SLEM/TLI baselines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from vllm_itl_base.config import BaseVLLMConfig
from vllm_itl_base.core import decode_ids
from vllm_itl_base.hf_proposer import (
    BaseProposal,
    HeterogeneousDraftProposer,
    SamplingRequest,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TargetRuntime:
    model_id: str
    tokenizer_id: str
    target_vocab_size: int
    trust_remote_code: bool
    max_model_len: int


@dataclass(frozen=True)
class _TPRuntime:
    rank: int
    world_size: int
    group: Any | None


class VllmITLBaseProposer:
    """Drop-in replacement for vLLM 0.15.1 ``NgramProposer``.

    The class returns target-token proposals just like vLLM's built-in ngram
    proposer. For TLI it also stores target-vocabulary draft probability rows,
    which the companion GPUModelRunner patch passes into vLLM's rejection
    sampler on the next step.
    """

    is_itl_base_proposer = True

    def __init__(self, vllm_config: Any) -> None:
        assert vllm_config.speculative_config is not None
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        self.k = int(self.speculative_config.num_speculative_tokens)
        self.config = BaseVLLMConfig.from_env()
        if not self.config.draft_model:
            raise ValueError(
                "VLLM_ITL_BASE_DRAFT_MODEL must be set. Use "
                "vllm-itl-base-serve --itl-base-draft-model ..."
            )

        self.tp_runtime = _detect_tp_runtime()
        if self.config.draft_tp_rank >= self.tp_runtime.world_size:
            raise ValueError(
                "VLLM_ITL_BASE_DRAFT_TP_RANK must be smaller than tensor "
                f"parallel size ({self.config.draft_tp_rank} >= "
                f"{self.tp_runtime.world_size})."
            )
        if (
            self.tp_runtime.world_size > 1
            and self.config.draft_tp_rank != 0
            and getattr(self.tp_runtime.group, "mq_broadcaster", None) is not None
        ):
            raise ValueError(
                "This vLLM TP runtime supports object broadcast only from TP "
                "rank 0; set VLLM_ITL_BASE_DRAFT_TP_RANK=0."
            )
        self._runs_draft_model = self.tp_runtime.rank == self.config.draft_tp_rank
        self.target_runtime = self._build_target_runtime(vllm_config)
        self.target_tokenizer = (
            self._load_target_tokenizer() if self._runs_draft_model else None
        )
        self._hf_proposer: HeterogeneousDraftProposer | None = None
        self._draft_probs_by_req_id: dict[str, list[Any]] = {}
        self._target_ids_by_req_id: dict[str, list[int]] = {}
        self._last_methods_by_req_id: dict[str, str] = {}
        if self._runs_draft_model:
            logger.info(
                "ITL_BASE TP rank %s/%s will run draft model %s.",
                self.tp_runtime.rank,
                self.tp_runtime.world_size,
                self.config.draft_model,
            )
        elif self.tp_runtime.world_size > 1:
            logger.info(
                "ITL_BASE TP rank %s/%s will receive draft proposals from "
                "TP rank %s; local draft model load is skipped.",
                self.tp_runtime.rank,
                self.tp_runtime.world_size,
                self.config.draft_tp_rank,
            )

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        if self._runs_draft_model:
            self._ensure_hf_proposer()

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: Any,
        token_ids_cpu: Any,
        slot_mappings: Any = None,
        sampling_metadata: Any = None,
        req_ids: Sequence[str] | None = None,
    ) -> list[list[int]]:
        req_ids = list(req_ids or [f"row:{i}" for i in range(len(sampled_token_ids))])
        if self._runs_draft_model:
            try:
                draft_token_ids = self._propose_local(
                    sampled_token_ids,
                    num_tokens_no_spec,
                    token_ids_cpu,
                    sampling_metadata=sampling_metadata,
                    req_ids=req_ids,
                )
                payload = self._export_payload(req_ids, draft_token_ids)
            except Exception as exc:
                if self.tp_runtime.world_size > 1:
                    self._broadcast_payload(
                        {
                            "error": f"{type(exc).__name__}: {exc}",
                            "draft_token_ids": [],
                        }
                    )
                raise
        else:
            payload = None

        payload = self._broadcast_payload(payload)
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(
                "ITL_BASE proposal failed on draft TP rank "
                f"{self.config.draft_tp_rank}: {payload['error']}"
            )
        if self._runs_draft_model:
            return draft_token_ids
        return self._apply_payload(req_ids, payload)

    def _propose_local(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: Any,
        token_ids_cpu: Any,
        *,
        sampling_metadata: Any = None,
        req_ids: Sequence[str],
    ) -> list[list[int]]:
        proposer = self._ensure_hf_proposer()
        draft_token_ids: list[list[int]] = []

        for index, sampled_ids in enumerate(sampled_token_ids):
            req_id = str(req_ids[index])
            self._draft_probs_by_req_id.pop(req_id, None)
            self._target_ids_by_req_id.pop(req_id, None)
            self._last_methods_by_req_id.pop(req_id, None)

            if not sampled_ids:
                draft_token_ids.append([])
                continue

            num_tokens = int(num_tokens_no_spec[index])
            if num_tokens >= self.target_runtime.max_model_len:
                draft_token_ids.append([])
                continue

            max_target_tokens = min(
                self.k,
                self.target_runtime.max_model_len - num_tokens,
            )
            if max_target_tokens <= 0:
                draft_token_ids.append([])
                continue

            current_target_ids = _slice_row(token_ids_cpu, index, num_tokens)
            current_text = decode_ids(self.target_tokenizer, current_target_ids)
            is_greedy = _is_request_greedy(sampling_metadata, index)
            method = self.config.method_for_request(is_greedy=is_greedy)
            sampling = _sampling_request(sampling_metadata, index)

            proposal = proposer.propose(
                req_id,
                current_text,
                current_target_ids,
                max_target_tokens=max_target_tokens,
                method=method,
                sampling=sampling,
            )
            target_ids = list(proposal.target_token_ids[:max_target_tokens])
            if method == "tli":
                self._store_tli_probs(req_id, proposal, len(target_ids))
            self._target_ids_by_req_id[req_id] = target_ids
            self._last_methods_by_req_id[req_id] = method
            if self.config.log_proposals:
                logger.info(
                    "ITL_BASE proposal req=%s method=%s target_tokens=%d "
                    "draft_tokens=%d cache=%s",
                    req_id,
                    method,
                    len(target_ids),
                    len(proposal.draft_token_ids),
                    proposal.cache_event,
                )
            draft_token_ids.append(target_ids)

        return draft_token_ids

    def _export_payload(
        self,
        req_ids: Sequence[str],
        draft_token_ids: list[list[int]],
    ) -> dict[str, Any]:
        req_id_set = {str(req_id) for req_id in req_ids}
        return {
            "draft_token_ids": [
                [int(token_id) for token_id in row] for row in draft_token_ids
            ],
            "draft_probs_by_req_id": {
                req_id: [_tensor_to_cpu(row) for row in rows]
                for req_id, rows in self._draft_probs_by_req_id.items()
                if req_id in req_id_set
            },
            "target_ids_by_req_id": {
                req_id: [int(token_id) for token_id in token_ids]
                for req_id, token_ids in self._target_ids_by_req_id.items()
                if req_id in req_id_set
            },
            "last_methods_by_req_id": {
                req_id: str(method)
                for req_id, method in self._last_methods_by_req_id.items()
                if req_id in req_id_set
            },
        }

    def _apply_payload(
        self,
        req_ids: Sequence[str],
        payload: Any,
    ) -> list[list[int]]:
        if not isinstance(payload, dict):
            raise RuntimeError("ITL_BASE TP broadcast did not return a proposal payload.")

        for req_id in (str(req_id) for req_id in req_ids):
            self._draft_probs_by_req_id.pop(req_id, None)
            self._target_ids_by_req_id.pop(req_id, None)
            self._last_methods_by_req_id.pop(req_id, None)

        self._draft_probs_by_req_id.update(
            {
                str(req_id): list(rows)
                for req_id, rows in payload.get("draft_probs_by_req_id", {}).items()
            }
        )
        self._target_ids_by_req_id.update(
            {
                str(req_id): [int(token_id) for token_id in token_ids]
                for req_id, token_ids in payload.get("target_ids_by_req_id", {}).items()
            }
        )
        self._last_methods_by_req_id.update(
            {
                str(req_id): str(method)
                for req_id, method in payload.get("last_methods_by_req_id", {}).items()
            }
        )
        return [
            [int(token_id) for token_id in row]
            for row in payload.get("draft_token_ids", [])
        ]

    def _broadcast_payload(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        if self.tp_runtime.world_size == 1:
            assert payload is not None
            return payload
        if self.tp_runtime.group is None:
            raise RuntimeError(
                "ITL_BASE detected tensor parallelism but could not access the "
                "vLLM TP communication group."
            )
        return self.tp_runtime.group.broadcast_object(
            payload,
            src=self.config.draft_tp_rank,
        )

    def take_draft_probs(
        self,
        *,
        req_ids: Sequence[str],
        num_draft_tokens: Sequence[int],
        device: Any,
        vocab_size: int,
    ) -> Any | None:
        req_ids = [str(req_id) for req_id in req_ids]
        has_tli_rows = any(
            int(count) > 0 and req_id in self._draft_probs_by_req_id
            for req_id, count in zip(req_ids, num_draft_tokens, strict=False)
        )
        if not has_tli_rows:
            for req_id, count in zip(req_ids, num_draft_tokens, strict=False):
                if int(count) > 0:
                    self._target_ids_by_req_id.pop(req_id, None)
            return None

        import torch

        rows: list[Any] = []
        missing: list[str] = []
        for req_id, count in zip(req_ids, num_draft_tokens, strict=False):
            count = int(count)
            if count <= 0:
                continue
            stored = self._draft_probs_by_req_id.pop(req_id, None)
            if stored is None or len(stored) < count:
                if self._last_methods_by_req_id.get(req_id) == "slem":
                    target_ids = self._target_ids_by_req_id.pop(req_id, [])
                    rows.extend(
                        _one_hot_prob_row(token_id, vocab_size=vocab_size, device=device)
                        for token_id in target_ids[:count]
                    )
                    continue
                missing.append(req_id)
                continue
            self._target_ids_by_req_id.pop(req_id, None)
            rows.extend(stored[:count])

        if not rows:
            return None
        if missing and self.config.strict_tli_probs:
            raise RuntimeError(
                "Missing TLI draft probability rows for request ids: "
                f"{', '.join(missing)}"
            )

        normalized_rows = [
            _resize_prob_row(row, vocab_size=vocab_size, device=device)
            for row in rows
        ]
        return torch.stack(normalized_rows, dim=0).contiguous()

    def stats_snapshot(self) -> dict[str, int]:
        proposer = self._hf_proposer
        if proposer is None:
            return {}
        return proposer.stats.snapshot()

    def _store_tli_probs(
        self,
        req_id: str,
        proposal: BaseProposal,
        num_target_ids: int,
    ) -> None:
        rows = proposal.draft_prob_rows
        if rows is None or num_target_ids <= 0:
            return
        row_list = list(rows)[:num_target_ids]
        if len(row_list) != num_target_ids:
            if self.config.strict_tli_probs:
                raise RuntimeError(
                    "TLI proposal did not return one draft probability row per "
                    "target draft token."
                )
            return
        self._draft_probs_by_req_id[str(req_id)] = row_list

    def _ensure_hf_proposer(self) -> HeterogeneousDraftProposer:
        if self.target_tokenizer is None:
            raise RuntimeError(
                "ITL_BASE draft proposer was requested on a TP rank that is not "
                "configured to run the draft model."
            )
        if self._hf_proposer is None:
            logger.info(
                "Loading ITL_BASE draft model %s for target tokenizer %s.",
                self.config.draft_model,
                self.target_runtime.tokenizer_id,
            )
            self._hf_proposer = HeterogeneousDraftProposer(
                draft_model_path=str(self.config.draft_model),
                target_tokenizer=self.target_tokenizer,
                target_vocab_size=self.target_runtime.target_vocab_size,
                config=self.config,
                trust_remote_code=self.target_runtime.trust_remote_code,
            )
        return self._hf_proposer

    def _load_target_tokenizer(self) -> object:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            self.target_runtime.tokenizer_id,
            trust_remote_code=self.target_runtime.trust_remote_code,
        )

    @staticmethod
    def _build_target_runtime(vllm_config: Any) -> _TargetRuntime:
        model_config = vllm_config.model_config
        model_id = str(getattr(model_config, "model"))
        tokenizer_id = str(getattr(model_config, "tokenizer", None) or model_id)
        trust_remote_code = bool(getattr(model_config, "trust_remote_code", False))
        max_model_len = int(getattr(model_config, "max_model_len"))
        get_vocab_size = getattr(model_config, "get_vocab_size", None)
        if callable(get_vocab_size):
            target_vocab_size = int(get_vocab_size())
        else:
            target_vocab_size = int(getattr(model_config, "vocab_size", 0))
        if target_vocab_size <= 0:
            target_vocab_size = _tokenizer_len(tokenizer_id, trust_remote_code)
        return _TargetRuntime(
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            target_vocab_size=target_vocab_size,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
        )


def _slice_row(token_ids_cpu: Any, index: int, length: int) -> tuple[int, ...]:
    row = token_ids_cpu[index, :length]
    tolist = getattr(row, "tolist", None)
    if callable(tolist):
        row = tolist()
    return tuple(int(token_id) for token_id in row)


def _is_request_greedy(sampling_metadata: Any, index: int) -> bool:
    if sampling_metadata is None:
        return True
    if bool(getattr(sampling_metadata, "all_greedy", False)):
        return True
    temperature = getattr(sampling_metadata, "temperature", None)
    if temperature is None:
        return False
    return float(_tensor_item(temperature, index, 1.0)) == 0.0


def _sampling_request(sampling_metadata: Any, index: int) -> SamplingRequest:
    if sampling_metadata is None:
        return SamplingRequest(temperature=0.0)
    if _is_request_greedy(sampling_metadata, index):
        return SamplingRequest(temperature=0.0)
    return SamplingRequest(
        temperature=float(_tensor_item(getattr(sampling_metadata, "temperature", None), index, 1.0)),
        top_k=int(_tensor_item(getattr(sampling_metadata, "top_k", None), index, -1)),
        top_p=float(_tensor_item(getattr(sampling_metadata, "top_p", None), index, 1.0)),
    )


def _tensor_item(value: Any, index: int, default: float | int) -> float | int:
    if value is None:
        return default
    try:
        item = value[index]
    except Exception:
        item = value
    if hasattr(item, "item"):
        item = item.item()
    return item


def _resize_prob_row(row: Any, *, vocab_size: int, device: Any) -> Any:
    import torch

    row = row.to(device=device, dtype=torch.float32)
    width = int(row.shape[-1])
    if width == vocab_size:
        return row
    if width > vocab_size:
        return row[:vocab_size]
    padded = torch.zeros((vocab_size,), dtype=torch.float32, device=device)
    padded[:width] = row
    return padded


def _one_hot_prob_row(token_id: int, *, vocab_size: int, device: Any) -> Any:
    import torch

    row = torch.zeros((vocab_size,), dtype=torch.float32, device=device)
    token_id = int(token_id)
    if 0 <= token_id < vocab_size:
        row[token_id] = 1.0
    return row


def _tensor_to_cpu(row: Any) -> Any:
    detach = getattr(row, "detach", None)
    if callable(detach):
        return detach().cpu().float().contiguous()
    return row


def _detect_tp_runtime() -> _TPRuntime:
    try:
        from vllm.distributed import (  # type: ignore
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
            get_tp_group,
        )
    except Exception:
        return _TPRuntime(rank=0, world_size=1, group=None)

    try:
        rank = int(get_tensor_model_parallel_rank())
        world_size = int(get_tensor_model_parallel_world_size())
        group = get_tp_group() if world_size > 1 else None
        return _TPRuntime(rank=rank, world_size=world_size, group=group)
    except Exception:
        logger.debug("Could not detect vLLM tensor-parallel runtime.", exc_info=True)
        return _TPRuntime(rank=0, world_size=1, group=None)


def _tokenizer_len(tokenizer_id: str, trust_remote_code: bool) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=trust_remote_code,
    )
    return int(len(tokenizer))
