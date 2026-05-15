"""HF draft proposer for first-paper SLEM/TLI baselines."""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

from vllm_itl_base.core import (
    VocabIntersection,
    decode_ids,
    encode_text,
    slem_target_proxies_from_assistant_window,
)

from .config import BaseVLLMConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplingRequest:
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0


@dataclass(frozen=True)
class BaseProposal:
    method: str
    draft_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    draft_prob_rows: object | None
    cache_event: str
    draft_context_tokens: int


@dataclass
class DraftRequestState:
    rid: str
    text: str
    input_ids: tuple[int, ...]
    past_key_values: object
    next_token_logits: object


@dataclass
class DraftProposerStats:
    proposals: int = 0
    slem_proposals: int = 0
    tli_proposals: int = 0
    proposed_target_tokens: int = 0
    proposed_draft_tokens: int = 0
    cache_hits: int = 0
    cache_extensions: int = 0
    cache_rebuilds: int = 0
    cache_evictions: int = 0
    empty_proposals: int = 0
    failed_proposals: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "proposals": self.proposals,
            "slem_proposals": self.slem_proposals,
            "tli_proposals": self.tli_proposals,
            "proposed_target_tokens": self.proposed_target_tokens,
            "proposed_draft_tokens": self.proposed_draft_tokens,
            "cache_hits": self.cache_hits,
            "cache_extensions": self.cache_extensions,
            "cache_rebuilds": self.cache_rebuilds,
            "cache_evictions": self.cache_evictions,
            "empty_proposals": self.empty_proposals,
            "failed_proposals": self.failed_proposals,
        }


class HeterogeneousDraftProposer:
    """Generate SLEM or TLI proposals from an ordinary HF draft model."""

    def __init__(
        self,
        *,
        draft_model_path: str,
        target_tokenizer: object,
        target_vocab_size: int,
        config: BaseVLLMConfig,
        trust_remote_code: bool,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.target_tokenizer = target_tokenizer
        self.target_vocab_size = int(target_vocab_size)
        self.draft_tokenizer = AutoTokenizer.from_pretrained(
            draft_model_path,
            trust_remote_code=trust_remote_code,
        )

        model_kwargs: dict[str, object] = {"trust_remote_code": trust_remote_code}
        if config.draft_dtype != "auto":
            model_kwargs["torch_dtype"] = _torch_dtype(torch, config.draft_dtype)
        else:
            model_kwargs["torch_dtype"] = "auto"
        if config.draft_device_map:
            model_kwargs["device_map"] = config.draft_device_map

        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_path,
            **model_kwargs,
        )
        if not config.draft_device_map and config.draft_device:
            self.draft_model.to(config.draft_device)
        self.draft_model.eval()

        self.intersection: VocabIntersection | None = None
        self._valid_assistant_ids = None
        self._valid_target_ids = None

        self._states: OrderedDict[str, DraftRequestState] = OrderedDict()
        self.stats = DraftProposerStats()

    def propose(
        self,
        rid: str,
        current_text: str,
        current_target_ids: Sequence[int],
        *,
        max_target_tokens: int,
        method: str,
        sampling: SamplingRequest | None = None,
    ) -> BaseProposal:
        self.stats.proposals += 1
        if max_target_tokens <= 0:
            self.stats.empty_proposals += 1
            return BaseProposal(method, (), (), None, "disabled", 0)

        try:
            if method == "slem":
                proposal = self._propose_slem(
                    rid,
                    current_text,
                    current_target_ids,
                    max_target_tokens=max_target_tokens,
                )
                self.stats.slem_proposals += 1
            elif method == "tli":
                proposal = self._propose_tli(
                    rid,
                    current_text,
                    max_target_tokens=max_target_tokens,
                    sampling=sampling or SamplingRequest(),
                )
                self.stats.tli_proposals += 1
            else:
                raise ValueError(f"Unsupported ITL_BASE method: {method}")

            if not proposal.target_token_ids:
                self.stats.empty_proposals += 1
            self.stats.proposed_target_tokens += len(proposal.target_token_ids)
            self.stats.proposed_draft_tokens += len(proposal.draft_token_ids)
            return proposal
        except Exception:
            self.stats.failed_proposals += 1
            raise

    def evict(self, rids: Sequence[str]) -> None:
        for rid in rids:
            if self._states.pop(str(rid), None) is not None:
                self.stats.cache_evictions += 1

    def clear(self) -> None:
        evicted = len(self._states)
        self._states.clear()
        self.stats.cache_evictions += evicted

    def cache_size(self) -> int:
        return len(self._states)

    def _propose_slem(
        self,
        rid: str,
        current_text: str,
        current_target_ids: Sequence[int],
        *,
        max_target_tokens: int,
    ) -> BaseProposal:
        import torch

        state, cache_event = self._ensure_state(rid, current_text)
        max_draft_tokens = self.config.max_draft_tokens
        if max_draft_tokens is None:
            max_draft_tokens = max(max_target_tokens * 4, max_target_tokens + 4)

        draft_ids: list[int] = []
        proxy_ids: tuple[int, ...] = ()
        generation_ids = list(state.input_ids)
        generation_past = self._fork_past_key_values(state.past_key_values)
        logits = state.next_token_logits
        context_len = len(state.input_ids)

        with torch.inference_mode():
            for _ in range(max_draft_tokens):
                next_token = int(torch.argmax(logits, dim=-1)[0])
                draft_ids.append(next_token)
                proxy_ids = slem_target_proxies_from_assistant_window(
                    target_tokenizer=self.target_tokenizer,
                    assistant_tokenizer=self.draft_tokenizer,
                    current_target_ids=current_target_ids,
                    assistant_context_ids=state.input_ids,
                    assistant_new_ids=draft_ids,
                    assistant_lookbehind=self.config.assistant_lookbehind,
                    target_lookbehind=self.config.target_lookbehind,
                    add_special_tokens=self.config.add_special_tokens,
                )
                if len(proxy_ids) >= max_target_tokens:
                    break

                generation_ids.append(next_token)
                context_len += 1
                logits, generation_past = self._forward_one(
                    token_id=next_token,
                    full_ids=generation_ids,
                    context_len=context_len,
                    past_key_values=generation_past,
                )
                eos_token_id = getattr(self.draft_tokenizer, "eos_token_id", None)
                if eos_token_id is not None and next_token == int(eos_token_id):
                    break

        return BaseProposal(
            method="slem",
            draft_token_ids=tuple(draft_ids),
            target_token_ids=tuple(int(token_id) for token_id in proxy_ids[:max_target_tokens]),
            draft_prob_rows=None,
            cache_event=cache_event,
            draft_context_tokens=len(state.input_ids),
        )

    def _propose_tli(
        self,
        rid: str,
        current_text: str,
        *,
        max_target_tokens: int,
        sampling: SamplingRequest,
    ) -> BaseProposal:
        import torch

        state, cache_event = self._ensure_state(rid, current_text)
        self._ensure_intersection_tensors()
        assert self._valid_assistant_ids is not None
        assert self._valid_target_ids is not None

        max_draft_tokens = self.config.max_draft_tokens or max_target_tokens
        max_draft_tokens = min(max_draft_tokens, max_target_tokens)
        draft_ids: list[int] = []
        target_ids: list[int] = []
        prob_rows: list[object] = []

        generation_ids = list(state.input_ids)
        generation_past = self._fork_past_key_values(state.past_key_values)
        logits = state.next_token_logits
        context_len = len(state.input_ids)

        with torch.inference_mode():
            for _ in range(max_draft_tokens):
                target_probs, selected_assistant_id, selected_target_id = (
                    self._sample_tli_token(logits, sampling=sampling)
                )
                prob_rows.append(target_probs)
                draft_ids.append(selected_assistant_id)
                target_ids.append(selected_target_id)

                generation_ids.append(selected_assistant_id)
                context_len += 1
                logits, generation_past = self._forward_one(
                    token_id=selected_assistant_id,
                    full_ids=generation_ids,
                    context_len=context_len,
                    past_key_values=generation_past,
                )
                eos_token_id = getattr(self.draft_tokenizer, "eos_token_id", None)
                if eos_token_id is not None and selected_assistant_id == int(eos_token_id):
                    break

            # The final slot is useful for engines that need a target-only
            # bonus distribution. vLLM consumes only the draft-token rows.
            if prob_rows:
                target_probs, _, _ = self._sample_tli_token(logits, sampling=sampling)
                prob_rows.append(target_probs)

        return BaseProposal(
            method="tli",
            draft_token_ids=tuple(draft_ids),
            target_token_ids=tuple(target_ids),
            draft_prob_rows=prob_rows,
            cache_event=cache_event,
            draft_context_tokens=len(state.input_ids),
        )

    def _sample_tli_token(self, logits, *, sampling: SamplingRequest):
        import torch

        assert self._valid_assistant_ids is not None
        assert self._valid_target_ids is not None

        valid_assistant_ids = self._valid_assistant_ids
        valid_target_ids = self._valid_target_ids
        if valid_assistant_ids.device != logits.device:
            valid_assistant_ids = valid_assistant_ids.to(logits.device, non_blocking=True)
        if valid_target_ids.device != logits.device:
            valid_target_ids = valid_target_ids.to(logits.device, non_blocking=True)

        valid_logits = logits[0, valid_assistant_ids].float()
        temperature = max(float(sampling.temperature), 1e-5)
        scaled = valid_logits / temperature
        probs = torch.softmax(scaled, dim=-1)
        probs = _renormalize_top_k_top_p(probs, top_k=sampling.top_k, top_p=sampling.top_p)

        if sampling.temperature <= 0:
            index = int(torch.argmax(probs).item())
        else:
            index = int(torch.multinomial(probs, num_samples=1).item())

        target_probs = torch.zeros(
            (self.target_vocab_size,),
            dtype=torch.float32,
            device=logits.device,
        )
        target_probs.scatter_(0, valid_target_ids, probs)
        return (
            target_probs,
            int(valid_assistant_ids[index].item()),
            int(valid_target_ids[index].item()),
        )

    def _ensure_intersection_tensors(self) -> None:
        import torch

        if self.intersection is None:
            self.intersection = VocabIntersection.from_tokenizers(
                target_tokenizer=self.target_tokenizer,
                assistant_tokenizer=self.draft_tokenizer,
                target_vocab_size=self.target_vocab_size,
            )
            self.intersection.require_non_empty()
            if len(self.intersection.assistant_ids) < self.config.tli_min_intersection:
                raise ValueError(
                    "Token-level intersection is too small: "
                    f"{len(self.intersection.assistant_ids)} < "
                    f"{self.config.tli_min_intersection}."
                )
            logger.info(
                "Initialized TLI vocabulary intersection: assistant=%s target=%s common=%s",
                self.intersection.assistant_vocab_size,
                self.intersection.target_vocab_size,
                len(self.intersection.assistant_ids),
            )

        if self._valid_assistant_ids is None or self._valid_target_ids is None:
            device = self._input_device()
            self._valid_assistant_ids = torch.tensor(
                self.intersection.assistant_ids,
                dtype=torch.long,
                device=device,
            )
            self._valid_target_ids = torch.tensor(
                self.intersection.target_ids,
                dtype=torch.long,
                device=device,
            )

    def _ensure_state(self, rid: str, current_text: str) -> tuple[DraftRequestState, str]:
        import torch

        rid = str(rid)
        context_ids = tuple(self._context_ids(current_text))
        if not context_ids:
            raise ValueError("draft context must contain at least one token.")

        cached = self._states.get(rid) if self.config.enable_draft_cache else None
        if cached is not None and cached.input_ids == context_ids:
            self._states.move_to_end(rid)
            self.stats.cache_hits += 1
            return cached, "hit"

        if (
            cached is not None
            and cached.past_key_values is not None
            and len(context_ids) > len(cached.input_ids)
            and context_ids[: len(cached.input_ids)] == cached.input_ids
        ):
            suffix = context_ids[len(cached.input_ids) :]
            suffix_tensor = torch.tensor(
                [list(suffix)],
                dtype=torch.long,
                device=self._input_device(),
            )
            attention_mask = torch.ones(
                (1, len(context_ids)),
                dtype=torch.long,
                device=suffix_tensor.device,
            )
            with torch.inference_mode():
                outputs = self.draft_model(
                    input_ids=suffix_tensor,
                    attention_mask=attention_mask,
                    past_key_values=cached.past_key_values,
                    use_cache=True,
                )
            state = DraftRequestState(
                rid=rid,
                text=current_text,
                input_ids=context_ids,
                past_key_values=getattr(outputs, "past_key_values", None),
                next_token_logits=outputs.logits[:, -1, :],
            )
            self._store_state(state)
            self.stats.cache_extensions += 1
            return state, "extend"

        input_tensor = torch.tensor(
            [list(context_ids)],
            dtype=torch.long,
            device=self._input_device(),
        )
        attention_mask = torch.ones_like(input_tensor)
        with torch.inference_mode():
            outputs = self.draft_model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                use_cache=True,
            )
        state = DraftRequestState(
            rid=rid,
            text=current_text,
            input_ids=context_ids,
            past_key_values=getattr(outputs, "past_key_values", None),
            next_token_logits=outputs.logits[:, -1, :],
        )
        self._store_state(state)
        self.stats.cache_rebuilds += 1
        return state, "rebuild"

    def _forward_one(
        self,
        *,
        token_id: int,
        full_ids: Sequence[int],
        context_len: int,
        past_key_values: object,
    ):
        import torch

        if past_key_values is None:
            input_tensor = torch.tensor(
                [list(full_ids)],
                dtype=torch.long,
                device=self._input_device(),
            )
            attention_mask = torch.ones_like(input_tensor)
        else:
            input_tensor = torch.tensor(
                [[int(token_id)]],
                dtype=torch.long,
                device=self._input_device(),
            )
            attention_mask = torch.ones(
                (1, context_len),
                dtype=torch.long,
                device=input_tensor.device,
            )
        outputs = self.draft_model(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outputs.logits[:, -1, :], getattr(outputs, "past_key_values", None)

    def _store_state(self, state: DraftRequestState) -> None:
        if not self.config.enable_draft_cache:
            return
        self._states[state.rid] = state
        self._states.move_to_end(state.rid)
        while len(self._states) > self.config.max_cached_requests:
            self._states.popitem(last=False)
            self.stats.cache_evictions += 1

    def _context_ids(self, text: str) -> tuple[int, ...]:
        token_ids = encode_text(
            self.draft_tokenizer,
            text,
            add_special_tokens=self.config.add_special_tokens,
        )
        if self.config.max_context_tokens is not None:
            token_ids = token_ids[-self.config.max_context_tokens :]
        return token_ids

    def _input_device(self):
        try:
            return self.draft_model.device
        except AttributeError:
            return next(self.draft_model.parameters()).device

    def _fork_past_key_values(self, past_key_values):
        if past_key_values is None:
            return None
        if not self.config.clone_draft_cache:
            return past_key_values
        return _clone_cache(past_key_values)

    def debug_decode_target(self, token_ids: Sequence[int]) -> str:
        return decode_ids(self.target_tokenizer, token_ids)


def _clone_cache(value):
    if hasattr(value, "to_legacy_cache"):
        return value
    if isinstance(value, tuple):
        return tuple(_clone_cache(item) for item in value)
    if isinstance(value, list):
        return [_clone_cache(item) for item in value]
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    return value


def _torch_dtype(torch, dtype_name: str):
    normalized = dtype_name.lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported VLLM_ITL_BASE_DRAFT_DTYPE: {dtype_name}"
        ) from exc


def _renormalize_top_k_top_p(probs, *, top_k: int, top_p: float):
    import torch

    probs = probs.float()
    if top_k is not None and int(top_k) > 0 and int(top_k) < probs.numel():
        top_values, top_indices = torch.topk(probs, int(top_k))
        filtered = torch.zeros_like(probs)
        filtered.scatter_(0, top_indices, top_values)
        probs = filtered

    if top_p is not None and 0 < float(top_p) < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= float(top_p)
        if keep.numel() > 0:
            keep[0] = True
        filtered = torch.zeros_like(probs)
        filtered.scatter_(0, sorted_indices[keep], sorted_probs[keep])
        probs = filtered

    total = probs.sum()
    if not torch.isfinite(total) or total <= 0:
        return torch.full_like(probs, 1.0 / probs.numel())
    return probs / total
