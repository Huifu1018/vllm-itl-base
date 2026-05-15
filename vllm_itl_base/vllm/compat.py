"""Compatibility patch for vLLM 0.15.1 speculative decoding."""

from __future__ import annotations

import importlib
import logging
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

_PATCHED = False


def install_patch() -> None:
    """Install the ITL_BASE proposer into vLLM's ngram spec-decode slot.

    vLLM 0.15.1 has no public out-of-tree proposer registry for speculative
    decoding. The least invasive stable hook is the ngram proposer: it accepts
    target-token proposals and already supports deterministic draft verification.
    TLI additionally needs draft probabilities, so this patch also passes
    proposer-owned probability rows into the existing rejection sampler.
    """

    global _PATCHED
    if _PATCHED:
        return

    from vllm_itl_base.vllm.proposer import VllmITLBaseProposer

    ngram_mod = importlib.import_module("vllm.v1.spec_decode.ngram_proposer")
    setattr(ngram_mod, "NgramProposer", VllmITLBaseProposer)

    try:
        runner_mod = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    except Exception:
        logger.exception("Failed to patch vLLM GPUModelRunner for ITL_BASE.")
        raise

    setattr(runner_mod, "NgramProposer", VllmITLBaseProposer)
    _patch_gpu_model_runner(runner_mod)
    _PATCHED = True
    logger.info("Installed vLLM ITL_BASE patch for vLLM 0.15.1.")


def _patch_gpu_model_runner(runner_mod: ModuleType) -> None:
    cls = getattr(runner_mod, "GPUModelRunner")
    if not hasattr(cls, "_itl_base_original_sample"):
        cls._itl_base_original_sample = cls._sample
        cls._sample = _patched_sample
    if not hasattr(cls, "_itl_base_original_propose_draft_token_ids"):
        cls._itl_base_original_propose_draft_token_ids = cls.propose_draft_token_ids
        cls.propose_draft_token_ids = _patched_propose_draft_token_ids


def _is_itl_drafter(drafter: object) -> bool:
    return bool(getattr(drafter, "is_itl_base_proposer", False))


def _patched_sample(
    self: Any,
    logits: Any,
    spec_decode_metadata: Any,
) -> Any:
    drafter = getattr(self, "drafter", None)
    if not _is_itl_drafter(drafter) or spec_decode_metadata is None:
        return self._itl_base_original_sample(logits, spec_decode_metadata)

    sampling_metadata = self.input_batch.sampling_metadata
    self.input_batch.update_async_output_token_ids()
    if spec_decode_metadata is None:
        return self.sampler(logits=logits, sampling_metadata=sampling_metadata)

    if self.use_async_scheduling and self._draft_token_req_ids is not None:
        draft_token_ids_cpu, _ = self._get_draft_token_ids_cpu()
        self.input_batch.update_async_spec_token_ids(draft_token_ids_cpu)

    draft_probs = None
    if logits is not None and hasattr(drafter, "take_draft_probs"):
        draft_probs = drafter.take_draft_probs(
            req_ids=list(self.input_batch.req_ids),
            num_draft_tokens=list(spec_decode_metadata.num_draft_tokens),
            device=logits.device,
            vocab_size=int(logits.shape[-1]),
        )

    return self.rejection_sampler(
        spec_decode_metadata,
        draft_probs,
        logits,
        sampling_metadata,
    )


def _patched_propose_draft_token_ids(
    self: Any,
    scheduler_output: Any,
    sampled_token_ids: Any,
    sampling_metadata: Any,
    hidden_states: Any,
    sample_hidden_states: Any,
    aux_hidden_states: Any,
    spec_decode_metadata: Any,
    common_attn_metadata: Any,
    slot_mappings: Any,
) -> Any:
    spec_config = self.speculative_config
    drafter = getattr(self, "drafter", None)
    if (
        spec_config is not None
        and spec_config.method == "ngram"
        and _is_itl_drafter(drafter)
    ):
        if not isinstance(sampled_token_ids, list):
            raise TypeError("ITL_BASE ngram path expects CPU list sampled_token_ids.")
        return drafter.propose(
            sampled_token_ids,
            self.input_batch.num_tokens_no_spec,
            self.input_batch.token_ids_cpu,
            slot_mappings=slot_mappings,
            sampling_metadata=sampling_metadata,
            req_ids=list(self.input_batch.req_ids),
        )

    return self._itl_base_original_propose_draft_token_ids(
        scheduler_output,
        sampled_token_ids,
        sampling_metadata,
        hidden_states,
        sample_hidden_states,
        aux_hidden_states,
        spec_decode_metadata,
        common_attn_metadata,
        slot_mappings,
    )
