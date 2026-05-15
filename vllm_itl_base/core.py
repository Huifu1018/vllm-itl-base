"""First-paper heterogeneous-vocabulary helpers.

The paper's deployable baselines split into two practical paths:

* SLEM/UAG for deterministic decoding: translate draft tokens through text and
  align the re-tokenized suffix.
* TLI/USD for probabilistic decoding: restrict the draft distribution to the
  token-level vocabulary intersection, then map probabilities to target ids.

This module is intentionally framework-light. Torch is imported only by the
runtime proposer so these helpers remain unit-testable without model packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


def _as_int_tuple(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def tokenizer_vocab(tokenizer: object) -> dict[str, int]:
    """Return a tokenizer vocabulary as a normal dict."""

    vocab = getattr(tokenizer, "get_vocab", None)
    if callable(vocab):
        return dict(vocab())
    vocab = getattr(tokenizer, "vocab", None)
    if isinstance(vocab, Mapping):
        return dict(vocab)
    raise TypeError("tokenizer must expose get_vocab() or vocab.")


def _leading_space_marker(tokenizer: object) -> str | None:
    """Infer the tokenizer-specific visible marker for a leading space token."""

    try:
        ids = tokenizer(" ", add_special_tokens=False)["input_ids"]
    except Exception:
        return None
    if not ids:
        return None
    try:
        tokens = tokenizer.convert_ids_to_tokens(ids)
    except Exception:
        return None
    if not tokens:
        return None
    token = tokens[0]
    return token[0] if token else None


def normalize_assistant_vocab_keys(
    target_tokenizer: object,
    assistant_tokenizer: object,
    assistant_vocab: Mapping[str, int],
) -> dict[str, int]:
    """Normalize leading-space markers before intersecting vocabularies.

    This mirrors the practical detail from the Transformers UAG/TLI
    implementation: BPE tokenizers can represent a leading space using
    different visible prefix characters. Replacing only the first marker keeps
    the intersection useful without changing non-prefix token text.
    """

    target_space = _leading_space_marker(target_tokenizer)
    assistant_space = _leading_space_marker(assistant_tokenizer)
    if not target_space or not assistant_space or target_space == assistant_space:
        return dict(assistant_vocab)

    normalized: dict[str, int] = {}
    for token, token_id in assistant_vocab.items():
        if token.startswith(assistant_space):
            token = token.replace(assistant_space, target_space, 1)
        normalized[token] = int(token_id)
    return normalized


@dataclass(frozen=True)
class VocabIntersection:
    """Assistant-to-target id mapping for Token-Level Intersection."""

    assistant_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    assistant_to_target: dict[int, int]
    target_to_assistant: dict[int, int]
    assistant_vocab_size: int
    target_vocab_size: int

    @classmethod
    def from_tokenizers(
        cls,
        *,
        target_tokenizer: object,
        assistant_tokenizer: object,
        target_vocab_size: int | None = None,
    ) -> "VocabIntersection":
        target_vocab = tokenizer_vocab(target_tokenizer)
        assistant_vocab_raw = tokenizer_vocab(assistant_tokenizer)
        assistant_vocab = normalize_assistant_vocab_keys(
            target_tokenizer,
            assistant_tokenizer,
            assistant_vocab_raw,
        )

        pairs: list[tuple[int, int]] = []
        for token, assistant_id in assistant_vocab.items():
            target_id = target_vocab.get(token)
            if target_id is not None:
                pairs.append((int(assistant_id), int(target_id)))
        pairs.sort(key=lambda pair: pair[0])

        assistant_to_target = {assistant_id: target_id for assistant_id, target_id in pairs}
        target_to_assistant = {}
        for assistant_id, target_id in pairs:
            target_to_assistant.setdefault(target_id, assistant_id)

        if target_vocab_size is None:
            target_vocab_size = max(target_vocab.values()) + 1 if target_vocab else 0
        assistant_vocab_size = (
            max(assistant_vocab_raw.values()) + 1 if assistant_vocab_raw else 0
        )
        return cls(
            assistant_ids=tuple(assistant_id for assistant_id, _ in pairs),
            target_ids=tuple(target_id for _, target_id in pairs),
            assistant_to_target=assistant_to_target,
            target_to_assistant=target_to_assistant,
            assistant_vocab_size=assistant_vocab_size,
            target_vocab_size=int(target_vocab_size),
        )

    def suppress_assistant_ids(self) -> tuple[int, ...]:
        supported = set(self.assistant_ids)
        return tuple(
            token_id
            for token_id in range(self.assistant_vocab_size)
            if token_id not in supported
        )

    def require_non_empty(self) -> None:
        if not self.assistant_ids:
            raise ValueError(
                "Target and draft tokenizers have no token-level intersection. "
                "Use SLEM/greedy mode or choose a different draft model."
            )


@dataclass(frozen=True)
class DiagonalMatch:
    source_start: int
    target_start: int
    length: int


def longest_diagonal_match(
    source: Sequence[int],
    target: Sequence[int],
) -> DiagonalMatch | None:
    """Return the longest contiguous equal-token diagonal between two lists."""

    if not source or not target:
        return None

    best = DiagonalMatch(0, 0, 0)
    for source_start, source_token in enumerate(source):
        for target_start, target_token in enumerate(target):
            if source_token != target_token:
                continue
            length = 1
            while (
                source_start + length < len(source)
                and target_start + length < len(target)
                and source[source_start + length] == target[target_start + length]
            ):
                length += 1
            if length > best.length:
                best = DiagonalMatch(source_start, target_start, length)
    return best if best.length > 0 else None


def new_tokens_after_reencoded_window(
    *,
    target_suffix: Sequence[int],
    reencoded_window: Sequence[int],
) -> tuple[int, ...]:
    """Extract newly generated target ids from a re-encoded assistant window.

    This is the string-level exact-match alignment used by UAG/SLEM. The
    overlap between the old target suffix and the re-encoded assistant window is
    identified by the longest contiguous diagonal. Everything after that
    overlap in the re-encoded window is considered newly proposed target tokens.
    """

    target_suffix = _as_int_tuple(target_suffix)
    reencoded_window = _as_int_tuple(reencoded_window)
    if not reencoded_window:
        return ()
    match = longest_diagonal_match(target_suffix, reencoded_window)
    if match is None:
        return reencoded_window
    start = match.target_start + match.length
    return reencoded_window[start:]


def decode_ids(tokenizer: object, token_ids: Sequence[int]) -> str:
    """Decode token ids while preserving tokenizer boundary behavior."""

    token_ids = [int(token_id) for token_id in token_ids]
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    except TypeError:
        return tokenizer.decode(token_ids)


def encode_text(tokenizer: object, text: str, *, add_special_tokens: bool = False) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    return _as_int_tuple(encoded["input_ids"])


def slem_target_proxies_from_assistant_window(
    *,
    target_tokenizer: object,
    assistant_tokenizer: object,
    current_target_ids: Sequence[int],
    assistant_context_ids: Sequence[int],
    assistant_new_ids: Sequence[int],
    assistant_lookbehind: int,
    target_lookbehind: int,
    add_special_tokens: bool = False,
) -> tuple[int, ...]:
    """Convert assistant draft tokens to target ids using SLEM/UAG alignment."""

    if not assistant_new_ids:
        return ()

    assistant_prefix = _as_int_tuple(assistant_context_ids)[-assistant_lookbehind:]
    assistant_window = assistant_prefix + _as_int_tuple(assistant_new_ids)
    reencoded_window = encode_text(
        target_tokenizer,
        decode_ids(assistant_tokenizer, assistant_window),
        add_special_tokens=add_special_tokens,
    )
    target_suffix = _as_int_tuple(current_target_ids)[-target_lookbehind:]
    proxy_ids = new_tokens_after_reencoded_window(
        target_suffix=target_suffix,
        reencoded_window=reencoded_window,
    )

    if proxy_ids:
        return proxy_ids

    # Conservative fallback: translate only the new assistant text. This avoids
    # returning an empty proposal when a tokenizer pair has no suffix overlap in
    # the chosen lookbehind window.
    return encode_text(
        target_tokenizer,
        decode_ids(assistant_tokenizer, assistant_new_ids),
        add_special_tokens=add_special_tokens,
    )
