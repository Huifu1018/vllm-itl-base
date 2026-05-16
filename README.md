# vLLM ITL Base

`vllm-itl-base` integrates the first paper's heterogeneous-vocabulary
speculative decoding baselines with **vLLM 0.15.1**.

Paper: `Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies`

## What It Implements

- **SLEM / UAG path**: use any HF draft model, decode draft tokens to text,
  re-tokenize with the target tokenizer, then align the suffix and propose
  target-token ids to vLLM.
- **TLI / USD path**: build the token-level vocabulary intersection, map draft
  probability rows into target-vocabulary space, and pass them to vLLM's
  rejection sampler.
- **vLLM engine integration**: installed through a `vllm.general_plugins`
  entry point and launched with `vllm-itl-base-serve`.

This is not vLLM's ordinary same-tokenizer `draft_model` mode. It is for cases
where target and draft tokenizers are different, and you do not have a
specialized draft model.

## Install

Use a clean Python environment that can install vLLM 0.15.1.

```bash
uv venv
source .venv/bin/activate
uv pip install "vllm-itl-base[vllm] @ git+https://github.com/Huifu1018/vllm-itl-base.git"
vllm-itl-base-preflight
```

With pip:

```bash
python -m pip install "vllm-itl-base[vllm] @ git+https://github.com/Huifu1018/vllm-itl-base.git"
```

The package is version-pinned for vLLM `0.15.1`. Run `vllm-itl-base-preflight`
after installation; `vllm_version_ok` should be `True`.

## Serve A Model

Basic SLEM example:

```bash
vllm-itl-base-serve nvidia/MiniMax-M2.7-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --generation-config vllm \
  --tensor-parallel-size 4 \
  --itl-base-draft-model Qwen/Qwen2.5-1.5B-Instruct \
  --itl-base-draft-device cuda:0 \
  --itl-base-draft-tp-rank 0 \
  --itl-base-method slem \
  --itl-base-num-speculative-tokens 2 \
  --itl-base-max-context-tokens 2048
```

Sampling-oriented TLI example:

```bash
vllm-itl-base-serve nvidia/MiniMax-M2.7-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --generation-config vllm \
  --tensor-parallel-size 4 \
  --itl-base-draft-model Qwen/Qwen2.5-1.5B-Instruct \
  --itl-base-draft-device cuda:0 \
  --itl-base-draft-tp-rank 0 \
  --itl-base-method tli \
  --itl-base-num-speculative-tokens 2 \
  --itl-base-max-context-tokens 2048
```

`auto` mode:

```bash
vllm-itl-base-serve nvidia/MiniMax-M2.7-NVFP4 \
  --trust-remote-code \
  --generation-config vllm \
  --tensor-parallel-size 4 \
  --itl-base-draft-model Qwen/Qwen2.5-1.5B-Instruct \
  --itl-base-draft-device cuda:0 \
  --itl-base-method auto
```

`auto` uses SLEM for all-greedy batches and TLI for non-greedy batches.
When tensor parallelism is enabled, only the configured draft TP rank runs the
HF draft model; the proposal tokens and TLI probability rows are broadcast to
the other TP ranks.

## How Arguments Work

Pass ordinary `vllm serve` arguments as usual. The wrapper consumes only
`--itl-base-*` flags and forwards everything else to vLLM.

Internally the wrapper injects this vLLM speculative config:

```json
{
  "method": "ngram",
  "model": "ngram",
  "num_speculative_tokens": 5,
  "prompt_lookup_min": 1,
  "prompt_lookup_max": 1
}
```

The vLLM `ngram` method is only used as an engine hook. The proposer is replaced
by `VllmITLBaseProposer` at runtime.

## Useful Flags

- `--itl-base-draft-model`: HF path for the ordinary draft model.
- `--itl-base-method`: `slem`, `tli`, or `auto`.
- `--itl-base-num-speculative-tokens`: draft tokens per step.
- `--itl-base-max-draft-tokens`: cap assistant tokens generated before SLEM
  alignment returns enough target tokens.
- `--itl-base-assistant-lookbehind`: assistant-token context window for SLEM.
- `--itl-base-target-lookbehind`: target-token suffix window for SLEM.
- `--itl-base-draft-device`: move the HF draft model to a device, for example
  `cuda:0`. If this and `--itl-base-draft-device-map` are omitted,
  Transformers normally leaves the draft on CPU, which is too slow for serving.
- `--itl-base-draft-device-map`: pass a Transformers `device_map`, for example
  `auto`.
- `--itl-base-draft-dtype`: `auto`, `float16`, `bfloat16`, or `float32`.
- `--itl-base-draft-tp-rank`: local tensor-parallel rank that loads and runs
  the HF draft model. Default: `0`; keep this at `0` for vLLM runtimes that
  use message-queue object broadcast.
- `--no-itl-base-draft-cache`: disable draft KV cache reuse.
- `--itl-base-log-proposals`: log proposal length and cache events.

Environment variables with the same names are also supported, using the
`VLLM_ITL_BASE_` prefix.

## Check Acceptance Rate

vLLM already exports speculative decoding metrics. For Prometheus:

```promql
rate(vllm:spec_decode_num_accepted_tokens_total[1m])
/
rate(vllm:spec_decode_num_draft_tokens_total[1m])
```

Accepted tokens per step:

```promql
rate(vllm:spec_decode_num_accepted_tokens_total[1m])
/
rate(vllm:spec_decode_num_drafts[1m])
```

## Practical Defaults

For MiniMax/Kimi/GLM/DeepSeek targets without a dedicated draft model, start
with:

- method: `slem`
- draft: `Qwen/Qwen2.5-1.5B-Instruct` or another small instruction model
- `num_speculative_tokens`: `2` or `3` until acceptance is stable
- `--itl-base-draft-device cuda:0`, or another visible GPU/spare GPU
- `--itl-base-max-context-tokens 2048` for the first benchmark pass
- temperature: `0` for the first benchmark pass
- `--generation-config vllm` to avoid model-card sampling defaults changing
  the benchmark

Use `tli` when the workload is sampling-heavy and the draft/target tokenizers
have a useful token-level intersection. If vLLM logs show `Avg Draft acceptance
rate: 0.0%`, disable TLI for that draft/target pair or switch to SLEM.

## Limitations

- This package is tied to vLLM `0.15.1` internals.
- The draft model is loaded through Hugging Face Transformers, separate from
  vLLM's target model executor. Under tensor parallelism it is loaded only on
  `--itl-base-draft-tp-rank`.
- TLI stores full target-vocabulary probability rows for draft positions; use a
  modest `num_speculative_tokens` for very large vocabularies.
- SLEM quality depends strongly on tokenizer alignment and draft model quality.

## Development Checks

```bash
python -m unittest discover -s tests
python -m compileall vllm_itl_base
python -m build
```
