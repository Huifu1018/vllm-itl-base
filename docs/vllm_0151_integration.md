# vLLM 0.15.1 Integration Notes

This package targets vLLM 0.15.1 exactly.

vLLM's `draft_model` speculative path checks that target and draft vocabulary
sizes are equal, so it cannot directly serve the first paper's heterogeneous
vocabulary setup. The integration therefore uses the built-in `ngram` slot as
the engine hook because that slot accepts already-translated target token ids.

The package installs a `vllm.general_plugins` entry point and a launcher:

- `vllm-itl-base-serve` sets `VLLM_ITL_BASE_*` environment variables.
- The launcher rewrites vLLM's speculative config to `method=ngram`.
- The plugin replaces vLLM's `NgramProposer` with `VllmITLBaseProposer`.
- For TLI, the plugin also patches `GPUModelRunner._sample` so draft
  probability rows are passed to vLLM's existing rejection sampler.

Supported routes:

- `slem`: ordinary HF draft model generates greedy assistant tokens; the text
  is decoded and re-tokenized by the target tokenizer; suffix alignment returns
  target-token proposals.
- `tli`: ordinary HF draft model samples over the token-level vocabulary
  intersection; probability rows are mapped into target-vocabulary space before
  verification.
- `auto`: uses `slem` for greedy requests and `tli` for non-greedy requests.

The implementation deliberately avoids patching vLLM source files. The tradeoff
is that this package is version-pinned to vLLM 0.15.1 internals.
