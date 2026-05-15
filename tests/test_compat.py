import sys
import types
import unittest
from unittest.mock import patch


class CompatTests(unittest.TestCase):
    def test_install_patch_replaces_ngram_and_runner_methods(self):
        modules = _fake_vllm_modules()
        with patch.dict(sys.modules, modules):
            import vllm_itl_base.vllm.compat as compat

            compat._PATCHED = False
            compat.install_patch()

            runner_cls = modules["vllm.v1.worker.gpu_model_runner"].GPUModelRunner
            ngram_mod = modules["vllm.v1.spec_decode.ngram_proposer"]
            self.assertTrue(hasattr(runner_cls, "_itl_base_original_sample"))
            self.assertEqual(ngram_mod.NgramProposer.__name__, "VllmITLBaseProposer")


def _fake_vllm_modules():
    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.spec_decode": types.ModuleType("vllm.v1.spec_decode"),
        "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
    }
    ngram_mod = types.ModuleType("vllm.v1.spec_decode.ngram_proposer")

    class OriginalNgram:
        pass

    ngram_mod.NgramProposer = OriginalNgram
    modules["vllm.v1.spec_decode.ngram_proposer"] = ngram_mod

    runner_mod = types.ModuleType("vllm.v1.worker.gpu_model_runner")

    class GPUModelRunner:
        def _sample(self, logits, spec_decode_metadata):
            return None

        def propose_draft_token_ids(self, *args, **kwargs):
            return None

    runner_mod.GPUModelRunner = GPUModelRunner
    runner_mod.NgramProposer = OriginalNgram
    modules["vllm.v1.worker.gpu_model_runner"] = runner_mod
    return modules


if __name__ == "__main__":
    unittest.main()
