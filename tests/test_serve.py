import json
import os
import unittest
from unittest.mock import patch

from vllm_itl_base.cli.serve import (
    _enable_vllm_plugin,
    _set_env_from_args,
    _rewrite_or_add_speculative_config,
)


class ServeTests(unittest.TestCase):
    def test_adds_speculative_config(self):
        argv = _rewrite_or_add_speculative_config(["target"], num_speculative_tokens=4)
        self.assertIn("--speculative-config", argv)
        config = json.loads(argv[argv.index("--speculative-config") + 1])
        self.assertEqual(config["method"], "ngram")
        self.assertEqual(config["num_speculative_tokens"], 4)

    def test_rewrites_existing_speculative_config(self):
        argv = _rewrite_or_add_speculative_config(
            ["--speculative-config", '{"disable_by_batch_size":16}'],
            num_speculative_tokens=3,
        )
        config = json.loads(argv[1])
        self.assertEqual(config["disable_by_batch_size"], 16)
        self.assertEqual(config["model"], "ngram")
        self.assertEqual(config["prompt_lookup_min"], 1)

    def test_enable_vllm_plugin_appends(self):
        with patch.dict(os.environ, {"VLLM_PLUGINS": "foo"}, clear=True):
            _enable_vllm_plugin()
            self.assertEqual(os.environ["VLLM_PLUGINS"], "foo,vllm_itl_base")

    def test_sets_draft_tp_rank_env(self):
        args = type(
            "Args",
            (),
            {
                "itl_base_method": "slem",
                "itl_base_max_draft_tokens": None,
                "itl_base_max_context_tokens": None,
                "itl_base_draft_device": None,
                "itl_base_draft_device_map": None,
                "itl_base_draft_dtype": None,
                "itl_base_draft_tp_rank": 2,
                "itl_base_assistant_lookbehind": None,
                "itl_base_target_lookbehind": None,
                "itl_base_max_cached_requests": None,
                "itl_base_tli_min_intersection": None,
                "itl_base_add_special_tokens": None,
                "itl_base_draft_cache": None,
                "itl_base_strict_tli_probs": None,
                "itl_base_log_proposals": None,
            },
        )()
        with patch.dict(os.environ, {}, clear=True):
            _set_env_from_args(args, "draft")
            self.assertEqual(os.environ["VLLM_ITL_BASE_DRAFT_TP_RANK"], "2")


if __name__ == "__main__":
    unittest.main()
