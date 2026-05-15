import json
import os
import unittest
from unittest.mock import patch

from vllm_itl_base.cli.serve import (
    _enable_vllm_plugin,
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


if __name__ == "__main__":
    unittest.main()
