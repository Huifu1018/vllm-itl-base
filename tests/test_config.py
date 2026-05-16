import unittest
from unittest.mock import patch

from vllm_itl_base.config import BaseVLLMConfig


class ConfigTests(unittest.TestCase):
    def test_auto_method_selection(self):
        config = BaseVLLMConfig(method="auto")
        self.assertEqual(config.method_for_request(is_greedy=True), "slem")
        self.assertEqual(config.method_for_request(is_greedy=False), "tli")

    def test_env_validation(self):
        with patch.dict("os.environ", {"VLLM_ITL_BASE_METHOD": "bad"}):
            with self.assertRaises(ValueError):
                BaseVLLMConfig.from_env()

    def test_draft_tp_rank_from_env(self):
        with patch.dict("os.environ", {"VLLM_ITL_BASE_DRAFT_TP_RANK": "2"}):
            self.assertEqual(BaseVLLMConfig.from_env().draft_tp_rank, 2)

    def test_draft_tp_rank_rejects_negative_values(self):
        with patch.dict("os.environ", {"VLLM_ITL_BASE_DRAFT_TP_RANK": "-1"}):
            with self.assertRaises(ValueError):
                BaseVLLMConfig.from_env()


if __name__ == "__main__":
    unittest.main()
