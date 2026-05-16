import unittest

from vllm_itl_base.hf_proposer import _clone_cache


class HFProposerCacheTests(unittest.TestCase):
    def test_hf_cache_object_is_cloned_not_reused(self):
        class FakeCache:
            def __init__(self, values=None):
                self.values = list(values or [1])

            def get_seq_length(self):
                return len(self.values)

            def to_legacy_cache(self):
                return tuple(self.values)

        cache = FakeCache()
        cloned = _clone_cache(cache)

        cache.values.append(2)

        self.assertIsInstance(cloned, FakeCache)
        self.assertIsNot(cloned, cache)
        self.assertEqual(cloned.get_seq_length(), 1)

    def test_legacy_cache_fallback_preserves_cache_api(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class FakeCache:
            def __init__(self, legacy_cache=None):
                self.legacy_cache = legacy_cache
                self.source_tensor = torch.tensor([1])

            def __deepcopy__(self, memo):
                raise TypeError("force legacy fallback")

            def get_seq_length(self):
                return 1

            def to_legacy_cache(self):
                return ((self.source_tensor,),)

            @classmethod
            def from_legacy_cache(cls, legacy_cache):
                return cls(legacy_cache=legacy_cache)

        cache = FakeCache()
        cloned = _clone_cache(cache)

        self.assertIsInstance(cloned, FakeCache)
        self.assertEqual(cloned.get_seq_length(), 1)
        self.assertIsNot(cloned.legacy_cache[0][0], cache.source_tensor)
        self.assertEqual(int(cloned.legacy_cache[0][0][0].item()), 1)

    def test_nested_tensor_cache_is_cloned(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        tensor = torch.tensor([1])
        cache = ((tensor,),)
        cloned = _clone_cache(cache)

        tensor[0] = 2

        self.assertEqual(int(cloned[0][0][0].item()), 1)


if __name__ == "__main__":
    unittest.main()
