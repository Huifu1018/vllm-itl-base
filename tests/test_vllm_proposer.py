import types
import unittest

from vllm_itl_base.vllm.proposer import VllmITLBaseProposer, _TPRuntime


class VllmProposerTests(unittest.TestCase):
    def test_payload_contains_only_current_request_state(self):
        proposer = VllmITLBaseProposer.__new__(VllmITLBaseProposer)
        proposer._draft_probs_by_req_id = {"req-a": ["row-a"], "old": ["row-old"]}
        proposer._target_ids_by_req_id = {"req-a": [10, 11], "old": [99]}
        proposer._last_methods_by_req_id = {"req-a": "tli", "old": "slem"}

        payload = proposer._export_payload(["req-a"], [[10, 11]])

        self.assertEqual(payload["draft_token_ids"], [[10, 11]])
        self.assertEqual(payload["draft_probs_by_req_id"], {"req-a": ["row-a"]})
        self.assertEqual(payload["target_ids_by_req_id"], {"req-a": [10, 11]})
        self.assertEqual(payload["last_methods_by_req_id"], {"req-a": "tli"})

    def test_apply_payload_replaces_rank_local_request_state(self):
        proposer = VllmITLBaseProposer.__new__(VllmITLBaseProposer)
        proposer._draft_probs_by_req_id = {"req-a": ["stale"]}
        proposer._target_ids_by_req_id = {"req-a": [1]}
        proposer._last_methods_by_req_id = {"req-a": "slem"}

        draft_ids = proposer._apply_payload(
            ["req-a"],
            {
                "draft_token_ids": [[20]],
                "draft_probs_by_req_id": {"req-a": ["row"]},
                "target_ids_by_req_id": {"req-a": [20]},
                "last_methods_by_req_id": {"req-a": "tli"},
            },
        )

        self.assertEqual(draft_ids, [[20]])
        self.assertEqual(proposer._draft_probs_by_req_id, {"req-a": ["row"]})
        self.assertEqual(proposer._target_ids_by_req_id, {"req-a": [20]})
        self.assertEqual(proposer._last_methods_by_req_id, {"req-a": "tli"})

    def test_broadcast_uses_configured_tp_source_rank(self):
        class FakeGroup:
            def broadcast_object(self, payload, src=0):
                return {"payload": payload, "src": src}

        proposer = VllmITLBaseProposer.__new__(VllmITLBaseProposer)
        proposer.tp_runtime = _TPRuntime(rank=1, world_size=2, group=FakeGroup())
        proposer.config = types.SimpleNamespace(draft_tp_rank=1)

        result = proposer._broadcast_payload({"draft_token_ids": [[1]]})

        self.assertEqual(result["src"], 1)
        self.assertEqual(result["payload"], {"draft_token_ids": [[1]]})


if __name__ == "__main__":
    unittest.main()
