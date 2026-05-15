import unittest

from vllm_itl_base.core import (
    VocabIntersection,
    longest_diagonal_match,
    new_tokens_after_reencoded_window,
    slem_target_proxies_from_assistant_window,
)


class FakeTokenizer:
    def __init__(self, vocab, id_to_text=None, space_marker=None):
        self._vocab = dict(vocab)
        self._id_to_token = {idx: token for token, idx in self._vocab.items()}
        self._id_to_text = id_to_text or {
            idx: token.replace("Ġ", " ").replace("▁", " ")
            for token, idx in self._vocab.items()
        }
        self._space_marker = space_marker

    def get_vocab(self):
        return dict(self._vocab)

    def __len__(self):
        return max(self._vocab.values()) + 1

    def convert_ids_to_tokens(self, ids):
        return [self._id_to_token[int(idx)] for idx in ids]

    def decode(self, ids, **kwargs):
        return "".join(self._id_to_text[int(idx)] for idx in ids)

    def __call__(self, text, add_special_tokens=False, **kwargs):
        if text == " " and self._space_marker is not None:
            return {"input_ids": [self._vocab[self._space_marker]]}
        ids = []
        cursor = 0
        pieces = sorted(
            ((token.replace("Ġ", " ").replace("▁", " "), idx) for token, idx in self._vocab.items()),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        while cursor < len(text):
            for piece, idx in pieces:
                if text.startswith(piece, cursor):
                    ids.append(idx)
                    cursor += len(piece)
                    break
            else:
                cursor += 1
        return {"input_ids": ids}


class CoreTests(unittest.TestCase):
    def test_vocab_intersection_normalizes_space_marker(self):
        target = FakeTokenizer({"Ġhello": 10, "world": 11, "Ġ": 12}, space_marker="Ġ")
        assistant = FakeTokenizer({"▁hello": 3, "other": 4, "▁": 5}, space_marker="▁")

        mapping = VocabIntersection.from_tokenizers(
            target_tokenizer=target,
            assistant_tokenizer=assistant,
            target_vocab_size=20,
        )

        self.assertEqual(mapping.assistant_to_target[3], 10)
        self.assertEqual(mapping.target_to_assistant[10], 3)
        self.assertIn(4, mapping.suppress_assistant_ids())

    def test_longest_diagonal_match(self):
        match = longest_diagonal_match([9, 1, 2, 3], [7, 1, 2, 8])
        self.assertIsNotNone(match)
        self.assertEqual((match.source_start, match.target_start, match.length), (1, 1, 2))

    def test_new_tokens_after_reencoded_window(self):
        new_tokens = new_tokens_after_reencoded_window(
            target_suffix=[1, 2, 3],
            reencoded_window=[9, 2, 3, 4, 5],
        )
        self.assertEqual(new_tokens, (4, 5))

    def test_slem_extracts_new_target_tokens(self):
        target = FakeTokenizer({"hello": 1, " world": 2, "!": 3})
        assistant = FakeTokenizer({"hello": 10, " world": 11, "!": 12})
        proxies = slem_target_proxies_from_assistant_window(
            target_tokenizer=target,
            assistant_tokenizer=assistant,
            current_target_ids=[1],
            assistant_context_ids=[10],
            assistant_new_ids=[11, 12],
            assistant_lookbehind=2,
            target_lookbehind=2,
        )
        self.assertEqual(proxies, (2, 3))


if __name__ == "__main__":
    unittest.main()
