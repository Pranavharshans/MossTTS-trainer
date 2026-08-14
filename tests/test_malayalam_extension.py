from __future__ import annotations

import unittest


try:
    import sentencepiece  # noqa: F401
except ImportError:
    sentencepiece = None

try:
    import torch  # noqa: F401
except ImportError:
    torch = None


@unittest.skipUnless(sentencepiece is not None, "sentencepiece is not installed")
class MalayalamTokenizerExtensionTest(unittest.TestCase):
    def test_append_preserves_all_base_piece_ids(self) -> None:
        from sentencepiece import sentencepiece_model_pb2 as model_pb2

        from finetuning.build_malayalam_tokenizer import append_pieces

        base = model_pb2.ModelProto()
        for text, piece_type in (
            ("<unk>", model_pb2.ModelProto.SentencePiece.UNKNOWN),
            ("a", model_pb2.ModelProto.SentencePiece.NORMAL),
            ("b", model_pb2.ModelProto.SentencePiece.NORMAL),
        ):
            piece = base.pieces.add()
            piece.piece = text
            piece.type = piece_type
        before = [piece.piece for piece in base.pieces]

        extended = append_pieces(base, ["മ", "മലയാളം"])

        self.assertEqual([piece.piece for piece in extended.pieces[:3]], before)
        self.assertEqual([piece.piece for piece in extended.pieces[3:]], ["മ", "മലയാളം"])
        self.assertEqual(extended.trainer_spec.vocab_size, 5)


@unittest.skipUnless(torch is not None, "torch is not installed")
class DatasetPackingSafetyTest(unittest.TestCase):
    def test_overlength_target_is_rejected_instead_of_truncated(self) -> None:
        from types import SimpleNamespace

        from finetuning.dataset import MossTTSNanoSFTDataset

        class Tokenizer:
            @staticmethod
            def encode(text: str, add_special_tokens: bool = False) -> list[int]:
                del add_special_tokens
                return [7] * len(text)

        config = SimpleNamespace(
            n_vq=2,
            audio_pad_token_id=0,
            pad_token_id=0,
            im_start_token_id=1,
            audio_start_token_id=2,
            audio_end_token_id=3,
            audio_user_slot_token_id=4,
            audio_assistant_slot_token_id=5,
        )
        dataset = MossTTSNanoSFTDataset(
            [{"text": "മലയാളം", "audio_codes": [[1, 2]] * 600}],
            tokenizer=Tokenizer(),
            model_config=config,
            max_length=512,
        )

        with self.assertRaisesRegex(ValueError, "silent target truncation is disabled"):
            _ = dataset[0]


if __name__ == "__main__":
    unittest.main()
