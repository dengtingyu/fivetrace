import unittest

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from ai_detector.features import FeatureExtractor, LM_NAMES, STYLE_NAMES


class TinyTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [2+ord(c) % 41 for c in text]


class BatchedFeatureTests(unittest.TestCase):
    def test_padding_and_overlapping_contexts_preserve_teacher_forced_features(self):
        torch.manual_seed(123)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=43, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                    head_dim=8, max_position_embeddings=2048, attention_dropout=0.)).eval()
        extractor = FeatureExtractor.__new__(FeatureExtractor)
        extractor.model, extractor.tokenizer = model, TinyTokenizer()
        extractor.torch, extractor.device = torch, "cpu"
        extractor.cache, extractor.fingerprint = None, "test"
        texts = ["甲乙丙丁", "教师说明实验条件与结果。"*19, "模型预测需要完整且连续的前文。"*67]
        batch = extractor.extract_many(texts, batch_size=3)
        for text, (embedding, stats) in zip(texts, batch):
            individual, single_stats = extractor.extract(text)
            self.assertEqual(stats["scored_token_count"], len(text)-1)
            np.testing.assert_allclose(embedding, individual, atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose([stats[n] for n in LM_NAMES+STYLE_NAMES],
                                       [single_stats[n] for n in LM_NAMES+STYLE_NAMES], atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
