"""Runtime regression tests use synthetic inputs and no downloaded models."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ai_detector.v3_inference import V3Detector, raw_head_key
from ai_detector.language_windows import language_document_structure
from ai_detector.paragraphs import detection_window_spans, prepare_detection_paragraphs


class OffsetTokenizer:
    def __call__(self, text, **kwargs):
        return dict(offset_mapping=[(i, i+1) for i in range(len(text))])


class V3RuntimeTests(unittest.TestCase):
    def test_raw_classifier_identity_does_not_merge_whitespace_variants(self):
        self.assertNotEqual(raw_head_key('synthetic body'), raw_head_key('synthetic body\n'))

    def test_runtime_and_training_use_identical_window_bodies(self):
        from ai_detector.language_windows import language_window_spans
        tokenizer = OffsetTokenizer()
        text = ('A synthetic sentence for a reproducible window. '*40)+'\n'+('Another synthetic paragraph. '*30)
        geometry = language_document_structure(text, tokenizer)
        bodies = [w['text'] for p in geometry for w in p['windows']]
        _, _, pp = prepare_detection_paragraphs(text)
        actual = [p['text'][a:b] for p in pp for a, b in detection_window_spans(p['text'], window_chars=600, window_splitter=lambda t: language_window_spans(t, tokenizer))]
        self.assertEqual(bodies, actual)

    def test_fusion_receives_continuous_raw_head_logit_and_numeric_features(self):
        received = []
        class Fusion:
            def predict_proba(self, x):
                received.append(x.copy()); p = 1/(1+np.exp(-x[:, 0])); return np.column_stack([1-p, p])
        detector = V3Detector.__new__(V3Detector); detector.bundle = dict(fusion=Fusion())
        detector.head_logits = lambda texts: [-2.25, 1.75]
        detector.reference = lambda: SimpleNamespace(extract_many=lambda texts, batch_size: [(None, dict(numeric=.4)), (None, dict(numeric=.7))])
        with patch('ai_detector.v3_inference.repair_numeric_vector', side_effect=lambda stats: np.array([stats['numeric']])):
            scores = detector.score_windows(['synthetic a', 'synthetic b'])
        np.testing.assert_array_equal(received[0], [[-2.25, .4], [1.75, .7]])
        self.assertLess(scores[0]['ai_tendency_score'], .5)
        self.assertGreater(scores[1]['ai_tendency_score'], .5)


if __name__ == '__main__':
    unittest.main()
