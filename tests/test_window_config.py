import unittest

from ai_detector.paragraphs import analyze_paragraphs


class WindowConfigurationTests(unittest.TestCase):
    def test_matching_shorter_training_windows_keeps_one_paragraph_vote(self):
        text = "甲乙丙。"*225
        predict = lambda part: {"ai_tendency_score": .9, "text_classifier_score": .8, "features": {}}
        limits = {"ai_at_or_above": .5, "human_below": .5}
        old = analyze_paragraphs(text, predict, limits, {}, window_chars=1000)
        matched = analyze_paragraphs(text, predict, limits, {}, window_chars=600)
        self.assertEqual(len(old["paragraphs"][0]["windows"]), 1)
        self.assertEqual(len(matched["paragraphs"][0]["windows"]), 2)
        self.assertEqual(matched["total_paragraphs"], 1)
        self.assertEqual(matched["summary"]["ai"]["count"], 1)
        self.assertEqual(matched["ai_paragraph_ratio"], 1.)


if __name__ == "__main__":
    unittest.main()
