import json
import unittest

from ai_detector.paragraphs import analyze_paragraphs, prepare_detection_paragraphs, report_to_markdown, split_paragraphs, summarize_paragraphs
from ai_detector.training import binary_thresholds
import numpy as np

LIMITS = {"human_below": .5, "ai_at_or_above": .5}


def predict(text):
    score = .9 if text.lstrip().startswith("甲") else .1
    return {"ai_tendency_score": score, "text_classifier_score": score, "features": {}}


class ParagraphTests(unittest.TestCase):
    def test_rates_count_paragraphs_not_characters_or_mean_scores(self):
        result = analyze_paragraphs("甲"*100 + "\n\n" + "乙"*900, predict, LIMITS, {})
        self.assertEqual(result["summary"]["ai"]["count"], 1)
        self.assertEqual(result["summary"]["human"]["count"], 1)
        self.assertEqual(result["ai_paragraph_ratio"], .5)
        self.assertEqual(result["human_paragraph_ratio"], .5)
        self.assertNotIn("document_score", result)
        self.assertEqual(result["feature_groups_used"], [1, 2, 3, 4, 5])

    def test_empty_lines_and_offsets(self):
        raw = "\r\n　" + "甲"*120 + "　\r\n\r\n\t" + "乙"*150 + "\r\n"
        text, parts = split_paragraphs(raw)
        self.assertEqual(len(parts), 2)
        for part in parts:
            self.assertEqual(text[part["start"]:part["end"]], part["text"])
        self.assertEqual([p["characters"] for p in parts], [120, 150])

    def test_short_paragraph_merge_keeps_all_sources(self):
        raw = "\n".join(["标题", "甲"*50, "乙"*60, "丙"*200, "尾注"*5])
        text, source, parts = prepare_detection_paragraphs(raw)
        self.assertEqual(len(parts), 2)
        self.assertEqual([p["source_paragraphs"] for p in parts], [[1, 2, 3], [4, 5]])
        self.assertEqual(sum(p["characters"] for p in parts), sum(p["characters"] for p in source))
        self.assertTrue(all(p["characters"] >= 100 for p in parts))
        for part in parts:
            self.assertEqual(text[part["start"]:part["end"]], part["text"])

    def test_empty_or_too_short_article_is_not_fabricated_as_human(self):
        for text in ["\n\t ", "甲"*20 + "\n" + "乙"*30]:
            with self.assertRaises(ValueError):
                analyze_paragraphs(text, predict, LIMITS, {})

    def test_long_paragraph_is_counted_once(self):
        result = analyze_paragraphs("甲"*2100, predict, LIMITS, {})
        part = result["paragraphs"][0]
        self.assertGreater(len(part["windows"]), 1)
        self.assertEqual("".join(w["text"] for w in part["windows"]), part["text"])
        self.assertEqual(result["total_paragraphs"], 1)
        self.assertEqual(result["summary"]["ai"]["percentage"], 100.)

    def test_long_mixed_evidence_still_returns_binary_paragraph(self):
        result = analyze_paragraphs("甲"*1000 + "乙"*1000, predict, LIMITS, {})
        self.assertEqual(result["paragraphs"][0]["ai_tendency_score"], .5)
        self.assertEqual(result["paragraphs"][0]["category"], "ai")
        self.assertEqual(result["ai_paragraph_ratio"], 1.)
        self.assertNotIn("不确定", json.dumps(result, ensure_ascii=False))

    def test_sparse_long_windows_are_merged_before_scoring(self):
        seen = []
        def checked_predict(text):
            self.assertGreaterEqual(sum(not c.isspace() for c in text), 100)
            seen.append(text)
            return predict(text)
        raw = "甲"*100 + " "*3000 + "乙"*100
        result = analyze_paragraphs(raw, checked_predict, LIMITS, {})
        self.assertEqual("".join(seen), raw)
        self.assertEqual(result["total_paragraphs"], 1)

    def test_model_failure_is_reported_without_a_fake_label(self):
        def failed(text):
            raise ValueError("invalid feature")
        with self.assertRaisesRegex(RuntimeError, "第 1 段"):
            analyze_paragraphs("甲"*100, failed, LIMITS, {})

    def test_percentages_sum_to_100_after_rounding(self):
        result = summarize_paragraphs([{"category": "ai"}, {"category": "human"}, {"category": "human"}])
        self.assertAlmostEqual(result["ai"]["percentage"]+result["human"]["percentage"], 100.)
        self.assertEqual(result["ai"]["percentage"], 33.3)
        self.assertEqual(result["human"]["percentage"], 66.7)

    def test_markdown_contains_counts_and_binary_results(self):
        result = analyze_paragraphs("甲"*100 + "\n" + "乙"*100, predict, LIMITS, {})
        markdown = report_to_markdown(result)
        self.assertIn("| AI 段落 | 1 | 50.0% |", markdown)
        self.assertIn("| 人工段落 | 1 | 50.0% |", markdown)
        self.assertNotIn("不确定", markdown)

    def test_binary_threshold_has_no_reject_band(self):
        y = np.array([0]*100 + [1]*100)
        scores = np.concatenate([np.linspace(0, .8, 100), np.linspace(.2, 1, 100)])
        limits = binary_thresholds(y, scores)
        self.assertEqual(limits["human_below"], limits["ai_at_or_above"])
        self.assertLessEqual(np.mean(scores[y == 0] >= limits["ai_at_or_above"]), .05)


if __name__ == "__main__":
    unittest.main()
