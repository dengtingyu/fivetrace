import json
import unittest
from pathlib import Path
from unittest.mock import patch
import evaluate
from ai_detector.paragraphs import analyze_paragraphs
from ai_detector.config import STANDARD_ARTIFACT_DIR
from ai_detector.expanded_data import phase

class ReleaseProtocolTests(unittest.TestCase):
    def test_release_defaults_to_v3(self):
        self.assertEqual(STANDARD_ARTIFACT_DIR.name,"fivetrace_v3_mdeberta")
    def test_ai_only_metrics_do_not_invent_binary_performance(self):
        r=evaluate.metrics([dict(label=1,score=.2,paragraphs=1,AI_paragraphs=0),dict(label=1,score=.8,paragraphs=1,AI_paragraphs=1)])
        self.assertEqual(r["AI_recall"],.5)
        self.assertIsNone(r["human_FPR"]);self.assertIsNone(r["macro_F1"]);self.assertIsNone(r["AUROC"])
    def test_binary_boundary_includes_exact_half(self):
        rows=[dict(label=y,score=s,paragraphs=1,AI_paragraphs=int(s>=.5)) for y,s in [(0,.5),(0,.1),(1,.49),(1,.9)]]
        r=evaluate.metrics(rows)
        self.assertEqual(r["human_FPR"],.5);self.assertEqual(r["AI_recall"],.5)
    def test_document_score_is_character_weighted_not_paragraph_vote(self):
        report=analyze_paragraphs("甲"*110+"\n"+"乙"*330,lambda t:dict(ai_tendency_score=.9 if t[0]=="甲" else .1,text_classifier_score=0.,features={}),dict(human_below=.5,ai_at_or_above=.5),{},window_chars=600)
        self.assertEqual(report["ai_paragraph_ratio"],.5)
        self.assertAlmostEqual(evaluate.document_score(report),.3)
    def test_source_role_does_not_depend_on_generator_or_label(self):
        original="synthetic-source-family"
        roles={phase(original) for generator in ["human","GPT","Qwen"] for label in [0,1]}
        self.assertEqual(len(roles),1)
    def test_published_json_has_no_per_record_payloads(self):
        root=Path(__file__).resolve().parents[1]
        def check(value):
            if isinstance(value,dict):
                self.assertFalse(set(value)&{"sample_ids","text","question","response","instruction","embedding","windows","private_regression"})
                for item in value.values():check(item)
            elif isinstance(value,list):
                for item in value:check(item)
        for file in (root/"results").glob("*.json"):check(json.loads(file.read_text("utf-8")))
        external=json.loads((root/"results/external_v2.json").read_text("utf-8"))
        sources={r["dataset"] for r in external["tables"]["source_language_track"]}
        self.assertNotIn("local_generation_regression",sources)
        for row in external["mage_author_recipe"]["tables"]:
            self.assertIn(row["dataset"],sources)
            self.assertEqual(row["eligibility"],"strict_known_comparators")
if __name__=="__main__":unittest.main()
