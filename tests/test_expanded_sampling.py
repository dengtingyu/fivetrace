import unittest
from collections import Counter

from ai_detector.expanded_data import sample_units


class ExpandedSamplingTests(unittest.TestCase):
    def test_sources_and_lengths_stay_matched_despite_generator_imbalance(self):
        rows = []
        for source, length_bin, human_count in [("papers", 0, 5), ("qa", 1, 9)]:
            for label, generators in [(0, ["human"]), (1, ["GPT", "Qwen", "DeepSeek"])]:
                for generator in generators:
                    for i in range(human_count if not label else 20):
                        rows.append({"id": f"{source}:{label}:{generator}:{i}", "split": "head_train",
                                     "source": source, "domain": "example", "length_bin": length_bin,
                                     "label": label, "generator": generator})
        rows.append({"id": "unpaired", "split": "head_train", "source": "only_AI", "domain": "example",
                     "length_bin": 0, "label": 1, "generator": "GPT"})
        chosen = sample_units(rows, "head_train", per_class=12)
        self.assertEqual(Counter(r["label"] for r in chosen), {0: 12, 1: 12})
        strata = [Counter((r["source"], r["domain"], r["length_bin"]) for r in chosen if r["label"] == label)
                  for label in [0, 1]]
        self.assertEqual(strata[0], strata[1])
        self.assertEqual(len({r["id"] for r in chosen}), len(chosen))
        self.assertGreater(len({r["generator"] for r in chosen if r["label"]}), 1)
        self.assertNotIn("only_AI", {r["source"] for r in chosen})


if __name__ == "__main__":
    unittest.main()
