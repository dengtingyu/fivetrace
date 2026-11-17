"""Published aggregates retain failure conditions and disclose their scope."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class V3PublicationTests(unittest.TestCase):
    def test_external_chart_values_are_complete_binary_metrics_for_same_population(self):
        result = json.loads((ROOT/'results/v3_external.json').read_text(encoding='utf-8'))
        self.assertEqual(len(result['primary']), 8)
        for column in result['primary']:
            sizes = {(v['records'], v['human'], v['ai']) for v in column['models'].values()}
            self.assertEqual(len(sizes), 1)
            n, h, a = sizes.pop(); self.assertEqual(n, h+a)
            self.assertGreater(h, 0); self.assertGreater(a, 0)
            self.assertEqual(column['parent_exposure_matches'], 0)
            for point in column['models'].values():
                self.assertTrue(0 <= point['human_FPR'] <= 1)
                self.assertTrue(0 <= point['AI_recall'] <= 1)
                self.assertEqual(len(point['macro_F1_CI95']), 2)
        self.assertEqual([c['key'] for c in result['primary']], [c['key'] for c in result['original_regression']])

    def test_paired_differences_match_the_displayed_points(self):
        result = json.loads((ROOT/'results/v3_external.json').read_text(encoding='utf-8'))
        for column in result['primary']:
            for model, difference in column['differences'].items():
                self.assertAlmostEqual(difference['macro_F1'], column['models']['v3']['macro_F1']-column['models'][model]['macro_F1'])

    def test_release_files_match_the_reviewed_allowlist(self):
        manifest = json.loads((ROOT/'PUBLICATION_MANIFEST.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['version'], '3.0.0')
        for relative, expected in manifest['files'].items():
            self.assertEqual(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest(), expected, relative)


if __name__ == '__main__': unittest.main()
