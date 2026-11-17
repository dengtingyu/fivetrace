"""Local split import must reject leakage and remain usable by training IO."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import benchmark_v1_api as api
import prepare_v3_data as prep


def synthetic_splits():
    result = {}
    counter = 0
    for phase in ['train', 'development', 'calibration', 'test']:
        result[phase] = []
        for role in (['reference', 'classifier', 'fusion'] if phase == 'train' else [None]):
            for lang in ['zh', 'en']:
                counter += 1
                for label in [0, 1]:
                    r = dict(id=f'synthetic-{counter}-{label}', group_id=f'parent-{counter}', text=f'Synthetic unit {counter} class {label}. '+('Only generated test fixture content. '*8), label=label, language=lang, domain='news', generator='human' if label == 0 else 'synthetic-model', family='human' if label == 0 else 'synthetic-family')
                    if role: r['component_role'] = role
                    result[phase].append(r)
    return result


class LocalSplitTests(unittest.TestCase):
    def test_imported_splits_are_readable_by_the_training_and_test_API(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'prepared'
            with patch.object(prep, 'OUT', root), patch.object(api, 'OUT', root):
                prep.prepare(synthetic_splits())
                train = list(api.load_training(root/'protocols/A', 'reference'))
                self.assertEqual(len(train), 4)
                self.assertTrue(all(set(r) == {'id', 'text', 'label', 'group_id', 'language'} for r in train))
                self.assertEqual(len(list(api.load_test_inputs(root/'protocols/A'))), 4)
                with self.assertRaises(FileExistsError): prep.prepare(synthetic_splits())

    def test_cross_split_topic_reuse_is_rejected(self):
        splits = synthetic_splits()
        for r in splits['test'][:2]: r['group_id'] = splits['train'][0]['group_id']
        with self.assertRaisesRegex(ValueError, 'Source groups overlap'): prep.validate(splits)

    def test_renamed_duplicate_body_in_another_split_is_rejected(self):
        splits = synthetic_splits(); splits['test'][0]['text'] = splits['train'][0]['text'].upper()
        with self.assertRaisesRegex(ValueError, 'Canonical body overlaps'): prep.validate(splits)

    def test_component_roles_cannot_split_a_source_pair(self):
        splits = synthetic_splits(); splits['train'][0]['component_role'] = 'fusion'
        with self.assertRaisesRegex(ValueError, 'crosses component roles'): prep.validate(splits)

    def test_unpaired_or_missing_language_cannot_reach_training(self):
        splits = synthetic_splits(); splits['development'] = splits['development'][:2]
        with self.assertRaisesRegex(ValueError, 'both labels'): prep.validate(splits)
        splits = synthetic_splits(); splits['test'].pop()
        with self.assertRaisesRegex(ValueError, 'one human/AI pair'): prep.validate(splits)


if __name__ == '__main__': unittest.main()
