import unittest
import tempfile
import pickle
from pathlib import Path
import torch
import numpy as np
from baseline_eval import aggregate_metrics, pooled_probabilities, sentence_token_chunks, load_numeric_checkpoint

class SimpleTokenizer:
    cls_token_id=1
    sep_token_id=2
    def num_special_tokens_to_add(self,pair=False):return 2
    def __call__(self,text,**kwargs):
        return dict(input_ids=list(range(10,10+len(text))),offset_mapping=[(i,i+1) for i in range(len(text))])

class RecentBaselineTests(unittest.TestCase):
    def test_edited_and_humanized_classes_both_count_as_ai(self):
        logits=torch.tensor([[0.,-10.,10.,-5.],[0.,-10.,-5.,10.]])
        self.assertTrue(bool((pooled_probabilities(logits,'tropa')>.99).all()))
    def test_author_subtraction_keeps_high_score_resolution(self):
        logits=torch.tensor([[0.,18.,-20.,-20.]])
        score=float(pooled_probabilities(logits,'tropa')[0])
        self.assertLess(score,1.0)
        self.assertGreater(score,.9999999)
    def test_chunking_preserves_every_token_once(self):
        text=('A sentence. Another sentence! '*110)+('z'*1000)
        chunks,total=sentence_token_chunks(SimpleTokenizer(),text)
        self.assertEqual([i for ids,_ in chunks for i in ids[1:-1]],list(range(10,10+len(text))))
        self.assertEqual(total,len(text))
        self.assertTrue(all(len(ids)<=768 for ids,_ in chunks))
    def test_ai_only_has_no_binary_metrics(self):
        result=aggregate_metrics([1,1],[.9,.3],[1,0])
        self.assertEqual(result['AI_recall'],.5)
        for name in ['human_FPR','macro_F1','AUROC']:self.assertIsNone(result[name])
    def test_numeric_metadata_is_allowed_and_unknown_pickle_globals_are_rejected(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'numeric.pt'
            torch.save(dict(weight=torch.ones(2),scale=np.array([1.,2.],dtype=np.float32)),path)
            result=load_numeric_checkpoint(path)
            self.assertEqual(result['scale'].tolist(),[1.,2.])
            del result
            other=Path(tmp)/'unsupported.pt'
            torch.save(dict(unexpected=SimpleNamespace(value=1)),other)
            with self.assertRaises(pickle.UnpicklingError):load_numeric_checkpoint(other)

if __name__=='__main__':unittest.main()
