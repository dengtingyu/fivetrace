"""V3 tests protect source isolation and the two independent tokenizer inputs."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit
import torch
from transformers import AutoModelForSequenceClassification,DebertaV2Config,BatchEncoding

import framework_v3_deberta as v3


class V3ProtocolTests(unittest.TestCase):
    def test_superseded_pipeline_cannot_launch_old_self_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'effective_model_map.json').write_text(json.dumps(dict(variant='FiveTrace-V3-mDeBERTa')))
            with patch.object(v3.original,'OUT',root),patch.object(v3.original,'prepare',side_effect=AssertionError('Must reject before preparing or launching GPU work')) as prepare:
                with self.assertRaisesRegex(RuntimeError,'superseded'):v3.original.pipeline(wait_references=True)
                prepare.assert_not_called()

    def test_training_group_overlap_is_rejected(self):
        rr=[dict(id=str(i),group_id='g'+str(i),component_role=name) for i,name in enumerate(['classifier','fusion','reference'])]
        self.assertEqual(len(v3.validate_roles(rr,{'heldout'})['classifier']),1)
        with self.assertRaisesRegex(ValueError,'evaluation|development'):v3.validate_roles(rr,{'g0'})
        rr[1]['group_id']='g0'
        with self.assertRaisesRegex(ValueError,'overlap'):v3.validate_roles(rr,set())

    def test_test_geometry_is_not_read_before_fusion_fit(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(v3,'prepare',return_value={}),patch.object(v3,'WORK',Path(temporary)),patch.object(v3,'MODEL',Path(temporary)/'model'),patch.object(v3.shared,'phase_rows',side_effect=AssertionError('Test body read before fit')) as reader:
            with self.assertRaises(AssertionError):v3.geometry('test')
            reader.assert_not_called()

    def test_head_cache_rejects_another_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'head').mkdir();p=root/'head/fit_before_scoring.json';p.write_text(json.dumps(dict(model_sha256='checkpoint-A')))
            with patch.object(v3,'WORK',root),patch.object(v3,'MODEL',root):
                db=v3.head_cache();db.close();p.write_text(json.dumps(dict(model_sha256='checkpoint-B')))
                with self.assertRaisesRegex(ValueError,'mix head checkpoints'):v3.head_cache()

    def test_raw_head_inputs_remain_distinct_after_Qwen_normalization(self):
        a='Some repeated body text. '*10;b=a+'\n'
        self.assertEqual(v3.shared.score_key(a),v3.shared.score_key(b))
        self.assertNotEqual(v3.head_key(a),v3.head_key(b))
        class Fusion:
            def predict_proba(self,x):
                p=expit(x[:,0]);return np.column_stack([1-p,p])
        records=[dict(id='a'),dict(id='b')]
        g={r['id']:[dict(characters=100,windows=[dict(key='same-Qwen-key',head_key=r['id'],characters=100)])] for r in records}
        result=v3.fused_scores(dict(fusion=Fusion()),records,g,{'same-Qwen-key':(None,np.array([.1,.2]))},{'a':-2.,'b':2.})
        self.assertAlmostEqual(result['a'],float(expit(-2.)));self.assertAlmostEqual(result['b'],float(expit(2.)))

    def test_head_only_aggregation_uses_probabilities_and_all_windows(self):
        g={'d':[dict(characters=400,windows=[dict(head_key='a',characters=100),dict(head_key='b',characters=300)])]}
        result=v3.score_head_documents([dict(id='d')],g,dict(a=-4.,b=2.))
        self.assertAlmostEqual(result['d'],float((expit(-4.)+3*expit(2.))/4))

    def test_tokenization_keeps_raw_body_and_declares_context_cap(self):
        calls=[]
        class Tokenizer:
            def __call__(self,texts,**kwargs):
                calls.append((texts,kwargs));return dict(input_ids=[[1,2]],attention_mask=[[1,1]])
        with patch.object(v3,'prepare',return_value=dict(first_feature=dict(max_tokens=512))):v3.encode(Tokenizer(),['## Heading\nOriginal  body'])
        self.assertEqual(calls[0][0],['## Heading\nOriginal  body'])
        self.assertEqual(calls[0][1],dict(padding=False,truncation=True,max_length=512))

    def test_native_tokenizer_uses_identical_relative_files_and_rejects_false(self):
        directory=v3.ROOT/'models/framework-base-mdeberta'
        fake=lambda text:text
        with patch('transformers.AutoTokenizer.from_pretrained',return_value=fake) as loader:
            self.assertIs(v3.shared.local_tokenizer(directory,use_fast=False),fake)
            address=loader.call_args.args[0]
            self.assertEqual(Path(address).resolve(),directory.resolve())
            self.assertFalse(Path(address).is_absolute())
            self.assertEqual(loader.call_args.kwargs,dict(local_files_only=True,use_fast=False))
        with patch('transformers.AutoTokenizer.from_pretrained',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'Tokenizer failed'):v3.shared.local_tokenizer(directory,use_fast=False)

    def test_fusion_receives_continuous_logits(self):
        class Head(torch.nn.Module):
            def forward(self,input_ids,attention_mask):
                x=input_ids[:,0].float();return SimpleNamespace(logits=torch.stack([-x,x],dim=-1))
        class Tokenizer:
            def pad(self,rr,**kwargs):return BatchEncoding(dict(input_ids=torch.tensor([r['input_ids'] for r in rr]),attention_mask=torch.tensor([r['attention_mask'] for r in rr])))
        values=v3.first_logits(Head(),Tokenizer(),[],encoded=[dict(input_ids=[1,2],attention_mask=[1,1]),dict(input_ids=[2,3],attention_mask=[1,1])],device='cpu')
        self.assertEqual(values,[2.,4.])

    def test_mdeberta_classifier_backpropagates_into_encoder_and_head(self):
        torch.set_num_threads(2)
        config=DebertaV2Config(vocab_size=40,hidden_size=16,num_hidden_layers=1,num_attention_heads=4,intermediate_size=32,max_position_embeddings=40,relative_attention=True,position_buckets=8,pos_att_type=['p2c','c2p'],num_labels=2)
        model=AutoModelForSequenceClassification.from_config(config)
        result=model(input_ids=torch.tensor([[1,3,4,5,2],[1,4,3,6,2]]),attention_mask=torch.ones(2,5,dtype=torch.long),labels=torch.tensor([0,1]))
        self.assertTrue(torch.isfinite(result.loss));result.loss.backward()
        self.assertTrue(any('classifier' in n and p.grad is not None and float(p.grad.abs().sum())>0 for n,p in model.named_parameters()))
        self.assertTrue(any('deberta' in n and p.grad is not None and float(p.grad.abs().sum())>0 for n,p in model.named_parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_comparison_selects_V3_without_relabeling_original_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);v3dir=root/'A/v3';old=root/'A/fivetrace';v3dir.mkdir(parents=True);old.mkdir()
            protocol=v3dir/'protocol_before_training.json';protocol.write_text('{}',encoding='utf-8')
            m=dict(records=4,macro_F1=.9,AI_recall=.8,human_FPR=.1,AUROC=.95)
            (v3dir/'metrics.json').write_text(json.dumps(dict(overall=m,strata=dict(language=dict(en=m,zh=m)))),encoding='utf-8')
            (old/'metrics.json').write_text(json.dumps(dict(overall=dict(m,macro_F1=.1))),encoding='utf-8')
            (root/'effective_model_map.json').write_text(json.dumps(dict(variant='FiveTrace-V3-mDeBERTa',protocol_sha256=v3.sha(protocol),models=dict(fivetrace='A/v3'))),encoding='utf-8')
            with patch.object(v3.shared,'OUT',root),patch.object(v3.shared,'prepare',return_value=dict(frameworks=['fivetrace','xlmr','desklib','conpara','mage'])),patch.object(v3.shared,'phase_rows',side_effect=AssertionError('No test labels needed for a partial status report')):
                result=v3.shared.report()
            self.assertEqual(result['fivetrace']['overall']['macro_F1'],.9)
            summary=json.loads((root/'comparison.json').read_text(encoding='utf-8'));self.assertFalse(summary['complete']);self.assertEqual(summary['self_variant'],'FiveTrace-V3-mDeBERTa')


if __name__=='__main__':unittest.main()
