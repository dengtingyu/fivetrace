"""Check fair calibration, body processing and actual differentiable structures."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import AutoModel,AutoModelForSequenceClassification,XLMRobertaConfig,LongformerConfig,DebertaV2Config

import framework_trainers_v1 as tr


class ProtocolTests(unittest.TestCase):
    def test_calibration_cannot_depend_on_AI_scores(self):
        records=[dict(id=str(i),label=0) for i in range(100)]+[dict(id='AI',label=1)]
        first={str(i):i/100 for i in range(100)};first['AI']=.999
        second=dict(first,AI=.001)
        a=tr.calibration_thresholds(records,first);b=tr.calibration_thresholds(records,second)
        self.assertEqual(a,b)
        self.assertLessEqual(a['0.01']['empirical_human_FPR'],.01)
        self.assertLessEqual(a['0.05']['empirical_human_FPR'],.05)

    def test_calibration_with_ties_remains_conservative(self):
        records=[dict(id=str(i),label=0) for i in range(100)]
        cuts=tr.calibration_thresholds(records,{str(i):.5 for i in range(100)})
        self.assertEqual(cuts['0.01']['empirical_human_FPR'],0.)

    def test_human_and_AI_share_relative_window_selection(self):
        windows=[dict(key=str(i),characters=100,text='x'*100) for i in range(12)]
        first=tr.representative(dict(id='human',group_id='same-parent'),[dict(windows=windows)])
        second=tr.representative(dict(id='AI',group_id='same-parent'),[dict(windows=windows)])
        self.assertEqual(first,second)

    def test_aggregation_counts_all_windows_with_character_weights(self):
        geometry={'doc':[dict(characters=400,windows=[dict(key='a',characters=100),dict(key='b',characters=300)]),dict(characters=100,windows=[dict(key='c',characters=100)])]}
        scores=tr.aggregate_scores([dict(id='doc')],geometry,dict(a=1.,b=0.,c=1.))
        self.assertAlmostEqual(scores['doc'],.4)

    def test_conpara_normalizes_only_inference(self):
        captured=[]
        class Tok:
            def __call__(self,texts,**kwargs):
                captured.append(texts);return dict(input_ids=[[1,2] for _ in texts],attention_mask=[[1,1] for _ in texts])
        records=[dict(id='1',text='Ａ\u200bＢ   text')]
        with patch.object(tr,'normalizer',return_value=lambda text:'AB text'):
            tr.encoder_tokens('conpara',Tok(),records,False);tr.encoder_tokens('conpara',Tok(),records,True)
        self.assertEqual(captured[0][0],records[0]['text'])
        self.assertEqual(captured[1][0],'AB text')


class ArchitectureTests(unittest.TestCase):
    def test_all_four_structures_have_real_encoder_and_head_gradients(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in tr.ENCODERS:
                directory=root/'models'/tr.ENCODERS[name]['directory'];directory.mkdir(parents=True,exist_ok=True);(directory/'source.json').write_text(json.dumps(dict(task_detector_weights=False)))
            configs=dict(xlmr=XLMRobertaConfig(vocab_size=40,hidden_size=16,num_hidden_layers=1,num_attention_heads=4,intermediate_size=32,max_position_embeddings=40,num_labels=2),
                mage=LongformerConfig(vocab_size=40,hidden_size=16,num_hidden_layers=1,num_attention_heads=4,intermediate_size=32,max_position_embeddings=40,attention_window=4,num_labels=2),
                deberta=DebertaV2Config(vocab_size=40,hidden_size=16,num_hidden_layers=1,num_attention_heads=4,intermediate_size=32,max_position_embeddings=40,relative_attention=True,position_buckets=8,pos_att_type=['p2c','c2p']))
            for name in tr.ENCODERS:
                config=configs.get(name,configs['deberta'])
                def encoder(*args,**kwargs):return AutoModel.from_config(config)
                def classifier(*args,**kwargs):return AutoModelForSequenceClassification.from_config(config)
                with self.subTest(framework=name),patch.object(tr,'ROOT',root),patch('transformers.AutoModel.from_pretrained',side_effect=encoder),patch('transformers.AutoModelForSequenceClassification.from_pretrained',side_effect=classifier):
                    model=tr.new_encoder(name)
                    output=model(input_ids=torch.tensor([[0,5,6,7,8,9,2,1],[0,7,6,5,9,8,2,1]]),attention_mask=torch.tensor([[1,1,1,1,1,1,1,0]]*2),labels=torch.tensor([0,1]))
                    logits=tr.output_logits(output);self.assertEqual(tuple(logits.shape),(2,1 if name=='desklib' else 2))
                    loss=output['loss'] if isinstance(output,dict) else output.loss;self.assertTrue(torch.isfinite(loss));loss.backward()
                    gradients=[(key,value.grad) for key,value in model.named_parameters() if value.grad is not None]
                    self.assertTrue(all(torch.isfinite(g).all() for key,g in gradients))
                    self.assertTrue(any('classifier' in key and float(g.abs().sum())>0 for key,g in gradients))
                    self.assertTrue(any('classifier' not in key and float(g.abs().sum())>0 for key,g in gradients))


if __name__=='__main__':unittest.main()
