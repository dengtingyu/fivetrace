"""Contract tests for leakage prevention, readers and evaluation failures."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

import benchmark_v1_api as api
from benchmark_v1_policy import generator,canonical
from export_benchmark_v1 import balanced_pairs,select_pairs


class MetadataTests(unittest.TestCase):
    def test_generator_vendor_boundaries(self):
        for model in ['gpt-j-6b','gpt-neox-20b']:
            self.assertEqual(generator(model)[1],'eleutherai')
        self.assertEqual(generator('OpenAI_gpt4o')[:2],('gpt-4o','openai'))
        self.assertEqual(generator('Claude-3.5-Haiku')[1],'anthropic')
        self.assertEqual(generator('LLaDA-8B')[1],'llada')

    def test_unknown_size_is_not_inferred_as_llama(self):
        for value in ['7B','13B','30B','unknown']:
            self.assertEqual(generator(value)[1],'unknown')

    def test_canonical_aliases_share_identity(self):
        self.assertEqual(canonical('ＡＢＣ_，中文 123'),canonical('abc 中文１２３'))

    def test_sampling_never_reuses_parent_with_many_variants(self):
        candidates={i:[dict(root=i,language=lang,domain='news',length_bin='long',family=f,generator=f) for f in ['qwen','openai']] for i,lang in [(1,'zh'),(2,'zh'),(3,'en'),(4,'en')]}
        result=balanced_pairs(candidates,100,'adversarial')
        self.assertEqual(len(result),4)
        self.assertEqual(len({r['root'] for r in result}),4)
        self.assertEqual(Counter(r['language'] for r in result),{'zh':2,'en':2})
        self.assertNotIn(1,{r['root'] for r in select_pairs(candidates,100,'exclude',[1])})


class ReaderAndMetricTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.patch=patch.object(api,'OUT',self.root);self.patch.start()
        self.folder=self.root/'protocols/A';self.folder.mkdir(parents=True)
        (self.root/'data').mkdir()
        self.train=[];self.test=[]
        for number,(phase,role,lang) in enumerate([('train','reference','zh'),('train','classifier','en'),('train','fusion','zh'),('test',None,'zh'),('test',None,'en')]):
            for label in [0,1]:
                r=dict(id=f'doc-{number}-{label}',group_id=f'g-{number}',label=label,language=lang,family='human' if label==0 else 'openai',generator='human' if label==0 else 'gpt-4o',domain='news',length_bin='long')
                if role:r['component_role']=role
                (self.train if phase=='train' else self.test).append(r)
        all_rows=[dict(id=r['id'],text='Only the body '+r['id']) for r in self.train+self.test]
        pq.write_table(pa.Table.from_pylist(all_rows),self.root/'data/master.parquet')
        splits={}
        for phase,rr in [('train',self.train),('development',[]),('calibration',[]),('test',self.test)]:
            file=self.folder/f'{phase}.jsonl';self.write(file,rr)
            splits[phase]=dict(file=file.relative_to(self.root).as_posix(),records=len(rr),sha256=api.sha(file))
        self.protocol=dict(name='A-covered-generator',splits=splits,master=dict(sha256=api.sha(self.root/'data/master.parquet')))
        self.save_protocol()
        (self.root/'protocols/index.json').write_text(json.dumps(dict(protocols=['A'])),encoding='utf-8')
        (self.folder/'pairs.json').write_text(json.dumps(dict(test=[dict(human_id=self.test[i]['id'],AI_id=self.test[i+1]['id']) for i in [0,2]])),encoding='utf-8')
        self.pred=self.root/'predictions.jsonl'

    def tearDown(self):
        self.patch.stop();self.tmp.cleanup()

    def write(self,path,rr):
        path.write_text(''.join(json.dumps(r)+'\n' for r in rr),encoding='utf-8')

    def save_protocol(self):
        (self.folder/'protocol.json').write_text(json.dumps(self.protocol),encoding='utf-8')

    def predictions(self,values):
        self.write(self.pred,[dict(id=r['id'],score=s) for r,s in zip(self.test,values)])

    def test_component_reader_only_returns_its_own_bodies(self):
        rr=list(api.load_training(self.folder,'reference'))
        self.assertEqual({r['id'] for r in rr},{r['id'] for r in self.train if r['component_role']=='reference'})
        self.assertTrue(all(r['text']=='Only the body '+r['id'] for r in rr))
        self.assertTrue(all('generator' not in r and 'family' not in r for r in rr))

    def test_test_inputs_have_no_labels_or_provenance(self):
        rr=list(api.load_test_inputs(self.folder));self.assertEqual(len(rr),4)
        self.assertTrue(all(set(r)=={'id','text'} for r in rr))

    def test_modified_manifest_is_rejected(self):
        with (self.folder/'train.jsonl').open('a',encoding='utf-8') as stream:stream.write('\n')
        with self.assertRaisesRegex(ValueError,'changed'):list(api.load_training(self.folder))

    def test_modified_master_is_rejected(self):
        pq.write_table(pa.Table.from_pylist([dict(id='fake',text='fake')]),self.root/'data/master.parquet')
        with self.assertRaisesRegex(ValueError,'master changed'):list(api.load_training(self.folder))

    def test_held_family_is_rejected_before_any_training_body(self):
        self.protocol.update(name='B-family',excluded_family='openai');self.save_protocol()
        with self.assertRaisesRegex(ValueError,'Held-out family'):next(api.load_training(self.folder))

    def test_missing_prediction_cannot_improve_reported_coverage(self):
        self.predictions([.1,.9,.2])
        with self.assertRaisesRegex(ValueError,'missing=1'):api.evaluate(self.folder,self.pred,bootstrap=0)

    def test_duplicate_prediction_is_rejected(self):
        self.predictions([.1,.9,.2,.8])
        with self.pred.open('a') as stream:stream.write(json.dumps(dict(id=self.test[0]['id'],score=.2))+'\n')
        with self.assertRaisesRegex(ValueError,'Duplicate'):api.evaluate(self.folder,self.pred,bootstrap=0)

    def test_nonfinite_scores_and_boolean_are_rejected(self):
        for value in [float('nan'),float('inf'),-.1,1.1,True,None]:
            self.predictions([value,.9,.2,.8])
            with self.assertRaisesRegex(ValueError,'finite score'):api.evaluate(self.folder,self.pred,bootstrap=0)

    def test_confusion_matrix_counts_human_false_positives(self):
        self.predictions([.8,.9,.1,.2]);r=api.evaluate(self.folder,self.pred,bootstrap=0)['overall']
        self.assertEqual(r['confusion_matrix'],[[1,1],[1,1]])
        self.assertEqual(r['AI_recall'],.5);self.assertEqual(r['human_FPR'],.5)
        self.assertEqual(r['macro_F1'],.5);self.assertEqual(r['AUROC'],.75)

    def test_group_bootstrap_preserves_matched_pairs(self):
        self.predictions([.1,.9,.2,.8]);original=api.metric;seen=[]
        def validate(rr,scores,threshold):
            if len(rr)==len(self.test):
                groups=Counter((r['group_id'],r['label']) for r in rr)
                for g in {r['group_id'] for r in rr}:self.assertEqual(groups[g,0],groups[g,1])
                seen.append(1)
            return original(rr,scores,threshold)
        with patch.object(api,'metric',side_effect=validate):r=api.evaluate(self.folder,self.pred,bootstrap=30)
        self.assertGreaterEqual(len(seen),31)
        self.assertEqual(r['source_group_bootstrap']['overall_macro_F1_95CI'],[1.,1.])

    def test_inactive_protocol_is_rejected(self):
        (self.root/'protocols/index.json').write_text(json.dumps(dict(protocols=[])),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'active version'):api.load_manifest(self.folder,'train')

    def test_invalid_threshold_cannot_produce_results(self):
        self.predictions([.1,.9,.2,.8])
        for threshold in [float('nan'),float('inf'),-1,2,True]:
            with self.assertRaisesRegex(ValueError,'Threshold'):api.evaluate(self.folder,self.pred,threshold,0)


if __name__=='__main__':unittest.main()
