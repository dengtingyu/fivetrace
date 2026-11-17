"""Download recovery must resume bytes and expose a failed coordinator."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import framework_comparison_v1 as acquisition
import framework_v3_deberta as v3


class Response:
    status=206
    def __init__(self,body,content_range):
        self.body=io.BytesIO(body);self.headers={'Content-Range':content_range}
    def __enter__(self):return self
    def __exit__(self,*args):self.body.close()
    def read(self,n):return self.body.read(n)


class DownloadRecoveryTests(unittest.TestCase):
    def test_partial_TLS_response_resumes_at_next_byte_and_changes_host(self):
        expected=b'1234567';calls=[]
        def fetch(request,timeout):
            calls.append(request)
            if len(calls)==1:return Response(expected[:3],'bytes 0-6/7')
            self.assertEqual(request.get_header('Range'),'bytes=3-6')
            return Response(expected[3:],'bytes 3-6/7')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);target=root/'checkpoint.bin'
            with patch.object(acquisition,'ROOT',root),patch('urllib.request.urlopen',side_effect=fetch),patch.object(acquisition.time,'sleep'),patch.dict(acquisition.os.environ,{'FRAMEWORK_WEIGHT_HOST':'https://huggingface.co'}):
                acquisition.download_weight(dict(repo='unit/base',revision='fixed',directory='unit'), 'checkpoint.bin',7,hashlib.sha256(expected).hexdigest(),target)
            self.assertEqual(target.read_bytes(),expected)
            self.assertTrue(calls[0].full_url.startswith('https://huggingface.co/'))
            self.assertTrue(calls[1].full_url.startswith('https://hf-mirror.com/'))

    def test_checksum_mismatch_never_publishes_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);target=root/'checkpoint.bin'
            with patch.object(acquisition,'ROOT',root),patch('urllib.request.urlopen',return_value=Response(b'wrong','bytes 0-4/5')):
                with self.assertRaisesRegex(AssertionError,'hash mismatch'):
                    acquisition.download_weight(dict(repo='unit/base',revision='fixed',directory='unit'),'checkpoint.bin',5,hashlib.sha256(b'right').hexdigest(),target)
            self.assertFalse(target.exists())


class CoordinatorRecoveryTests(unittest.TestCase):
    def test_v3_only_schedule_never_contains_baseline_training(self):
        steps=v3.pipeline_steps(v3_only=True)
        self.assertEqual(len(steps),7)
        self.assertEqual(steps[-1],('evaluate',None))
        self.assertTrue(all(name is None for stage,name in steps))
        self.assertFalse(any(stage.startswith('encoder-') for stage,name in steps))

    def test_current_scope_excludes_baselines_without_changing_frozen_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'run_scope.json').write_text(json.dumps(dict(frameworks=['fivetrace'],protocol='A')))
            frozen=dict(frameworks=['fivetrace','xlmr','desklib','conpara','mage'])
            with patch.object(v3.shared,'OUT',root),patch.object(v3.shared,'prepare',return_value=frozen):
                self.assertEqual(v3.shared.planned_frameworks(),['fivetrace'])
            self.assertEqual(len(frozen['frameworks']),5)

    def test_dead_adopted_worker_is_reported_instead_of_waiting_forever(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(v3,'OUT',Path(temporary)),patch.object(v3,'process_alive',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'exited before completing'):v3.wait_for_fit_worker(123)

    def test_adopted_pipeline_continues_after_features_without_restarting_GPU_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);out=root/'run';model=out/'A/v3';refs=out/'A/fivetrace';refs.mkdir(parents=True);(refs/'references_complete.json').write_text('{}')
            base=root/'models'/v3.SPEC['directory'];base.mkdir(parents=True);(base/'source.json').write_text('{}')
            (model/'head').mkdir(parents=True);(model/'head/fit_before_scoring.json').write_text('{}');(model/'head_scores_fit_complete.json').write_text('{}')
            with patch.object(v3,'OUT',out),patch.object(v3,'ROOT',root),patch.object(v3,'MODEL',model),patch.object(v3,'prepare',return_value={}),patch.object(v3.shared,'planned_frameworks',return_value=['fivetrace']),patch.object(v3,'wait_for_fit_worker',return_value=['head-train','head-fit-scores','qwen-fit-features']) as adopt,patch.object(v3.shared,'report'),patch.object(v3.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as child:
                v3.pipeline(v3_only=True,adopt_fit_worker=123)
                adopt.assert_called_once_with(123)
                self.assertEqual([call.args[0][-1] for call in child.call_args_list],['fusion-fit','head-test-scores','qwen-test-features','evaluate'])
                self.assertTrue(all(call.args[0][1]=='-u' for call in child.call_args_list))
            state=json.loads((out/'pipeline_state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['status'],'completed');self.assertEqual(state['planned_frameworks'],['fivetrace'])

    def test_acquisition_failure_returns_immediately_without_two_hour_wait(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);out=root/'run';refs=out/'A/fivetrace';refs.mkdir(parents=True);(refs/'references_complete.json').write_text('{}')
            with patch.object(v3,'OUT',out),patch.object(v3,'ROOT',root),patch.object(v3,'prepare',return_value={}),patch.object(v3.subprocess,'run',return_value=SimpleNamespace(returncode=1)) as child:
                with self.assertRaisesRegex(RuntimeError,'acquisition failed'):v3.pipeline()
                self.assertEqual(child.call_count,1)
                self.assertEqual(child.call_args.args[0][-1],'acquire')

    def test_failure_replaces_running_status_and_preserves_failed_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);out=root/'run';model=out/'A/v3';model.mkdir(parents=True)
            (out/'pipeline_state.json').write_text(json.dumps(dict(status='running',stage='acquiring_mdeberta_pretrained_weights',completed=['references'])))
            with patch.object(v3,'OUT',out),patch.object(v3,'MODEL',model):v3.record_failure('pipeline',RuntimeError('download failed'))
            value=json.loads((out/'pipeline_state.json').read_text(encoding='utf-8'))
            self.assertEqual(value['status'],'failed');self.assertEqual(value['stage'],'acquiring_mdeberta_pretrained_weights')
            self.assertEqual(value['completed'],['references']);self.assertIn('download failed',value['error'])
            self.assertTrue((out/value['failure_file']).exists())


if __name__=='__main__':unittest.main()
