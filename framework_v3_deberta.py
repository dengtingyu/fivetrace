"""V3 candidate: replace only feature 1 with a fresh multilingual DeBERTa."""
from collections import Counter
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import time

import joblib
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import framework_comparison_v1 as original
import framework_trainers_v1 as shared
from benchmark_v1_api import load_manifest,metric
from train_fivetrace import save,sha,now,rows,jsonl

ROOT=original.ROOT
OUT=original.OUT
WORK=original.WORK/'v3_mdeberta_fp32'
MODEL=OUT/'A/fivetrace_v3_mdeberta'
SEED=original.SEED
# Microsoft's pretrained multilingual checkpoint, not any released AI detector.
SPEC=dict(repo='microsoft/mdeberta-v3-base',revision='a0484667b22365f84929a935b5e50a51f71f159d',directory='framework-base-mdeberta')


def progress(stage,**kwargs):
    value=dict(at=now(),stage=stage,variant='FiveTrace-V3-mDeBERTa',**kwargs)
    save(MODEL/'progress.json',value);save(OUT/'progress.json',value);print(json.dumps(value),flush=True)


def head_key(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()


def validate_roles(data,evaluation_groups):
    roles={name:[r for r in data if r['component_role']==name] for name in ['classifier','fusion','reference']}
    groups={name:{r['group_id'] for r in rr} for name,rr in roles.items()}
    for name,parents in groups.items():
        if not parents:raise ValueError('Missing source groups for '+name)
        if parents&evaluation_groups:raise ValueError('Training overlaps development/calibration/test ancestry')
        for other in groups:
            if other!=name and parents&groups[other]:raise ValueError('Classifier, fusion and reference ancestry overlap')
    return roles


def prepare():
    original.prepare();WORK.mkdir(parents=True,exist_ok=True);MODEL.mkdir(parents=True,exist_ok=True)
    path=MODEL/'protocol_before_training.json'
    if path.exists():
        value=json.loads(path.read_text(encoding='utf-8'))
        assert value['parent_protocol_sha256']==sha(OUT/'protocol_before_training.json')
        assert value['dataset_verification_sha256']==sha(original.DATA/'verification.json')
        return value
    assert not (OUT/'A/fivetrace/fit_before_test.json').exists(),'Do not relabel an already tested original-head run'
    data=shared.phase_rows('train');evaluation_groups=set()
    for phase in ['development','calibration','test']:
        rr,_=load_manifest(original.DATA/'protocols/A',phase)
        evaluation_groups.update(r['group_id'] for r in rr)
    roles=validate_roles(data,evaluation_groups);classifier=roles['classifier'];fusion=roles['fusion'];reference=roles['reference']
    p=dict(at=now(),name='FiveTrace-V3-mDeBERTa',user_request='V3 replaces feature 1 with DeBERTa and tests it.',
        parent_protocol_sha256=sha(OUT/'protocol_before_training.json'),dataset_verification_sha256=sha(original.DATA/'verification.json'),
        base=SPEC,first_feature=dict(architecture='AutoModelForSequenceClassification: mDeBERTa-v3-base with a fresh two-class head',
            score_for_fusion='raw logit[AI]-logit[human], not rounded 0/1 decisions',train_role='classifier',train_documents=len(classifier),
            epochs=3,lr=2e-5,batch=2,accumulation=8,max_tokens=512,weight_decay=.01,warmup_fraction=.06,
            optimizer='Full-parameter FP32 AdamW; BF16 autocast; gradient checkpointing',
            selection='Development equal-language-domain document Macro-F1, then AUROC; no test selection',
            text_policy='One deterministic actual Qwen-geometry window per source document in classifier training; raw body passed to its own tokenizer, prefix capped at 512 tokens; all runtime windows scored for development and test.'),
        unchanged_features_2_to_5='Exactly the ongoing fresh Qwen Base/AI/human reference training, LM loss/rank and reliability-aware sentence/lexical statistics; share original reference weights and window cache.',
        reference_training_protocol_unchanged=True,reference_directory='A/fivetrace/adapters',fusion_train_role='fusion',fusion_documents=len(fusion),
        fusion_C=original.prepare()['fivetrace']['fusion_C'],fusion_retrained=True,
        group_isolation='Classifier, reference and fusion source components remain disjoint; do not reuse the all-TRAIN Desklib baseline as feature 1.',
        comparator_policy='Single primary self-developed framework is V3; the four predeclared end-to-end baselines retain their original same-data recipes.',
        ablation='Feature-1-only document scores from the same selected head; auxiliary diagnostic, not another primary framework.',
        calibration='Default 0.5 plus 1%/5% empirical human-CALIBRATION-FPR thresholds, separately for fusion and feature-1-only.',
        test_records=original.prepare()['phases']['test']['records'],legacy_v3_directories_overwritten=False,activate_as_default=False,data_uploaded=False)
    save(path,p)
    save(OUT/'effective_model_map.json',dict(at=now(),variant='FiveTrace-V3-mDeBERTa',protocol_sha256=sha(path),models=dict(fivetrace='A/fivetrace_v3_mdeberta'),
        original_self_head_plan='Superseded before supervised fitting and test; original references continue unchanged.'))
    save(OUT/'queue_switch_to_v3.json',dict(at=now(),stopped='Only the idle original pipeline coordinator',reference_training_interrupted=False,new_pipeline='framework_v3_deberta.py pipeline',test_scores_seen=False))
    progress('v3_protocol_frozen',classifier_documents=len(classifier),fusion_documents=len(fusion),reference_training_continues=True)
    return p


def acquire():
    import urllib.request
    prepare();directory=ROOT/'models'/SPEC['directory'];directory.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen('https://huggingface.co/api/models/'+SPEC['repo']+'/revision/'+SPEC['revision']+'?blobs=true',timeout=30) as response:metadata=json.load(response)
    assert metadata['sha']==SPEC['revision'];files={r['rfilename']:r for r in metadata['siblings']}
    for name in ['config.json','tokenizer_config.json','tokenizer.json','spm.model','special_tokens_map.json']:
        if name not in files:continue
        target=directory/name
        if not target.exists() or target.stat().st_size!=files[name]['size']:
            with urllib.request.urlopen('https://huggingface.co/'+SPEC['repo']+'/resolve/'+SPEC['revision']+'/'+name,timeout=60) as response:body=response.read()
            assert len(body)==files[name]['size'];target.write_bytes(body)
        if files[name].get('lfs'):assert sha(target)==files[name]['lfs']['sha256']
    name='model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin';info=files[name];target=directory/name
    if not target.exists() or sha(target)!=info['lfs']['sha256']:original.download_weight(SPEC,name,info['size'],info['lfs']['sha256'],target)
    assert sha(target)==info['lfs']['sha256']
    save(directory/'source.json',dict(**SPEC,task_detector_weights=False,files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in directory.iterdir() if p.is_file() and p.name!='source.json'}))
    progress('mdeberta_pretrained_checkpoint_verified',bytes=info['size'],revision=SPEC['revision'])


def geometry(phase):
    from transformers import AutoTokenizer
    prepare();file=WORK/(phase+'_geometry.json');manifest=WORK/(phase+'_geometry_provenance.json')
    if manifest.exists():
        p=json.loads(manifest.read_text(encoding='utf-8'));assert p['sha256']==sha(file)
        return json.loads(file.read_text(encoding='utf-8'))
    if phase=='test':assert (MODEL/'fit_before_test.json').exists()
    rr=shared.phase_rows(phase)
    if phase=='train':rr=[r for r in rr if r['component_role'] in ['classifier','fusion']]
    tokenizer=AutoTokenizer.from_pretrained(original.BASE,local_files_only=True);g=shared.structures(rr,tokenizer)
    if phase=='train':g={r['id']:[dict(characters=shared.representative(r,g[r['id']])['characters'],windows=[shared.representative(r,g[r['id']])])] for r in rr}
    for pp in g.values():
        for p in pp:
            for w in p['windows']:w['head_key']=head_key(w['text'])
    save(file,g);save(manifest,dict(at=now(),sha256=sha(file),document_records=len(g),phase=phase,tokenizer='Original Qwen runtime geometry; raw-body SHA256 separately identifies the DeBERTa input.'))
    return g


def load_head():
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    directory=ROOT/'models'/SPEC['directory'];source=json.loads((directory/'source.json').read_text(encoding='utf-8'))
    assert source['revision']==SPEC['revision'] and source['task_detector_weights'] is False
    # This is the original pretrained MLM, with a randomly initialized task head.
    model=AutoModelForSequenceClassification.from_pretrained(directory,num_labels=2,local_files_only=True)
    # Use the native SentencePiece tokenizer: no generated-protobuf conversion
    # or dependency upgrade in the environment shared by the running Qwen job.
    tokenizer=shared.local_tokenizer(directory,use_fast=False)
    return model,tokenizer


def encode(tokenizer,texts):
    data=tokenizer(texts,padding=False,truncation=True,max_length=prepare()['first_feature']['max_tokens'])
    return [dict(input_ids=ids,attention_mask=mask) for ids,mask in zip(data['input_ids'],data['attention_mask'])]


def first_logits(model,tokenizer,texts,encoded=None,batch_size=8,device='cuda'):
    import torch
    if encoded is None:encoded=encode(tokenizer,texts)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model.eval();values=[]
    with torch.inference_mode():
        for offset in range(0,len(encoded),batch_size):
            batch=tokenizer.pad(encoded[offset:offset+batch_size],padding=True,return_tensors='pt').to(device);batch.pop('token_type_ids',None)
            logits=model(**batch).logits.float();values.extend((logits[:,1]-logits[:,0]).cpu().double().tolist())
    if not np.isfinite(values).all():raise RuntimeError('Nonfinite DeBERTa logits')
    return values


def score_head_documents(records,g,lookup):
    probabilities={k:float(expit(v)) for k,v in lookup.items()}
    remapped={r['id']:[dict(characters=p['characters'],windows=[dict(key=w['head_key'],characters=w['characters']) for w in p['windows']]) for p in g[r['id']]] for r in records}
    return shared.aggregate_scores(records,remapped,probabilities)


def train_head():
    import torch
    from transformers import get_linear_schedule_with_warmup
    p=prepare();recipe=p['first_feature'];directory=MODEL/'head';directory.mkdir(parents=True,exist_ok=True)
    if (directory/'fit_before_scoring.json').exists():return
    train=[r for r in shared.phase_rows('train') if r['component_role']=='classifier'];development=shared.phase_rows('development');tg=geometry('train');dg=geometry('development')
    assert len(train)==recipe['train_documents']
    dev_windows={w['head_key']:w['text'] for r in development for ps in dg[r['id']] for w in ps['windows']};dev_items=list(dev_windows.items())
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);np.random.seed(SEED);random.seed(SEED)
    model,tokenizer=load_head();model=model.float().cuda();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    train_text=[tg[r['id']][0]['windows'][0]['text'] for r in train];tokens=encode(tokenizer,train_text);dev_tokens=encode(tokenizer,[t for k,t in dev_items]);parameters=[p for p in model.parameters() if p.requires_grad]
    assert all(p.requires_grad for p in model.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=recipe['lr'],weight_decay=recipe['weight_decay'],foreach=False)
    batches=math.ceil(len(train)/recipe['batch']);steps_epoch=math.ceil(batches/recipe['accumulation']);steps=steps_epoch*recipe['epochs'];scheduler=get_linear_schedule_with_warmup(optimizer,int(steps*recipe['warmup_fraction']),steps)
    checkpoint=directory/'resume.pt';current=dict(epoch=0,next_batch=0,step=0,best=[-1.,-1.],history=[],elapsed=0.)
    if checkpoint.exists():
        cp=torch.load(checkpoint,map_location='cpu',weights_only=True);assert cp['protocol_sha256']==sha(MODEL/'protocol_before_training.json')
        model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler']);current=cp['state'];torch.set_rng_state(cp['rng']);torch.cuda.set_rng_state_all(cp['cuda_rng']);del cp
    started=time.monotonic();prior=current['elapsed']
    def save_checkpoint():
        current['elapsed']=prior+time.monotonic()-started;temp=checkpoint.with_suffix('.tmp')
        torch.save(dict(protocol_sha256=sha(MODEL/'protocol_before_training.json'),model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),state=current,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temp);temp.replace(checkpoint)
    save(directory/'sample_manifest.json',dict(classifier_ids=[r['id'] for r in train],classifier_groups=sorted({r['group_id'] for r in train}),data_sha256=original.prepare()['phases']['train']['sha256']))
    progress('mdeberta_first_feature_training',completed=current['step'],total=steps,training_documents=len(train),trainable_parameters=sum(p.numel() for p in parameters))
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(current['epoch'],recipe['epochs']):
        order=list(range(len(train)));random.Random(SEED+epoch).shuffle(order);begin=current['next_batch'] if epoch==current['epoch'] else 0;model.train();running=0;count=0
        for index in range(begin,batches):
            indices=order[index*recipe['batch']:(index+1)*recipe['batch']];batch=tokenizer.pad([tokens[i] for i in indices],padding=True,return_tensors='pt').to('cuda');batch.pop('token_type_ids',None)
            labels=torch.tensor([train[i]['label'] for i in indices],device='cuda');divisor=min(recipe['accumulation'],batches-index//recipe['accumulation']*recipe['accumulation'])
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch,labels=labels).loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite DeBERTa training loss')
            (loss/divisor).backward();running+=float(loss.detach());count+=1
            if (index+1)%recipe['accumulation'] and index+1<batches:continue
            torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);current['step']+=1;current.update(epoch=epoch,next_batch=index+1)
            if current['step']%25==0 or index+1==batches:
                if current['step']%250==0 or index+1==batches:save_checkpoint()
                progress('mdeberta_first_feature_training',completed=current['step'],total=steps,epoch=epoch+1,loss=running/max(1,count),elapsed_seconds=prior+time.monotonic()-started)
                running=count=0
        values=first_logits(model,tokenizer,[t for k,t in dev_items],encoded=dev_tokens);lookup={k:value for (k,text),value in zip(dev_items,values)}
        scores=score_head_documents(development,dg,lookup);m=shared.development_metrics(development,scores);rank=[m['equal_cell_macro_F1'],m['AUROC']]
        current['history'].append(dict(epoch=epoch+1,development=m))
        if rank>current['best']:
            current.update(best=rank,best_epoch=epoch+1);temp=directory/'best_model.tmp';torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},temp);temp.replace(directory/'best_model.pt')
        current.update(epoch=epoch+1,next_batch=0);save_checkpoint();progress('mdeberta_head_epoch_complete',epoch=epoch+1,development=m)
    save(directory/'fit_before_scoring.json',dict(at=now(),variant='V3',model_sha256=sha(directory/'best_model.pt'),base_source_sha256=sha(ROOT/'models'/SPEC['directory']/'source.json'),selected_epoch=current['best_epoch'],history=current['history'],elapsed_seconds=current['elapsed'],trainable_parameters=sum(p.numel() for p in parameters),train_documents=len(train),all_encoder_parameters_trained=True,test_labels_used=False,protocol_sha256=sha(MODEL/'protocol_before_training.json')))
    del model,optimizer,parameters,batch,labels,loss;gc.collect();torch.cuda.empty_cache()


def head_cache():
    db=sqlite3.connect(WORK/'head_logits.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('CREATE TABLE IF NOT EXISTS logits(key TEXT PRIMARY KEY,value REAL NOT NULL)');db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)')
    frozen=json.loads((MODEL/'head/fit_before_scoring.json').read_text(encoding='utf-8'));old=db.execute("SELECT value FROM metadata WHERE key='model_sha256'").fetchone()
    if old and old[0]!=frozen['model_sha256']:
        db.close();raise ValueError('Do not mix head checkpoints in one cache')
    elif not old:db.execute('INSERT INTO metadata VALUES (?,?)',('model_sha256',frozen['model_sha256']));db.commit()
    return db


def cache_head_scores(phase='fit'):
    import torch
    prepare();frozen=json.loads((MODEL/'head/fit_before_scoring.json').read_text(encoding='utf-8'));assert sha(MODEL/'head/best_model.pt')==frozen['model_sha256']
    if phase=='test':assert (MODEL/'fit_before_test.json').exists()
    names=['test'] if phase=='test' else ['train','development','calibration'];wanted={}
    for name in names:
        g=geometry(name);rr=shared.phase_rows(name)
        if name=='train':rr=[r for r in rr if r['component_role']=='fusion']
        for r in rr:
            for ps in g[r['id']]:
                for w in ps['windows']:wanted[w['head_key']]=w['text']
    db=head_cache();done={r[0] for r in db.execute('SELECT key FROM logits')};pending=[(k,t) for k,t in wanted.items() if k not in done]
    if pending:
        model,tokenizer=load_head();model=model.float().cuda();model.load_state_dict(torch.load(MODEL/'head/best_model.pt',map_location='cpu',weights_only=True));pending.sort(key=lambda kt:len(kt[1]));started=time.monotonic()
        try:
            for offset in range(0,len(pending),64):
                part=pending[offset:offset+64];values=first_logits(model,tokenizer,[t for k,t in part])
                db.executemany('INSERT INTO logits VALUES (?,?)',[(k,float(v)) for (k,text),v in zip(part,values)]);db.commit()
                if offset%256==0 or offset+64>=len(pending):progress('mdeberta_window_scoring',phase=phase,completed=min(offset+64,len(pending)),total=len(pending),cached=len(wanted)-len(pending),elapsed_seconds=time.monotonic()-started)
        finally:del model;gc.collect();torch.cuda.empty_cache()
    db.close();save(MODEL/f'head_scores_{phase}_complete.json',dict(at=now(),windows=len(wanted),model_sha256=frozen['model_sha256']))


def heads():
    db=head_cache();result=dict(db.execute('SELECT key,value FROM logits'));db.close();return result


def fused_scores(bundle,records,g,numeric,logits):
    # The raw-body head key and normalized-Qwen key are different identities.
    # Preserve both, even when two raw windows normalize to one Qwen input.
    pairs={(w['key'],w['head_key']) for r in records for ps in g[r['id']] for w in ps['windows']};pairs=sorted(pairs);values={}
    for offset in range(0,len(pairs),512):
        part=pairs[offset:offset+512];x=np.stack([np.concatenate([[logits[hk]],numeric[qk][1]]) for qk,hk in part]);scores=bundle['fusion'].predict_proba(x)[:,1]
        values.update({key:float(value) for key,value in zip(part,scores)})
    remapped={r['id']:[dict(characters=ps['characters'],windows=[dict(key=(w['key'],w['head_key']),characters=w['characters']) for w in ps['windows']]) for ps in g[r['id']]] for r in records}
    return shared.aggregate_scores(records,remapped,values)


def fit():
    prepare()
    if (MODEL/'fit_before_test.json').exists():return
    assert (OUT/'A/fivetrace/features_fit_complete.json').exists() and (MODEL/'head_scores_fit_complete.json').exists()
    data=[r for r in shared.phase_rows('train') if r['component_role']=='fusion'];development=shared.phase_rows('development');calibration=shared.phase_rows('calibration');g={name:geometry(name) for name in ['train','development','calibration']}
    numeric=shared.five_arrays();logits=heads();windows=[g['train'][r['id']][0]['windows'][0] for r in data];x=np.stack([np.concatenate([[logits[w['head_key']]],numeric[w['key']][1]]) for w in windows]);y=np.array([r['label'] for r in data])
    best=None;history=[]
    for C in prepare()['fusion_C']:
        fusion=make_pipeline(StandardScaler(),LogisticRegression(C=C,class_weight='balanced',max_iter=2000,random_state=SEED)).fit(x,y);bundle=dict(fusion=fusion)
        scores=fused_scores(bundle,development,g['development'],numeric,logits);m=shared.development_metrics(development,scores);rank=[m['equal_cell_macro_F1'],m['AUROC']]
        history.append(dict(C=C,development=m))
        if best is None or rank>best[0]:best=(rank,bundle,C,m)
    bundle=best[1];bundle.update(variant='FiveTrace-V3-mDeBERTa',first_feature=SPEC,head_checkpoint=(MODEL/'head/best_model.pt').relative_to(ROOT).as_posix(),head_checkpoint_sha256=sha(MODEL/'head/best_model.pt'),Qwen_reference_directory='artifacts/framework_comparison_v1_20261115/A/fivetrace/adapters',extractor_metadata=json.loads((OUT/'A/fivetrace/extractor_metadata.json').read_text(encoding='utf-8')),threshold=.5,fresh_task_training=True)
    fused_cal=fused_scores(bundle,calibration,g['calibration'],numeric,logits);first_cal=score_head_documents(calibration,g['calibration'],logits);cuts=shared.calibration_thresholds(calibration,fused_cal);first_cuts=shared.calibration_thresholds(calibration,first_cal)
    joblib.dump(bundle,MODEL/'detector.joblib');refs=json.loads((OUT/'A/fivetrace/references_complete.json').read_text(encoding='utf-8'))
    save(MODEL/'fit_before_test.json',dict(at=now(),variant='FiveTrace-V3-mDeBERTa',model_sha256=sha(MODEL/'detector.joblib'),head_sha256=bundle['head_checkpoint_sha256'],reference_sha256=refs['branches'],selected_fusion_C=best[2],development=best[3],fusion_candidates=history,calibration=cuts,first_feature_only_calibration=first_cuts,protocol_sha256=sha(MODEL/'protocol_before_training.json'),test_scores_seen=False))
    progress('v3_fusion_fitted_before_test',development=best[3],selected_C=best[2])


def evaluate():
    prepare();frozen=json.loads((MODEL/'fit_before_test.json').read_text(encoding='utf-8'));assert sha(MODEL/'detector.joblib')==frozen['model_sha256'];assert sha(MODEL/'head/best_model.pt')==frozen['head_sha256']
    assert (OUT/'A/fivetrace/features_test_complete.json').exists() and (MODEL/'head_scores_test_complete.json').exists()
    test=shared.phase_rows('test');g=geometry('test');numeric=shared.five_arrays();logits=heads();bundle=joblib.load(MODEL/'detector.joblib');scores=fused_scores(bundle,test,g,numeric,logits)
    shared.write_results('fivetrace',test,scores,frozen,out_override=MODEL)
    first=score_head_documents(test,g,logits);file=MODEL/'first_feature_only_predictions.jsonl';jsonl(file,[dict(id=r['id'],score=first[r['id']]) for r in test])
    from benchmark_v1_api import evaluate as evaluate_protocol
    report=evaluate_protocol(original.DATA/'protocols/A',file,bootstrap=200);report.update(ablation='V3 feature 1 only; same selected head and 6,704 classifier training documents',head_sha256=frozen['head_sha256'],calibrated_operating_points={k:dict(calibration=v,test=metric(test,first,v['threshold'])) for k,v in frozen['first_feature_only_calibration'].items()})
    save(MODEL/'first_feature_only_metrics.json',report);shared.report();progress('v3_testing_complete',records=len(test))


def pipeline_steps(v3_only=False):
    steps=[('head-train',None),('head-fit-scores',None),('qwen-fit-features',None),('fusion-fit',None),('head-test-scores',None),('qwen-test-features',None),('evaluate',None)]
    if not v3_only:
        for name in original.ENCODERS:steps.extend([('encoder-train',name),('encoder-evaluate',name)])
    return steps


def process_alive(pid):
    if os.name!='nt':
        try:os.kill(pid,0);return True
        except ProcessLookupError:return False
    import ctypes
    from ctypes import wintypes
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];api.OpenProcess.restype=wintypes.HANDLE
    api.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];api.GetExitCodeProcess.restype=wintypes.BOOL
    api.CloseHandle.argtypes=[wintypes.HANDLE];api.CloseHandle.restype=wintypes.BOOL
    handle=api.OpenProcess(0x1000,False,pid)
    if not handle:
        code=ctypes.get_last_error()
        if code in [87,1168]:return False
        raise OSError(code,'Unable to inspect adopted feature worker')
    try:
        code=wintypes.DWORD()
        if not api.GetExitCodeProcess(handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
        return code.value==259
    finally:api.CloseHandle(handle)


def wait_for_fit_worker(pid):
    completed=OUT/'A/fivetrace/features_fit_complete.json'
    while not completed.exists():
        if not process_alive(pid):raise RuntimeError('Adopted feature worker exited before completing; restart from its feature cache')
        save(OUT/'pipeline_state.json',dict(at=now(),status='running',stage='qwen-fit-features',adopted_worker_pid=pid,completed=['head-train','head-fit-scores'],self_variant='FiveTrace-V3-mDeBERTa',scope='V3-only protocol A'))
        time.sleep(10)
    manifest=json.loads(completed.read_text(encoding='utf-8'))
    for phase,digest in manifest['geometries'].items():
        if sha(original.WORK/(phase+'_five_geometry.json'))!=digest:raise RuntimeError('Completed feature geometry changed')
    return ['head-train','head-fit-scores','qwen-fit-features']


def pipeline(v3_only=False,adopt_fit_worker=None):
    prepare();started=time.monotonic()
    while not (OUT/'A/fivetrace/references_complete.json').exists():
        if time.monotonic()-started>6*3600:raise TimeoutError('Reference trainer did not finish; inspect reference_training.log')
        save(OUT/'pipeline_state.json',dict(at=now(),status='running',stage='waiting_for_existing_reference_training',self_variant='FiveTrace-V3-mDeBERTa'));time.sleep(10)
    if not (ROOT/'models'/SPEC['directory']/'source.json').exists():
        # Own and check the acquisition child, rather than waiting for a file
        # for two hours after an independent downloader has already failed.
        save(OUT/'pipeline_state.json',dict(at=now(),status='running',stage='acquiring_mdeberta_pretrained_weights',self_variant='FiveTrace-V3-mDeBERTa'))
        with (OUT/'v3_backbone_acquire_owned.log').open('a',encoding='utf-8') as log:
            child=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'acquire'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if child.returncode:raise RuntimeError('mDeBERTa acquisition failed; see v3_backbone_acquire_owned.log')
        if not (ROOT/'models'/SPEC['directory']/'source.json').exists():raise RuntimeError('Acquisition returned without a verified checkpoint')
    selected=shared.planned_frameworks()
    if selected==['fivetrace']:v3_only=True
    steps=pipeline_steps(v3_only=v3_only)
    completed=wait_for_fit_worker(adopt_fit_worker) if adopt_fit_worker is not None else []
    if completed:
        if not (MODEL/'head/fit_before_scoring.json').exists() or not (MODEL/'head_scores_fit_complete.json').exists():raise RuntimeError('Adopted worker lacks completed first-feature stages')
    for stage,name in steps:
        if stage in completed and name is None:continue
        if name:
            started=time.monotonic()
            while not (ROOT/'models'/original.ENCODERS[name]['directory']/'source.json').exists():
                if time.monotonic()-started>2*3600:raise TimeoutError('Baseline backbone unavailable: '+name)
                save(OUT/'pipeline_state.json',dict(at=now(),status='running',stage='waiting_for_backbone',model=name,self_variant='FiveTrace-V3-mDeBERTa'));time.sleep(10)
        key=stage+('_'+name if name else '');command=[sys.executable,'-u',str(Path(__file__).resolve()),stage]
        if name:command=[sys.executable,'-u',str(ROOT/'framework_comparison_v1.py'),stage,'--model',name]
        save(OUT/'pipeline_state.json',dict(at=now(),status='running',stage=key,completed=completed,self_variant='FiveTrace-V3-mDeBERTa'))
        with (OUT/('v3_'+key+'.log')).open('a',encoding='utf-8') as log:child=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if child.returncode:
            save(OUT/'pipeline_state.json',dict(at=now(),status='failed',stage=key,returncode=child.returncode,completed=completed,self_variant='FiveTrace-V3-mDeBERTa'));raise RuntimeError('V3 stage failed: '+key)
        completed.append(key);shared.report()
    save(OUT/'pipeline_state.json',dict(at=now(),status='completed',completed=completed,self_variant='FiveTrace-V3-mDeBERTa',protocol='A',planned_frameworks=selected,B_not_scheduled=True,data_uploaded=False))


def record_failure(stage,error):
    failure=dict(at=now(),stage=stage,error=repr(error))
    save(MODEL/(stage+'_failure.json'),failure)
    if stage=='pipeline':
        previous_path=OUT/'pipeline_state.json'
        previous=json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else {}
        save(previous_path,dict(previous,at=now(),status='failed',error=repr(error),failure_file=(MODEL/(stage+'_failure.json')).relative_to(OUT).as_posix()))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','acquire','head-train','head-fit-scores','head-test-scores','qwen-fit-features','qwen-test-features','fusion-fit','evaluate','pipeline']);parser.add_argument('--v3-only',action='store_true');parser.add_argument('--adopt-fit-worker',type=int);args=parser.parse_args()
    try:
        if args.stage=='prepare':prepare()
        elif args.stage=='acquire':acquire()
        elif args.stage=='head-train':train_head()
        elif args.stage=='head-fit-scores':cache_head_scores('fit')
        elif args.stage=='head-test-scores':cache_head_scores('test')
        elif args.stage=='qwen-fit-features':shared.five_features('fit',train_roles=['fusion'])
        elif args.stage=='qwen-test-features':shared.five_features('test',fit_dir=MODEL)
        elif args.stage=='fusion-fit':fit()
        elif args.stage=='evaluate':evaluate()
        elif args.stage=='pipeline':pipeline(v3_only=args.v3_only,adopt_fit_worker=args.adopt_fit_worker)
    except Exception as error:
        record_failure(args.stage,error);raise
