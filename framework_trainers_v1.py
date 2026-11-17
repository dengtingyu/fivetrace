"""Training/evaluation stages for fresh framework adaptations on frozen A."""
from collections import defaultdict,Counter
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import time

import joblib
import numpy as np
from scipy.special import expit
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from framework_comparison_v1 import ROOT,DATA,WORK,OUT,BASE,SEED,ENCODERS,prepare,state
from benchmark_v1_api import evaluate,metric,load_manifest
from train_fivetrace import save,sha,rows,jsonl,now


def phase_rows(phase):
    p=prepare();assert sha(WORK/p['phases'][phase]['file'])==p['phases'][phase]['sha256']
    return rows(WORK/p['phases'][phase]['file'])


def local_tokenizer(directory,use_fast=True):
    from transformers import AutoTokenizer
    # SentencePiece's Windows loader may fail on an absolute non-ASCII path.
    # The pipeline runs from ROOT; a relative address names identical files.
    try:address=os.path.relpath(directory,Path.cwd())
    except ValueError:address=str(directory)
    tokenizer=AutoTokenizer.from_pretrained(address,local_files_only=True,use_fast=use_fast)
    if not callable(tokenizer):raise RuntimeError('Tokenizer failed to load its verified SentencePiece vocabulary: '+address)
    return tokenizer


def development_metrics(records,scores):
    values=metric(records,scores,.5);cells=defaultdict(list)
    for r in records:cells[r['language'],r['domain']].append(r)
    values['equal_cell_macro_F1']=float(np.mean([metric(rr,scores,.5)['macro_F1'] for rr in cells.values()]))
    return values


def calibration_thresholds(records,scores):
    # Use human CALIBRATION only. Report realized TEST FPR at these fixed cuts.
    human=np.array([scores[r['id']] for r in records if r['label']==0]);assert len(human)
    result={}
    for alpha in [.01,.05]:
        k=min(len(human)-1,int(math.floor(len(human)*alpha)))
        threshold=float(np.nextafter(np.sort(human)[::-1][k],np.inf))
        result[str(alpha)]=dict(threshold=threshold,empirical_human_FPR=float(np.mean(human>=threshold)),human_records=len(human))
    return result


def score_key(text):
    from ai_detector.data import normalize
    from ai_detector.expanded_data import clean_text
    return hashlib.sha256(normalize(clean_text(text)).encode('utf-8')).hexdigest()


def feature_db():
    path=WORK/'fivetrace_windows.sqlite';db=sqlite3.connect(path);db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS features(key TEXT PRIMARY KEY,embedding BLOB,numeric BLOB)')
    return db


def structures(records,tokenizer):
    from ai_detector.language_windows import language_document_structure
    return {r['id']:language_document_structure(r['text'],tokenizer) for r in records}


def representative(record,geometry):
    windows=[w for p in geometry for w in p['windows']]
    fraction=int(hashlib.sha256((str(SEED)+'training_window'+record['group_id']).encode()).hexdigest()[:12],16)/16**12
    return windows[min(len(windows)-1,int(fraction*len(windows)))]


def five_features(phase='fit',train_roles=None,fit_dir=None):
    import torch
    from transformers import AutoTokenizer
    from ai_detector.repair_features import RepairFeatureExtractor,repair_numeric_vector
    assert (OUT/'A/fivetrace/references_complete.json').exists()
    if phase=='test':assert ((Path(fit_dir) if fit_dir is not None else OUT/'A/fivetrace')/'fit_before_test.json').exists()
    phases=['test'] if phase=='test' else ['train','development','calibration']
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    needed={};locations={}
    for name in phases:
        rr=phase_rows(name)
        if name=='train':rr=[r for r in rr if r['component_role'] in (train_roles if train_roles is not None else ['classifier','fusion'])]
        geom=structures(rr,tokenizer)
        if name=='train':geom={r['id']:[dict(characters=representative(r,geom[r['id']])['characters'],windows=[representative(r,geom[r['id']])])] for r in rr}
        # Keep full metadata/source labels out of model input; local geometry is
        # only body text, identity and aggregation weights.
        for ps in geom.values():
            for p in ps:
                for w in p['windows']:needed[w['key']]=w['text']
        locations[name]=geom
        save(WORK/f'{name}_five_geometry.json',geom)
    db=feature_db();done={r[0] for r in db.execute('SELECT key FROM features')}
    pending=[(k,t) for k,t in needed.items() if k not in done]
    if pending:
        ex=RepairFeatureExtractor(BASE,device='cuda',adapter_dir=OUT/'A/fivetrace/adapters')
        metadata=OUT/'A/fivetrace/extractor_metadata.json'
        if metadata.exists():assert json.loads(metadata.read_text(encoding='utf-8'))==ex.metadata
        else:save(metadata,ex.metadata)
        pending.sort(key=lambda kt:len(tokenizer.encode(kt[1],add_special_tokens=False)));started=time.monotonic()
        try:
            for offset in range(0,len(pending),32):
                part=pending[offset:offset+32];features=ex.extract_many([t for k,t in part],batch_size=16)
                for (key,text),(embedding,stats) in zip(part,features):
                    db.execute('INSERT OR REPLACE INTO features VALUES (?,?,?)',(key,embedding.astype(np.float32).tobytes(),repair_numeric_vector(stats).astype(np.float64).tobytes()))
                db.commit()
                if offset%128==0 or offset+32>=len(pending):state('fivetrace_feature_extraction',phase=phase,completed=min(offset+32,len(pending)),total=len(pending),cached=len(needed)-len(pending),elapsed_seconds=time.monotonic()-started)
        finally:ex.close()
        del ex;gc.collect();torch.cuda.empty_cache()
    db.close();save(OUT/f'A/fivetrace/features_{phase}_complete.json',dict(at=now(),unique_windows=len(needed),train_component_roles=train_roles if train_roles is not None else ['classifier','fusion'],geometries={name:sha(WORK/f'{name}_five_geometry.json') for name in phases}))


def five_arrays():
    db=feature_db();values={k:(np.frombuffer(e,dtype=np.float32).copy(),np.frombuffer(v,dtype=np.float64).copy()) for k,e,v in db.execute('SELECT key,embedding,numeric FROM features')};db.close();return values


def aggregate_scores(records,geometry,window_scores):
    result={}
    for r in records:
        ps=geometry[r['id']]
        scores=[float(np.average([window_scores[w['key']] for w in p['windows']],weights=[w['characters'] for w in p['windows']])) for p in ps]
        result[r['id']]=float(np.average(scores,weights=[p['characters'] for p in ps]))
    return result


def five_predict(bundle,records,geometry,values):
    keys=list({w['key'] for r in records for p in geometry[r['id']] for w in p['windows']});scores={}
    for offset in range(0,len(keys),512):
        part=keys[offset:offset+512];embedding=np.stack([values[k][0] for k in part]);numeric=np.stack([values[k][1] for k in part]);logits=bundle['head'].decision_function(embedding)
        p=bundle['fusion'].predict_proba(np.column_stack([logits,numeric]))[:,1]
        scores.update({k:float(v) for k,v in zip(part,p)})
    return aggregate_scores(records,geometry,scores)


def five_fit():
    out=OUT/'A/fivetrace'
    if (out/'fit_before_test.json').exists():return
    assert (out/'features_fit_complete.json').exists()
    values=five_arrays();data=phase_rows('train');development=phase_rows('development');calibration=phase_rows('calibration')
    geom={name:json.loads((WORK/f'{name}_five_geometry.json').read_text(encoding='utf-8')) for name in ['train','development','calibration']}
    def array(role):
        rr=[r for r in data if r['component_role']==role];keys=[geom['train'][r['id']][0]['windows'][0]['key'] for r in rr]
        return np.stack([values[k][0] for k in keys]),np.stack([values[k][1] for k in keys]),np.array([r['label'] for r in rr])
    he,hv,hy=array('classifier');fe,fv,fy=array('fusion');best=None;history=[]
    for hc in prepare()['fivetrace']['head_C']:
        head=make_pipeline(StandardScaler(),LogisticRegression(C=hc,class_weight='balanced',max_iter=2000,random_state=SEED)).fit(he,hy)
        xx=np.column_stack([head.decision_function(fe),fv])
        for fc in prepare()['fivetrace']['fusion_C']:
            fusion=make_pipeline(StandardScaler(),LogisticRegression(C=fc,class_weight='balanced',max_iter=2000,random_state=SEED)).fit(xx,fy)
            bundle=dict(head=head,fusion=fusion);scores=five_predict(bundle,development,geom['development'],values);m=development_metrics(development,scores)
            history.append(dict(head_C=hc,fusion_C=fc,development=m));rank=(m['equal_cell_macro_F1'],m['AUROC'])
            if best is None or rank>best[0]:best=(rank,bundle,hc,fc,m)
        state('fivetrace_supervised_fit',head_C=hc,completed_candidates=len(history),total_candidates=9)
    bundle=best[1];bundle.update(extractor_metadata=json.loads((out/'extractor_metadata.json').read_text(encoding='utf-8')),threshold=.5,framework='five-feature-proxy-LM',geometry='language-aware-600chars-384tokens',fresh_task_training=True)
    calibration_scores=five_predict(bundle,calibration,geom['calibration'],values);cuts=calibration_thresholds(calibration,calibration_scores)
    joblib.dump(bundle,out/'detector.joblib')
    save(out/'fit_before_test.json',dict(at=now(),model_sha256=sha(out/'detector.joblib'),head_C=best[2],fusion_C=best[3],development=best[4],candidates=history,calibration=cuts,protocol_sha256=sha(OUT/'protocol_before_training.json'),test_scores_seen=False))
    state('fivetrace_fitted_before_test',development=best[4])


def five_evaluate():
    out=OUT/'A/fivetrace';frozen=json.loads((out/'fit_before_test.json').read_text(encoding='utf-8'));assert sha(out/'detector.joblib')==frozen['model_sha256']
    assert (out/'features_test_complete.json').exists();test=phase_rows('test');geometry=json.loads((WORK/'test_five_geometry.json').read_text(encoding='utf-8'))
    scores=five_predict(joblib.load(out/'detector.joblib'),test,geometry,five_arrays());write_results('fivetrace',test,scores,frozen)


def normalizer():
    path=ROOT/'models/ConPara/unicode_preprocessing_v2.py';spec=importlib.util.spec_from_file_location('comparison_conpara_normalization',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Author normalise also collapses whitespace in src/conpara.py.
    import re
    return lambda text:re.sub(r'\s+',' ',module.unicode_normalize(str(text))).strip()


def new_encoder(name):
    import torch
    from torch import nn
    from transformers import AutoModel,AutoModelForSequenceClassification
    spec=ENCODERS[name];base=ROOT/'models'/spec['directory'];assert json.loads((base/'source.json').read_text(encoding='utf-8'))['task_detector_weights'] is False
    if name in ['xlmr','mage']:
        model=AutoModelForSequenceClassification.from_pretrained(base,local_files_only=True,num_labels=2,ignore_mismatched_sizes=True)
        return model
    class Pooled(nn.Module):
        def __init__(self):
            super().__init__();self.encoder=AutoModel.from_pretrained(base,local_files_only=True);hidden=self.encoder.config.hidden_size
            self.classifier=nn.Linear(hidden,1) if name=='desklib' else nn.Sequential(nn.Linear(hidden,hidden//2),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden//2,2))
        def forward(self,input_ids,attention_mask,labels=None,**kwargs):
            hidden=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
            if name=='desklib':
                mask=attention_mask.unsqueeze(-1).float();pooled=(hidden.float()*mask).sum(1)/mask.sum(1).clamp_min(1.)
                logits=self.classifier(pooled);loss=nn.functional.binary_cross_entropy_with_logits(logits[:,0],labels.float()) if labels is not None else None
            else:
                logits=self.classifier(hidden[:,0]);loss=nn.functional.cross_entropy(logits,labels,weight=torch.tensor([1.5,1.],device=logits.device)) if labels is not None else None
            return dict(logits=logits,loss=loss)
    return Pooled()


def output_logits(output):return output['logits'] if isinstance(output,dict) else output.logits


def encoder_tokens(name,tokenizer,records,inference):
    spec=ENCODERS[name];transform=normalizer() if name=='conpara' and inference else lambda s:s
    # Per-method context budget is frozen before any fitting or test scoring.
    encoded=tokenizer([transform(r['text']) for r in records],truncation=True,max_length=spec['max_tokens'],padding=False)
    return [dict(input_ids=ids,attention_mask=mask) for ids,mask in zip(encoded['input_ids'],encoded['attention_mask'])]


def encoder_scores(name,model,tokenizer,records,encoded=None):
    import torch
    if encoded is None:encoded=encoder_tokens(name,tokenizer,records,True)
    model.eval();scores={};batch_size=2 if name!='mage' else 1
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for offset in range(0,len(records),batch_size):
            batch=tokenizer.pad(encoded[offset:offset+batch_size],padding=True,return_tensors='pt').to('cuda');batch.pop('token_type_ids',None)
            logits=output_logits(model(**batch)).double()
            p=logits[:,0].sigmoid() if name=='desklib' else logits.softmax(-1)[:,1]
            scores.update({r['id']:float(np.clip(v,1e-12,1-1e-12)) for r,v in zip(records[offset:offset+batch_size],p.cpu().tolist())})
    return scores


def encoder_train(name):
    import torch
    from transformers import AutoTokenizer,get_linear_schedule_with_warmup
    prepare();out=OUT/'A'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'fit_before_test.json').exists():return
    spec=ENCODERS[name];recipe=prepare()['encoders_training'];base=ROOT/'models'/spec['directory'];torch.set_num_threads(4)
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);np.random.seed(SEED);random.seed(SEED)
    tokenizer=local_tokenizer(base,use_fast=name not in ['desklib','conpara']);model=new_encoder(name).float().cuda()
    backbone=model.encoder if hasattr(model,'encoder') else model
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if hasattr(model,'config'):model.config.use_cache=False
    train=phase_rows('train');development=phase_rows('development');tokens=encoder_tokens(name,tokenizer,train,False);dev_tokens=encoder_tokens(name,tokenizer,development,True)
    parameters=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(parameters,lr=spec['lr'],weight_decay=.01,foreach=False)
    batches=math.ceil(len(train)/spec['batch']);steps_epoch=math.ceil(batches/spec['accumulation']);total_steps=steps_epoch*recipe['max_epochs'];scheduler=get_linear_schedule_with_warmup(optimizer,int(.06*total_steps),total_steps)
    checkpoint=out/'resume.pt';current=dict(epoch=0,next_batch=0,step=0,best=[-1.,-1.],history=[],elapsed=0.)
    if checkpoint.exists():
        cp=torch.load(checkpoint,map_location='cpu',weights_only=True);assert cp['protocol_sha256']==sha(OUT/'protocol_before_training.json')
        model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler']);current=cp['state'];torch.set_rng_state(cp['rng']);torch.cuda.set_rng_state_all(cp['cuda_rng']);del cp
    started=time.monotonic();prior=current['elapsed']
    def save_checkpoint():
        current['elapsed']=prior+time.monotonic()-started;temp=checkpoint.with_suffix('.tmp')
        # Locally produced numerical state only; safe weights-only restoration.
        torch.save(dict(protocol_sha256=sha(OUT/'protocol_before_training.json'),model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),state=current,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temp);temp.replace(checkpoint)
    optimizer.zero_grad(set_to_none=True);state('encoder_training',framework=name,completed=current['step'],total=total_steps,parameters=sum(p.numel() for p in parameters))
    for epoch in range(current['epoch'],recipe['max_epochs']):
        order=list(range(len(train)));random.Random(SEED+epoch).shuffle(order);begin=current['next_batch'] if epoch==current['epoch'] else 0;running=0;count=0;model.train()
        for index in range(begin,batches):
            selected=order[index*spec['batch']:(index+1)*spec['batch']];batch=tokenizer.pad([tokens[i] for i in selected],padding=True,return_tensors='pt').to('cuda');batch.pop('token_type_ids',None);batch['labels']=torch.tensor([train[i]['label'] for i in selected],device='cuda')
            divisor=min(spec['accumulation'],batches-(index//spec['accumulation'])*spec['accumulation'])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(**batch);loss=output['loss'] if isinstance(output,dict) else output.loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite '+name+' loss')
            (loss/divisor).backward();running+=float(loss.detach());count+=1
            if (index+1)%spec['accumulation'] and index+1<batches:continue
            torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);current['step']+=1;current.update(epoch=epoch,next_batch=index+1)
            if current['step']%25==0 or index+1==batches:
                if current['step']%250==0 or index+1==batches:save_checkpoint()
                state('encoder_training',framework=name,completed=current['step'],total=total_steps,epoch=epoch+1,loss=running/max(1,count),elapsed_seconds=prior+time.monotonic()-started,peak_cuda_GiB=torch.cuda.max_memory_allocated()/1024**3)
                running=count=0
        scores=encoder_scores(name,model,tokenizer,development,dev_tokens);m=development_metrics(development,scores);rank=[m['equal_cell_macro_F1'],m['AUROC']];current['history'].append(dict(epoch=epoch+1,development=m))
        if rank>current['best']:
            current.update(best=rank,best_epoch=epoch+1);torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'best_model.pt')
        current.update(epoch=epoch+1,next_batch=0);save_checkpoint();state('encoder_epoch_complete',framework=name,epoch=epoch+1,development=m)
    model.load_state_dict(torch.load(out/'best_model.pt',map_location='cpu',weights_only=True));calibration=phase_rows('calibration');cuts=calibration_thresholds(calibration,encoder_scores(name,model,tokenizer,calibration))
    save(out/'fit_before_test.json',dict(at=now(),model_sha256=sha(out/'best_model.pt'),base_source_sha256=sha(base/'source.json'),selected_epoch=current['best_epoch'],history=current['history'],calibration=cuts,training_documents=len(train),fresh_task_initialization=True,all_encoder_parameters_trained=True,trainable_parameters=sum(p.numel() for p in parameters),training_elapsed_seconds=current['elapsed'],max_training_tokens=spec['max_tokens'],peak_cuda_GiB=torch.cuda.max_memory_allocated()/1024**3,protocol_sha256=sha(OUT/'protocol_before_training.json'),test_scores_seen=False))
    del model,optimizer,parameters,output,loss,batch;gc.collect();torch.cuda.empty_cache()


def encoder_evaluate(name):
    import torch
    from transformers import AutoTokenizer
    out=OUT/'A'/name;frozen=json.loads((out/'fit_before_test.json').read_text(encoding='utf-8'));assert sha(out/'best_model.pt')==frozen['model_sha256']
    tokenizer=local_tokenizer(ROOT/'models'/ENCODERS[name]['directory'],use_fast=name not in ['desklib','conpara']);model=new_encoder(name).float().cuda();model.load_state_dict(torch.load(out/'best_model.pt',map_location='cpu',weights_only=True))
    test=phase_rows('test');target=out/'predictions.jsonl';done={r['id']:r['score'] for r in rows(target)} if target.exists() else {};assert set(done)<={r['id'] for r in test}
    pending=[r for r in test if r['id'] not in done];started=time.monotonic()
    with target.open('a',encoding='utf-8') as stream:
        for offset in range(0,len(pending),64):
            part=pending[offset:offset+64];scores=encoder_scores(name,model,tokenizer,part)
            for r in part:stream.write(json.dumps(dict(id=r['id'],score=scores[r['id']]))+'\n')
            stream.flush();done.update(scores)
            if offset%256==0 or offset+64>=len(pending):state('test_scoring',framework=name,completed=len(done),total=len(test),elapsed_seconds=time.monotonic()-started)
    del model;gc.collect();torch.cuda.empty_cache();write_results(name,test,done,frozen)


def write_results(name,records,scores,frozen,out_override=None):
    out=Path(out_override) if out_override is not None else OUT/'A'/name
    if not out.resolve().is_relative_to(OUT.resolve()):raise ValueError('Report path outside comparison run')
    path=out/'predictions.jsonl';jsonl(path,[dict(id=r['id'],score=scores[r['id']]) for r in records])
    report=evaluate(DATA/'protocols/A',path,threshold=.5,bootstrap=200)
    report.update(framework=name,variant=frozen.get('variant',name),fresh_same_data_training=True,model_sha256=frozen['model_sha256'],protocol_before_training_sha256=frozen['protocol_sha256'],calibrated_operating_points={k:dict(calibration=v,test=metric(records,scores,v['threshold'])) for k,v in frozen['calibration'].items()})
    save(out/'metrics.json',report);state('framework_complete',framework=name,records=len(records),metrics=report['overall'])


def planned_frameworks():
    available=prepare()['frameworks'];scope=OUT/'run_scope.json'
    if not scope.exists():return available
    selected=json.loads(scope.read_text(encoding='utf-8'))['frameworks']
    if not selected or len(selected)!=len(set(selected)) or any(name not in available for name in selected):raise ValueError('Invalid current comparison scope')
    return selected


def report():
    planned=planned_frameworks();result={};model_paths={name:OUT/'A'/name for name in planned};variant=None
    mapping=OUT/'effective_model_map.json'
    if mapping.exists():
        effective=json.loads(mapping.read_text(encoding='utf-8'));variant=effective['variant']
        for name,relative in effective['models'].items():
            target=(OUT/relative).resolve()
            if name not in prepare()['frameworks'] or not target.is_relative_to(OUT.resolve()):raise ValueError('Invalid effective model path')
            if name not in model_paths:continue
            model_paths[name]=target
        own=model_paths['fivetrace']/'protocol_before_training.json'
        assert sha(own)==effective['protocol_sha256']
    for name in planned:
        p=model_paths[name]/'metrics.json'
        if p.exists():result[name]=json.loads(p.read_text(encoding='utf-8'))
    paired={}
    if len(result)==len(planned):
        test=phase_rows('test');names=list(result);maps={name:{r['id']:r['score'] for r in rows(model_paths[name]/'predictions.jsonl')} for name in names}
        for scope in ['all','en','zh']:
            rr=test if scope=='all' else [r for r in test if r['language']==scope];grouped=defaultdict(dict)
            for r in rr:grouped[r['group_id']][r['label']]=r['id']
            assert all(set(pair)=={0,1} for pair in grouped.values())
            pairs=list(grouped.values());human=np.array([[maps[name][p[0]]>=.5 for p in pairs] for name in names]);AI=np.array([[maps[name][p[1]]>=.5 for p in pairs] for name in names]);n=len(pairs)
            def f1(h,a):
                fp=h.sum(axis=1);tp=a.sum(axis=1);tn=n-fp;fn=n-tp
                return (2*tp/np.maximum(1,2*tp+fp+fn)+2*tn/np.maximum(1,2*tn+fp+fn))/2
            point=f1(human,AI);rng=np.random.default_rng(SEED);values=[]
            for _ in range(200):
                sample=rng.integers(0,n,n);values.append(f1(human[:,sample],AI[:,sample]))
            values=np.array(values);ours=names.index('fivetrace')
            paired[scope]={name:dict(delta_macro_F1=float(point[ours]-point[i]),source_group_paired_95CI=np.quantile(values[:,ours]-values[:,i],[.025,.975]).tolist(),source_groups=n,replicates=200) for i,name in enumerate(names) if name!='fivetrace'}
        save(OUT/'completion_integrity.json',dict(at=now(),frameworks=names,self_variant=variant,unique_test_records=len(test),complete_prediction_ID_sets=True,prediction_files_sha256={name:sha(model_paths[name]/'predictions.jsonl') for name in names},protocol_sha256=sha(OUT/'protocol_before_training.json'),dataset_verification_sha256=sha(DATA/'verification.json'),models_trained_from_pretrained_only=True,default_unchanged=True,data_uploaded=False))
    save(OUT/'comparison.json',dict(at=now(),complete=len(result)==len(planned),planned_frameworks=planned,completed_frameworks=list(result),self_variant=variant,protocol='A same-data fresh training; B not run',models=result,paired_FiveTrace_minus_baseline=paired,default_changed=False,data_uploaded=False))
    title='V3 协议 A 评测' if planned==['fivetrace'] else '统一数据重训：框架对比'
    lines=['# '+title,'',f'自研候选：{variant or "original Qwen-representation feature 1"}。已完成 {len(result)}/{len(planned)}；未完成框架不填分数。当前仅协议 A，不能称 B 未见家族结果。','']
    if planned==['fivetrace']:lines+=['按用户最新要求，本轮只评测V3及协议A，四个开源框架训练／测试已延后。','']
    for scope,title in [('overall','中英整体'),('en','英文测试'),('zh','中文分层')]:
        lines+=['## '+title,'','| 框架 | 文本数 | Macro-F1 | AI检出率 | 人工误报率 | AUROC |','|---|---:|---:|---:|---:|---:|']
        for name,r in result.items():
            m=r['overall'] if scope=='overall' else r['strata']['language'][scope]
            label='FiveTrace V3 (mDeBERTa)' if name=='fivetrace' and variant=='FiveTrace-V3-mDeBERTa' else name
            lines.append(f"| {label} | {m['records']} | {m['macro_F1']:.4f} | {m['AI_recall']:.2%} | {m['human_FPR']:.2%} | {m['AUROC']:.4f} |")
        lines+=['']
    lines+=['','每种框架都从基础预训练权重新建任务参数。可用训练池相同，但组件分工、参数量、上下文长度、训练步数和计算预算不同，不能宣称完全控制这些变量。','',
        'XLM-R与Qwen为多语言骨干；DeBERTa/Longformer为英文骨干。应单列英文共同测试结果及双语覆盖差异，不把中文能力差异泛化为作者原模型的结论。','',
        'Desklib、ConPara、MAGE为公开结构／训练流程在新语料上的适配实验，最大学习轮数为3，不能冒充作者在原数据、原训练预算上的复现成绩。Tropa的原生四类监督不适用本二类主集，未另设一个伪四类对照。','',
        '本数据集已有研究设计暴露，基座预训练来源未知；从未使用测试标签拟合或选择阈值。数据和逐条预测仅本地保留。']
    (OUT/'对比报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
