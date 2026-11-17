"""Fresh, same-data framework comparison; no historical detector initialization."""
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
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from ai_detector.config import ROOT
from benchmark_v1_api import OUT as DATA,load_manifest,sha
from train_fivetrace import save,now,jsonl,rows

WORK=ROOT/'detector_research/framework_comparison_v1_20261115'
OUT=ROOT/'artifacts/framework_comparison_v1_20261115'
BASE=ROOT/'models/Qwen3-1.7B-Base'
SEED=20261115
ENCODERS={
 'xlmr':dict(repo='FacebookAI/xlm-roberta-base',revision='e73636d4f797dec63c3081bb6ed5c7b0bb3f2089',directory='framework-base-xlmr',kind='sequence_classifier',max_tokens=512,lr=2e-5,batch=2,accumulation=16),
 'desklib':dict(repo='microsoft/deberta-v3-large',revision='64a8c8eab3e352a784c658aef62be1662607476f',directory='framework-base-deberta',kind='masked_mean_linear',max_tokens=768,lr=1e-5,batch=1,accumulation=32),
 'conpara':dict(repo='microsoft/deberta-v3-large',revision='64a8c8eab3e352a784c658aef62be1662607476f',directory='framework-base-deberta',kind='CLS_MLP_raw_train_normalized_inference',max_tokens=512,lr=1e-5,batch=1,accumulation=32),
 'mage':dict(repo='allenai/longformer-base-4096',revision='301e6a42cb0d9976a6d6a26a079fef81c18aa895',directory='framework-base-longformer',kind='sequence_classifier_global_CLS',max_tokens=2048,lr=3e-5,batch=1,accumulation=16),
}


def state(stage,**kwargs):
    value=dict(at=now(),stage=stage,**kwargs);save(OUT/'progress.json',value)
    if 'framework' in kwargs:save(OUT/(str(kwargs['framework'])+'_progress.json'),value)
    print(json.dumps(value),flush=True)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    destination=OUT/'protocol_before_training.json'
    if destination.exists():
        p=json.loads(destination.read_text(encoding='utf-8'))
        assert p['dataset_verification_sha256']==sha(DATA/'verification.json')
        for phase,entry in p['phases'].items():assert sha(WORK/entry['file'])==entry['sha256']
        return p
    manifest={phase:load_manifest(DATA/'protocols/A',phase)[0] for phase in ['train','development','calibration','test']}
    inputs={r['id']:r for rr in manifest.values() for r in rr}
    master=json.loads((DATA/'build_complete.json').read_text(encoding='utf-8'))['master'];assert sha(DATA/master['path'])==master['sha256']
    texts={r['id']:r['text'] for r in pq.read_table(DATA/master['path'],columns=['id','text']).to_pylist() if r['id'] in inputs}
    phases={}
    for phase,rr in manifest.items():
        file=WORK/(phase+'.jsonl');jsonl(file,[dict(r,text=texts[r['id']]) for r in rr])
        phases[phase]=dict(file=file.name,records=len(rr),sha256=sha(file))
    protocol=dict(at=now(),seed=SEED,scope='A covered-generator comparison first; B schedule not yet selected by user',
        dataset_verification_sha256=sha(DATA/'verification.json'),phases=phases,
        frameworks=['fivetrace','xlmr','desklib','conpara','mage'],encoders=ENCODERS,
        initialization='Qwen Base / original pretrained encoder weights only; all task adapters and heads fresh. Never load old detector task checkpoints.',
        equal_data='All frameworks receive the same available training pool; exact counts are recorded in phases. FiveTrace uses the frozen source-disjoint reference/classifier/fusion roles; end-to-end encoders may use all TRAIN labels. Same development/calibration/test IDs.',
        fivetrace=dict(reference_epochs=3,reference_selection='Minimum equal-language development NLL',rank=8,alpha=16,targets=['q_proj','v_proj'],dropout=.05,max_training_tokens=256,batch=2,accumulation=4,lr=2e-4,
            training_document_coverage='Every reference document sampled once per epoch; deterministic original-token window, not a claim of full-document-token LM training.',
            development_reference_pairs_per_language=100,style='zh-en-reliability-v21',outer_Chinese_characters=600,outer_English_tokens=384,outer_English_minimum_tail=128,
            head_C=[.01,.05,.2],fusion_C=[.03,.1,1.],component_roles='reference/classifier/fusion from frozen A train; no cross-role task fitting'),
        encoders_training=dict(max_epochs=3,selection='Development equal-language-domain Macro-F1, then AUROC; test is not read by fitting code',optimizer='Full-parameter AdamW float32 states, BF16 autocast, gradient checkpointing, foreach=False',warmup_fraction=.06,weight_decay=.01,gradient_clip=1.,
            overflow_policy='Fail and report; do not silently freeze encoder or switch to task-trained checkpoints.',
            truncation='Document prefixes at the disclosed author-style context budget; all document IDs used. Do not claim equal token or compute budgets.'),
        calibration='Default 0.5 and separately calibrated 1%/5% empirical human-FPR thresholds, calibration split only. No test-label threshold search.',
        reporting='Full binary coverage; AI recall, human FPR, Macro-F1, AUROC, balanced accuracy, per language/domain/length/family, source-group bootstrap. English-native backbones disclosed separately from multilingual encoders.',
        tropa='Not a separate same-data reproduction: native 4-class supervision needs edited/humanized labels absent from strict binary TRAIN. Mean-pooling backbone is represented by the Desklib-style binary architecture; no claimed Tropa author reproduction.',
        method_names='Architecture/recipe adaptations on this new corpus, not reproduction of the authors published original-data scores.',
        source_code={p.relative_to(ROOT).as_posix():sha(p) for p in (WORK/'author_sources').glob('*') if p.is_file()},
        pretrained_Qwen_sha256=sha(BASE/'model.safetensors'),old_default_changed=False,test_labels_used_for_training=False,data_uploaded=False)
    save(destination,protocol);state('same_data_protocol_frozen',frameworks=protocol['frameworks'],training_records=phases['train']['records'])
    return protocol


def ref_train():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import get_peft_model,LoraConfig,TaskType
    p=prepare();recipe=p['fivetrace'];out=OUT/'A/fivetrace';out.mkdir(parents=True,exist_ok=True)
    data=[r for r in rows(WORK/'train.jsonl') if r['component_role']=='reference']
    development=rows(WORK/'development.jsonl');dev=[]
    for lang in ['zh','en']:
        groups={r['group_id'] for r in development if r['language']==lang}
        selected=set(sorted(groups,key=lambda g:hashlib.sha256((str(SEED)+'reference_dev'+g).encode()).hexdigest())[:recipe['development_reference_pairs_per_language']])
        dev.extend(r for r in development if r['group_id'] in selected)
    torch.set_num_threads(4);tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    def pack(examples):
        n=max(len(x['tokens']) for x in examples);ids=torch.full((len(examples),n),tokenizer.eos_token_id,device='cuda',dtype=torch.long);mask=torch.zeros_like(ids);labels=torch.full_like(ids,-100)
        for j,x in enumerate(examples):
            ids[j,:len(x['tokens'])]=torch.tensor(x['tokens'],device='cuda');mask[j,:len(x['tokens'])]=1;labels[j,:len(x['tokens'])]=ids[j,:len(x['tokens'])]
        return dict(input_ids=ids,attention_mask=mask,labels=labels)
    for name,label in [('ai',1),('human',0)]:
        directory=out/'adapters'/name;directory.mkdir(parents=True,exist_ok=True)
        if (directory/'training_complete.json').exists():continue
        torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);np.random.seed(SEED);random.seed(SEED)
        def encode(rr):
            result=[]
            for r in rr:
                tokens=tokenizer.encode(r['text'],add_special_tokens=False)
                if len(tokens)>recipe['max_training_tokens']:
                    start=random.Random(r['group_id']).randrange(len(tokens)-recipe['max_training_tokens']+1);tokens=tokens[start:start+recipe['max_training_tokens']]
                if len(tokens)<2:raise ValueError('Reference sample has fewer than two tokens')
                result.append(dict(id=r['id'],group_id=r['group_id'],language=r['language'],tokens=tokens))
            return result
        training=encode([r for r in data if r['label']==label]);validation=encode([r for r in dev if r['label']==label])
        save(directory/'sample_manifest.json',dict(train=[{k:v for k,v in r.items() if k!='tokens'} for r in training],development=[{k:v for k,v in r.items() if k!='tokens'} for r in validation]))
        model=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
        model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=recipe['rank'],lora_alpha=recipe['alpha'],target_modules=recipe['targets'],lora_dropout=recipe['dropout'],bias='none'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
        params=[(n,v) for n,v in model.named_parameters() if v.requires_grad];optimizer=torch.optim.AdamW([v for _,v in params],lr=recipe['lr'],weight_decay=.01,foreach=False)
        assert params and all('lora_' in n for n,v in params)
        def validation_loss():
            model.eval();values={}
            with torch.inference_mode():
                for lang in ['zh','en']:
                    selected=[r for r in validation if r['language']==lang];total=weight=0
                    for offset in range(0,len(selected),2):
                        b=pack(selected[offset:offset+2]);n=int((b['labels'][:,1:]!=-100).sum());total+=float(model(**b).loss)*n;weight+=n
                    values[lang]=total/weight
            model.train();return dict(by_language=values,macro=sum(values.values())/2)
        checkpoint=directory/'resume.pt';current=dict(epoch=0,next_batch=0,step=0,best=float('inf'),history=[],elapsed=0,trained_tokens=0)
        if checkpoint.exists():
            cp=torch.load(checkpoint,map_location='cpu',weights_only=True);assert cp['protocol_sha256']==sha(OUT/'protocol_before_training.json')
            for n,v in params:v.data.copy_(cp['adapters'][n].to(v.device))
            optimizer.load_state_dict(cp['optimizer']);current=cp['state'];torch.set_rng_state(cp['rng']);torch.cuda.set_rng_state_all(cp['cuda_rng']);del cp
        else:current['initial_development']=validation_loss()
        batches=math.ceil(len(training)/recipe['batch']);steps_epoch=math.ceil(batches/recipe['accumulation']);steps=steps_epoch*recipe['reference_epochs'];started=time.monotonic();prior=current['elapsed']
        def to_cpu(v):
            if isinstance(v,torch.Tensor):return v.detach().cpu()
            if isinstance(v,dict):return {k:to_cpu(x) for k,x in v.items()}
            if isinstance(v,list):return [to_cpu(x) for x in v]
            return v
        def save_checkpoint():
            current['elapsed']=prior+time.monotonic()-started
            temp=checkpoint.with_suffix('.tmp');torch.save(dict(protocol_sha256=sha(OUT/'protocol_before_training.json'),adapters={n:v.detach().cpu() for n,v in params},optimizer=to_cpu(optimizer.state_dict()),state=current,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temp);temp.replace(checkpoint)
        state('reference_training',framework='fivetrace',adapter=name,completed=current['step'],total=steps,training_documents=len(training),development_documents=len(validation))
        model.train();optimizer.zero_grad(set_to_none=True)
        for epoch in range(current['epoch'],recipe['reference_epochs']):
            order=list(range(len(training)));random.Random(SEED+epoch).shuffle(order);begin=current['next_batch'] if epoch==current['epoch'] else 0;running=0;count=0
            for index in range(begin,batches):
                examples=[training[j] for j in order[index*recipe['batch']:(index+1)*recipe['batch']]];b=pack(examples);loss=model(**b).loss
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite reference loss')
                last_group=min(recipe['accumulation'],batches-(index//recipe['accumulation'])*recipe['accumulation'])
                (loss/last_group).backward();running+=float(loss.detach());count+=1;current['trained_tokens']+=int((b['labels'][:,1:]!=-100).sum())
                if (index+1)%recipe['accumulation'] and index+1<batches:continue
                current['step']+=1;step=current['step'];warm=max(1,int(steps*.06));rate=recipe['lr']*(step/warm if step<=warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,steps-warm))))
                for group in optimizer.param_groups:group['lr']=rate
                norm=float(torch.nn.utils.clip_grad_norm_([v for _,v in params],1.));optimizer.step();optimizer.zero_grad(set_to_none=True)
                current.update(epoch=epoch,next_batch=index+1)
                record=dict(step=step,epoch=epoch+1,train_nll=running/max(1,count),lr=rate,gradient_norm=norm)
                if step%200==0 or index+1==batches:
                    value=validation_loss();record['development']=value
                    if value['macro']<current['best']:
                        current.update(best=value['macro'],best_step=step);model.save_pretrained(directory,safe_serialization=True)
                current['history'].append(record)
                if step%25==0 or index+1==batches:
                    save_checkpoint();state('reference_training',framework='fivetrace',adapter=name,completed=step,total=steps,epoch=epoch+1,train_nll=record['train_nll'],best_development=current['best'] if math.isfinite(current['best']) else None,elapsed_seconds=current['elapsed'],trained_tokens=current['trained_tokens'])
                running=count=0
            current.update(epoch=epoch+1,next_batch=0);save_checkpoint()
        assert (directory/'adapter_model.safetensors').exists() and current['step']==steps
        save(directory/'training_complete.json',dict(at=now(),state=current,selected='minimum equal-language development NLL',train_documents=len(training),development_documents=len(validation),adapter_sha256=sha(directory/'adapter_model.safetensors'),trainable_parameters=sum(v.numel() for n,v in params),fresh_initialization=True))
        del model,optimizer,params,loss,b;gc.collect();torch.cuda.empty_cache()
    save(out/'references_complete.json',dict(at=now(),branches={name:json.loads((out/f'adapters/{name}/training_complete.json').read_text(encoding='utf-8'))['adapter_sha256'] for name in ['ai','human']},old_task_weights_loaded=False))
    state('references_complete',framework='fivetrace')


def acquire():
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor,as_completed
    prepare();completed=set()
    for spec in ENCODERS.values():
        if spec['repo'] in completed:continue
        completed.add(spec['repo']);directory=ROOT/'models'/spec['directory'];directory.mkdir(parents=True,exist_ok=True)
        with urllib.request.urlopen('https://huggingface.co/api/models/'+spec['repo']+'/revision/'+spec['revision']+'?blobs=true',timeout=30) as response:metadata=json.load(response)
        assert metadata['sha']==spec['revision'];files={f['rfilename']:f for f in metadata['siblings']}
        weight='model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin'
        small=[name for name in ['config.json','tokenizer.json','tokenizer_config.json','sentencepiece.bpe.model','spm.model','vocab.json','merges.txt','special_tokens_map.json'] if name in files]
        for name in small:
            path=directory/name
            if not path.exists() or path.stat().st_size!=files[name]['size']:
                with urllib.request.urlopen('https://huggingface.co/'+spec['repo']+'/resolve/'+spec['revision']+'/'+name,timeout=60) as response:content=response.read()
                assert len(content)==files[name]['size'];path.write_bytes(content)
        f=files[weight];target=directory/weight;digest=f['lfs']['sha256']
        if not target.exists() or sha(target)!=digest:
            download_weight(spec,weight,f['size'],digest,target)
        assert sha(target)==digest
        save(directory/'source.json',dict(repo=spec['repo'],revision=spec['revision'],task_detector_weights=False,files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in directory.iterdir() if p.is_file() and p.name!='source.json'}))
        state('pretrained_backbone_verified',repo=spec['repo'],revision=spec['revision'],bytes=f['size'])


def download_weight(spec,filename,size,digest,target):
    """Bounded, resumable ranges; distinct URLs avoid stale range-cache replies."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor,as_completed
    directory=ROOT/'.downloads/framework_comparison_v1'/spec['directory'];directory.mkdir(parents=True,exist_ok=True)
    chunks=[(start,min(size,start+8*1024*1024)) for start in range(0,size,8*1024*1024)]
    def fetch(bounds):
        begin,end=bounds;path=directory/f'{begin}-{end}.part'
        failures=0
        preferred=os.environ.get('FRAMEWORK_WEIGHT_HOST','https://huggingface.co').rstrip('/')
        hosts=list(dict.fromkeys([preferred,'https://huggingface.co','https://hf-mirror.com']))
        while True:
            present=path.stat().st_size if path.exists() else 0
            if present==end-begin:return path
            if present>end-begin:raise ValueError('Oversized download range')
            offset=begin+present;piece_end=min(end,offset+1024*1024)
            host=hosts[failures%len(hosts)]
            url=host+'/'+spec['repo']+'/resolve/'+spec['revision']+'/'+filename+'?download=true&segment='+str(offset)+'-'+str(piece_end-1)+'&nonce='+str(time.time_ns())
            try:
                request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-{piece_end-1}'})
                with urllib.request.urlopen(request,timeout=35) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {offset}-{piece_end-1}/{size}':raise ValueError('Unexpected range response')
                    with path.open('ab') as stream:
                        remaining=piece_end-offset
                        while remaining:
                            part=response.read(min(64*1024,remaining))
                            if not part:raise OSError('Premature end of ranged response')
                            stream.write(part);remaining-=len(part)
                if path.stat().st_size!=piece_end-begin:raise OSError('Range byte count mismatch')
                failures=0
                if path.stat().st_size==end-begin:return path
            except Exception as error:
                failures+=1
                print(json.dumps(dict(stage='pretrained_download_retry',repo=spec['repo'],part_start=begin,bytes_present=path.stat().st_size if path.exists() else 0,consecutive_failures=failures,error=repr(error),host=host)),flush=True)
                if failures>=12:raise
                time.sleep(min(5,failures))
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures=[executor.submit(fetch,b) for b in chunks]
        for i,future in enumerate(as_completed(futures),1):
            future.result()
            if i%8==0 or i==len(chunks):
                print(json.dumps(dict(stage='pretrained_weight_download',repo=spec['repo'],completed_ranges=i,total_ranges=len(chunks),downloaded_bytes=sum(p.stat().st_size for p in directory.glob('*.part')),total_bytes=size,elapsed_seconds=time.monotonic()-started)),flush=True)
    temporary=target.with_suffix('.assembling')
    with temporary.open('wb') as stream:
        for begin,end in chunks:
            with (directory/f'{begin}-{end}.part').open('rb') as source:
                while part:=source.read(8*1024*1024):stream.write(part)
    assert temporary.stat().st_size==size and sha(temporary)==digest,'Pretrained checkpoint hash mismatch'
    temporary.replace(target)


def pipeline(wait_references=False):
    import subprocess
    effective=OUT/'effective_model_map.json'
    if effective.exists() and json.loads(effective.read_text(encoding='utf-8')).get('variant')=='FiveTrace-V3-mDeBERTa':
        raise RuntimeError('Original self-head pipeline was superseded; use framework_v3_deberta.py pipeline, and do not duplicate an already running queue')
    prepare();OUT.mkdir(parents=True,exist_ok=True);stages=[]
    if not wait_references:stages.append(('references',None))
    else:
        started=time.monotonic()
        while not (OUT/'A/fivetrace/references_complete.json').exists():
            if time.monotonic()-started>6*3600:raise TimeoutError('Adopted reference trainer did not complete; inspect its log, do not launch duplicate GPU work')
            save(OUT/'pipeline_state.json',dict(at=now(),stage='waiting_for_existing_reference_training',status='running'));time.sleep(10)
    stages += [('five-features',None),('five-fit',None),('five-test-features',None),('five-evaluate',None)]
    for name in ENCODERS:stages.extend([('encoder-train',name),('encoder-evaluate',name)])
    completed=[]
    for stage,model in stages:
        if model:
            started=time.monotonic()
            while not (ROOT/'models'/ENCODERS[model]['directory']/'source.json').exists():
                if time.monotonic()-started>2*3600:raise TimeoutError('Pretrained backbone unavailable; see backbone_acquisition.log')
                save(OUT/'pipeline_state.json',dict(at=now(),stage='waiting_for_backbone',model=model,status='running'));time.sleep(10)
        key=stage+('_'+model if model else '')
        command=[sys.executable,str(Path(__file__).resolve()),stage]+(['--model',model] if model else [])
        save(OUT/'pipeline_state.json',dict(at=now(),status='running',stage=key,completed=completed))
        with (OUT/(key+'.log')).open('a',encoding='utf-8') as log:
            child=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if child.returncode:
            save(OUT/'pipeline_state.json',dict(at=now(),status='failed',stage=key,returncode=child.returncode,completed=completed));raise RuntimeError('Framework stage failed: '+key)
        completed.append(key)
        from framework_trainers_v1 import report
        report()
    save(OUT/'pipeline_state.json',dict(at=now(),status='completed',completed=completed,frameworks=prepare()['frameworks'],protocol='A',B_not_scheduled=True))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','references','acquire','five-features','five-fit','five-test-features','five-evaluate','encoder-train','encoder-evaluate','report','pipeline']);parser.add_argument('--model',choices=list(ENCODERS));parser.add_argument('--wait-references',action='store_true');args=parser.parse_args()
    try:
        if args.stage in ['prepare','references','acquire']:{'prepare':prepare,'references':ref_train,'acquire':acquire}[args.stage]()
        elif args.stage=='pipeline':pipeline(args.wait_references)
        else:
            import framework_trainers_v1 as trainers
            if args.stage=='five-features':trainers.five_features('fit')
            elif args.stage=='five-test-features':trainers.five_features('test')
            elif args.stage=='five-fit':trainers.five_fit()
            elif args.stage=='five-evaluate':trainers.five_evaluate()
            elif args.stage=='encoder-train':trainers.encoder_train(args.model)
            elif args.stage=='encoder-evaluate':trainers.encoder_evaluate(args.model)
            elif args.stage=='report':trainers.report()
    except Exception as error:
        save(OUT/(args.stage+'_failure.json'),dict(at=now(),stage=args.stage,model=args.model,error=repr(error)))
        raise
