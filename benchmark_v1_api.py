"""Versioned local protocol readers and score evaluation. No detector training."""
from collections import Counter,defaultdict
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score,roc_auc_score,confusion_matrix

from build_benchmark_v1 import OUT


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8') as stream:return [json.loads(line) for line in stream if line.strip()]


def load_manifest(protocol_dir,phase):
    protocol_dir=Path(protocol_dir).resolve()
    if not protocol_dir.is_relative_to((OUT/'protocols').resolve()):raise ValueError('Protocol must be inside this benchmark version')
    index=OUT/'protocols/index.json'
    if index.exists() and protocol_dir.relative_to((OUT/'protocols').resolve()).as_posix() not in json.loads(index.read_text(encoding='utf-8'))['protocols']:
        raise ValueError('Protocol is not in the active version index')
    if phase not in ['train','development','calibration','test']:raise ValueError('Unknown dataset phase')
    p=json.loads((protocol_dir/'protocol.json').read_text(encoding='utf-8'));entry=p['splits'][phase]
    file=(OUT/entry['file']).resolve()
    if not file.is_relative_to(OUT.resolve()):raise ValueError('Manifest path escapes the benchmark')
    if sha(file)!=entry['sha256']:raise ValueError('Frozen split manifest changed')
    records=read(file)
    if len(records)!=entry['records'] or len({r['id'] for r in records})!=len(records):raise ValueError('Split count/identity mismatch')
    return records,p


def load_training(protocol_dir,component=None):
    if component not in [None,'reference','classifier','fusion']:raise ValueError('Unknown training component')
    records,protocol=load_manifest(protocol_dir,'train')
    if component:records=[r for r in records if r['component_role']==component]
    ids={r['id']:r for r in records}
    if not ids:raise ValueError('No records for this component')
    excluded=protocol.get('excluded_family')
    if excluded and any(r['label']==1 and r['family']==excluded for r in records):raise ValueError('Held-out family leaked into training')
    if protocol.get('name')=='B-method-diffusion' and any(r['family'] in ['llada','diffusiongemma','dream','sdar'] for r in records if r['label']==1):
        raise ValueError('Held-out generation mechanism leaked into training')
    verify_master(protocol)
    remaining=set(ids)
    for batch in pq.ParquetFile(OUT/'data/master.parquet').iter_batches(batch_size=512,columns=['id','text']):
        for value in batch.to_pylist():
            if value['id'] in ids:
                remaining.remove(value['id'])
                r=ids[value['id']]
                # Only the body is a model input. Source/model names are not
                # concatenated into prompts or features by this reader.
                yield dict(id=r['id'],text=value['text'],label=r['label'],group_id=r['group_id'],language=r['language'])
    if remaining:raise ValueError('Training bodies missing from master')


def verify_master(protocol):
    master=protocol.get('master') or json.loads((OUT/'protocols/A/protocol.json').read_text(encoding='utf-8'))['master']
    if sha(OUT/'data/master.parquet')!=master['sha256']:raise ValueError('Frozen text master changed')


def load_test_inputs(protocol_dir):
    records,protocol=load_manifest(protocol_dir,'test');ids={r['id'] for r in records};verify_master(protocol)
    remaining=set(ids)
    for batch in pq.ParquetFile(OUT/'data/master.parquet').iter_batches(batch_size=512,columns=['id','text']):
        for r in batch.to_pylist():
            if r['id'] in ids:
                remaining.remove(r['id']);yield r
    if remaining:raise ValueError('Test bodies missing from master')


def metric(records,scores,threshold):
    y=np.array([r['label'] for r in records]);s=np.array([scores[r['id']] for r in records]);p=s>=threshold
    human=y==0;ai=y==1;both=human.any() and ai.any()
    return dict(records=len(records),groups=len({r['group_id'] for r in records}),human=int(human.sum()),AI=int(ai.sum()),
                AI_recall=float(p[ai].mean()) if ai.any() else None,human_FPR=float(p[human].mean()) if human.any() else None,
                macro_F1=float(f1_score(y,p,labels=[0,1],average='macro',zero_division=0)) if both else None,
                balanced_accuracy=float((p[ai].mean()+1-p[human].mean())/2) if both else None,
                AUROC=float(roc_auc_score(y,s)) if both else None,confusion_matrix=confusion_matrix(y,p,labels=[0,1]).tolist())


def evaluate(protocol_dir,predictions,threshold=.5,bootstrap=200):
    if isinstance(threshold,bool) or not np.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Threshold must be finite and in [0,1]')
    if not isinstance(bootstrap,int) or bootstrap<0:raise ValueError('Bootstrap replicates must be a nonnegative integer')
    records,protocol=load_manifest(protocol_dir,'test');expected={r['id'] for r in records};scores={}
    for r in read(predictions):
        if r['id'] in scores:raise ValueError('Duplicate prediction ID')
        s=r.get('score')
        if isinstance(s,bool) or not isinstance(s,(int,float)) or not np.isfinite(s) or not 0<=s<=1:
            raise ValueError('Every input needs one finite score in [0,1]; unsupported/error cases must not be silently dropped')
        scores[r['id']]=float(s)
    if set(scores)!=expected:raise ValueError(f'Complete exact test coverage required: missing={len(expected-set(scores))}, extra={len(set(scores)-expected)}')
    result=dict(protocol=protocol.get('name'),prediction_sha256=sha(predictions),threshold=threshold,coverage=1.,
        overall=metric(records,scores,threshold),strata={},limitations=['Previously inventoried/exposed resources; no Base-pretraining decontamination claim.'])
    for field in ['language','domain','length_bin']:
        groups=defaultdict(list)
        for r in records:groups[r[field]].append(r)
        result['strata'][field]={k:metric(rr,scores,threshold) for k,rr in sorted(groups.items())}
    # A human control belongs to its matched generator pair, even though the
    # human text itself has no generator identity. Read pairs to preserve this.
    own_pairs=Path(protocol_dir)/'pairs.json'
    pairs=json.loads(own_pairs.read_text(encoding='utf-8')) if protocol.get('name')=='B-method-diffusion' else json.loads((OUT/'protocols/A/pairs.json').read_text(encoding='utf-8'))['test']
    rec={r['id']:r for r in records};family=defaultdict(dict);generator=defaultdict(dict)
    for pair in pairs:
        if pair['AI_id'] in rec and pair['human_id'] in rec:
            ai=rec[pair['AI_id']];family[ai['family']][ai['id']]=ai;family[ai['family']][pair['human_id']]=rec[pair['human_id']]
            key=ai['language']+':'+ai['generator']
            generator[key][ai['id']]=ai;generator[key][pair['human_id']]=rec[pair['human_id']]
    result['strata']['matched_generator_family']={f:metric(list(rr.values()),scores,threshold) for f,rr in sorted(family.items())}
    result['strata']['matched_generator_language']={g:metric(list(rr.values()),scores,threshold) for g,rr in sorted(generator.items())}
    family_values=[m['macro_F1'] for m in result['strata']['matched_generator_family'].values() if m['macro_F1'] is not None]
    result['equal_family_macro_F1']=float(np.mean(family_values)) if family_values else None
    strata=defaultdict(list)
    for r in records:strata[(r['language'],r['domain'])].append(r)
    cells={str(k):metric(rr,scores,threshold) for k,rr in sorted(strata.items())}
    valid=[m['macro_F1'] for m in cells.values() if m['macro_F1'] is not None]
    result['language_domain_cells']=cells
    result['primary_equal_cell_macro_F1']=float(np.mean(valid)) if valid else None
    result['worst_cell_macro_F1']=min(valid) if valid else None
    if bootstrap:
        groups=defaultdict(list)
        for r in records:groups[r['group_id']].append(r)
        ids=list(groups);rng=np.random.default_rng(20261115);values=[]
        for _ in range(bootstrap):
            sample=[r for i in rng.integers(0,len(ids),len(ids)) for r in groups[ids[i]]]
            m=metric(sample,scores,threshold)['macro_F1']
            if m is not None:values.append(m)
        result['source_group_bootstrap']=dict(replicates=bootstrap,seed=20261115,overall_macro_F1_95CI=np.quantile(values,[.025,.975]).tolist()) if values else None
    return result


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    export=sub.add_parser('export-train');export.add_argument('--protocol',type=Path,required=True);export.add_argument('--component',choices=['reference','classifier','fusion']);export.add_argument('--output',type=Path,required=True)
    inp=sub.add_parser('export-test-inputs');inp.add_argument('--protocol',type=Path,required=True);inp.add_argument('--output',type=Path,required=True)
    ev=sub.add_parser('evaluate');ev.add_argument('--protocol',type=Path,required=True);ev.add_argument('--predictions',type=Path,required=True);ev.add_argument('--output',type=Path,required=True);ev.add_argument('--threshold',type=float,default=.5);ev.add_argument('--bootstrap',type=int,default=200)
    args=p.parse_args()
    output=args.output.resolve()
    if output.is_relative_to(OUT.resolve()) and output.relative_to(OUT.resolve()).parts[0] in ['protocols','data','extensions','construction_history']:
        raise ValueError('Output would overwrite a frozen dataset asset')
    if output.is_relative_to(OUT.resolve()) and output.parent==OUT.resolve():
        raise ValueError('Write reports under validation/ or a separate directory')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.command=='evaluate':args.output.write_text(json.dumps(evaluate(args.protocol,args.predictions,args.threshold,args.bootstrap),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    else:
        iterator=load_training(args.protocol,args.component) if args.command=='export-train' else load_test_inputs(args.protocol)
        with args.output.open('w',encoding='utf-8') as stream:
            for r in iterator:stream.write(json.dumps(r,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
