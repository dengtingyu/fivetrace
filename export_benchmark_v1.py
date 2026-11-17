"""Deterministic source-group sampling helpers; no dataset is distributed."""
from collections import Counter,defaultdict,deque
import hashlib
from benchmark_v1_policy import SEED

def hashkey(*parts):return hashlib.sha256(('|'.join(map(str,(SEED,)+parts))).encode()).hexdigest()


def select_pairs(candidates,count_per_language,tag,exclude=()):
    used=set(exclude);selected=[];counts=Counter();buckets=defaultdict(list)
    for root,pp in candidates.items():
        if root in used:continue
        for p in pp:
            cell=(p['language'],p['domain'],p['length_bin'],p['family'])
            buckets[cell].append(p)
    for cell,pp in buckets.items():pp.sort(key=lambda p:hashkey(tag,p['root'],p['generator']),reverse=True)
    for lang in ['zh','en']:
        cells=sorted([k for k in buckets if k[0]==lang],key=lambda k:hashkey(tag,k))
        n=0
        while n<count_per_language:
            advanced=False
            for cell in cells:
                bucket=buckets[cell]
                while bucket and bucket[-1]['root'] in used:bucket.pop()
                if not bucket:continue
                p=bucket.pop();used.add(p['root']);selected.append(p);n+=1;counts[cell]+=1;advanced=True
                if n>=count_per_language:break
            if not advanced:break
    return selected


def balanced_pairs(candidates,target,tag,exclude=()):
    initial=select_pairs(candidates,target,tag,exclude)
    counts=Counter(p['language'] for p in initial);n=min(counts.get('zh',0),counts.get('en',0))
    if n<target:return select_pairs(candidates,n,tag,exclude)
    return initial
