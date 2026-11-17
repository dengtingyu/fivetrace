"""Validate four user-supplied local splits and freeze their manifests.

No corpus is downloaded or uploaded. Training rows must name their disjoint
reference/classifier/fusion role. All phases require complete human/AI pairs.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pyarrow as pa
import pyarrow.parquet as pq

from ai_detector.config import ROOT
from benchmark_v1_policy import length_bin
from train_fivetrace import rows, save, sha, jsonl, now

OUT=ROOT/"benchmarks/fivetrace_standard_v1"

def validate(splits):
    identities=set(); groups=set(); bodies=set()
    for phase, records in splits.items():
        local_groups=defaultdict(list); local_bodies=set()
        for r in records:
            if type(r.get("label")) is not int or r["label"] not in [0,1]: raise ValueError("Expected integer label 0/1")
            for field in ["id","group_id","text","language","domain","generator","family"]:
                if not isinstance(r.get(field),str) or not r[field]: raise ValueError("Missing "+field)
            if r["id"] in identities: raise ValueError("Repeated document ID")
            identities.add(r["id"])
            if r["language"] not in ["zh","en"]: raise ValueError("Expected zh/en language")
            if sum(not c.isspace() for c in r["text"])<100: raise ValueError("Input below 100 nonspace characters")
            value=re.sub(r"[\W_]+","",unicodedata.normalize("NFKC",r["text"]).casefold())
            if value in bodies: raise ValueError("Canonical body overlaps an earlier split")
            local_bodies.add(value); local_groups[r["group_id"]].append(r)
            if phase=="train" and r.get("component_role") not in ["reference","classifier","fusion"]: raise ValueError("Missing training component_role")
        if set(local_groups)&groups: raise ValueError("Source groups overlap splits")
        for rr in local_groups.values():
            if len(rr)!=2 or {r["label"] for r in rr}!={0,1}: raise ValueError("Each source group needs one human/AI pair")
            if len({(r["language"],r["domain"]) for r in rr})!=1: raise ValueError("A pair must share language and domain")
            if phase=="train" and len({r["component_role"] for r in rr})!=1: raise ValueError("A source group crosses component roles")
        if {(r["language"],r["label"]) for r in records}!={(lang,label) for lang in ["zh","en"] for label in [0,1]}: raise ValueError("Each phase needs both labels in Chinese and English")
        groups.update(local_groups); bodies.update(local_bodies)
    if {r["component_role"] for r in splits["train"]}!={"reference","classifier","fusion"}: raise ValueError("All three training roles are required")
    for role in ["reference","classifier","fusion"]:
        if {r["language"] for r in splits["train"] if r["component_role"]==role}!={"zh","en"}: raise ValueError("Each training role needs both languages")
    return splits

def prepare(splits):
    validate(splits)
    if OUT.exists(): raise FileExistsError("Refusing to overwrite an existing frozen local benchmark")
    OUT.mkdir(parents=True); meta={}; pairs={}; master=[]
    for phase, records in splits.items():
        manifest=[]; grouped=defaultdict(dict)
        for r in records:
            master.append(dict(id=r["id"],text=r["text"]))
            m={k:v for k,v in r.items() if k in ["id","label","group_id","language","domain","generator","family","component_role"]}
            m["characters"]=sum(not c.isspace() for c in r["text"]);m["length_bin"]=length_bin(r["language"],m["characters"])
            manifest.append(m); grouped[r["group_id"]][r["label"]]=r["id"]
        file=OUT/"protocols/A"/(phase+".jsonl");jsonl(file,manifest)
        meta[phase]=dict(file=file.relative_to(OUT).as_posix(),records=len(manifest),sha256=sha(file))
        pairs[phase]=[dict(human_id=p[0],AI_id=p[1]) for p in grouped.values()]
    target=OUT/"data/master.parquet";target.parent.mkdir();pq.write_table(pa.Table.from_pylist(master),target,compression="zstd")
    provenance=dict(path="data/master.parquet",sha256=sha(target))
    save(OUT/"protocols/A/pairs.json",pairs)
    save(OUT/"protocols/A/protocol.json",dict(name="A-covered-generator-local",splits=meta,master=provenance))
    save(OUT/"protocols/index.json",dict(protocols=["A"]))
    save(OUT/"build_complete.json",dict(master=provenance,custom_user_corpus=True))
    save(OUT/"verification.json",dict(at=now(),passed=True,master_sha256=sha(target),checks=["unique IDs","source-group split isolation","complete label pairs","source-disjoint training roles","canonical body split isolation"],scope="User-provided local splits; not a redistribution or reconstruction of the published experiment. Semantic duplicates and Base pretraining remain unaudited."))

def main():
    parser=argparse.ArgumentParser()
    for phase in ["train","development","calibration","test"]: parser.add_argument("--"+phase,type=Path,required=True)
    args=parser.parse_args();prepare({phase:rows(getattr(args,phase)) for phase in ["train","development","calibration","test"]})
    print("Local splits frozen. No data uploaded.")
if __name__=="__main__": main()
