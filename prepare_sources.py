"""Download fixed public corpora and one Base model; no uploads or dataset code."""
import hashlib
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from ai_detector.config import ROOT

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")
def digest(path):
    with path.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()
def download(item):
    path=ROOT/"detector_research/expanded_raw"/item["path"]
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size==item["bytes"] and digest(path)==item["sha256"]:return
    partial=path.with_suffix(path.suffix+".partial")
    request=urllib.request.Request(item["url"],headers={"User-Agent":"FiveTrace-V2"})
    with urllib.request.urlopen(request,timeout=120) as response,partial.open("wb") as stream:
        while chunk:=response.read(1024*1024):stream.write(chunk)
    if partial.stat().st_size!=item["bytes"] or digest(partial)!=item["sha256"]:
        raise RuntimeError("Pinned file checksum mismatch: "+item["path"])
    partial.replace(path)
def main():
    from huggingface_hub import snapshot_download
    manifest=json.loads((Path(__file__).resolve().parent/"results/sources.json").read_text("utf-8"))
    with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(download,manifest["datasets"]))
    base=manifest["base"];directory=ROOT/"models/Qwen3-1.7B-Base"
    snapshot_download(base["model_id"],revision=base["revision"],local_dir=directory,
        allow_patterns=["*.json","*.safetensors","*.txt","*.jinja","LICENSE"])
    for name,expected in base["weight_sha256"].items():
        if digest(directory/name)!=expected:raise RuntimeError("Base checksum mismatch")
    (directory/"source.json").write_text(json.dumps(base,indent=2)+"\n",encoding="utf-8")
    print("Pinned public inputs verified locally. Nothing uploaded.")
if __name__=="__main__":main()
