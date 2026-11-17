"""Shared atomic local IO helpers; no publication or training side effects."""
import hashlib,json,os,time
from pathlib import Path
from datetime import datetime,timezone

def now():return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Windows readers/antivirus can briefly deny atomic replacement. Preserve
    # the previous complete JSON and retry instead of aborting GPU inference.
    for attempt in range(20):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt==19:raise
            time.sleep(min(.05*(attempt+1),.25))


def jsonl(path,values):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='\n') as f:
        for r in values:f.write(json.dumps(r,ensure_ascii=False)+'\n')
