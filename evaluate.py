"""Evaluate a user-supplied local JSONL corpus; output aggregates only."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,roc_auc_score
from ai_detector.inference import Detector

def metrics(records):
    human=[r for r in records if r["label"]==0];ai=[r for r in records if r["label"]==1]
    y=np.array([r["label"] for r in records]);p=np.array([r["score"] for r in records]);both=bool(human and ai)
    return dict(records=len(records),human=len(human),ai=len(ai),
        human_FPR=sum(r["score"]>=.5 for r in human)/len(human) if human else None,
        AI_recall=sum(r["score"]>=.5 for r in ai)/len(ai) if ai else None,
        macro_F1=float(f1_score(y,p>=.5,average="macro")) if both else None,
        AUROC=float(roc_auc_score(y,p)) if both else None,
        human_paragraph_FPR=sum(r["AI_paragraphs"] for r in human)/sum(r["paragraphs"] for r in human) if human else None,
        AI_paragraph_recall=sum(r["AI_paragraphs"] for r in ai)/sum(r["paragraphs"] for r in ai) if ai else None)
def document_score(report):
    parts=report["paragraphs"]
    return sum(p["characters"]*p["ai_tendency_score"] for p in parts)/sum(p["characters"] for p in parts)
def main():
    parser=argparse.ArgumentParser(description="Each local JSONL record needs text and integer label (0 human, 1 AI).")
    parser.add_argument("file",type=Path);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--artifact-dir",type=Path);parser.add_argument("--device",default="cuda")
    args=parser.parse_args();detector=Detector(device=args.device,artifact_dir=args.artifact_dir);records=[];short=0
    try:
        for line in args.file.read_text("utf-8-sig").splitlines():
            if not line.strip():continue
            row=json.loads(line)
            if type(row.get("label")) is not int or row["label"] not in [0,1] or not isinstance(row.get("text"),str):
                raise ValueError("Expected text and integer label 0/1")
            if sum(not c.isspace() for c in row["text"])<100:short+=1;continue
            report=detector.analyze(row["text"])
            records.append(dict(label=row["label"],score=document_score(report),paragraphs=report["total_paragraphs"],
                AI_paragraphs=report["summary"]["ai"]["count"]))
    finally:detector.close()
    args.output.write_text(json.dumps(dict(threshold=.5,skipped_short=short,metrics=metrics(records)),indent=2)+"\n",encoding="utf-8")
if __name__=="__main__":main()
