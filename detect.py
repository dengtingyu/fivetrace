"""Analyze a local UTF-8 text file; no network requests."""
import argparse
import json
from pathlib import Path
from ai_detector.inference import Detector

def main():
    parser=argparse.ArgumentParser(description='FiveTrace local paragraph detector')
    parser.add_argument('file',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--artifact-dir',type=Path)
    args=parser.parse_args()
    detector=Detector(device=args.device,artifact_dir=args.artifact_dir)
    try: report=detector.analyze(args.file.read_text(encoding='utf-8-sig'))
    finally: detector.close()
    rendered=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if args.output: args.output.write_text(rendered,encoding='utf-8')
    else: print(rendered)

if __name__=='__main__': main()
