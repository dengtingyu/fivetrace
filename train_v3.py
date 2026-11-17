"""Run the V3 recipe on a validated local prepared benchmark.

The original experiment's datasets and trained weights are not distributed.
The separate legacy train.py still replays V2.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from ai_detector.config import ROOT

CODE=Path(__file__).resolve().parent

def run(script,*args):
    subprocess.run([sys.executable,"-u",str(CODE/script),*args],cwd=CODE,check=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["prepare","references","head","features","fit","test","all"]);args=parser.parse_args()
    stages=["prepare","references","head","features","fit","test"] if args.stage=="all" else [args.stage]
    for stage in stages:
        if stage=="prepare":
            from framework_comparison_v1 import OUT
            from train_fivetrace import save
            OUT.mkdir(parents=True,exist_ok=True);save(OUT/"run_scope.json",dict(frameworks=["fivetrace"],protocol="A"))
            run("framework_v3_deberta.py","prepare")
            run("framework_v3_deberta.py","acquire")
        elif stage=="references":run("framework_comparison_v1.py","references")
        elif stage=="head":run("framework_v3_deberta.py","head-train")
        elif stage=="features":
            run("framework_v3_deberta.py","head-fit-scores");run("framework_v3_deberta.py","qwen-fit-features")
        elif stage=="fit":run("framework_v3_deberta.py","fusion-fit")
        else:
            run("framework_v3_deberta.py","head-test-scores");run("framework_v3_deberta.py","qwen-test-features");run("framework_v3_deberta.py","evaluate")
if __name__=="__main__":main()
