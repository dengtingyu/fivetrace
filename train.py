"""Replay the selected V2 recipe locally; never publishes or uploads."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser(description="Local V2 training; Base and dataset download is separate.")
    parser.add_argument("stage",choices=["prepare","lora","fit","all"])
    args=parser.parse_args()
    for stage in (["prepare","lora","fit"] if args.stage=="all" else [args.stage]):
        command={"prepare":["-m","ai_detector.expanded_data"],"lora":[str(ROOT/"train_lora.py")],"fit":[str(ROOT/"train_detector.py")]}[stage]
        subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
if __name__=="__main__":main()
