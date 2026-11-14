import os
from pathlib import Path

ROOT = Path(os.environ.get("FIVETRACE_HOME", Path(__file__).resolve().parent.parent)).resolve()
DATA_DIR = ROOT / "detector_research" / "data"
MODEL_DIR = ROOT / "models" / "Qwen3-1.7B-Base"
ARTIFACT_DIR = ROOT / "artifacts"
STANDARD_ARTIFACT_DIR = ARTIFACT_DIR / "five_feature" / "Qwen3-1.7B-LoRA-v2"
MODEL_ID = "Qwen/Qwen3-1.7B-Base"
STANDARD_THRESHOLD = 0.5
SEED = 20261110
CONTEXT = 512
STRIDE = 384
LOSS_BLOCK = 128
FEATURE_VERSION = "qwen-hidden-last-nll-rank-v1"
