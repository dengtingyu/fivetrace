import json
from pathlib import Path

import joblib
import numpy as np

from .config import ARTIFACT_DIR, ROOT, STANDARD_ARTIFACT_DIR, STANDARD_THRESHOLD
from .features import FeatureExtractor, numeric_vector
from .paragraphs import analyze_paragraphs, document_windows, verdict


class Detector:
    def __init__(self, device="auto", artifact_dir=None):
        if artifact_dir is None and (ARTIFACT_DIR / "active_standard.json").exists():
            active = json.loads((ARTIFACT_DIR / "active_standard.json").read_text(encoding="utf-8"))
            selected = (ROOT / active["artifact_dir"]).resolve()
            if not selected.is_relative_to(ARTIFACT_DIR.resolve()):
                raise ValueError("标准模型路径必须位于 artifacts 目录内。")
            artifact_dir = selected
        artifact_dir = Path(artifact_dir) if artifact_dir is not None else (STANDARD_ARTIFACT_DIR if (STANDARD_ARTIFACT_DIR / "detector.joblib").exists() else ARTIFACT_DIR)
        self.artifact_dir = artifact_dir
        path = artifact_dir / "detector.joblib"
        if not path.exists():
            raise FileNotFoundError("尚无已训练模型，请先运行 train.cmd")
        self.bundle = joblib.load(path)
        if artifact_dir.resolve() == STANDARD_ARTIFACT_DIR.resolve():
            # The original validation cutoff remains in the frozen weight file.
            self.bundle['thresholds'] = dict(human_below=STANDARD_THRESHOLD,
                                           ai_at_or_above=STANDARD_THRESHOLD,
                                           policy='user_selected_fixed_0.5')
        training = self.bundle["extractor_metadata"]
        if device == "auto" and training["quantization"] == "dynamic_int8":
            device = "cpu"
        model_dir = ROOT / "models" / training["model_id"].split("/")[-1]
        variant = training.get("scorer_variant")
        if variant in {"expanded_base", "multi_lora"}:
            from .adapted_features import ExpandedFeatureExtractor, adapted_numeric_vector
            adapter_dir = ROOT / training["adapter_path"] if variant == "multi_lora" else None
            extractor_class = ExpandedFeatureExtractor
            if training.get("style_version") == "zh-en-pronouns-v1":
                raise ValueError("This V2 release requires compatible V2 feature metadata; V3 is a separate experiment.")
            self.extractor = extractor_class(model_dir=model_dir, device=device, adapter_dir=adapter_dir)
            self.vectorize = adapted_numeric_vector if variant == "multi_lora" else numeric_vector
        else:
            self.extractor = FeatureExtractor(model_dir=model_dir, device=device)
            self.vectorize = numeric_vector
        for field in ("revision", "feature_version", "context_tokens", "stride_tokens", "loss_block_tokens", "rank_ties", "quantization"):
            if self.extractor.metadata[field] != training[field]:
                raise ValueError(f"模型或特征配置与训练时不同：{field}；请使用与训练相同的设备类型，或重新训练。")
        for field in ("scorer_variant", "preprocess_version", "adapter_sha256", "document_window_characters", "style_version"):
            if field in training and self.extractor.metadata.get(field) != training[field]:
                raise ValueError(f"微调模型或预处理配置与训练时不同：{field}")

    def analyze(self, raw_text, progress=None):
        report = analyze_paragraphs(raw_text, self._predict_window, self.bundle["thresholds"],
                                    self.extractor.metadata, progress=progress,
                                    window_chars=self.extractor.metadata.get("document_window_characters", 1000))
        if self.extractor.metadata.get("scorer_variant"):
            report["notes"][-1] = "模型使用扩充后的学术、作文、新闻、问答、影评等公开语料训练；段落占比仍是模型分类统计。"
        if self.extractor.metadata.get("scorer_variant") == "multi_lora":
            report["notes"].append("第 2、3 项综合基础模型、AI-LoRA、人工-LoRA 及两适配分支的差值；未加前缀的损失与排名字段展示 AI-LoRA 分支。")
        return report

    def _predict_window(self, part):
        embedding, stats = self.extractor.extract(part)
        logits = float(self.bundle["head"].decision_function(embedding[None, :])[0])
        x = np.concatenate([[logits], self.vectorize(stats)])[None, :]
        return {"ai_tendency_score": float(self.bundle["fusion"].predict_proba(x)[0, 1]),
                "text_classifier_score": float(self.bundle["head"].predict_proba(embedding[None, :])[0, 1]),
                "features": stats}

    def close(self):
        self.extractor.close()


def read_text(path):
    path = Path(path)
    if path.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("第一版支持 .txt 和 .md，请将正文另存为纯文本。")
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文件编码，请另存为 UTF-8 文本。")
