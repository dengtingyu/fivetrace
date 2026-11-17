"""Frozen embeddings plus base/AI-LoRA/human-LoRA likelihood and rank features."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import ROOT
from .expanded_data import PREPROCESS_VERSION, clean_text
from .features import FeatureExtractor, LM_NAMES, STYLE_NAMES

ADAPTED_LM_NAMES = [f"{variant}_{name}" for variant in ["base", "ai", "human", "delta"] for name in LM_NAMES]
ADAPTED_NUMERIC_NAMES = ADAPTED_LM_NAMES + STYLE_NAMES


def adapter_hash(path):
    path = Path(path)
    digest = hashlib.sha256()
    for filename in ["adapter_config.json", "adapter_model.safetensors"]:
        digest.update(filename.encode())
        digest.update((path / filename).read_bytes())
    return digest.hexdigest()


def adapted_numeric_vector(stats):
    return np.array([stats[name] for name in ADAPTED_NUMERIC_NAMES], dtype=float)


class ExpandedFeatureExtractor:
    def __init__(self, model_dir, device="auto", adapter_dir=None):
        self.base = FeatureExtractor(model_dir=model_dir, device=device)
        self.device = self.base.device
        self.adapter_dir = Path(adapter_dir) if adapter_dir else None
        self.adapters = None
        self.reference_names = {"ai": "ai", "human": "human"}
        self.base_fingerprint = self.base.fingerprint
        self.metadata = dict(self.base.metadata, preprocess_version=PREPROCESS_VERSION,
                             document_window_characters=600,
                             scorer_variant="multi_lora" if adapter_dir else "expanded_base",
                             feature_version="base-ai-human-lora-nll-rank-v2" if adapter_dir else "expanded-base-nll-rank-v2")
        self.variant_fingerprints = {}
        if adapter_dir:
            if self.device == "cpu":
                raise ValueError("LoRA 参考模型使用 GPU FP16；当前版本不支持 CPU 动态量化适配器。")
            from peft import PeftModel
            paths = {name: self.adapter_dir / name for name in ["ai", "human"]}
            hashes = {name: adapter_hash(path) for name, path in paths.items()}
            self.metadata["adapter_sha256"] = hashes
            self.metadata["adapter_path"] = self.adapter_dir.relative_to(ROOT).as_posix()
            self.adapters = PeftModel.from_pretrained(self.base.model, str(paths["ai"]),
                                adapter_name="ai", is_trainable=False)
            self.adapters.load_adapter(str(paths["human"]), adapter_name="human", is_trainable=False)
            self.adapters.eval()
            # Keep the original causal-LM object so the existing extractor still reaches .model and .lm_head.
            self.base.model = self.adapters.get_base_model()
            for name in paths:
                self.variant_fingerprints[name] = hashlib.sha256(json.dumps(
                    {**self.base.metadata, "active_adapter": name, "adapter_sha256": hashes[name]}, sort_keys=True).encode()).hexdigest()
        self.fingerprint = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()

    def extract(self, raw_text, need_lm=True):
        return self.extract_many([raw_text], need_lm=need_lm)[0]

    def extract_single_reference(self, raw_text, need_lm=True):
        text = clean_text(raw_text)
        self.base.fingerprint = self.base_fingerprint
        if self.adapters is None:
            return self.base.extract(text, need_lm=need_lm)
        with self.adapters.disable_adapter():
            embedding, original = self.base.extract(text, need_lm=need_lm)
        if not need_lm:
            return embedding, original
        values = {"base": original}
        try:
            for name in ["ai", "human"]:
                self.adapters.set_adapter(self.reference_names[name])
                self.adapters.eval()
                self.base.fingerprint = self.variant_fingerprints[name]
                _, values[name] = self.base.extract(text, need_lm=True)
        finally:
            self.base.fingerprint = self.base_fingerprint
        # Keep legacy unprefixed fields as aliases for the active AI-style scorer in the UI.
        stats = dict(values["ai"])
        for variant, data in values.items():
            stats.update({f"{variant}_{name}": data[name] for name in LM_NAMES})
        stats.update({f"delta_{name}": values["human"][name]-values["ai"][name] for name in LM_NAMES})
        return embedding, stats

    def close(self):
        self.base.close()

    def extract_many(self, raw_texts, need_lm=True, batch_size=8):
        texts = [clean_text(text) for text in raw_texts]
        self.base.fingerprint = self.base_fingerprint
        if self.adapters is None:
            return self.base.extract_many(texts, need_lm=need_lm, batch_size=batch_size)
        with self.adapters.disable_adapter():
            original = self.base.extract_many(texts, need_lm=need_lm, batch_size=batch_size)
        if not need_lm:
            return original
        variants = {}
        try:
            for name in ["ai", "human"]:
                self.adapters.set_adapter(self.reference_names[name])
                self.adapters.eval()
                self.base.fingerprint = self.variant_fingerprints[name]
                variants[name] = self.base.extract_many(texts, need_lm=True, batch_size=batch_size)
        finally:
            self.base.fingerprint = self.base_fingerprint
        results = []
        for index, (embedding, base_stats) in enumerate(original):
            ai, human = variants["ai"][index][1], variants["human"][index][1]
            stats = dict(ai)
            for name, values in [("base", base_stats), ("ai", ai), ("human", human)]:
                stats.update({f"{name}_{key}": values[key] for key in LM_NAMES})
            stats.update({f"delta_{key}": human[key]-ai[key] for key in LM_NAMES})
            results.append((embedding, stats))
        return results
