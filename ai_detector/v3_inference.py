"""V3 inference: independently trained mDeBERTa plus Qwen reference features."""
import gc
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit

from .config import ROOT
from .language_windows import language_document_structure, language_window_spans
from .paragraphs import analyze_paragraphs
from .repair_features import RepairFeatureExtractor, repair_numeric_vector


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def raw_head_key(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class V3Detector:
    def __init__(self, device='cuda', artifact_dir=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not str(device).startswith('cuda'):
            raise ValueError('V3 reference adapters require a CUDA device; CPU quantization is not the validated recipe.')
        self.device = device
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.artifact_dir = Path(artifact_dir) if artifact_dir else ROOT/'artifacts/framework_comparison_v1_20261115/A/fivetrace_v3_mdeberta'
        # Only load a trusted, locally produced sklearn bundle.
        self.bundle = joblib.load(self.artifact_dir/'detector.joblib')
        if self.bundle.get('variant') != 'FiveTrace-V3-mDeBERTa':
            raise ValueError('Expected a V3 mDeBERTa artifact, not a historical V2/5-epoch checkpoint.')
        head_path = ROOT/self.bundle['head_checkpoint']
        if file_sha(head_path) != self.bundle['head_checkpoint_sha256']:
            raise ValueError('Selected classifier weights changed.')
        base = ROOT/'models'/self.bundle['first_feature']['directory']
        source = json.loads((base/'source.json').read_text(encoding='utf-8'))
        if source['revision'] != self.bundle['first_feature']['revision']:
            raise ValueError('Classifier tokenizer/configuration revision differs from fitting.')
        for name, entry in source['files'].items():
            if name.endswith(('.json', '.model')) and file_sha(base/name) != entry['sha256']:
                raise ValueError('Classifier configuration or vocabulary changed: '+name)
        self.head = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2, local_files_only=True).float().eval()
        self.head.load_state_dict(torch.load(head_path, map_location='cpu', weights_only=True))
        for parameter in self.head.parameters():
            parameter.requires_grad_(False)
        try:
            address = os.path.relpath(base, Path.cwd())
        except ValueError:
            address = str(base)
        self.head_tokenizer = AutoTokenizer.from_pretrained(address, local_files_only=True, use_fast=False)
        if not callable(self.head_tokenizer):
            raise RuntimeError('Failed to load the native SentencePiece tokenizer.')
        self.model_dir = ROOT/'models'/self.bundle['extractor_metadata']['model_id'].split('/')[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self.extractor = None

    def head_logits(self, texts, batch_size=8):
        import torch
        self.head.to(self.device); values = []
        try:
            with torch.inference_mode():
                for offset in range(0, len(texts), batch_size):
                    batch = self.head_tokenizer(texts[offset:offset+batch_size], padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
                    batch.pop('token_type_ids', None)
                    logits = self.head(**batch).logits.float()
                    values.extend((logits[:, 1]-logits[:, 0]).cpu().tolist())
                    del logits, batch
        finally:
            # Keep the classifier off GPU during the vocabulary-rank passes.
            self.head.cpu(); gc.collect(); torch.cuda.empty_cache()
        if not np.isfinite(values).all():
            raise RuntimeError('Nonfinite classifier logits.')
        return values

    def reference(self):
        if self.extractor is None:
            self.extractor = RepairFeatureExtractor(self.model_dir, device=self.device, adapter_dir=ROOT/self.bundle['Qwen_reference_directory'])
            if self.extractor.metadata != self.bundle['extractor_metadata']:
                self.extractor.close(); self.extractor = None
                raise ValueError('Reference models, adapters, precision or feature configuration differ from fitting.')
        return self.extractor

    def score_windows(self, texts, batch_size=4):
        logits = self.head_logits(texts)
        features = self.reference().extract_many(texts, batch_size=batch_size)
        x = np.stack([np.concatenate([[logit], repair_numeric_vector(stats)]) for logit, (_, stats) in zip(logits, features)])
        scores = self.bundle['fusion'].predict_proba(x)[:, 1]
        return [dict(ai_tendency_score=float(score), text_classifier_score=float(expit(logit)), features=stats)
                for score, logit, (_, stats) in zip(scores, logits, features)]

    def analyze(self, raw_text, progress=None):
        geometry = language_document_structure(raw_text, self.tokenizer)
        texts = list(dict.fromkeys(w['text'] for p in geometry for w in p['windows']))
        scored = dict(zip(texts, self.score_windows(texts)))
        report = analyze_paragraphs(raw_text, lambda text: scored[text],
            dict(human_below=.5, ai_at_or_above=.5, policy='fixed_0.5'),
            dict(self.bundle['extractor_metadata'], classifier='mDeBERTa-v3-base', variant=self.bundle['variant']),
            progress=progress, window_chars=600, window_splitter=lambda text: language_window_spans(text, self.tokenizer))
        report['notes'][-1] = 'V3 uses separately trained mDeBERTa classification, Qwen AI/human proxy loss/rank, and sentence/lexical evidence. Ratios count classified paragraphs.'
        return report

    def close(self):
        import torch
        if self.extractor is not None:
            self.extractor.close(); self.extractor = None
        if getattr(self, 'head', None) is not None:
            self.head.cpu(); self.head = None
        gc.collect(); torch.cuda.empty_cache()
