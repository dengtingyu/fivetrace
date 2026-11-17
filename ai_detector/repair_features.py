"""Versioned, language-aware style features; V2 reference models are unchanged."""
import hashlib
import json
import math
import re
from collections import Counter, defaultdict

import numpy as np

from .adapted_features import ExpandedFeatureExtractor, ADAPTED_NUMERIC_NAMES
from .data import normalize
from .expanded_data import clean_text
from .features import STYLE_NAMES, style_features, english_route

STYLE_VERSION = 'zh-en-reliability-v21'
FEATURE_VERSION = 'base-ai-human-lora-language-repair-v21'
EXTRA_NAMES = ['english_route', 'log_scored_tokens', 'multiple_loss_blocks',
               'log_nonspace_characters', 'log_sentence_count']
REPAIR_NUMERIC_NAMES = ADAPTED_NUMERIC_NAMES + EXTRA_NAMES
EN_STOP = set('a an the of and to in is are was were be been being for on with as at by from or that this these those it its'.split())
EN_PRONOUNS = set('i me my mine myself we us our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves this that these those who whom whose which'.split())
WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")


def repair_style(text, scored_tokens):
    text = normalize(clean_text(text))
    en = english_route(text)
    sentences = [s for s in re.split(r'[。！？!?；;\n]+|(?<!\d)\.(?!\d)', text) if s.strip()]
    if not en:
        result = style_features(text)
    else:
        lengths = np.array([len(WORD.findall(s)) for s in sentences], dtype=float)
        lengths = lengths[lengths > 0]
        mean = float(lengths.mean()) if len(lengths) else 0.
        std = float(lengths.std()) if len(lengths) else 0.
        words = [m.group().lower().replace('’', "'") for m in WORD.finditer(text)]
        counts = Counter(w for w in words if w not in EN_STOP)
        total = max(1, sum(counts.values()))
        probs = np.array(list(counts.values()), dtype=float) / total
        positions = defaultdict(list)
        for i, word in enumerate(words):positions[word].append(i)
        gaps = [float(np.mean(np.diff(positions[w])-1)) for w, n in counts.most_common(10) if n >= 2]
        pronouns = [i for i, word in enumerate(words) if word in EN_PRONOUNS]
        pg = np.diff(pronouns)
        values = [mean, std, std/max(mean, 1.), len(counts)/total,
                  float(-(probs*np.log(probs)).sum()) if len(probs) else 0.,
                  sum(n for _, n in counts.most_common(10))/total,
                  float(np.mean(gaps)) if gaps else 0., float(np.std(gaps)) if gaps else 0.,
                  len(pronouns)/max(1, len(words)), float(np.std(pg)) if len(pg) else 0.]
        result = dict(zip(STYLE_NAMES, values))
    result.update(english_route=float(en), log_scored_tokens=math.log1p(scored_tokens),
                  multiple_loss_blocks=float(scored_tokens > 128),
                  log_nonspace_characters=math.log1p(sum(not c.isspace() for c in text)),
                  log_sentence_count=math.log1p(len(sentences)))
    return result


def repair_numeric_vector(stats):
    return np.array([stats[n] for n in REPAIR_NUMERIC_NAMES], dtype=float)


def recall_repair_score(original, repaired, cutoff, enabled=True):
    """Add a calibrated positive branch without reducing the V2 score."""
    if not enabled:return float(original)
    from scipy.special import expit, logit
    branch=float(expit(logit(np.clip(repaired,1e-12,1-1e-12))-logit(np.clip(cutoff,1e-12,1-1e-12))))
    return max(float(original),branch)


class RepairFeatureExtractor(ExpandedFeatureExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata.update(style_version=STYLE_VERSION, feature_version=FEATURE_VERSION)
        self.fingerprint = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()

    def extract_many(self, texts, need_lm=True, batch_size=8):
        values = super().extract_many(texts, need_lm=need_lm, batch_size=batch_size)
        return [(e, dict(s, **repair_style(t, int(s.get('scored_token_count', 0))))) for t, (e, s) in zip(texts, values)]

    def extract_single_reference(self, text, need_lm=True):
        e, s = super().extract_single_reference(text, need_lm=need_lm)
        return e, dict(s, **repair_style(text, int(s.get('scored_token_count', 0))))
