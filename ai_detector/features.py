import hashlib
import json
import logging
import math
import re
import sqlite3
from collections import Counter, defaultdict

import numpy as np

from .config import ARTIFACT_DIR, CONTEXT, FEATURE_VERSION, LOSS_BLOCK, MODEL_DIR, STRIDE
from .data import normalize

LM_NAMES = ["nll_mean", "nll_block_std", "nll_block_max", "rank_top5", "rank_6_50",
            "rank_51_500", "rank_gt500", "log_rank_mean"]
STYLE_NAMES = ["sentence_length_mean", "sentence_length_std", "sentence_length_cv", "word_ttr",
               "word_entropy", "top10_word_share", "frequent_word_gap_mean", "frequent_word_gap_std",
               "pronoun_frequency", "pronoun_gap_std"]
STOP = set("的 了 和 与 及 或 是 在 为 对 将 被 把 由 于 而 也 都 就 很 更 最 有 无 并 等 以 a an the of and to in is are".split())
PRONOUNS = set("其 这种 此类 此种 以上 该 这些 此 这一 这类 他们 它们 其中".split())
_segmenter = None


def style_features(text):
    global _segmenter
    import jieba
    if _segmenter is None:
        jieba.setLogLevel(logging.ERROR)
        _segmenter = jieba.Tokenizer()
    sentences = [s for s in re.split(r"[。！？!?；;\n]+|(?<!\d)\.(?!\d)", text) if s.strip()]
    lengths = np.array([sum(not c.isspace() for c in s) for s in sentences], dtype=float)
    mean = float(lengths.mean()) if len(lengths) else 0.
    std = float(lengths.std()) if len(lengths) else 0.
    tokens = [(word.lower(), begin, end) for word, begin, end in _segmenter.tokenize(text, HMM=False)
              if any(c.isalnum() for c in word)]
    words = [w for w, _, _ in tokens]
    count = Counter(w for w in words if w not in STOP)
    positions = defaultdict(list)
    for word, begin, end in tokens:
        positions[word].append((begin, end))
    freq_total = max(1, sum(count.values()))
    probabilities = np.array(list(count.values()), dtype=float) / freq_total
    entropy = float(-(probabilities * np.log(probabilities)).sum()) if len(probabilities) else 0.
    gap_means = []
    for word, frequency in count.most_common(10):
        spans = positions[word]
        if frequency >= 2:
            gap_means.append(float(np.mean([spans[i][0] - spans[i-1][1] for i in range(1, len(spans))])))
    pronoun_positions = [begin for word, begin, _ in tokens if word in PRONOUNS]
    pronoun_gaps = np.diff(pronoun_positions)
    values = [mean, std, std / max(mean, 1.), len(count) / freq_total, entropy,
              sum(n for _, n in count.most_common(10)) / freq_total,
              float(np.mean(gap_means)) if gap_means else 0.,
              float(np.std(gap_means)) if gap_means else 0.,
              len(pronoun_positions) / max(1, len(words)),
              float(np.std(pronoun_gaps)) if len(pronoun_gaps) else 0.]
    return dict(zip(STYLE_NAMES, values))


def token_windows(token_count, context=CONTEXT, stride=STRIDE):
    if not 1 <= stride < context:
        raise ValueError("Require 1 <= stride < context")
    # Token zero has no preceding text and is intentionally not scored.
    for start in range(1, token_count, stride):
        end = min(token_count, start + stride)
        context_start = max(0, start - (context - stride))
        yield start, end, context_start


def summarize_lm(losses, ranks, block_size=LOSS_BLOCK):
    losses = np.asarray(losses, dtype=float)
    ranks = np.asarray(ranks, dtype=float)
    if not len(losses) or len(losses) != len(ranks):
        raise ValueError("Need aligned, nonempty losses and ranks")
    blocks = np.array([losses[start:start+block_size].mean() for start in range(0, len(losses), block_size)])
    values = [float(losses.mean()), float(blocks.std()), float(blocks.max()),
              float(np.mean(ranks <= 5)), float(np.mean((ranks > 5) & (ranks <= 50))),
              float(np.mean((ranks > 50) & (ranks <= 500))), float(np.mean(ranks > 500)),
              float(np.log(ranks).mean())]
    return dict(zip(LM_NAMES, values))


def token_statistics(logits, targets):
    """Exact next-token NLL and midrank in the full vocabulary; no top-k approximation."""
    import torch
    logits = logits.float()
    actual = logits.gather(1, targets[:, None])
    losses = torch.logsumexp(logits, dim=-1) - actual[:, 0]
    greater = (logits > actual).sum(dim=-1)
    tied = (logits == actual).sum(dim=-1)
    ranks = 1. + greater.float() + (tied.float() - 1.) / 2.
    return losses, ranks


class FeatureExtractor:
    def __init__(self, model_dir=MODEL_DIR, device="auto", cache=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model_dir = model_dir
        if not (model_dir / "model.safetensors").exists():
            raise FileNotFoundError("语言模型尚未下载完成，请先运行 prepare_model.py")
        source = json.loads((model_dir / "source.json").read_text(encoding="utf-8"))
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        torch.set_num_threads(min(8, torch.get_num_threads()))
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.metadata = {"model_id": source["model_id"], "revision": source["revision"],
                         "feature_version": FEATURE_VERSION, "context_tokens": CONTEXT,
                         "stride_tokens": STRIDE, "loss_block_tokens": LOSS_BLOCK,
                         "dtype": str(dtype), "quantization": "dynamic_int8" if self.device == "cpu" else "none",
                         "rank_ties": "midrank", "first_token_scored": False}
        self.fingerprint = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False,
            torch_dtype=dtype, attn_implementation="sdpa",
        ).to(self.device).eval()
        if self.device == "cpu":
            # Keep CPU inference practical without a multi-GB CUDA installation.
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True,
            )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.cache = None
        if cache:
            ARTIFACT_DIR.mkdir(exist_ok=True)
            self.cache = sqlite3.connect(ARTIFACT_DIR / "feature_cache.sqlite", check_same_thread=False)
            self.cache.execute("PRAGMA journal_mode=WAL")
            self.cache.execute("CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY, embedding BLOB, stats TEXT)")

    def extract(self, raw_text, need_lm=True):
        text = normalize(raw_text)
        key = hashlib.sha256((self.fingerprint + ("full:" if need_lm else "embedding:") + text).encode("utf-8")).hexdigest()
        if self.cache:
            row = self.cache.execute("SELECT embedding, stats FROM features WHERE key=?", (key,)).fetchone()
            if row:
                return np.frombuffer(row[0], dtype=np.float32).copy(), json.loads(row[1])
        torch = self.torch
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 2:
            raise ValueError("文本至少需要两个 token")
        losses, ranks, embeddings, weights = [], [], [], []
        with torch.inference_mode():
            for start, end, context_start in token_windows(len(ids)):
                tokens = torch.tensor([ids[context_start:end]], device=self.device)
                hidden = self.model.model(input_ids=tokens, use_cache=False).last_hidden_state[0]
                embeddings.append(hidden[-1].float().cpu().numpy())
                weights.append(end - start)
                for begin in range(start, end, 64) if need_lm else []:
                    stop = min(end, begin + 64)
                    predictors = hidden[begin-context_start-1:stop-context_start-1]
                    logits = self.model.lm_head(predictors)
                    targets = tokens[0, begin-context_start:stop-context_start]
                    nll, rank = token_statistics(logits, targets)
                    losses.extend(nll.cpu().tolist())
                    ranks.extend(rank.cpu().tolist())
                    del logits
        if need_lm and len(losses) != len(ids) - 1:
            raise RuntimeError("Token coverage mismatch")
        embedding = np.average(np.stack(embeddings), axis=0, weights=weights).astype(np.float32)
        base_stats = summarize_lm(losses, ranks) if need_lm else dict.fromkeys(LM_NAMES, 0.)
        stats = dict(base_stats, **style_features(text),
                     token_count=len(ids), scored_token_count=len(losses), model_windows=len(weights))
        if not np.isfinite(embedding).all() or not all(math.isfinite(v) for v in stats.values()):
            raise ValueError("Non-finite features")
        if self.cache:
            self.cache.execute("INSERT OR REPLACE INTO features VALUES (?, ?, ?)",
                               (key, embedding.tobytes(), json.dumps(stats)))
            self.cache.commit()
        return embedding, stats

    def close(self):
        if self.cache:
            self.cache.close()

    def extract_many(self, raw_texts, need_lm=True, batch_size=8):
        """Batch independent context windows; retain the exact teacher-forcing coverage."""
        results, pending, tasks = [None]*len(raw_texts), {}, []
        for index, raw_text in enumerate(raw_texts):
            text = normalize(raw_text)
            key = hashlib.sha256((self.fingerprint + ("full:" if need_lm else "embedding:") + text).encode("utf-8")).hexdigest()
            row = self.cache.execute("SELECT embedding,stats FROM features WHERE key=?", (key,)).fetchone() if self.cache else None
            if row:
                results[index] = np.frombuffer(row[0], dtype=np.float32).copy(), json.loads(row[1])
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 2:
                raise ValueError("文本至少需要两个 token")
            pending[index] = {"text": text, "key": key, "ids": ids, "embeddings": [], "weights": [],
                              "losses": np.full(len(ids)-1, np.nan), "ranks": np.full(len(ids)-1, np.nan)}
            for start, end, context_start in token_windows(len(ids)):
                tasks.append((index, start, end, context_start, ids[context_start:end]))
        torch = self.torch
        with torch.inference_mode():
            for offset in range(0, len(tasks), batch_size):
                batch = tasks[offset:offset+batch_size]
                length = max(len(task[4]) for task in batch)
                ids = torch.full((len(batch), length), self.tokenizer.eos_token_id, device=self.device, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for row_index, task in enumerate(batch):
                    ids[row_index, :len(task[4])] = torch.tensor(task[4], device=self.device)
                    mask[row_index, :len(task[4])] = 1
                hidden = self.model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
                for row_index, (index, start, end, context_start, context_ids) in enumerate(batch):
                    record = pending[index]
                    record["embeddings"].append(hidden[row_index, len(context_ids)-1].float().cpu().numpy())
                    record["weights"].append(end-start)
                    for begin in range(start, end, 256) if need_lm else []:
                        stop = min(end, begin+256)
                        logits = self.model.lm_head(hidden[row_index, begin-context_start-1:stop-context_start-1])
                        targets = ids[row_index, begin-context_start:stop-context_start]
                        losses, ranks = token_statistics(logits, targets)
                        record["losses"][begin-1:stop-1] = losses.cpu().numpy()
                        record["ranks"][begin-1:stop-1] = ranks.cpu().numpy()
                        del logits
                del hidden
        for index, record in pending.items():
            embedding = np.average(np.stack(record["embeddings"]), axis=0, weights=record["weights"]).astype(np.float32)
            if need_lm and (not np.isfinite(record["losses"]).all() or not np.isfinite(record["ranks"]).all()):
                raise RuntimeError("Batch token coverage mismatch")
            lm_stats = summarize_lm(record["losses"], record["ranks"]) if need_lm else dict.fromkeys(LM_NAMES, 0.)
            stats = dict(lm_stats, **style_features(record["text"]), token_count=len(record["ids"]),
                         scored_token_count=len(record["ids"])-1 if need_lm else 0, model_windows=len(record["weights"]))
            if not np.isfinite(embedding).all() or not all(math.isfinite(v) for v in stats.values()):
                raise ValueError("Non-finite batched features")
            if self.cache:
                self.cache.execute("INSERT OR REPLACE INTO features VALUES (?, ?, ?)",
                                   (record["key"], embedding.tobytes(), json.dumps(stats)))
            results[index] = embedding, stats
        if self.cache and pending:
            self.cache.commit()
        return results


def numeric_vector(stats):
    return np.array([stats[name] for name in LM_NAMES + STYLE_NAMES], dtype=float)
