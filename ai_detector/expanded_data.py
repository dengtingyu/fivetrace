"""Audited multi-source data preparation with source-group isolation."""
import csv
import hashlib
import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .config import ROOT, SEED
from .paragraphs import document_windows, prepare_detection_paragraphs

RAW = ROOT / "detector_research" / "expanded_raw"
PREPARED = ROOT / "detector_research" / "expanded_prepared"
PHASES = [("lora_train", .35), ("lora_dev", .40), ("head_train", .65),
          ("fusion_train", .80), ("validation", .90), ("test", 1.)]
HELDOUT_GENERATOR = "claude-3.5-haiku"
PREPROCESS_VERSION = "strip-markdown-abstract-v1"


def clean_text(text):
    text = unicodedata.normalize("NFKC", str(text)).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u200b", "").replace("**", "").replace("__", "")
    text = re.sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+", "", text)
    text = re.sub(r"^\s*(?:摘\s*要|abstract)\s*[:：]\s*", "", text, flags=re.I)
    return text.strip()


def canonical(text):
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).lower())


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            lo, hi = sorted([a, b])
            self.parent[hi] = lo


def phase(group):
    value = int(digest(f"{SEED}:{group}")[:12], 16)/16**12
    return next(name for name, edge in PHASES if value < edge)


def choose_unit(text, seed_key):
    text = clean_text(text)
    if sum(not c.isspace() for c in text) < 100:
        return None
    if any(token in text[:160] for token in ["作为一个人工智能", "作为AI", "作为一名人工智能", "作为一个AI", "作为语言模型", "作为一个语言模型"]):
        return None
    _, _, paragraphs = prepare_detection_paragraphs(text)
    candidates = []
    for p in paragraphs:
        for a, b in document_windows(p["text"], max_chars=600, min_tail=100):
            part = p["text"][a:b].strip()
            if sum(not c.isspace() for c in part) >= 100:
                candidates.append(part)
    if not candidates:
        return None
    # Same grouped source uses the same random index, independently of its label.
    rng = random.Random(int(digest(seed_key)[:16], 16))
    return candidates[rng.randrange(len(candidates))]


def load_documents():
    docs, audit = [], Counter()
    fields = {"composition": "composition_title", "film_review": "film_title", "news": "news_title",
              "paper": "paper_title", "question_answer": "question_title"}
    for domain, title_field in fields.items():
        files = sorted((RAW / "cred").glob(f"CReD_{domain}_*.csv"))
        if len(files) != 9:
            raise RuntimeError(f"C-ReD files incomplete: {domain}: {len(files)}")
        human_path = RAW / "cred" / f"CReD_{domain}_human.csv"
        with human_path.open(encoding="utf-8-sig", newline="") as stream:
            originals = {r["id"]: r for r in csv.DictReader(stream)}
        for path in files:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    generator = row["attribution"]
                    label = int(generator != "human")
                    if row["label"] != ("0" if label else "1"):
                        raise ValueError("C-ReD source label mismatch")
                    origin = originals[row["original_id"] if label else row["id"]]
                    if canonical(row[title_field]) != canonical(origin[title_field]):
                        raise ValueError("C-ReD origin/title mismatch")
                    group_key = ("question:" if domain == "question_answer" else f"cred:{domain}:") + canonical(origin[title_field])
                    docs.append({"id": f"cred:{domain}:{generator}:{row['id']}", "source": "C-ReD", "domain": domain,
                                 "generator": generator, "label": label, "group_key": group_key,
                                 "original_text": origin["text"], "text": row["text"]})
                    audit["C-ReD_raw"] += 1
    cudrt_path = RAW / "nlpcc" / "train_data.json"
    raw = cudrt_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != "d07059d8af0c7ba86629a8ceba3f09669d79f13356f09d0853861b53a8d5c01f":
        raise RuntimeError("CUDRT SHA-256 mismatch")
    for row in json.loads(raw.decode("utf-8-sig")):
        identifier = row["ID"]
        human = str(row["HWT"])
        generated = str(row["LGT"])
        generator = identifier.split("-")[0].lower()
        domain = "news" if "-news-" in identifier.lower() else "academic"
        group_key = "original:" + digest(canonical(human))
        # LGT is prefix-conditioned continuation. The verbatim human prefix is NOT AI training text.
        prefix = os.path.commonprefix([human, generated])
        ai_tail = generated[len(prefix):] if len(prefix) >= 20 else generated
        audit["CUDRT_raw_pairs"] += 1
        audit["CUDRT_human_prefix_removed"] += int(len(prefix) >= 20)
        for label, text, gen in [(0, human, "human"), (1, ai_tail, generator)]:
            docs.append({"id": f"cudrt:{identifier}:{label}", "source": "CUDRT", "domain": domain,
                         "generator": gen, "label": label, "group_key": group_key, "original_text": human,
                         "human_prefix_removed_characters": len(prefix) if label and len(prefix) >= 20 else 0,
                         "text": text})
        audit["CUDRT_HLT_excluded"] += 1
    for i, line in enumerate((RAW / "hc3" / "all.jsonl").read_text(encoding="utf-8-sig").splitlines()):
        row = json.loads(line)
        question = str(row["question"])
        group_key = "question:" + canonical(question)
        for field, label, generator in [("human_answers", 0, "human"), ("chatgpt_answers", 1, "chatgpt-early")]:
            for j, text in enumerate(row.get(field, [])):
                docs.append({"id": f"hc3:{i}:{label}:{j}", "source": "HC3", "domain": str(row.get("source", "qa")),
                             "generator": generator, "label": label, "group_key": group_key,
                             "original_text": "", "text": text})
                audit["HC3_raw"] += 1
    return docs, audit


def prepare():
    docs, audit = load_documents()
    uf = UnionFind()
    originals, prefixes, full_texts = {}, {}, {}
    for doc in docs:
        group = digest(doc["group_key"])
        doc["group"] = group
        uf.find(group)
        if doc["original_text"]:
            key = canonical(doc["original_text"])
            uf.union(group, originals.setdefault(digest(key), group))
            if len(key) >= 160:
                uf.union(group, prefixes.setdefault(digest(key[:160]), group))
        key = digest(canonical(clean_text(doc["text"])))
        uf.union(group, full_texts.setdefault(key, group))
    units = []
    for doc in docs:
        unit = choose_unit(doc["text"], uf.find(doc["group"]))
        if unit is None:
            audit["too_short_or_self_disclosure_excluded"] += 1
            continue
        units.append({k: v for k, v in doc.items() if k not in {"original_text", "group_key", "text"}} | {"text": unit})
    hashes = defaultdict(list)
    for row in units:
        hashes[digest(canonical(row["text"]))].append(row)
    output = []
    for key, rows in hashes.items():
        if len({r["label"] for r in rows}) > 1:
            audit["contradictory_label_units_excluded"] += len(rows)
            continue
        for row in rows[1:]:
            uf.union(rows[0]["group"], row["group"])
        # Keep one exact-normalized unit with provenance, not multiple votes.
        row = min(rows, key=lambda r: r["id"])
        row["duplicate_ids"] = [r["id"] for r in rows if r["id"] != row["id"]]
        output.append(row)
        audit["duplicate_units_removed"] += len(rows)-1
    for row in output:
        row["group"] = uf.find(row["group"])
        row["split"] = phase(row["group"])
        row["characters"] = sum(not c.isspace() for c in row["text"])
        row["length_bin"] = min(row["characters"]//200, 4)
    output = [r for r in output if r["generator"] != HELDOUT_GENERATOR or r["split"] == "test"]
    by_text = {}
    by_group = {}
    for row in output:
        assert by_text.setdefault(digest(canonical(row["text"])), row["split"]) == row["split"]
        assert by_group.setdefault(row["group"], row["split"]) == row["split"]
    PREPARED.mkdir(parents=True, exist_ok=True)
    with (PREPARED / "units.jsonl").open("w", encoding="utf-8") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {name: {"units": len(subset), "groups": len({r["group"] for r in subset}),
                      "labels": dict(Counter("AI" if r["label"] else "human" for r in subset)),
                      "sources": dict(Counter(r["source"] for r in subset)),
                      "generators": dict(Counter(r["generator"] for r in subset))}
               for name, _ in PHASES for subset in [[r for r in output if r["split"] == name]]}
    result = {"seed": SEED, "preprocess_version": PREPROCESS_VERSION, "audit": dict(audit),
              "summary": summary, "heldout_generator": HELDOUT_GENERATOR,
              "group_rules": ["same title/question and all generators together", "same normalized original text together",
                              "same first 160 normalized original characters together", "exact normalized unit duplicates together"],
              "near_duplicate_limit": "No full semantic near-duplicate clustering; beyond grouping, original-prefix and exact checks may miss paraphrased sources.",
              "excluded": ["CUDRT human verbatim prefix removed from AI positives", "CUDRT HLT not used as pure-generated training labels",
                           "user papers excluded from all model fitting", "official phase-2 test excluded from all model fitting"]}
    (PREPARED / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def sample_units(rows, split, per_class, seed=SEED):
    """Match source/domain/length counts across labels, then spread AI generators."""
    rng = random.Random(seed)
    pools = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        if row["split"] == split:
            pools[(row["source"], row["domain"], row["length_bin"])][row["label"]][row["generator"]].append(row)
    for labels in pools.values():
        for generators in labels.values():
            for pool in generators.values():
                rng.shuffle(pool)
    keys = [key for key, labels in pools.items() if labels.get(0) and labels.get(1)]
    keys.sort()
    rng.shuffle(keys)
    generator_keys = {(key, label): sorted(pools[key][label]) for key in keys for label in [0, 1]}
    positions = Counter()
    chosen = []
    count = 0
    while keys and count < per_class:
        next_keys = []
        for key in keys:
            if count >= per_class:
                break
            for label in [0, 1]:
                generators = generator_keys[(key, label)]
                position = positions[(key, label)] % len(generators)
                name = generators[position]
                chosen.append(pools[key][label][name].pop())
                if not pools[key][label][name]:
                    generators.pop(position)
                positions[(key, label)] += 1
            count += 1
            if all(generator_keys[(key, label)] for label in [0, 1]):
                next_keys.append(key)
        keys = next_keys
    rng.shuffle(chosen)
    return chosen


def read_units():
    return [json.loads(line) for line in (PREPARED / "units.jsonl").read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
