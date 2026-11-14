"""Train and compare expanded-data frozen and LoRA five-feature detectors."""
import gc
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np

from ai_detector.adapted_features import ADAPTED_NUMERIC_NAMES, ExpandedFeatureExtractor, adapted_numeric_vector
from ai_detector.config import ARTIFACT_DIR, ROOT
from ai_detector.expanded_data import HELDOUT_GENERATOR, PREPARED, PREPROCESS_VERSION, read_units, sample_units
from ai_detector.features import LM_NAMES, STYLE_NAMES
from ai_detector.training import binary_thresholds, evaluate, fit_linear

OUT = ARTIFACT_DIR / "five_feature" / "Qwen3-1.7B-LoRA-v2"
MODEL = ROOT / "models" / "Qwen3-1.7B-Base"


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def state(**values):
    dump(OUT / "training_state.json", {"updated_at": now(), **values})


def select_data():
    rows = read_units()
    seen = [r for r in rows if r["generator"] != HELDOUT_GENERATOR]
    unseen = [r for r in rows if r["generator"] in {HELDOUT_GENERATOR, "human"}]
    sets = {"head_train": sample_units(rows, "head_train", 4000),
            "fusion_train": sample_units(rows, "fusion_train", 1500),
            "validation": sample_units(rows, "validation", 800),
            "test_seen": sample_units(seen, "test", 1200),
            "test_unseen": sample_units(unseen, "test", 400)}
    for name, selected in sets.items():
        assert selected
        by_label = [Counter((r["source"], r["domain"], r["length_bin"]) for r in selected if r["label"] == label) for label in [0, 1]]
        assert by_label[0] == by_label[1], f"Source/length label confound: {name}"
    groups = {name: {r["group"] for r in selected} for name, selected in sets.items()}
    for a in ["head_train", "fusion_train", "validation"]:
        for b in sets:
            if a != b:
                assert not groups[a] & groups[b], (a, b)
    manifest = {name: {"samples": len(selected), "groups": len(groups[name]),
                      "labels": dict(Counter(str(r["label"]) for r in selected)),
                      "sources": dict(Counter(r["source"] for r in selected)),
                      "generators": dict(Counter(r["generator"] for r in selected)),
                      "sample_ids": [r["id"] for r in selected]}
                for name, selected in sets.items()}
    dump(OUT / "selected_manifest.json", manifest)
    return sets, manifest


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "lora_training.json").exists():
        raise RuntimeError("Run train_style_lora.py to completion first")
    start = time.monotonic()
    sets, manifest = select_data()
    feature_rows = list({r["id"]: r for name, selected in sets.items() if name != "head_train" for r in selected}.values())
    extractor = ExpandedFeatureExtractor(MODEL, device="cuda", adapter_dir=OUT / "adapters")
    print("Feature model:", json.dumps(extractor.metadata), flush=True)
    try:
        head_path = OUT / "head_features.npz"
        head_rows = sets["head_train"]
        if head_path.exists():
            saved = np.load(head_path, allow_pickle=False)
            assert saved["ids"].tolist() == [r["id"] for r in head_rows]
            head_matrix = saved["embeddings"]
        else:
            embeddings = []
            for i, row in enumerate(head_rows, start=1):
                embedding, _ = extractor.extract(row["text"], need_lm=False)
                embeddings.append(embedding)
                if i % 100 == 0 or i == len(head_rows):
                    state(stage="head_features", completed=i, total=len(head_rows), elapsed_seconds=time.monotonic()-start)
                    print(f"Head embeddings {i}/{len(head_rows)}; {time.monotonic()-start:.0f}s", flush=True)
            head_matrix = np.stack(embeddings)
            np.savez_compressed(head_path, ids=np.array([r["id"] for r in head_rows]), embeddings=head_matrix)
        head = fit_linear(head_matrix, np.array([r["label"] for r in head_rows]), .05)
        joblib.dump(head, OUT / "classification_head.joblib", compress=3)
        del head_matrix
        gc.collect()
        x_lora, statistics = [], []
        # Verify a diverse real batch against uncached individual calculation before batching the corpus.
        cache = extractor.base.cache
        extractor.base.cache = None
        try:
            examples = [row["text"] for row in feature_rows[:8]]
            batched = extractor.extract_many(examples)
            errors = []
            for text, (batch_embedding, batch_stats) in zip(examples, batched):
                single_embedding, single_stats = extractor.extract_single_reference(text)
                relative = float(np.linalg.norm(batch_embedding-single_embedding)/np.linalg.norm(single_embedding))
                numeric_error = float(np.max(np.abs(adapted_numeric_vector(batch_stats)-adapted_numeric_vector(single_stats))))
                assert relative < .01, ("batch embedding drift", relative)
                for variant in ["base", "ai", "human", "delta"]:
                    assert abs(batch_stats[f"{variant}_nll_mean"]-single_stats[f"{variant}_nll_mean"]) < .02
                    assert abs(batch_stats[f"{variant}_rank_top5"]-single_stats[f"{variant}_rank_top5"]) < .04
                errors.append({"relative_embedding_error": relative, "max_numeric_error": numeric_error})
            dump(OUT / "batch_numerical_check.json", {"batch_size": 8, "real_examples": len(examples), "errors": errors})
        finally:
            extractor.base.cache = cache
        for offset in range(0, len(feature_rows), 8):
            chunk = feature_rows[offset:offset+8]
            for embedding, stats in extractor.extract_many([row["text"] for row in chunk]):
                head_logit = float(head.decision_function(embedding[None, :])[0])
                x_lora.append(np.concatenate([[head_logit], adapted_numeric_vector(stats)]))
                statistics.append(stats)
            i = offset+len(chunk)
            if i % 200 == 0 or i == len(feature_rows):
                state(stage="fusion_features", completed=i, total=len(feature_rows), elapsed_seconds=time.monotonic()-start)
                print(f"Three-scorer full features {i}/{len(feature_rows)}; {time.monotonic()-start:.0f}s", flush=True)
        metadata = dict(extractor.metadata)
    finally:
        extractor.close()
    matrix = np.stack(x_lora)
    np.savez_compressed(OUT / "fusion_features.npz", ids=np.array([r["id"] for r in feature_rows]), features=matrix)
    labels = np.array([r["label"] for r in feature_rows])
    generators = np.array([r["generator"] for r in feature_rows])
    id_to_index = {r["id"]: i for i, r in enumerate(feature_rows)}
    masks = {name: np.array([id_to_index[r["id"]] for r in selected])
             for name, selected in sets.items() if name != "head_train"}
    matrices = {
        "expanded_lora_1_5": matrix,
        "expanded_frozen_1_5": np.stack([np.concatenate([[values[0]], [s[f"base_{n}"] for n in LM_NAMES], [s[n] for n in STYLE_NAMES]])
                                        for values, s in zip(x_lora, statistics)]),
        "expanded_1_4_5_only": np.stack([np.concatenate([[values[0]], [s[n] for n in STYLE_NAMES]])
                                       for values, s in zip(x_lora, statistics)])}
    feature_names = {"expanded_lora_1_5": ["text_classifier_logit"] + ADAPTED_NUMERIC_NAMES,
                     "expanded_frozen_1_5": ["text_classifier_logit"] + LM_NAMES + STYLE_NAMES,
                     "expanded_1_4_5_only": ["text_classifier_logit"] + STYLE_NAMES}
    models, scores, limits, validation = {}, {}, {}, {}
    for name, x in matrices.items():
        models[name] = fit_linear(x[masks["fusion_train"]], labels[masks["fusion_train"]], .1)
        scores[name] = models[name].predict_proba(x)[:, 1]
        limits[name] = dict(human_below=.5, ai_at_or_above=.5, policy="fixed_0.5")
        validation[name] = evaluate(labels[masks["validation"]], scores[name][masks["validation"]], limits[name], generators[masks["validation"]])
    eligible = ["expanded_lora_1_5"]
    selected = "expanded_lora_1_5"  # Replay the historically selected V2 architecture.
    dump(OUT / "selection_before_test.json", {"selected": selected, "validation": validation, "thresholds": limits, "at": now()})
    evaluations = {name: {role: evaluate(labels[mask], score[mask], limits[name], generators[mask])
                         for role, mask in masks.items() if role.startswith("test")}
                   for name, score in scores.items()}
    report = {"completed_at": now(), "elapsed_seconds": time.monotonic()-start, "selected_model": selected,
              "selected_data": manifest, "validation": validation, "thresholds": limits, "evaluation": evaluations,
              "data_manifest_sha256": hashlib.sha256((PREPARED / "manifest.json").read_bytes()).hexdigest(),
              "reference_model": metadata, "feature_names": feature_names,
              "lora_training": json.loads((OUT / "lora_training.json").read_text(encoding="utf-8")),
              "test_used_for_selection": False, "user_papers_used_for_training_or_selection": False}
    for name in eligible:
        directory = OUT / "candidates" / name
        directory.mkdir(parents=True, exist_ok=True)
        model_metadata = dict(metadata)
        if name == "expanded_frozen_1_5":
            model_metadata.pop("adapter_sha256")
            model_metadata.pop("adapter_path")
            model_metadata.update(scorer_variant="expanded_base", feature_version="expanded-base-nll-rank-v2")
        bundle = {"head": head, "fusion": models[name], "thresholds": limits[name], "extractor_metadata": model_metadata,
                  "feature_names": feature_names[name], "report": report,
                  "training_source_sha256": {"expanded_manifest": report["data_manifest_sha256"]}}
        joblib.dump(bundle, directory / "detector.joblib", compress=3)
        if name == selected:
            joblib.dump(bundle, OUT / "detector.joblib", compress=3)
    dump(OUT / "metrics.json", report)
    with (OUT / "test_predictions.jsonl").open("w", encoding="utf-8") as stream:
        chosen_ids = {r["id"] for role, values in sets.items() if role.startswith("test") for r in values}
        for i, row in enumerate(feature_rows):
            if row["id"] in chosen_ids:
                stream.write(json.dumps({k: row[k] for k in ["id", "group", "source", "domain", "generator", "label"]} |
                                        {name: float(score[i]) for name, score in scores.items()}, ensure_ascii=False) + "\n")
    render(report)
    state(stage="detector_training_completed", selected=selected)
    print(json.dumps({"selected": selected, "evaluation": evaluations}, ensure_ascii=False), flush=True)

def render(report):
    public={k:v for k,v in report.items() if k in ["selected_model","validation","evaluation","test_used_for_selection"]}
    (OUT/"training_summary.json").write_text(json.dumps(public,indent=2),encoding="utf-8")


if __name__=="__main__":main()
