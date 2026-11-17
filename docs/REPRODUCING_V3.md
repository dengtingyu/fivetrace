# Local V3 training and inference

This repository releases the implementation and aggregate evidence. It includes no datasets, exact experimental record-ID lists or trained weights. Training on a custom corpus does not reproduce the published numbers automatically.

1. Install a CUDA-compatible torch build and requirements.txt. Use a separate environment for baseline_eval.py as described in the historical evaluation docs.
2. `python prepare_sources.py --models-only` downloads/verifies the pinned Qwen3-1.7B-Base. This option does not download the old V2 training corpora. V3 preparation separately downloads the pinned original microsoft/mdeberta-v3-base (revision a0484667b22365f84929a935b5e50a51f71f159d), not a previously trained detector.
3. Prepare local train/development/calibration/test JSONL files with the fields below. Split original topics/documents before assigning roles. Semantically related variants need the same group_id. Do not use your intended external test corpus in any fitting/tuning role.
4. `python prepare_v3_data.py --train local/train.jsonl --development local/dev.jsonl --calibration local/calibration.jsonl --test local/test.jsonl` validates and freezes local manifests and a text parquet under the ignored benchmarks directory. It rejects an existing destination. Canonical exact body and source-group split checks do not replace an upstream semantic/lineage audit.
5. `python train_v3.py all` runs pinned preparation, fresh class-specific reference LoRAs, full-parameter classifier training, cached fitting features, fusion and test. Individual stages: prepare, references, head, features, fit, test. Completed weights/features are reused only with matching provenance. If interrupted, rerun the same stage to resume saved checkpoints/caches; do not change frozen data or recipes underneath a run.
6. `python detect.py local/article.txt --output local/report.json` uses V3 by default. `python evaluate.py local/test.jsonl --output local/aggregates.json` outputs aggregates only. These do not publish or upload.

| Field | Requirement |
|---|---|
| id | Unique local document identity across all four splits |
| text | Body string; at least 100 nonspace characters |
| label | Integer 0 human / 1 AI |
| group_id | Original-topic/source group; exactly one human/AI pair per group for the training/evaluation scaffold |
| language | zh or en; pair members have the same language |
| domain | Shared domain within each pair |
| generator | Declared generator for AI; human for a human control |
| family | Generator family for AI; human for a human control |
| component_role | TRAIN only: reference, classifier or fusion; every group has one role |

Each phase and training role requires both Chinese and English. Four phase group sets must be disjoint; classifier, reference and fusion groups must also be disjoint. The CLI uses supplied groups and does not infer missing semantic ancestry. Original-source audit, label validity and representative corpus design remain the user's responsibility. Native mDeBERTa SentencePiece tokenization uses relative filesystem addressing on Windows to handle non-ASCII workspace paths, avoiding a dependency upgrade or fast-tokenizer conversion.

`FIVETRACE_HOME` optionally selects the local runtime root. All commands use that root's models/data/artifacts. Default V3 artifact: artifacts/framework_comparison_v1_20261115/A/fivetrace_v3_mdeberta. The date-bearing path is an implementation namespace, not a claim that custom data comes from the original experiment. Do not load untrusted sklearn joblib bundles. CUDA inference is the validated reference path; CPU dynamic-quantized LoRA inference is not supported.

Legacy V2: `python train.py all` runs its historical recipe; pass its artifact explicitly to detect.py/evaluate.py with --artifact-dir. Legacy results remain in V2_RELEASE.md and the pre-V3 release/commit history. The project's local research application's active model is not changed by publishing this repository.
