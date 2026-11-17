**English** | [简体中文](README.zh-CN.md)

# FiveTrace V3

### Learn from generated text. Detect locally. Zero target-model API calls.

[Method](docs/METHOD.md) · [External comparison](docs/V3_EXTERNAL.md) · [Protocol](docs/PROTOCOL.md) · [Local training](docs/REPRODUCING_V3.md) · [V2 history](docs/V2_RELEASE.md)

![MIT license](https://img.shields.io/badge/License-MIT-blue)
![Version 3.0.0](https://img.shields.io/badge/Version-3.0.0-blue)
![Inference API calls 0](https://img.shields.io/badge/Target_API_calls-0-brightgreen)
![Source and aggregates](https://img.shields.io/badge/Release-source_%26_aggregates-lightgrey)

**FiveTrace V3 combines a fine-tuned mDeBERTa classifier with Qwen corpus-proxy loss, token rank, sentence and lexical evidence.** Detection does not deploy or query the original GPT, Claude or other generator: it runs one local Qwen backbone, two LoRA adapters and a separate mDeBERTa classifier.

Across **6 external sources and 8 conditions**, V3 has the strictly highest Macro-F1 point estimate in **6/8** conditions. We show every condition, including failures, with AI recall and human false positives. This is an official-checkpoint comparison, not a same-training-data or full-leaderboard ranking.

![FiveTrace V3 versus official open detectors](assets/benchmark_v3.svg)

**Exclude known fitting ancestry, then compare identical inputs.** Each abstract condition retains 118 topic pairs after removing V3 train/development/calibration group matches. Original full populations remain separate regression results; the other five columns keep their frozen subsets. Related sources/generators and historical evaluations were seen during development, so these are not blind or unseen-generator claims.

Each cell shows **Macro-F1 ↑**, then **AI recall ↑ (human FPR ↓)**. Bold marks the highest point estimate per column.

| Detector | TextSight | AI-Detector-Benchmark | ToolBenchmarks raw | COLING 2025 test slice | Intel peer review |
|---|---:|---:|---:|---:|---:|
| **FiveTrace V3** | **0.9433**<br>97.92% (1.56%) | **0.9746**<br>96.34% (1.60%) | 0.4352<br>29.67% (0.67%) | **0.8455**<br>76.00% (6.67%) | 0.8487<br>92.80% (18.00%) |
| Desklib v1.01 (2025) | 0.7906<br>97.92% (8.55%) | 0.8424<br>97.12% (25.60%) | 0.6142<br>56.67% (6.00%) | 0.7274<br>85.67% (39.33%) | 0.7996<br>100.00% (52.00%) |
| Tropa-mini (2026) | 0.9140<br>100.00% (2.80%) | 0.9675<br>93.98% (1.00%) | **0.6168**<br>54.83% (0.00%) | 0.8367<br>82.67% (15.33%) | **0.9324**<br>98.40% (14.00%) |
| ConPara (AACL 2026) | 0.7283<br>68.75% (7.62%) | 0.7041<br>91.62% (45.40%) | 0.5829<br>55.67% (15.33%) | 0.8127<br>92.67% (29.67%) | 0.9187<br>98.40% (18.00%) |
| DetectRL-X XLM-R (2026) | 0.8646<br>91.67% (3.89%) | 0.9095<br>83.77% (3.00%) | 0.4011<br>26.33% (4.67%) | 0.7647<br>60.00% (5.67%) | 0.6877<br>73.60% (14.00%) |
| MAGE author (ACL 2024) | 0.6068<br>47.92% (12.75%) | 0.6036<br>83.51% (56.40%) | 0.5161<br>46.67% (18.00%) | 0.7409<br>79.67% (31.33%) | 0.7220<br>87.20% (38.00%) |


Each cell shows **Macro-F1 ↑**, then **AI recall ↑ (human FPR ↓)**. Bold marks the highest point estimate per column.

| Detector | LLaDA abstract | LLaDA rephrase | LLaMA abstract |
|---|---:|---:|---:|
| **FiveTrace V3** | **1.0000**<br>100.00% (0.00%) | **0.9490**<br>89.83% (0.00%) | **1.0000**<br>100.00% (0.00%) |
| Desklib v1.01 (2025) | 0.9746<br>100.00% (5.08%) | 0.8010<br>66.10% (5.08%) | 0.9746<br>100.00% (5.08%) |
| Tropa-mini (2026) | 0.7126<br>46.61% (0.00%) | 0.3519<br>1.69% (0.00%) | 0.9146<br>83.05% (0.00%) |
| ConPara (AACL 2026) | 0.9788<br>100.00% (4.24%) | 0.7037<br>48.31% (4.24%) | 0.9788<br>100.00% (4.24%) |
| DetectRL-X XLM-R (2026) | 0.8686<br>86.44% (12.71%) | 0.8431<br>81.36% (12.71%) | 0.9362<br>100.00% (12.71%) |
| MAGE author (ACL 2024) | 0.9190<br>100.00% (16.10%) | 0.6450<br>47.46% (16.10%) | 0.9148<br>99.15% (16.10%) |


| Condition | Human / AI |
|---|---:|
| TextSight | 643 / 48 |
| AI-Detector-Benchmark | 500 / 382 |
| ToolBenchmarks raw | 150 / 600 |
| LLaDA abstract | 118 / 118 |
| LLaDA rephrase | 118 / 118 |
| LLaMA abstract | 118 / 118 |
| COLING 2025 test slice | 300 / 300 |
| Intel peer review | 50 / 250 |

**Comparison scope:** weights and native thresholds are fixed; external labels do not tune parameters. Original training sources and context budgets differ. Abstract columns reuse human controls. COLING contains 75 humanized machine texts among its 300 official negatives; its FPR is official-negative-class FPR.

[Full metrics, AUROC, paired confidence intervals and overlap counts](docs/V3_EXTERNAL.md). A highest point estimate alone does not establish a statistically significant difference.

## What changed from V2

| Evidence | V3 implementation |
|---|---|
| 1 | Full-parameter mDeBERTa-v3-base fine-tuning; continuous AI-minus-human logit |
| 2 | Base / AI-LoRA / human-LoRA causal loss and branch differences |
| 3 | Exact full-vocabulary token midrank statistics and branch differences |
| 4 | Language-aware sentence-length statistics |
| 5 | Language-aware lexical/repetition/pronoun statistics and reliability inputs |

Reference adapters were freshly trained on both Chinese and English. Classifier, reference and fusion fitting use disjoint source groups. Chinese character / English token windows aggregate into paragraphs. The user-facing AI ratio counts AI-labelled detection paragraphs; benchmark document scores instead use character-weighted probabilities.

## Separate result: covered-generator protocol A

This separate experiment is not the official-checkpoint ranking above. On 20,000 independent test documents, balanced by label/language at threshold 0.5, V3 achieves **99.45% Macro-F1**, **99.72% AI recall** and **0.81% human FPR**. The same feature-1 classifier alone scores 98.42%, 99.35% and 2.50%, respectively. Equal-family Macro-F1 is 95.69%; small GLM strata remain weak. Protocol B has not been evaluated. These results use repaired FP32 classifier inference with unchanged task weights, fusion and threshold.

[Protocol-A aggregates](results/v3_protocol_a.json) · [Training scope and phase timings](results/v3_training.json)

## Run locally

This **source-and-aggregate release includes no trained weights or datasets**. Train with your own validated local corpus, or supply a compatible locally produced V3 artifact. It is not a download-and-run pretrained detector.

```bash
pip install -r requirements.txt
python prepare_sources.py --models-only
python prepare_v3_data.py --train local/train.jsonl --development local/dev.jsonl --calibration local/calibration.jsonl --test local/test.jsonl
python train_v3.py all
python detect.py local/article.txt --output local/report.json
python evaluate.py local/test.jsonl --output local/aggregates.json
python -m unittest discover -s tests -v
```

Install a suitable CUDA PyTorch build first. See [required split fields and training stages](docs/REPRODUCING_V3.md). `FIVETRACE_HOME` selects an optional local runtime root. V3 needs a CUDA device; the classifier is offloaded before Qwen rank computation. Training and reference extraction were run on an 8 GB RTX 5060 Laptop, but this is not a universal memory guarantee.

`train.py` remains the explicitly legacy V2 trainer. Select a compatible V2 artifact with `--artifact-dir` for legacy inference. [V2 documentation](docs/V2_RELEASE.md) preserves its original measurements.


## Interpretation

Scores are model judgments, not proof of authorship or the true amount of AI assistance. Human false positives matter, particularly for academic, short or new-domain text. Groups 2/3 measure local corpus proxies, not the original generator probabilities. Representative corpora affect coverage. No controlled percentage latency or monetary advantage over target generators has been measured.

Please also cite the upstream models, detectors and corpora listed in the full report and V2 history.
