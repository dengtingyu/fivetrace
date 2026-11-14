# Recent detector comparison / 近期检测器对比

Frozen public English external inputs; six detectors, seven disclosed modes. No V2 refitting or test-label threshold calibration. Dataset rows, IDs, per-record predictions and weights are not distributed.

## Checkpoints and native policies

| Detector | Release | Public source | Input and decision policy |
|---|---|---|---|
| Desklib v1.01 | 2025 | [model](https://huggingface.co/desklib/ai-text-detector-v1.01) | Original first 768 tokens, max-length padding, mean pool, sigmoid ≥ 0.5. |
| Tropa-mini | 2026 | [model](https://huggingface.co/wasitaigeneratedcom/tropa-mini-ai-text-detector) | Sentence-aligned ≤768-token chunks; token-length-weighted mean of 1−P(human), including AI/edited/humanized classes; ≥0.9755912190656111. |
| DeBERTa-ConPara | AACL-IJCNLP 2026 | [paper](https://arxiv.org/abs/2610.00883), [author code](https://github.com/SES-Lab-OTH/deberta-conpara), [checkpoint](https://huggingface.co/SES-Lab-OTH/deberta-conpara) | Author Unicode normalization and whitespace collapse; first 512 tokens; CLS + 2-layer head; AI−human logit >2.9423828125 (stored checkpoint threshold). |
| DetectRL-X XLM-R | 2026 | [official checkpoint](https://huggingface.co/WUJUNCHAO/DetectRL-X-XLM-RoBERTa-Detector-All) | Original first 512 tokens, AI-class softmax ≥0.5; reuse frozen scores. |
| MAGE | ACL 2024 | [paper](https://aclanthology.org/2024.acl-long.3/), [weights](https://huggingface.co/yaful/MAGE), [deployment](https://github.com/yafuly/MAGE) | Author preprocessing and class-0 raw logit >3.08583984375. Raw first-4096 softmax ≥0.5 retained as a separate sensitivity mode. |
| FiveTrace V2 | this release | [method](METHOD.md) | Production paragraph/sliding-window character-weighted document score ≥0.5; unchanged Chinese-trained V2. |

Tropa long-input chunking implements the model card's guidance locally; no unique official sentence splitter is specified. Published low-FPR calibration labels do not guarantee the same FPR on our domains. These checkpoints all support local inference; zero target-generator API dependence is a design benefit of our proxy references, not a claim unique among classifiers.

ConPara author source revision: `b6388ca8b95b75c057f5e6dfa3ee5b7c3f92eced`; checkpoint `a5f896f20f14259cf8a966fa4ae2012bd8d80164`; Base tokenizer/config `64a8c8eab3e352a784c658aef62be1662607476f`. Its author scoring and normalization are reused without modification; loading is restricted to safe torch.load(weights_only=True), strict state keys and verified local Base assets. Checkpoint/configuration/code hashes are in [aggregate metadata](../results/recent_comparison.json).

All new baselines use Transformers 4.57.6 and FP32 weights; V2 weights and its installed environment were preserved. ConPara uses an isolated protobuf runtime for its SentencePiece vocabulary. Measured NumPy versions are 2.5.x, above the author ConPara requirements file's <2 bound; the inference NumPy operations are unchanged, but the environment is not claimed identical to the paper's. Native policies have different context budgets; this is a deployment-policy comparison, not a controlled compute/latency comparison.

Before any new predictions, we replaced the candidate MELD public mirror with the official ConPara release because original-vs-mirror weight identity could not be independently verified. This source-quality selection amendment did not use detector scores. DivEye's official two-GPU demo pipeline exceeds this machine; the selected ParaTrace checkpoint is gated. No scores for those candidates are invented or compared.

## Matched primary conditions

| Generator / condition | Detector | Human / AI | Macro-F1 [95% CI] | Observed human FPR | AI recall | AUROC |
|---|---|---:|---:|---:|---:|---:|
| LLaDA / abstract_case | **FiveTrace V2** | 499/499 | 0.9920 [0.9860, 0.9970] | 0.80% | 99.20% | 0.9977 |
| LLaDA / abstract_case | Desklib v1.01 · 2025 | 499/499 | 0.9599 [0.9478, 0.9709] | 8.02% | 100.00% | 0.9999 |
| LLaDA / abstract_case | Tropa-mini · 2026 | 499/499 | 0.7291 [0.6993, 0.7543] | 0.00% | 49.30% | 0.9982 |
| LLaDA / abstract_case | DeBERTa-ConPara · AACL 2026 | 499/499 | 0.9870 [0.9800, 0.9940] | 2.20% | 99.60% | 0.9998 |
| LLaDA / abstract_case | DetectRL-X XLM-R · 2026 | 499/499 | 0.8778 [0.8577, 0.8948] | 11.62% | 87.17% | 0.9488 |
| LLaDA / abstract_case | MAGE · ACL 2024 · author | 499/499 | 0.9336 [0.9193, 0.9478] | 13.23% | 100.00% | 0.9975 |
| LLaDA / abstract_case | MAGE · raw sensitivity† | 499/499 | 0.9910 [0.9854, 0.9965] | 1.80% | 100.00% | 1.0000 |
| LLaDA / rephrese_case | **FiveTrace V2** | 497/497 | 0.7068 [0.6821, 0.7355] | 0.80% | 46.28% | 0.8558 |
| LLaDA / rephrese_case | Desklib v1.01 · 2025 | 497/497 | 0.7989 [0.7751, 0.8200] | 7.85% | 68.21% | 0.9255 |
| LLaDA / rephrese_case | Tropa-mini · 2026 | 497/497 | 0.3639 [0.3488, 0.3807] | 0.00% | 2.82% | 0.7398 |
| LLaDA / rephrese_case | DeBERTa-ConPara · AACL 2026 | 497/497 | 0.6967 [0.6685, 0.7213] | 2.21% | 45.67% | 0.9224 |
| LLaDA / rephrese_case | DetectRL-X XLM-R · 2026 | 497/497 | 0.8448 [0.8230, 0.8636] | 11.67% | 80.68% | 0.9188 |
| LLaDA / rephrese_case | MAGE · ACL 2024 · author | 497/497 | 0.6567 [0.6299, 0.6823] | 13.28% | 47.28% | 0.7807 |
| LLaDA / rephrese_case | MAGE · raw sensitivity† | 497/497 | 0.7952 [0.7715, 0.8171] | 1.81% | 62.17% | 0.9511 |
| LLaMA / abstract_case | **FiveTrace V2** | 499/499 | 0.9960 [0.9920, 0.9990] | 0.80% | 100.00% | 1.0000 |
| LLaMA / abstract_case | Desklib v1.01 · 2025 | 499/499 | 0.9599 [0.9478, 0.9709] | 8.02% | 100.00% | 1.0000 |
| LLaMA / abstract_case | Tropa-mini · 2026 | 499/499 | 0.9366 [0.9234, 0.9508] | 0.00% | 87.37% | 0.9993 |
| LLaMA / abstract_case | DeBERTa-ConPara · AACL 2026 | 499/499 | 0.9890 [0.9830, 0.9950] | 2.20% | 100.00% | 1.0000 |
| LLaMA / abstract_case | DetectRL-X XLM-R · 2026 | 499/499 | 0.9417 [0.9285, 0.9543] | 11.62% | 100.00% | 0.9970 |
| LLaMA / abstract_case | MAGE · ACL 2024 · author | 499/499 | 0.9316 [0.9174, 0.9467] | 13.23% | 99.60% | 0.9950 |
| LLaMA / abstract_case | MAGE · raw sensitivity† | 499/499 | 0.9900 [0.9834, 0.9960] | 1.80% | 99.80% | 0.9998 |

## Other public English sources

Every row uses the common supported-ID intersection across all seven modes in its source/condition. No AI-only F1, AUROC or human FPR is invented. More detailed source/model strata are in the aggregate JSON. Source-wise Wilson rate intervals are descriptive per-input bounds, not cluster-adjusted intervals for correlated sources; paired topic intervals are used for primary comparisons.

| Source / condition | Detector | Human / AI | Macro-F1 | Human FPR | AI recall | AUROC | Unsupported inputs excluded |
|---|---|---:|---:|---:|---:|---:|---:|
| AI-Detector-Benchmark-2026 / binary | **FiveTrace V2** | 500/382 | 0.9675 | 1.00% | 93.98% | 0.9963 | 0 |
| AI-Detector-Benchmark-2026 / binary | Desklib v1.01 · 2025 | 500/382 | 0.8424 | 25.60% | 97.12% | 0.9745 | 0 |
| AI-Detector-Benchmark-2026 / binary | Tropa-mini · 2026 | 500/382 | 0.9675 | 1.00% | 93.98% | 0.9965 | 0 |
| AI-Detector-Benchmark-2026 / binary | DeBERTa-ConPara · AACL 2026 | 500/382 | 0.7041 | 45.40% | 91.62% | 0.8874 | 0 |
| AI-Detector-Benchmark-2026 / binary | DetectRL-X XLM-R · 2026 | 500/382 | 0.9095 | 3.00% | 83.77% | 0.9665 | 0 |
| AI-Detector-Benchmark-2026 / binary | MAGE · ACL 2024 · author | 500/382 | 0.6036 | 56.40% | 83.51% | 0.7521 | 0 |
| AI-Detector-Benchmark-2026 / binary | MAGE · raw sensitivity† | 500/382 | 0.5570 | 67.00% | 89.53% | 0.7669 | 0 |
| Data-Prep-Bench / ai_only_qa | **FiveTrace V2** | 0/1168 | — | — | 13.18% | — | 0 |
| Data-Prep-Bench / ai_only_qa | Desklib v1.01 · 2025 | 0/1168 | — | — | 48.20% | — | 0 |
| Data-Prep-Bench / ai_only_qa | Tropa-mini · 2026 | 0/1168 | — | — | 6.42% | — | 0 |
| Data-Prep-Bench / ai_only_qa | DeBERTa-ConPara · AACL 2026 | 0/1168 | — | — | 22.35% | — | 0 |
| Data-Prep-Bench / ai_only_qa | DetectRL-X XLM-R · 2026 | 0/1168 | — | — | 11.82% | — | 0 |
| Data-Prep-Bench / ai_only_qa | MAGE · ACL 2024 · author | 0/1168 | — | — | 27.14% | — | 0 |
| Data-Prep-Bench / ai_only_qa | MAGE · raw sensitivity† | 0/1168 | — | — | 49.32% | — | 0 |
| Diffusion-Generated-Text / new_mechanism_ai_only | **FiveTrace V2** | 0/690 | — | — | 85.51% | — | 0 |
| Diffusion-Generated-Text / new_mechanism_ai_only | Desklib v1.01 · 2025 | 0/690 | — | — | 96.81% | — | 0 |
| Diffusion-Generated-Text / new_mechanism_ai_only | Tropa-mini · 2026 | 0/690 | — | — | 54.64% | — | 0 |
| Diffusion-Generated-Text / new_mechanism_ai_only | DeBERTa-ConPara · AACL 2026 | 0/690 | — | — | 91.88% | — | 0 |
| Diffusion-Generated-Text / new_mechanism_ai_only | DetectRL-X XLM-R · 2026 | 0/690 | — | — | 21.16% | — | 0 |
| Diffusion-Generated-Text / new_mechanism_ai_only | MAGE · ACL 2024 · author | 0/690 | — | — | 93.33% | — | 0 |
| Diffusion-Generated-Text / new_mechanism_ai_only | MAGE · raw sensitivity† | 0/690 | — | — | 94.93% | — | 0 |
| Diffusion-vs-AR-arXiv / unverified_pairing_ai_only | **FiveTrace V2** | 0/500 | — | — | 74.00% | — | 0 |
| Diffusion-vs-AR-arXiv / unverified_pairing_ai_only | Desklib v1.01 · 2025 | 0/500 | — | — | 94.20% | — | 0 |
| Diffusion-vs-AR-arXiv / unverified_pairing_ai_only | Tropa-mini · 2026 | 0/500 | — | — | 27.00% | — | 0 |
| Diffusion-vs-AR-arXiv / unverified_pairing_ai_only | DeBERTa-ConPara · AACL 2026 | 0/500 | — | — | 66.00% | — | 0 |
| Diffusion-vs-AR-arXiv / unverified_pairing_ai_only | DetectRL-X XLM-R · 2026 | 0/500 | — | — | 94.80% | — | 0 |
| Diffusion-vs-AR-arXiv / unverified_pairing_ai_only | MAGE · ACL 2024 · author | 0/500 | — | — | 72.20% | — | 0 |
| Diffusion-vs-AR-arXiv / unverified_pairing_ai_only | MAGE · raw sensitivity† | 0/500 | — | — | 77.40% | — | 0 |
| EvalDetectBench-WritingBench / binary | **FiveTrace V2** | 0/29 | — | — | 89.66% | — | 6 |
| EvalDetectBench-WritingBench / binary | Desklib v1.01 · 2025 | 0/29 | — | — | 93.10% | — | 6 |
| EvalDetectBench-WritingBench / binary | Tropa-mini · 2026 | 0/29 | — | — | 100.00% | — | 6 |
| EvalDetectBench-WritingBench / binary | DeBERTa-ConPara · AACL 2026 | 0/29 | — | — | 86.21% | — | 6 |
| EvalDetectBench-WritingBench / binary | DetectRL-X XLM-R · 2026 | 0/29 | — | — | 65.52% | — | 6 |
| EvalDetectBench-WritingBench / binary | MAGE · ACL 2024 · author | 0/29 | — | — | 62.07% | — | 6 |
| EvalDetectBench-WritingBench / binary | MAGE · raw sensitivity† | 0/29 | — | — | 82.76% | — | 6 |
| TextSight-2026-09 / binary | **FiveTrace V2** | 643/48 | 0.8594 | 3.11% | 83.33% | 0.9743 | 0 |
| TextSight-2026-09 / binary | Desklib v1.01 · 2025 | 643/48 | 0.7906 | 8.55% | 97.92% | 0.9846 | 0 |
| TextSight-2026-09 / binary | Tropa-mini · 2026 | 643/48 | 0.9140 | 2.80% | 100.00% | 0.9990 | 0 |
| TextSight-2026-09 / binary | DeBERTa-ConPara · AACL 2026 | 643/48 | 0.7283 | 7.62% | 68.75% | 0.8984 | 0 |
| TextSight-2026-09 / binary | DetectRL-X XLM-R · 2026 | 643/48 | 0.8646 | 3.89% | 91.67% | 0.9903 | 0 |
| TextSight-2026-09 / binary | MAGE · ACL 2024 · author | 643/48 | 0.6068 | 12.75% | 47.92% | 0.6742 | 0 |
| TextSight-2026-09 / binary | MAGE · raw sensitivity† | 643/48 | 0.5340 | 25.19% | 56.25% | 0.7158 | 0 |
| ToolBenchmarks-2026-10 / binary | **FiveTrace V2** | 150/600 | 0.3318 | 0.00% | 16.83% | 0.8825 | 0 |
| ToolBenchmarks-2026-10 / binary | Desklib v1.01 · 2025 | 150/600 | 0.6142 | 6.00% | 56.67% | 0.8718 | 0 |
| ToolBenchmarks-2026-10 / binary | Tropa-mini · 2026 | 150/600 | 0.6168 | 0.00% | 54.83% | 0.8984 | 0 |
| ToolBenchmarks-2026-10 / binary | DeBERTa-ConPara · AACL 2026 | 150/600 | 0.5829 | 15.33% | 55.67% | 0.7821 | 0 |
| ToolBenchmarks-2026-10 / binary | DetectRL-X XLM-R · 2026 | 150/600 | 0.4011 | 4.67% | 26.33% | 0.7857 | 0 |
| ToolBenchmarks-2026-10 / binary | MAGE · ACL 2024 · author | 150/600 | 0.5161 | 18.00% | 46.67% | 0.6779 | 0 |
| ToolBenchmarks-2026-10 / binary | MAGE · raw sensitivity† | 150/600 | 0.5181 | 32.00% | 52.67% | 0.6284 | 0 |
| ToolBenchmarks-2026-10 / edited | **FiveTrace V2** | 0/150 | — | — | 5.33% | — | 0 |
| ToolBenchmarks-2026-10 / edited | Desklib v1.01 · 2025 | 0/150 | — | — | 33.33% | — | 0 |
| ToolBenchmarks-2026-10 / edited | Tropa-mini · 2026 | 0/150 | — | — | 22.67% | — | 0 |
| ToolBenchmarks-2026-10 / edited | DeBERTa-ConPara · AACL 2026 | 0/150 | — | — | 26.00% | — | 0 |
| ToolBenchmarks-2026-10 / edited | DetectRL-X XLM-R · 2026 | 0/150 | — | — | 8.67% | — | 0 |
| ToolBenchmarks-2026-10 / edited | MAGE · ACL 2024 · author | 0/150 | — | — | 24.67% | — | 0 |
| ToolBenchmarks-2026-10 / edited | MAGE · raw sensitivity† | 0/150 | — | — | 35.33% | — | 0 |

## Paired differences from V2

Joint bootstrap: 500 resamples of deduplicated human topic families, preserving human/AI pairs and cross-model dependence. Positive Macro-F1 differences favor V2; an interval spanning zero does not establish a clear separation. Rate intervals use Wilson 95%.

| Condition | V2 minus comparator | Macro-F1 difference | Paired 95% CI |
|---|---|---:|---:|
| LLaDA / abstract_case | v2_minus_desklib | +0.0321 | [+0.0221, +0.0442] |
| LLaDA / abstract_case | v2_minus_tropa | +0.2629 | [+0.2367, +0.2947] |
| LLaDA / abstract_case | v2_minus_conpara | +0.0050 | [-0.0030, +0.0126] |
| LLaDA / abstract_case | v2_minus_xlmr | +0.1142 | [+0.0962, +0.1363] |
| LLaDA / abstract_case | v2_minus_mage_author | +0.0584 | [+0.0432, +0.0716] |
| LLaDA / abstract_case | v2_minus_mage_raw | +0.0010 | [-0.0080, +0.0090] |
| LLaDA / rephrese_case | v2_minus_desklib | -0.0921 | [-0.1232, -0.0579] |
| LLaDA / rephrese_case | v2_minus_tropa | +0.3430 | [+0.3115, +0.3752] |
| LLaDA / rephrese_case | v2_minus_conpara | +0.0101 | [-0.0212, +0.0423] |
| LLaDA / rephrese_case | v2_minus_xlmr | -0.1380 | [-0.1682, -0.1039] |
| LLaDA / rephrese_case | v2_minus_mage_author | +0.0502 | [+0.0147, +0.0853] |
| LLaDA / rephrese_case | v2_minus_mage_raw | -0.0883 | [-0.1221, -0.0556] |
| LLaMA / abstract_case | v2_minus_desklib | +0.0361 | [+0.0256, +0.0482] |
| LLaMA / abstract_case | v2_minus_tropa | +0.0594 | [+0.0442, +0.0736] |
| LLaMA / abstract_case | v2_minus_conpara | +0.0070 | [+0.0000, +0.0140] |
| LLaMA / abstract_case | v2_minus_xlmr | +0.0543 | [+0.0412, +0.0685] |
| LLaMA / abstract_case | v2_minus_mage_author | +0.0644 | [+0.0492, +0.0786] |
| LLaMA / abstract_case | v2_minus_mage_raw | +0.0060 | [-0.0020, +0.0140] |

## Coverage and source limits

| Mode | Eligible | Scored | Input cap/truncation observed | Below author word-count recommendation |
|---|---:|---:|---:|---:|
| v2 | 6861 | 6861 | not recorded | — |
| desklib | 6861 | 6861 | 216 | — |
| tropa | 6861 | 6861 | 0 | 823 |
| conpara | 6861 | 6861 | 481 | 1257 |
| xlmr | 6861 | 6861 | not recorded | — |
| mage_author | 6861 | 6855 | not recorded | — |
| mage_raw | 6861 | 6861 | not recorded | — |

- Original case collection, labels and source-only quality exclusions remain frozen. The misindexed LLaMA rephrasing CSV is excluded from topic-matched primary results. The earlier score-independent resolution of shared human-abstract identities is retained.
- Existing family exclusions audit known V2/V3 fitting/development/validation and known XLM-R training overlap. Complete new-comparator training manifests and foundation-model pretraining are not audited. No claim of universal decontamination or unseen data for every baseline.
- Old historical tests influenced prior research; reused human controls and correlated source conditions cannot be added as independent authors. Some old records are pre-cut units rather than complete articles.
- New baselines are English-only in this expansion. Original Chinese results remain in [the original report](RESULTS.md). V2 English performance is transfer, not bilingual fine-tuning.
- AI-only and edited-only collections report recall separately. Source-provided generation labels and model attribution are not independently proven.
- Our new-decoder transfer experiment uses [the academic source](https://github.com/ismailtrm/ceng_404/tree/7786b2bd1bfd080c3f16343164c64a3d7cf39e9f) and [additional diffusion outputs](https://huggingface.co/datasets/paoche11/diffusion-generated-text/tree/d1d8c8593a6b7db5c446f497fd000a0ce74dcded); input format and genre differences prevent a causal claim about generation architecture.

## Reproducing local baseline inference

Use a separate environment to preserve the V2 runtime. `baseline_eval.py --prepare` explicitly downloads and hash-verifies pinned model assets and native source; it performs no inference or code execution. The evaluation command checks those hashes again, reads local English JSONL (`text`, integer `label` 0/1, optional `language`) and writes aggregate metrics only. Text is never uploaded. ConPara executes its verified local author scorer/normalizer; its .pt is loaded with weights_only=True.

```bash
python -m venv .venv-eval
# Activate it; install a compatible PyTorch CUDA build first.
pip install -r requirements-eval.txt
python baseline_eval.py --detector desklib --model-dir local_models/Desklib-v1.01 --prepare
python baseline_eval.py --detector desklib --model-dir local_models/Desklib-v1.01 --input local_english.jsonl --output local_aggregate.json
```

The repository does not distribute the frozen texts or record-ID lists. This entry point reproduces native inference on locally supplied data; it does not claim a one-command replay of privately retained benchmark rows.
