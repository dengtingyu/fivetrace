# V2 external evaluation

Frozen V2 weights, cutoff 0.5. Public-source aggregate results only. Rows in the same family or condition may share controls; do not sum them as independent authors. No claim of a complete leaderboard or universal superiority.

## Matched academic generation and rephrasing

Shared full-text processing is controlled separately from native context limits and the MAGE author recipe. MAGE raw softmax=0.5 is not its published deployment boundary.

| Generator | Task | Detector | Mode | Families | Human FPR | AI recall | Macro-F1 | AUROC |
|---|---|---|---|---:|---:|---:|---:|---:|
| LLaDA | abstract_case | v2 | shared_fulltext | 499 | 0.80% | 99.20% | 0.9920 | 0.9977 |
| LLaDA | abstract_case | xlmr | shared_fulltext | 499 | 0.00% | 80.96% | 0.9039 | 0.8780 |
| LLaDA | abstract_case | xlmr | native_context | 499 | 11.62% | 87.17% | 0.8778 | 0.9488 |
| LLaDA | abstract_case | mage | shared_fulltext | 499 | 100.00% | 100.00% | 0.3333 | 0.9689 |
| LLaDA | abstract_case | mage | raw_context_limit | 499 | 1.80% | 100.00% | 0.9910 | 1.0000 |
| LLaDA | rephrese_case | v2 | shared_fulltext | 497 | 0.80% | 46.28% | 0.7068 | 0.8558 |
| LLaDA | rephrese_case | xlmr | shared_fulltext | 497 | 0.00% | 67.40% | 0.8326 | 0.7686 |
| LLaDA | rephrese_case | xlmr | native_context | 497 | 11.67% | 80.68% | 0.8448 | 0.9188 |
| LLaDA | rephrese_case | mage | shared_fulltext | 497 | 100.00% | 71.63% | 0.2637 | 0.5045 |
| LLaDA | rephrese_case | mage | raw_context_limit | 497 | 1.81% | 62.17% | 0.7952 | 0.9511 |
| LLaMA | abstract_case | v2 | shared_fulltext | 499 | 0.80% | 100.00% | 0.9960 | 1.0000 |
| LLaMA | abstract_case | xlmr | shared_fulltext | 499 | 0.00% | 93.99% | 0.9699 | 0.9988 |
| LLaMA | abstract_case | xlmr | native_context | 499 | 11.62% | 100.00% | 0.9417 | 0.9970 |
| LLaMA | abstract_case | mage | shared_fulltext | 499 | 100.00% | 99.40% | 0.3320 | 0.7821 |
| LLaMA | abstract_case | mage | raw_context_limit | 499 | 1.80% | 99.80% | 0.9900 | 0.9998 |
| LLaDA | abstract_case | mage_author_recipe | published_preprocess_and_boundary | 499 | 13.23% | 100.00% | 0.9336 | 0.9975 |
| LLaDA | rephrese_case | mage_author_recipe | published_preprocess_and_boundary | 497 | 13.28% | 47.28% | 0.6567 | 0.7807 |
| LLaMA | abstract_case | mage_author_recipe | published_preprocess_and_boundary | 499 | 13.23% | 99.60% | 0.9316 | 0.9950 |

## Public sources under the common full-text policy

| Source | Language | Condition | Detector | Human/AI | Human FPR | AI recall | Macro-F1 |
|---|---|---|---|---:|---:|---:|---:|
| AI-Detector-Benchmark-2026 | en | binary | mage | 500/382 | 75.40% | 92.67% | 50.70% |
| AI-Detector-Benchmark-2026 | en | binary | v2 | 500/382 | 1.00% | 93.98% | 96.75% |
| AI-Detector-Benchmark-2026 | en | binary | xlmr | 500/382 | 3.60% | 67.28% | 82.66% |
| C-ReD | zh | in_domain_heldout | v2 | 804/1099 | 0.37% | 99.55% | 99.57% |
| C-ReD | zh | in_domain_heldout | xlmr | 804/1099 | 19.40% | 98.64% | 90.52% |
| CLUE-CSL | zh | binary | v2 | 200/0 | 1.00% | — | — |
| CLUE-CSL | zh | binary | xlmr | 200/0 | 35.00% | — | — |
| CUDRT | zh | in_domain_heldout | v2 | 104/106 | 3.85% | 67.92% | 81.58% |
| CUDRT | zh | in_domain_heldout | xlmr | 104/106 | 10.58% | 55.66% | 71.64% |
| Data-Prep-Bench | en | ai_only_qa | mage | 0/1168 | — | 49.32% | — |
| Data-Prep-Bench | en | ai_only_qa | v2 | 0/1168 | — | 13.18% | — |
| Data-Prep-Bench | en | ai_only_qa | xlmr | 0/1168 | — | 11.82% | — |
| Diffusion-Generated-Text | en | new_mechanism_ai_only | mage | 0/690 | — | 99.42% | — |
| Diffusion-Generated-Text | en | new_mechanism_ai_only | v2 | 0/690 | — | 85.51% | — |
| Diffusion-Generated-Text | en | new_mechanism_ai_only | xlmr | 0/690 | — | 16.52% | — |
| Diffusion-vs-AR-arXiv | en | new_mechanism | mage | 499/1496 | 100.00% | 90.37% | 40.39% |
| Diffusion-vs-AR-arXiv | en | new_mechanism | v2 | 499/1496 | 0.80% | 81.89% | 84.08% |
| Diffusion-vs-AR-arXiv | en | new_mechanism | xlmr | 499/1496 | 0.00% | 80.82% | 83.53% |
| Diffusion-vs-AR-arXiv | en | unverified_pairing_diagnostic | mage | 0/500 | — | 94.40% | — |
| Diffusion-vs-AR-arXiv | en | unverified_pairing_diagnostic | v2 | 0/500 | — | 74.00% | — |
| Diffusion-vs-AR-arXiv | en | unverified_pairing_diagnostic | xlmr | 0/500 | — | 87.40% | — |
| EvalDetectBench-WritingBench | en | binary | mage | 0/35 | — | 97.14% | — |
| EvalDetectBench-WritingBench | en | binary | v2 | 0/35 | — | 82.86% | — |
| EvalDetectBench-WritingBench | en | binary | xlmr | 0/35 | — | 14.29% | — |
| EvalDetectBench-WritingBench | mixed | binary | v2 | 0/6 | — | 100.00% | — |
| EvalDetectBench-WritingBench | mixed | binary | xlmr | 0/6 | — | 33.33% | — |
| EvalDetectBench-WritingBench | zh | binary | v2 | 0/19 | — | 100.00% | — |
| EvalDetectBench-WritingBench | zh | binary | xlmr | 0/19 | — | 63.16% | — |
| HC3 | zh | in_domain_heldout | v2 | 382/382 | 0.26% | 100.00% | 99.87% |
| HC3 | zh | in_domain_heldout | xlmr | 382/382 | 9.42% | 60.47% | 74.96% |
| M4-qazh | zh | binary | v2 | 100/200 | 1.00% | 95.00% | 95.96% |
| M4-qazh | zh | binary | xlmr | 100/200 | 0.00% | 41.00% | 60.52% |
| NLPCC2025-STORAL | zh | binary | v2 | 100/100 | 0.00% | 100.00% | 100.00% |
| NLPCC2025-STORAL | zh | binary | xlmr | 100/100 | 29.00% | 100.00% | 85.19% |
| NLPCC2025-STORAL | zh | mixed | v2 | 100/100 | 0.00% | 94.00% | 97.00% |
| NLPCC2025-STORAL | zh | mixed | xlmr | 100/100 | 29.00% | 85.00% | 77.89% |
| NLPCC2025-STORAL | zh | robustness | v2 | 353/500 | 0.57% | 97.80% | 98.44% |
| NLPCC2025-STORAL | zh | robustness | xlmr | 353/500 | 24.36% | 81.80% | 78.66% |
| TextSight-2026-09 | en | binary | mage | 643/48 | 36.55% | 52.08% | 46.11% |
| TextSight-2026-09 | en | binary | v2 | 643/48 | 3.11% | 83.33% | 85.94% |
| TextSight-2026-09 | en | binary | xlmr | 643/48 | 2.49% | 54.17% | 77.42% |
| TextSight-2026-09 | mixed | binary | v2 | 1/0 | 0.00% | — | — |
| TextSight-2026-09 | mixed | binary | xlmr | 1/0 | 0.00% | — | — |
| ToolBenchmarks-2026-10 | en | binary | mage | 150/600 | 45.33% | 57.00% | 50.60% |
| ToolBenchmarks-2026-10 | en | binary | v2 | 150/600 | 0.00% | 16.83% | 33.18% |
| ToolBenchmarks-2026-10 | en | binary | xlmr | 150/600 | 3.33% | 21.50% | 36.50% |
| ToolBenchmarks-2026-10 | en | edited | mage | 0/150 | — | 48.00% | — |
| ToolBenchmarks-2026-10 | en | edited | v2 | 0/150 | — | 5.33% | — |
| ToolBenchmarks-2026-10 | en | edited | xlmr | 0/150 | — | 4.00% | — |
| WritingPreferenceBench-ZH | zh | binary | v2 | 0/675 | — | 95.11% | — |
| WritingPreferenceBench-ZH | zh | binary | xlmr | 0/675 | — | 76.44% | — |
| Zhiyin | zh | binary | v2 | 0/516 | — | 78.49% | — |
| Zhiyin | zh | binary | xlmr | 0/516 | — | 64.53% | — |

## Reading these results

- V2 was fitted on C-ReD/HC3/CUDRT: holdouts from these sources are source-familiar, not entirely new domains.
- The LLaDA/LLaMA academic collection and the additional diffusion outputs were not used for V2 fitting or threshold selection. Known fitting-family overlaps are excluded.
- LLaMA rephrasing has source-index alignment problems and is excluded from matched primary comparisons. Its scores are diagnostic only.
- The three-family diffusion output collection has no matched human controls: recall only, no invented FPR/F1/AUROC.
- The MAGE author recipe is the same detector under an additional published preprocessing/boundary, not an extra detector. Unsupported-context counts are explicit in its public-source coverage; all fresh inputs are supported.
- The original short-window MAGE comparison is an input-policy stress test. Do not cite its high false-positive rate as MAGE author-default performance.
- Native first-512/4096-token source aggregates, per-generator/cohort breakdowns, confidence intervals, and the separate author recipe are in [external_v2.json](../results/external_v2.json).
- The five-epoch V3 DetectRL-X score is not a V2 result and is not used in this release.

## Public source references

- [Can You Detect the Difference?](https://arxiv.org/abs/2507.10475) / [pinned data](https://github.com/ismailtrm/ceng_404/tree/7786b2bd1bfd080c3f16343164c64a3d7cf39e9f).
- [Additional diffusion outputs](https://huggingface.co/datasets/paoche11/diffusion-generated-text/tree/d1d8c8593a6b7db5c446f497fd000a0ce74dcded).
- [DetectRL-X XLM-R](https://huggingface.co/WUJUNCHAO/DetectRL-X-XLM-RoBERTa-Detector-All), [MAGE weights](https://huggingface.co/yaful/MAGE) and [MAGE published deployment recipe](https://github.com/yafuly/MAGE/blob/6d11f851184b9f04166f952ddc1f47727f36710f/deployment/utils.py).
- Original multi-source holdouts: [C-ReD](https://github.com/HeraldofLight/C-ReD), [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese), [NLPCC/CUDRT](https://github.com/NLP2CT/NLPCC-2026-Task6-Detection).
