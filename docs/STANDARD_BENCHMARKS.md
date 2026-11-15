# Additional standard benchmarks / 新增标准 benchmark

Two additional datasets, not a renamed set of abstract conditions. All six checkpoints and their operating points remain frozen. These are local sampled test slices, not official full-test rankings.

## Sources and predeclared sampling

- **COLING 2025 GenAI Task 1:** [organizer repository](https://github.com/mbzuai-nlp/COLING-2025-Workshop-on-MGT-Detection-Task1), [official public data folder](https://drive.google.com/drive/folders/1Mz8vTnqi7truGrc05v6kWaod6mEK7Enj). Use `test_set_en_with_label.jsonl`, released with gold labels by the organizers. No training/dev file is scored. Select 300 inputs per official label, balanced across eligible source/task strata. The resulting clean sample has 12 strata with 25 inputs per label per stratum.
- **Intel AI Peer Review Detection Benchmark:** [author dataset](https://huggingface.co/datasets/IntelLabs/AI-Peer-Review-Detection-Benchmark), [paper](https://arxiv.org/abs/2502.19614). Use only `test/*/ICLR2024.test.*.csv`, pinned dataset revision `b80f6524702eb2250139b744430a9a85478586bc`; calibration and extended files are not used. Match human review identities across GPT-4o, Claude, Gemini, Llama and Qwen; select one review identity per paper and 50 papers by fixed hash. Each paper supplies one human review and five generated reviews.
- **Local stress condition:** from the selected COLING test texts, choose 100 per label by fixed hash and insert a zero-width space inside every seventh ASCII word of length at least four. Apply the same operation to both labels; removing inserted characters exactly recovers the original text. This is derived data, not another independent dataset or an official attack score.
- Selection seed: `standard-extra-20261114-v1`. The plan was recorded before acquisition/selection, and the full selected-input checksum and model identities were frozen before inference. No test-label threshold search or weight updates.
- Minimum length: 100 nonspace characters, matching the existing production eligibility rule. Canonical duplicates and conflicting labels are handled before sampling. Known fitting/prefix matches for V2/V3/XLM-R and earlier exposed external inputs are excluded, propagating paper families where available; five known-overlap families were removed.
- Intel uses the same Summary/Strengths/Weaknesses/Questions sections and headings for both labels. Ratings, confidence, paper IDs and source labels are never passed to a detector.
- Google Drive publishes no independently pinned checksum here: byte ranges/total size were validated and a full SHA-256 was recorded at acquisition. Intel complete-file SHA-256 matches the pinned public LFS metadata. Dataset rows and record IDs stay local.

## Label semantics: official negative class is not pure-human provenance

**COLING label semantics (post-hoc audit):** 75 of the 300 official human-class negatives are MixSet `gpt4_humanize`/`2llama_humanize` texts. The [MixSet authors](https://llm-coauthor.github.io/) describe humanize as editing machine-generated text. Consequently, human FPR in this column means **official-negative-class FPR**, not exclusively false accusations on purely human writing. This applies to every detector; all 600 original labels/inputs and published scores are retained.

This clarification was added after the frozen evaluation. It changes interpretation, not labels, selection, thresholds, predictions or metrics. Mixed/edited-source errors should be read separately from the other source strata.

## Fixed native policies

Same [checkpoint identities and native input policies](RECENT_COMPARISON.md) as the earlier comparison. V2 uses the production paragraph/sliding-window score at 0.5; Desklib first768 at0.5; Tropa native sentence chunks at0.9755912190656111; ConPara normalized first512, margin >2.9423828125; XLM-R first512 at0.5; MAGE original preprocessing and raw class0 logit >3.08583984375, CPU FP32, maximum4096. MAGE author decisions were checked against its pinned original detect function.

Every comparison uses the supported-ID intersection for that cohort. Intel overall cohorts require a complete human+five-generator paper family; each generator-specific comparison requires its human/AI pair. Stress comparisons require both clean and perturbed versions supported by all detectors. Unsupported counts remain visible. Input budgets differ, so this is a deployment-policy comparison rather than equal-compute evaluation.

## Main conditions

| Dataset / condition | Detector | Human / machine-label | Families | Macro-F1 [95% CI] | Human FPR | Machine-label recall | AUROC | Excluded unsupported inputs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| COLING2025-English-test / clean | FiveTrace V2 | 300/300 | 590 | 0.8250 [0.7939, 0.8542] | 18.33% | 83.33% | 0.8658 | 0 |
| COLING2025-English-test / clean | Desklib (2025) | 300/300 | 590 | 0.7274 [0.6914, 0.7650] | 39.33% | 85.67% | 0.8644 | 0 |
| COLING2025-English-test / clean | Tropa-mini (2026) | 300/300 | 590 | 0.8367 [0.8065, 0.8631] | 15.33% | 82.67% | 0.9093 | 0 |
| COLING2025-English-test / clean | ConPara (AACL 2026) | 300/300 | 590 | 0.8127 [0.7806, 0.8431] | 29.67% | 92.67% | 0.8851 | 0 |
| COLING2025-English-test / clean | DetectRL-X XLM-R (2026) | 300/300 | 590 | 0.7647 [0.7267, 0.7994] | 5.67% | 60.00% | 0.8495 | 0 |
| COLING2025-English-test / clean | MAGE author (ACL 2024) | 300/300 | 590 | 0.7409 [0.7064, 0.7774] | 31.33% | 79.67% | 0.8080 | 0 |
| COLING2025-English-test / clean_matched_for_stress | FiveTrace V2 | 100/100 | 198 | 0.8700 [0.8199, 0.9152] | 12.00% | 86.00% | 0.9246 | 0 |
| COLING2025-English-test / clean_matched_for_stress | Desklib (2025) | 100/100 | 198 | 0.7427 [0.6803, 0.7967] | 35.00% | 84.00% | 0.8762 | 0 |
| COLING2025-English-test / clean_matched_for_stress | Tropa-mini (2026) | 100/100 | 198 | 0.8445 [0.7940, 0.8887] | 10.00% | 79.00% | 0.9121 | 0 |
| COLING2025-English-test / clean_matched_for_stress | ConPara (AACL 2026) | 100/100 | 198 | 0.8589 [0.8086, 0.9039] | 23.00% | 95.00% | 0.9258 | 0 |
| COLING2025-English-test / clean_matched_for_stress | DetectRL-X XLM-R (2026) | 100/100 | 198 | 0.7906 [0.7310, 0.8492] | 6.00% | 65.00% | 0.8616 | 0 |
| COLING2025-English-test / clean_matched_for_stress | MAGE author (ACL 2024) | 100/100 | 198 | 0.7494 [0.6910, 0.8045] | 30.00% | 80.00% | 0.8132 | 0 |
| COLING2025-English-test / local_zero_width_stress | FiveTrace V2 | 100/100 | 198 | 0.8750 [0.8258, 0.9193] | 11.00% | 86.00% | 0.9223 | 0 |
| COLING2025-English-test / local_zero_width_stress | Desklib (2025) | 100/100 | 198 | 0.7895 [0.7358, 0.8405] | 16.00% | 74.00% | 0.8490 | 0 |
| COLING2025-English-test / local_zero_width_stress | Tropa-mini (2026) | 100/100 | 198 | 0.8144 [0.7593, 0.8660] | 13.00% | 76.00% | 0.8964 | 0 |
| COLING2025-English-test / local_zero_width_stress | ConPara (AACL 2026) | 100/100 | 198 | 0.8589 [0.8086, 0.9039] | 23.00% | 95.00% | 0.9258 | 0 |
| COLING2025-English-test / local_zero_width_stress | DetectRL-X XLM-R (2026) | 100/100 | 198 | 0.3333 [0.3024, 0.3620] | 0.00% | 0.00% | 0.5592 | 0 |
| COLING2025-English-test / local_zero_width_stress | MAGE author (ACL 2024) | 100/100 | 198 | 0.5454 [0.4756, 0.6117] | 12.00% | 29.00% | 0.7072 | 0 |
| COLING2025-English-test / zero_width_matched_stress | FiveTrace V2 | 100/100 | 198 | 0.8750 [0.8258, 0.9193] | 11.00% | 86.00% | 0.9223 | 0 |
| COLING2025-English-test / zero_width_matched_stress | Desklib (2025) | 100/100 | 198 | 0.7895 [0.7358, 0.8405] | 16.00% | 74.00% | 0.8490 | 0 |
| COLING2025-English-test / zero_width_matched_stress | Tropa-mini (2026) | 100/100 | 198 | 0.8144 [0.7593, 0.8660] | 13.00% | 76.00% | 0.8964 | 0 |
| COLING2025-English-test / zero_width_matched_stress | ConPara (AACL 2026) | 100/100 | 198 | 0.8589 [0.8086, 0.9039] | 23.00% | 95.00% | 0.9258 | 0 |
| COLING2025-English-test / zero_width_matched_stress | DetectRL-X XLM-R (2026) | 100/100 | 198 | 0.3333 [0.3024, 0.3620] | 0.00% | 0.00% | 0.5592 | 0 |
| COLING2025-English-test / zero_width_matched_stress | MAGE author (ACL 2024) | 100/100 | 198 | 0.5454 [0.4756, 0.6117] | 12.00% | 29.00% | 0.7072 | 0 |
| Intel-peer-review-ICLR2024 / clean | FiveTrace V2 | 50/250 | 50 | 0.8969 [0.8560, 0.9345] | 12.00% | 95.20% | 0.9497 | 0 |
| Intel-peer-review-ICLR2024 / clean | Desklib (2025) | 50/250 | 50 | 0.7996 [0.7228, 0.8644] | 52.00% | 100.00% | 0.9535 | 0 |
| Intel-peer-review-ICLR2024 / clean | Tropa-mini (2026) | 50/250 | 50 | 0.9324 [0.8918, 0.9695] | 14.00% | 98.40% | 0.9665 | 0 |
| Intel-peer-review-ICLR2024 / clean | ConPara (AACL 2026) | 50/250 | 50 | 0.9187 [0.8704, 0.9577] | 18.00% | 98.40% | 0.9524 | 0 |
| Intel-peer-review-ICLR2024 / clean | DetectRL-X XLM-R (2026) | 50/250 | 50 | 0.6877 [0.6590, 0.7149] | 14.00% | 73.60% | 0.8530 | 0 |
| Intel-peer-review-ICLR2024 / clean | MAGE author (ACL 2024) | 50/250 | 50 | 0.7220 [0.6645, 0.7774] | 38.00% | 87.20% | 0.8266 | 0 |

## Matched Intel review generators

| Generator | Detector | Human / AI | Macro-F1 | Human FPR | AI recall | AUROC |
|---|---|---:|---:|---:|---:|---:|
| claude | FiveTrace V2 | 50/50 | 0.8193 | 12.00% | 76.00% | 0.9004 |
| claude | Desklib (2025) | 50/50 | 0.7211 | 52.00% | 100.00% | 0.8860 |
| claude | Tropa-mini (2026) | 50/50 | 0.9297 | 14.00% | 100.00% | 0.9660 |
| claude | ConPara (AACL 2026) | 50/50 | 0.8895 | 18.00% | 96.00% | 0.8924 |
| claude | DetectRL-X XLM-R (2026) | 50/50 | 0.3749 | 14.00% | 8.00% | 0.6272 |
| claude | MAGE author (ACL 2024) | 50/50 | 0.6497 | 38.00% | 68.00% | 0.6748 |
| gemini | FiveTrace V2 | 50/50 | 0.9398 | 12.00% | 100.00% | 0.9532 |
| gemini | Desklib (2025) | 50/50 | 0.7211 | 52.00% | 100.00% | 0.9428 |
| gemini | Tropa-mini (2026) | 50/50 | 0.9297 | 14.00% | 100.00% | 0.9512 |
| gemini | ConPara (AACL 2026) | 50/50 | 0.8895 | 18.00% | 96.00% | 0.9184 |
| gemini | DetectRL-X XLM-R (2026) | 50/50 | 0.9197 | 14.00% | 98.00% | 0.9216 |
| gemini | MAGE author (ACL 2024) | 50/50 | 0.6885 | 38.00% | 76.00% | 0.7048 |
| gpt4o | FiveTrace V2 | 50/50 | 0.9398 | 12.00% | 100.00% | 0.9612 |
| gpt4o | Desklib (2025) | 50/50 | 0.7211 | 52.00% | 100.00% | 0.9780 |
| gpt4o | Tropa-mini (2026) | 50/50 | 0.9297 | 14.00% | 100.00% | 0.9756 |
| gpt4o | ConPara (AACL 2026) | 50/50 | 0.9093 | 18.00% | 100.00% | 0.9680 |
| gpt4o | DetectRL-X XLM-R (2026) | 50/50 | 0.9297 | 14.00% | 100.00% | 0.9604 |
| gpt4o | MAGE author (ACL 2024) | 50/50 | 0.7742 | 38.00% | 94.00% | 0.8372 |
| llama | FiveTrace V2 | 50/50 | 0.9398 | 12.00% | 100.00% | 0.9656 |
| llama | Desklib (2025) | 50/50 | 0.7211 | 52.00% | 100.00% | 0.9748 |
| llama | Tropa-mini (2026) | 50/50 | 0.8998 | 14.00% | 94.00% | 0.9532 |
| llama | ConPara (AACL 2026) | 50/50 | 0.9093 | 18.00% | 100.00% | 0.9908 |
| llama | DetectRL-X XLM-R (2026) | 50/50 | 0.7362 | 14.00% | 62.00% | 0.8120 |
| llama | MAGE author (ACL 2024) | 50/50 | 0.7933 | 38.00% | 98.00% | 0.9480 |
| qwen | FiveTrace V2 | 50/50 | 0.9398 | 12.00% | 100.00% | 0.9680 |
| qwen | Desklib (2025) | 50/50 | 0.7211 | 52.00% | 100.00% | 0.9860 |
| qwen | Tropa-mini (2026) | 50/50 | 0.9197 | 14.00% | 98.00% | 0.9864 |
| qwen | ConPara (AACL 2026) | 50/50 | 0.9093 | 18.00% | 100.00% | 0.9924 |
| qwen | DetectRL-X XLM-R (2026) | 50/50 | 0.9297 | 14.00% | 100.00% | 0.9440 |
| qwen | MAGE author (ACL 2024) | 50/50 | 0.8029 | 38.00% | 100.00% | 0.9684 |

## COLING source/task strata

Official labels are retained, including Mixset mixed-authorship cases and CUDRT refinement/summary tasks. A positive label is the organizer's binary target, not a verified 100%-AI article or a paragraph-level AI fraction. These strata are conditions within one benchmark, not independent benchmarks.

| Source / task | Detector | Human / machine-label | Macro-F1 | Human FPR | Machine-label recall | AUROC |
|---|---|---:|---:|---:|---:|---:|
| CUDRT/GPT3.5_QA | FiveTrace V2 | 25/25 | 0.9800 | 0.00% | 96.00% | 1.0000 |
| CUDRT/GPT3.5_QA | Desklib (2025) | 25/25 | 0.9398 | 12.00% | 100.00% | 0.9968 |
| CUDRT/GPT3.5_QA | Tropa-mini (2026) | 25/25 | 0.9800 | 4.00% | 100.00% | 1.0000 |
| CUDRT/GPT3.5_QA | ConPara (AACL 2026) | 25/25 | 0.9800 | 4.00% | 100.00% | 1.0000 |
| CUDRT/GPT3.5_QA | DetectRL-X XLM-R (2026) | 25/25 | 0.7211 | 0.00% | 48.00% | 0.9280 |
| CUDRT/GPT3.5_QA | MAGE author (ACL 2024) | 25/25 | 0.8140 | 36.00% | 100.00% | 0.9600 |
| CUDRT/GPT3.5_Refine | FiveTrace V2 | 25/25 | 0.7917 | 0.00% | 60.00% | 0.9616 |
| CUDRT/GPT3.5_Refine | Desklib (2025) | 25/25 | 0.7565 | 12.00% | 64.00% | 0.9008 |
| CUDRT/GPT3.5_Refine | Tropa-mini (2026) | 25/25 | 0.8990 | 0.00% | 80.00% | 0.9792 |
| CUDRT/GPT3.5_Refine | ConPara (AACL 2026) | 25/25 | 0.8996 | 16.00% | 96.00% | 0.9808 |
| CUDRT/GPT3.5_Refine | DetectRL-X XLM-R (2026) | 25/25 | 0.8358 | 0.00% | 68.00% | 0.9984 |
| CUDRT/GPT3.5_Refine | MAGE author (ACL 2024) | 25/25 | 0.5572 | 36.00% | 48.00% | 0.5520 |
| CUDRT/GPT3.5_Summary | FiveTrace V2 | 25/25 | 0.8996 | 4.00% | 84.00% | 0.9552 |
| CUDRT/GPT3.5_Summary | Desklib (2025) | 25/25 | 0.3333 | 84.00% | 56.00% | 0.5008 |
| CUDRT/GPT3.5_Summary | Tropa-mini (2026) | 25/25 | 0.4746 | 28.00% | 28.00% | 0.4880 |
| CUDRT/GPT3.5_Summary | ConPara (AACL 2026) | 25/25 | 0.9195 | 0.00% | 84.00% | 0.9632 |
| CUDRT/GPT3.5_Summary | DetectRL-X XLM-R (2026) | 25/25 | 0.4907 | 24.00% | 28.00% | 0.3248 |
| CUDRT/GPT3.5_Summary | MAGE author (ACL 2024) | 25/25 | 0.2360 | 96.00% | 52.00% | 0.1936 |
| LLM-DetectAIve-IELTS/IELTS Essay | FiveTrace V2 | 25/25 | 0.9599 | 0.00% | 92.00% | 1.0000 |
| LLM-DetectAIve-IELTS/IELTS Essay | Desklib (2025) | 25/25 | 0.8140 | 36.00% | 100.00% | 1.0000 |
| LLM-DetectAIve-IELTS/IELTS Essay | Tropa-mini (2026) | 25/25 | 0.8990 | 20.00% | 100.00% | 0.9872 |
| LLM-DetectAIve-IELTS/IELTS Essay | ConPara (AACL 2026) | 25/25 | 0.7688 | 44.00% | 100.00% | 0.9888 |
| LLM-DetectAIve-IELTS/IELTS Essay | DetectRL-X XLM-R (2026) | 25/25 | 0.9398 | 0.00% | 88.00% | 1.0000 |
| LLM-DetectAIve-IELTS/IELTS Essay | MAGE author (ACL 2024) | 25/25 | 0.8782 | 24.00% | 100.00% | 0.8864 |
| Mixset/NarrativeQA_LLMs | FiveTrace V2 | 25/25 | 0.3333 | 76.00% | 44.00% | 0.3120 |
| Mixset/NarrativeQA_LLMs | Desklib (2025) | 25/25 | 0.3676 | 76.00% | 52.00% | 0.3728 |
| Mixset/NarrativeQA_LLMs | Tropa-mini (2026) | 25/25 | 0.5383 | 40.00% | 48.00% | 0.5104 |
| Mixset/NarrativeQA_LLMs | ConPara (AACL 2026) | 25/25 | 0.4485 | 84.00% | 88.00% | 0.3600 |
| Mixset/NarrativeQA_LLMs | DetectRL-X XLM-R (2026) | 25/25 | 0.4624 | 8.00% | 16.00% | 0.3536 |
| Mixset/NarrativeQA_LLMs | MAGE author (ACL 2024) | 25/25 | 0.6599 | 36.00% | 68.00% | 0.6528 |
| Mixset/SQuAD1_LLMs | FiveTrace V2 | 25/25 | 0.3559 | 72.00% | 44.00% | 0.3920 |
| Mixset/SQuAD1_LLMs | Desklib (2025) | 25/25 | 0.4492 | 68.00% | 60.00% | 0.4320 |
| Mixset/SQuAD1_LLMs | Tropa-mini (2026) | 25/25 | 0.5572 | 36.00% | 48.00% | 0.5296 |
| Mixset/SQuAD1_LLMs | ConPara (AACL 2026) | 25/25 | 0.4167 | 76.00% | 64.00% | 0.4352 |
| Mixset/SQuAD1_LLMs | DetectRL-X XLM-R (2026) | 25/25 | 0.3443 | 12.00% | 4.00% | 0.2704 |
| Mixset/SQuAD1_LLMs | MAGE author (ACL 2024) | 25/25 | 0.6400 | 36.00% | 64.00% | 0.7216 |
| Mixset/TruthfulQA_LLMs | FiveTrace V2 | 25/25 | 0.6250 | 56.00% | 84.00% | 0.6720 |
| Mixset/TruthfulQA_LLMs | Desklib (2025) | 25/25 | 0.4415 | 88.00% | 96.00% | 0.7032 |
| Mixset/TruthfulQA_LLMs | Tropa-mini (2026) | 25/25 | 0.7348 | 40.00% | 88.00% | 0.8720 |
| Mixset/TruthfulQA_LLMs | ConPara (AACL 2026) | 25/25 | 0.4286 | 88.00% | 92.00% | 0.6032 |
| Mixset/TruthfulQA_LLMs | DetectRL-X XLM-R (2026) | 25/25 | 0.5385 | 12.00% | 28.00% | 0.6432 |
| Mixset/TruthfulQA_LLMs | MAGE author (ACL 2024) | 25/25 | 0.7537 | 40.00% | 92.00% | 0.9312 |
| NLPeer-ARR22/Academic peer review | FiveTrace V2 | 25/25 | 1.0000 | 0.00% | 100.00% | 1.0000 |
| NLPeer-ARR22/Academic peer review | Desklib (2025) | 25/25 | 1.0000 | 0.00% | 100.00% | 1.0000 |
| NLPeer-ARR22/Academic peer review | Tropa-mini (2026) | 25/25 | 1.0000 | 0.00% | 100.00% | 1.0000 |
| NLPeer-ARR22/Academic peer review | ConPara (AACL 2026) | 25/25 | 0.9600 | 4.00% | 96.00% | 0.9952 |
| NLPeer-ARR22/Academic peer review | DetectRL-X XLM-R (2026) | 25/25 | 0.8990 | 0.00% | 80.00% | 1.0000 |
| NLPeer-ARR22/Academic peer review | MAGE author (ACL 2024) | 25/25 | 0.9000 | 12.00% | 92.00% | 0.9648 |
| NLPeer-COLING20/Academic peer review | FiveTrace V2 | 25/25 | 1.0000 | 0.00% | 100.00% | 1.0000 |
| NLPeer-COLING20/Academic peer review | Desklib (2025) | 25/25 | 0.9800 | 4.00% | 100.00% | 1.0000 |
| NLPeer-COLING20/Academic peer review | Tropa-mini (2026) | 25/25 | 1.0000 | 0.00% | 100.00% | 1.0000 |
| NLPeer-COLING20/Academic peer review | ConPara (AACL 2026) | 25/25 | 1.0000 | 0.00% | 100.00% | 1.0000 |
| NLPeer-COLING20/Academic peer review | DetectRL-X XLM-R (2026) | 25/25 | 0.9195 | 0.00% | 84.00% | 0.9840 |
| NLPeer-COLING20/Academic peer review | MAGE author (ACL 2024) | 25/25 | 0.8798 | 8.00% | 84.00% | 0.9584 |
| NLPeer-F1000/Academic peer review | FiveTrace V2 | 25/25 | 1.0000 | 0.00% | 100.00% | 1.0000 |
| NLPeer-F1000/Academic peer review | Desklib (2025) | 25/25 | 0.8990 | 20.00% | 100.00% | 1.0000 |
| NLPeer-F1000/Academic peer review | Tropa-mini (2026) | 25/25 | 1.0000 | 0.00% | 100.00% | 1.0000 |
| NLPeer-F1000/Academic peer review | ConPara (AACL 2026) | 25/25 | 0.8792 | 20.00% | 96.00% | 0.9872 |
| NLPeer-F1000/Academic peer review | DetectRL-X XLM-R (2026) | 25/25 | 0.9195 | 0.00% | 84.00% | 0.9872 |
| NLPeer-F1000/Academic peer review | MAGE author (ACL 2024) | 25/25 | 0.8595 | 20.00% | 92.00% | 0.9520 |
| ieltsduck/ielts | FiveTrace V2 | 25/25 | 0.9800 | 4.00% | 100.00% | 1.0000 |
| ieltsduck/ielts | Desklib (2025) | 25/25 | 0.7917 | 40.00% | 100.00% | 1.0000 |
| ieltsduck/ielts | Tropa-mini (2026) | 25/25 | 0.9195 | 16.00% | 100.00% | 1.0000 |
| ieltsduck/ielts | ConPara (AACL 2026) | 25/25 | 0.9195 | 16.00% | 100.00% | 1.0000 |
| ieltsduck/ielts | DetectRL-X XLM-R (2026) | 25/25 | 0.9800 | 4.00% | 100.00% | 0.9968 |
| ieltsduck/ielts | MAGE author (ACL 2024) | 25/25 | 0.8990 | 20.00% | 100.00% | 0.9600 |
| peersum/None | FiveTrace V2 | 25/25 | 0.9400 | 8.00% | 96.00% | 0.9760 |
| peersum/None | Desklib (2025) | 25/25 | 0.8358 | 32.00% | 100.00% | 1.0000 |
| peersum/None | Tropa-mini (2026) | 25/25 | 1.0000 | 0.00% | 100.00% | 1.0000 |
| peersum/None | ConPara (AACL 2026) | 25/25 | 0.9600 | 4.00% | 96.00% | 0.9984 |
| peersum/None | DetectRL-X XLM-R (2026) | 25/25 | 0.9200 | 8.00% | 92.00% | 0.9696 |
| peersum/None | MAGE author (ACL 2024) | 25/25 | 0.7565 | 12.00% | 64.00% | 0.8032 |

## Uncertainty, exposure and boundaries

- Macro-F1 and V2-minus-comparator intervals use 500 joint bootstrap resamples of known paper/input families. Unavailable topic-parent IDs prevent complete cluster identification. Detailed differences are in the aggregate JSON; an interval spanning zero does not establish separation.
- Source-wise Wilson rate bounds in JSON are descriptive per-input bounds; they are not generally cluster-adjusted for reused controls or multiple outputs from the same paper. The per-generator Intel pairs have one human and one AI output per selected paper.
- COLING includes CUDRT/MAGE/HC3/M4-family sources. V2 was fitted on C-ReD/HC3-Chinese/CUDRT; ConPara was trained on HC3/M4/MAGE/RAID-family data. Exact/prefix exclusions do not make source-familiar conditions globally unseen. Full comparator training manifests and Base pretraining exposure are not audited.
- Existing external tests influenced this research. New source selection was motivated by the need for more domains; it is not a claim of a preregistered blind study or universal detector ranking.
- The sample sizes are pilots within standard benchmark test sets. Human review controls are shared across five generators; source/task and local perturbation conditions are not independent datasets or independent author counts.
- Zero-width insertion tests a particular input-format perturbation. It does not establish robustness to paraphrasing, semantic edits or human-assisted rewriting. V2 removes zero-width spaces in its feature preprocessing; remaining paragraph/window boundaries can still change.
- The original RAID test labels are withheld. We did not substitute RAID training rows/community re-splits and call them official test scores, nor upload per-record predictions to an external scoring service.
- Data and labels are retained locally. Only code, documentation, plots of aggregate metrics and aggregate JSON are published.

## Aggregate metadata

[Source hashes, sample counts, fixed thresholds, overlap audit, coverage and per-model metrics](../results/standard_comparison.json). Native scoring can be replayed on local JSONL with `baseline_eval.py`; the repository intentionally does not distribute selected texts or record-ID lists.
