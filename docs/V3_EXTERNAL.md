# V3: external benchmarks versus official checkpoints

Primary comparison: six external sources / eight conditions. This is separate from the new covered-generator protocol-A test. Source-familiar and historically evaluated resources; not a blind or same-data-training ranking.

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


## Sampling and ancestry audit

| Condition | Original H/AI | Primary H/AI | Original known V3 ancestry hits |
|---|---:|---:|---:|
| TextSight | 643/48 | 643/48 | 0 |
| AI-Detector-Benchmark | 500/382 | 500/382 | 0 |
| ToolBenchmarks raw | 150/600 | 150/600 | 0 |
| LLaDA abstract | 499/499 | 118/118 | 762 |
| LLaDA rephrase | 497/497 | 118/118 | 758 |
| LLaMA abstract | 499/499 | 118/118 | 762 |
| COLING 2025 test slice | 300/300 | 300/300 | 0 |
| Intel peer review | 50/250 | 50/250 | 0 |

Before V3 external predictions, we canonicalized bodies (NFKC, casefold, punctuation/space removal), joined the frozen document registry and its original-source group graph, and excluded groups assigned to V3 TRAIN, DEVELOPMENT or CALIBRATION. A match in any source case propagates to known family members, including controls and alternative outputs. Remove both members of matched abstract pairs and entire Intel paper groups if either intersects. This is a conservative known-overlap screen; it does not establish that unknown semantic variants, Base pretraining or other detectors’ full training corpora are absent.

The original benchmark files, labels, sampling and previous scores are unchanged. All detectors use exactly the same retained IDs within each column. Abstract primary sets have 118 pairs per condition and share controls; do not sum these as independent human authors.

## Operating points and cache provenance

V3 uses its frozen task weights and threshold 0.5. Desklib uses 0.5; Tropa uses its published 0.9755912190656111 pooled-AI cutoff; ConPara uses the author checkpoint margin threshold (2.942383), with author Unicode/whitespace normalization. XLM-R uses the original first-512-token policy; MAGE uses its author preprocessing and decision boundary, not the separate raw sensitivity mode.

The official checkpoint/config/code revisions and hashes are in [aggregate metadata](../results/v3_external.json). We reuse exact-input baseline predictions from the verified frozen V2-era evaluations. Input and prediction-file hashes were fixed before V3 scoring, and source/native decisions are preserved rather than recalculating every model at 0.5. New V3 model and head hashes are also bound to that pre-prediction protocol. No official comparator was retrained.

V3 covers every language-aware detection window; official context policies differ (XLM-R 512, ConPara 512, Desklib 768, Tropa sentence chunks, MAGE author preprocessing/context). No matched-compute or equal-token claim.

## Classifier numerical precision

Production parity checks revealed BF16 classifier batch-shape sensitivity on this GPU. The same input had raw margin 5.53125 in its external batch and 6.328125 alone. We retained those original measurements locally, froze a numerical execution amendment, disabled classifier autocast/TF32, and recomputed both external and A results in FP32 without refitting task weights, fusion or thresholds. Reference features and all baseline predictions/inputs remain unchanged. Source/precision protocol hashes are published. The amendment followed inspection of the earlier BF16 scores and is disclosed as a consistency correction, not a new blind experiment.

## Label semantics

COLING 2025 is a fixed 600-record official-English-test subset, not a full leaderboard score. Its 300 official negatives include 75 MixSet gpt4_humanize/2llama_humanize examples. Authors define humanize as editing machine-generated text. Accordingly FPR is the official-negative-class error rate, not exclusively false accusations against pure human writing. Original labels are unchanged. Intel is a 50-paper paired slice with 50 human reviews and five generators × 50 generated reviews; humans are reused across generator comparisons.

## AUROC and uncertainty

| Condition | Detector | AUROC | Macro-F1 95% CI |
|---|---|---:|---|
| TextSight | FiveTrace V3 | 0.997570 | 0.9032–0.9741 |
| TextSight | Desklib v1.01 (2025) | 0.984578 | 0.7251–0.8465 |
| TextSight | Tropa-mini (2026) | 0.999028 | 0.8607–0.9532 |
| TextSight | ConPara (AACL 2026) | 0.898425 | 0.6643–0.7954 |
| TextSight | DetectRL-X XLM-R (2026) | 0.990312 | 0.7855–0.9292 |
| TextSight | MAGE author (ACL 2024) | 0.674248 | 0.5492–0.6699 |
| AI-Detector-Benchmark | FiveTrace V3 | 0.995974 | 0.9609–0.9854 |
| AI-Detector-Benchmark | Desklib v1.01 (2025) | 0.974539 | 0.8182–0.8675 |
| AI-Detector-Benchmark | Tropa-mini (2026) | 0.996508 | 0.9542–0.9797 |
| AI-Detector-Benchmark | ConPara (AACL 2026) | 0.887440 | 0.6733–0.7369 |
| AI-Detector-Benchmark | DetectRL-X XLM-R (2026) | 0.966545 | 0.8857–0.9307 |
| AI-Detector-Benchmark | MAGE author (ACL 2024) | 0.752099 | 0.5705–0.6345 |
| ToolBenchmarks raw | FiveTrace V3 | 0.861389 | 0.3987–0.4679 |
| ToolBenchmarks raw | Desklib v1.01 (2025) | 0.871844 | 0.5809–0.6541 |
| ToolBenchmarks raw | Tropa-mini (2026) | 0.898433 | 0.5787–0.6592 |
| ToolBenchmarks raw | ConPara (AACL 2026) | 0.782100 | 0.5483–0.6209 |
| ToolBenchmarks raw | DetectRL-X XLM-R (2026) | 0.785733 | 0.3605–0.4407 |
| ToolBenchmarks raw | MAGE author (ACL 2024) | 0.677911 | 0.4783–0.5551 |
| LLaDA abstract | FiveTrace V3 | 1.000000 | 1.0000–1.0000 |
| LLaDA abstract | Desklib v1.01 (2025) | 1.000000 | 0.9533–0.9915 |
| LLaDA abstract | Tropa-mini (2026) | 0.997702 | 0.6580–0.7635 |
| LLaDA abstract | ConPara (AACL 2026) | 0.999928 | 0.9576–0.9958 |
| LLaDA abstract | DetectRL-X XLM-R (2026) | 0.940893 | 0.8220–0.9090 |
| LLaDA abstract | MAGE author (ACL 2024) | 0.992818 | 0.8841–0.9533 |
| LLaDA rephrase | FiveTrace V3 | 0.992818 | 0.9190–0.9746 |
| LLaDA rephrase | Desklib v1.01 (2025) | 0.925524 | 0.7508–0.8501 |
| LLaDA rephrase | Tropa-mini (2026) | 0.735529 | 0.3333–0.3788 |
| LLaDA rephrase | ConPara (AACL 2026) | 0.892344 | 0.6495–0.7516 |
| LLaDA rephrase | DetectRL-X XLM-R (2026) | 0.907929 | 0.7965–0.8813 |
| LLaDA rephrase | MAGE author (ACL 2024) | 0.773269 | 0.5972–0.6911 |
| LLaMA abstract | FiveTrace V3 | 1.000000 | 1.0000–1.0000 |
| LLaMA abstract | Desklib v1.01 (2025) | 1.000000 | 0.9533–0.9915 |
| LLaMA abstract | Tropa-mini (2026) | 0.997558 | 0.8797–0.9490 |
| LLaMA abstract | ConPara (AACL 2026) | 0.999713 | 0.9576–0.9958 |
| LLaMA abstract | DetectRL-X XLM-R (2026) | 0.997199 | 0.9060–0.9661 |
| LLaMA abstract | MAGE author (ACL 2024) | 0.988150 | 0.8797–0.9470 |
| COLING 2025 test slice | FiveTrace V3 | 0.873311 | 0.8162–0.8732 |
| COLING 2025 test slice | Desklib v1.01 (2025) | 0.864356 | 0.6942–0.7611 |
| COLING 2025 test slice | Tropa-mini (2026) | 0.909311 | 0.8064–0.8642 |
| COLING 2025 test slice | ConPara (AACL 2026) | 0.885056 | 0.7813–0.8462 |
| COLING 2025 test slice | DetectRL-X XLM-R (2026) | 0.849489 | 0.7313–0.7992 |
| COLING 2025 test slice | MAGE author (ACL 2024) | 0.808011 | 0.7060–0.7738 |
| Intel peer review | FiveTrace V3 | 0.948240 | 0.7940–0.8995 |
| Intel peer review | Desklib v1.01 (2025) | 0.953520 | 0.7228–0.8644 |
| Intel peer review | Tropa-mini (2026) | 0.966480 | 0.8891–0.9695 |
| Intel peer review | ConPara (AACL 2026) | 0.952400 | 0.8670–0.9570 |
| Intel peer review | DetectRL-X XLM-R (2026) | 0.853040 | 0.6563–0.7152 |
| Intel peer review | MAGE author (ACL 2024) | 0.826640 | 0.6683–0.7769 |

Intervals jointly resample known topic/paper/input families 500 times across models; they are descriptive and may miss residual dependence where upstream topic IDs are unknown. Rate intervals in the JSON are per-input Wilson intervals, not generally cluster-adjusted.

## Paired V3 minus comparator Macro-F1

| Condition | Comparator | Difference | Paired 95% CI |
|---|---|---:|---|
| TextSight | Desklib v1.01 (2025) | +0.1527 | +0.1030–+0.2120 |
| TextSight | Tropa-mini (2026) | +0.0294 | -0.0144–+0.0786 |
| TextSight | ConPara (AACL 2026) | +0.2150 | +0.1441–+0.2773 |
| TextSight | DetectRL-X XLM-R (2026) | +0.0787 | +0.0077–+0.1556 |
| TextSight | MAGE author (ACL 2024) | +0.3365 | +0.2678–+0.4019 |
| AI-Detector-Benchmark | Desklib v1.01 (2025) | +0.1322 | +0.1038–+0.1591 |
| AI-Detector-Benchmark | Tropa-mini (2026) | +0.0071 | -0.0104–+0.0235 |
| AI-Detector-Benchmark | ConPara (AACL 2026) | +0.2705 | +0.2360–+0.3050 |
| AI-Detector-Benchmark | DetectRL-X XLM-R (2026) | +0.0650 | +0.0438–+0.0886 |
| AI-Detector-Benchmark | MAGE author (ACL 2024) | +0.3709 | +0.3383–+0.4028 |
| ToolBenchmarks raw | Desklib v1.01 (2025) | -0.1790 | -0.2215–-0.1424 |
| ToolBenchmarks raw | Tropa-mini (2026) | -0.1817 | -0.2327–-0.1324 |
| ToolBenchmarks raw | ConPara (AACL 2026) | -0.1477 | -0.1966–-0.1052 |
| ToolBenchmarks raw | DetectRL-X XLM-R (2026) | +0.0341 | -0.0068–+0.0713 |
| ToolBenchmarks raw | MAGE author (ACL 2024) | -0.0810 | -0.1353–-0.0317 |
| LLaDA abstract | Desklib v1.01 (2025) | +0.0254 | +0.0085–+0.0467 |
| LLaDA abstract | Tropa-mini (2026) | +0.2874 | +0.2365–+0.3420 |
| LLaDA abstract | ConPara (AACL 2026) | +0.0212 | +0.0042–+0.0424 |
| LLaDA abstract | DetectRL-X XLM-R (2026) | +0.1314 | +0.0910–+0.1780 |
| LLaDA abstract | MAGE author (ACL 2024) | +0.0810 | +0.0467–+0.1159 |
| LLaDA rephrase | Desklib v1.01 (2025) | +0.1481 | +0.0975–+0.2006 |
| LLaDA rephrase | Tropa-mini (2026) | +0.5971 | +0.5573–+0.6323 |
| LLaDA rephrase | ConPara (AACL 2026) | +0.2454 | +0.1891–+0.3125 |
| LLaDA rephrase | DetectRL-X XLM-R (2026) | +0.1059 | +0.0591–+0.1527 |
| LLaDA rephrase | MAGE author (ACL 2024) | +0.3040 | +0.2473–+0.3535 |
| LLaMA abstract | Desklib v1.01 (2025) | +0.0254 | +0.0085–+0.0467 |
| LLaMA abstract | Tropa-mini (2026) | +0.0854 | +0.0510–+0.1203 |
| LLaMA abstract | ConPara (AACL 2026) | +0.0212 | +0.0042–+0.0424 |
| LLaMA abstract | DetectRL-X XLM-R (2026) | +0.0638 | +0.0339–+0.0940 |
| LLaMA abstract | MAGE author (ACL 2024) | +0.0852 | +0.0530–+0.1203 |
| COLING 2025 test slice | Desklib v1.01 (2025) | +0.1181 | +0.0773–+0.1557 |
| COLING 2025 test slice | Tropa-mini (2026) | +0.0089 | -0.0256–+0.0407 |
| COLING 2025 test slice | ConPara (AACL 2026) | +0.0328 | -0.0074–+0.0686 |
| COLING 2025 test slice | DetectRL-X XLM-R (2026) | +0.0808 | +0.0491–+0.1090 |
| COLING 2025 test slice | MAGE author (ACL 2024) | +0.1046 | +0.0645–+0.1497 |
| Intel peer review | Desklib v1.01 (2025) | +0.0490 | -0.0208–+0.1307 |
| Intel peer review | Tropa-mini (2026) | -0.0837 | -0.1307–-0.0304 |
| Intel peer review | ConPara (AACL 2026) | -0.0701 | -0.1082–-0.0309 |
| Intel peer review | DetectRL-X XLM-R (2026) | +0.1610 | +0.1010–+0.2192 |
| Intel peer review | MAGE author (ACL 2024) | +0.1267 | +0.0527–+0.2055 |

## Original frozen populations: regression only

These larger populations include known V3 task fitting/tuning ancestry in the abstract columns. They are retained for transparency and V2-era continuity, not used for the primary chart or independent-generalization claims.

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
| **FiveTrace V3** | **0.9970**<br>99.40% (0.00%) | **0.9465**<br>89.34% (0.00%) | **1.0000**<br>100.00% (0.00%) |
| Desklib v1.01 (2025) | 0.9599<br>100.00% (8.02%) | 0.7989<br>68.21% (7.85%) | 0.9599<br>100.00% (8.02%) |
| Tropa-mini (2026) | 0.7291<br>49.30% (0.00%) | 0.3639<br>2.82% (0.00%) | 0.9366<br>87.37% (0.00%) |
| ConPara (AACL 2026) | 0.9870<br>99.60% (2.20%) | 0.6967<br>45.67% (2.21%) | 0.9890<br>100.00% (2.20%) |
| DetectRL-X XLM-R (2026) | 0.8778<br>87.17% (11.62%) | 0.8448<br>80.68% (11.67%) | 0.9417<br>100.00% (11.62%) |
| MAGE author (ACL 2024) | 0.9336<br>100.00% (13.23%) | 0.6567<br>47.28% (13.28%) | 0.9316<br>99.60% (13.23%) |


## Sources and reproducibility

- [Desklib](https://huggingface.co/desklib/ai-text-detector-v1.01)
- [Tropa-mini](https://huggingface.co/wasitaigeneratedcom/tropa-mini-ai-text-detector)
- [ConPara](https://github.com/SES-Lab-OTH/deberta-conpara)
- [Official DetectRL-X XLM-R](https://huggingface.co/WUJUNCHAO/DetectRL-X-XLM-RoBERTa-Detector-All)
- [MAGE](https://huggingface.co/yaful/MAGE)

Pinned upstream dataset provenance and original source-specific protocols remain in [recent V2 comparison](RECENT_COMPARISON.md) and [standard benchmark notes](STANDARD_BENCHMARKS.md). [V3 source training](REPRODUCING_V3.md) implements the local pipeline; original dataset rows, frozen record-ID manifests and weights are intentionally not distributed. Aggregate results alone do not reconstruct the exact experimental split.
