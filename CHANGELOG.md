# Changelog

## 3.0.0 — 2026-11-17

- Replace feature 1 with a fully fine-tuned multilingual mDeBERTa-v3-base classifier. Keep its continuous logit for fusion with Qwen corpus-proxy loss/rank and sentence/lexical evidence.
- Train fresh Chinese/English AI and human reference adapters, with separate source groups for the classifier, references and fusion.
- Compare the frozen V3 detector with five official open checkpoints on six existing external sources and eight conditions. V3 has the highest Macro-F1 point estimate in six conditions; ToolBenchmarks and Intel peer review remain weaknesses. Report AI recall, human FPR and paired confidence intervals alongside Macro-F1.
- Exclude known V3 fitting ancestry from every detector's primary inputs, preserve complete pairs and retain the original populations as regression aggregates. This is a source-familiar official-checkpoint comparison, not a blind or equal-data ranking.
- Repair BF16 classifier batch-shape sensitivity with FP32 inference and disabled TF32/autocast. Recompute external and protocol-A results without changing task weights, fusion or thresholds; disclose that original scores had already been inspected.
- Publish a V2-style comparison figure, English/Chinese READMEs, a public aggregate-only figure renderer, local split validation and the V3 training/inference recipe. Preserve the historical V2 documents and results.
- Distribute source and aggregates only. Datasets, writing samples, record identifiers, individual predictions, caches, adapters and model weights stay local. See `PUBLICATION_MANIFEST.json` and `docs/PRIVACY.md`.

## 2.0.0 — 2026-11-14

Historical corpus-trained Qwen reference scoring and fusion release. See the archived [V2 release](docs/V2_RELEASE.md) and [Chinese version](docs/V2_RELEASE.zh-CN.md).
