# V2 protocol

## Fitting recipe

Pinned sources: C-ReD, HC3-Chinese and the fixed CUDRT train file. The downloader verifies complete-file SHA256 and byte counts; file metadata, public URLs and revisions are in results/sources.json. CUDRT exact human prefixes are removed from generated continuations, and HLT is not used as pure-generated training data.

Source grouping ties same-topic C-ReD variants, HC3 questions, exact normalized originals, their160-character canonical prefixes, and exact chosen units. Deterministic family roles: LoRA train35%, LoRA dev5%, head25%, fusion15%, validation10%, test10%. Claude-3.5-Haiku is held out of all fitting/development roles. Sampling matches human/AI source, domain and length strata and spreads AI generators.

LoRA quotas per label: train3000, dev160. Head training4000, fusion1500 and validation800 per label. Test_seen1200 and test_unseen400 per label; some human controls repeat, so do not sum them as independent authors. Group sets are isolated across fitting, development and test roles. Exact historical role counts are in results/training.json; no sample IDs are distributed.

One-epoch rank8 causal-LM adapters use max256-token deterministic crops, batch2, accumulation4, AdamW, peakLR2e-4 and warmup/cosine; minimum development NLL selects each branch. Head C=0.05, fusion C=0.1 with standardization and balanced classes. The public recipe replays the historically selected expanded_lora_1_5 architecture; it does not reselect between frozen and LoRA candidate architectures. Runtime and public evaluation use the fixed0.5 threshold. Historical validation-calibrated cutoffs are not substituted into these external numbers.

## Input and aggregation

At least100 nonspace characters are required. Merge adjacent short source paragraphs; internally split long detection paragraphs into approximately600-character windows and merge sparse tails. Reference contexts are512 tokens with384-token target stride; token zero is excluded. Rank ties use exact full-vocabulary midrank.

Character-weight window scores into paragraph scores, then character-weight paragraphs for the benchmark input/document score. Apply0.5 to classify each input. The user-facing AI percentage instead counts AI-labelled detection paragraphs; internal windows do not add votes. Paragraph metrics inherit source-document labels, not individually annotated authorship.

## Frozen external evaluation

All12,747 locally scorable inputs were processed, but public aggregates exclude private documents/local generation regressions and26known fitting-family matches. The public eligible-source unique count is in results/protocol.json; per-condition counts can reuse human controls. Some original historical records are detection units rather than full articles.

The fresh academic paired collection is pinned to7786b2bd1bfd080c3f16343164c64a3d7cf39e9f. Inspecting source alignment identified LLaMA rephrase index shifts; that condition is AI-only diagnostic and excluded from primary pairing without guessed reindexing. Match controls by deduplicated human input identity. Bootstrap topic families jointly across detectors500times; use Wilson95% intervals for recall/FPR to avoid zero-width zero-error intervals.

Seven fixed model/length strata from the pinned diffusion-generated-text revision each select100prompts by fixed hash, one response per prompt. The deduplicated collection has690inputs and no human controls. Report recall only.

Our V2 uses full input and0.5. Official XLM-R uses both common full-text windows and original first512tokens. MAGE uses common windows, a raw original-context softmax0.5 control, and separately its pinned public preprocessing/class0-logit>3.08583984375 rule. Published MAGE threshold is not fitted here. It is the same detector in multiple modes, not multiple independent methods. Six old English inputs exceed the author-mode4096-token limit; fresh inputs all fit.

Known fitting/development/validation overlap is removed conservatively using canonical exact/prefix and source-family propagation. Semantic overlap, Base pretraining and all MAGE training exposure are not fully ruled out. Source formats, tasks and genres vary, so this experiment does not isolate generation architecture as a causal factor. Historical tests influenced previous designs and become regression checks; they are not pristine future test data.

## Re-evaluate locally

Use evaluate.py with your own local JSONL records containing text and integer label0/1. It uses the same production paragraph aggregation and fixed0.5 boundary and writes only aggregate metrics. Shared controls/families and mixed-authorship conditions require separate grouping and appropriate resampling for formal comparisons. Upstream source links and published aggregate facets are in docs/RESULTS.md and results/external_v2.json. Datasets and frozen per-record ID lists are intentionally not distributed.
