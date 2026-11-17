> Historical V2 release. These scores and architecture are not V3.

# Method and resource model

## Corpus-trained reference features

Let q_base be a frozen Qwen3-1.7B-Base. Train q_AI and q_human by causal likelihood on disjoint, labelled AI and human corpora through rank-8 LoRA adapters. The deployed reference is one backbone with separately switchable adapters. No target generator is loaded or queried during detection.

For each input window, teacher forcing supplies token negative log likelihood and exact full-vocabulary midrank statistics under Base, AI-LoRA and human-LoRA. Include human-minus-AI differences. These are proxy-reference measurements; they are not the original generator's true likelihood or logits. The current AI branch mixes generator corpora rather than training one adapter per generator.

Group 1 is a logistic head on frozen Base hidden representations. Groups 4 and 5 describe sentence lengths and lexical distributions. A fitted StandardScaler/logistic model combines the head logit, 32 likelihood/rank values and 10 style values. The runtime threshold is 0.5.

## What is saved

- Required deployed target-generator weights: none.
- Required target-generator API calls during detection: zero.
- Reference backbone: one Qwen3-1.7B-Base, with two small adapters.
- Reference computations: Base plus both adapter passes; exact rank calculation still touches the full vocabulary. Sliding contexts may add passes for long windows.

The benefit is removal of target-generator deployment/API dependencies. Collection of representative corpora, training, local GPU computation and storage still cost resources. No controlled latency, throughput or monetary comparison against target generators has been measured. Local adapters do not make an arbitrary unseen generator reliably detectable.

## Measured local training

Each adapter has 1,605,632 trainable parameters. AI and human adapters each use 3,000 training units, 160 development units, rank8/alpha16, q_proj/v_proj, dropout0.05, BF16 causal-LM training, max256 tokens, batch2 with accumulation4, and one epoch. Select the checkpoint with minimum class-specific development NLL. The stored AI best step is375; human best step300, from375 total updates each. The two LoRA phases (including their development evaluations) took 476.6 + 495.1 seconds, about16.2minutes, on an8GB RTX5060 Laptop; about4.18GiB peak allocated CUDA memory. This excludes preceding tokenization/model loading, feature extraction, classifier/fusion fitting, downloads and benchmarks. No reliable single end-to-end training time was retained; cached/resumed fitting timings cannot establish it. Group1 fits a logistic head on8,000 labelled frozen-Base representations; fusion uses3,000 separate units. See results/training.json.

## Scope of the contribution

The project uses generator corpora to learn compact, class-conditional likelihood/rank references, combined with three additional evidence groups. The deployed model stays fixed in size while its corpus coverage can be changed offline. Proxy scores describe the trained local references and are not the target generator's true likelihood. New corpus mixtures need their own independent validation.
