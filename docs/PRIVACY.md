# Publication boundary

This repository contains selected source, documentation, synthetic tests and aggregate statistics only. No dataset rows, source documents, real example paragraphs, sample-ID lists, individual predictions, embeddings, token sequences, adapters, Base weights, private paths or credentials are included.

prepare_sources.py downloads public pinned resources into ignored local directories. train.py runs local training. detect.py and evaluate.py use local model files; no uploading or repository publishing is implemented. The project implements no telemetry; third-party packages retain their own behavior.

Results are copied by a conservative aggregate-field selection. Private documents and local generation regressions are not included in public source tables. Source-file hashes and public URLs describe upstream provenance, not dataset redistribution. Runtime manifests/predictions may contain record identifiers and must stay local.

PUBLICATION_MANIFEST.json is an explicit reviewed allowlist with file checksums. Gitignore is an additional guard, not a substitute for checking staged files. This release ships source and aggregates, not ready-to-run trained weights; download the upstream Base and train locally.
