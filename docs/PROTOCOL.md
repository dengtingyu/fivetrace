# V3 evaluation protocols

## Primary public external comparison

Use the previously frozen V2-era external benchmark inputs and official public detector checkpoints. Freeze V3 task weights, official native decisions, input/prediction hashes and the selection before external scoring. Do not fit or recalibrate on external labels. Evaluate all original eight plotted conditions; additionally remove known V3 train/development/calibration source-group matches jointly across every detector for the primary comparison. Preserve complete abstract topic pairs and review-paper groups. Original populations remain explicitly separate regression results. See V3_EXTERNAL.md and results/v3_external.json.

These are six external sources with eight conditions, not eight independent datasets. Related source and generator exposure exists. Official detectors have different training corpora, context and native thresholds. Therefore this is an official-checkpoint comparison under disclosed conditions, not equal-data framework training or an unseen-generator/SOTA claim. Historical inspection and feedback prevent claiming a pristine blind test. Unknown semantic duplicates and Base pretraining remain unaudited.

## Numerical execution amendment

The initial BF16 classifier evaluation showed material single-vs-batch logit differences on this device. FP32 inference with autocast/TF32 disabled is now used for published external and A results and the production runtime. This was decided after inspecting original scores, solely to repair numerical consistency; head, fusion, reference weights, sample membership and threshold .5 remain frozen. Original BF16 evidence remains local and is not substituted selectively by column. No refitting or test-label threshold search follows the repair.

## Separate standard protocol A

The newly constructed standard corpus has 26,824 train, 4,000 development, 4,000 calibration and 20,000 test documents, balanced by language and label. Original-source groups are disjoint across phases and across the reference/classifier/fusion roles. Test generation identities are covered in fitting, while test articles and their source groups are independent. Original official-test ancestry is excluded from fitting. Model/epoch/fusion selection uses development, and 1%/5% human-FPR calibration thresholds use only human calibration records. Primary threshold is .5. Require one finite [0,1] score for every exact test ID: missing, repeated, extra or invalid scores fail evaluation.

The V3 classifier-only auxiliary result uses the same selected 6,704-document classifier. Full fusion receives additional 6,708 source-disjoint fitting documents, so the auxiliary result does not isolate each feature group's causal value at identical supervised fitting budgets. Family counts are uneven; report language/domain and equal-family metrics as well as pooled scores. B-family/B-method evaluation is not completed or advertised. No official detector was newly trained on A in this release. A scores are not mixed into the external comparison chart.

## Publication

All data, original record IDs, predictions, caches, adapters and weights stay local. The release contains source, synthetic tests and aggregate metrics. Reconstructing the exact published experiment requires independently obtaining the original frozen data/splits; the custom local-split preparation CLI is not a reconstruction of those source collections. V2 historical documentation is retained separately.
