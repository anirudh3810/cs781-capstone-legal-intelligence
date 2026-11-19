# CS781 Phase 4 plan — Court Judgment Prediction

Status: implementation prepared, 15 October 2026. The Phase 4 notebook is built and locally smoke tested with mock Phase 4 cases and the real four Phase 1 recovery checkpoints. Official Phase 4 training has not started because the gated task files are not present locally.

## Verified requirements and current limits

- Task: predict one binary court outcome (`0` or `1`) for each complete case. The primary metric is **macro F1**, so both classes matter. Inputs are `Phase_4_CJPE/Train/train.json`, `Phase_4_CJPE/Train/dev.json`, and `Phase_4_CJPE/Test/test.json`. The test labels are private.
- A submission is a ZIP containing exactly root-level `prediction.json`; that JSON is a list of `{"id": original_test_id, "pred": 0_or_1}` records, one per official test ID.
- At least six substantially different experiments/submissions are mandatory. At least five must use different fine-tuning/adaptation techniques, including **LORMA**; a sixth must be **knowledge distillation** from a larger or stronger teacher. All four of the student's Phase 1 architectures must be represented across the six. Hyperparameter-only changes and head-only changes do not count as distinct methods.
- Every experiment starts from the student's Phase 1 weights. No other pretrained model weights, external legal corpus, or external LLM/API can train the task.
- The local PDF places Phase 4 on 17–21 October 2026 and says it is a dynamic document. The three Phase 4 JSON files are not currently present in the workspace, and the gated Hugging Face repo could not be inspected anonymously. Dataset size, schema details beyond the PDF, class balance, text lengths, and feasibility timings remain unknown. Check the current official documents and data again at implementation time.

Sources: `Capstone Document 1.pdf` section 10 and `Capstone Document 2.pdf` sections 18–19.

## Lessons from Phase 2 and Phase 3

- Phase 2 Span V3 used staged adaptation of more Phase 1 base-03 layers and improved its three-fold local score from the V2 baseline 0.453 to selected hybrid 0.486. Its report warns that source/threshold choices used the same out-of-fold labels, making the local estimate optimistic. The user reported a private Phase 2 score of 0.5276.
- Phase 3's latest HighF1 artifact completed five document folds, but pooled out-of-fold macro F1 was **0.424** versus **0.533** on the ten-document development set. The wide branch was not promoted. Earlier Phase 3 experiments also plateaued near 0.5 privately despite more training. Longer training, a larger head, or repeated tuning on a small dev set was therefore insufficient evidence of generalization.
- Phase 3's rare labels caused a difficult multiclass ceiling, but Phase 4 is binary and needs a fresh data audit. Do not transfer Phase 3 model weights or task labels; transfer the engineering lessons: full-document coverage, real held-out evaluation, conservative selection, and recoverable jobs.

## Data investigation before architecture is frozen

1. Download only the three official Phase 4 files from the pinned Hugging Face revision using the Kaggle `HF_TOKEN` secret. Record revision and SHA-256 hashes. Inspect actual JSON shape and label field; reject duplicate/missing IDs and invalid labels.
2. Count train/dev/test cases and each class; summarize tokens per case under all four Phase 1 tokenizers, fraction exceeding each context limit, document sections, and first/middle/last token coverage.
3. Check exact and near-duplicate case texts across splits, related case IDs, obvious outcome phrases, empty/truncated cases, and whether train and test have comparable lengths. Keep the official dev split untouched for a final check.
4. Implement the official macro-F1 calculation if the course evaluator is accessible; otherwise use `sklearn` macro F1 for labels `[0,1]` and explicitly verify the convention against the evaluation script before submission.

## Common long-document classifier

Use the same document task head and comparable chunk coverage for the five supervised adaptation experiments, changing the required adaptation technique and Phase 1 base. Tokenize each case with its base's own tokenizer. Split into overlapping chunks that fit that base's context window. Always cover the beginning and end; sample middle chunks during training and cover all chunks at inference. A section/position embedding and gated attention pooling over chunk embeddings produce a document representation. Combine attention, mean, and max pooling so one decisive passage is not lost. Train with **case-level** binary cross-entropy: every chunk in a case contributes to one outcome, rather than being assigned a noisy copy of the case label. Use class-balanced document sampling or modest class weights only if the audit confirms imbalance.

For the causal bases 01 and 04, put a classification sentinel after the chunk and pool its final state; for base 03, use its prefix-hybrid attention to pool the chunk; for base 02, pool encoder states. Avoid a fixed first-1024-token truncation. Compare beginning-only, ending-only, and complete hierarchical coverage as controlled ablations; these are diagnostic comparisons, not substitutes for the five distinct adaptation methods.

The four Phase 1 checkpoints contain approximately 107M (base 01), 61M (base 02), 52M (base 03), and 68M (base 04) parameters. These are measured from the saved Phase 1 state dictionaries. The architecture/head details will be finalized after the Phase 4 length and class audit.

## Six required experiment candidates

| Submission | Phase 1 base | Distinct method | Planned role |
|---|---|---|---|
| 1 | 01, 107M | Full fine-tuning | Strong supervised teacher candidate with hierarchical document aggregation. |
| 2 | 03, 52M | Partial, layer-wise fine-tuning | Train upper pretrained blocks and document head; compare with full adaptation. |
| 3 | 04, 68M | LORMA | Mandatory multiplicative low-rank adaptation on attention/MLP layers, with the same document objective. |
| 4 | 02, 61M | AdaLoRA | Adaptive-rank parameter-efficient adaptation, primarily on its encoder. |
| 5 | 03, 52M | Bottleneck adapters | Insert trainable adapters while freezing most Phase 1 weights. |
| 6 | 03, 52M student | Knowledge distillation | Train from the Phase 1 base 03 plus case labels and soft logits from the larger base-01 teacher; use fold-safe teacher predictions for validation. |

The table satisfies five techniques + one distillation experiment and covers all four bases. If a method fails its validation gate, improve that method or run an additional genuinely different method; do not relabel a learning-rate change as a new submission. Each final submission must yield its own prediction file. An ensemble may be tested separately but should not obscure the six required method comparisons.

## Validation, model selection, and performance gates

1. Use stratified **case-level** outer folds within `train.json`; keep all chunks of a case in one fold. If duplicates/related cases are found, group them before splitting. Start with a fixed pilot split to measure throughput and validate I/O, then use at least three outer folds for every final submission candidate. The number of folds can be increased if dataset size and time permit.
2. Within each outer training portion, make an **inner validation split** for checkpoint selection and threshold calibration. Never select an epoch, threshold, ensemble weight, or head using the outer held-out cases being scored. This removes the Phase 2/3 source of optimistic out-of-fold estimates. Choose checkpoints by inner macro F1, not training loss.
3. Freeze a threshold from the inner split using a small predeclared grid, then predict the outer held-out fold once. Pool those untouched predictions to report macro F1, both class-specific precision/recall/F1, confusion matrices, calibration, length-bin scores, and fold dispersion. Use the median inner-selected threshold for the final model. Evaluate the frozen method/threshold on official dev once as a transfer check. A large outer-fold/dev gap triggers a data-leakage or shift investigation before submission.
4. Keep majority-class and Phase-1 frozen-feature baselines for context; these diagnostic baselines do not count toward the six methods. A model is a strong submission candidate only if it beats both baselines in held-out macro F1, does not collapse on either class, and has a compatible official-dev result. A score cannot be guaranteed before Codabench evaluates the private test.
5. Distillation uses teacher models trained only on each training fold when evaluating the student on that fold. For the final fit, train the teacher on allowed final training data and distill the student from its logits with a supervised label loss. Compare student F1, parameter count, memory, and inference time with the teacher.

## GPU and runtime plan

- Use two-GPU DistributedDataParallel with one process per GPU, mixed precision, dynamic token-bucket batches, pinned-memory prefetch, gradient accumulation, activation checkpointing, and a low backbone learning rate with a higher document-head learning rate. Benchmark one epoch before choosing chunk count and per-slot epoch budgets. Measure actual utilization and tokens/sec; adjust bottlenecks based on the measurements.
- Train sufficiently long to reach a held-out F1 plateau, with warmup, LR decay, clipping, and early stopping. Longer training is conditional on F1 improving; the Phase 3 curves show that lower training loss alone is not a reason to keep training.
- Reserve compute for all six methods and validation, rather than exhausting the session on the first model. Six full experiments plus folds may exceed one Kaggle session. Make the future Run All notebook idempotent across sessions: completed stages are reused, incomplete stages resume, and an early finished submission is available before later experiments run.

## Crash recovery and output contract for the future notebook

- Save model trainable state, optimizer, scheduler, scaler, RNG state, fold assignment, data/config hashes, OOF probabilities, metrics, test predictions, and per-stage status atomically every 10–15 minutes. Maintain a known-good prior checkpoint. Archive only after DDP workers exit; the Phase 3 Sections run hung when archiving inside distributed workers.
- Stage the four Phase 1 weights once, share immutable bases, store completed fine-tuned deltas or selected model states, and discard completed optimizer states after final verification to limit disk use. The final **all-artifacts ZIP** must include enough information to reconstruct all six outputs without another HF download, plus source/config/manifest and checksums. Verify a restore in a clean directory before calling it complete.
- A ZIP in `/kaggle/working` alone does not survive every fatal Kaggle session. Persist or download completed recovery archives between sessions (or save a Kaggle version); never claim full crash immunity from local checkpointing alone. `HF_TOKEN` is for reading the course data and does not imply permission or capability to upload artifacts elsewhere.
- Create six distinct `submission_XX.zip` packages, each with only root-level `prediction.json`; validate exact test ID set, one row per ID, integer `pred` in `{0,1}`, ZIP contents, and round-trip JSON. Record local F1 and eventual Codabench score for every slot, without representing local scores as private results.

The implementation is in `../CS781_Phase4_Six_Submissions_RunAll.ipynb`, with source and local tests in this directory. Its first Run All stage audits the released data; model fit and private score remain unverified until that Kaggle execution.
