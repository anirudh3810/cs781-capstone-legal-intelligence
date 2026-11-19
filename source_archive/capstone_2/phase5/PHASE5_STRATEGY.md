# CS781 Phase 5 strategy: Legal Statute Identification

Prepared 21 October 2026. This is a plan, not a claim of trained Phase 5 performance. The official Phase 5 JSON files and evaluator are not in this workspace yet. The course document schedules Phase 5 for **22-26 October 2026**, ending **26 October, 11:59 PM IST**; check the dynamic course document for changes when the phase opens.

## 1. What the submission must do

- Predict a **set of integer statute IDs** for each test case: `[{'id': original_id, 'pred': [id1, id2, ...]}, ...]` in a root-level `prediction.json` inside a ZIP. IDs must be valid, unique within each case, and cover the exact official test ID set.
- Optimize the official **macro F1** over statute labels. Micro F1 and subset accuracy are secondary. Check the evaluator's label-universe and zero-division behavior before reproducing it locally; the PDF says its label dimension comes from the reference set.
- Produce at least six substantially different experiments/submissions. Five need different fine-tuning/adaptation methods, **LORMA is compulsory**, one must use **knowledge distillation**, and the set must use all four of the student's Phase 1 architectures. Changes only to learning rate, epochs, rank, or head do not count as new techniques.
- Start every counted experiment from the student's Phase 1 weights. Do not load another pretrained model, external legal corpus, statutes, or external LLM/API for training.

Primary source: `../Capstone Document 1.pdf`, section 11. `../Capstone Document 2.pdf` supplies the archive and ID checklist. The document is dynamic, so recheck it on release day.

## 2. What the previous phases actually show

| Evidence | Consequence for Phase 5 |
|---|---|
| Four recoveries contain real Phase 1 checkpoints: base 01 decoder-only **82.0M** parameters, base 02 encoder-decoder **61.3M**, base 03 prefix hybrid **52.2M**, base 04 local/global causal **55.1M**. Base 03 has bidirectional prefix attention; base 02 has a bidirectional encoder. | Reuse these exact sources, checkpoints, and each checkpoint's own tokenizer. Classification should use the bidirectional path for 02/03. Parameter counts should come from checkpoint manifests, not an older planning estimate. |
| Phase 2 Span V3 improved its V2 three-fold local F1 around **0.453** to selected hybrid around **0.486**, but its own report says epoch/source/threshold selection reused out-of-fold labels. | Use independent selection and evaluation data; never quote a tuned OOF score as an unbiased estimate. |
| Phase 3 HighF1 artifact reported pooled OOF macro F1 **0.424** and official-dev **0.533**. Other Phase 3 notes show a fallback package could repeat prior predictions despite a new run. | Save both held-out predictions and exact prediction hashes. Require a new package to differ and avoid selecting by a small dev-only gain. |
| Phase 4's main artifact has four model reports (full/base01, partial/base03, LORMA/base04, AdaLoRA/base02), with dev macro F1 **0.730, 0.755, 0.665, 0.612**. Every report is marked **PROVISIONAL_0_FOLDS** and private F1 is unknown. Separate Phase 4 ZIPs for slots 5/6 exist, but their training reports are not in the main artifact. | Base 03 partial and base 01 full are first pilots, **not** guaranteed Phase 5 winners. Complete validation before claiming generalization. |
| Phase 4 data had 37,387 train / 3,505 dev / 1,517 test rows, repeated case IDs/text, and roughly 26% of train and 40% of dev/test cases exceeded eight 640-token chunks under its token audits. | Treat these numbers only as a proxy until Phase 5 is released. Group duplicates and related IDs, and explicitly measure how much evidence any chunk budget misses. |

Sources inspected: the four `../phase1_submission0*_recovery.zip` manifests, `../CS781_Phase2_Span_V3_Report.zip`, `../CS781_Phase3_HighF1_All_Artifacts.zip`, `../CS781_Phase4_All_Artifacts.zip`, `../phase4/cjpe_runner.py`, the Phase 2 method modules, the Phase 3 high-F1 runner, and Phase 1 model sources.

## 3. Release-day data audit: decide the architecture from these measurements

1. Pin the official dataset revision. Download only `Phase_5_LSI/Train/train.json`, `Train/dev.json`, and `Test/test.json` through the existing `HF_TOKEN` pattern. Save revision, SHA-256, schema, row counts, original ID types, and a sample of real rows. Verify no labels are present in test.
2. Build a **stable row key** separate from the original case ID. Count repeated IDs, exact text duplicates, near duplicates, contradictory label sets for identical text, and overlaps across train/dev/test. Group related cases before any split. Never deduplicate test IDs in the output.
3. Determine the label universe: number of statute IDs, positives per label, how many labels appear only in dev, labels per document, co-occurrence, and train/dev prevalence shift. Report bins such as 1-2, 3-9, 10-29, and 30+ positive training documents. An ID absent from training cannot be learned reliably from supervised labels.
4. Tokenize with **all four original tokenizers**. Record median, p90/p95/p99 document length, fraction covered by each proposed chunk policy, and where statute citations/section references occur (beginning, middle, end). Inspect a sample of long positives and errors manually without using test labels.
5. Obtain the official Phase 5 evaluation script when available. Match its exact macro-F1 label dimension, behavior for absent labels/empty predictions, and ID validity rules. A local `sklearn` score is only a provisional proxy until this is checked.

**Decision gate:** if the label count is moderate (roughly hundreds), use a dense multi-label output layer. If it is very large, benchmark a train-derived candidate-label stage, but use it only if held-out candidate recall is near complete, including rare labels. Numeric statute IDs have no inherent text meaning; do not pretend their embeddings contain statute descriptions.

## 4. Recommended model for the main submission

Use a **Phase 1 based, evidence-aware hierarchical multi-label classifier**. This retains one document-level decision while allowing a statute to be triggered by a passage deep in the case.

1. Split a case into overlapping tokenizer-specific chunks. Reserve at least one beginning and one ending chunk. During training, add middle chunks chosen by a label-agnostic policy: coverage across document positions plus citation/Act/section-density windows. Rotate sampled middle windows across epochs. At inference, cover all chunks when feasible, or use a **validated** retrieval budget with a measured coverage/recall curve. Do not silently retain a fixed first eight chunks.
2. Pass each chunk through its Phase 1 backbone. For base 03 use its bidirectional prefix mode; for base 02 use encoder states; for causal bases 01/04 use the final valid token or an explicit classification marker. Preserve chunk position and coarse section cues where they can be detected from the document itself.
3. Project chunk vectors to a shared width, apply a small ordered chunk encoder or gated attention, and form a global document vector. Add **label-specific evidence logits** from each chunk. A simple starting rule is `document_logit[label] = global_head(document)[label] + max_chunk(evidence_head(chunk)[label])`. Compare max with a top-two average on validation. This lets one decisive cited provision survive global pooling without assigning every chunk the document's full label set.
4. Train with **one multi-hot target per case** and document-level `BCEWithLogitsLoss`. Give every case equal weight regardless of length. Start with unweighted BCE; compare capped/square-root positive weights and asymmetric loss only as controlled ablations. Check macro F1 by label-frequency bin so a rare-label improvement does not merely create many false positives.
5. Maintain a cheap train-only TF-IDF/character n-gram one-vs-rest diagnostic baseline and a citation-density coverage baseline. These are analysis controls, not counted submissions. If the sparse signal adds held-out value, a small residual lexical feature branch can be tested inside the Phase 1 based model; it must use only official Phase 5 training text.

This is a conditional design: actual chunk budget, width, loss weights, and evidence selection must be set from the release-day audit and throughput pilot. An optional label-co-occurrence head is justified only if it improves grouped held-out macro F1; it can otherwise suppress independent rare labels.

## 5. Six counted submissions

Use the **same input coverage, output head, case-level objective, and evaluation protocol** as far as each base permits, so the differences really measure adaptation. The prior Phase 4 dev ranking motivates the order, but does not predict Phase 5 scores.

| Slot | Phase 1 base | Distinct technique | Why it is in the set |
|---|---|---|---|
| 01 | 03 prefix hybrid | **Partial/layer-wise fine-tuning** of upper blocks plus task head | First score candidate: bidirectional document reading, good Phase 4 provisional dev result, lower compute than full adaptation. |
| 02 | 01 decoder-only | **Full fine-tuning** plus task head | Larger 82M teacher candidate and a controlled full-adaptation comparison. |
| 03 | 04 local/global causal | **LORMA** on selected attention/MLP projections | Mandatory method and fourth-family representation; verify the multiplicative update is real and trainable. |
| 04 | 02 encoder-decoder | **Bottleneck adapters** in the encoder plus task head | Covers the encoder-decoder family while retaining its bidirectional encoder. |
| 05 | 03 prefix hybrid | **AdaLoRA** on selected attention/MLP projections | Fifth adaptation technique on the strongest initial backbone candidate. |
| 06 | 03 prefix hybrid student | **Knowledge distillation** from the best eligible stronger/larger trained teacher (initially slot 02) | Multi-label soft-target transfer plus hard labels; quantify quality and inference cost. |

If a method fails to learn, repair its implementation or use another genuinely distinct permitted technique (for example QLoRA or prefix tuning), and preserve the five-technique rule. A probability ensemble can be a **seventh, optional** score attempt after the six counted methods are complete; it cannot replace one of the required techniques.

## 6. Validation and thresholding: where macro F1 is won or lost

- Freeze one **grouped multi-label holdout** from training before trying architectures. Keep duplicate texts, repeated IDs, and related cases together; balance common labels approximately across groups. For labels with fewer than three positive groups, document the impossibility of stable fold estimates instead of manufacturing a balanced score.
- Every slot first runs a small throughput/integrity pilot, then trains on the same development protocol. Choose epoch and loss settings on an **inner** training split; score the untouched grouped holdout once. Keep the official dev split for transfer and threshold checks. If time permits, run three grouped outer folds for the top two or three methods; prioritize six genuine finished submissions before broad cross-validation.
- Report **official-style macro F1**, micro F1, subset accuracy, precision/recall, predicted versus actual label cardinality, score by label-frequency bin and document length, and fold dispersion. Compare the Phase 1 model to sparse diagnostic and common-label baselines. Keep confusion/co-occurrence audits for frequently mixed IDs.
- Start with a **global probability threshold** chosen on inner validation. For labels with enough positive examples, allow regularized per-label thresholds shrunk toward the global threshold; rare labels retain the global threshold or a prevalence-based prior. Tune a small predeclared grid. Never tune hundreds of independent thresholds against a tiny dev split. Record whether any top-k/cardinality correction really improves held-out macro F1 before using it.
- Use any held-out fold only once for scoring. If pooled OOF predictions are later used to choose final thresholds, label that OOF score as **selection-biased** and retain the original fold scores produced with inner-selected thresholds as the honest estimate. The official dev may be used for final threshold selection because the course permits it, but a dev score after such tuning is not an unbiased prediction of private F1.
- Distillation validation must train the teacher without labels from the student evaluation fold. Compare hard-label-only student and distilled student on the same fold. Do not let the teacher's failure on rare statutes erase student recall; inspect per-label teacher F1 before assigning a large soft-loss weight.

## 7. Training and compute budget

1. Benchmark preprocessing plus 500-1,000 cases with one representative backbone and 2/4/8 chunks. Log tokens/sec, cases/sec, GPU utilization on both devices, memory, and time per validation pass. The Phase 4 histories showed long per-case epochs and low reported peak memory, so batch multiple cases and flatten their chunks where possible; do not assume DDP is faster without timing it.
2. Pre-tokenize each base once. Use length buckets and dynamic token budgets, mixed precision, gradient accumulation, gradient clipping, warmup/decay, and activation checkpointing only when needed. Use a lower backbone rate and higher new-head rate, then let held-out F1 choose the checkpoint. Training loss alone is not the stopping metric.
3. Estimate a slot's full runtime from the pilot before starting it. Run both GPUs as one DDP job when scaling is measured; if communication/data-loader overhead dominates, run independent slots on separate GPUs. Reserve the last day for calibration, restore tests, and actual Codabench evaluation rather than consuming it with another training run.
4. Make every stage idempotent and resumable. Save selected trainable weights, optimizer/scheduler/scaler/RNG states, data and source hashes, group split, per-label metrics, dev/holdout/test probabilities, and status atomically. Archive **after** distributed workers exit. Persist or download a recovery ZIP between Kaggle sessions; a file left only in `/kaggle/working` can be lost after a fatal session.

### Suggested release-week sequence (IST)

| Date | Deliverable |
|---|---|
| **22 Oct** | Verify current rules/evaluator, audit Phase 5 data, build exact output validator and sparse diagnostic, run chunk-coverage/throughput pilot, start slot 01. |
| **23 Oct** | Complete and validate slots 01 and 02; obtain first two valid ZIPs and archived probabilities. |
| **24 Oct** | Complete mandatory LORMA slot 03 and encoder-decoder slot 04; diagnose rare-label errors, not just headline score. |
| **25 Oct** | Complete AdaLoRA slot 05 and distillation slot 06; run fair grouped validation for leading candidates and freeze calibration. |
| **26 Oct** | Refit selected final models if justified, verify recovery and all six distinct ZIPs, submit early enough for Codabench's 15-30 minute or longer queue, record private scores, and use any remaining allowed attempt only for a validated improvement. |

This timetable is contingent on dataset size and Kaggle GPU availability. If the pilot predicts more compute than available, reduce training chunk budget using measured coverage, shorten weak methods after honest holdout evidence, and prioritize completing six legitimate methods plus a strong slot 01/02. Do not relabel a hyperparameter change as a new submission.

## 8. Packaging and final evidence

For **each** `submission_XX.zip`, open the archive and require exactly root-level `prediction.json`; parse round-trip; check exact test ID multiset, no duplicate rows, integer statute IDs only, valid label universe, no duplicate ID inside `pred`, and prediction hash different from every other counted submission. Save a JSON report with method, Phase 1 checkpoint hash, data revision, trainable parameter count, selected epoch, threshold policy, grouped held-out metrics, dev metrics, test label-count distribution, runtime, and ZIP SHA-256. Keep a reconstruction archive with the exact model/tokenizer sources, Phase 1 provenance, adapter/full checkpoint, configs, label mapping, and probability arrays. Restore one package in a clean directory before finalizing.

Record Codabench scores separately from local estimates. The private test F1 is unknown until scored. Include truthful AI-assistance disclosure in the final course report, consistent with the workspace's existing `../phase2/LLM_USE_DECLARATION.md`.
