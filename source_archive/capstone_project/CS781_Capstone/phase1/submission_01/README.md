# CS781 Phase 1 — Submission 1

This directory contains the reproducible end-to-end implementation for the first Phase 1 submission. The main artifact edited for execution is:

- `../../../CS781_Phase1_Submission01_Infrastructure_FIXED.ipynb`

The notebook contains gated Hugging Face acquisition, streaming JSON inspection, compressed disk-backed document preprocessing, a deterministic byte-level BPE integration, an approximately 82M-parameter decoder-only Transformer, a mixed-source causal-LM training loop, checkpoint resume support, evaluator integration, KV-cached generation, and packaging. It does not contain trained weights, fabricated results, or scientific conclusions. Tokenizer training, full model training, and official inference remain explicitly gated.

## Recommended experiment framing

The implemented intervention replaces the conventional GELU feed-forward sublayer with a parameter-matched SwiGLU sublayer. The recorded control keeps the official data, tokenizer, parameter/training-token budget, split, optimizer, schedule, and evaluation procedure fixed.

The experiment must still be run and compared against the recorded control before any scientific conclusion is made.

## Fresh Kaggle session

1. Either attach the official Phase 1 data as a private Kaggle dataset or enable the notebook's Hugging Face download gate. Add `HF_TOKEN` as a Kaggle secret if access is restricted. Attach the official evaluator and, when resuming, exactly one recovery archive.
2. Enable the intended GPU accelerator.
3. Run notebook sections 1–3.
4. Inspect the printed repository paths, local paths, file sizes, JSON layouts, and sampled schemas. Resolve duplicate copies through explicit file overrides.
5. Review the 5% source-stratified document validation split, compressed document-store size, overlap audit, per-source counts, and balanced document-draw mixture; preserve the manifests.
6. Restore the registered 32k byte-level BPE tokenizer, or explicitly authorize its one-time training from `TRAIN_DOCS` only.
7. Build the decoder model and inspect the parameter, causality, and reconstruction validation reports.
8. Run the disposable infrastructure/model tests, followed by the gated 1024-token GPU optimizer-and-checkpoint smoke test.
9. Review measured GPU memory, disk use, hypothesis, ablation, and training budget; set both explicit full-training authorization gates, then run the expensive training cell.
10. Load the supplied official evaluator and evaluate only on the fixed local development examples.
11. Generate official test continuations, validate all IDs and exact 128-word outputs, and create `submission/submission.zip`.
12. Create `recovery/phase1_submission01_recovery.zip`, then run final verification.

## Resume after interruption

1. Attach one previously validated recovery archive and the same official datasets.
2. Run environment, discovery, and recovery sections.
3. Validate and extract the selected archive.
4. Reconstruct the registered tokenizer and verify its fingerprint.
5. Reconstruct the model from the preserved source/configuration.
6. Recreate the optimizer, scheduler, scaler, loader, and stateful sampler.
7. Call the checkpoint restore utility and verify dataset/tokenizer fingerprints. A history tail written after the latest valid checkpoint is backed up and rolled back only when it is a strict byte-for-byte extension of that checkpoint's history.
8. Continue through the student's training function from the restored step/epoch/batch cursor.
9. Download the automatic training recovery snapshot at milestones and export a newly validated final recovery archive before the session ends.

## Output distinction

- `submission/submission.zip` is the Codabench package and contains only `prediction.json` at the ZIP root.
- `recovery/phase1_submission01_recovery.zip` is the internal reproducibility archive and must never be uploaded as the Codabench submission.

## Missing source materials

At notebook creation time, the local workspace did not contain the official Phase 1 JSON files or official metric scripts. The notebook therefore supports a gated download from the professor's Hugging Face repository, searches both Kaggle inputs and the download directory, refuses ambiguous matches, and requires explicit evaluator function/argument configuration after inspection.

