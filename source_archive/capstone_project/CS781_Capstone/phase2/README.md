# CS781 Phase 2 — six-submission Kaggle notebook

Open `CS781_Phase2_Six_Submission_Infrastructure.ipynb` in Kaggle. The runner
is embedded in notebook cells. Attach `phase2_method_common.py` and all six
`phase2_method_XX.py` modules as Kaggle input files.

## What is included

- Exact-name discovery of the six official Phase-2 files and all four Phase-1
  recovery ZIPs under `/kaggle/input`, with conflict errors for duplicates.
  Missing Phase-2 files are downloaded using the enabled Kaggle `HF_TOKEN`
  secret from the gated official Hugging Face dataset. If any file must be
  downloaded, attached Phase-2 files are cross-checked against that pinned
  dataset revision.
- Selective staging of model source, tokenizer, and selected checkpoint from
  each Phase-1 ZIP; ZIP CRC and SHA-256 checks. The older 01/02 archives lack
  internal SHA manifests; their assurance is correspondingly weaker.
- Strict Phase-2 document/gold-ID, 12-label, character-offset, and optional
  entity-text validation. The released JSON schema must still be inspected.
- Deterministic document-level validation split, a local strict macro-F1
  approximation, and six-method/all-four-base rule checks.
- GPU detection, mixed precision, all-visible-GPU DataParallel, fused AdamW
  when supported, a representative smoke test, stateful batches, complete
  optimizer/scheduler/scaler/RNG checkpoints, known-good previous checkpoints,
  and verified per-experiment recovery ZIPs.
- Five distinct NER adaptations (full, partial, additive LoRA, multiplicative
  LoRMA+, adapters) and one distillation experiment using the completed
  slot-1 NER model as the larger teacher. Each uses a BIOUL tag head plus
  character-boundary offsets so predictions can end inside BPE tokens.
- Per-ID resumable test predictions, strict output validation, and six distinct
  root-level `prediction.json`/`submission.zip` packages. Incomplete or
  duplicate predictions block the six-submission completion marker.

## Kaggle inputs and Run All

1. Accept access to `Exploration-Lab/CS-781-Capstone` on Hugging Face. Add a
   Hugging Face read token as an enabled Kaggle Secret named exactly `HF_TOKEN`
   and enable Kaggle Internet. Run All downloads the six Phase-2 files. As an
   offline alternative, attach `fold1-text.json`, `fold1-gold.json`,
   `fold2-text.json`, `fold2-gold.json`, `labels.txt`, and `fold3-text.json`
   as Kaggle Inputs; the notebook will not request a token if all are present.
2. Attach all four **your own trained** Phase-1 recovery ZIPs and one copy each of
   `phase2_method_common.py` and `phase2_method_01.py` through
   `phase2_method_06.py` as a private Kaggle dataset. Do not attach duplicate
   basenames from multiple datasets; exact-name discovery refuses ambiguity.
3. Inspect actual gold JSON structure and the official evaluation script. The
   notebook intentionally raises if offset semantics or fields differ from
   what it can validate. Do not normalize away original character positions.
4. Enable a Kaggle GPU. `RUN_PHASE2=True` and `RUN_MODE='auto'` are already set.
   Run All trains and packages completed slots sequentially. Both GPUs are used
   for batched forward/backward when Kaggle presents two CUDA devices; actual
   utilization and timing must be measured on the real T4 environment.
5. Download every generated `phase2_submissionXX_recovery.zip` before the
   Kaggle session expires. Attach those recoveries in the next session and Run
   All again. Auto mode verifies and resumes their checkpoints. It does not
   silently restart an incomplete slot at step zero.
6. For generation without training data, attach the four Phase-1 archives and
   six completed Phase-2 recovery ZIPs. The Hugging Face downloader fetches
   `labels.txt` and `fold3-text.json` unless you attach them.
   Set `RUN_MODE='inference_only'` and `RESUME_EXPECTED_SLOTS={1,2,3,4,5,6}`.
   The notebook restores each frozen training configuration from its selected
   checkpoint and does not open the four training/gold files.
7. When `six_submission_validation.json` reports six structurally valid and
   distinct packages, download each slot's `submission.zip` and submit each
   separately to Phase 2 on Codabench. Record actual acceptance and scores.

**Academic integrity:** These six methods and their initial settings were
AI-assisted. `STUDENT_AUTHORSHIP_CONFIRMED=False` is deliberately unchanged.
The course documents restrict LLM authorship of assessed model architecture,
tuning, and conclusions. Do not represent this as independently student-authored
work; ask the instructor whether and how it may be used. The notebook does
**not** contain trained Phase-2 models, promise six outputs in one session,
guarantee full GPU utilization, or upload to Codabench. It recovers to the
last verified checkpoint.
