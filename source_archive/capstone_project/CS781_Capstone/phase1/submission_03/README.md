# CS781 Phase 1 - Submission 03

This directory contains Kaggle infrastructure for a student-owned, single-stack
prefix-hybrid legal language model.

The architectural experiment is deliberately different from the earlier two:

- Submission 01: globally causal decoder-only Transformer.
- Submission 02: separate bidirectional encoder and causal decoder connected by
  encoder-decoder cross-attention.
- Submission 03: one shared Transformer stack with a segment-aware mask. Prompt
  tokens are bidirectional; continuation tokens are causal and condition on the
  full prompt. There is no separate decoder stack or cross-attention module.

## Files

- `notebook/CS781_Phase1_Submission03_Infrastructure_FIXED.ipynb`: generated
  Kaggle notebook.
- `notebook/build_submission03_notebook.py`: reproducible notebook builder.
- `model/prefix_hybrid_lm_TEMPLATE.py`: student implementation contract only.
- `LLM_USE_DECLARATION.md`: truthful scope of AI assistance for this artifact.

## Required student work

1. Independently implement `prefix_hybrid_lm.py` from course material.
2. Supply the exact numerical architecture and analytical parameter count.
3. Keep the runtime total within 50M-300M parameters.
4. Choose and justify the corpus cap, optimizer settings, context split,
   training budget, decoding policies, hypothesis, and matched controls.
5. Run the integration and representative Kaggle GPU save/restore tests.
6. Train from scratch using only the official Phase 1 legal corpus.
7. Preserve checkpoints and recovery archives outside the Kaggle session.
8. Generate every official continuation, validate `submission.zip`, submit it
   to Codabench, and record only real results.

Do not set the notebook's student attestations until they are truthful.

