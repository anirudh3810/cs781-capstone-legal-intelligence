# Phase 1 Submission 04

This folder contains a separate Kaggle notebook and a contract for the fourth
architecture experiment: an efficient causal Transformer using local/global
attention. Grouped-query attention and RoPE are part of the proposed model
design; BPE supplies token IDs.

The notebook inherits official-data schema inspection, a bounded disk cache,
deterministic splitting, two GPU training support, mixed precision, fused AdamW
when available, complete checkpoints, portable recovery archives, optional local
evaluation, resumable prediction generation, and validated Codabench packaging.

An AI-assisted reference `efficient_causal_lm.py` now implements a 14-layer,
55.1M-parameter causal local/global Transformer with GQA, RoPE, and bounded
cached decoding. The course document prohibits LLM authorship of the whole
assessed model and hyperparameter tuning. Do not represent this reference as
independently student-authored; the notebook keeps those declarations false.

The configured target is 15,000 optimizer steps, with a 9.25-hour per-session
guard. A Kaggle smoke test measures throughput and VRAM; there is no
guarantee the target fits within ten hours or reaches a particular score.

## Kaggle run order

1. Attach the four official Phase 1 training files, official test prompts, and
   `efficient_causal_lm.py` under `/kaggle/input`. The first run trains a fresh
   25k byte-level BPE tokenizer on the bounded official training split.
2. Read the updated LLM-use declaration and course policy. Independently
   revise/author the assessed architecture if required; do not set false
   student-ownership attestations.
3. Select Kaggle T4×2 and Run All. `execution_mode="auto"` performs preprocessing,
   model checks, a GPU smoke test, and then training. Read the smoke report:
   15,000 steps are a target, not a ten-hour guarantee.
4. Download the verified `training_recovery.latest.zip` or final recovery ZIP.
   Attach it to a later Kaggle session and Run All to resume unfinished training
   or prediction generation. Every successful official ID is saved immediately.
5. `submission.zip` is packaged only when every official ID has a validated
   128-word continuation. Download that ZIP and upload it manually to Codabench;
   record the real acceptance and score.

RoPE is positional encoding inside attention; BPE is the tokenizer. Neither
accuracy nor completion of all test predictions is guaranteed by this
infrastructure. Unresolved IDs are saved for resumption and prevent an invalid
submission ZIP.

