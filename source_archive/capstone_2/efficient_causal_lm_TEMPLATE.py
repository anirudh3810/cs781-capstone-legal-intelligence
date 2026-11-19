"""Student-owned Submission 04 model contract; this is intentionally not a model.

Create an independently authored `efficient_causal_lm.py` with the functions below.
The model must start from random weights, have 50M–300M trainable parameters, and
be trained only on the official Phase 1 legal corpus. The intended architectural
experiment is an efficient causal local/global attention Transformer with
grouped-query attention and RoPE inside attention. BPE is the tokenizer.

Required public API:

* make_model_config(vocab_size) -> mapping including vocab_size and
  max_sequence_length, plus every setting required to reconstruct the model;
* build_model(config) -> new, randomly initialized torch.nn.Module;
* model(input_ids=..., attention_mask=...) -> logits [B, T, V];
* model.prefill_context(input_ids) -> cache;
* model.decode_cached(next_input_ids, cache) -> (last_logits [B, V], new_cache).

Document and test the exact local/global causal visibility rule. Future tokens
must never affect earlier logits. Cached decoding must agree with full forward
logits and keep its memory bounded according to the documented attention rule.
"""

from __future__ import annotations

from typing import Any, Mapping


def make_model_config(vocab_size: int) -> Mapping[str, Any]:
    raise NotImplementedError("Student must provide the complete numerical configuration")


def build_model(config: Mapping[str, Any]):
    raise NotImplementedError("Student must implement the assessed efficient causal model")

