# Phase 1 model implementation

This directory now contains the source embedded and imported by the corrected notebook:

- `course_bpe.py` implements the deterministic byte-level BPE adapter and persistence contract.
- `decoder_lm.py` implements the Pre-LayerNorm causal decoder with learned positions, tied embeddings, and a SwiGLU feed-forward sublayer.

The notebook expects a PyTorch model that:

- accepts token IDs shaped `[batch, sequence]`;
- returns vocabulary logits shaped `[batch, sequence, vocabulary]`;
- can be reconstructed from a JSON-serializable model configuration;
- loads its own `state_dict` strictly;
- optionally exposes a deterministic greedy `generate` interface.

Before training, the notebook checks forward/backward execution, finite logits and gradients, causal future-token isolation, context length, vocabulary compatibility, strict reconstruction, and the official 50M–300M parameter constraint.

Do not place pretrained language-model weights here. Preserve these exact sources and the tokenizer fingerprint with every recovery package.

