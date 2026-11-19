"""Student implementation contract for CS781 Phase 1, Submission 03.

Copy this file to ``prefix_hybrid_lm.py`` only after implementing the model
yourself.  The capstone rules prohibit an LLM from writing the whole assessed
architecture or selecting/tuning its scientific hyperparameters, so this file
deliberately contains no model implementation and no numerical configuration.

Required attention semantics for each example with prefix length ``p``:

* prefix query ``i < p`` may attend to every valid prefix key ``j < p``;
* continuation query ``i >= p`` may attend to every valid prefix key and to
  continuation keys ``p <= j <= i``;
* padded keys are never visible.

Required public API:

* ``make_model_config(vocab_size) -> Mapping[str, Any]``
* ``build_model(config) -> torch.nn.Module``
* the model forward method accepts ``input_ids``, ``attention_mask``, and
  ``prefix_lengths`` and returns logits shaped ``[batch, sequence, vocab]``.

You may alternatively expose ``forward_adapter(model, input_ids,
attention_mask, prefix_lengths)`` if your model uses a different internal
signature.  The saved configuration must contain ``vocab_size`` and
``max_sequence_length`` and must reconstruct the exact architecture.
"""

from __future__ import annotations

from typing import Any, Mapping


def make_model_config(vocab_size: int) -> Mapping[str, Any]:
    """Return your frozen numerical configuration.

    Include every value needed to reconstruct the model exactly.  Preserve your
    analytical parameter derivation separately and verify the runtime total is
    between 50,000,000 and 300,000,000 parameters.
    """

    raise NotImplementedError("Student must supply the independently authored numerical configuration")


def build_model(config: Mapping[str, Any]):
    """Construct a freshly initialized model from ``config`` without pretrained weights."""

    raise NotImplementedError("Student must supply the independently authored prefix-hybrid model")

