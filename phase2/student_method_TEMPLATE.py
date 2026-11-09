"""Student-owned Phase 2 method contract. This file is NOT a working NER method.

Create six independently authored phase2_method_01.py ... phase2_method_06.py
files. At least five must implement genuinely distinct non-distillation
adaptation techniques, including LORMA, and one must implement knowledge
distillation. Use the selected Phase-1 model supplied by the notebook.

Every module must expose the functions below. The returned torch.nn.Module's
forward(**batch) must return a finite loss tensor or {'loss': loss_tensor}.
The infrastructure supplies mixed precision, optimizer/checkpoints,
validation, multi-GPU DataParallel, recovery, and packaging.

The module owns the scientific method: task head, Phase-1 hidden-state bridge,
trainable/frozen parameter selection, span alignment and decoding, long-text
windowing, teacher/student loss (when applicable), and all hyperparameters.
Never use other pretrained weights, external data, or an external LLM API.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def build_task_model(base_model, tokenizer, labels: Sequence[str],
                     config: Mapping[str, Any], *, teacher_model=None,
                     teacher_tokenizer=None):
    """Return a new trainable torch.nn.Module starting from the supplied base."""
    raise NotImplementedError('Student implements the assessed Phase-2 method')


def collate_batch(rows, tokenizer, labels: Sequence[str], config: Mapping[str, Any],
                  *, teacher_tokenizer=None) -> Mapping[str, Any]:
    """Return tensor kwargs for task.forward; retain original character offsets."""
    raise NotImplementedError('Student implements BPE/character span alignment')


def predict_spans(model, text: str, tokenizer, labels: Sequence[str],
                  config: Mapping[str, Any]) -> list[list[Any]]:
    """Return [start_char, end_char_exclusive, official_label] spans for this text."""
    raise NotImplementedError('Student implements long-document span decoding')


# Optional: define build_scheduler(optimizer, config) if your student-designed
# method needs a nonconstant schedule. The notebook checkpoints its state.
