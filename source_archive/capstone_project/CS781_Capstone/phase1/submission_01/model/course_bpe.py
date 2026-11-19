"""Deterministic byte-level BPE tokenizer used by the Phase 1 notebook.

The implementation is deliberately small: it wraps the `tokenizers` BPE model,
exposes the interface expected by the notebook, and persists every artifact
needed to recreate the token-to-ID mapping in a later Kaggle session.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence


SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


class CourseBPETokenizer:
    """Minimal tokenizer adapter with a stable encode/decode/save contract."""

    def __init__(self, backend):
        self.backend = backend
        self.pad_token_id = backend.token_to_id(SPECIAL_TOKENS[0])
        self.unk_token_id = backend.token_to_id(SPECIAL_TOKENS[1])
        self.bos_token_id = backend.token_to_id(SPECIAL_TOKENS[2])
        self.eos_token_id = backend.token_to_id(SPECIAL_TOKENS[3])

    def __len__(self) -> int:
        return int(self.backend.get_vocab_size())

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids = list(self.backend.encode(text, add_special_tokens=False).ids)
        if add_special_tokens:
            ids = [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self.backend.decode([int(value) for value in ids], skip_special_tokens=skip_special_tokens)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 32_000,
        min_frequency: int = 2,
    ) -> "CourseBPETokenizer":
        try:
            from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        except ImportError as exc:
            raise RuntimeError(
                "The `tokenizers` package is required. Kaggle normally includes it; "
                "otherwise install a pinned version before this cell."
            ) from exc

        if vocab_size < 1_024:
            raise ValueError("vocab_size must be at least 1024")
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")

        backend = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS[1]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=int(vocab_size),
            min_frequency=int(min_frequency),
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        backend.train_from_iterator((str(text) for text in texts), trainer=trainer)
        tokenizer = cls(backend)
        if len(tokenizer) != vocab_size:
            raise RuntimeError(
                f"BPE training produced {len(tokenizer)} tokens, expected exactly {vocab_size}. "
                "Inspect the corpus and do not begin model training with a changing vocabulary."
            )
        return tokenizer

    def save_pretrained(self, destination: str | Path) -> tuple[str, ...]:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        tokenizer_json = destination / "tokenizer.json"
        self.backend.save(str(tokenizer_json))
        # Save the explicit BPE vocabulary and merge table as well as the unified
        # tokenizer.json, so the learned mapping is auditable without code.
        model_files = tuple(self.backend.model.save(str(destination), "course_bpe"))
        config = {
            "tokenizer_class": type(self).__name__,
            "tokenizer_file": tokenizer_json.name,
            "model_type": "byte_level_bpe",
            "special_tokens": list(SPECIAL_TOKENS),
            "special_token_ids": {
                "pad_token_id": self.pad_token_id,
                "unk_token_id": self.unk_token_id,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
            },
            "vocab_size": len(self),
        }
        (destination / "tokenizer_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return (str(tokenizer_json), *model_files, str(destination / "tokenizer_config.json"))

    @classmethod
    def from_pretrained(cls, location: str | Path) -> "CourseBPETokenizer":
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("The `tokenizers` package is required to restore the BPE tokenizer") from exc

        location = Path(location)
        tokenizer_path = location / "tokenizer.json"
        config_path = location / "tokenizer_config.json"
        if not tokenizer_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"Incomplete tokenizer directory: {location}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        tokenizer = cls(Tokenizer.from_file(str(tokenizer_path)))
        if len(tokenizer) != int(config["vocab_size"]):
            raise RuntimeError("Saved tokenizer vocabulary size does not match tokenizer_config.json")
        actual_ids = {
            "pad_token_id": tokenizer.pad_token_id,
            "unk_token_id": tokenizer.unk_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if actual_ids != config["special_token_ids"]:
            raise RuntimeError("Saved tokenizer special-token mapping is inconsistent")
        return tokenizer
