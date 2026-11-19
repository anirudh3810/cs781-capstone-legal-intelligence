"""Phase 1 decoder-only Transformer trained from scratch.

The architecture uses Pre-LayerNorm causal self-attention, learned absolute
positions, a parameter-matched SwiGLU feed-forward intervention, and tied input
and output token embeddings. No pretrained weights or external model code are
used.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32_000
    context_length: int = 1_024
    hidden_size: int = 768
    num_layers: int = 8
    num_heads: int = 12
    ffn_hidden_size: int = 2_048
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    tie_word_embeddings: bool = True
    activation: str = "swiglu"
    architecture: str = "phase1_preln_decoder"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.context_length <= 1:
            raise ValueError("vocab_size and context_length must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_layers <= 0 or self.ffn_hidden_size <= 0:
            raise ValueError("num_layers and ffn_hidden_size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.activation != "swiglu":
            raise ValueError("This submission implements the recorded SwiGLU intervention")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelConfig":
        return cls(**dict(values))


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = float(config.dropout)
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, sequence, width = hidden.shape
        qkv = self.qkv(hidden)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        past_length = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            if past_key.shape[:2] != key.shape[:2] or past_key.shape[3] != key.shape[3]:
                raise ValueError("Incompatible attention cache")
            past_length = past_key.shape[2]
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        # PyTorch's fused implementation evaluates softmax(QK^T/sqrt(d_k) + M)V
        # with the strict upper triangle masked when is_causal=True.
        if past_length:
            query_positions = past_length + torch.arange(sequence, device=hidden.device)
            key_positions = torch.arange(key.shape[2], device=hidden.device)
            allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            attended = F.scaled_dot_product_attention(
                query, key, value, attn_mask=allowed,
                dropout_p=self.dropout if self.training else 0.0, is_causal=False,
            )
        else:
            attended = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0, is_causal=True,
            )
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, width)
        result = self.output(attended)
        return (result, (key, value)) if use_cache else result


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.down = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(hidden)) * self.up(hidden)))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attention = CausalSelfAttention(config)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ffn = SwiGLUFeedForward(config)

    def forward(
        self,
        hidden: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attention_result = self.attention(
            self.attention_norm(hidden), past_key_value=past_key_value, use_cache=use_cache
        )
        if use_cache:
            attention_output, present = attention_result
        else:
            attention_output, present = attention_result, None
        hidden = hidden + self.attention_dropout(attention_output)
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return (hidden, present) if use_cache else hidden


class CausalDecoderLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.context_length, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.apply(self._initialize)
        # GPT-style residual projection scaling keeps the initial residual stream stable.
        residual_std = 0.02 / math.sqrt(2 * config.num_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.output.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.down.weight, mean=0.0, std=residual_std)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def get_config(self) -> dict[str, Any]:
        return self.config.to_dict()

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise TypeError("input_ids must be a LongTensor shaped [batch, sequence]")
        _, sequence = input_ids.shape
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values must contain one entry per decoder block")
        past_length = 0 if past_key_values is None else int(past_key_values[0][0].shape[2])
        if sequence + past_length > self.config.context_length:
            raise ValueError(
                f"cached sequence length {sequence + past_length} exceeds context length {self.config.context_length}"
            )
        if input_ids.numel() and (int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size):
            raise ValueError("input_ids contain a token outside the configured vocabulary")
        positions = torch.arange(past_length, past_length + sequence, device=input_ids.device)
        hidden = self.embedding_dropout(self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :])
        presents = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            if self.gradient_checkpointing and self.training and not use_cache:
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                block_result = block(hidden, past_key_value=past, use_cache=use_cache)
                if use_cache:
                    hidden, present = block_result
                    presents.append(present)
                else:
                    hidden = block_result
        logits = self.lm_head(self.final_norm(hidden))
        return (logits, tuple(presents)) if use_cache else logits


def build_model(config: Mapping[str, Any] | ModelConfig) -> CausalDecoderLM:
    if not isinstance(config, ModelConfig):
        config = ModelConfig.from_dict(config)
    return CausalDecoderLM(config)
