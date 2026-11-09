"""RoPE prefix-hybrid language model for CS781 Phase 1 Submission 03.

The prompt prefix is encoded bidirectionally and the continuation is decoded
causally inside one shared Transformer stack.  The module is randomly
initialized and depends only on PyTorch.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def make_model_config(vocab_size: int) -> Mapping[str, Any]:
    """Return every value required to reconstruct the exact model."""

    return {
        "architecture": "rope_prefix_hybrid_transformer",
        "vocab_size": int(vocab_size),
        "max_sequence_length": 1280,
        "hidden_size": 384,
        "num_layers": 24,
        "num_heads": 6,
        "head_dim": 64,
        "ffn_size": 1024,
        "dropout": 0.10,
        "attention_dropout": 0.0,
        "normalization": "pre_rms_norm",
        "normalization_epsilon": 1e-6,
        "activation": "swiglu",
        "position_representation": "rope",
        "rotary_base": 10_000.0,
        "rotary_fraction": 1.0,
        "tie_word_embeddings": True,
        "linear_bias": True,
        "initializer_std": 0.02,
    }


def analytical_parameter_count(config: Mapping[str, Any]) -> int:
    """Calculate the exact trainable parameter count for this implementation."""

    vocab = int(config["vocab_size"])
    hidden = int(config["hidden_size"])
    layers = int(config["num_layers"])
    inner = int(config["ffn_size"])
    use_bias = bool(config["linear_bias"])
    bias = 1 if use_bias else 0

    token_embedding = vocab * hidden
    attention_per_layer = 4 * (hidden * hidden + bias * hidden)
    swiglu_per_layer = (
        2 * (hidden * inner + bias * inner)
        + inner * hidden
        + bias * hidden
    )
    two_rms_norms_per_layer = 2 * hidden
    final_rms_norm = hidden
    output_bias = vocab
    return int(
        token_embedding
        + layers
        * (attention_per_layer + swiglu_per_layer + two_rms_norms_per_layer)
        + final_rms_norm
        + output_bias
    )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.epsilon = float(epsilon)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        variance = states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = states * torch.rsqrt(variance + self.epsilon).to(states.dtype)
        return normalized * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, maximum: int, base: float) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        inverse_frequency = 1.0 / (
            float(base) ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(maximum, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse_frequency)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cosine", angles.cos(), persistent=False)
        self.register_buffer("sine", angles.sin(), persistent=False)

    def forward(self, length: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cosine[:length].to(dtype=dtype)[None, None, :, :],
            self.sine[:length].to(dtype=dtype)[None, None, :, :],
        )

    def position_slice(
        self, start: int, length: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if start < 0 or length < 1 or start + length > self.cosine.shape[0]:
            raise ValueError("Cached RoPE position exceeds the configured context")
        end = start + length
        return (
            self.cosine[start:end].to(dtype=dtype)[None, None, :, :],
            self.sine[start:end].to(dtype=dtype)[None, None, :, :],
        )


def _rotate_half(states: torch.Tensor) -> torch.Tensor:
    half = states.shape[-1] // 2
    return torch.cat((-states[..., half:], states[..., :half]), dim=-1)


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        query * cosine + _rotate_half(query) * sine,
        key * cosine + _rotate_half(key) * sine,
    )


class PrefixHybridAttention(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        heads = int(config["num_heads"])
        if hidden % heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden
        self.num_heads = heads
        self.head_dim = hidden // heads
        if self.head_dim != int(config["head_dim"]):
            raise ValueError("Configured head_dim does not match hidden_size / num_heads")
        self.attention_dropout = float(config["attention_dropout"])
        use_bias = bool(config["linear_bias"])
        self.query = nn.Linear(hidden, hidden, bias=use_bias)
        self.key = nn.Linear(hidden, hidden, bias=use_bias)
        self.value = nn.Linear(hidden, hidden, bias=use_bias)
        self.output = nn.Linear(hidden, hidden, bias=use_bias)

    def _heads(self, states: torch.Tensor) -> torch.Tensor:
        batch, length, _ = states.shape
        return states.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        states: torch.Tensor,
        visibility_mask: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        query = self._heads(self.query(states))
        key = self._heads(self.key(states))
        value = self._heads(self.value(states))
        query, key = _apply_rope(query, key, cosine, sine)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=visibility_mask[:, None, :, :],
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        batch, _, length, _ = attended.shape
        attended = attended.transpose(1, 2).contiguous().view(
            batch, length, self.hidden_size
        )
        return self.output(attended)

    def forward_cached(
        self,
        states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        previous: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Attend over a complete prefix or append one causal continuation token."""

        query = self._heads(self.query(states))
        key = self._heads(self.key(states))
        value = self._heads(self.value(states))
        query, key = _apply_rope(query, key, cosine, sine)
        if previous is not None:
            key = torch.cat((previous[0], key), dim=2)
            value = torch.cat((previous[1], value), dim=2)
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        batch, _, length, _ = attended.shape
        output = attended.transpose(1, 2).contiguous().view(
            batch, length, self.hidden_size
        )
        return self.output(output), (key, value)


class SwiGLU(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        inner = int(config["ffn_size"])
        use_bias = bool(config["linear_bias"])
        self.gate = nn.Linear(hidden, inner, bias=use_bias)
        self.up = nn.Linear(hidden, inner, bias=use_bias)
        self.down = nn.Linear(inner, hidden, bias=use_bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(states)) * self.up(states))


class PrefixHybridBlock(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        epsilon = float(config["normalization_epsilon"])
        dropout = float(config["dropout"])
        self.attention_norm = RMSNorm(hidden, epsilon)
        self.attention = PrefixHybridAttention(config)
        self.ffn_norm = RMSNorm(hidden, epsilon)
        self.ffn = SwiGLU(config)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        states: torch.Tensor,
        visibility_mask: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        states = states + self.residual_dropout(
            self.attention(
                self.attention_norm(states), visibility_mask, cosine, sine
            )
        )
        states = states + self.residual_dropout(self.ffn(self.ffn_norm(states)))
        return states

    def forward_cached(
        self,
        states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        previous: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attended, cache = self.attention.forward_cached(
            self.attention_norm(states), cosine, sine, previous
        )
        states = states + self.residual_dropout(attended)
        states = states + self.residual_dropout(self.ffn(self.ffn_norm(states)))
        return states, cache


class PrefixHybridLanguageModel(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.vocab_size = int(config["vocab_size"])
        self.max_sequence_length = int(config["max_sequence_length"])
        self.hidden_size = int(config["hidden_size"])
        self.num_layers = int(config["num_layers"])
        if self.vocab_size < 4 or self.max_sequence_length < 2:
            raise ValueError("Invalid vocabulary or sequence length")
        if str(config["normalization"]) != "pre_rms_norm":
            raise ValueError("This implementation requires pre_rms_norm")
        if str(config["activation"]) != "swiglu":
            raise ValueError("This implementation requires swiglu")
        if str(config["position_representation"]) != "rope":
            raise ValueError("This implementation requires RoPE")
        if not bool(config["tie_word_embeddings"]):
            raise ValueError("This implementation ties input and output word embeddings")
        rotary_fraction = float(config["rotary_fraction"])
        if rotary_fraction != 1.0:
            raise ValueError("This implementation uses full-head RoPE")

        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.embedding_dropout = nn.Dropout(float(config["dropout"]))
        self.rotary = RotaryEmbedding(
            int(config["head_dim"]),
            self.max_sequence_length,
            float(config["rotary_base"]),
        )
        self.blocks = nn.ModuleList(
            PrefixHybridBlock(config) for _ in range(self.num_layers)
        )
        self.final_norm = RMSNorm(
            self.hidden_size, float(config["normalization_epsilon"])
        )
        self.output_bias = nn.Parameter(torch.zeros(self.vocab_size))
        self.gradient_checkpointing = False
        self.apply(self._initialize_weights)
        residual_std = float(config["initializer_std"]) / math.sqrt(2 * self.num_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.output.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.down.weight, mean=0.0, std=residual_std)

        exact = analytical_parameter_count(config)
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != exact:
            raise RuntimeError(
                f"Analytical parameter count {exact:,} differs from runtime {actual:,}"
            )

    def _initialize_weights(self, module: nn.Module) -> None:
        std = float(self.config["initializer_std"])
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    @torch.no_grad()
    def prefill_prefix(
        self, prefix_ids: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Encode the whole prompt bidirectionally and retain each layer's keys/values."""

        if self.training:
            raise RuntimeError("Cached generation requires model.eval()")
        if prefix_ids.ndim != 2 or not 1 <= prefix_ids.shape[1] <= self.max_sequence_length:
            raise ValueError("prefix_ids must have shape [batch, valid_prefix_length]")
        states = self.embedding_dropout(
            self.token_embedding(prefix_ids) * math.sqrt(self.hidden_size)
        )
        cosine, sine = self.rotary(prefix_ids.shape[1], states.dtype)
        caches = []
        for block in self.blocks:
            states, layer_cache = block.forward_cached(states, cosine, sine)
            caches.append(layer_cache)
        return tuple(caches)

    @torch.no_grad()
    def decode_cached(
        self,
        current_ids: torch.Tensor,
        caches: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        """Append one token and return its next-token logits and updated layer caches."""

        if self.training:
            raise RuntimeError("Cached generation requires model.eval()")
        if current_ids.ndim != 2 or current_ids.shape[1] != 1:
            raise ValueError("current_ids must have shape [batch, 1]")
        if len(caches) != self.num_layers or not caches:
            raise ValueError("One key/value cache is required for every Transformer layer")
        position = caches[0][0].shape[2]
        if position >= self.max_sequence_length:
            raise ValueError("Cached decoding exceeded the configured context")
        states = self.embedding_dropout(
            self.token_embedding(current_ids) * math.sqrt(self.hidden_size)
        )
        cosine, sine = self.rotary.position_slice(position, 1, states.dtype)
        updated = []
        for block, previous in zip(self.blocks, caches):
            states, layer_cache = block.forward_cached(
                states, cosine, sine, previous
            )
            updated.append(layer_cache)
        logits = F.linear(
            self.final_norm(states), self.token_embedding.weight, self.output_bias
        )
        return logits[:, -1, :], tuple(updated)

    def _prefix_hybrid_visibility(
        self,
        attention_mask: torch.Tensor,
        prefix_lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch, length = attention_mask.shape
        positions = torch.arange(length, device=attention_mask.device)
        query_position = positions[None, :, None]
        key_position = positions[None, None, :]
        boundary = prefix_lengths[:, None, None]
        query_is_prefix = query_position < boundary
        key_is_prefix = key_position < boundary
        causal_continuation = (key_position >= boundary) & (
            key_position <= query_position
        )
        visible = torch.where(
            query_is_prefix,
            key_is_prefix,
            key_is_prefix | causal_continuation,
        )
        return visible & attention_mask[:, None, :].to(dtype=torch.bool)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        prefix_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, length = input_ids.shape
        if length > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {length} exceeds maximum {self.max_sequence_length}"
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        attention_mask = attention_mask.to(dtype=torch.bool)
        if prefix_lengths is None:
            raise ValueError("prefix_lengths is required")
        prefix_lengths = prefix_lengths.to(device=input_ids.device, dtype=torch.long)
        if prefix_lengths.shape != (batch,):
            raise ValueError("prefix_lengths must have shape [batch]")
        valid_lengths = attention_mask.sum(dim=1)
        if bool(((prefix_lengths < 0) | (prefix_lengths > valid_lengths)).any()):
            raise ValueError("Each prefix length must be within its valid token span")

        visibility = self._prefix_hybrid_visibility(attention_mask, prefix_lengths)
        states = self.embedding_dropout(
            self.token_embedding(input_ids) * math.sqrt(self.hidden_size)
        )
        cosine, sine = self.rotary(length, states.dtype)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                states = checkpoint(
                    block,
                    states,
                    visibility,
                    cosine,
                    sine,
                    use_reentrant=False,
                )
            else:
                states = block(states, visibility, cosine, sine)
        states = self.final_norm(states)
        return F.linear(states, self.token_embedding.weight, self.output_bias)


def build_model(config: Mapping[str, Any]) -> PrefixHybridLanguageModel:
    """Construct a newly initialized prefix-hybrid model."""

    return PrefixHybridLanguageModel(dict(config))

