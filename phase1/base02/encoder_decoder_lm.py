"""Submission 02 encoder-decoder Transformer language model.

The model is trained from scratch and depends only on PyTorch.  It uses a
bidirectional pre-norm encoder and a causal pre-norm decoder with
encoder-decoder cross-attention.  Token embeddings are shared by the encoder
and decoder and tied to the output projection.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def make_model_config(vocab_size: int) -> dict[str, Any]:
    """Return the complete numerical configuration needed for reconstruction."""

    return {
        "architecture": "pre_norm_encoder_decoder_transformer",
        "vocab_size": int(vocab_size),
        "encoder_layers": 6,
        "decoder_layers": 6,
        "hidden_size": 512,
        "attention_heads": 8,
        "ffn_size": 2048,
        # Training still uses 512/256-token pairs.  The larger positional
        # tables avoid unnecessarily truncating 512-word prompts and leave a
        # safer generation budget for 128-word continuations.
        "max_encoder_length": 1024,
        "max_decoder_length": 512,
        "dropout": 0.1,
        "attention_dropout": 0.0,
        "activation": "gelu",
        "normalization": "pre_layer_norm",
        "position_representation": "learned_absolute",
        "embedding_tying": "shared_encoder_decoder_and_output",
        "layer_norm_epsilon": 1e-5,
        "initializer_std": 0.02,
    }


def _require_int(config: Mapping[str, Any], name: str, minimum: int = 1) -> int:
    value = int(config[name])
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


class MultiHeadAttention(nn.Module):
    """Small, explicit multi-head attention used for self- and cross-attention."""

    def __init__(self, hidden_size: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        self.hidden_size = int(hidden_size)
        self.heads = int(heads)
        self.head_dim = hidden_size // heads
        self.dropout = float(dropout)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def project_query(self, states: torch.Tensor) -> torch.Tensor:
        return self._split_heads(self.query(states))

    def project_key_value(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._split_heads(self.key(states)), self._split_heads(self.value(states))

    def attend_projected(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        *,
        causal: bool = False,
    ) -> torch.Tensor:
        """Attend using already projected heads, enabling inference KV caching."""

        attention_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.ndim != 2:
                raise ValueError("key_padding_mask must have shape [batch, key_length]")
            attention_mask = key_padding_mask.to(dtype=torch.bool)[:, None, None, :]

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        batch, _, query_length, _ = attended.shape
        attended = attended.transpose(1, 2).contiguous().view(
            batch, query_length, self.hidden_size
        )
        return self.output(attended)

    def forward(
        self,
        query_states: torch.Tensor,
        key_value_states: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        *,
        causal: bool = False,
    ) -> torch.Tensor:
        query = self.project_query(query_states)
        key, value = self.project_key_value(key_value_states)
        return self.attend_projected(
            query, key, value, key_padding_mask, causal=causal
        )


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(hidden_size, ffn_size)
        self.output = nn.Linear(ffn_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.output(self.dropout(F.gelu(self.input(states), approximate="tanh")))


class EncoderBlock(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        epsilon = float(config["layer_norm_epsilon"])
        dropout = float(config["dropout"])
        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=epsilon)
        self.self_attention = MultiHeadAttention(
            hidden_size, int(config["attention_heads"]), float(config["attention_dropout"])
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_size, eps=epsilon)
        self.feed_forward = FeedForward(hidden_size, int(config["ffn_size"]), dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor, encoder_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.self_attention_norm(states)
        states = states + self.residual_dropout(
            self.self_attention(normalized, normalized, encoder_mask, causal=False)
        )
        states = states + self.residual_dropout(
            self.feed_forward(self.feed_forward_norm(states))
        )
        return states


class DecoderBlock(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        epsilon = float(config["layer_norm_epsilon"])
        dropout = float(config["dropout"])
        heads = int(config["attention_heads"])
        attention_dropout = float(config["attention_dropout"])
        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=epsilon)
        self.self_attention = MultiHeadAttention(hidden_size, heads, attention_dropout)
        self.cross_attention_norm = nn.LayerNorm(hidden_size, eps=epsilon)
        self.cross_attention = MultiHeadAttention(hidden_size, heads, attention_dropout)
        self.feed_forward_norm = nn.LayerNorm(hidden_size, eps=epsilon)
        self.feed_forward = FeedForward(hidden_size, int(config["ffn_size"]), dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        states: torch.Tensor,
        encoder_states: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.self_attention_norm(states)
        states = states + self.residual_dropout(
            self.self_attention(normalized, normalized, causal=True)
        )
        states = states + self.residual_dropout(
            self.cross_attention(
                self.cross_attention_norm(states),
                encoder_states,
                encoder_mask,
                causal=False,
            )
        )
        states = states + self.residual_dropout(
            self.feed_forward(self.feed_forward_norm(states))
        )
        return states

    def prepare_cross_attention(
        self, encoder_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project encoder keys/values once for all autoregressive steps."""

        return self.cross_attention.project_key_value(encoder_states)

    def forward_step(
        self,
        states: torch.Tensor,
        encoder_mask: torch.Tensor,
        cache: Mapping[str, torch.Tensor | None],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Decode one position and append only its self-attention keys/values."""

        normalized = self.self_attention_norm(states)
        query = self.self_attention.project_query(normalized)
        new_key, new_value = self.self_attention.project_key_value(normalized)
        previous_key = cache.get("self_key")
        previous_value = cache.get("self_value")
        key = new_key if previous_key is None else torch.cat((previous_key, new_key), dim=2)
        value = (
            new_value
            if previous_value is None
            else torch.cat((previous_value, new_value), dim=2)
        )
        # The single query is the newest position, so every cached key is in
        # its causal past and no triangular mask is required.
        states = states + self.residual_dropout(
            self.self_attention.attend_projected(query, key, value, causal=False)
        )

        cross_key = cache.get("cross_key")
        cross_value = cache.get("cross_value")
        if cross_key is None or cross_value is None:
            raise RuntimeError("Cross-attention cache was not prepared")
        cross_query = self.cross_attention.project_query(
            self.cross_attention_norm(states)
        )
        states = states + self.residual_dropout(
            self.cross_attention.attend_projected(
                cross_query,
                cross_key,
                cross_value,
                encoder_mask,
                causal=False,
            )
        )
        states = states + self.residual_dropout(
            self.feed_forward(self.feed_forward_norm(states))
        )
        return states, {
            "self_key": key,
            "self_value": value,
            "cross_key": cross_key,
            "cross_value": cross_value,
        }


class EncoderDecoderLanguageModel(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.vocab_size = _require_int(config, "vocab_size", 4)
        self.hidden_size = _require_int(config, "hidden_size")
        self.max_encoder_length = _require_int(config, "max_encoder_length")
        self.max_decoder_length = _require_int(config, "max_decoder_length")
        encoder_layers = _require_int(config, "encoder_layers")
        decoder_layers = _require_int(config, "decoder_layers")
        _require_int(config, "attention_heads")
        _require_int(config, "ffn_size")

        if str(config.get("activation")) != "gelu":
            raise ValueError("This implementation supports activation='gelu'")
        if str(config.get("normalization")) != "pre_layer_norm":
            raise ValueError("This implementation supports normalization='pre_layer_norm'")
        if str(config.get("position_representation")) != "learned_absolute":
            raise ValueError("This implementation supports learned absolute positions")
        if str(config.get("embedding_tying")) != "shared_encoder_decoder_and_output":
            raise ValueError("Encoder, decoder, and output embeddings must be tied")

        epsilon = float(config["layer_norm_epsilon"])
        dropout = float(config["dropout"])
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.encoder_position_embedding = nn.Embedding(
            self.max_encoder_length, self.hidden_size
        )
        self.decoder_position_embedding = nn.Embedding(
            self.max_decoder_length, self.hidden_size
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.ModuleList(
            EncoderBlock(config) for _ in range(encoder_layers)
        )
        self.decoder_blocks = nn.ModuleList(
            DecoderBlock(config) for _ in range(decoder_layers)
        )
        self.encoder_final_norm = nn.LayerNorm(self.hidden_size, eps=epsilon)
        self.decoder_final_norm = nn.LayerNorm(self.hidden_size, eps=epsilon)
        self.output_bias = nn.Parameter(torch.zeros(self.vocab_size))
        self.gradient_checkpointing = False
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module) -> None:
        std = float(self.config["initializer_std"])
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = bool(enabled)

    def _embed(
        self, input_ids: torch.Tensor, position_embedding: nn.Embedding, maximum: int
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input IDs must have shape [batch, length]")
        if input_ids.shape[1] > maximum:
            raise ValueError(f"Sequence length {input_ids.shape[1]} exceeds maximum {maximum}")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        states = self.token_embedding(input_ids) * math.sqrt(self.hidden_size)
        return self.embedding_dropout(states + position_embedding(positions)[None, :, :])

    def encode(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the bidirectional source representation exactly once."""

        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones_like(
                encoder_input_ids, dtype=torch.bool
            )
        if encoder_attention_mask.shape != encoder_input_ids.shape:
            raise ValueError("encoder_attention_mask must match encoder_input_ids")
        encoder_attention_mask = encoder_attention_mask.to(dtype=torch.bool)
        encoder_states = self._embed(
            encoder_input_ids, self.encoder_position_embedding, self.max_encoder_length
        )
        for block in self.encoder_blocks:
            if self.gradient_checkpointing and self.training:
                encoder_states = checkpoint(
                    block, encoder_states, encoder_attention_mask, use_reentrant=False
                )
            else:
                encoder_states = block(encoder_states, encoder_attention_mask)
        return self.encoder_final_norm(encoder_states), encoder_attention_mask

    def prepare_generation(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Cache the encoder and every decoder layer's cross-attention K/V."""

        encoder_states, encoder_attention_mask = self.encode(
            encoder_input_ids, encoder_attention_mask
        )
        layer_caches = []
        for block in self.decoder_blocks:
            cross_key, cross_value = block.prepare_cross_attention(encoder_states)
            layer_caches.append(
                {
                    "self_key": None,
                    "self_value": None,
                    "cross_key": cross_key,
                    "cross_value": cross_value,
                }
            )
        return {
            "encoder_attention_mask": encoder_attention_mask,
            "layers": layer_caches,
            "position": 0,
        }

    def decode_step(
        self,
        token_ids: torch.Tensor,
        generation_state: Mapping[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return next-token logits using incremental decoder KV caches."""

        if token_ids.ndim == 1:
            token_ids = token_ids[:, None]
        if token_ids.ndim != 2 or token_ids.shape[1] != 1:
            raise ValueError("decode_step expects one token per batch item")
        position = int(generation_state["position"])
        if position >= self.max_decoder_length:
            raise ValueError("Incremental decoding exceeded max_decoder_length")
        position_ids = torch.tensor([position], device=token_ids.device)
        states = self.token_embedding(token_ids) * math.sqrt(self.hidden_size)
        states = self.embedding_dropout(
            states + self.decoder_position_embedding(position_ids)[None, :, :]
        )
        old_caches = list(generation_state["layers"])
        if len(old_caches) != len(self.decoder_blocks):
            raise ValueError("Generation cache layer count does not match the model")
        new_caches = []
        encoder_mask = generation_state["encoder_attention_mask"]
        for block, layer_cache in zip(self.decoder_blocks, old_caches):
            states, new_cache = block.forward_step(states, encoder_mask, layer_cache)
            new_caches.append(new_cache)
        states = self.decoder_final_norm(states)
        logits = F.linear(states[:, 0], self.token_embedding.weight, self.output_bias)
        return logits, {
            "encoder_attention_mask": encoder_mask,
            "layers": new_caches,
            "position": position + 1,
        }

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_states, encoder_attention_mask = self.encode(
            encoder_input_ids, encoder_attention_mask
        )

        decoder_states = self._embed(
            decoder_input_ids, self.decoder_position_embedding, self.max_decoder_length
        )
        for block in self.decoder_blocks:
            if self.gradient_checkpointing and self.training:
                decoder_states = checkpoint(
                    block,
                    decoder_states,
                    encoder_states,
                    encoder_attention_mask,
                    use_reentrant=False,
                )
            else:
                decoder_states = block(
                    decoder_states, encoder_states, encoder_attention_mask
                )
        decoder_states = self.decoder_final_norm(decoder_states)
        return F.linear(decoder_states, self.token_embedding.weight, self.output_bias)


def build_model(config: Mapping[str, Any]) -> EncoderDecoderLanguageModel:
    """Construct a newly initialized model from a complete saved configuration."""

    return EncoderDecoderLanguageModel(dict(config))


def prepare_generation(
    model: EncoderDecoderLanguageModel,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: torch.Tensor | None,
) -> dict[str, Any]:
    """Notebook adapter for one-time encoder and cross-attention caching."""

    return model.prepare_generation(encoder_input_ids, encoder_attention_mask)


def decode_step(
    model: EncoderDecoderLanguageModel,
    token_ids: torch.Tensor,
    generation_state: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Notebook adapter for one-token cached autoregressive decoding."""

    return model.decode_step(token_ids, generation_state)

