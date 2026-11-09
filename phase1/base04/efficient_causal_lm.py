"""Submission 04 causal local/global Transformer.

This is an AI-assisted reference implementation. The course's student-authorship
declaration must not be set to true on the basis of this file alone.

For query position i, visible keys are exactly the union of:
  * max(0, i - local_window + 1) ... i; and
  * j <= i with j % global_stride == 0.
The global positions are ordinary causal tokens, not bidirectional memory.
Their hidden states carry earlier context through preceding layers. The cache
retains the anchor keys and only the last local_window - 1 ordinary keys.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def make_model_config(vocab_size: int) -> dict[str, Any]:
    if not 1024 <= int(vocab_size) <= 100_000:
        raise ValueError("Unexpected BPE vocabulary size")
    return {
        "vocab_size": int(vocab_size),
        "max_sequence_length": 1536,
        "hidden_size": 512,
        "num_layers": 14,
        "num_query_heads": 8,
        "num_kv_heads": 2,
        "intermediate_size": 1536,
        "local_window": 256,
        "global_stride": 128,
        "rope_base": 10_000.0,
        "residual_dropout": 0.05,
        "tie_embeddings": True,
    }


class RMSNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return (x.float() * scale).to(x.dtype) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class SparseCausalGQA(nn.Module):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        h = int(config["hidden_size"])
        self.num_query_heads = int(config["num_query_heads"])
        self.num_kv_heads = int(config["num_kv_heads"])
        if h % self.num_query_heads or self.num_query_heads % self.num_kv_heads:
            raise ValueError("Invalid grouped-query head dimensions")
        self.head_dim = h // self.num_query_heads
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        self.groups = self.num_query_heads // self.num_kv_heads
        self.window = int(config["local_window"])
        self.stride = int(config["global_stride"])
        self.max_length = int(config["max_sequence_length"])
        if not 1 <= self.window <= self.max_length or self.stride < 1:
            raise ValueError("Invalid local/global attention span")
        kv_width = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(h, h)
        self.k_proj = nn.Linear(h, kv_width)
        self.v_proj = nn.Linear(h, kv_width)
        self.o_proj = nn.Linear(h, h)

        positions = torch.arange(self.max_length, dtype=torch.long)
        inv_freq = 1.0 / (float(config["rope_base"]) **
                          (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        angles = torch.outer(positions.float(), inv_freq)
        self.register_buffer("rope_cos", torch.cat((angles.cos(), angles.cos()), -1), persistent=False)
        self.register_buffer("rope_sin", torch.cat((angles.sin(), angles.sin()), -1), persistent=False)


    def visible_positions(self, position: int) -> tuple[int, ...]:
        if not 0 <= position < self.max_length:
            raise ValueError("Position outside model context")
        start = max(0, position - self.window + 1)
        return tuple(sorted(set(range(start, position + 1)) | set(range(0, start, self.stride))))

    def _project(self, x: torch.Tensor, position_offset: int = 0):
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.num_kv_heads, self.groups, self.head_dim)
        q = q.permute(0, 2, 3, 1, 4)
        k = self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cosine = self.rope_cos[position_offset:position_offset + length].to(q.dtype)
        sine = self.rope_sin[position_offset:position_offset + length].to(q.dtype)
        if cosine.shape[0] != length:
            raise ValueError("Sequence exceeds max_sequence_length")
        q = q * cosine[None, None, None] + _rotate_half(q) * sine[None, None, None]
        k = k * cosine[None, None] + _rotate_half(k) * sine[None, None]
        return q, k, v

    def _finish(self, values: torch.Tensor) -> torch.Tensor:
        batch, _, _, length, _ = values.shape
        return self.o_proj(values.permute(0, 3, 1, 2, 4).reshape(batch, length, -1))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        batch, length, _ = x.shape
        q, k, v = self._project(x)
        # The fused SDPA kernel is substantially more T4-friendly than a Python
        # gather/einsum over [batch, heads, queries, visible_keys, head_dim].
        # Its mask still enforces the exact local/global rule above.
        positions = torch.arange(length, device=x.device)
        distance = positions[:, None] - positions[None, :]
        visible = (distance >= 0) & ((distance < self.window) |
                                      (positions[None, :] % self.stride == 0))
        visible = visible[None, None] & attention_mask[:, None, None, :]
        query = q.reshape(batch, self.num_query_heads, length, self.head_dim)
        key = k.repeat_interleave(self.groups, dim=1)
        value = v.repeat_interleave(self.groups, dim=1)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=visible, dropout_p=0.0, is_causal=False)
        attended = attended.reshape(batch, self.num_kv_heads, self.groups, length, self.head_dim)
        return self._finish(attended), k, v

    def decode_one(self, x: torch.Tensor, old_k: torch.Tensor, old_v: torch.Tensor,
                   position: int):
        q, new_k, new_v = self._project(x, position)
        keys = torch.cat((old_k, new_k), dim=2)
        values = torch.cat((old_v, new_v), dim=2)
        scores = torch.einsum("bhgtd,bhkd->bhgtk", q, keys).float()
        probabilities = torch.softmax(scores * (self.head_dim ** -0.5), -1)
        result = torch.einsum("bhgtk,bhkd->bhgtd", probabilities.to(values.dtype), values)
        return self._finish(result), keys, values


class CausalBlock(nn.Module):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        h, intermediate = int(config["hidden_size"]), int(config["intermediate_size"])
        self.attention_norm = RMSNorm(h)
        self.attention = SparseCausalGQA(config)
        self.feedforward_norm = RMSNorm(h)
        self.gate_proj = nn.Linear(h, intermediate)
        self.up_proj = nn.Linear(h, intermediate)
        self.down_proj = nn.Linear(intermediate, h)
        self.dropout = nn.Dropout(float(config["residual_dropout"]))

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attended, _, _ = self.attention(self.attention_norm(x), attention_mask)
        x = x + self.dropout(attended)
        return x + self.dropout(self._mlp(self.feedforward_norm(x)))

    def forward_with_kv(self, x: torch.Tensor, attention_mask: torch.Tensor):
        attended, k, v = self.attention(self.attention_norm(x), attention_mask)
        x = x + self.dropout(attended)
        return x + self.dropout(self._mlp(self.feedforward_norm(x))), k, v

    def decode_one(self, x: torch.Tensor, old_k: torch.Tensor,
                   old_v: torch.Tensor, position: int):
        attended, keys, values = self.attention.decode_one(
            self.attention_norm(x), old_k, old_v, position)
        x = x + attended
        return x + self._mlp(self.feedforward_norm(x)), keys, values


class EfficientCausalLM(nn.Module):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.config = dict(config)
        h = int(config["hidden_size"])
        self.max_sequence_length = int(config["max_sequence_length"])
        self.local_window = int(config["local_window"])
        self.global_stride = int(config["global_stride"])
        self.embedding = nn.Embedding(int(config["vocab_size"]), h)
        self.blocks = nn.ModuleList(CausalBlock(config) for _ in range(int(config["num_layers"])))
        self.final_norm = RMSNorm(h)
        self.output = nn.Linear(h, int(config["vocab_size"]), bias=True)
        if config.get("tie_embeddings") is not True:
            raise ValueError("Submission 04 requires tied token/output weights")
        self.output.weight = self.embedding.weight
        self.gradient_checkpointing = False
        self.apply(self._initialize)
        for block in self.blocks:
            nn.init.normal_(block.attention.o_proj.weight, std=0.02 / math.sqrt(2 * len(self.blocks)))
            nn.init.normal_(block.down_proj.weight, std=0.02 / math.sqrt(2 * len(self.blocks)))

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = bool(enabled)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.max_sequence_length:
            raise ValueError("Expected [batch, sequence] IDs within max_sequence_length")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask shape mismatch")
        x = self.embedding(input_ids)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, attention_mask, use_reentrant=False)
            else:
                x = block(x, attention_mask)
        return self.output(self.final_norm(x))

    @torch.inference_mode()
    def prefill_context(self, input_ids: torch.Tensor) -> dict[str, Any]:
        if self.training:
            raise RuntimeError("Call model.eval() before cached inference")
        if input_ids.ndim != 2 or not 1 <= input_ids.shape[1] <= self.max_sequence_length:
            raise ValueError("Invalid prefill sequence")
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)
        start = max(0, length - self.local_window + 1)
        retained = (positions % self.global_stride == 0) | (positions >= start)
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        x = self.embedding(input_ids)
        layers = []
        for block in self.blocks:
            x, k, v = block.forward_with_kv(x, mask)
            layers.append((k[:, :, retained, :], v[:, :, retained, :]))
        return {"layers": tuple(layers), "positions": positions[retained], "next_position": length}

    @torch.inference_mode()
    def decode_cached(self, next_input_ids: torch.Tensor, cache: Mapping[str, Any]):
        if self.training:
            raise RuntimeError("Call model.eval() before cached inference")
        if next_input_ids.ndim != 2 or next_input_ids.shape[1] != 1:
            raise ValueError("decode_cached requires one token per batch row")
        position = int(cache["next_position"])
        if position >= self.max_sequence_length:
            raise ValueError("Cached decoding exceeded max_sequence_length")
        old_positions = cache["positions"]
        if len(cache["layers"]) != len(self.blocks):
            raise ValueError("Cache layer count mismatch")
        next_positions = torch.cat((old_positions, old_positions.new_tensor([position])))
        next_local_start = max(0, position - self.local_window + 2)
        retain = (next_positions % self.global_stride == 0) | (next_positions >= next_local_start)
        x = self.embedding(next_input_ids)
        next_layers = []
        for block, (old_k, old_v) in zip(self.blocks, cache["layers"]):
            if old_k.shape[0] != x.shape[0] or old_k.shape[2] != old_positions.numel():
                raise ValueError("Cache batch or position dimensions mismatch")
            x, keys, values = block.decode_one(x, old_k, old_v, position)
            next_layers.append((keys[:, :, retain, :], values[:, :, retain, :]))
        logits = self.output(self.final_norm(x))[:, -1, :]
        return logits, {"layers": tuple(next_layers), "positions": next_positions[retain],
                        "next_position": position + 1}


def build_model(config: Mapping[str, Any]) -> EfficientCausalLM:
    required = set(make_model_config(int(config["vocab_size"])))
    if set(config) != required:
        raise ValueError(f"Model configuration keys mismatch: {set(config) ^ required}")
    return EfficientCausalLM(config)


def expected_parameter_count(config: Mapping[str, Any]) -> int:
    """Closed-form trainable count, including all biases and tied output weight."""
    vocab = int(config["vocab_size"])
    width = int(config["hidden_size"])
    depth = int(config["num_layers"])
    intermediate = int(config["intermediate_size"])
    kv_width = width * int(config["num_kv_heads"]) // int(config["num_query_heads"])
    attention = (width * width + width) * 2 + (width * kv_width + kv_width) * 2
    feedforward = (width * intermediate + intermediate) * 2 + (intermediate * width + width)
    return vocab * width + depth * (attention + feedforward + 2 * width) + width + vocab
