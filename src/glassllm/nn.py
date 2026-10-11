"""The building blocks of the Transformer. Each marked line is one traceable step."""
from __future__ import annotations

import math

import numpy as np

from .autograd import Tensor, masked_fill, repeat_kv, rope, silu, softmax
from .trace import Tracer, record


class Module:
    def named_parameters(self, prefix: str = "") -> dict[str, Tensor]:
        params: dict[str, Tensor] = {}
        for name, value in vars(self).items():
            if isinstance(value, Tensor) and value.requires_grad:
                params[prefix + name] = value
            elif isinstance(value, Module):
                params.update(value.named_parameters(f"{prefix}{name}."))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        params.update(item.named_parameters(f"{prefix}{name}.{i}."))
        return params

    def parameters(self) -> list[Tensor]:
        return list(self.named_parameters().values())


def init_param(rng: np.random.Generator, shape: tuple[int, ...], std: float, dtype) -> Tensor:
    return Tensor(rng.normal(0.0, std, shape).astype(dtype), requires_grad=True)


class RMSNorm(Module):
    def __init__(self, d: int, eps: float, dtype):
        self.eps = eps
        self.gamma = Tensor(np.ones(d, dtype=dtype), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        mean_square = (x * x).mean(axis=-1, keepdims=True)
        return x / (mean_square + self.eps) ** 0.5 * self.gamma


def rope_angles(d_k: int, max_len: int, base: float) -> tuple[np.ndarray, np.ndarray]:
    """cos and sin of m * theta_i, with theta_i = base^(-2i/d_k). Shape (max_len, d_k/2)."""
    if d_k <= 0 or d_k % 2 or max_len < 0 or base <= 0:
        raise ValueError("d_k must be a positive even number, max_len nonnegative and base positive")
    i = np.arange(d_k // 2)
    theta = base ** (-2.0 * i / d_k)
    angles = np.outer(np.arange(max_len), theta)
    return np.cos(angles), np.sin(angles)


def causal_mask(n_queries: int, n_keys: int, start: int) -> np.ndarray:
    """True where a query may NOT look: key position j is after query position start + i."""
    if n_queries < 0 or n_keys < 0 or start < 0:
        raise ValueError("query count, key count and start must be nonnegative")
    query_pos = start + np.arange(n_queries)[:, None]
    key_pos = np.arange(n_keys)[None, :]
    return key_pos > query_pos


class KVCache:
    """Keys and values already computed for earlier positions, per layer."""

    def __init__(self):
        self.k: np.ndarray | None = None
        self.v: np.ndarray | None = None


class Attention(Module):
    def __init__(self, d: int, n_heads: int, n_kv_heads: int, max_len: int, rope_base: float,
                 rng: np.random.Generator, std: float, dtype):
        if n_heads <= 0 or n_kv_heads <= 0 or d <= 0 or d % n_heads:
            raise ValueError("d and n_heads must be positive and d must be divisible by n_heads")
        if n_heads % n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.d_k = d // n_heads
        if self.d_k % 2:
            raise ValueError("d / n_heads must be even for RoPE")
        self.W_Q = init_param(rng, (d, n_heads * self.d_k), std, dtype)
        self.W_K = init_param(rng, (d, n_kv_heads * self.d_k), std, dtype)
        self.W_V = init_param(rng, (d, n_kv_heads * self.d_k), std, dtype)
        self.W_O = init_param(rng, (n_heads * self.d_k, d), std, dtype)
        cos, sin = rope_angles(self.d_k, max_len, rope_base)
        self.cos, self.sin = cos.astype(dtype), sin.astype(dtype)

    def _split_heads(self, x: Tensor, heads: int) -> Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, heads, self.d_k).transpose(0, 2, 1, 3)

    def __call__(self, x: Tensor, layer: int, trace: Tracer | None = None,
                 cache: KVCache | None = None, start: int = 0) -> Tensor:
        b, n, _ = x.shape
        if start < 0 or start + n > len(self.cos):
            raise ValueError(f"positions [{start}, {start + n}) do not fit in max_len={len(self.cos)}")
        q = self._split_heads(x @ self.W_Q, self.n_heads)  # @step attn.q
        k = self._split_heads(x @ self.W_K, self.n_kv_heads)  # @step attn.k
        v = self._split_heads(x @ self.W_V, self.n_kv_heads)  # @step attn.v
        record(trace, "attn.q", q, layer)
        record(trace, "attn.k", k, layer)
        record(trace, "attn.v", v, layer)

        cos, sin = self.cos[start:start + n], self.sin[start:start + n]
        q = rope(q, cos, sin)  # @step attn.rope_q
        k = rope(k, cos, sin)  # @step attn.rope_k
        record(trace, "attn.rope_q", q, layer)
        record(trace, "attn.rope_k", k, layer)

        if cache is not None:
            # Inference only: glue the new keys and values after the cached ones.
            if cache.k is not None:
                k = Tensor(np.concatenate([cache.k, k.data], axis=2))
                v = Tensor(np.concatenate([cache.v, v.data], axis=2))
            cache.k, cache.v = k.data, v.data

        repeats = self.n_heads // self.n_kv_heads
        k_all, v_all = repeat_kv(k, repeats), repeat_kv(v, repeats)

        scores = (q @ k_all.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)  # @step attn.scores
        record(trace, "attn.scores", scores, layer)
        mask = causal_mask(n, k_all.shape[2], start)
        scores = masked_fill(scores, mask, -np.inf)  # @step attn.mask
        record(trace, "attn.mask", scores, layer)
        weights = softmax(scores, axis=-1)  # @step attn.softmax
        record(trace, "attn.softmax", weights, layer)
        mixed = weights @ v_all  # @step attn.mix
        record(trace, "attn.mix", mixed, layer)

        concat = mixed.transpose(0, 2, 1, 3).reshape(b, n, self.n_heads * self.d_k)
        out = concat @ self.W_O  # @step attn.out
        record(trace, "attn.out", out, layer)
        return out


class SwiGLU(Module):
    def __init__(self, d: int, rng: np.random.Generator, std: float, dtype):
        if d <= 0:
            raise ValueError("d must be positive")
        hidden = int(8 * d / 3)
        self.W1 = init_param(rng, (d, hidden), std, dtype)
        self.W3 = init_param(rng, (d, hidden), std, dtype)
        self.W2 = init_param(rng, (hidden, d), std, dtype)

    def __call__(self, x: Tensor, layer: int, trace: Tracer | None = None) -> Tensor:
        gate = x @ self.W1  # @step ffn.gate
        up = x @ self.W3  # @step ffn.up
        act = silu(gate) * up  # @step ffn.act
        down = act @ self.W2  # @step ffn.down
        record(trace, "ffn.gate", gate, layer)
        record(trace, "ffn.up", up, layer)
        record(trace, "ffn.act", act, layer)
        record(trace, "ffn.down", down, layer)
        return down


class Block(Module):
    def __init__(self, cfg, rng: np.random.Generator, dtype):
        self.norm1 = RMSNorm(cfg.d_model, cfg.eps, dtype)
        self.attn = Attention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.max_len, cfg.rope_base,
                              rng, cfg.init_std, dtype)
        self.norm2 = RMSNorm(cfg.d_model, cfg.eps, dtype)
        self.ffn = SwiGLU(cfg.d_model, rng, cfg.init_std, dtype)

    def __call__(self, x: Tensor, layer: int, trace: Tracer | None = None,
                 cache: KVCache | None = None, start: int = 0) -> Tensor:
        h = self.norm1(x)  # @step norm.attn
        record(trace, "norm.attn", h, layer)
        x = x + self.attn(h, layer, trace, cache, start)  # @step resid.attn
        record(trace, "resid.attn", x, layer)
        h = self.norm2(x)  # @step norm.ffn
        record(trace, "norm.ffn", h, layer)
        x = x + self.ffn(h, layer, trace)  # @step resid.ffn
        record(trace, "resid.ffn", x, layer)
        return x
