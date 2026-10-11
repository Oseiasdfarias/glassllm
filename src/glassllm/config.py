"""Model hyperparameters."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    vocab_size: int = 1024
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    max_len: int = 128
    rope_base: float = 10000.0
    eps: float = 1e-6
    init_std: float = 0.02

    def __post_init__(self) -> None:
        for name in ("vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "max_len"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("d_model / n_heads must be even for RoPE")
        if self.rope_base <= 0 or self.eps < 0 or self.init_std < 0:
            raise ValueError("rope_base must be positive; eps and init_std must be nonnegative")
