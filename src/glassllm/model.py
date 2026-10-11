"""GlassLM decoder: RMSNorm, RoPE, grouped-query attention, SwiGLU and tied embeddings."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .autograd import Tensor, cross_entropy, cross_entropy_grad, embedding, no_grad, softmax_np, transpose
from .config import Config
from .nn import Block, KVCache, Module, RMSNorm, init_param
from .trace import Tracer, record


@dataclass(frozen=True)
class GenStep:
    token: int
    probs: np.ndarray
    trace: Tracer | None


def next_token_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Normalize next-token logits using a positive sampling temperature."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0; use greedy=True to always take the most likely token")
    return softmax_np(logits / temperature)  # @step probs


class GlassLM(Module):
    def __init__(self, cfg: Config, seed: int = 0, dtype=np.float32):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self.E = init_param(rng, (cfg.vocab_size, cfg.d_model), cfg.init_std, dtype)
        self.blocks = [Block(cfg, rng, dtype) for _ in range(cfg.n_layers)]
        self.norm_f = RMSNorm(cfg.d_model, cfg.eps, dtype)

    def new_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.blocks]

    def forward(self, ids, trace: Tracer | None = None,
                cache: list[KVCache] | None = None, start: int = 0) -> Tensor:
        """Return logits of shape (batch, positions, vocabulary size)."""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.ndim != 2:
            raise ValueError("ids must be a one- or two-dimensional array")
        batch, length = ids.shape
        if start < 0 or start + length > self.cfg.max_len:
            raise ValueError(f"positions [{start}, {start + length}) do not fit in max_len={self.cfg.max_len}")
        if trace is not None and batch != 1:
            raise ValueError("the tracer follows a single sequence; pass batch size 1")

        x = embedding(self.E, ids)  # @step embed
        record(trace, "embed", x)
        for i, block in enumerate(self.blocks):
            x = block(x, i, trace, None if cache is None else cache[i], start)
        h = self.norm_f(x)  # @step norm.final
        record(trace, "norm.final", h)
        logits = h @ transpose(self.E, (1, 0))  # @step logits
        record(trace, "logits", logits)
        return logits

    __call__ = forward

    def loss(self, ids, targets, trace: Tracer | None = None) -> Tensor:
        """Return mean cross-entropy for one target per input token."""
        logits = self.forward(ids, trace=trace)
        batch, length, vocab = logits.shape
        flat_targets = np.asarray(targets).reshape(batch * length)
        result = cross_entropy(logits.reshape(batch * length, vocab), flat_targets)  # @step loss
        record(trace, "loss", result, batched=False)
        if trace is not None:
            grad = cross_entropy_grad(logits.data.reshape(batch * length, vocab), flat_targets)
            record(trace, "loss.grad", grad.reshape(batch, length, vocab))
        return result

    def generate(self, ids, n_new: int, temperature: float = 1.0, greedy: bool = False,
                 seed: int | None = None, trace: bool = False) -> Iterator[GenStep]:
        """Yield tokens with cached inference and an optional fresh trace per step.

        Once the context is full, rebuild it from the last max_len tokens,
        resetting positions to match a full forward on that trailing window.
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0; use greedy=True to always take the most likely token")
        if not isinstance(trace, bool):
            raise TypeError("trace must be a bool; use trace=True for a fresh Tracer in each GenStep")
        ids = [int(token) for token in ids]
        if not ids:
            raise ValueError("the prompt is empty; give at least one token")
        max_len = self.cfg.max_len
        ids = ids[-max_len:]
        rng = np.random.default_rng(seed)
        cache, start, pending = self.new_cache(), 0, ids
        for _ in range(n_new):
            if start + len(pending) > max_len:
                cache, start, pending = self.new_cache(), 0, ids
            tracer = Tracer() if trace else None
            with no_grad():
                logits = self.forward(pending, trace=tracer, cache=cache, start=start)
                start += len(pending)
                probs = next_token_probs(logits.data[0, -1].astype(np.float64), temperature)
                token = int(np.argmax(probs)) if greedy else int(rng.choice(len(probs), p=probs))  # @step sample
                record(tracer, "probs", probs, batched=False)
                record(tracer, "sample", np.array(token), batched=False)
            ids = (ids + [token])[-max_len:]
            pending = [token]
            yield GenStep(token, probs, tracer)

    def save(self, path: str | Path, fp16: bool = True) -> None:
        arrays = {name: p.data.astype(np.float16 if fp16 else p.data.dtype)
                  for name, p in self.named_parameters().items()}
        np.savez_compressed(path, __config__=np.array(json.dumps(asdict(self.cfg))), **arrays)

    @classmethod
    def load(cls, path: str | Path, dtype=np.float32) -> GlassLM:
        with np.load(path) as data:
            cfg = Config(**json.loads(str(data["__config__"])))
            model = cls(cfg, dtype=dtype)
            for name, parameter in model.named_parameters().items():
                parameter.data = data[name].astype(dtype)
        return model
