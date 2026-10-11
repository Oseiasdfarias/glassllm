"""AdamW, gradient clipping and the learning rate schedule."""
from __future__ import annotations

import math

import numpy as np

from .autograd import Tensor


class AdamW:
    def __init__(self, params: list[Tensor], lr: float = 3e-3, betas=(0.9, 0.95),
                 eps: float = 1e-8, weight_decay: float = 0.1):
        self.params = list(params)
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self._steps = [0 for _ in self.params]
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self, lr: float | None = None) -> None:
        lr = self.lr if lr is None else lr
        beta1, beta2 = self.betas
        self.t += 1
        for i, (p, m, v) in enumerate(zip(self.params, self.m, self.v)):
            if p.grad is None:
                continue
            self._steps[i] += 1
            step = self._steps[i]
            m *= beta1
            m += (1 - beta1) * p.grad
            v *= beta2
            v += (1 - beta2) * p.grad * p.grad
            m_hat = m / (1 - beta1**step)
            v_hat = v / (1 - beta2**step)
            if p.data.ndim >= 2:
                p.data -= lr * self.weight_decay * p.data
            p.data -= (lr * m_hat / (np.sqrt(v_hat) + self.eps)).astype(p.data.dtype)


def clip_grad_norm(params: list[Tensor], max_norm: float) -> float:
    """Return the global norm before scaling gradients to at most max_norm."""
    grads = [p.grad for p in params if p.grad is not None]
    # Accumulate in float64 so large, finite float32 gradients do not overflow.
    total = math.sqrt(sum(float(np.square(g, dtype=np.float64).sum()) for g in grads))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * scale
    return total


def lr_schedule(step: int, total_steps: int, base_lr: float, warmup: int,
                min_ratio: float = 0.1) -> float:
    """Linear warmup, then cosine decay down to min_ratio * base_lr."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return base_lr * (min_ratio + (1 - min_ratio) * cosine)
