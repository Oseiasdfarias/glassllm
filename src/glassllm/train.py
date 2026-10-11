"""Train on random corpus windows with next-token targets and AdamW."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .autograd import no_grad
from .model import GlassLM
from .optim import AdamW, clip_grad_norm, lr_schedule


_TRAINING_TIMEOUT_SECONDS = 2 * 60 * 60


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("training exceeded the 2-hour deadline (including validation)")


def get_batch(data: np.ndarray, batch_size: int, seq_len: int,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample corpus windows and the same windows shifted one token right."""
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be positive")
    if len(data) <= seq_len:
        raise ValueError(f"need more than seq_len={seq_len} tokens, got {len(data)}")
    starts = rng.integers(0, len(data) - seq_len, size=batch_size)
    x = np.stack([data[s:s + seq_len] for s in starts])
    y = np.stack([data[s + 1:s + seq_len + 1] for s in starts])
    return x, y


def _evaluate(model: GlassLM, data: np.ndarray, batch_size: int, seq_len: int,
              n_batches: int, seed: int, deadline: float | None) -> float:
    if n_batches <= 0:
        raise ValueError("n_batches must be positive")
    rng = np.random.default_rng(seed)
    losses = []
    with no_grad():
        for _ in range(n_batches):
            _check_deadline(deadline)
            x, y = get_batch(data, batch_size, seq_len, rng)
            losses.append(float(model.loss(x, y).data))
            _check_deadline(deadline)
    return float(np.mean(losses))


def evaluate(model: GlassLM, data: np.ndarray, batch_size: int, seq_len: int,
             n_batches: int = 10, seed: int = 1234) -> float:
    """Measure mean loss on deterministic windows without tracking gradients."""
    return _evaluate(model, data, batch_size, seq_len, n_batches, seed, deadline=None)


def train(model: GlassLM, data: np.ndarray, steps: int, batch_size: int = 16,
          seq_len: int = 128, lr: float = 3e-3, warmup: int = 100, seed: int = 0,
          val_data: np.ndarray | None = None, eval_every: int = 100,
          keep_best: bool = False, log: Callable[[str], None] | None = print) -> list[dict]:
    """Train in place, optionally restoring the best measured validation weights.

    Raise TimeoutError after two hours, including validation. The deadline is
    checked between NumPy operations; an individual operation cannot be interrupted.
    A completed best checkpoint is also restored when the deadline expires.
    """
    if steps < 0 or warmup < 0 or eval_every <= 0:
        raise ValueError("steps and warmup must be nonnegative; eval_every must be positive")
    t0 = time.monotonic()
    deadline = t0 + _TRAINING_TIMEOUT_SECONDS
    rng = np.random.default_rng(seed)
    params = model.parameters()
    opt = AdamW(params, lr=lr)
    history = []
    best_val, best_weights = float("inf"), None
    try:
        for step in range(steps):
            _check_deadline(deadline)
            x, y = get_batch(data, batch_size, seq_len, rng)
            loss = model.loss(x, y)
            _check_deadline(deadline)
            opt.zero_grad()
            loss.backward()
            _check_deadline(deadline)
            clip_grad_norm(params, 1.0)
            step_lr = lr_schedule(step, steps, lr, warmup)
            _check_deadline(deadline)
            opt.step(step_lr)
            _check_deadline(deadline)
            entry = {"step": step, "loss": float(loss.data), "lr": step_lr}
            last = step == steps - 1
            if val_data is not None and (step % eval_every == 0 or last):
                entry["val_loss"] = _evaluate(model, val_data, batch_size, seq_len,
                                               n_batches=10, seed=1234, deadline=deadline)
                if keep_best and entry["val_loss"] < best_val:
                    best_val = entry["val_loss"]
                    best_weights = [p.data.copy() for p in params]
            history.append(entry)
            if log is not None and (step % eval_every == 0 or last):
                val = f"  val {entry['val_loss']:.3f}" if "val_loss" in entry else ""
                log(f"step {step:5d}  loss {entry['loss']:.3f}{val}  "
                    f"lr {step_lr:.2e}  {time.monotonic() - t0:.0f}s")
            _check_deadline(deadline)
    finally:
        if best_weights is not None:
            for p, w in zip(params, best_weights):
                p.data[...] = w
            if log is not None:
                log(f"restored the weights with the best validation loss ({best_val:.3f})")
    return history
