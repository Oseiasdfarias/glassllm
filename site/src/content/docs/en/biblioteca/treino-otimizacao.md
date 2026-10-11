---
title: "Training and optimization"
description: "Token windows, evaluation, train, AdamW, clipping and learning-rate scheduling."
---

## Prepare batches

Imports: `from glassllm.train import get_batch, evaluate, train` and `from glassllm.optim import AdamW, clip_grad_norm, lr_schedule`. These functions are not root exports.

`get_batch(data: np.ndarray, batch_size: int, seq_len: int, rng: np.random.Generator) -> tuple[np.ndarray,np.ndarray]` receives one-dimensional token IDs. It samples `batch_size` windows and returns `(batch_size,seq_len)` `x,y`, preserving the data dtype. Each target is the next token: `y[:, :-1] == x[:, 1:]`. Sampling includes the last window with complete targets and may repeat windows.

Batch size and sequence length must be positive and `len(data)>seq_len`, otherwise `ValueError` is raised. The function neither tokenizes text nor validates every ID. Model usage also requires `seq_len<=cfg.max_len`.

```python
import numpy as np
from glassllm.train import get_batch

data = np.arange(12, dtype=np.int64)
x, y = get_batch(data, batch_size=2, seq_len=4, rng=np.random.default_rng(0))
assert x.shape == y.shape == (2, 4)
np.testing.assert_array_equal(y, x + 1)
```

## Evaluate

`evaluate(model: GlassLM, data: np.ndarray, batch_size: int, seq_len: int, n_batches: int = 10, seed: int = 1234) -> float` averages losses from sampled batches. A fixed seed reproduces windows; this is not an exhaustive corpus pass. It requires `n_batches>0` as well as get_batch and forward conditions.

It runs inside `no_grad` and changes neither weights nor existing gradients. It does not call `.eval()`: there is no dropout or batch-normalization mode. The metric is mean natural-log cross-entropy per token, not accuracy. Perplexity is `np.exp(loss)`, compared using the same tokenizer and data.

## train

The full signature is:

```text
train(model, data, steps, batch_size=16, seq_len=128, lr=0.003,
      warmup=100, seed=0, val_data=None, eval_every=100,
      keep_best=False, log=print) -> list[dict]
```

`model` is modified in place; `data/val_data` are one-dimensional IDs. `steps` counts updates rather than epochs. Each iteration samples a batch, calculates loss, resets parameter gradients, runs backward, clips global norm to 1, computes learning rate and calls AdamW. Each call creates a new optimizer: calling `train` again does not resume prior moments.

| Argument | Effect |
| --- | --- |
| `steps` | Nonnegative integer; zero performs no updates |
| `batch_size`, `seq_len` | Training windows; positive and compatible with corpus/context |
| `lr` | Base learning rate; no complete sign/finite-value validation |
| `warmup` | Nonnegative warmup count |
| `seed` | Window generator; does not reinitialize model weights |
| `val_data` | Optional validation corpus large enough for batches |
| `eval_every` | Positive evaluation and logging interval |
| `keep_best` | With validation, restores the best measured weights |
| `log` | String callback; `None` disables messages |

History contains one dictionary per completed update: zero-based `step`, pre-update batch `loss`, and applied `lr`. With validation, `val_loss` appears when `step % eval_every == 0` and at the final step. Validation evaluates weights **after** the update, with 10 batches and seed 1234.

```python
import numpy as np
from glassllm import Config, GlassLM
from glassllm.train import evaluate, train

cfg = Config(vocab_size=8, d_model=8, n_layers=1,
             n_heads=2, n_kv_heads=1, max_len=4)
model = GlassLM(cfg, seed=0)
data = np.tile(np.arange(8, dtype=np.int64), 5)
history = train(model, data, steps=3, batch_size=2, seq_len=4,
                warmup=1, val_data=data, eval_every=2,
                keep_best=True, log=None)
assert [entry["step"] for entry in history] == [0, 1, 2]
assert [entry["step"] for entry in history if "val_loss" in entry] == [0, 2]
best = min(entry["val_loss"] for entry in history if "val_loss" in entry)
np.testing.assert_allclose(evaluate(model, data, 2, 4), best)
```

With `keep_best=True`, weights are copied whenever measured validation improves and restored at the end. There is no early stopping or automatic file save. Without validation, `keep_best` has no effect. Optimizer state and final gradients are not restored with the weights.

## Deadline and failures

Training has a fixed two-hour deadline including validation. The clock is checked between NumPy operations; a currently running operation cannot be interrupted. Expiry raises `TimeoutError` without returning partial history. A previously measured best set of weights is restored in `finally`, including when execution fails; without a completed best set, weights remain at the last reached state.

`steps<0`, `warmup<0` or `eval_every<=0` raise `ValueError` before updates. Corpus, shape and ID failures arise in called components. A failing log callback also interrupts execution. Supply valid finite controls and remember that ordinary training is CPU NumPy.

## AdamW

`AdamW(params: list[Tensor], lr: float = 0.003, betas=(0.9,0.95), eps: float = 1e-8, weight_decay: float = 0.1)` stores parameters, `m/v` moments, global `t` and per-parameter update counts. Beta2 and learning-rate defaults are project choices rather than another framework's full default set.

`zero_grad() -> None` sets every parameter's `grad=None`. `step(lr: float | None = None) -> None` uses an override for that update only, preserving the base `lr`. Parameters without gradients are skipped, including decay and bias correction. Participating parameters update moving averages of the gradient and its square, correct bias with their own update count and receive an adaptive step.

Decoupled weight decay applies only to `ndim>=2` arrays, such as projection and embedding matrices. RMSNorm gamma vectors receive no decay. There is no complete validation of betas, shapes, dtype or learning rate; supply floating parameters and mathematically valid controls. Optimizer state has no save/load API.

```python
import numpy as np
from glassllm import Tensor
from glassllm.optim import AdamW, clip_grad_norm, lr_schedule

parameter = Tensor([0.0, 0.0], requires_grad=True)
optimizer = AdamW([parameter], lr=0.1, weight_decay=0.0)
optimizer.zero_grad()
((parameter - 3.0) ** 2).sum().backward()
before = clip_grad_norm([parameter], max_norm=1.0)
assert before > 1.0
assert np.linalg.norm(parameter.grad) <= 1.0
optimizer.step()
assert np.all(parameter.data > 0.0)
assert lr_schedule(0, 10, 1.0, warmup=2) == 0.5
assert lr_schedule(20, 10, 1.0, warmup=2) == 0.1
```

## Clipping and learning rate

`clip_grad_norm(params: list[Tensor], max_norm: float) -> float` returns global L2 norm **before** clipping. Float64 square accumulation avoids overflow for large float32 gradients. Above the limit, all present gradients are multiplied by `max_norm/(norm+1e-6)`; `None` is ignored. No gradients gives 0. Supply `max_norm>0`; negative or NaN limits have no dedicated validation.

`lr_schedule(step: int, total_steps: int, base_lr: float, warmup: int, min_ratio: float = 0.1) -> float` uses `base_lr*(step+1)/warmup` while `step<warmup`, then cosine decay from base to minimum. `warmup=0` starts at base. Cosine progress is capped at 1; `step>=total_steps` yields the minimum once warmup has finished. Actual updates run from 0 to `total_steps-1`, so the last rate may remain slightly above minimum.

References are [Adam](https://arxiv.org/abs/1412.6980), [AdamW](https://arxiv.org/abs/1711.05101) and the cosine shape from [SGDR](https://arxiv.org/abs/1608.03983). The project uses one cosine decay with warmup, without SGDR warm restarts. The [file guide](../cli-arquivos/) explains weight saving and float16 export validation.
