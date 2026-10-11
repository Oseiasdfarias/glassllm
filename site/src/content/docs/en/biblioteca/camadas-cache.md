---
title: "Layers and cache"
description: "Module, RMSNorm, GQA attention, RoPE, SwiGLU, blocks and K/V state."
---

## Module and initialization

All symbols on this page belong to `glassllm.nn`. `Module` supplies `named_parameters(prefix: str = "") -> dict[str,Tensor]` and `parameters() -> list[Tensor]`. Discovery traverses gradient-requiring Tensor attributes, Module attributes and lists of Modules. Arbitrary dictionaries and tuples are not traversed; duplicate references have no general deduplication. The standard model stores the tied embedding only once.

`init_param(rng: np.random.Generator, shape: tuple[int,...], std: float, dtype) -> Tensor` samples a zero-mean normal distribution with standard deviation `std`, converts it to `dtype` and requests gradients. Call it outside `no_grad` for parameters to be discovered. It changes no global seed: samples come from the supplied `rng`.

## RMSNorm

`RMSNorm(d: int, eps: float, dtype)` creates `(d,)` `gamma`, initialized to ones. `norm(x: Tensor) -> Tensor` applies `x / sqrt(mean(x², axis=-1)+eps) * gamma`, preserving shape. It subtracts no mean and has neither beta nor moving batch statistics. The last input dimension must match `d`.

Ordinary usage requires positive `d` and `eps`; the standalone class does not repeat every Config validation. With `eps=0`, a zero vector has a zero denominator. The [RMSNorm paper](https://arxiv.org/abs/1910.07467) motivates this normalization; the project normalizes the full final axis.

## Angles and masks

`rope_angles(d_k: int, max_len: int, base: float) -> tuple[np.ndarray,np.ndarray]` generates `(max_len,d_k/2)` cosine and sine for `position * base**(-2*i/d_k)`. It requires a positive even dimension, nonnegative length and positive base, otherwise raises `ValueError`. Arrays have the formula's NumPy dtype; Attention converts them to the weight dtype.

`causal_mask(n_queries: int, n_keys: int, start: int) -> np.ndarray` returns `(n_queries,n_keys)` booleans. Query position is `start+i`; key `j > start+i` is True and blocked. Negative counts or start raise `ValueError`. A cached mask need not be square.

```python
import numpy as np
from glassllm.nn import causal_mask, rope_angles

np.testing.assert_array_equal(causal_mask(2, 3, start=1),
                              [[False, False, True], [False, False, False]])
cos, sin = rope_angles(4, 3, 10000.0)
assert cos.shape == sin.shape == (3, 2)
np.testing.assert_allclose(cos[0], [1.0, 1.0])
np.testing.assert_allclose(sin[0], [0.0, 0.0])
```

## Attention

`Attention(d, n_heads, n_kv_heads, max_len, rope_base, rng, std, dtype)` creates `(d,d)` `W_Q`, `(d,Hkv*dk)` `W_K/W_V`, and `(d,d)` `W_O`, where `dk=d/n_heads`. Dimensions must be positive and divisible, Q head count must be a multiple of K/V count, and `dk` must be even. Structural violations raise `ValueError`.

`attention(x: Tensor, layer: int, trace: Tracer | None = None, cache: KVCache | None = None, start: int = 0) -> Tensor` accepts and returns `(B,n,d)`. It projects, splits heads, rotates Q/K, incorporates cache, repeats K/V for Q groups, scales scores by `sqrt(dk)`, masks, normalizes, mixes V, concatenates and projects. There is no bias, dropout or cross-attention.

`layer` labels snapshots rather than selecting another layer's parameters. Positions outside the RoPE table raise `ValueError`. Traced Q has `Hq` heads and K/V has `Hkv`: with the 4/2 defaults, Q 0–1 share K/V 0 and Q 2–3 share K/V 1.

## KVCache and incremental execution

`KVCache()` starts with `k=None`, `v=None`. After a prefix it holds `(B,Hkv,m,dk)` arrays: rotated keys and values before group repetition. Each call concatenates new positions on axis 2 and replaces both arrays. Cache is neither a parameter nor checkpoint state.

Use cache for inference inside `no_grad`. Array concatenation disconnects history from the graph, so this is not a chunked-training mechanism. Never reuse caches across prompts, models or different batches. There is no `reset` method: create a fresh cache, normally with `model.new_cache()`.

```python
import numpy as np
from glassllm import Config, GlassLM, no_grad

model = GlassLM(Config(vocab_size=8, d_model=8, n_layers=1,
                      n_heads=2, n_kv_heads=1, max_len=4))
cache = model.new_cache()
assert cache[0].k is None and cache[0].v is None
with no_grad():
    full = model([1, 2, 3]).data[:, -1:]
    model([1, 2], cache=cache, start=0)
    assert cache[0].k.shape == (1, 1, 2, 4)
    last = model([3], cache=cache, start=2).data
assert cache[0].k.shape == (1, 1, 3, 4)
np.testing.assert_allclose(last, full, rtol=1e-5, atol=1e-6)
```

Manual calls require `start=m` before appending and total size no larger than `max_len`. The layer does not validate every cache/offset mismatch; `generate` maintains consistency and rebuilds a full window at the limit. Traced K/V represent **new positions**; scores and softmax also include older cached positions.

## SwiGLU and Block

`SwiGLU(d: int, rng: np.random.Generator, std: float, dtype)` requires `d>0` and uses `hidden=int(8*d/3)`. `W1/W3` are `(d,hidden)` and `W2` is `(hidden,d)`. `ffn(x: Tensor, layer: int, trace: Tracer | None = None) -> Tensor` computes `SiLU(x@W1) * (x@W3)`, then `@W2`. Input/output are `(B,n,d)`; gate, up and act are `(B,n,hidden)`. Four FFN events are recorded per block.

`Block(cfg, rng: np.random.Generator, dtype)` groups `norm1`, `attn`, `norm2` and `ffn`. `block(x: Tensor, layer: int, trace=None, cache=None, start=0) -> Tensor` performs residual sums `h=x+Attention(RMSNorm(x))` and `out=h+SwiGLU(RMSNorm(h))`. It preserves `(B,n,d)` and records 18 operations including Attention and FFN. Each block owns its separate cache.

Scaled attention comes from [Attention Is All You Need](https://arxiv.org/abs/1706.03762), position from [RoPE](https://arxiv.org/abs/2104.09864), K/V sharing from [GQA](https://arxiv.org/abs/2305.13245) and gating from [GLU Variants](https://arxiv.org/abs/2002.05202). Here GQA is trained directly; the code does not implement the paper's multi-head checkpoint conversion. Use the [interface graph](../../interface/arquitetura/) to connect each branch with its snapshots.
