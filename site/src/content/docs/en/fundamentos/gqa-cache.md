---
title: "Heads, GQA and K/V cache"
description: "Sharing keys and values between heads, incremental inference and rebuilding the context window."
---

## Four queries, two sets of K/V

The default configuration has four Q heads and two K/V heads, each with dimension 32. In *grouped-query attention* (GQA), a group of queries shares the same keys and values, while each query produces its own scores and weights.

```text
Q head 0 ─┐
          ├─ uses K/V head 0
Q head 1 ─┘

Q head 2 ─┐
          ├─ uses K/V head 1
Q head 3 ─┘
```

`repeat_kv(x, repeats=2)` in `src/glassllm/autograd.py` expands K/V for matrix multiplication: `(B,2,m,32)` becomes `(B,4,m,32)` in order `[0,0,1,1]`. It uses `np.repeat`, which materializes the expansion. The memory saving described below applies to stored cache arrays, rather than every temporary structure.

The backward pass sums contributions from heads sharing K/V. For example, repeating `[[1,2],[3,4]]` twice per head and summing the entire output returns gradient `[[2,2],[2,2]]` to the original tensor. These are shared projections, rather than four independently parameterized K/V heads.

## What the cache stores

`KVCache` in `src/glassllm/nn.py` contains arrays `k` and `v`, initially `None`. `GlassLM.new_cache()` creates one cache for each of the four layers.

| Incremental-stage tensor | Default shape, batch 1 |
| --- | --- |
| New Q | `(1,4,1,32)` |
| New K/V before append | `(1,2,1,32)` |
| Stored K/V after append | `(1,2,m,32)` |
| Scores and weights | `(1,4,1,m)` |
| Attention output | `(1,1,128)` |

K is stored **after** RoPE and V after projection. The cache does not store attention weights or all block intermediates. New Q/K/V are computed only for new tokens; Q queries all stored keys. Concatenation copies old arrays, so this educational implementation does not use preallocated buffers.

## Prefill and the next token

The first prediction processes the whole prompt (*prefill*). The next prediction passes only the newly selected token, with `start` equal to the stored length. Compare it with a full forward pass:

```python
import numpy as np
from glassllm import Config, GlassLM, no_grad

cfg = Config(vocab_size=16, d_model=8, n_layers=1,
             n_heads=2, n_kv_heads=1, max_len=8)
model = GlassLM(cfg, seed=0)
with no_grad():
    cache = model.new_cache()
    model.forward([4, 5, 6], cache=cache)
    incremental = model.forward([7], cache=cache, start=3).data[0, -1]
    full = model.forward([4, 5, 6, 7]).data[0, -1]
print(cache[0].k.shape)
print(np.allclose(incremental, full, atol=1e-7))
```

Verified output: `(1,1,4,4)` and `True`; the maximum difference was approximately `2.78 × 10⁻¹⁷`. The reduced configuration simply keeps the example fast. Caching is for inference: concatenating arrays disconnects the history's gradient path. Train with a complete forward pass and no cache.

## Memory and the window limit

For K and V across all layers:

```text
memory = 2 · B · L · m · H_KV · dₖ · bytes_per_element
```

With batch 1, four layers, 128 positions, two K/V heads and dimension 32, this is 65,536 elements: 256 KiB in float32 or 512 KiB in float64. Check `cache[0].k.dtype` and sum `k.nbytes + v.nbytes`: engine operations may promote activations to float64 even when parameters are float32. These totals exclude scores, gradients and `Tracer` snapshots. GQA halves stored cache memory relative to four K/V heads of the same size.

When the next insertion would exceed `max_len=128`, `GlassLM.generate` discards the cache and reprocesses the last 128 tokens, restarting positions at zero. It does not simply remove the first cached key: that would not recreate the states obtained with the truncated history.

## Visualization and reference

In the inspector, `attn.k` and `attn.v` show **new** projections recorded before concatenation. `attn.scores` and `attn.softmax` span the full key length `m`. The graph's cache node represents that structural connection and has no dedicated snapshot.

Ainslie et al. introduce GQA in [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), 2023. This project implements head sharing without the paper's checkpoint conversion procedure. Continue with [causal attention](../atencao-causal/) or [generation](../saida-geracao/).
