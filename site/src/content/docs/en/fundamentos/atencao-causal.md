---
title: "Causal attention: Q, K and V"
description: "From projections to masking and value mixing, with actual dimensions and a complete numerical example."
---

## Querying context

Attention produces a vector for each position by combining the content it may observe. Q represents the current position's query, K contains the keys compared with that query, and V contains the values to mix. These are distinct learned projections. Their names describe the computation by analogy, rather than literal questions or labels in natural language.

In `Attention.__call__`, located in `src/glassllm/nn.py`, the block's RMSNorm has already normalized the input. The sequence is:

```text
X → Q/K/V → RoPE on Q/K → cache → GQA expansion
  → QKᵀ/√dₖ → causal mask → softmax → mix V → W_O
```

## Actual dimensions

Let `B` denote batch size, `n` new queries and `m` total keys. Without caching, `m=n`; with caching, `m=start+n`.

| Stage | Default dimensions |
| --- | --- |
| Input X | `(B, n, 128)` |
| W_Q / W_K / W_V | `(128,128)` / `(128,64)` / `(128,64)` |
| Q split into heads | `(B, 4, n, 32)` |
| K/V before GQA | `(B, 2, m, 32)` |
| K/V after repeat_kv | `(B, 4, m, 32)` |
| Scores and weights | `(B, 4, n, m)` |
| Per-head mix | `(B, 4, n, 32)` |
| Concatenation / output | `(B, n, 128)` |

The code applies [RoPE](../rope/) before storing new keys in the cache. [GQA](../gqa-cache/) shares two K/V heads among four Q heads. `W_O`, with shape `(128,128)`, combines the concatenated outputs. None of these projections has a bias.

## Scores, masking and normalization

```text
S[h,i,j] = dot(Q[h,i], K[h,j]) / sqrt(32)
M[i,j] = −∞ if j > start+i; 0 otherwise
A[h,i,:] = softmax(S[h,i,:] + M[i,:])
Y[h,i,:] = Σ_j A[h,i,j] · V[h,j,:]
```

The divisor `√dₖ` controls the scale of dot products. Under the explanatory assumption of independent coordinates with zero mean and unit variance, a dot product has variance `dₖ`; dividing by its square root normalizes that variance. This motivates the factor without exactly describing learned vectors.

`causal_mask(n, m, start)` returns `True` at forbidden positions. `masked_fill` replaces those scores with `−∞`. Since `exp(−∞)=0`, future positions receive zero weight. Softmax subtracts each row's maximum before exponentiating, preserving probabilities and reducing overflow risk. Each valid row sums to approximately 1, subject to numerical rounding.

## Complete numerical example

This example uses one two-dimensional head to keep the calculation short:

```python
import numpy as np
from glassllm.nn import causal_mask
from glassllm.autograd import softmax_np

Q = np.array([[1., 0.], [1., 1.], [0., 1.]])
K = Q.copy()
V = np.array([[2., 0.], [0., 2.], [2., 2.]])
scores = Q @ K.T / np.sqrt(2)
masked = np.where(causal_mask(3, 3, 0), -np.inf, scores)
weights = softmax_np(masked)
print(np.round(weights, 6))
print(np.round(weights @ V, 6))
```

Verified output:

```text
Weights: [[1.000000, 0.000000, 0.000000],
          [0.330238, 0.669762, 0.000000],
          [0.197776, 0.401112, 0.401112]]
Output:  [[2.000000, 0.000000],
          [0.660477, 1.339523],
          [1.197776, 1.604448]]
```

The first position observes only itself. The second mixes the first two rows of V. The third uses the entire prefix. The example starts after the projections and does not include RoPE or W_O.

## Reading snapshots accurately

The inspector exposes `attn.q`, `attn.k`, `attn.v`, `attn.rope_q`, `attn.rope_k`, `attn.scores`, `attn.mask`, `attn.softmax`, `attn.mix` and `attn.out`. Select layer 0–3. Q and attention weights have four heads; projected K/V have two. `Tracer` removes the batch axis, so a selected `attn.softmax` head has shape `(n,m)`.

Attention weights describe context mixing within a head. Output probabilities allocate mass to the 1,024 next-token candidates. A high attention weight alone does not establish causality or fully explain a prediction.

## Reference

Scaled dot-product attention and decoder masking appear in [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al., 2017. This project uses a decoder with RMSNorm, RoPE, GQA and SwiGLU, rather than reproducing the original encoder–decoder architecture in full.
