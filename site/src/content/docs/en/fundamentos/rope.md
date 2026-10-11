---
title: "Position and RoPE"
description: "Rotating adjacent Q/K coordinates, frequencies, relative positions and cache offsets."
---

## Position enters through rotations

Attention needs to distinguish a token's position as well as its content. `glassllm` uses *Rotary Position Embedding* (RoPE): it rotates adjacent pairs of Q and K coordinates using position-dependent angles. V remains unrotated, and no positional embedding table is added to the input.

Each head has `dₖ=32`, giving 16 pairs. `rope_angles(d_k, max_len, base)` in `src/glassllm/nn.py` computes sine and cosine tables. `rope(x, cos, sin)` in `src/glassllm/autograd.py` applies the transformation and its derivative.

## Frequencies and equation

For pair index `i`, starting at zero:

```text
θ_i = 10000^(−2i/32)                i = 0, ..., 15
angle at position p = p · θ_i

x'[2i]   = x[2i] · cos(pθ_i) − x[2i+1] · sin(pθ_i)
x'[2i+1] = x[2i] · sin(pθ_i) + x[2i+1] · cos(pθ_i)
```

The first pair has angular frequency 1 radian per position; the second approximately 0.562341; the last 0.000177828. Faster pairs change more between nearby positions, while slower pairs vary over a larger scale. The default tables have shape `(128,16)` and are buffers rather than trainable parameters.

A rotation preserves each pair's norm, subject to rounding. Its backward pass uses the opposite angle: the transpose of a rotation matrix is its inverse. The code therefore applies `_rotate(g, cos, -sin)` to the gradient.

## Why relative positions appear

If `R_p` is the rotation at position p, the dot product between a query at p and a key at q can be rewritten as:

```text
dot(R_p · Q, R_q · K) = Qᵀ · R_(q−p) · K
```

The positional difference appears in the dot product. This does not mean the entire model depends only on distances: content, causal masking, window boundaries and other layers also affect its output.

## A two-dimensional calculation

```python
import numpy as np
from glassllm import Tensor
from glassllm.autograd import rope
from glassllm.nn import rope_angles

cos, sin = rope_angles(d_k=2, max_len=2, base=10000.)
x = Tensor([[[[1., 0.], [1., 0.]]]])
y = rope(x, cos, sin).data
print(np.round(y[0, 0], 6))
print(round(float(y[0, 0, 0] @ y[0, 0, 1]), 6))
```

Verified output:

```text
[[1.000000, 0.000000],
 [0.540302, 0.841471]]
0.540302
```

Both vectors start identical. Position zero has angle zero; position one has angle one radian. Their dot product becomes `cos(1)`, while both retain norm 1.

## RoPE during cached generation

`Attention.__call__` selects `cos[start:start+n]` and `sin[start:start+n]`. During the first pass, prompt positions are `0…n−1`. During an incremental pass, the new query and key use position `start`; previous keys are already rotated in the cache.

Rotating stored keys again, or always assigning position zero to the new token, would be incorrect. `GlassLM.generate` maintains `start` while the window can hold the history. When it fills, the model rebuilds K/V from the last 128 tokens and restarts window positions at zero. This matches a full forward pass over the trailing window, rather than unlimited absolute positions.

## Inspector checks and reference

Compare `attn.q` with `attn.rope_q`, or `attn.k` with `attn.rope_k`, in the same layer and head. Their shapes stay unchanged; their values change. Q has four heads and K has two. At position zero, the rotation is the identity, so identical values are expected.

The model requires an even `dₖ` and rejects positions outside `max_len`. RoPE does not automatically enable longer contexts in this implementation. Su et al. introduce the mechanism in [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), first preprint in 2021. See also [GQA and caching](../gqa-cache/).
