---
title: "FFN and SwiGLU"
description: "Two projection branches, the SiLU activation and the return from 341 hidden coordinates to 128."
---

## Transforming each token separately

Attention mixes context positions. The feed-forward network (FFN) transforms each position's vector without directly consulting other tokens. Its input already contains contextual information from attention, but the operation itself is applied independently at each position.

`SwiGLU` in `src/glassllm/nn.py` has three learned matrices: `W1` for the *gate* branch, `W3` for *up* and `W2` for the *down* projection. None of these projections has a bias.

```text
                 ┌─ @W1 → gate → SiLU ─┐
Normalized X ────┤                     ⊙ → @W2 → output
                 └─ @W3 → up ──────────┘
```

The branches multiply coordinate by coordinate, rather than through matrix multiplication. The gate activation modulates the up branch's content. This gate is not a probability: `SiLU(z)` may be negative and is not bounded by `[0,1]`.

## Equations and dimensions

```text
G = X @ W1
U = X @ W3
SiLU(z) = z · sigmoid(z)
A = SiLU(G) ⊙ U
Y = A @ W2
```

| Value | Default shape |
| --- | --- |
| Normalized X | `(B,n,128)` |
| W1 and W3 | `(128,341)` |
| Gate, up and activation | `(B,n,341)` |
| W2 | `(341,128)` |
| Output Y | `(B,n,128)` |

The code computes `hidden = int(8*d/3)`: with `d=128`, this is **341**, not 342 or 512. The three matrices total `3 × 128 × 341 = 130,944` parameters per block. The rule approximates the `8d²` parameter budget of a conventional width-`4d` FFN, rounded to an integer.

## Verified activation example

This example starts after projection to focus on multiplying the two branches:

```python
import numpy as np
from glassllm import Tensor
from glassllm.autograd import silu

gate = Tensor([-1., 0., 1.], requires_grad=True)
up = Tensor([2., 3., 4.], requires_grad=True)
act = silu(gate) * up
print(np.round(act.data, 6))
print(np.round(act.data @ np.ones((3, 1)), 6))
act.sum().backward()
print(np.round(gate.grad, 6))
print(np.round(up.grad, 6))
```

Verified output:

```text
Activation:          [-0.537883, 0.000000, 2.924234]
Down with weights 1: [2.386351]
Gate gradient:       [0.144659, 1.500000, 3.710682]
Up gradient:         [-0.268941, 0.000000, 0.731059]
```

Even when `gate=0`, the derivative with respect to the gate is nonzero: `SiLU'(0)=0.5`, multiplied by `up=3`, gives 1.5. The gradient with respect to up is zero at that coordinate because `SiLU(0)=0`.

## How gradients traverse the branches

If `D` is the gradient received by A:

```text
∂L/∂U = D ⊙ SiLU(G)
∂L/∂G = D ⊙ U ⊙ sigmoid(G) ⊙ (1 + G ⊙ (1−sigmoid(G)))
```

Projection backward follows the rule for `Y=XW`: the input gradient uses `grad_Y @ W.T`; the weight gradient sums products of inputs and output gradients across positions and batch elements. Contributions from W1 and W3 return to and add at their shared input. The residual path adds another separate contribution.

`silu` in `src/glassllm/autograd.py` uses a stable sigmoid expression based on `exp(−abs(x))`. Its derivative is implemented directly, while multiplications and projections use general autograd rules.

## Snapshots, limitations and reference

In replay order, `norm.ffn` shows normalized input; `ffn.gate` and `ffn.up` show the two branches; `ffn.act` shows **SiLU(gate) multiplied by up**, rather than SiLU alone; `ffn.down` shows projection back to 128; and `resid.ffn` shows addition to the residual stream. `Tracer` exposes shapes without the batch axis.

Negative values are valid. A large coordinate alone does not identify a concept or word. This FFN contains no dropout. Shazeer introduces gated variants in [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202), 2020. The project implements SwiGLU with the width specified above. Continue with [normalization and residuals](../normalizacao-residuos/) or [output and generation](../saida-geracao/).
