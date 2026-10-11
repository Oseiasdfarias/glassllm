---
title: "RMSNorm, pre-norm and residuals"
description: "Controlling vector scale and adding two residual branches per block, including epsilon and gradient behavior."
---

## Controlling each position's scale

Each token enters the blocks as a 128-coordinate vector. RMSNorm adjusts its scale using the mean of squared coordinates. It acts along the last axis, without mixing tokens or batch elements.

```text
r(x) = sqrt((x₀² + … + x₁₂₇²)/128 + ε)
RMSNorm(x) = γ ⊙ x/r(x)

γ: (128,), learned and initialized to 1
ε: 0.000001 in the default configuration
```

`RMSNorm.__call__` in `src/glassllm/nn.py` expresses the operation using `Tensor`: square, mean, add epsilon, square root, divide and multiply by `gamma`. It has no separate custom backward function; autograd differentiates each part of the expression.

Unlike LayerNorm, RMSNorm does not subtract the mean or add a beta parameter. A nonzero constant vector remains uncentered. With positive epsilon, an all-zero vector remains zero without division by zero.

## Numerical example and epsilon

```python
import numpy as np
from glassllm import Tensor
from glassllm.nn import RMSNorm

norm = RMSNorm(d=2, eps=1e-6, dtype=np.float64)
a = norm(Tensor([[3., 4.]])).data
b = norm(Tensor([[6., 8.]])).data
print(np.round(a, 6))
print(b - a)
```

Verified output:

```text
[[0.848528, 1.131371]]
[[2.54558422e-08, 3.39411230e-08]]
```

The mean square of `[3,4]` is 12.5, giving divisor `sqrt(12.5 + 10⁻⁶)`. Scaling the input by 2 produces nearly the same output, but not exactly. For `c>0`:

```text
RMSNorm(c·x) = γ ⊙ x / sqrt(mean(x²) + ε/c²)
```

Scale invariance is exact in the ideal expression with `ε=0`, for nonzero vectors. With the code's epsilon, it is approximate when vector energy dominates epsilon. A nonuniform gamma also changes scaling coordinate by coordinate. The output should not be described as having an invariably fixed norm or unit variance.

## Two residual additions per block

`Block.__call__` normalizes **before** each branch, an arrangement called pre-norm:

```text
H  = X + Attention(RMSNorm₁(X))
X' = H + SwiGLU(RMSNorm₂(H))
```

The residual stream keeps shape `(B,n,128)`. Attention writes information gathered from context; [SwiGLU](../swiglu/) writes a per-position transformation. Each branch returns the same shape as its input so addition is possible.

```text
X ─────────────────────── + ── H ────────────────── + ── X'
 └─ RMSNorm₁ → Attention ─┘      └─ RMSNorm₂ → FFN ─┘
```

Each block has two 128-parameter gamma vectors. Four blocks contribute eight RMSNorm layers, and one final RMSNorm precedes logits: nine normalizations, totaling 1,152 parameters. Residual additions introduce no parameters.

## The gradient path

For a residual transformation `y=x+F(x)`, the Jacobian is `I+J_F`. Backpropagation has a direct identity contribution as well as the branch contribution. This helps explain how residuals support optimization.

The path **does not guarantee** that gradients never vanish or explode. Contributions may cancel, as with `F(x)=−x`, and products of Jacobians can amplify values. Pre-norm does not remove the need for an appropriate learning rate, initialization and gradient control during training.

## Map inspection and references

Compare `norm.attn` with `resid.attn`, and `norm.ffn` with `resid.ffn`, in the same layer. Normalization snapshots show branch inputs; residual snapshots show completed additions. `norm.final` belongs to the model output, after the fourth block. The inspector removes the batch axis and displays `(n,128)`.

Zhang and Sennrich introduce RMSNorm in [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467), 2019. Xiong et al. analyze pre-norm placement in [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745), 2020. That paper analyzes LayerNorm, whereas this project uses RMSNorm. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), He et al., first preprint in 2015, is a historical reference for residual connections in vision, not a description of this decoder.
