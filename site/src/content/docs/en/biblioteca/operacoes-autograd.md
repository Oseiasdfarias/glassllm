---
title: "Differentiable operations"
description: "Signatures, shapes, derivatives and stability of glassllm.autograd functions."
---

## Common convention

Every function on this page is imported from `glassllm.autograd`, not the root. Except for `softmax_np` and `cross_entropy_grad`, they return `Tensor`. Numeric inputs and arrays become constant Tensors; a result participates in the graph when an input requires gradients and `no_grad` is inactive. Broadcasting, axis, shape and domain errors mainly follow NumPy rules.

Backward applies the chain rule and sums dimensions expanded by broadcasting. Gradients concern Tensors, not index selectors, masks, constant exponents, repeat counts or trigonometric tables. Operations support real data and first-order derivatives.

## Arithmetic and matrix products

| Signature | Forward and backward |
| --- | --- |
| `add(a, b)` | `a+b`; sends `g` to both, reducing broadcast axes |
| `neg(a)` | `-a`; sends `-g` |
| `mul(a, b)` | `a*b`; sends `g*b` and `g*a` |
| `div(a, b)` | `a/b`; sends `g/b` and `-g*a/b²` |
| `power(a, exponent)` | `a**exponent`; sends `g*exponent*a**(exponent-1)`; exponent zero sends zero |
| `matmul(a, b)` | `@` with vectors and batches; uses products with transposed final axes and undoes promotions |
| `exp(a)` | `exp(a)`; sends `g*exp(a)` |
| `log(a)` | Natural logarithm; sends `g/a` |

Division by zero, log of nonpositive values and powers outside the real domain have no dedicated error policy: warnings or inf/NaN may result. Plain `exp` does not clip. Distribution-specific numerical protections are described below.

## Reductions, shapes and selection

| Signature | Contract |
| --- | --- |
| `tsum(a, axis=None, keepdims=False)` | Sum; backward expands and broadcasts `g` to the original shape |
| `mean(a, axis=None, keepdims=False)` | `tsum / count`; accepts an axis or axis tuple |
| `max(a, axis=None, keepdims=False)` | Maximum; shares `g` equally across tied winners |
| `reshape(a, shape)` | New shape with equal element count; backward restores original shape |
| `transpose(a, axes)` | Permutation; `axes=None` reverses axes; backward uses inverse permutation |
| `concat(tensors, axis=0)` | Concatenates a nonempty Tensor sequence; backward splits each input's segment |
| `where(condition, a, b)` | Chooses `a` where True and `b` otherwise; only selected branches receive gradients |

`tsum` avoids confusion with Python's `sum`; `max` is a submodule function and a `Tensor.max` method. `condition` should be a shape-compatible boolean NumPy mask. Axes and element counts are not differentiable.

## Embedding and masks

`embedding(weight, ids)` directly looks up rows without a one-hot matrix. Weight `(V,d)` and IDs `(B,n)` produce `(B,n,d)`; backward accumulates repeated-row gradients with scatter-add. IDs must be valid integers; indexing retains NumPy semantics, including negative indices.

`masked_fill(a, mask, value)` is `where(mask, value, a)`. `mask=True` means **replace/block**, not allow. Masked positions receive no gradient from `a`. For attention, `value=-np.inf` produces zero future-position weights after softmax, provided each row has at least one unmasked key.

```python
import numpy as np
from glassllm import Tensor
from glassllm.autograd import embedding, masked_fill, softmax

weight = Tensor(np.zeros((4, 2)), requires_grad=True)
embedding(weight, np.array([1, 2, 1])).sum().backward()
np.testing.assert_array_equal(weight.grad[1], [2.0, 2.0])
np.testing.assert_array_equal(weight.grad[2], [1.0, 1.0])
scores = Tensor(np.zeros((3, 3)), requires_grad=True)
mask = np.triu(np.ones((3, 3), dtype=bool), k=1)
p = softmax(masked_fill(scores, mask, -np.inf))
assert np.all(p.data[mask] == 0)
np.testing.assert_allclose(p.data.sum(-1), 1.0)
```

## Distributions and activations

| Signature | Return and rule |
| --- | --- |
| `softmax_np(x, axis=-1)` | Array `exp(x-max) / sum(exp(x-max))`; no graph |
| `softmax(a, axis=-1)` | Same-shaped Tensor; backward `p*(g-sum(g*p))` on the chosen axis |
| `log_softmax(a, axis=-1)` | Stable log-probability Tensor; backward `g-exp(y)*sum(g)` |
| `sigmoid(a)` | `1/(1+exp(-a))`; stable implementation uses `exp(-abs(a))`; backward `g*s*(1-s)` |
| `silu(a)` | `a*sigmoid(a)`; backward `g*s*(1+a*(1-s))` |

Softmax and log-softmax subtract the maximum to avoid overflow for large finite logits. An all-`-inf` row or NaN/+inf input can still yield NaN. Sigmoid handles finite extremes such as ±10,000 without exponential overflow; SiLU uses the same care but does not promise finite results for non-finite inputs.

```python
import numpy as np
from glassllm import Tensor
from glassllm.autograd import log_softmax, sigmoid, silu, softmax_np

np.testing.assert_allclose(softmax_np([[1000.0, 1000.0]]), [[0.5, 0.5]])
np.testing.assert_allclose(log_softmax(Tensor([[1000.0, 1000.0]])).data,
                           [[-np.log(2.0), -np.log(2.0)]])
np.testing.assert_allclose(sigmoid(Tensor([-10000.0, 0.0, 10000.0])).data,
                           [0.0, 0.5, 1.0])
x = Tensor([0.0], requires_grad=True)
silu(x).sum().backward()
np.testing.assert_allclose(x.grad, [0.5])
```

## Cross-entropy

`cross_entropy(logits, targets)` expects `(N,V)` logits and integer `(N,)` targets; it returns scalar mean `-log p[target]`. It accepts neither one-hot target distributions nor `ignore_index`, label smoothing, class weights or configurable reduction. Do not apply softmax beforehand: input is unnormalized logits.

`cross_entropy_grad(logits, targets) -> np.ndarray` returns `(softmax(logits)-one_hot(targets))/N` shaped `(N,V)`, without a graph or modifying logits. Loss backward uses this function. Inputs require at least one sample and valid indices; incompatibilities raise NumPy errors and an empty batch can yield NaN.

```python
import numpy as np
from glassllm import Tensor
from glassllm.autograd import cross_entropy, cross_entropy_grad

logits = Tensor([[0.0, 0.0]], requires_grad=True)
targets = np.array([1], dtype=np.int64)
loss = cross_entropy(logits, targets)
np.testing.assert_allclose(loss.data, np.log(2.0))
loss.backward()
np.testing.assert_allclose(logits.grad, [[0.5, -0.5]])
np.testing.assert_allclose(cross_entropy_grad(logits.data, targets), logits.grad)
```

## RoPE and K/V repetition

`rope(x, cos, sin)` returns a same-shaped Tensor; the last axis needs coordinate pairs. `(n,dk/2)` `cos/sin` tables broadcast over batch and heads of `(B,H,n,dk)` input. Each adjacent `(even,odd)` pair rotates; backward applies the transposed rotation, negating the sine. Only `x` is differentiable. The function neither generates angles nor manually validates every input; use `rope_angles` from the [layer reference](../camadas-cache/).

`repeat_kv(x, repeats)` expects `(B,Hkv,n,dk)` and repeats adjacent heads to produce `(B,Hkv*repeats,n,dk)`. Backward groups the head axis and sums repetitions. Model usage requires a positive integer `repeats`; this function provides neither a repeat-count derivative nor dedicated GQA validation.

```python
import numpy as np
from glassllm import Tensor
from glassllm.autograd import repeat_kv, rope

x = Tensor([[1.0, 2.0, 3.0, 4.0]])
rotated = rope(x, np.zeros((1, 2)), np.ones((1, 2)))
np.testing.assert_allclose(rotated.data, [[-2.0, 1.0, -4.0, 3.0]])
heads = Tensor(np.array([1.0, 2.0]).reshape(1, 2, 1, 1), requires_grad=True)
repeated = repeat_kv(heads, 2)
np.testing.assert_array_equal(repeated.data.ravel(), [1, 1, 2, 2])
repeated.sum().backward()
np.testing.assert_array_equal(heads.grad.ravel(), [2, 2])
```

Foundations include [automatic differentiation](https://arxiv.org/abs/1502.05767), [RoPE](https://arxiv.org/abs/2104.09864) and [SwiGLU](https://arxiv.org/abs/2002.05202). These are mathematical building blocks, not complete reproductions of every paper's architecture or experiments.
