---
title: "Tensor and autograd engine"
description: "Arrays, operators, reverse differentiation, gradient accumulation and no_grad."
---

## Construct a Tensor

`from glassllm import Tensor, no_grad` is the public import. `Tensor(data, parents=(), backward=None, requires_grad=False)` converts `data` with `np.asarray`, imposing no dtype and not always making a copy. Ordinary usage supplies only `data` and `requires_grad`; `parents` and `backward` are the operation-construction mechanism.

| Attribute | Behavior |
| --- | --- |
| `data` | NumPy array of current values; may share memory with the input |
| `grad` | Initially `None`; accumulated gradient array after backward |
| `requires_grad` | True when requested or inherited from a parent, while tracking is enabled |
| `shape`, `ndim` | Properties reading the array |

Use real float32 or float64 arrays for derivatives. Integer data is accepted, but this does not provide discrete differentiation or integer-parameter training. There is no `.detach()`, `.item()`, `.T` or implicit PyTorch conversion; read `.data`, use `float(t.data)` for scalars and `.transpose()` for axes.

## Operators and methods

| Syntax | Operation and limit |
| --- | --- |
| `a+b`, `a-b`, `a*b`, `a/b` | Elementwise, with broadcasting and reverse operations such as `2-a` |
| `-a`, `a**exponent` | Negation and constant-exponent power; no exponent derivative |
| `a @ b` | NumPy matrix multiplication, vector promotion and batch broadcasting |
| `a[index]` | NumPy indexing; backward adds contributions at repeated indices |
| `a.sum(axis=None, keepdims=False)` | Reduction over an axis, axis tuple or all axes |
| `a.mean(axis=None, keepdims=False)` | Sum divided by the reduced element count |
| `a.max(axis=None, keepdims=False)` | Maximum; shares gradients across ties |
| `a.reshape(*shape)` | Separate dimensions or tuple/list, including `-1` |
| `a.transpose(*axes)` | No arguments reverses axes; separate axes or tuple/list accepted |

Operations return new Tensors; incompatible shapes and indices raise NumPy errors. `@` is implemented for a Tensor on the left; there is no dedicated `__rmatmul__`. Compare values using NumPy on `.data`: the class provides neither differentiable comparisons nor indexed assignment such as `a[i] = ...`.

## backward and seed gradients

`backward(grad=None) -> None` traverses the graph in reverse topological order. With no `grad`, it uses `ones_like(data)`: for a nonscalar output, that differentiates the **sum of its entries**. Supply a gradient matching the output shape for another vector-Jacobian product. An output with `requires_grad=False` does nothing.

The algorithm adds contributions from shared paths. Leaves accumulate gradients across calls; intermediate gradients are cleared per traversal to avoid reusing old values. Before a new training update, use `p.grad = None` or `optimizer.zero_grad()`. `backward` neither resets every leaf gradient nor updates parameters.

```python
import numpy as np
from glassllm import Tensor

x = Tensor([3.0], requires_grad=True)
y = x * x + x
y.backward()
np.testing.assert_allclose(x.grad, [7.0])
y.backward()
np.testing.assert_allclose(x.grad, [14.0])
x.grad = None
y.backward(np.array([2.0]))
np.testing.assert_allclose(x.grad, [14.0])
```

Broadcasting replicates values during forward, so backward **sums** replica contributions to recover the original shape. Indexing needs the same care when an embedding ID appears multiple times.

```python
import numpy as np
from glassllm import Tensor

x = Tensor(np.ones((2, 3)), requires_grad=True)
bias = Tensor([1.0, 2.0, 3.0], requires_grad=True)
(x + bias).sum().backward()
np.testing.assert_allclose(bias.grad, [2.0, 2.0, 2.0])
np.testing.assert_allclose(x.grad, np.ones((2, 3)))
ties = Tensor([2.0, 2.0, 1.0], requires_grad=True)
ties.max().backward()
np.testing.assert_allclose(ties.grad, [0.5, 0.5, 0.0])
```

## no_grad

`no_grad()` is a context manager: results constructed inside it retain neither parents nor backward functions. It does not erase an existing parameter's `requires_grad` or reset gradients. Nested scopes and exceptions restore the previous state on exit.

```python
from glassllm import Tensor, no_grad

x = Tensor([3.0], requires_grad=True)
with no_grad():
    y = x * 2
    assert not y.requires_grad
    y.backward()
assert x.grad is None
assert (x * 2).requires_grad
```

Tracking state is a module-global variable, not an isolated thread or coroutine context. Do not overlap training and inference in one process assuming automatic scope isolation.

## Graph limitations

Backward produces NumPy arrays without building a graph of the derivatives themselves: there is no higher-order autograd. Operations outside the documented functions, such as `np.sin(t.data)`, break the chain. Mutating `.data` after forward can invalidate values still used by backward closures; finish backward before updating parameters. Complex data has no conjugation-specific rules and is outside this engine's contract.

The project implements the reverse mode discussed in [Baydin et al.'s automatic differentiation survey](https://arxiv.org/abs/1502.05767), with a small set of vectorized operations rather than full framework infrastructure. The [functional reference](../operacoes-autograd/) explains derivatives and stability.
