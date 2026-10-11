"""A tiny reverse-mode autograd engine on top of NumPy.

Each Tensor remembers its parents and a function that pushes gradients back to
them. backward() visits the graph in reverse topological order: micrograd's
idea, with arrays instead of scalars.
"""
from __future__ import annotations

import contextlib

import numpy as np

_grad_enabled = True


@contextlib.contextmanager
def no_grad():
    """Run without tracking gradients or recording the graph (inference)."""
    global _grad_enabled
    previous = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = previous


def _unbroadcast(grad, shape):
    # Broadcasting copies values; its backward pass sums those copies.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, parents=(), backward=None, requires_grad=False):
        self.data = np.asarray(data)
        self.grad = None
        self.requires_grad = _grad_enabled and (
            requires_grad or any(parent.requires_grad for parent in parents)
        )
        self._parents = tuple(parents) if self.requires_grad else ()
        self._backward = backward if self.requires_grad else None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    def _accumulate(self, grad):
        if self.requires_grad:
            grad = _unbroadcast(np.asarray(grad), self.shape)
            self.grad = grad if self.grad is None else self.grad + grad

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            grad = np.ones_like(self.data)
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif id(node) not in seen:
                seen.add(id(node))
                stack.append((node, True))
                stack.extend((parent, False) for parent in node._parents)
        # Leaves accumulate across calls; intermediate gradients belong to this
        # traversal only, otherwise shared paths get counted again next time.
        for node in order:
            if node._parents:
                node.grad = None
        self._accumulate(grad)
        for node in reversed(order):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)

    def __add__(self, other):
        return add(self, other)

    __radd__ = __add__

    def __sub__(self, other):
        return add(self, neg(other))

    def __rsub__(self, other):
        return add(other, neg(self))

    def __mul__(self, other):
        return mul(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    def __neg__(self):
        return neg(self)

    def __pow__(self, exponent):
        return power(self, exponent)

    def __matmul__(self, other):
        return matmul(self, other)

    def __getitem__(self, index):
        def back(g):
            grad = np.zeros(self.shape, dtype=g.dtype)
            np.add.at(grad, index, g)
            self._accumulate(grad)

        return Tensor(self.data[index], (self,), back)

    def sum(self, axis=None, keepdims=False):
        return tsum(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return mean(self, axis, keepdims)

    def max(self, axis=None, keepdims=False):
        return max(self, axis, keepdims)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = axes[0]
        return transpose(self, axes or None)


def _t(x):
    return x if isinstance(x, Tensor) else Tensor(x)


def add(a, b):
    a, b = _t(a), _t(b)

    def back(g):
        a._accumulate(g)
        b._accumulate(g)

    return Tensor(a.data + b.data, (a, b), back)


def neg(a):
    a = _t(a)

    def back(g):
        a._accumulate(-g)

    return Tensor(-a.data, (a,), back)


def mul(a, b):
    a, b = _t(a), _t(b)

    def back(g):
        a._accumulate(g * b.data)
        b._accumulate(g * a.data)

    return Tensor(a.data * b.data, (a, b), back)


def div(a, b):
    a, b = _t(a), _t(b)

    def back(g):
        a._accumulate(g / b.data)
        b._accumulate(-g * a.data / b.data**2)

    return Tensor(a.data / b.data, (a, b), back)


def power(a, exponent):
    a = _t(a)

    def back(g):
        if exponent == 0:
            a._accumulate(np.zeros_like(g))
        else:
            a._accumulate(g * exponent * a.data ** (exponent - 1))

    return Tensor(a.data**exponent, (a,), back)


def matmul(a, b):
    a, b = _t(a), _t(b)
    left = a.data[np.newaxis, :] if a.ndim == 1 else a.data
    right = b.data[:, np.newaxis] if b.ndim == 1 else b.data

    def back(g):
        # NumPy promotes vectors to matrices, then removes the inserted axes.
        if a.ndim == b.ndim == 1:
            g = g.reshape(1, 1)
        elif a.ndim == 1:
            g = np.expand_dims(g, -2)
        elif b.ndim == 1:
            g = np.expand_dims(g, -1)
        grad_a = g @ np.swapaxes(right, -1, -2)
        grad_b = np.swapaxes(left, -1, -2) @ g
        a._accumulate(np.squeeze(grad_a, -2) if a.ndim == 1 else grad_a)
        b._accumulate(np.squeeze(grad_b, -1) if b.ndim == 1 else grad_b)

    return Tensor(a.data @ b.data, (a, b), back)


def exp(a):
    a = _t(a)
    y = np.exp(a.data)

    def back(g):
        a._accumulate(g * y)

    return Tensor(y, (a,), back)


def log(a):
    a = _t(a)

    def back(g):
        a._accumulate(g / a.data)

    return Tensor(np.log(a.data), (a,), back)


def tsum(a, axis=None, keepdims=False):
    a = _t(a)

    def back(g):
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        a._accumulate(np.broadcast_to(g, a.shape))

    return Tensor(a.data.sum(axis=axis, keepdims=keepdims), (a,), back)


def mean(a, axis=None, keepdims=False):
    a = _t(a)
    count = a.data.size if axis is None else np.prod([a.shape[i] for i in np.atleast_1d(axis)])
    return tsum(a, axis, keepdims) / float(count)


def max(a, axis=None, keepdims=False):
    """Maximum reduction; ties share the gradient equally."""
    a = _t(a)
    maxima = a.data.max(axis=axis, keepdims=True)
    winners = a.data == maxima
    counts = winners.sum(axis=axis, keepdims=True)

    def back(g):
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        a._accumulate(g * winners / counts)

    return Tensor(a.data.max(axis=axis, keepdims=keepdims), (a,), back)


def reshape(a, shape):
    a = _t(a)

    def back(g):
        a._accumulate(g.reshape(a.shape))

    return Tensor(a.data.reshape(shape), (a,), back)


def transpose(a, axes):
    a = _t(a)
    axes = tuple(reversed(range(a.ndim))) if axes is None else tuple(axes)
    inverse = np.argsort([axis % a.ndim for axis in axes])

    def back(g):
        a._accumulate(g.transpose(inverse))

    return Tensor(a.data.transpose(axes), (a,), back)


def concat(tensors, axis=0):
    tensors = tuple(_t(tensor) for tensor in tensors)
    boundaries = np.cumsum([tensor.shape[axis] for tensor in tensors])[:-1]

    def back(g):
        for tensor, part in zip(tensors, np.split(g, boundaries, axis=axis)):
            tensor._accumulate(part)

    return Tensor(np.concatenate([tensor.data for tensor in tensors], axis=axis), tensors, back)


def where(condition, a, b):
    a, b = _t(a), _t(b)

    def back(g):
        a._accumulate(np.where(condition, g, 0.0))
        b._accumulate(np.where(condition, 0.0, g))

    return Tensor(np.where(condition, a.data, b.data), (a, b), back)


def embedding(weight, ids):
    """Row lookup E[ids], without constructing a one-hot matrix."""
    return _t(weight)[np.asarray(ids)]


def masked_fill(a, mask, value):
    return where(mask, value, a)


def softmax_np(x, axis=-1):
    x = np.asarray(x)
    # Subtracting a constant preserves probabilities and prevents overflow.
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def softmax(a, axis=-1):
    a = _t(a)
    y = softmax_np(a.data, axis)

    def back(g):
        # dp_i/ds_j = p_i (delta_ij - p_j).
        a._accumulate(y * (g - (g * y).sum(axis=axis, keepdims=True)))

    return Tensor(y, (a,), back)


def log_softmax(a, axis=-1):
    a = _t(a)
    z = a.data - a.data.max(axis=axis, keepdims=True)
    y = z - np.log(np.exp(z).sum(axis=axis, keepdims=True))

    def back(g):
        a._accumulate(g - np.exp(y) * g.sum(axis=axis, keepdims=True))

    return Tensor(y, (a,), back)


def _sigmoid_np(x):
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def sigmoid(a):
    a = _t(a)
    y = _sigmoid_np(a.data)

    def back(g):
        a._accumulate(g * y * (1.0 - y))

    return Tensor(y, (a,), back)


def silu(a):
    a = _t(a)
    s = _sigmoid_np(a.data)

    def back(g):
        a._accumulate(g * s * (1.0 + a.data * (1.0 - s)))

    return Tensor(a.data * s, (a,), back)


def cross_entropy_grad(logits, targets):
    """Gradient of mean cross-entropy with respect to logits: (p - y) / N."""
    targets = np.asarray(targets)
    p = softmax_np(logits, axis=-1)
    p[np.arange(len(targets)), targets] -= 1.0  # @step loss.grad
    return p / len(targets)


def cross_entropy(logits, targets):
    """Mean -log softmax(logits)[target]; logits: (N, V), targets: (N,)."""
    logits, targets = _t(logits), np.asarray(targets)
    z = logits.data - logits.data.max(axis=-1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    loss = -log_p[np.arange(len(targets)), targets].mean()

    def back(g):
        logits._accumulate(g * cross_entropy_grad(logits.data, targets))

    return Tensor(loss, (logits,), back)


def _rotate(x, cos, sin):
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


def rope(x, cos, sin):
    """Rotate each adjacent coordinate pair by its position-dependent angle."""
    x = _t(x)

    def back(g):
        # A rotation's transpose is the rotation by the opposite angle.
        x._accumulate(_rotate(g, cos, -np.asarray(sin)))

    return Tensor(_rotate(x.data, cos, sin), (x,), back)


def repeat_kv(x, repeats):
    """GQA: (B, H_kv, n, d_k) -> (B, H_kv * repeats, n, d_k)."""
    x = _t(x)
    b, h_kv, n, d_k = x.shape

    def back(g):
        x._accumulate(g.reshape(b, h_kv, repeats, n, d_k).sum(axis=2))

    return Tensor(np.repeat(x.data, repeats, axis=1), (x,), back)
