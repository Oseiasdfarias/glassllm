import numpy as np
import pytest

from glassllm.autograd import (Tensor, add, concat, cross_entropy, cross_entropy_grad,
                               div, embedding, exp, log, log_softmax, masked_fill,
                               matmul, max as tensor_max, mean, mul, neg, no_grad,
                               power, repeat_kv, reshape, rope, sigmoid, silu,
                               softmax, softmax_np, transpose, tsum, where)

RNG = np.random.default_rng(0)


def numeric_grad(f, x, eps=1e-6):
    """Central finite difference of the scalar f() with respect to every entry of x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        plus = f()
        x[i] = old - eps
        minus = f()
        x[i] = old
        grad[i] = (plus - minus) / (2 * eps)
    return grad


def check(op, *shapes, positive=False):
    """Compare autograd with finite differences for loss = sum(op(*inputs) * w)."""
    arrays = [RNG.normal(size=s) for s in shapes]
    if positive:
        arrays = [np.abs(a) + 0.5 for a in arrays]
    tensors = [Tensor(a, requires_grad=True) for a in arrays]
    out = op(*tensors)
    w = RNG.normal(size=out.shape)
    (out * w).sum().backward()
    for a, t in zip(arrays, tensors):
        expected = numeric_grad(lambda: float((op(*[Tensor(b) for b in arrays]).data * w).sum()), a)
        np.testing.assert_allclose(t.grad, expected, rtol=1e-5, atol=1e-7)


def test_elementwise_ops():
    check(lambda a, b: a + b, (3, 4), (3, 4))
    check(lambda a, b: a - b, (3, 4), (3, 4))
    check(lambda a, b: a * b, (3, 4), (3, 4))
    check(lambda a, b: a / b, (3, 4), (3, 4), positive=True)
    check(lambda a: a**0.5, (3, 4), positive=True)
    check(lambda a: exp(a), (3, 4))
    check(lambda a: log(a), (3, 4), positive=True)
    check(lambda a: silu(a), (3, 4))


def test_broadcasting_sums_the_gradient():
    check(lambda a, b: a * b, (2, 3, 4), (4,))
    check(lambda a, b: a + b, (2, 3, 4), (3, 1))


def test_matmul_batched_and_shared_weight():
    check(lambda a, b: a @ b, (3, 4), (4, 5))
    check(lambda a, b: a @ b, (2, 3, 4), (4, 5))
    check(lambda a, b: a @ b, (2, 2, 3, 4), (2, 2, 4, 3))


def test_reductions_and_shapes():
    check(lambda a: a.sum(axis=-1), (3, 4))
    check(lambda a: a.mean(axis=-1, keepdims=True), (3, 4))
    check(lambda a: a.reshape(4, 3), (3, 4))
    check(lambda a: transpose(a, (1, 0, 2)), (2, 3, 4))


def test_softmax_gradient_and_rows_sum_to_one():
    check(lambda a: softmax(a, axis=-1), (3, 5))
    y = softmax(Tensor(RNG.normal(size=(4, 6)))).data
    np.testing.assert_allclose(y.sum(axis=-1), 1.0)


def test_softmax_does_not_overflow_with_large_scores():
    y = softmax(Tensor(np.array([[1000.0, 1001.0, 999.0]]))).data
    assert np.all(np.isfinite(y))
    np.testing.assert_allclose(y.sum(), 1.0)


def test_masked_fill_blocks_gradient_and_softmax_gives_zero_weight():
    mask = np.triu(np.ones((4, 4), dtype=bool), k=1)
    check(lambda a: softmax(masked_fill(a, mask, -np.inf)), (4, 4))
    y = softmax(masked_fill(Tensor(RNG.normal(size=(4, 4))), mask, -np.inf)).data
    assert np.all(y[mask] == 0)


def test_cross_entropy_gradient_is_p_minus_y():
    logits = RNG.normal(size=(5, 7))
    targets = np.array([0, 3, 6, 2, 2])
    t = Tensor(logits, requires_grad=True)
    cross_entropy(t, targets).backward()
    p = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    y = np.eye(7)[targets]
    np.testing.assert_allclose(t.grad, (p - y) / 5, rtol=1e-6, atol=1e-10)


def test_embedding_accumulates_repeated_tokens():
    ids = np.array([[1, 2, 1]])
    check(lambda w: embedding(w, ids), (4, 3))


def test_rope_gradient_and_relative_position_property():
    d_k = 8
    theta = 10000.0 ** (-2 * np.arange(d_k // 2) / d_k)
    angles = np.outer(np.arange(64), theta)
    cos, sin = np.cos(angles), np.sin(angles)
    check(lambda x: rope(x, cos[:3], sin[:3]), (1, 2, 3, d_k))

    q, k = RNG.normal(size=d_k), RNG.normal(size=d_k)

    def score(m, n):
        qm = rope(Tensor(q[None]), cos[m:m + 1], sin[m:m + 1]).data[0]
        kn = rope(Tensor(k[None]), cos[n:n + 1], sin[n:n + 1]).data[0]
        return qm @ kn

    np.testing.assert_allclose(score(5, 2), score(40, 37))


def test_repeat_kv_gradient():
    check(lambda x: repeat_kv(x, 2), (1, 2, 3, 4))


def test_no_grad_builds_no_graph():
    a = Tensor(RNG.normal(size=3), requires_grad=True)
    with no_grad():
        b = a * 2
    assert b._parents == ()


def test_backward_through_shared_node_adds_both_paths():
    a = Tensor(np.array([3.0]), requires_grad=True)
    b = a * a + a  # d/da = 2a + 1 = 7
    b.backward()
    np.testing.assert_allclose(a.grad, [7.0])


@pytest.mark.parametrize("bad", [np.inf, np.nan])
def test_nonfinite_input_does_not_crash_silu(bad):
    out = silu(Tensor(np.array([bad, -1e4, 1e4]))).data
    assert out.shape == (3,)


@pytest.mark.parametrize("op", [
    lambda a: 2 + a, lambda a: 2 - a, lambda a: 2 * a,
    lambda a: 2 / a, lambda a: -a,
])
def test_reverse_arithmetic_gradients(op):
    check(op, (2, 3), positive=True)


@pytest.mark.parametrize("shapes", [
    ((4,), (4,)), ((4,), (4, 3)), ((3, 4), (4,)),
    ((4,), (2, 4, 3)), ((2, 3, 4), (4,)),
    ((2, 1, 3, 4), (1, 2, 4, 3)),
])
def test_matmul_vector_promotion_and_batch_broadcast(shapes):
    check(lambda a, b: a @ b, *shapes)


@pytest.mark.parametrize("axis, keepdims", [
    (None, False), ((0, 2), False), ((-1, 0), True), ((), False),
])
def test_sum_and_mean_tuple_axes(axis, keepdims):
    check(lambda a: a.sum(axis=axis, keepdims=keepdims), (2, 3, 4))
    check(lambda a: a.mean(axis=axis, keepdims=keepdims), (2, 3, 4))


def test_tensor_transpose_default_tuple_and_variadic_axes():
    check(lambda a: a.transpose(), (2, 3, 4))
    check(lambda a: a.transpose((2, 0, 1)), (2, 3, 4))
    check(lambda a: a.transpose(2, 0, 1), (2, 3, 4))
    check(lambda a: a.reshape((4, 6)), (2, 3, 4))


@pytest.mark.parametrize("axis, keepdims", [
    (None, False), (-1, False), ((0, 2), True), ((0, 2), False),
])
def test_max_gradient_for_unique_extrema(axis, keepdims):
    check(lambda a: tensor_max(a, axis=axis, keepdims=keepdims), (2, 3, 4))


def test_max_splits_gradient_between_tied_extrema():
    a = Tensor([[2.0, 2.0, 1.0], [0.0, 3.0, 1.0]], requires_grad=True)
    a.max(axis=1).sum().backward()
    np.testing.assert_allclose(a.grad, [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]])


@pytest.mark.parametrize("axis, shapes", [
    (0, ((2, 3), (1, 3))), (-1, ((2, 2), (2, 3))),
])
def test_concat_routes_gradient_to_each_input(axis, shapes):
    check(lambda a, b: concat((a, b), axis=axis), *shapes)


def test_where_routes_broadcast_gradients_to_selected_branch():
    mask = np.array([[True, False, True], [False, True, False]])
    check(lambda a, b: where(mask, a, b), (2, 3), (3,))


@pytest.mark.parametrize("index", [
    (slice(None), slice(1, None)), ([0, 0, 2], [1, 1, 3]),
    np.array([True, False, True]), (None, Ellipsis, slice(None, None, 2)),
])
def test_indexing_scatter_adds_gradient_including_repeated_indices(index):
    check(lambda a: a[index], (3, 4))


def test_sigmoid_gradient_and_extreme_probabilities():
    check(sigmoid, (2, 3))
    with np.errstate(over="raise", invalid="raise"):
        actual = sigmoid(Tensor([-1e4, 0.0, 1e4])).data
    np.testing.assert_allclose(actual, [0.0, 0.5, 1.0])


@pytest.mark.parametrize("axis", [0, -1])
def test_log_softmax_gradient_and_large_scores(axis):
    check(lambda a: log_softmax(a, axis), (2, 3))
    actual = log_softmax(Tensor([[1000.0, 1000.0]])).data
    np.testing.assert_allclose(actual, [[-np.log(2.0), -np.log(2.0)]])


def test_cross_entropy_finite_difference_and_extreme_loss():
    targets = np.array([1, 0, 2])
    check(lambda a: cross_entropy(a, targets), (3, 4))
    a = Tensor([[1000.0, -1000.0]], requires_grad=True)
    loss = cross_entropy(a, np.array([1]))
    loss.backward()
    np.testing.assert_allclose(loss.data, 2000.0)
    np.testing.assert_allclose(a.grad, [[1.0, -1.0]])
    np.testing.assert_allclose(cross_entropy_grad(a.data, np.array([1])), a.grad)


def test_forward_operations_match_hand_checked_values():
    a, b = Tensor([[1.0, 2.0], [3.0, 4.0]]), Tensor([2.0, 4.0])
    for actual, expected in [
        (add(a, b), [[3, 6], [5, 8]]), (neg(a), [[-1, -2], [-3, -4]]),
        (mul(a, b), [[2, 8], [6, 16]]), (div(a, b), [[0.5, 0.5], [1.5, 1]]),
        (power(a, 2), [[1, 4], [9, 16]]), (matmul(a, b), [10, 22]),
        (tsum(a, 0), [4, 6]), (mean(a, 1), [1.5, 3.5]),
        (reshape(a, (4,)), [1, 2, 3, 4]), (transpose(a, (1, 0)), [[1, 3], [2, 4]]),
        (concat((a, a), axis=1), [[1, 2, 1, 2], [3, 4, 3, 4]]),
        (where([True, False], a, b), [[1, 4], [3, 4]]),
        (embedding(a, [1, 0, 1]), [[3, 4], [1, 2], [3, 4]]),
    ]:
        np.testing.assert_allclose(actual.data, expected)
    np.testing.assert_allclose(exp(Tensor([0.0, 1.0])).data, [1, np.e])
    np.testing.assert_allclose(log(Tensor([1.0, np.e])).data, [0, 1])
    np.testing.assert_allclose(softmax_np([[1000.0, 1000.0]]), [[0.5, 0.5]])


def test_nested_no_grad_restores_tracking_and_blocks_backpropagation():
    a = Tensor([3.0], requires_grad=True)
    with no_grad():
        b = a * 2
        with no_grad():
            c = a + 1
        d = a / 2
        for value in (b, c, d):
            assert not value.requires_grad
            value.sum().backward()
    assert a.grad is None
    (a * 2).sum().backward()
    np.testing.assert_allclose(a.grad, [2.0])


def test_no_grad_restores_tracking_after_exception():
    a = Tensor([3.0], requires_grad=True)
    with pytest.raises(RuntimeError):
        with no_grad():
            raise RuntimeError("leave inference")
    (a * 2).backward()
    np.testing.assert_allclose(a.grad, [2.0])


def test_repeat_kv_identity_respects_no_grad():
    a = Tensor(np.ones((1, 2, 3, 4)), requires_grad=True)
    with no_grad():
        b = repeat_kv(a, 1)
    assert not b.requires_grad


def test_repeated_backward_accumulates_leaves_without_reusing_intermediate_gradients():
    a = Tensor([3.0], requires_grad=True)
    b = a * a
    c = b * 2 + b
    c.backward()
    np.testing.assert_allclose(a.grad, [18.0])
    c.backward()
    np.testing.assert_allclose(a.grad, [36.0])
    c.backward(np.array([2.0]))
    np.testing.assert_allclose(a.grad, [72.0])


def test_backward_uses_custom_gradient_without_changing_constants():
    a = Tensor([2.0, 3.0], requires_grad=True)
    constant = Tensor([5.0, 7.0])
    (a * constant).backward(np.array([2.0, 4.0]))
    np.testing.assert_allclose(a.grad, [10.0, 28.0])
    assert constant.grad is None


def test_rope_rotates_adjacent_coordinates_and_repeat_kv_preserves_head_order():
    a = Tensor([[1.0, 2.0, 3.0, 4.0]])
    np.testing.assert_allclose(rope(a, np.zeros((1, 2)), np.ones((1, 2))).data,
                               [[-2.0, 1.0, -4.0, 3.0]])
    heads = Tensor(np.array([1.0, 2.0]).reshape(1, 2, 1, 1))
    np.testing.assert_allclose(repeat_kv(heads, 3).data.ravel(), [1, 1, 1, 2, 2, 2])
    check(lambda x: repeat_kv(x, 1), (1, 2, 3, 4))


def test_silu_finite_extremes_have_correct_values_and_gradients():
    a = Tensor([-1e4, 0.0, 1e4], requires_grad=True)
    with np.errstate(over="raise", invalid="raise"):
        result = silu(a)
        result.sum().backward()
    np.testing.assert_allclose(result.data, [0.0, 0.0, 1e4])
    np.testing.assert_allclose(a.grad, [0.0, 0.5, 1.0])


def test_backward_on_leaf_accumulates_each_seed():
    a = Tensor([3.0], requires_grad=True)
    a.backward()
    a.backward(np.array([2.0]))
    np.testing.assert_allclose(a.grad, [3.0])


def test_power_zero_has_zero_gradient_even_at_zero_input():
    a = Tensor([0.0, 2.0], requires_grad=True)
    with np.errstate(divide="raise", invalid="raise"):
        result = a**0
        result.sum().backward()
    np.testing.assert_allclose(result.data, [1.0, 1.0])
    np.testing.assert_allclose(a.grad, [0.0, 0.0])
