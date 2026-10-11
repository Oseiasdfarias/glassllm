import numpy as np
import pytest

from glassllm.autograd import Tensor, no_grad
from glassllm.model import Config, GlassLM
from glassllm.trace import Tracer


CFG = Config(vocab_size=50, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_len=16)


@pytest.fixture
def model():
    result = GlassLM(CFG, seed=0, dtype=np.float64)
    rng = np.random.default_rng(1)
    for parameter in result.parameters():
        parameter.data = rng.normal(0, 0.2, parameter.shape)
    return result


def test_next_token_probs_is_stable_and_applies_temperature():
    from glassllm.model import next_token_probs

    expected = np.exp([-1.0, -0.5, 0.0])
    expected /= expected.sum()
    np.testing.assert_allclose(next_token_probs(np.array([1001.0, 1002.0, 1003.0]), 2.0), expected)


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_next_token_probs_rejects_non_positive_temperature(temperature):
    from glassllm.model import next_token_probs

    with pytest.raises(ValueError, match="temperature.*greedy=True"):
        next_token_probs(np.array([1.0, 2.0]), temperature)


def test_kv_cache_gives_the_same_logits_as_full_forward(model):
    ids = [3, 17, 4, 4, 42, 9]
    with no_grad():
        full = model.forward(ids).data[0]
        cache = model.new_cache()
        prefix = model.forward(ids[:4], cache=cache, start=0).data[0]
        step5 = model.forward(ids[4:5], cache=cache, start=4).data[0]
        step6 = model.forward(ids[5:6], cache=cache, start=5).data[0]
    np.testing.assert_allclose(np.concatenate([prefix, step5, step6]), full, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("prompt", [[3, 17, 4], list(range(16)), list(range(1, 40))])
@pytest.mark.parametrize("greedy", [False, True])
def test_generation_matches_full_trailing_window_logits_probs_and_tokens(model, prompt, greedy):
    seq = list(prompt)
    rng = np.random.default_rng(7)
    for step in model.generate(prompt, CFG.max_len + 3, temperature=0.7, greedy=greedy, seed=7, trace=True):
        with no_grad():
            full_logits = model.forward(seq[-CFG.max_len:]).data[0, -1]
        expected = np.exp((full_logits - full_logits.max()) / 0.7)
        expected /= expected.sum()
        np.testing.assert_allclose(step.trace.get("logits").value[-1], full_logits, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(step.probs, expected, rtol=1e-9, atol=1e-12)
        expected_token = int(np.argmax(expected)) if greedy else int(rng.choice(CFG.vocab_size, p=expected))
        assert step.token == expected_token
        seq.append(step.token)
    assert len(seq) == len(prompt) + CFG.max_len + 3
    assert prompt == seq[:len(prompt)]


def test_generation_uses_cached_single_tokens_until_window_is_full(model):
    steps = list(model.generate([1, 2, 3], CFG.max_len + 2, greedy=True, trace=True))
    lengths = [step.trace.get("embed").shape[0] for step in steps]
    assert lengths == [3] + [1] * 13 + [CFG.max_len] * 4


def test_same_seed_same_tokens(model):
    a = [step.token for step in model.generate([1, 2], 8, seed=7)]
    b = [step.token for step in model.generate([1, 2], 8, seed=7)]
    assert a == b


def test_each_step_has_probabilities_and_a_fresh_independent_trace(model):
    from glassllm.model import GenStep

    steps = list(model.generate([1, 2, 3], 3, trace=True, seed=0))
    for step in steps:
        assert isinstance(step, GenStep)
        assert step.probs.shape == (CFG.vocab_size,)
        assert step.probs.sum() == pytest.approx(1.0)
        assert step.trace.get("sample").value == step.token
        np.testing.assert_array_equal(step.trace.get("probs").value, step.probs)
        assert step.trace.records[0].order == 0
    assert len({id(step.trace) for step in steps}) == len(steps)
    snapshot = steps[0].trace.get("probs").value.copy()
    steps[0].probs[:] = 0
    np.testing.assert_array_equal(steps[0].trace.get("probs").value, snapshot)
    steps[0].trace.records.clear()
    assert steps[1].trace.get("sample").value == steps[1].token
    assert next(model.generate([1], 1)).trace is None


def test_empty_prompt_is_a_clear_error(model):
    with pytest.raises(ValueError, match="empty"):
        next(model.generate([], 1))


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_non_positive_temperature_is_a_clear_error(model, temperature):
    with pytest.raises(ValueError, match="temperature.*greedy=True"):
        next(model.generate([1], 1, temperature=temperature))


@pytest.mark.parametrize("trace", [Tracer(), 1, "yes", None])
def test_trace_requires_a_bool_instead_of_silently_accepting_a_tracer(model, trace):
    with pytest.raises(TypeError, match="trace.*bool"):
        next(model.generate([1], 1, trace=trace))


def test_zero_new_tokens_yields_nothing(model):
    assert list(model.generate([1, 2], 0)) == []


def test_generation_past_max_len_slides_the_window(model):
    steps = list(model.generate([1, 2, 3], CFG.max_len * 3, seed=0))
    assert len(steps) == CFG.max_len * 3


def assert_caller_backward_works():
    leaf = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    result = (leaf * leaf).sum()
    assert result.requires_grad
    result.backward()
    np.testing.assert_array_equal(leaf.grad, [4.0, 6.0])


def test_generation_restores_autograd_between_yields_after_close_and_exhaustion(model):
    generator = model.generate([1, 2], 3)
    try:
        next(generator)
        assert_caller_backward_works()
        next(generator)
        assert_caller_backward_works()
    finally:
        generator.close()
    assert_caller_backward_works()
    list(model.generate([1], 1))
    assert_caller_backward_works()


@pytest.mark.parametrize("close_first", [0, 1])
def test_interleaved_generators_restore_autograd_when_closed_in_either_order(model, close_first):
    first = model.generate([1, 2], 3)
    second = model.generate([3, 4], 3)
    try:
        next(first)
        assert_caller_backward_works()
        next(second)
        assert_caller_backward_works()
        next(first)
        assert_caller_backward_works()
        generators = [first, second]
        generators[close_first].close()
        assert_caller_backward_works()
        next(generators[1 - close_first])
        assert_caller_backward_works()
    finally:
        first.close()
        second.close()
    assert_caller_backward_works()


def test_generation_preserves_an_enclosing_no_grad_scope(model):
    with no_grad():
        generator = model.generate([1], 2)
        next(generator)
        assert not Tensor([1.0], requires_grad=True).requires_grad
        generator.close()
        assert not Tensor([1.0], requires_grad=True).requires_grad
    assert_caller_backward_works()


def test_generation_forward_does_not_build_autograd_graphs(model, monkeypatch):
    original_forward = model.forward
    outputs = []

    def observe_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        outputs.append(output)
        return output

    monkeypatch.setattr(model, "forward", observe_forward)
    list(model.generate([1, 2], 3))
    assert len(outputs) == 3
    assert all(not output.requires_grad for output in outputs)
    assert_caller_backward_works()
