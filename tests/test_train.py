import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from glassllm.autograd import Tensor
from glassllm.config import Config
from glassllm.model import GlassLM
from glassllm.optim import AdamW, clip_grad_norm, lr_schedule
from glassllm.train import evaluate, get_batch, train


def test_adamw_finds_the_minimum_of_a_quadratic():
    w = Tensor(np.zeros(3), requires_grad=True)
    opt = AdamW([w], lr=0.1, weight_decay=0.0)
    for _ in range(300):
        opt.zero_grad()
        ((w - 3.0) * (w - 3.0)).sum().backward()
        opt.step()
    np.testing.assert_allclose(w.data, 3.0, atol=1e-2)


def test_weight_decay_only_touches_matrices():
    vec = Tensor(np.ones(3), requires_grad=True)
    mat = Tensor(np.ones((2, 2)), requires_grad=True)
    opt = AdamW([vec, mat], lr=0.1, weight_decay=0.5)
    vec.grad, mat.grad = np.zeros(3), np.zeros((2, 2))
    opt.step()
    np.testing.assert_allclose(vec.data, 1.0)
    np.testing.assert_allclose(mat.data, 0.95)


def test_adamw_bias_correction_lr_override_and_missing_gradient():
    w = Tensor(np.array([1.0]), requires_grad=True)
    unused = Tensor(np.ones((2, 2)), requires_grad=True)
    opt = AdamW([w, unused], lr=0.1, betas=(0.5, 0.5), eps=0.0)
    w.grad = np.array([2.0])
    opt.step(lr=0.2)
    np.testing.assert_allclose(w.data, [0.8])
    w.grad = np.array([0.0])
    opt.step()
    # m_hat = 2/3, v_hat = 4/3 on the second update.
    np.testing.assert_allclose(w.data, [0.8 - 0.1 / np.sqrt(3)])
    np.testing.assert_array_equal(unused.data, np.ones((2, 2)))
    opt.zero_grad()
    assert w.grad is None


def test_adamw_bias_correction_counts_updates_per_parameter():
    early, late = (Tensor(np.array([1.0]), requires_grad=True) for _ in range(2))
    opt = AdamW([early, late], lr=0.1, betas=(0.5, 0.5), eps=0.0,
                weight_decay=0.0)
    early.grad = np.array([1.0])
    late.grad = None
    opt.step()
    early.grad = np.array([0.0])
    late.grad = np.array([1.0])
    opt.step()
    # The late parameter's first update must receive first-step bias correction.
    np.testing.assert_allclose(late.data, [0.9])


def test_clip_grad_norm():
    a = Tensor(np.zeros(2), requires_grad=True)
    a.grad = np.array([3.0, 4.0])
    assert clip_grad_norm([a], 1.0) == pytest.approx(5.0)
    assert np.linalg.norm(a.grad) == pytest.approx(1.0, rel=1e-5)


def test_clip_grad_norm_is_global_and_leaves_small_gradients_alone():
    a, b, unused = (Tensor(np.zeros(1), requires_grad=True) for _ in range(3))
    a.grad, b.grad = np.array([3.0]), np.array([4.0])
    assert clip_grad_norm([a, b, unused], 10.0) == pytest.approx(5.0)
    np.testing.assert_array_equal(a.grad, [3.0])
    assert clip_grad_norm([a, b, unused], 1.0) == pytest.approx(5.0)
    np.testing.assert_allclose(a.grad, [0.6], rtol=1e-5)
    np.testing.assert_allclose(b.grad, [0.8], rtol=1e-5)
    assert clip_grad_norm([unused], 1.0) == 0.0


def test_lr_schedule_warms_up_then_decays():
    lrs = [lr_schedule(s, 100, 1.0, warmup=10) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1)
    assert lrs[9] == pytest.approx(1.0)
    assert lrs[-1] == pytest.approx(0.1, abs=0.01)
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:]))


def test_lr_schedule_without_warmup_and_after_completion():
    assert lr_schedule(0, 10, 2.0, warmup=0) == 2.0
    assert lr_schedule(5, 10, 2.0, warmup=0) == pytest.approx(1.1)
    assert lr_schedule(20, 10, 2.0, warmup=0) == pytest.approx(0.2)


def test_get_batch_targets_are_shifted_by_one():
    data = np.arange(100)
    x, y = get_batch(data, 4, 10, np.random.default_rng(0))
    assert x.shape == y.shape == (4, 10)
    np.testing.assert_array_equal(y, x + 1)


def test_get_batch_includes_the_last_valid_window():
    x, y = get_batch(np.arange(5), 3, 4, np.random.default_rng(0))
    np.testing.assert_array_equal(x, [[0, 1, 2, 3]] * 3)
    np.testing.assert_array_equal(y, [[1, 2, 3, 4]] * 3)


def test_get_batch_rejects_tiny_corpus():
    with pytest.raises(ValueError, match="seq_len"):
        get_batch(np.arange(5), 2, 10, np.random.default_rng(0))


def tiny_model(seed=0):
    return GlassLM(Config(vocab_size=8, d_model=8, n_layers=1, n_heads=2,
                          n_kv_heads=1, max_len=4), seed=seed)


def test_evaluate_is_deterministic_without_modifying_gradients():
    model = tiny_model()
    data = np.tile(np.arange(8), 5)
    for p in model.parameters():
        p.grad = np.ones_like(p.data)
    weights = [p.data.copy() for p in model.parameters()]
    first = evaluate(model, data, 2, 4, n_batches=3)
    assert np.isfinite(first)
    assert first == evaluate(model, data, 2, 4, n_batches=3)
    for p, w in zip(model.parameters(), weights):
        np.testing.assert_array_equal(p.data, w)
        np.testing.assert_array_equal(p.grad, np.ones_like(p.data))
    assert model.loss(data[:4], data[1:5]).requires_grad


def test_training_learns_a_repeating_pattern():
    cfg = Config(vocab_size=8, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, max_len=16)
    model = GlassLM(cfg, seed=0)
    data = np.tile(np.arange(8), 50)
    history = train(model, data, steps=150, batch_size=8, seq_len=16, lr=1e-2,
                    warmup=10, log=None)
    assert history[0]["loss"] > 1.5
    assert history[-1]["loss"] < 0.3


def test_keep_best_ends_with_the_best_validation_weights():
    cfg = Config(vocab_size=8, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, max_len=16)
    model = GlassLM(cfg, seed=0)
    data = np.tile(np.arange(8), 50)
    val = np.tile(np.arange(8)[::-1], 10)
    history = train(model, data, steps=120, batch_size=8, seq_len=16, lr=1e-2,
                    warmup=10, val_data=val, eval_every=10, keep_best=True, log=None)
    best = min(h["val_loss"] for h in history if "val_loss" in h)
    assert best < history[-1]["val_loss"]
    assert evaluate(model, val, 8, 16) == pytest.approx(best)


def test_training_seed_history_validation_cadence_and_logs():
    data = np.tile(np.arange(8), 5)
    a, b = tiny_model(), tiny_model()
    logs = []
    kwargs = dict(steps=4, batch_size=2, seq_len=4, warmup=1,
                  val_data=data, eval_every=2)
    history = train(a, data, **kwargs, log=logs.append)
    assert history == train(b, data, **kwargs, log=None)
    assert [h["step"] for h in history] == [0, 1, 2, 3]
    assert [h["step"] for h in history if "val_loss" in h] == [0, 2, 3]
    assert all(set(h) >= {"step", "loss", "lr"} for h in history)
    assert len(logs) == 3
    assert all("val " in line for line in logs)
    for p, q in zip(a.parameters(), b.parameters()):
        np.testing.assert_array_equal(p.data, q.data)


def deadline_model(monkeypatch, expire_on_validation=None):
    training = importlib.import_module("glassllm.train")
    clock = {"now": 0.0, "validation_batches": 0}
    monkeypatch.setattr(training, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
    model = tiny_model()
    real_loss = model.loss
    best_weights = []

    def timed_loss(x, y):
        result = real_loss(x, y)
        if result.requires_grad and expire_on_validation is None:
            clock["now"] = 7201.0
        elif not result.requires_grad:
            clock["validation_batches"] += 1
            if clock["validation_batches"] == 10:
                best_weights.extend(p.data.copy() for p in model.parameters())
            if clock["validation_batches"] == expire_on_validation:
                clock["now"] = 7201.0
        return result

    monkeypatch.setattr(model, "loss", timed_loss)
    return model, clock, best_weights


def test_training_stops_before_update_when_loss_exceeds_deadline(monkeypatch):
    model, _, _ = deadline_model(monkeypatch)
    original = [p.data.copy() for p in model.parameters()]
    with pytest.raises(TimeoutError, match="2.hour"):
        train(model, np.tile(np.arange(8), 5), steps=2, batch_size=2, seq_len=4, log=None)
    for p, w in zip(model.parameters(), original):
        np.testing.assert_array_equal(p.data, w)


def test_validation_time_counts_toward_training_deadline(monkeypatch):
    model, clock, _ = deadline_model(monkeypatch, expire_on_validation=1)
    data = np.tile(np.arange(8), 5)
    with pytest.raises(TimeoutError, match="2.hour"):
        train(model, data, steps=2, batch_size=2, seq_len=4, val_data=data, log=None)
    assert clock["validation_batches"] == 1
    # no_grad must unwind even when validation is interrupted.
    assert model.loss(data[:4], data[1:5]).requires_grad


def test_deadline_restores_best_completed_validation_weights(monkeypatch):
    model, _, best_weights = deadline_model(monkeypatch, expire_on_validation=11)
    data = np.tile(np.arange(8), 5)
    with pytest.raises(TimeoutError, match="2.hour"):
        train(model, data, steps=3, batch_size=2, seq_len=4, val_data=data,
              eval_every=1, keep_best=True, log=None)
    assert best_weights
    for p, w in zip(model.parameters(), best_weights):
        np.testing.assert_array_equal(p.data, w)


def test_clipping_large_float32_gradients_remains_finite():
    p = Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)
    p.grad = np.array([3e20, 4e20], dtype=np.float32)
    with np.errstate(over="raise"):
        assert clip_grad_norm([p], 1.0) == pytest.approx(5e20)
    np.testing.assert_allclose(p.grad, [0.6, 0.8], rtol=1e-5)


@pytest.mark.parametrize("batch_size, seq_len", [(0, 4), (2, 0), (-1, 4), (2, -1)])
def test_get_batch_rejects_nonpositive_dimensions(batch_size, seq_len):
    with pytest.raises(ValueError):
        get_batch(np.arange(20), batch_size, seq_len, np.random.default_rng(0))


def test_evaluate_rejects_zero_batches():
    with pytest.raises(ValueError, match="n_batches"):
        evaluate(tiny_model(), np.arange(8), 2, 4, n_batches=0)


@pytest.mark.parametrize("kwargs", [{"steps": -1}, {"eval_every": 0}, {"warmup": -1}])
def test_train_rejects_invalid_controls_before_mutating_weights(kwargs):
    model = tiny_model()
    original = [p.data.copy() for p in model.parameters()]
    settings = dict(steps=2, batch_size=2, seq_len=4, log=None)
    settings.update(kwargs)
    with pytest.raises(ValueError):
        train(model, np.arange(8), **settings)
    for p, w in zip(model.parameters(), original):
        np.testing.assert_array_equal(p.data, w)
