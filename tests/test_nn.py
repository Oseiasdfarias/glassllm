"""Layer contracts and independent float64 forward/backward parity."""
import numpy as np
import pytest
import torch

from glassllm.autograd import Tensor, no_grad, rope
from glassllm.config import Config
from glassllm.nn import Attention, Block, KVCache, Module, RMSNorm, SwiGLU, causal_mask, init_param, rope_angles
from glassllm.steps import LAYER_STEPS
from glassllm.trace import Tracer

from tests import torch_reference as ref

torch.set_num_threads(1)


def assert_parity(module, values, forward, reference):
    """A weighted loss exercises every output and all parameter paths."""
    rng = np.random.default_rng(12)
    for p in module.parameters():
        p.data = rng.normal(0, 0.3, size=p.shape)
    x = Tensor(values, requires_grad=True)
    tx = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    params = ref.to_torch(module)
    out = forward(x)
    expected = reference(params, tx)
    np.testing.assert_allclose(out.data, expected.detach().numpy(), rtol=1e-8, atol=1e-10)
    seed = rng.normal(size=out.shape)
    (out * seed).sum().backward()
    (expected * torch.tensor(seed)).sum().backward()
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), rtol=1e-6, atol=1e-10, err_msg="input")
    for name, p in module.named_parameters().items():
        assert p.grad is not None and params[name].grad is not None, name
        np.testing.assert_allclose(p.grad, params[name].grad.numpy(), rtol=1e-6, atol=1e-10, err_msg=name)


def test_config_defaults_construct_the_planned_block():
    cfg = Config()
    assert (cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
            cfg.max_len, cfg.rope_base, cfg.eps, cfg.init_std) == (1024, 128, 4, 4, 2, 128, 10000.0, 1e-6, 0.02)
    block = Block(cfg, np.random.default_rng(0), np.float32)
    assert block.attn.W_K.shape == (128, 64)
    assert all(p.data.dtype == np.float32 for p in block.parameters())


@pytest.mark.parametrize("name", ["vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "max_len"])
@pytest.mark.parametrize("value", [0, -1])
def test_config_rejects_nonpositive_dimensions_before_division(name, value):
    with pytest.raises(ValueError, match=name):
        Config(**{name: value})


@pytest.mark.parametrize("fields", [dict(d_model=32, n_heads=3), dict(n_heads=4, n_kv_heads=3),
                                   dict(d_model=12, n_heads=4, n_kv_heads=2)])
def test_config_rejects_incompatible_head_dimensions(fields):
    with pytest.raises(ValueError):
        Config(**fields)


def test_rope_angles_match_independent_frequencies():
    cos, sin = rope_angles(8, 10, 10000.0)
    tc, ts = ref.rope_angles(8, 10, 10000.0)
    assert cos.shape == sin.shape == (10, 4)
    np.testing.assert_allclose(cos[0], 1.0)
    np.testing.assert_allclose(sin[0], 0.0)
    np.testing.assert_allclose(cos, tc.numpy(), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(sin, ts.numpy(), rtol=1e-8, atol=1e-10)
    assert np.all(np.diff(np.arctan2(sin[1], cos[1])) < 0)


def test_rope_forward_and_input_gradient_match_pytorch_at_nonzero_positions():
    values = np.random.default_rng(5).normal(size=(2, 3, 4, 8))
    x = Tensor(values, requires_grad=True)
    tx = torch.tensor(values, requires_grad=True)
    cos, sin = rope_angles(8, 9, 700.0)
    tc, ts = ref.rope_angles(8, 9, 700.0)
    actual = rope(x, cos[3:7], sin[3:7])
    expected = ref.rope(tx, tc[3:7], ts[3:7])
    np.testing.assert_allclose(actual.data, expected.detach().numpy(), rtol=1e-8, atol=1e-10)
    seed = np.random.default_rng(6).normal(size=values.shape)
    (actual * seed).sum().backward()
    (expected * torch.tensor(seed)).sum().backward()
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), rtol=1e-6, atol=1e-10)


def test_causal_mask_with_and_without_cache():
    np.testing.assert_array_equal(causal_mask(3, 3, 0), [[False, True, True], [False, False, True], [False, False, False]])
    assert not causal_mask(1, 5, 4).any()
    np.testing.assert_array_equal(causal_mask(2, 5, 3), [[False, False, False, False, True], [False] * 5])


def test_rmsnorm_ignores_input_scale():
    norm = RMSNorm(6, 1e-12, np.float64)
    x = np.random.default_rng(0).normal(size=(2, 6))
    np.testing.assert_allclose(norm(Tensor(x)).data, norm(Tensor(10 * x)).data)
    np.testing.assert_allclose(np.sqrt((norm(Tensor(x)).data ** 2).mean(axis=-1)), 1.0)


def test_rmsnorm_forward_input_and_gamma_gradients_match_pytorch():
    norm = RMSNorm(8, 1e-6, np.float64)
    values = np.random.default_rng(0).normal(size=(2, 5, 8))
    assert_parity(norm, values, norm, lambda p, x: ref.rmsnorm(x, p["gamma"], norm.eps))


def test_swiglu_hidden_size_keeps_about_eight_d_squared_parameters():
    ffn = SwiGLU(48, np.random.default_rng(0), 0.02, np.float64)
    assert ffn.W1.shape == ffn.W3.shape == (48, 128)
    assert ffn.W2.shape == (128, 48)
    assert abs(sum(p.data.size for p in ffn.parameters()) - 8 * 48 * 48) <= 3 * 48


def test_swiglu_forward_input_and_parameter_gradients_match_pytorch():
    ffn = SwiGLU(8, np.random.default_rng(0), 0.02, np.float64)
    values = np.random.default_rng(0).normal(size=(2, 5, 8))
    assert_parity(ffn, values, lambda x: ffn(x, 0), ref.swiglu)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_attention_and_gqa_forward_input_and_parameter_gradients_match_pytorch(kv_heads):
    cfg = Config(d_model=16, n_heads=4, n_kv_heads=kv_heads, max_len=12, rope_base=700.0)
    attn = Attention(16, 4, kv_heads, 12, 700.0, np.random.default_rng(0), 0.02, np.float64)
    values = np.random.default_rng(0).normal(size=(2, 5, 16))
    assert_parity(attn, values, lambda x: attn(x, 0), lambda p, x: ref.attention(p, x, cfg))


def test_block_forward_input_and_parameter_gradients_match_pytorch():
    cfg = Config(d_model=16, n_heads=4, n_kv_heads=2, max_len=12, rope_base=700.0)
    block = Block(cfg, np.random.default_rng(0), np.float64)
    values = np.random.default_rng(0).normal(size=(2, 5, 16))
    assert_parity(block, values, lambda x: block(x, 0), lambda p, x: ref.block(p, x, cfg))


def test_named_parameters_follow_the_module_tree():
    block = Block(Config(d_model=16, n_heads=2, n_kv_heads=1), np.random.default_rng(0), np.float64)
    names = {"norm1.gamma", "attn.W_Q", "attn.W_K", "attn.W_V", "attn.W_O",
             "norm2.gamma", "ffn.W1", "ffn.W2", "ffn.W3"}
    assert set(block.named_parameters()) == names
    parent = Module()
    parent.blocks = [block]
    parent.frozen = Tensor(np.ones(3))
    assert set(parent.named_parameters("model.")) == {"model.blocks.0." + name for name in names}
    assert all(p.requires_grad for p in parent.parameters())


def test_gqa_key_and_value_projections_are_smaller():
    attn = Attention(16, 4, 2, 12, 10000.0, np.random.default_rng(0), 0.02, np.float64)
    assert attn.W_Q.shape == attn.W_O.shape == (16, 16)
    assert attn.W_K.shape == attn.W_V.shape == (16, 8)


def test_init_param_is_reproducible_and_uses_requested_dtype():
    actual = init_param(np.random.default_rng(2), (3, 4), 0.3, np.float32)
    expected = np.random.default_rng(2).normal(0.0, 0.3, (3, 4)).astype(np.float32)
    np.testing.assert_array_equal(actual.data, expected)
    assert actual.requires_grad
    assert actual.data.dtype == np.float32


def test_attention_cache_chunks_match_full_causal_forward():
    attn = Attention(16, 4, 2, 12, 700.0, np.random.default_rng(0), 0.3, np.float64)
    values = np.random.default_rng(1).normal(size=(2, 6, 16))
    cache = KVCache()
    with no_grad():
        full = attn(Tensor(values), 0).data
        first = attn(Tensor(values[:, :3]), 0, None, cache, 0).data
        second = attn(Tensor(values[:, 3:5]), 0, None, cache, 3).data
        third = attn(Tensor(values[:, 5:]), 0, None, cache, 5).data
    np.testing.assert_allclose(np.concatenate((first, second, third), axis=1), full, rtol=1e-8, atol=1e-10)
    assert cache.k.shape == cache.v.shape == (2, 2, 6, 4)


def test_block_trace_records_all_layer_steps_and_causal_weights():
    block = Block(Config(d_model=16, n_heads=4, n_kv_heads=2), np.random.default_rng(0), np.float64)
    trace = Tracer()
    out = block(Tensor(np.random.default_rng(1).normal(size=(1, 5, 16))), 2, trace)
    assert set(trace.step_ids()) == set(LAYER_STEPS)
    np.testing.assert_array_equal(trace.get("resid.ffn", layer=2).value, out.data[0])
    weights = trace.get("attn.softmax", layer=2, head=1).value
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0)
    assert np.all(np.triu(weights, k=1) == 0)
