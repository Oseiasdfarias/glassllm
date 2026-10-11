import numpy as np
import pytest
import torch
import torch.nn.functional as F

from glassllm.model import Config, GlassLM
from glassllm.steps import LAYER_STEPS
from glassllm.trace import Tracer

from . import torch_reference as ref


CFG = Config(vocab_size=50, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_len=16)
IDS = np.array([[3, 17, 4, 4, 42, 9, 1]])
TARGETS = np.array([[17, 4, 4, 42, 9, 1, 0]])


@pytest.fixture
def model():
    model = GlassLM(CFG, seed=0, dtype=np.float64)
    rng = np.random.default_rng(1)
    for parameter in model.parameters():
        parameter.data = rng.normal(0, 0.2, parameter.shape)
    return model


def test_forward_matches_independent_pytorch_decoder(model):
    expected = ref.forward(ref.to_torch(model), IDS, CFG).detach().numpy()
    actual = model(IDS).data
    assert actual.shape == (1, len(IDS[0]), CFG.vocab_size)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)


def test_loss_and_all_parameter_gradients_match_pytorch(model):
    loss = model.loss(IDS, TARGETS)
    loss.backward()
    params = ref.to_torch(model)
    logits = ref.forward(params, IDS, CFG)
    expected_loss = F.cross_entropy(logits.reshape(-1, CFG.vocab_size),
                                    torch.as_tensor(TARGETS.ravel()))
    expected_loss.backward()
    assert float(loss.data) == pytest.approx(expected_loss.item(), rel=1e-8, abs=1e-10)
    for name, parameter in model.named_parameters().items():
        np.testing.assert_allclose(parameter.grad, params[name].grad.numpy(),
                                   rtol=1e-6, atol=1e-10, err_msg=name)


def test_tied_embedding_receives_lookup_and_output_gradients(model):
    model.loss(IDS, TARGETS).backward()
    params = ref.to_torch(model)
    lookup = params["E"].detach().clone().requires_grad_()
    output = params["E"].detach().clone().requires_grad_()
    logits = ref.forward(params, IDS, CFG, input_embedding=lookup, output_embedding=output)
    F.cross_entropy(logits.reshape(-1, CFG.vocab_size),
                    torch.as_tensor(TARGETS.ravel())).backward()
    assert np.linalg.norm(lookup.grad.numpy()) > 0
    assert np.linalg.norm(output.grad.numpy()) > 0
    np.testing.assert_allclose(model.E.grad, lookup.grad.numpy() + output.grad.numpy(),
                               rtol=1e-6, atol=1e-10)
    assert set(model.named_parameters()) == {"E", "norm_f.gamma"} | {
        f"blocks.{i}.{name}" for i in range(CFG.n_layers) for name in (
            "norm1.gamma", "attn.W_Q", "attn.W_K", "attn.W_V", "attn.W_O",
            "norm2.gamma", "ffn.W1", "ffn.W3", "ffn.W2")}


def test_parameter_count_uses_one_embedding_matrix(model):
    d, layers, v = CFG.d_model, CFG.n_layers, CFG.vocab_size
    attention = (2 + 2 * CFG.n_kv_heads / CFG.n_heads) * d * d
    ffn = 3 * d * int(8 * d / 3)
    norms = 2 * d
    assert sum(p.data.size for p in model.parameters()) == layers * (
        attention + ffn + norms) + v * d + d


def test_trace_records_global_and_every_layer_forward_value(model):
    tracer = Tracer()
    logits = model.forward(IDS, trace=tracer).data
    np.testing.assert_array_equal(tracer.get("embed").value, model.E.data[IDS[0]])
    normalized = tracer.get("norm.final").value
    np.testing.assert_allclose(normalized @ model.E.data.T, logits[0], rtol=1e-8, atol=1e-10)
    np.testing.assert_array_equal(tracer.get("logits").value, logits[0])
    for layer in range(CFG.n_layers):
        for step_id in LAYER_STEPS:
            assert tracer.get(step_id, layer=layer).value.size > 0
    weights = tracer.get("attn.softmax", layer=0, head=1).value
    assert weights.shape == (len(IDS[0]), len(IDS[0]))
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0)
    assert np.all(np.triu(weights, k=1) == 0)


def test_loss_trace_gradient_is_actual_logit_derivative(model):
    tracer = Tracer()
    actual_loss = model.loss(IDS, TARGETS, trace=tracer)
    assert tracer.get("loss").value == pytest.approx(float(actual_loss.data))
    params = ref.to_torch(model)
    logits = ref.forward(params, IDS, CFG)
    logits.retain_grad()
    F.cross_entropy(logits.reshape(-1, CFG.vocab_size),
                    torch.as_tensor(TARGETS.ravel())).backward()
    np.testing.assert_allclose(tracer.get("loss.grad").value, logits.grad.numpy()[0],
                               rtol=1e-8, atol=1e-10)


def test_cache_advances_same_logits_as_full_forward(model):
    cache = model.new_cache()
    assert len(cache) == CFG.n_layers
    first = model.forward(IDS[:, :3], cache=cache, start=0).data
    second = model.forward(IDS[:, 3:], cache=cache, start=3).data
    np.testing.assert_allclose(np.concatenate([first, second], axis=1),
                               model.forward(IDS).data, rtol=1e-8, atol=1e-10)


def test_trace_rejects_batches(model):
    with pytest.raises(ValueError, match="single sequence"):
        model.forward(np.zeros((2, 3), dtype=int), trace=Tracer())


def test_forward_rejects_positions_beyond_max_len(model):
    with pytest.raises(ValueError, match="max_len"):
        model.forward(np.zeros(CFG.max_len + 1, dtype=int))
    with pytest.raises(ValueError, match="max_len"):
        model.forward(np.zeros(3, dtype=int), start=CFG.max_len - 2)


def test_save_and_load_roundtrip(model, tmp_path):
    path = tmp_path / "m.npz"
    model.save(path, fp16=False)
    loaded = GlassLM.load(path, dtype=np.float64)
    assert loaded.cfg == CFG
    np.testing.assert_allclose(loaded.forward(IDS).data, model.forward(IDS).data,
                               rtol=1e-8, atol=1e-10)
