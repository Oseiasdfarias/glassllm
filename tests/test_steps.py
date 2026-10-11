import numpy as np
import pytest

from glassllm.trace import Tracer

from glassllm.autograd import Tensor
from glassllm.steps import STEPS, load_step_index, scan_markers
from glassllm.trace import record


def test_unknown_step_id_is_rejected():
    with pytest.raises(KeyError, match="steps.py"):
        Tracer().record("attn.typo", [1.0], batched=False)


def test_every_marker_is_in_the_catalog_and_every_catalog_entry_is_marked():
    assert set(scan_markers()) == set(STEPS)


def test_step_index_file_is_up_to_date():
    assert load_step_index() == scan_markers(), "run: uv run python scripts/build_step_index.py"


def test_forward_loss_and_generation_record_the_complete_catalog():
    from glassllm.model import Config, GlassLM

    cfg = Config(vocab_size=20, d_model=16, n_layers=2, n_heads=4, n_kv_heads=2, max_len=8)
    model = GlassLM(cfg, seed=0)
    trace = Tracer()
    model.loss([1, 2, 3], [2, 3, 4], trace=trace)
    step = next(model.generate([1, 2, 3], 1, trace=True))
    assert set(trace.step_ids()) | set(step.trace.step_ids()) == set(STEPS)


def test_catalog_covers_all_model_equations():
    expected = {
        "embed", "norm.attn", "attn.q", "attn.k", "attn.v", "attn.rope_q",
        "attn.rope_k", "attn.scores", "attn.mask", "attn.softmax", "attn.mix",
        "attn.out", "resid.attn", "norm.ffn", "ffn.gate", "ffn.up", "ffn.act",
        "ffn.down", "resid.ffn", "norm.final", "logits", "probs", "sample",
        "loss", "loss.grad",
    }
    assert set(STEPS) == expected
    for step_id in expected:
        tracer = Tracer()
        tracer.record(step_id, [1.0], layer=0 if step_id.startswith(("attn.", "ffn.", "resid.")) or step_id in {"norm.attn", "norm.ffn"} else None, batched=False)
        assert tracer.get(step_id).equation
        assert tracer.get(step_id).latex


def test_records_snapshot_batch_tensor_and_order():
    tracer = Tracer()
    source = Tensor(np.arange(12.0).reshape(1, 3, 4))
    tracer.record("embed", source)
    source.data[:] = -1
    record(tracer, "logits", np.ones((1, 3, 7)))
    tracer.record("embed", [[4.0]], batched=False)
    np.testing.assert_array_equal(tracer.get("embed").value, np.arange(12.0).reshape(3, 4))
    assert [rec.order for rec in tracer.records] == [0, 1, 2]
    assert [rec.shape for rec in tracer.records] == [(3, 4), (3, 7), (1, 1)]
    assert tracer.step_ids() == ["embed", "logits"]


def test_full_head_arrays_are_selected_by_layer_and_head():
    tracer = Tracer()
    tracer.record("attn.q", np.arange(24).reshape(1, 2, 3, 4), layer=0)
    tracer.record("attn.q", np.arange(24, 48).reshape(1, 2, 3, 4), layer=1)
    selected = tracer.get("attn.q", layer=1, head=1)
    np.testing.assert_array_equal(selected.value, np.arange(36, 48).reshape(3, 4))
    assert (selected.head, selected.layer, selected.order) == (1, 1, 1)
    assert tracer.get("attn.q", layer=1).shape == (2, 3, 4)


def test_explicit_head_records_are_not_sliced_again():
    tracer = Tracer()
    record(tracer, "attn.q", np.arange(6).reshape(1, 2, 3), 0, True, head=1)
    tracer.record("attn.q", np.full((2, 3), 7), layer=0, batched=False, head=0)
    selected = tracer.get("attn.q", layer=0, head=0)
    assert selected.shape == (2, 3)
    np.testing.assert_array_equal(selected.value, np.full((2, 3), 7))
    assert tracer.get("attn.q", layer=0, head=1).shape == (2, 3)
    assert tracer.get("attn.q", layer=0).head == 1
    with pytest.raises(KeyError, match="head=2"):
        tracer.get("attn.q", layer=0, head=2)


@pytest.mark.parametrize("selector", [-1, 1.5, True])
def test_invalid_layer_and_head_selectors(selector):
    tracer = Tracer()
    with pytest.raises(ValueError, match="layer"):
        tracer.record("attn.q", [[1]], layer=selector, batched=False)
    with pytest.raises(ValueError, match="head"):
        tracer.record("attn.q", [[1]], batched=False, head=selector)
    with pytest.raises(ValueError, match="layer"):
        tracer.get("attn.q", layer=selector)
    with pytest.raises(ValueError, match="head"):
        tracer.get("attn.q", head=selector)


def test_nonexistent_selectors_have_clear_errors():
    tracer = Tracer()
    tracer.record("attn.q", np.zeros((2, 3, 4)), layer=0, batched=False)
    tracer.record("embed", [[1]], batched=False)
    with pytest.raises(KeyError, match="layer=3"):
        tracer.get("attn.q", layer=3)
    with pytest.raises(ValueError, match="head"):
        tracer.get("attn.q", head=2)
    with pytest.raises(ValueError, match="no head axis"):
        tracer.get("embed", head=0)
    with pytest.raises(ValueError, match="no head axis"):
        tracer.record("embed", [[1]], batched=False, head=0)


def test_batched_record_requires_one_sequence():
    tracer = Tracer()
    for value in (np.zeros((2, 3)), np.zeros((0, 3)), np.array(3)):
        with pytest.raises(ValueError, match="batch"):
            tracer.record("embed", value)
    tracer.record("loss", 2.0, batched=False)
    assert tracer.get("loss").shape == ()


def test_optional_helper_is_noop_without_tracer():
    record(None, "unknown", object(), -1, True, head=-1)


def test_scanner_matches_comments_not_strings(tmp_path):
    (tmp_path / "a.py").write_text('"""# @step embed"""\ntext = "# @step logits"\nx = 1  # @step embed\n# @step logits\n', encoding="utf-8")
    assert scan_markers(tmp_path) == {
        "embed": {"file": "glassllm/a.py", "line": 3},
        "logits": {"file": "glassllm/a.py", "line": 4},
    }


def test_scanner_rejects_duplicate_markers(tmp_path):
    (tmp_path / "a.py").write_text("# @step embed\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("# @step embed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="marked twice"):
        scan_markers(tmp_path)


def test_index_script_writes_loadable_json(tmp_path, monkeypatch):
    import runpy
    from pathlib import Path
    import glassllm.steps as steps

    (tmp_path / "a.py").write_text("# @step embed\n", encoding="utf-8")
    output = tmp_path / "step_index.json"
    monkeypatch.setattr(steps, "INDEX_PATH", output)
    original_scan = steps.scan_markers
    monkeypatch.setattr(steps, "scan_markers", lambda: original_scan(tmp_path))
    runpy.run_path(str(Path(__file__).parents[1] / "scripts/build_step_index.py"), run_name="__main__")
    assert steps.load_step_index() == {"embed": {"file": "glassllm/a.py", "line": 1}}
