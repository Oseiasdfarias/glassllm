from __future__ import annotations

import numpy as np
import pytest

from glassllm import load_pretrained
from glassllm.cli import main
from glassllm.config import Config
from glassllm.model import GlassLM
from glassllm.tokenizer import Tokenizer

TEXT = "a atenção compara uma pergunta com várias chaves e mistura os valores.\n" * 40


@pytest.fixture
def model_dir(tmp_path):
    tok = Tokenizer.train(TEXT, vocab_size=60)
    tok.save(tmp_path / "en.tokenizer.json")
    GlassLM(Config(vocab_size=tok.vocab_size, d_model=32, n_layers=1, max_len=32), seed=0).save(
        tmp_path / "en.npz"
    )
    return tmp_path


def run(capsys, model_dir, *args):
    main(["--model-dir", str(model_dir), *args])
    return capsys.readouterr().out


def test_tokenize_accepts_lang_after_subcommand_and_lists_merges(capsys, model_dir):
    out = run(capsys, model_dir, "tokenize", "--lang", "en", "a atenção", "--merges")
    assert "▁" in out and "→" in out


def test_generate_accepts_n_alias_and_prints_prompt(capsys, model_dir):
    out = run(capsys, model_dir, "generate", "a atenção", "--n", "2", "--seed", "0")
    assert out.startswith("a atenção")


def test_steps_runs_without_pretrained_assets(capsys, tmp_path):
    main(["--model-dir", str(tmp_path / "missing"), "steps"])
    assert "attn.softmax" in capsys.readouterr().out


def test_trace_empty_selector_result_is_a_single_cli_error(model_dir, capsys):
    with pytest.raises(SystemExit, match="error:") as error:
        main(["--model-dir", str(model_dir), "trace", "a", "--step", "attn.softmax", "--layer", "99"])
    assert "Traceback" not in str(error.value)
    assert capsys.readouterr().err.strip().count("\n") == 0


def test_trace_rejects_head_on_non_head_step(model_dir):
    with pytest.raises(SystemExit, match="error:"):
        main(["--model-dir", str(model_dir), "trace", "a", "--step", "embed", "--head", "99"])


def test_missing_model_is_a_single_cli_error(tmp_path, capsys):
    with pytest.raises(SystemExit, match="error:"):
        main(["--model-dir", str(tmp_path), "generate", "hello"])
    assert capsys.readouterr().err.strip().count("\n") == 0


def test_empty_prompt_and_bad_temperature_are_cli_errors(model_dir):
    with pytest.raises(SystemExit, match="empty"):
        main(["--model-dir", str(model_dir), "generate", ""])
    with pytest.raises(SystemExit, match="temperature"):
        main(["--model-dir", str(model_dir), "generate", "a", "--temperature", "0"])


def test_unknown_characters_still_generate(capsys, model_dir):
    assert run(capsys, model_dir, "generate", "a 🙂 atenção", "--n", "1", "--seed", "0").startswith(
        "a 🙂 atenção"
    )


@pytest.mark.parametrize("step", ["attn.q", "attn.k", "attn.v", "attn.mix"])
def test_attention_feature_columns_are_not_mislabeled_as_tokens(model_dir, capsys, step):
    # Four heads in d_model=32 gives d_k=8. Eight single-character tokens
    # make the displayed Q matrix ambiguous by shape alone.
    main(["--model-dir", str(model_dir), "trace", "🙂🙂🙂🙂🙂🙂🙂🙂", "--step", step])
    out = capsys.readouterr().out
    assert "f0" in out and "f7" in out
    assert "shape (8, 8)" in out
    header = out.splitlines()[4]
    assert "f0" in header and "⁇" not in header


def test_trace_scores_columns_are_tokens(model_dir, capsys):
    text = "a b c"
    main(["--model-dir", str(model_dir), "trace", text, "--step", "attn.scores", "--head", "0"])
    out = capsys.readouterr().out
    _, tok = load_pretrained("en", model_dir)
    assert all(token in out.splitlines()[4] for token in tok.tokens(tok.encode(text)))


def test_trace_rejects_out_of_range_head_as_a_cli_error(model_dir):
    with pytest.raises(SystemExit, match="error:.*head"):
        main(["--model-dir", str(model_dir), "trace", "a b", "--step", "attn.q", "--head", "99"])


def test_train_uses_prepared_language_corpus_and_saves_smoke_model(tmp_path, model_dir, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "en_train.txt").write_text(TEXT, encoding="utf-8")
    out = run(capsys, model_dir, "train", "--lang", "en", "--steps", "1", "--seq-len", "8",
              "--batch-size", "1", "--d-model", "32", "--layers", "1", "--log-every", "1",
              "--out", str(tmp_path / "trained.npz"))
    assert "step" in out
    assert GlassLM.load(tmp_path / "trained.npz").cfg.d_model == 32


def test_train_does_not_require_pretrained_weights(tmp_path, model_dir, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "en_train.txt").write_text(TEXT, encoding="utf-8")
    (model_dir / "en.npz").unlink()
    out = run(capsys, model_dir, "train", "--lang", "en", "--steps", "1", "--seq-len", "8",
              "--batch-size", "1", "--d-model", "32", "--layers", "1", "--log-every", "1",
              "--out", str(tmp_path / "trained.npz"))
    assert "step" in out
    assert GlassLM.load(tmp_path / "trained.npz").cfg.d_model == 32


def test_format_matrix_marks_negative_infinity():
    from glassllm.cli import format_matrix

    result = format_matrix(np.array([[0.5, -np.inf]]), ["a"], ["a", "b"])
    assert "-inf" in result
