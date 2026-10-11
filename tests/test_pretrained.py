"""Portable loader and reproducible asset tooling; fixtures are test-local only."""
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from glassllm import LANGUAGES, load_pretrained
from glassllm.config import Config
from glassllm.model import GlassLM
from glassllm.tokenizer import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
TEXT = "Alice and Capitu look at a little book.\n" * 100


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def model_dir(tmp_path):
    tok = Tokenizer.train(TEXT, 64)
    for lang in ("pt", "en"):
        tok.save(tmp_path / f"{lang}.tokenizer.json")
        GlassLM(Config(vocab_size=tok.vocab_size, d_model=16, n_layers=1,
                       n_heads=2, n_kv_heads=1, max_len=16), seed=0).save(tmp_path / f"{lang}.npz")
    return tmp_path


@pytest.mark.parametrize("lang", ("pt", "en"))
def test_loader_loads_local_weights_and_generates(lang, model_dir):
    from glassllm import GenStep

    assert lang in LANGUAGES
    model, tok = load_pretrained(lang, model_dir)
    assert tok.vocab_size == model.cfg.vocab_size
    steps = list(model.generate(tok.encode("Alice"), 10, seed=0))
    assert len(steps) == 10
    assert all(isinstance(step, GenStep) and 0 <= step.token < tok.vocab_size for step in steps)


def test_unknown_language_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="pt, en"):
        load_pretrained("fr", tmp_path)


@pytest.mark.parametrize("missing", ("en.npz", "en.tokenizer.json"))
def test_missing_asset_is_a_clear_error(model_dir, missing):
    (model_dir / missing).unlink()
    with pytest.raises(FileNotFoundError, match="pretrained.*en"):
        load_pretrained("en", model_dir)


def test_loader_rejects_a_tokenizer_with_a_different_vocabulary(model_dir):
    Tokenizer.train("xyz", 7).save(model_dir / "en.tokenizer.json")
    with pytest.raises(ValueError, match="vocab"):
        load_pretrained("en", model_dir)


def test_default_loader_supports_zip_package_resources(model_dir, tmp_path, monkeypatch):
    import glassllm

    archive = tmp_path / "package.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in model_dir.glob("en.*"):
            bundle.write(path, f"glassllm/weights/{path.name}")
    with zipfile.ZipFile(archive) as bundle:
        monkeypatch.setattr(glassllm.resources, "files", lambda package: zipfile.Path(bundle, "glassllm/"))
        model, tok = load_pretrained("en")
    assert len(list(model.generate(tok.encode("Alice"), 2, seed=0))) == 2


def test_clean_normalizes_and_removes_gutenberg_boilerplate():
    prepare = script("prepare_corpus")
    raw = ("header\r\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\r\n"
           "Cafe\u0301\t  book\r\n\r\n\r\nnext\r\n"
           "*** END OF THE PROJECT GUTENBERG EBOOK X ***\r\nfooter")
    assert prepare.clean(raw) == "Café book\n\nnext\n"


def test_clean_rejects_missing_gutenberg_markers():
    with pytest.raises(ValueError, match="Gutenberg"):
        script("prepare_corpus").clean("A book without delimiters")


def test_prepare_preserves_per_book_last_five_percent(tmp_path, monkeypatch):
    prepare = script("prepare_corpus")
    import io

    # External network only is replaced; cleaning, all eight books and writes are real.
    body = "line\n" * 100
    raw = ("*** START OF THE PROJECT GUTENBERG EBOOK X ***\n" + body +
           "*** END OF THE PROJECT GUTENBERG EBOOK X ***").encode()
    responses = {f"https://www.gutenberg.org/cache/epub/{book}/pg{book}.txt": raw
                 for book in (55752, 54829, 55682, 56737, 55797, 11, 12, 620)}
    monkeypatch.setattr(prepare.urllib.request, "urlopen",
                        lambda url, timeout: io.BytesIO(responses.pop(url)))
    out = tmp_path / "nested" / "data"
    prepare.main(["--out", str(out)])
    assert (out / "pt_train.txt").read_text() == "\n".join(["line\n" * 96] * 5)
    assert (out / "pt_val.txt").read_text() == "\n".join(["line\n" * 4] * 5)
    assert (out / "en_train.txt").read_text() == "\n".join(["line\n" * 96] * 3)
    assert (out / "en_val.txt").read_text() == "\n".join(["line\n" * 4] * 3)
    assert not responses


def test_tokenizer_script_creates_missing_nested_output_directory(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "en_train.txt").write_text(TEXT, encoding="utf-8")
    out = tmp_path / "missing" / "weights"
    script("train_tokenizer").main(["--lang", "en", "--data", str(data),
                                    "--out", str(out), "--vocab-size", "64"])
    assert Tokenizer.load(out / "en.tokenizer.json").decode(
        Tokenizer.load(out / "en.tokenizer.json").encode(TEXT)) == TEXT


def test_training_metadata_reproduces_restored_fp32_and_fp16_loss(tmp_path, monkeypatch):
    from glassllm.train import evaluate

    data, out = tmp_path / "data", tmp_path / "weights"
    data.mkdir()
    out.mkdir()
    tok = Tokenizer.train(TEXT, 64)
    tok.save(out / "en.tokenizer.json")
    for split in ("train", "val"):
        (data / f"en_{split}.txt").write_text(TEXT, encoding="utf-8")
    module = script("train_model")
    captured = []
    original_save = GlassLM.save

    def observe_save(model, path, fp16=True):
        # Keep a real fp32 copy to independently reproduce the metadata measurement.
        fp32 = tmp_path / "before_quantization.npz"
        original_save(model, fp32, fp16=False)
        captured.append(GlassLM.load(fp32))
        original_save(model, path, fp16=fp16)

    monkeypatch.setattr(GlassLM, "save", observe_save)
    module.main(["--lang", "en", "--steps", "1", "--batch-size", "1",
                 "--data", str(data), "--out", str(out)])
    meta = json.loads((out / "en.meta.json").read_text())
    val = np.array(tok.encode(TEXT))
    settings = (meta["eval_batch_size"], meta["eval_seq_len"])
    keywords = {"n_batches": meta["eval_n_batches"], "seed": meta["eval_seed"]}
    fp32_loss = evaluate(captured[0], val, *settings, **keywords)
    fp16_loss = evaluate(GlassLM.load(out / "en.npz"), val, *settings, **keywords)
    assert meta["steps"] == 1
    assert meta["train_tokens"] == meta["val_tokens"] == len(val)
    assert meta["val_loss_fp32"] == fp32_loss
    assert meta["val_loss"] == fp16_loss
    assert abs(fp16_loss - fp32_loss) <= 0.01
    assert captured[0].cfg == Config(vocab_size=tok.vocab_size)
