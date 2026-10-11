import pytest

from glassllm.tokenizer import SPACE, SPECIALS, UNK_TEXT, Tokenizer, pre_tokenize

TEXT = (
    "a atenção compara uma pergunta com várias chaves e mistura os valores.\n"
    "a rede aprende a prever o próximo token e a perda mede o erro.\n"
) * 30


def test_pre_tokenize_marks_spaces_and_keeps_newlines():
    assert pre_tokenize("ola mundo\nfim") == ["ola", f"{SPACE}mundo", "\n", "fim"]


def test_train_reaches_vocab_size_and_is_deterministic():
    a = Tokenizer.train(TEXT, vocab_size=80)
    b = Tokenizer.train(TEXT, vocab_size=80)
    assert a.vocab_size == 80
    assert a.merges == b.merges
    assert a.vocab[: len(SPECIALS)] == SPECIALS


def test_roundtrip():
    tok = Tokenizer.train(TEXT, vocab_size=120)
    for text in ["a atenção compara", "  dois  espaços ", "o erro.\nnova linha", ""]:
        assert tok.decode(tok.encode(text)) == text


def test_frequent_words_become_single_tokens():
    tok = Tokenizer.train(TEXT, vocab_size=200)
    assert tok.tokens(tok.encode(" pergunta")) == [f"{SPACE}pergunta"]


def test_merges_never_cross_a_space():
    tok = Tokenizer.train(TEXT, vocab_size=200)
    for token in tok.vocab:
        assert SPACE not in token[1:].lstrip(SPACE) or set(token) == {SPACE}


def test_unknown_characters_become_unk():
    tok = Tokenizer.train(TEXT, vocab_size=80)
    ids = tok.encode("a 🙂 a")
    assert tok.unk_id in ids
    assert tok.decode(ids) == f"a {UNK_TEXT} a"


def test_encode_verbose_shows_each_merge():
    tok = Tokenizer.train(TEXT, vocab_size=200)
    [entry] = tok.encode_verbose(" pergunta")
    assert entry["chunk"] == f"{SPACE}pergunta"
    assert entry["steps"][0] == list(f"{SPACE}pergunta")
    assert entry["steps"][-1] == [f"{SPACE}pergunta"]
    assert len(entry["steps"]) > 1
    assert all(len(after) < len(before) for before, after in zip(entry["steps"], entry["steps"][1:]))
    assert all("".join(step) == f"{SPACE}pergunta" for step in entry["steps"])


def test_save_and_load(tmp_path):
    tok = Tokenizer.train(TEXT, vocab_size=100)
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.vocab == tok.vocab
    assert loaded.merges == tok.merges
    assert loaded.encode(TEXT[:50]) == tok.encode(TEXT[:50])
    assert loaded.decode(loaded.encode("a atenção")) == "a atenção"


def test_decomposed_unicode_roundtrips_as_nfc():
    tok = Tokenizer.train("café atenção", vocab_size=40)
    assert pre_tokenize("cafe\u0301 atenc\u0327a\u0303o") == ["café", f"{SPACE}atenção"]
    assert tok.decode(tok.encode("cafe\u0301 atenc\u0327a\u0303o")) == "café atenção"
    assert tok.encode("cafe\u0301") == tok.encode("café")


def test_tiny_corpus_stops_when_no_pairs_remain():
    tok = Tokenizer.train("abab", vocab_size=100)
    assert tok.merges == [("a", "b"), ("ab", "ab")]
    assert tok.vocab_size == 8
    assert tok.tokens(tok.encode("abab")) == ["abab"]
    assert tok.decode(tok.encode("abab")) == "abab"


def test_pair_frequency_ties_are_broken_alphabetically():
    tok = Tokenizer.train("ba ab", vocab_size=8)
    assert tok.merges == [("a", "b")]


def test_vocab_size_must_cover_specials_and_alphabet():
    with pytest.raises(ValueError, match="smaller than specials \\+ alphabet"):
        Tokenizer.train("abc", vocab_size=6)


def test_empty_training_corpus_keeps_specials_and_unknown_fallback():
    tok = Tokenizer.train("", vocab_size=100)
    assert tok.vocab == SPECIALS
    assert tok.encode("") == []
    assert tok.decode(tok.encode("x")) == UNK_TEXT


def test_decode_skips_control_tokens():
    tok = Tokenizer.train("a", vocab_size=5)
    assert tok.decode([0, 2, 4, 1, 3]) == f"a{UNK_TEXT}"
