---
title: "Tokenizer: complete reference"
description: "Character BPE, normalization, merges, IDs, decoding and persistence."
---

## Construction and vocabulary

Import `Tokenizer` from the root; `pre_tokenize`, `SPACE`, `SPECIALS` and `UNK_TEXT` come from `glassllm.tokenizer`. `Tokenizer(alphabet: list[str], merges: list[tuple[str,str]])` constructs a vocabulary containing special tokens, the alphabet and merge results, in that order. It copies the supplied lists, learns no merges and does not validate every possible manually supplied vocabulary combination.

| Attribute or constant | Contents |
| --- | --- |
| `SPECIALS` | `["<pad>", "<unk>", "<bos>", "<eos>"]`, IDs 0–3 |
| `SPACE` | `"▁"`, represents an ASCII space |
| `UNK_TEXT` | `"⁇"`, decoded text for `<unk>` |
| `alphabet`, `merges` | Characters and learned pairs, in order |
| `ranks` | Pair → merge priority dictionary |
| `vocab` | Complete list of token strings indexed by ID |
| `token_to_id`, `unk_id` | Reverse dictionary and unknown ID, normally 1 |
| `vocab_size` | Integer property, `len(vocab)` |

Use `tok.vocab_size` without parentheses. Read token strings with `tok.tokens(ids)` or `tok.vocab[id]`; there is no `id_to_token` method.

## Normalization and learning

`pre_tokenize(text: str) -> list[str]` normalizes Unicode to NFC, replaces each ASCII space with `▁` and splits chunks. Newlines are separate chunks; spaces before a word belong to its chunk. Merges never cross chunk boundaries. Other whitespace, such as tabs, is not universally converted to `▁`. A literal `▁` is treated as the marker, so arbitrary Unicode strings are not fully reversible.

`Tokenizer.train(text: str, vocab_size: int) -> Tokenizer` counts chunks, starts with the observed alphabet and repeatedly chooses the most frequent adjacent pair. Pair ties are resolved alphabetically; each merge is applied without overlap. The requested size must accommodate four specials plus the alphabet, or `ValueError` is raised. It is an upper bound: a small corpus may run out of pairs first.

```python
from glassllm import Tokenizer
from glassllm.tokenizer import pre_tokenize

assert pre_tokenize("Olá mundo\n") == ["Olá", "▁mundo", "\n"]
tok = Tokenizer.train("banana banana\n", vocab_size=32)
ids = tok.encode("banana banana\n")
assert tok.decode(ids) == "banana banana\n"
assert tok.vocab_size <= 32
assert len(tok.tokens(ids)) == len(ids)
assert tok.encode("🧪") == [tok.unk_id]
assert tok.decode([tok.unk_id]) == "⁇"
```

## Encode and explain merges

`encode(text: str) -> list[int]` processes each chunk, applying the available pair with the lowest rank first. A piece absent from the vocabulary becomes `unk_id`. It adds no padding, BOS or EOS and imposes no length limit. Empty text produces `[]`, which is not a valid prompt for `GlassLM.generate`.

`encode_verbose(text: str) -> list[dict]` returns one entry per chunk: `{"chunk": str, "steps": list[list[str]]}`. The first state contains characters; later states contain pieces after each applied merge. These are neither IDs nor probabilities. Even a chunk without merges has an initial state. This method records states for explanation even when ordinary encoding could use the internal cache.

```python
from glassllm import Tokenizer

tok = Tokenizer.train("banana banana\n", 32)
detail = tok.encode_verbose("banana")
assert detail[0]["chunk"] == "banana"
assert detail[0]["steps"][0] == list("banana")
assert detail[0]["steps"][-1] == tok.tokens(tok.encode("banana"))
```

## Tokens and text

`tokens(ids: list[int]) -> list[str]` uses Python list indexing into `vocab`. IDs above the range raise `IndexError`; negative indices follow list semantics rather than being rejected, so validate external IDs. `decode(ids: list[int]) -> str` joins pieces, converts `▁` to spaces, renders `<unk>` as `⁇` and omits the other three specials. It cannot recover text lost to normalization or unknown characters.

## Save and load

`save(path: str | Path) -> None` writes UTF-8 JSON with `version: 1`, `alphabet` and `merges`. It overwrites the destination and does not create parent directories. `Tokenizer.load(path: str | Path) -> Tokenizer` reads the lists and reconstructs ranks and vocabulary. The current reader neither negotiates versions nor validates a full schema: missing keys raise `KeyError`, invalid JSON raises a reader error and filesystem failures raise `OSError`.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from glassllm import Tokenizer

tok = Tokenizer.train("banana banana\n", 32)
with TemporaryDirectory() as directory:
    path = Path(directory) / "tokenizer.json"
    tok.save(path)
    restored = Tokenizer.load(path)
    assert restored.encode("banana") == tok.encode("banana")
    assert restored.vocab == tok.vocab
```

## Relationship to the papers

The project uses the subword BPE merging idea described by [Sennrich, Haddow and Birch](https://arxiv.org/abs/1508.07909). Its implementation starts with **Unicode characters**, not bytes, and provides no byte fallback. The visible space marker resembles [SentencePiece](https://arxiv.org/abs/1808.06226), but this is a custom tokenizer: it neither loads SentencePiece models nor implements all its algorithms. Follow the calculations in [Tokens and BPE](../../fundamentos/tokens-bpe/).
