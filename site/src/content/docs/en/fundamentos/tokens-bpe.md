---
title: "Tokens and BPE"
description: "How the tokenizer converts text to IDs, learns merges and preserves spaces and newlines."
---

## From text to IDs

The Transformer consumes integers. Each ID identifies a vocabulary entry: a letter, punctuation mark, subword or word learned through merges. A token does not necessarily correspond to a word or a byte.

`glassllm` implements **BPE over Unicode characters** in `src/glassllm/tokenizer.py`. It does not use the SentencePiece library and has no byte fallback mechanism. The Portuguese and English models have separate vocabularies of 1,024 entries. An ID can represent different text in each language.

```text
Text → NFC normalization → chunks → BPE merges → pieces → IDs
IDs → pieces → remove special tokens → replace ▁ with spaces → text
```

## Normalization and boundaries

`pre_tokenize(text)` first normalizes Unicode to NFC, treating a letter plus a combining accent as its equivalent composed form. It then replaces ordinary spaces with `▁` and separates chunks using a regular expression. Spaces preceding a word may belong to that word's chunk; a newline forms its own chunk. Merges never cross these boundaries.

For example, `pre_tokenize("Era uma vez\n")` produces `['Era', '▁uma', '▁vez', '\n']`. The `▁` marker makes a space visible in the token panel. It does not add a letter to the decoded text.

| Special token | Fixed ID | Decoding |
| --- | --- | --- |
| `<pad>` | 0 | Omitted |
| `<unk>` | 1 | `⁇` |
| `<bos>` | 2 | Omitted |
| `<eos>` | 3 | Omitted |

`encode` does not automatically insert BOS or EOS. Unknown characters become `<unk>`, so decoding cannot recover characters missing from the alphabet. NFC also means the original Unicode encoding may not be preserved byte for byte.

## How merges are learned

`Tokenizer.train(text, vocab_size)` counts chunks and adjacent pairs. Each round selects the most frequent pair, replaces non-overlapping occurrences and records the merge. Frequency ties are resolved by alphabetical pair order, making training deterministic for the same text.

The initial vocabulary contains the four special tokens and the observed alphabet. Each merge adds an entry. Training stops before the requested size when no pairs remain. Requesting fewer entries than the alphabet plus specials raises `ValueError`.

During encoding, the **learned** merge order determines the next pair to combine. Looking for the longest vocabulary piece is insufficient: `ranks` retains each merge's priority.

## Reproducible example

This small corpus lets you inspect every stage:

```python
from glassllm import Tokenizer

t = Tokenizer.train("banana banana", vocab_size=12)
print(t.merges)
print(t.encode("banana banana"))
print(t.encode_verbose("banana")[0]["steps"])
print(t.decode(t.encode("banana banana")))
```

Verified output:

```text
[('a', 'n'), ('an', 'a'), ('an', 'ana'), ('b', 'anana')]
[11, 7, 11]
[['b', 'a', 'n', 'a', 'n', 'a'],
 ['b', 'an', 'an', 'a'], ['b', 'an', 'ana'],
 ['b', 'anana'], ['banana']]
banana banana
```

The second chunk's space stays separate because no `('▁', 'banana')` merge was learned. With this tokenizer, `t.encode("🙂")` returns `[1]`.

## Connection to the interface and engine

With the project's packaged Portuguese tokenizer, `"Era uma vez"` maps to `[35, 120, 219, 373]`, with pieces `['E', 'ra', '▁uma', '▁vez']`. Use `tokens(ids)` to inspect pieces and `decode(ids)` to read text. This API has no `id_to_token` method.

The token panel displays these IDs. The next stage, [`embed`](../embeddings/), uses each ID to look up a row in `E`. `encode_verbose` exposes merge stages; these are not `Tracer` snapshots, which begin inside the model.

## References and limitations

Sennrich, Haddow and Birch discuss BPE subword units in [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909), first preprint in 2015. Kudo and Richardson's 2018 [SentencePiece paper](https://arxiv.org/abs/1808.06226) provides context for explicit space representation. This implementation has its own chunk, normalization and merge rules. The papers' results are not evaluations of this small model.
