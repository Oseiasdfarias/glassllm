"""Character byte-pair encoding with NFC text and explicit space markers.

Chunks preserve newlines and word boundaries. Training merges the most frequent
adjacent pair, breaking frequency ties alphabetically, until the requested
vocabulary size is reached or no pairs remain.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

SPACE = "▁"
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
UNK_TEXT = "⁇"
_CHUNK = re.compile(r"\n|▁*[^▁\n]+|▁+")


def pre_tokenize(text: str) -> list[str]:
    """Normalize to NFC and split into chunks that merges cannot cross."""
    text = unicodedata.normalize("NFC", text).replace(" ", SPACE)
    return _CHUNK.findall(text)


def _apply_merge(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    out = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


class Tokenizer:
    def __init__(self, alphabet: list[str], merges: list[tuple[str, str]]):
        self.alphabet = list(alphabet)
        self.merges = [tuple(pair) for pair in merges]
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.vocab = SPECIALS + self.alphabet + [a + b for a, b in self.merges]
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.unk_id = self.token_to_id["<unk>"]
        self._cache: dict[str, list[str]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(cls, text: str, vocab_size: int) -> Tokenizer:
        """Learn up to ``vocab_size`` tokens; small corpora may exhaust pairs."""
        words = Counter(tuple(chunk) for chunk in pre_tokenize(text))
        alphabet = sorted({character for word in words for character in word})
        minimum = len(SPECIALS) + len(alphabet)
        if vocab_size < minimum:
            raise ValueError(
                f"vocab_size={vocab_size} is smaller than specials + alphabet ({minimum})"
            )
        merges = []
        for _ in range(vocab_size - minimum):
            pairs = Counter()
            for word, frequency in words.items():
                for pair in zip(word, word[1:]):
                    pairs[pair] += frequency
            if not pairs:
                break
            best = min(pairs, key=lambda pair: (-pairs[pair], pair))
            merges.append(best)
            merged_words = Counter()
            for word, frequency in words.items():
                merged_words[_apply_merge(word, best)] += frequency
            words = merged_words
        return cls(alphabet, merges)

    def _merge_chunk(
        self, chunk: str, steps: list[list[str]] | None = None
    ) -> list[str]:
        if steps is None and chunk in self._cache:
            return self._cache[chunk]
        symbols = tuple(chunk)
        if steps is not None:
            steps.append(list(symbols))
        while len(symbols) > 1:
            ranked = [
                (self.ranks[pair], pair)
                for pair in zip(symbols, symbols[1:])
                if pair in self.ranks
            ]
            if not ranked:
                break
            symbols = _apply_merge(symbols, min(ranked)[1])
            if steps is not None:
                steps.append(list(symbols))
        result = list(symbols)
        self._cache[chunk] = result
        return result

    def encode(self, text: str) -> list[int]:
        return [
            self.token_to_id.get(piece, self.unk_id)
            for chunk in pre_tokenize(text)
            for piece in self._merge_chunk(chunk)
        ]

    def encode_verbose(self, text: str) -> list[dict]:
        """Return each chunk's successive states, including its characters."""
        out = []
        for chunk in pre_tokenize(text):
            steps: list[list[str]] = []
            self._merge_chunk(chunk, steps)
            out.append({"chunk": chunk, "steps": steps})
        return out

    def tokens(self, ids: list[int]) -> list[str]:
        return [self.vocab[i] for i in ids]

    def decode(self, ids: list[int]) -> str:
        pieces = []
        for token in self.tokens(ids):
            if token == "<unk>":
                pieces.append(UNK_TEXT)
            elif token not in SPECIALS:
                pieces.append(token)
        return "".join(pieces).replace(SPACE, " ")

    def save(self, path: str | Path) -> None:
        data = {
            "version": 1,
            "alphabet": self.alphabet,
            "merges": [list(pair) for pair in self.merges],
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["alphabet"], data["merges"])
