"""Train a language's BPE tokenizer and save it next to the model weights."""
from __future__ import annotations

import argparse
from pathlib import Path

from glassllm import LANGUAGES, Tokenizer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", required=True, choices=LANGUAGES)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="src/glassllm/weights")
    args = parser.parse_args(argv)
    text = (Path(args.data) / f"{args.lang}_train.txt").read_text(encoding="utf-8")
    tok = Tokenizer.train(text, args.vocab_size)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.lang}.tokenizer.json"
    tok.save(path)
    ids = tok.encode(text)
    ratio = len(text) / len(ids) if ids else 0.0
    print(f"{args.lang}: vocab {tok.vocab_size}, {len(ids):,} tokens, "
          f"{ratio:.2f} characters per token -> {path}")


if __name__ == "__main__":
    main()
