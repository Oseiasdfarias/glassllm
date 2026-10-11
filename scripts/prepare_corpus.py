"""Download eight public domain books and split each book's final 5% for validation."""
from __future__ import annotations

import argparse
import re
import unicodedata
import urllib.request
from pathlib import Path


BOOKS = {
    "pt": {  # Machado de Assis
        55752: "Dom Casmurro",
        54829: "Memorias Posthumas de Braz Cubas",
        55682: "Quincas Borba",
        56737: "Esau e Jacob",
        55797: "Memorial de Ayres",
    },
    "en": {  # Lewis Carroll
        11: "Alice's Adventures in Wonderland",
        12: "Through the Looking-Glass",
        620: "Sylvie and Bruno",
    },
}
URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
START = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)
END = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK")


def clean(raw: str) -> str:
    """Remove Gutenberg delimiters and normalize whitespace and Unicode to NFC."""
    text = raw.replace("\r\n", "\n")
    start, end = START.search(text), END.search(text)
    if start is None or end is None or start.end() >= end.start():
        raise ValueError("missing or invalid Project Gutenberg header/footer markers")
    text = unicodedata.normalize("NFC", text[start.end():end.start()])
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for lang, books in BOOKS.items():
        train_parts, val_parts = [], []
        for book_id, title in books.items():
            with urllib.request.urlopen(URL.format(id=book_id), timeout=30) as response:
                text = clean(response.read().decode("utf-8-sig"))
            cut = text.index("\n", int(len(text) * 0.95)) + 1
            train_parts.append(text[:cut])
            val_parts.append(text[cut:])
            print(f"{lang}: {title}: {len(text):,} characters")
        (out / f"{lang}_train.txt").write_text("\n".join(train_parts), encoding="utf-8")
        (out / f"{lang}_val.txt").write_text("\n".join(val_parts), encoding="utf-8")


if __name__ == "__main__":
    main()
