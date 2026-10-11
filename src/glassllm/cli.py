"""Terminal interface for inspecting and training small glassllm models."""
from __future__ import annotations

import argparse
from importlib import resources
import sys
from pathlib import Path

import numpy as np

from . import load_pretrained
from .config import Config
from .model import GlassLM
from .steps import HEAD_STEPS, LAYER_STEPS, STEPS
from .tokenizer import Tokenizer
from .trace import Tracer
from .train import train

MAX_COLS = 8


def _label(token: str) -> str:
    return token.replace("\n", "\\n")


def format_matrix(values: np.ndarray, row_labels: list[str], col_labels: list[str] | None = None) -> str:
    """Format a 2-D value matrix with labeled rows and a bounded number of columns."""
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("matrix display requires a two-dimensional value")
    if len(row_labels) != values.shape[0]:
        raise ValueError("row label count must match matrix rows")
    cols = values.shape[1]
    if col_labels is None:
        col_labels = [str(j) for j in range(cols)]
    if len(col_labels) != cols:
        raise ValueError("column label count must match matrix columns")
    shown = min(cols, MAX_COLS)
    width = max(8, *(len(c) for c in col_labels[:shown]))
    left = max((len(label) for label in row_labels), default=0)
    header = " " * left + " " + " ".join(c[:width].rjust(width) for c in col_labels[:shown])
    lines = [header + ("  …" if cols > shown else "")]
    for label, row in zip(row_labels, values):
        cells = " ".join(f"{v:{width}.3f}" if np.isfinite(v) else "-inf".rjust(width) for v in row[:shown])
        lines.append(label.rjust(left) + " " + cells + ("  …" if cols > shown else ""))
    return "\n".join(lines)


def _matrix_labels(step_id: str, tokens: list[str], shape: tuple[int, ...]) -> tuple[list[str], list[str]]:
    """Choose axes from the semantic step, never by comparing coincidental dimensions."""
    if step_id in {"attn.scores", "attn.mask", "attn.softmax"}:
        return tokens[:shape[0]], tokens[:shape[1]]
    if step_id in {"attn.q", "attn.k", "attn.v", "attn.rope_q", "attn.rope_k", "attn.mix"}:
        return tokens[:shape[0]], [f"f{i}" for i in range(shape[1])]
    if step_id.startswith("attn."):
        return tokens[:shape[0]], [f"f{i}" for i in range(shape[1])]
    if step_id in {"embed", "norm.attn", "attn.out", "resid.attn", "norm.ffn", "ffn.gate", "ffn.up",
                   "ffn.act", "ffn.down", "resid.ffn", "norm.final"}:
        return tokens[:shape[0]], [f"f{i}" for i in range(shape[1])]
    return [str(i) for i in range(shape[0])], [str(i) for i in range(shape[1])]


def cmd_tokenize(args, model: GlassLM | None, tok: Tokenizer) -> None:
    ids = tok.encode(args.text)
    for token, token_id in zip(tok.tokens(ids), ids):
        print(f"{token_id:5d}  {_label(token)}")
    if args.merges:
        print()
        for entry in tok.encode_verbose(args.text):
            print(" → ".join(" ".join(_label(part) for part in state) for state in entry["steps"]))


def cmd_generate(args, model: GlassLM, tok: Tokenizer) -> None:
    ids = tok.encode(args.text)
    print(args.text, end="", flush=True)
    for step in model.generate(ids, args.n, temperature=args.temperature, greedy=args.greedy, seed=args.seed):
        print(tok.decode([step.token]), end="", flush=True)
    print()


def cmd_trace(args, model: GlassLM, tok: Tokenizer) -> None:
    ids = tok.encode(args.text)
    labels = [_label(token) for token in tok.tokens(ids)]
    step_id = args.step
    tracer = Tracer()
    if step_id in {"probs", "sample"}:
        tracer = next(model.generate(ids, 1, greedy=True, trace=True)).trace
    elif step_id in {"loss", "loss.grad"}:
        if len(ids) < 2:
            raise ValueError("loss trace needs at least two tokens")
        model.loss(np.array(ids[:-1]), np.array(ids[1:]), trace=tracer)
        labels = labels[:-1]
    else:
        model.forward(ids, trace=tracer)

    if args.layer is not None and step_id not in LAYER_STEPS:
        raise ValueError(f"step {step_id!r} has no layer selector")
    if args.head is not None and step_id not in HEAD_STEPS:
        raise ValueError(f"step {step_id!r} has no head selector")
    layer = (args.layer if args.layer is not None else 0) if step_id in LAYER_STEPS else None
    head = (args.head if args.head is not None else 0) if step_id in HEAD_STEPS else None
    rec = tracer.get(step_id, layer=layer, head=head)
    selectors = (f"  layer {layer}" if layer is not None else "") + (f"  head {head}" if head is not None else "")
    print(f"{step_id}{selectors}")
    print(f"  {rec.equation}")
    print(f"  shape {rec.shape}\n")

    value = rec.value
    if step_id == "probs":
        for index in np.argsort(value)[::-1][:10]:
            print(f"{_label(tok.vocab[index]):>14}  {value[index]:.3f}")
    elif step_id == "sample":
        print(f"next token: {_label(tok.vocab[int(value)])!r} (id {int(value)})")
    elif value.ndim == 0:
        print(f"{float(value):.4f}")
    elif value.ndim == 2:
        rows, cols = _matrix_labels(step_id, labels, value.shape)
        print(format_matrix(value, rows, cols))


def cmd_steps(args, model, tok) -> None:
    for step_id, info in STEPS.items():
        print(f"{step_id:14} {info.plain}")


def cmd_train(args, model: GlassLM, tok: Tokenizer) -> None:
    corpus_path = Path(args.corpus) if args.corpus else Path("data") / f"{args.lang}_train.txt"
    try:
        text = corpus_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"prepared corpus {corpus_path} is unavailable; run scripts/prepare_corpus.py first"
        ) from exc
    data = np.asarray(tok.encode(text), dtype=np.int64)
    cfg = Config(vocab_size=tok.vocab_size, d_model=args.d_model, n_layers=args.layers,
                 n_heads=4, n_kv_heads=2, max_len=args.seq_len)
    fresh = GlassLM(cfg, seed=args.seed)
    train(fresh, data, steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len,
          lr=args.lr, warmup=min(100, args.steps // 10 + 1), seed=args.seed, eval_every=args.log_every)
    if args.out:
        fresh.save(args.out)
        print(f"saved {args.out}")


def _add_lang_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lang", choices=["pt", "en"], default=argparse.SUPPRESS,
                        help="which language model to use (default: en)")
    parser.add_argument("--model-dir", default=argparse.SUPPRESS, help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glassllm", description="A small Transformer built to be looked at.")
    parser.add_argument("--lang", default="en", choices=["pt", "en"], help="which language model to use")
    parser.add_argument("--model-dir", default=None, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tokenize", help="split text into BPE tokens")
    _add_lang_option(p)
    p.add_argument("text")
    p.add_argument("--merges", action="store_true", help="show every merge, from characters to tokens")
    p.set_defaults(func=cmd_tokenize)

    p = sub.add_parser("generate", help="continue a text, one token at a time")
    _add_lang_option(p)
    p.add_argument("text")
    p.add_argument("-n", "--n", type=int, default=40, help="number of new tokens")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--greedy", action="store_true", help="always take the most likely token")
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("trace", help="print the equation and values of one step")
    _add_lang_option(p)
    p.add_argument("text")
    p.add_argument("--step", default="attn.softmax", choices=list(STEPS))
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--head", type=int, default=None)
    p.set_defaults(func=cmd_trace)

    p = sub.add_parser("steps", help="list every traceable step and its equation")
    p.set_defaults(func=cmd_steps)

    p = sub.add_parser("train", help="train from scratch on a prepared language corpus")
    _add_lang_option(p)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--out", default=None, help="where to save the weights (.npz)")
    p.add_argument("--corpus", default=None,
                   help="development-only local corpus override (default: data/{lang}_train.txt)")
    p.set_defaults(func=cmd_train)
    return parser


def _load_tokenizer(lang: str, model_dir: str | None) -> Tokenizer:
    """Load the prepared tokenizer without requiring pretrained model weights."""
    if model_dir is not None:
        path = Path(model_dir) / f"{lang}.tokenizer.json"
        return Tokenizer.load(path)
    path = resources.files("glassllm") / "weights" / f"{lang}.tokenizer.json"
    with resources.as_file(path) as tokenizer_path:
        return Tokenizer.load(tokenizer_path)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "steps":
        args.func(args, None, None)
        return
    try:
        if args.command == "train":
            tok = _load_tokenizer(args.lang, args.model_dir)
            args.func(args, None, tok)
        else:
            model, tok = load_pretrained(args.lang, args.model_dir)
            args.func(args, model, tok)
    except (ValueError, FileNotFoundError, KeyError, IndexError, OSError) as exc:
        message = str(exc).splitlines()[0]
        raise SystemExit(f"error: {message}") from None


if __name__ == "__main__":
    main(sys.argv[1:])
