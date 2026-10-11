"""Train one language with glassllm, restore its best checkpoint, and validate fp16 export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from glassllm import Config, GlassLM, LANGUAGES, Tokenizer
from glassllm.train import evaluate, train


MAX_SECONDS = 2 * 60 * 60
EVAL_SEED = 1234
EVAL_N_BATCHES = 10


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", required=True, choices=LANGUAGES)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="src/glassllm/weights")
    args = parser.parse_args(argv)
    deadline = time.monotonic() + MAX_SECONDS

    def check_deadline() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("pretrained model run exceeded the two-hour deadline")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.load(out / f"{args.lang}.tokenizer.json")

    def read(split: str) -> np.ndarray:
        text = (Path(args.data) / f"{args.lang}_{split}.txt").read_text(encoding="utf-8")
        return np.array(tok.encode(text), dtype=np.int64)

    train_ids, val_ids = read("train"), read("val")
    cfg = Config(vocab_size=tok.vocab_size)
    model = GlassLM(cfg, seed=0)
    check_deadline()
    train(model, train_ids, steps=args.steps, batch_size=args.batch_size, seq_len=cfg.max_len,
          lr=args.lr, warmup=100, seed=0, val_data=val_ids, eval_every=100, keep_best=True)

    # Measure the restored checkpoint and its fp16 export on identical windows.
    settings = {"batch_size": args.batch_size, "seq_len": cfg.max_len,
                "n_batches": EVAL_N_BATCHES, "seed": EVAL_SEED}
    check_deadline()
    fp32_loss = evaluate(model, val_ids, **settings)
    check_deadline()
    path = out / f"{args.lang}.npz"
    model.save(path, fp16=True)
    check_deadline()
    fp16_loss = evaluate(GlassLM.load(path), val_ids, **settings)
    check_deadline()
    if not np.isfinite(fp32_loss) or not np.isfinite(fp16_loss) or abs(fp16_loss - fp32_loss) > 0.01:
        raise ValueError(f"fp16 validation loss {fp16_loss} differs from fp32 {fp32_loss} by more than 0.01")
    meta = {"steps": args.steps, "val_loss_fp32": fp32_loss, "val_loss": fp16_loss,
            "train_tokens": len(train_ids), "val_tokens": len(val_ids),
            "eval_batch_size": args.batch_size, "eval_seq_len": cfg.max_len,
            "eval_n_batches": EVAL_N_BATCHES, "eval_seed": EVAL_SEED}
    (out / f"{args.lang}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"saved {path}: val loss {fp16_loss:.3f} (fp16; fp32 {fp32_loss:.3f})")


if __name__ == "__main__":
    main()
