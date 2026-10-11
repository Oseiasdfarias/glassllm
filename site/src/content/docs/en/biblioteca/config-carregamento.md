---
title: "Configuration and pretrained models"
description: "Config fields, validation, PT/EN models and loading existing local assets."
---

## Config

Import: `from glassllm import Config`. This mutable dataclass has the following signature. It describes architecture; it contains no weights and performs no training.

```text
Config(vocab_size=1024, d_model=128, n_layers=4, n_heads=4,
       n_kv_heads=2, max_len=128, rope_base=10000.0,
       eps=1e-6, init_std=0.02)
```

| Field | Contract and effect |
| --- | --- |
| `vocab_size` | Positive integer; embedding rows and output logits |
| `d_model` | Positive integer; state width and residual dimension |
| `n_layers` | Positive integer; number of Transformer blocks |
| `n_heads` | Positive integer; number of Q heads |
| `n_kv_heads` | Positive integer; number of K/V heads |
| `max_len` | Positive integer; window and RoPE table length |
| `rope_base` | Positive; rotation frequency base |
| `eps` | Nonnegative; RMSNorm denominator stabilizer |
| `init_std` | Nonnegative; random weight standard deviation |

The six integer fields reject booleans, zero, negative values and objects that are not `int`. `d_model` must be divisible by `n_heads`; `n_heads` must be divisible by `n_kv_heads`; `d_model / n_heads` must be even for coordinate-pair rotation. Violations raise `ValueError` during construction.

Validation runs in `__post_init__`, not when attributes are subsequently changed. Treat an existing model's configuration as fixed: changing `cfg.d_model` does not resize matrices. Real-valued fields have no dedicated finite-value validation for inputs such as NaN; supply finite numbers and keep `eps` positive when handling zero vectors.

```python
from dataclasses import asdict
from glassllm import Config, GlassLM

cfg = Config(vocab_size=8, d_model=8, n_layers=1,
             n_heads=2, n_kv_heads=1, max_len=4)
model = GlassLM(cfg, seed=0)
assert asdict(cfg)["n_kv_heads"] == 1
assert model([1, 2]).shape == (1, 2, 8)
try:
    Config(d_model=127)
except ValueError:
    pass
else:
    raise AssertionError("invalid head dimensions must fail")
```

## load_pretrained

Import: `from glassllm import load_pretrained, LANGUAGES, __version__`.

`load_pretrained(lang: str, model_dir: str | Path | None = None) -> tuple[GlassLM, Tokenizer]` reads an existing file pair and returns a model with float32 parameters and its tokenizer. `LANGUAGES == ("pt", "en")`; `__version__ == "0.1.0"`. `lang` is case sensitive and accepts neither variants such as `pt-BR` nor automatic language detection.

With `model_dir=None`, the function uses `glassllm/weights/` package resources. An alternate directory must contain `{lang}.npz` and `{lang}.tokenizer.json`. `{lang}.meta.json` files describe training but are neither required nor returned. There is no download, account, remote API or automatic replacement with random weights when assets are missing.

```python
from glassllm import LANGUAGES, load_pretrained

assert LANGUAGES == ("pt", "en")
for lang in LANGUAGES:
    model, tokenizer = load_pretrained(lang)
    assert model.cfg.vocab_size == tokenizer.vocab_size == 1024
    assert sum(p.data.size for p in model.parameters()) == 852608
    step = next(model.generate(tokenizer.encode("Alice"), 1, greedy=True))
    assert step.probs.shape == (1024,)
```

The loader raises `ValueError` for an unknown language, `FileNotFoundError` for a missing file, and `ValueError` for mismatched vocabulary sizes. Malformed JSON or NPZ files can raise reader-specific exceptions. Equal vocabulary sizes do not verify identical token ordering: preserve the pair used during training.

## Included asset configuration

PT and EN use `Config()`: 4 blocks, dimension 128, 4 Q heads, 2 K/V heads and a 128-token window. Each head has 32 coordinates; SwiGLU uses `int(8 * 128 / 3) = 341` hidden units. The 852,608 parameter count includes `E` once because the output projection shares that matrix.

Loading trained weights does not change the model's limits: this remains a small educational text-continuation model, without a chat protocol or instruction-following behavior. To create another architecture, construct `GlassLM(Config(...), seed=...)`; transport its weights with [save/load](../glasslm/).
