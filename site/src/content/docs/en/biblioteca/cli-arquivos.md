---
title: "CLI, files and scripts"
description: "Terminal commands, asset formats, corpus preparation and local training."
---

## Run the CLI

Local installation registers `glassllm = glassllm.cli:main`. In a prepared checkout environment, commands also run through `uv run python -m glassllm.cli`. `--lang` accepts `pt` or `en`, defaulting to `en`; it may precede or follow a subcommand offering this option. `--model-dir` selects local assets named `{lang}.npz` and `{lang}.tokenizer.json`.

```bash
uv run glassllm --lang pt tokenize "Era uma vez" --merges
uv run glassllm generate "Alice" --lang en -n 3 --greedy --seed 0
uv run glassllm trace "Alice was" --lang en --step attn.softmax --layer 0 --head 0
uv run glassllm steps
```

The first three use packaged weights and tokenizers. `steps` needs no assets because it reads the catalog. In the current implementation `tokenize` also loads model weights; use the Python tokenizer API alone if only its file is available. `train` loads only a tokenizer and initializes fresh weights.

## Subcommands and arguments

| Command | Arguments and result |
| --- | --- |
| `tokenize TEXT` | Prints ID and string per token; `--merges` shows each chunk's successive states |
| `generate TEXT` | `-n/--n=40`, `--temperature=0.8`, `--greedy=False`, `--seed=None`; prints prompt and continuation |
| `trace TEXT` | `--step=attn.softmax`, `--layer=None`, `--head=None`; prints equation, shape and values |
| `steps` | Lists the 25 operation IDs and equations |
| `train` | `--steps=200`, `--batch-size=8`, `--seq-len=64`, `--d-model=64`, `--layers=2`, `--lr=0.003`, `--seed=0`, `--log-every=20`, `--out=None`, `--corpus=None` |

`trace` accepts catalog IDs only. Block operations default to layer 0; head operations default to head 0. Inapplicable selectors raise errors. For `probs/sample` it performs greedy generation; for `loss/loss.grad` it shifts the sequence and requires at least two tokens; other operations perform full forward. Matrices show all rows and at most 8 columns, with ellipses for the rest. Probabilities display the top 10 candidates, unlike the interface's six.

CLI training reads `data/{lang}_train.txt` or explicit `--corpus`. It uses an existing tokenizer, always 4 Q heads and 2 K/V heads, and `max_len=seq_len`; `d_model` must meet divisibility requirements. Warmup is `min(100, steps//10+1)`. Unlike `scripts/train_model.py`, it performs no validation or best-weight restoration. `--out` saves an NPZ on completion; without it the model lives only in the process.

```bash
uv run glassllm train --lang pt --steps 3 --batch-size 2 --seq-len 8 --d-model 8 --layers 1 --corpus data/pt_train.txt --out /tmp/glassllm-cli-demo.npz
```

This last command needs a prepared corpus. Choose a different destination to preserve existing experiments; `save` overwrites files. CLI generation provides neither complete per-token trace, top-k/top-p, automatic EOS stop nor a character limit.

## cli module API

| Signature | Programmatic use |
| --- | --- |
| `main(argv: list[str] | None = None) -> None` | Parses and dispatches; `None` uses process arguments |
| `build_parser() -> argparse.ArgumentParser` | Constructs all subcommand parsers |
| `format_matrix(values: np.ndarray, row_labels: list[str], col_labels: list[str] | None = None) -> str` | Requires a 2D matrix and matching label counts; default column labels are indices |
| `cmd_tokenize(args, model: GlassLM | None, tok: Tokenizer) -> None` | Prints tokens and optional merges |
| `cmd_generate(args, model: GlassLM, tok: Tokenizer) -> None` | Consumes the generator and prints text |
| `cmd_trace(args, model: GlassLM, tok: Tokenizer) -> None` | Computes and prints a selected operation |
| `cmd_steps(args, model, tok) -> None` | Prints the catalog; ignores model/tokenizer |
| `cmd_train(args, model: GlassLM, tok: Tokenizer) -> None` | Creates and trains a fresh model; model argument is unused |

Handlers expect a namespace compatible with `build_parser`; use `main` as the normal entry point. `format_matrix` does not modify arrays and limits columns through `MAX_COLS=8`. It renders every non-finite value as `-inf`, without distinguishing NaN and +inf; inspect original arrays when that distinction matters.

```python
import numpy as np
from glassllm.cli import build_parser, format_matrix, main

args = build_parser().parse_args(["generate", "Alice", "--lang", "en", "-n", "1"])
assert args.lang == "en" and args.n == 1
table = format_matrix(np.array([[1.0, 2.0]]), ["token"], ["f0", "f1"])
assert "token" in table and "1.000" in table
main(["steps"])
```

`main` converts common value, key, index and filesystem errors into `SystemExit("error: ...")`. argparse handles argument syntax errors; training timeouts are not among the converted exceptions. Internal handlers can raise errors directly when called outside main.

## Produced files

| File | Contents and reader |
| --- | --- |
| `{lang}.npz` | Named matrices and JSON `__config__`; `GlassLM.load` |
| `{lang}.tokenizer.json` | `version`, `alphabet`, `merges`; `Tokenizer.load` |
| `{lang}.meta.json` | Steps, token counts, FP32/FP16 losses and evaluation settings; metadata rather than a checkpoint |
| `step_index.json` | ID → source file/line; `load_step_index` |
| Local `.whl` | Python package with modules and prepared assets; also used by the browser |

NPZ files do not include corpus, tokenizer or optimizer. A wheel filename must follow the full wheel convention, such as `glassllm-0.1.0-py3-none-any.whl`; a URL ending only in `glassllm` cannot be installed by micropip. The released package is available with `python -m pip install glassllm`; the browser lab continues to serve its own wheel to Pyodide.

## Checkout scripts

Run scripts from the repository root with the package installed in the environment.

```bash
uv run python scripts/prepare_corpus.py --out data
uv run python scripts/train_tokenizer.py --lang pt --vocab-size 1024 --data data --out /tmp/glassllm-training
uv run python scripts/train_model.py --lang pt --steps 3 --batch-size 2 --data data --out /tmp/glassllm-training
uv run python scripts/build_step_index.py
```

`prepare_corpus.py` downloads eight books, removes Gutenberg delimiters, normalizes NFC and whitespace, and reserves approximately the final 5% of each book for validation, cutting at the following newline. It requires networking and does not randomly interleave passages from the same book between splits. `train_tokenizer.py` learns from training text only and writes its tokenizer to the output directory.

`train_model.py` requires that tokenizer in the same directory. Defaults are 3,000 steps, batch 16 and full Config. It trains with validation and `keep_best=True`, saves float16 weights and compares export validation loss with FP32 on identical windows; non-finite losses or absolute differences above 0.01 raise `ValueError`. Its two-hour deadline also covers evaluation and export. Metadata is written only after these checks.

`build_step_index.py` refreshes line locations after engine changes. `build_api_inventory.py` rebuilds the documentation's symbol inventory; it does not validate signatures through execution. See [project integration](../../projeto/integracao-browser/) and [training](../treino-otimizacao/) for the contracts used by these scripts.
