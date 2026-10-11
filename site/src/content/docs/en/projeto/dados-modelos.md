---
title: Data, models and reproducibility
description: PT/EN corpus provenance, checkpoint configuration and the meaning of metrics shipped with the package.
---

## Text provenance

The distributed models are separate for each language. `scripts/prepare_corpus.py` downloads eight Project Gutenberg books: five by Machado de Assis for PT and three by Lewis Carroll for EN. The pages for [Dom Casmurro](https://www.gutenberg.org/ebooks/55752) and [Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11) identify these works and their files. This is the script's actual corpus, rather than a general internet corpus.

| Language | Book | Gutenberg ID |
| --- | --- | --- |
| PT | Dom Casmurro | 55752 |
| PT | Memorias Posthumas de Braz Cubas | 54829 |
| PT | Quincas Borba | 55682 |
| PT | Esau e Jacob | 56737 |
| PT | Memorial de Ayres | 55797 |
| EN | Alice's Adventures in Wonderland | 11 |
| EN | Through the Looking-Glass | 12 |
| EN | Sylvie and Bruno | 620 |

The script removes Gutenberg headers and footers, normalizes Unicode to NFC, compresses horizontal whitespace and reduces repeated blank lines. It reserves approximately the last 5% of each book for validation, moving the split to a newline, then concatenates training and validation parts separately by language. This evaluates later passages of the same works, rather than generalization to unseen authors or domains.

## Tokenizers and configuration

Each language has its own BPE tokenizer with 1024 entries. Identical IDs may represent different text in PT and EN; load the tokenizer matching the checkpoint. Tokenizer files, `.npz` weights and `.meta.json` metadata live in `src/glassllm/weights/`.

| Property | Distributed models |
| --- | --- |
| Vocabulary | 1024 |
| Model dimension | 128 |
| Transformer blocks | 4 |
| Query heads | 4 |
| K/V heads | 2 |
| Dimension per head | 32 |
| SwiGLU hidden dimension | 341 |
| Context window | 128 tokens |
| Unique parameters | 852,608 |

The embedding matrix is shared with output projection. The parameter count includes unique parameters returned by `model.parameters()`. Export stores float16 weights by default; `load_pretrained` reconstructs float32 parameters. Intermediate operations can promote precision according to NumPy rules, so inspect array `dtype` when estimating memory.

## Distributed metadata

These values come from packaged metadata. Loss is mean cross-entropy using natural logarithms; smaller values indicate greater probability assigned to targets in the evaluated windows.

| Metric | PT | EN |
| --- | --- | --- |
| Requested training steps | 3000 | 3000 |
| Training-set tokens | 694,356 | 229,390 |
| Validation-set tokens | 38,009 | 12,573 |
| Float32 checkpoint loss | 3.4348712394 | 4.1935762033 |
| Loss after float16 export and reload | 3.4348720647 | 4.1935892849 |

Recorded evaluation uses 10 batches of 16 sequences with 128 positions and seed 1234: 20,480 sampled targets per evaluation. This is not the validation set's unique-token count. Sampled windows can overlap and use targets shifted one position; average loss does not necessarily cover the entire corpus.

Metadata records requested steps, rather than the best iteration's index. `keep_best=True` restores the best weights seen in periodic validation before export, so the distributed checkpoint can come from an earlier evaluation than the final training step. The loop uses zero-based step indices: it evaluates after updates at indices 0, 100, 200 and so on, plus the last index even when it falls outside that interval.

PT and EN losses do not directly rank language quality: tokenization and data differ. They also do not measure factual accuracy, general knowledge or conversational skill. This small model can repeat, produce incomplete text or generate incoherent continuations.

## Reproducing the pipeline

From an installed checkout, these commands prepare corpora and show tokenizer and training options. Commands with `--help` do not train. Downloading needs a network connection; running training writes to the directory selected by `--out`. Choose a different output directory to preserve distributed checkpoints and create that language's tokenizer there before starting `train_model.py`.

```sh
uv run python scripts/prepare_corpus.py --out data
uv run python scripts/train_tokenizer.py --help
uv run python scripts/train_model.py --help
```

`scripts/train_model.py` uses seed 0 for initialization and batch sampling, default learning rate `3e-3`, 100 warmup steps, periodic evaluation every 100 indices and best-checkpoint selection. It compares float32 loss with float16-export loss on identical windows, rejecting non-finite values or differences above 0.01. The script checks a two-hour budget around its phases. `train` also checks its deadline between update phases and validation batches. These checks do not interrupt an executing NumPy operation; a `TimeoutError` ends the attempt without promising that export and metadata will be produced.

Keep configuration, Python/NumPy versions, downloaded files, seeds and training arguments to reproduce a run. Network data can change; a seed does not replace preserving the corpus. `model.save()` stores configuration and weights, rather than optimizer state or training history.

## Checking a checkpoint

```python
from glassllm import load_pretrained

model, tokenizer = load_pretrained("pt")
print(tokenizer.vocab_size)
print(sum(parameter.data.size for parameter in model.parameters()))
print(model.cfg.max_len)
```

Distributed assets return `1024`, `852608` and `128`. To repeat a continuation, preserve language, prompt, checkpoint, temperature, selection mode and seed. The lab supports most-likely selection; the Python API also accepts `seed` for sampling. See [generation](../../biblioteca/glasslm/) and [training](../../biblioteca/treino-otimizacao/).
