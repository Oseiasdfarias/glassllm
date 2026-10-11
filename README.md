![glassllm — see every step of a small language model](https://raw.githubusercontent.com/Oseiasdfarias/glassllm/main/docs/assets/banner-light.svg)

# glassllm

A small Transformer written from scratch in NumPy, built to be looked at. Follow a token through
the model and inspect the equation, source code and values produced at each step.

[![PyPI version](https://img.shields.io/pypi/v/glassllm)](https://pypi.org/project/glassllm/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![NumPy only](https://img.shields.io/badge/runtime-NumPy%20only-4fb3c2)](https://numpy.org/)
[![MIT license](https://img.shields.io/badge/license-MIT-4fb3c2)](https://github.com/Oseiasdfarias/glassllm/blob/main/LICENSE)

**English** · [Português](https://github.com/Oseiasdfarias/glassllm/blob/main/README.pt-BR.md) ·
[Documentation](https://oseiasdfarias.github.io/glassllm/)

glassllm is an educational language model and Python library. Its decoder-only Transformer,
tokenizer, automatic differentiation engine, training code and tracing tools are implemented
with NumPy. Pretrained Portuguese and English models are included in the package.

## Install

```bash
python -m pip install glassllm
```

The library supports Python 3.10 and later. NumPy is its only runtime dependency. The pretrained
weights and tokenizers are included, so inference works immediately after installation.

## Generate text

The `glassllm` command uses the included English model by default. Select `pt` for Portuguese.

```bash
glassllm --lang en generate "Alice was beginning to" --n 40
glassllm --lang pt generate "Capitu olhou para" --n 40
```

Use greedy decoding for repeatable output, or inspect the available model operations:

```bash
glassllm --lang en generate "Alice was beginning to" --n 40 --greedy
glassllm steps
glassllm trace "Alice was beginning to" --step attn.softmax --layer 0 --head 1
```

## Inspect a calculation

Pass a `Tracer` to the model to record its actual intermediate values. Each record includes the
equation, tensor shape and source location for a tagged operation.

```python
from glassllm import Tracer, load_pretrained

model, tokenizer = load_pretrained("en")
token_ids = tokenizer.encode("Alice was beginning to")

tracer = Tracer()
model.forward(token_ids, trace=tracer)

step = tracer.get("attn.softmax", layer=0, head=1)
print(step.equation)
print(step.shape)
print(step.value)
```

List all traceable operations with `glassllm steps`. The same tagged calculations are explained
in the [library documentation](https://oseiasdfarias.github.io/glassllm/en/docs/biblioteca/api-publica/).

## What happens inside the model

Each prediction follows the same sequence of real calculations. The website can replay all 77
recorded operations for a prediction and lets you inspect their equations, tensors and source.

```text
text → BPE tokens → embeddings → 4 Transformer blocks → logits → probabilities → next token
                                  attention + SwiGLU
```

The model uses causal self-attention, RoPE, grouped-query attention, an attention KV cache,
RMSNorm, SwiGLU and tied input/output embeddings. It has 852,608 parameters and runs on CPU.
It is deliberately small for learning; it is not intended for production text generation.

## Interactive site and documentation

Try the [live bilingual lab](https://oseiasdfarias.github.io/glassllm/). It runs the same Python
wheel in a Pyodide worker in your browser. The network view shows the complete model architecture,
and slow motion steps through each recorded calculation at a readable pace.

The documentation includes 35 pages in each language, with interface guides, inference and
training flowcharts, mathematical foundations, API reference, examples and research references:

- [Documentation in English](https://oseiasdfarias.github.io/glassllm/en/docs/)
- [Documentação em português](https://oseiasdfarias.github.io/glassllm/pt/docs/)

## Training and data

The included models were trained on public-domain books by Machado de Assis (Portuguese) and
Lewis Carroll (English), sourced from [Project Gutenberg](https://www.gutenberg.org/). Training
scripts and dataset preparation live in the repository. PyTorch is used only for development
and parity checks; it is not installed when users install the library.

## Contributing

The source code, tests, design notes and issue tracker are on
[GitHub](https://github.com/Oseiasdfarias/glassllm). See the
[contribution guide](https://github.com/Oseiasdfarias/glassllm/blob/main/CONTRIBUTING.md) before
opening a pull request.

The interactive lab and complete library documentation are available now. The guided chapter
experience is planned for a later phase.

## License

The Python library is licensed under the [MIT License](https://github.com/Oseiasdfarias/glassllm/blob/main/LICENSE).
Documentation text is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
