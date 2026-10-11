<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Oseiasdfarias/glassllm/develop/docs/assets/banner-dark.svg">
  <img alt="glassllm: see every step of an LLM" src="https://raw.githubusercontent.com/Oseiasdfarias/glassllm/develop/docs/assets/banner-light.svg" width="100%">
</picture>

**A small Transformer written from scratch in NumPy, built to be looked at.**<br>
Pick any step of the model and see the equation, the line of code that computes it,<br>
and the real numbers it produced for your sentence.

[![Status](https://img.shields.io/badge/status-pre--alpha-c98a12)](#roadmap)
[![License: MIT](https://img.shields.io/badge/license-MIT-0f6e7a)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-0f6e7a)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/runtime%20deps-numpy%20only-0f6e7a)](#design-principles)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-0f6e7a)](CONTRIBUTING.md)

**English** · [Português](README.pt-BR.md)

</div>

---

> [!NOTE]
> **Phases 1 and 2 are implemented and validated locally.** The bilingual Astro site runs the library
> and pretrained models in a Pyodide worker. The PyPI release workflow is configured; its Trusted
> Publisher still needs one-time setup on PyPI. GitHub Pages deployment and the chapter path remain pending.

## Why

Most explanations of LLMs stop at diagrams, and most real code hides the math behind
`torch.nn` and autograd. You end up knowing the words without seeing the machine run.

glassllm closes that gap. Every computation in the model is tagged, recorded and linked to
its equation, so you can follow a single token from text to prediction and check every
number along the way.

## What you get

| | |
|---|---|
| **A Python library** (source in this repository) | NumPy tokenizer, autograd engine, decoder-only Transformer, tracing, training code, CLI and trained Portuguese and English models. The tagged PyPI release is pending. |
| **A trace mode** | Pass a `Tracer` to the model and every tagged step is recorded with its layer, head, shape and value. |
| **A command line tool** | Tokenize text, inspect model steps, generate text and trace model values with the included PT/EN assets. |
| **An interactive site** | A PT/EN lab with real generation, attention, probabilities and the complete Transformer architecture. Zoom into the diagram, replay all 77 calculations per prediction and inspect equations, matrices and code. It runs in a Pyodide worker; the chapter path is planned for Phase 3. |

## Quickstart

From the repository checkout, install the development environment and run either language model:

```bash
uv sync
uv run glassllm --lang en generate "Alice was beginning to" --n 40
uv run glassllm --lang pt generate "Capitu olhou para" --n 40
uv run glassllm steps
```

## Run the site

With Node.js 22.12+ and `uv` installed:

```bash
cd site
npm ci
npm run dev
```

The first run builds the Python wheel for the browser lab. The lab downloads the pinned Pyodide
runtime and the wheel when you start generation.

In the lab, open **Explore the network**. The map shows all four blocks with Q/K/V, RoPE,
GQA, K/V cache, causal masking, residuals and SwiGLU. Generate a token and click an operation
to inspect its actual tensor and source line; the head selector lets you compare attention
heads. **Replay flow** and **Next calculation** follow the order recorded by the model.
Zoom, block navigation and full screen help you explore the diagram.

Enable **Slow motion** to follow one operation at a time, with its explanation, equation
and values. Reading time ranges from 1 to 8 seconds per calculation (3 seconds by default).
**Pause flow**, **Continue**, **Previous calculation** and **Next calculation** let you study
at your own pace. The diagram follows the current step; uncheck **Follow current step** to
pan freely. Resuming generation finishes the current prediction's flow before generating
another token. This mode slows the visualization of actual recorded calculations.

The **Documentation** link opens 35 complete pages per language, with inference and training
flowcharts, interface guides, foundations, API reference, examples and BibTeX bibliography.
There are 34 actual screenshots of components and runtime states. To regenerate screenshots, keep
the server running and execute `npm run capture:docs` in `site/`, with Node.js 24 and Chrome.
Set `GLASSLLM_CAPTURE_URL` if the server uses a different port; the default is 4322.

## Preview

The chapter and free-mode images below are design mockups. The live site currently includes the
bilingual overview and the free generation lab; the chapter experience is planned for Phase 3.

**Chapter screen.** Theory, real code and values side by side. The highlighted term in the
equation, the highlighted line of code and the matrix all refer to the same step.

<img alt="Chapter screen: equation, code and attention weights side by side" src="https://raw.githubusercontent.com/Oseiasdfarias/glassllm/develop/docs/assets/preview-chapter.png" width="100%">

**Free mode.** Type a prompt, press play, and watch each token travel through the model,
get a probability and join the text.

<img alt="Free mode: token by token generation loop" src="https://raw.githubusercontent.com/Oseiasdfarias/glassllm/develop/docs/assets/preview-loop.png" width="100%">

## How it works

One core library feeds both the terminal and the browser.

```mermaid
flowchart LR
    subgraph core["glassllm (Python + NumPy)"]
        tok[Tokenizer BPE] --> model[GlassLM]
        model --> tracer[Tracer]
    end
    core --> wheel[(wheel)]
    wheel --> pypi[PyPI<br/>pip install glassllm]
    wheel --> site[Astro site<br/>Pyodide in a Web Worker]
    pypi --> cli[CLI and notebooks]
    site --> pages[GitHub Pages]
```

Inside the model, a token goes through these steps. Each one is a tagged step you can inspect.

```mermaid
flowchart TB
    T["text"] --> B["BPE tokens"] --> E["embedding · E[ids]"]
    E --> N1["RMSNorm"] --> A["attention<br/>Q, K, V → RoPE → QKᵀ/√dₖ → mask → softmax → ·V"]
    A --> R1(("+")) --> N2["RMSNorm"] --> F["SwiGLU FFN"] --> R2(("+"))
    E -. residual .-> R1
    R1 -. residual .-> R2
    R2 -->|"× 4 blocks"| NF["final RMSNorm"] --> L["logits · Eᵀ"] --> P["softmax(z / T)"] --> S["sample next token"]
```

### Trace mode

Each step is marked in the source with a comment, and the tracer records the value when the
model runs. Example:

```python
from glassllm import load_pretrained, Tracer

model, tokenizer = load_pretrained("en")
ids = tokenizer.encode("Alice was beginning to")

tracer = Tracer()
model.forward(ids, trace=tracer)

step = tracer.get("attn.softmax", layer=0, head=1)
print(step.equation)   # softmax(QKᵀ/√dₖ + M)
print(step.value)      # the (n, n) attention weights for this sentence
```

```python
# glassllm/nn.py
weights = softmax(scores, axis=-1)  # @step attn.softmax
```

## The learning path

| # | Chapter | You will see |
|---|---|---|
| 0 | Overview | The whole flow, from text to the next token |
| 1 | BPE tokenization | How merges turn text into tokens |
| 2 | Embedding | Token IDs becoming vectors |
| 3 | Attention | Q, K, V, scores, softmax and the weighted mix |
| 4 | Causal mask | Why the future gets weight zero |
| 5 | RoPE | Rotations that encode relative position |
| 6 | Multi-head and GQA | Heads side by side, shared K and V |
| 7 | Residual and RMSNorm | The residual stream and why pre-norm |
| 8 | SwiGLU FFN | The gated per-token network |
| 9 | Logits, loss and training | Cross-entropy, the `p − y` gradient, a live mini training run |
| 10 | Generation loop and KV cache | Sampling, temperature and caching |

The full derivations, in Portuguese, are already in [docs/equacoes.md](docs/equacoes.md).

## Roadmap

- [x] Design spec ([docs/specs](docs/specs/2026-10-10-glassllm-design.md))
- [x] Equations and derivations ([docs/equacoes.md](docs/equacoes.md))
- [x] **Phase 1: library.** PT/EN corpora and models are built; tests pass on Python 3.10 and 3.13; the wheel was installed and smoke-tested outside the checkout. The PyPI release workflow is configured; publication awaits PyPI Trusted Publisher setup and a tagged release.
- [x] **Phase 2: site and free mode.** Astro, PT/EN pages, Pyodide worker, generation loop and local build
- [ ] **Phase 2 deployment.** Publish the site to GitHub Pages from `main`
- [ ] **Phase 3: chapters.** Chapters 0 to 10 in English and Portuguese

## Design principles

- **Nothing hidden.** Runtime depends on NumPy only. Gradients come from a small autograd
  engine you can read in one sitting.
- **Verified, not approximate.** Every layer is tested against a PyTorch reference, forward
  and backward.
- **Small on purpose.** About 0.9M parameters, so it runs in a browser tab. The goal is to
  understand, not to write good prose.
- **One source of truth.** The site runs the wheel built from the same Python library.
- **Two languages.** Interface and chapters in English and Portuguese.

## Repository layout

```
glassllm/
├── src/glassllm/      library implementation
├── tests/             tests, including PyTorch parity
├── scripts/           corpus, tokenizer and model training
├── site/              bilingual Astro site and Pyodide free lab
└── docs/
    ├── equacoes.md    equations and derivations (PT)
    ├── specs/         design documents
    ├── prototipo/     the original PyTorch prototype
    └── assets/        images used in this README
```

## Contributing

Contributions are welcome, and you do not need to be an ML expert to help. Useful places to
start:

- **Explanations.** Clearer wording, better intuitions, fixes in the derivations.
- **Translations.** Every chapter exists in English and Portuguese.
- **Tests.** More parity checks and edge cases.
- **Visuals.** Better ways to show a matrix, a rotation or a gradient.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request, and follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Acknowledgements

glassllm builds on ideas from these works:

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017.
- Su et al., [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864), 2021.
- Shazeer, [*GLU Variants Improve Transformer*](https://arxiv.org/abs/2002.05202), 2020.
- Zhang and Sennrich, [*Root Mean Square Layer Normalization*](https://arxiv.org/abs/1910.07467), 2019.
- Ainslie et al., [*GQA: Training Generalized Multi-Query Transformer Models*](https://arxiv.org/abs/2305.13245), 2023.
- Sennrich et al., [*Neural Machine Translation of Rare Words with Subword Units*](https://arxiv.org/abs/1508.07909), 2016.
- Kudo and Richardson, [*SentencePiece*](https://arxiv.org/abs/1808.06226), 2018.
- Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and [nanoGPT](https://github.com/karpathy/nanoGPT), for showing how much fits in a small, readable codebase.

Training text comes from public domain books on [Project Gutenberg](https://www.gutenberg.org/):
novels by Machado de Assis (Portuguese) and Lewis Carroll (English).

## License

Code under the [MIT License](LICENSE). Chapter texts under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
