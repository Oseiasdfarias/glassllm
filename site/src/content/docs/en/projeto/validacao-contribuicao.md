---
title: Validation and contribution
description: Verify engine, documentation and visualization changes while keeping equations, tensors and source connected.
---

## What the tests check

The library tests cover autograd, layers, models, generation, tokenizers, training, assets, CLI and step markers. A PyTorch reference exists only in development to compare values and gradients; the runtime package depends on NumPy.

| Area | Checked evidence |
| --- | --- |
| Autograd | PyTorch and finite-difference gradients; broadcasting, repeated indices and shared paths |
| Layers | RMSNorm, RoPE, causal/GQA attention and SwiGLU forward and backward |
| Model | Shapes, loss, shared parameters and save/load |
| Generation | Cache versus full forward, window rebuild, temperature, seeds and per-prediction traces |
| Tokenizer | Merges, Unicode normalization, spaces, unknowns and persistence |
| Training | Shifted batches, AdamW, clipping, schedule, evaluation and best checkpoint |
| CLI and assets | Arguments, output, PT/EN files and loading errors |
| Steps | Correspondence between `STEPS`, source markers and source index |

Parity in one case does not prove that every possible configuration works. Each test covers particular inputs and tolerances; numerical changes require examining affected shapes and dtypes as well as values.

## Running checks

At the checkout root:

```sh
uv sync --group dev
uv run pytest -q
uv run glassllm --help
```

`tests.yml` runs the suite on Python 3.10 and 3.13. For a limited change, start with the relevant test file; run the full suite for shared contracts, autograd or model changes. The development group installs PyTorch from the CPU index configured in `pyproject.toml`.

For the site, from `site/`:

```sh
npm ci
npm run check
npm run build
```

`check` prepares the wheel and checks Astro/TypeScript; `build` generates static pages and copies assets. These commands alone do not establish that Python loaded in the browser. Generate an actual token, select an operation, compare shape and source, and check pause/resume. When changing routes or navigation, visit PT and EN pages, including paper links and images.

## Updating derived sources

```sh
uv run python scripts/build_step_index.py
uv run python scripts/build_api_inventory.py
```

The first rebuilds `src/glassllm/step_index.json`; the second generates `site/src/data/api-inventory.json` from the Python AST. Run them when marker locations or signatures change. The inventory supplies names and signatures; documentation explains contracts, returns and effects.

PT/EN content lives in `site/src/content/docs/{pt,en}/`. `docs-plan.ts` defines routes and groups; the renderer imports matching Markdown and rejects missing content. After changing a function, review both language references and the foundations that use it.

## Checking interface and screenshots

With the local server running on port 4322, Node.js 24 and Chrome installed, run from `site/`:

```sh
npm run capture:docs
```

The script records 17 screenshots per language, using the actual model for prediction, tracing, causal masking and the 128-token limit. It also exercises an unavailable wheel and recovery. `manifest.json` records each image's language, prompt, selection mode, temperature, operation, head, viewport and dimensions. Use `GLASSLLM_CAPTURE_URL` for another server or `GLASSLLM_CHROME` for a different executable path.

Check keyboard behavior too: visible focus, tab controls, index expansion, links and full screen. Widths of 390 and 320 px help detect overflow but do not replace reading on an actual device. During capture, the inspector is expanded so the matrix and source remain readable; its values still come from the model.

## Contributing a change

`CONTRIBUTING.md` describes the repository process. `develop` is the integration branch; `main` holds released versions. A PR should describe the problem, resulting behavior and relevant validation. Include screenshots for presentation changes.

A new operation needs a `# @step` marker, a catalog equation, an updated source index and an explanation in both languages. Prefer names corresponding to equations and keep the engine executable with NumPy. Architecture changes should explain differences from the referenced papers. Library and site publication remain separate from local verification.
