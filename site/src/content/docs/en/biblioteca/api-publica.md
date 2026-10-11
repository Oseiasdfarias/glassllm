---
title: "API map and conventions"
description: "Imports, types, dimensions and contracts of the NumPy glassllm engine."
---

## Choose an entry level

The `glassllm` package exports 12 names: `Config`, `GenStep`, `GlassLM`, `LANGUAGES`, `STEPS`, `StepRecord`, `Tensor`, `Tokenizer`, `Tracer`, `load_pretrained`, `no_grad` and `__version__`. They cover configuration, generation, loading and model inspection. Mathematical operations, layers, optimizers and training helpers are imported from their submodules.

```python
import numpy as np
from glassllm import Config, GlassLM, Tensor, Tracer, no_grad

cfg = Config(vocab_size=8, d_model=8, n_layers=1,
             n_heads=2, n_kv_heads=1, max_len=4)
model = GlassLM(cfg, seed=0)
trace = Tracer()
with no_grad():
    logits = model([1, 2, 3], trace=trace)
assert logits.shape == (1, 3, 8)
assert not logits.requires_grad
np.testing.assert_allclose(
    trace.get("attn.softmax", layer=0, head=0).value.sum(-1), 1.0,
    rtol=1e-5,
)
```

This example uses random weights to check contracts. To generate with trained weights, use `model, tokenizer = load_pretrained("en")`. Library publication remains deferred, so installation uses the checkout or a local wheel.

## Import paths

| Module | Symbols and purpose |
| --- | --- |
| `glassllm.config` | `Config`: dimensions, limits and validation |
| `glassllm.model` | `GlassLM`, `GenStep`, `next_token_probs`: logits, loss and generation |
| `glassllm.tokenizer` | `Tokenizer`, `pre_tokenize`, `SPACE`, `SPECIALS`, `UNK_TEXT`: custom BPE |
| `glassllm.trace` | `Tracer`, `StepRecord`, `record`: copied snapshots |
| `glassllm.steps` | `StepInfo`, `STEPS`, `LAYER_STEPS`, `HEAD_STEPS`, `scan_markers`, `load_step_index`: equations and source locations |
| `glassllm.autograd` | `Tensor`, `no_grad` and differentiable operations: reverse graph |
| `glassllm.nn` | `Module`, `RMSNorm`, `Attention`, `SwiGLU`, `Block`, `KVCache`, parameter, mask and angle helpers |
| `glassllm.optim` | `AdamW`, `clip_grad_norm`, `lr_schedule`: parameter updates |
| `glassllm.train` | `get_batch`, `evaluate`, `train`: windows, evaluation and training |
| `glassllm.cli` | `main`, `build_parser`, `format_matrix`, `cmd_*` handlers: terminal |

Names starting with `_` are internal helpers. This is not the PyTorch module interface: `Module` has no `.train()`, `.eval()`, `.to()` or `.state_dict()`. The engine runs on NumPy; the browser runs the same Python through Pyodide.

## Shape and indexing conventions

| Symbol | Meaning | Included model default |
| --- | --- | --- |
| `B` | batch | 1 in the interface and trace |
| `n` | positions received by this call | up to 128 |
| `m` | key positions, including cache | up to 128 |
| `V` | vocabulary size | 1024 |
| `d` | model dimension | 128 |
| `Hq`, `Hkv` | query and key/value heads | 4 and 2 |
| `dk = d / Hq` | head dimension | 32 |

IDs and targets are integer indices; positions, layers and heads are zero based. `forward` accepts `(n,)` or `(B,n)` IDs and returns a `(B,n,V)` `Tensor`. The batch axis remains even when the input was a list. The tracer removes a unit batch axis: a full attention record is `(Hq,n,m)`, or `(n,m)` after head selection. The tokenizer does not automatically add BOS/EOS.

## Types, state and errors

`Tensor.data` is a `numpy.ndarray`; `Tensor.grad` starts as `None`. Parameters usually use float32; derivative examples may use float64. There is no GPU device, symbolic type system or second-order differentiation. Direct NumPy operations on `.data` are outside the graph.

`forward`, `loss` and `generate` do not update weights. `train` and `AdamW.step` modify the existing object; `forward` also fills a supplied cache. `generate` owns its cache and creates a fresh tracer for each token when `trace=True`.

Explicit validation primarily raises `ValueError`, `TypeError`, `KeyError` and `FileNotFoundError`. Incompatible shapes, out-of-range IDs and malformed files can also raise NumPy or JSON errors. The library does not wrap every internal exception in one error class. Because `generate` is a generator, its validation runs only when iteration advances.

Continue with [Configuration and loading](../config-carregamento/), [GlassLM](../glasslm/) or [Tensor and autograd](../tensor-autograd/).
