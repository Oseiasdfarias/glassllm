---
title: "GlassLM and GenStep"
description: "Inference, loss, cached generation, parameter inspection and Transformer checkpoints."
---

## Construct and list parameters

`from glassllm import GlassLM, GenStep` imports the model class and a prediction result. `GlassLM(cfg: Config, seed: int = 0, dtype=np.float32)` initializes random weights with a local NumPy generator. Equal configuration, dtype and seed reproduce initialization; these are not the trained weights provided by `load_pretrained`.

The model inherits `Module`. `named_parameters(prefix: str = "") -> dict[str,Tensor]` traverses attributes, submodules and lists of submodules. Names such as `E`, `blocks.0.attn.W_Q` and `norm_f.gamma` become checkpoint entries. `parameters() -> list[Tensor]` returns these same objects in order, not copies: changing `.data` changes the model. Output uses the transpose of `E`, without another parameter.

## forward and `__call__`

`forward(ids, trace: Tracer | None = None, cache: list[KVCache] | None = None, start: int = 0) -> Tensor` returns `(B,n,V)` logits. `model(ids, ...)` aliases it. IDs are converted to `int64`; a `(n,)` input gains batch 1. It rejects inputs with other dimensionalities, negative positions, `start+n > max_len`, and tracing with batch size other than 1.

IDs must belong to the vocabulary. NumPy indexing accepts negative IDs, and the engine has no dedicated semantic validation to reject them. Floating-point inputs are converted to integers too, so validate external data first. Empty sequence lengths can pass portions of forward but are not valid generation input.

Here `trace` is a **Tracer object**, unlike the boolean argument to `generate`. Without `no_grad`, logits normally retain the parameter graph. A supplied cache is populated per block; its list must contain one item per block. `start` is the absolute offset of current positions within the window. Cache and offset must agree; see [layers and cache](../camadas-cache/).

```python
import numpy as np
from glassllm import Config, GlassLM, Tracer, no_grad

model = GlassLM(Config(vocab_size=8, d_model=8, n_layers=1,
                      n_heads=2, n_kv_heads=1, max_len=4), seed=0)
trace = Tracer()
logits = model.forward([1, 2], trace=trace)
assert logits.shape == (1, 2, 8)
assert trace.get("embed").shape == (2, 8)
with no_grad():
    reference = model([1, 2, 3]).data[:, -1:]
    cache = model.new_cache()
    model([1, 2], cache=cache)
    incremental = model([3], cache=cache, start=2).data
np.testing.assert_allclose(incremental, reference, rtol=1e-5, atol=1e-6)
```

## loss

`loss(ids, targets, trace: Tracer | None = None) -> Tensor` performs a full forward, flattens positions and returns scalar mean cross-entropy. Supply one target per input position; the method does not shift text automatically. For next-token prediction, pass `ids=data[:-1]`, `targets=data[1:]`. Targets must be valid integer indices with `B*n` entries.

The loss uses stable log-softmax with no generation temperature. Call `loss.backward()` to accumulate parameter gradients. With tracing it records `loss` and `loss.grad`; the latter is the analytical derivative with respect to logits, not evidence that `backward()` has run. Loss training uses no cache.

## generate and GenStep

`generate(ids, n_new: int, temperature: float = 1.0, greedy: bool = False, seed: int | None = None, trace: bool = False) -> Iterator[GenStep]` generates up to `n_new` tokens. Supply a nonempty ID sequence. The prompt is truncated to its last `max_len` tokens. Zero new tokens yields no results; currently a negative `n_new` also yields no results without dedicated validation.

Each `GenStep` is a frozen dataclass with `token: int`, `probs: np.ndarray` shaped `(V,)`, and `trace: Tracer | None`. The float64 distribution covers the whole vocabulary, not only the interface's six candidates. Frozen fields prevent reassignment, but the array itself remains mutable.

Temperature must be positive even for `greedy=True`; zero and negative values raise `ValueError`. Greedy selection uses `argmax`; sampling uses the NumPy generator initialized by `seed`. `trace` accepts only a bool, otherwise raises `TypeError`; `trace=True` creates a fresh tracer per result. These errors appear when the generator advances, not when it is merely created.

```python
import numpy as np
from glassllm import Config, GlassLM

model = GlassLM(Config(vocab_size=8, d_model=8, n_layers=1,
                      n_heads=2, n_kv_heads=1, max_len=4), seed=0)
steps = list(model.generate([1, 2], 3, seed=42, trace=True))
again = list(model.generate([1, 2], 3, seed=42))
assert [s.token for s in steps] == [s.token for s in again]
assert steps[0].trace is not steps[1].trace
assert int(steps[0].trace.get("sample").value) == steps[0].token
np.testing.assert_allclose(steps[0].probs.sum(), 1.0)
```

The prefix is evaluated once; subsequent calls receive the last token and reuse K/V. When the window is full, the cache is rebuilt from the last `max_len` IDs with positions reset to zero. Output therefore matches a full forward on that window rather than extrapolating the RoPE table. Generation does not automatically stop at EOS. `no_grad` is restored before every `yield`, allowing caller training operations between results.

`from glassllm.model import next_token_probs` provides `next_token_probs(logits: np.ndarray, temperature: float) -> np.ndarray`: softmax over the last axis of `logits / temperature`, leaving the input unchanged. It requires positive temperature and performs neither sampling nor top-k/top-p filtering.

## new_cache, save and load

`new_cache() -> list[KVCache]` creates separate empty caches for the blocks. `save(path: str | Path, fp16: bool = True) -> None` writes compressed NPZ with a JSON `__config__` and named parameter arrays. Default conversion to float16 occurs **in the file**, leaving memory weights unchanged. `fp16=False` preserves each parameter's dtype. Existing destinations are overwritten, parent directories are not created, and NumPy may append `.npz` to filenames without that extension.

`GlassLM.load(path: str | Path, dtype=np.float32) -> GlassLM` rebuilds the Config and loads arrays converted to the requested dtype. Missing entries raise `KeyError`; invalid files raise NumPy reader errors. The current loader does not explicitly compare every parameter shape to the architecture before assigning it; use files produced by `save`.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from glassllm import Config, GlassLM, no_grad

model = GlassLM(Config(vocab_size=8, d_model=8, n_layers=1,
                      n_heads=2, n_kv_heads=1, max_len=4))
with TemporaryDirectory() as directory:
    path = Path(directory) / "model.npz"
    model.save(path, fp16=False)
    restored = GlassLM.load(path)
    with no_grad():
        np.testing.assert_array_equal(model([1, 2]).data, restored([1, 2]).data)
```

Checkpoints do not contain the tokenizer, corpus, cache, trace, gradients or AdamW state. Save the tokenizer separately to resume an experiment; resuming identical optimization state would additionally require optimizer serialization, which this version does not provide. The architecture combines [Transformer attention](https://arxiv.org/abs/1706.03762) and [tied embeddings](https://arxiv.org/abs/1608.05859), with RMSNorm, RoPE, GQA and SwiGLU changes explained in the foundations pages.
