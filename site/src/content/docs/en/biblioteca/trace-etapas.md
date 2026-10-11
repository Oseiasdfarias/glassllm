---
title: "Tracer, records and step catalog"
description: "Actual snapshots, layer and head selection, 25 operations and source-code links."
---

## Record an execution

`from glassllm import Tracer, StepRecord, STEPS` imports inspection APIs. `Tracer()` starts with an empty `records: list[StepRecord]`. It copies values at marked model locations: it neither pauses computations nor independently computes attention, and it does not record every elementary NumPy operation. Interface slow motion replays these snapshots after execution.

`record(step_id: str, value, layer: int | None = None, batched: bool = True, *, head: int | None = None) -> None` accepts a Tensor or an array-convertible value. `step_id` must exist in `STEPS`, otherwise `KeyError` is raised. `batched=True` requires and removes a leading axis of size 1; a scalar or larger batch raises `ValueError`. Use `batched=False` for scalars, probability vectors or values already without batch.

The recorded array is an independent copy of the input. `order` is its zero-based list position. Supplied layer and head selectors must be nonnegative integers; booleans are rejected. Heads are only permitted for `HEAD_STEPS` IDs. The tracer does not validate a layer against a model or require a layer for block IDs; that is the caller's responsibility.

```python
import numpy as np
from glassllm import Tracer

trace = Tracer()
value = np.ones((1, 2, 3), dtype=np.float32)
trace.record("embed", value)
value[:] = 7
record = trace.get("embed")
assert record.shape == (2, 3)
np.testing.assert_array_equal(record.value, np.ones((2, 3)))
assert record.order == 0
trace.record("sample", np.array(4), batched=False)
assert int(trace.get("sample").value) == 4
```

The submodule helper `glassllm.trace.record(trace: Tracer | None, step_id, value, layer=None, batched=True, *, head=None) -> None` invokes the method when a tracer exists and does nothing for `trace=None`. Layers use this adapter.

## Query without ambiguity

`get(step_id: str, layer: int | None = None, head: int | None = None) -> StepRecord` returns the **first** matching snapshot. Omitting a layer does not mean the last layer; omitting a head returns the stored array. Reusing a tracer appends records across calls, so the first value remains the oldest. Prefer one tracer per forward or inspect `records` directly for a timeline.

A full head snapshot has shape `(H,n,...)`. `get(..., head=h)` selects the leading axis and returns a new StepRecord with `head=h`, keeping its `order` and equation. That slice may share memory with the full snapshot; do not modify it if the observation must be preserved. A record stored with explicit `head=h` is already sliced and matches only that selector.

Negative, boolean or noninteger selectors and out-of-range heads raise `ValueError`; missing records raise `KeyError`. Selecting a head for an ID without a head axis also raises `ValueError`. `step_ids() -> list[str]` returns distinct IDs in order of first occurrence, without repetitions across blocks.

```python
import numpy as np
from glassllm import Config, GlassLM

model = GlassLM(Config(vocab_size=8, d_model=8, n_layers=1,
                      n_heads=2, n_kv_heads=1, max_len=4))
step = next(model.generate([1, 2, 3], 1, greedy=True, trace=True))
assert step.trace is not None
attention = step.trace.get("attn.softmax", layer=0, head=1)
assert attention.shape == (3, 3)
assert attention.head == 1
np.testing.assert_allclose(attention.value.sum(-1), 1.0, rtol=1e-5)
assert len(step.trace.records) == 23
```

## StepRecord and StepInfo

`StepRecord(id: str, layer: int | None, value: np.ndarray, order: int, head: int | None = None)` is a frozen dataclass. `shape` returns `value.shape`; `equation` reads the plain text in `STEPS[id]`; `latex` reads its LaTeX formula. The class does not make `value` immutable. `StepInfo(latex: str, plain: str)`, from `glassllm.steps`, is the frozen dataclass storing both equation formats.

## The 25 IDs and repeated events

`STEPS` contains 25 IDs. `LAYER_STEPS` contains the 18 operations executed per block; `HEAD_STEPS` contains the 9 that retain a head axis. Equations are identical in PT/EN; prose and labels live on the site.

| Step | Recorded shape, without batch | Head |
| --- | --- | --- |
| `embed`, `norm.final` | `(n,d)` | No |
| `norm.attn`, `resid.attn`, `norm.ffn`, `resid.ffn` | `(n,d)` per block | No |
| `attn.q`, `attn.rope_q` | `(Hq,n,dk)` | Q |
| `attn.k`, `attn.v`, `attn.rope_k` | `(Hkv,n,dk)` for new positions | K/V |
| `attn.scores`, `attn.mask`, `attn.softmax` | `(Hq,n,m)` including cached keys | Q |
| `attn.mix` | `(Hq,n,dk)` | Q |
| `attn.out`, `ffn.down` | `(n,d)` per block | No |
| `ffn.gate`, `ffn.up`, `ffn.act` | `(n,int(8*d/3))` per block | No |
| `logits` | `(n,V)` | No |
| `probs` | `(V,)` | No |
| `sample`, `loss` | Scalar | No |
| `loss.grad` | `(n,V)` | No |

For `L` blocks, forward records `18*L + 3` events. Generation adds `probs` and `sample`: 4 blocks yield **77 events**, although only 23 distinct IDs execute. Loss adds `loss` and `loss.grad` to the 75 forward events, also yielding 77, with another set of 23 IDs. Generation never includes loss.

## Equation and source line

`scan_markers(package_dir: Path = PACKAGE_DIR) -> dict[str,dict]` scans Python comments `# @step <id>` in immediate `.py` modules and returns `{id: {"file": "glassllm/x.py", "line": n}}`, with one-based lines. Duplicate markers raise `ValueError`; lookalike text inside strings does not count. `load_step_index() -> dict[str,dict]` reads the packaged `step_index.json`; it does not automatically recompute line numbers.

```python
from glassllm import STEPS
from glassllm.steps import HEAD_STEPS, LAYER_STEPS, load_step_index, scan_markers

assert len(STEPS) == 25
assert len(LAYER_STEPS) == 18
assert len(HEAD_STEPS) == 9
assert scan_markers() == load_step_index()
assert load_step_index()["attn.softmax"]["file"] == "glassllm/nn.py"
```

After moving engine lines, run `python scripts/build_step_index.py`. The index identifies the operation's source location, not a call-specific stack trace. The [interface inspector](../../interface/inspetor/) displays samples with non-finite values such as `−∞`; they are actual causal-mask values.
