---
title: "First experiment: site and Python"
description: "Compare an actual prediction in the lab and library and inspect attention, probabilities and the causal mask."
---

## Produce a prediction you can compare

Use the same prompt, language and deterministic selection in both environments. The English lab opens with `Alice was beginning to`; the Portuguese lab opens with `Capitu olhou para`. Each model has a different tokenizer, so do not reuse IDs from one language in the other.

1. Open the [lab](../../../) and keep the initial prompt.
2. Under **Token selection**, choose **Most likely**. Keep temperature at `0.8`.
3. Click **Step once** and wait for initial loading.
4. Read **Latest token**, the continuation and the **BPE tokens** boxes.
5. Compare **Attention** with **Next possibilities**. One panel describes the context; the other describes the output.

With the bundled assets, the English prompt produces five IDs: `494, 160, 729, 633, 111`, corresponding to `Alice`, `▁was`, `▁begin`, `ning`, `▁to`. The next greedy choice is ID `97`, displayed as `▁the` and decoded as ` the`. These results were verified against the checkout's weights; retraining or replacing the assets can change them.

## Repeat the prediction in Python

After [installation](../instalacao/), save and run this code with `uv run python file.py`:

```python
import numpy as np
from glassllm import load_pretrained

model, tokenizer = load_pretrained("en")
ids = tokenizer.encode("Alice was beginning to")
step = next(model.generate(
    ids, n_new=1, temperature=0.8, greedy=True, trace=True
))

print(ids)
print(tokenizer.tokens(ids))
print(step.token, repr(tokenizer.decode([step.token])))
print(len(step.trace.records))
assert np.isclose(step.probs.sum(), 1.0)

attention = step.trace.get("attn.softmax", layer=3, head=0)
print(attention.shape)
print(attention.value[-1])
assert np.allclose(attention.value.sum(axis=-1), 1.0)
```

The `GenStep` contains the selected ID, the full output distribution and the prediction's trace. For this prompt, there are 77 records and one head's attention has shape `(5, 5)`. `layer=3` means the fourth block; `head=0` means the first head. The lab's compact panel uses the last row of this matrix.

To repeat in Portuguese, load `"pt"` and use `"Capitu olhou para"`. The bundled assets produce six input tokens and greedy next-token ID `132`, decoded as ` o`. Do not compare the numerical meaning of IDs across vocabularies.

## Check a calculation in the diagram

Open **Explore the network**, go to **Block 1** and select **Causal mask**. The inspector shows the equation, dimensions, values and line applying the mask. In the first prediction, the entire prefix is computed; future positions in the matrix receive `−∞`.

You can inspect the same tensor without starting generation:

```python
from glassllm import Tracer

tracer = Tracer()
model.forward(ids, trace=tracer)
mask = tracer.get("attn.mask", layer=0, head=0)
print(mask.equation)
print(mask.value)
```

`forward` records logits and internal operations, but not selection or the generation distribution: those two steps belong to `generate`. In the English example, one head's mask contains 10 `−∞` positions above the diagonal. After softmax, those positions receive zero weight.

## Study without losing the current prediction

Enable **Slow motion** and click **Replay flow**. Use **Pause flow** at an interesting operation; **Previous calculation** and **Next calculation** walk through the same records. You do not need a new prediction to revisit the tensor.

When you generate another token, panels describe the new prediction. With the cache, queries usually have only one position and the attention matrix becomes `1 × m`. For a clean experiment with another prompt or selection method, click **Reset**, edit the input and start again. Continue with [slow motion](../../interface/slow-motion/) and the [inspector](../../interface/inspetor/).
