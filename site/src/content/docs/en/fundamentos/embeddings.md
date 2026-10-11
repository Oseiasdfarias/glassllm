---
title: "Embedding and shared weights"
description: "Looking up vectors by token ID and reusing the same matrix to produce logits."
---

## An ID selects a row

Tokenizer IDs are indices, not measurements: token 200 is not semantically twice token 100. The embedding matrix `E` associates each vocabulary entry with a learned vector. In the default model it has shape `1,024 × 128` and contains 131,072 parameters.

```text
ids: (B, n)          E: (V, d)
X = E[ids]           X: (B, n, d)

B = batch size; n = input positions
V = 1024; d = 128
```

A one-hot representation would let us write `X = one_hot(ids) @ E`, but constructing that matrix is unnecessary. `embedding(weight, ids)` directly indexes a `Tensor`. The function lives in `src/glassllm/autograd.py`; `GlassLM.forward` calls it at the `embed` snapshot.

## A three-token example

A small matrix makes the lookup and its gradient visible:

```python
import numpy as np
from glassllm import Tensor
from glassllm.autograd import embedding

E = Tensor([[1., 0.], [0., 1.], [1., 1.]], requires_grad=True)
x = embedding(E, [0, 2, 0])
print(x.data)
x.sum().backward()
print(E.grad)
print(np.array([[2., 1.]]) @ E.data.T)
```

Verified output:

```text
Looked-up vectors:  [[1, 0], [1, 1], [1, 0]]
Gradient of E:      [[2, 2], [0, 0], [1, 1]]
Logits for h=[2,1]: [[2, 1, 3]]
```

ID 0 occurs twice, so its row receives two contributions. Row 1 is never looked up and receives zero in this example. Indexing's backward pass uses `np.add.at` to add gradients correctly for repeated indices.

## The same matrix at output

After four blocks and the final RMSNorm, each position has a vector `H`. `GlassLM.forward` computes:

```text
H: (B, n, 128)
logits = H @ E.T
logits: (B, n, 1024)
```

A token's logit is the dot product between `H` and its row of `E`. In the example, the third token gets logit 3 because `[2,1] · [1,1] = 3`. A logit is a real-valued score, not yet a probability.

This sharing, called *weight tying*, avoids a second matrix of 131,072 parameters. `E` is a single `Tensor`, reused at input and through `transpose(self.E, (1, 0))` at output. There is no separately trainable copy and no output projection bias.

During training, gradients from the lookup add to gradients from the output projection. A token absent from the input may still receive an output-side gradient because cross-entropy covers the entire vocabulary. Together, the blocks, normalization layers and `E` contain **852,608 parameters** in the default model.

## What to inspect in the lab

Select `embed` in the map to inspect shape `(n, 128)`: `Tracer` removes the batch axis when `B=1`. The inspector displays a numerical sample rather than all coordinates when the tensor exceeds the display limit. Select `logits` to follow shape `(n, 1024)`. The graph's input-to-output weight connection represents shared `E`.

On the second generated token with caching, the new `embed` snapshot may have only one position. Previous context stays in K/V and is not embedded again in that pass. See [GQA and caching](../gqa-cache/) and [logits and generation](../saida-geracao/).

## Interpretation and reference

Embedding coordinates do not have fixed human-readable meanings. Training determines their geometry; a high dot product does not prove a particular linguistic relationship. The model also adds no position layer directly to `E`: [RoPE](../rope/) introduces position through Q and K.

Press and Wolf study shared input and output embeddings in [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859), first preprint in 2016. This project implements weight tying without reproducing all of the paper's experiments or regularization proposals.
