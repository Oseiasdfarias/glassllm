---
title: "Loss and backpropagation"
description: "Shifted targets, mean cross-entropy and the (p − y)/N gradient in the reverse-mode engine."
---

## Learning to predict the successor

Training presents an input sequence and a target sequence shifted one token to the right. For IDs `[4,5,6,7]`, the input may be `[4,5,6]` and the target `[5,6,7]`. The causal mask prevents a position from observing the successor it must predict, even though all positions are processed in parallel.

`get_batch` in `src/glassllm/train.py` samples these windows. `GlassLM.loss(ids, targets)` receives both arrays; it does not shift targets automatically. Forward produces logits `(B,n,V)`, which loss reshapes into `(N,V)`, with `N=B×n` and one target per row.

## Mean cross-entropy

```text
p[t,:] = softmax(z[t,:])
L = −(1/N) · Σ_t log(p[t, target[t]])

For one position with target c:
−log p_c = −z_c + log Σ_j exp(z_j)
```

The code uses a stable log-softmax expression, subtracting the maximum logit before exponentiation. `cross_entropy(logits, targets)` in `src/glassllm/autograd.py` returns a scalar `Tensor`. It averages all positions; this function has no ignore-PAD setting or padding mask.

With an exactly uniform distribution, each target has probability `1/V` and loss equals `ln(V)`. For `V=1024`, that is approximately **6.931472 nats**. Random initialization may be near this reference but is not necessarily uniform. The unit nats follows from the natural logarithm used in code.

## Where p minus y comes from

Differentiating one position's loss with respect to logit i gives:

```text
∂(−log p_c)/∂z_i = p_i − 1[i=c]
∂L/∂z[t,i] = (p[t,i] − 1[i=target[t]]) / N
```

`y` is the target's one-hot vector. The gradient is negative at the target and positive elsewhere. A gradient descent update therefore tends to raise the target logit and lower competitors. Dividing by N is essential because the loss is defined as a mean rather than a sum.

## Two-target example

```python
import numpy as np
from glassllm import Tensor
from glassllm.autograd import cross_entropy

logits = Tensor([[2., 1., 0.], [0., 1., 2.]], requires_grad=True)
loss = cross_entropy(logits, [0, 2])
loss.backward()
print(round(float(loss.data), 6))
print(np.round(logits.grad, 6))
```

Verified output:

```text
0.407606
[[-0.167380, 0.122364, 0.045015],
 [ 0.045015, 0.122364,-0.167380]]
```

Both targets have probability approximately 0.665241. The loss is `−ln(0.665241) ≈ 0.407606`. Each gradient row sums to approximately zero, consistent with the softmax distribution summing to 1.

## Reverse mode in Tensor

A differentiable forward pass builds an operation graph. Each `Tensor` stores its parents and a function that distributes gradients to them. `Tensor.backward()` visits this graph in reverse topological order, starting with gradient 1 for a scalar loss.

```text
ids → embedding → blocks → logits → loss
          ↑         ↑        ↑       │
          └─────────┴────────┴───────┘ backward
```

At each operation, the chain rule turns output gradients into input contributions. Shared paths add: this includes E reused at input and output, repeated lookup IDs and both SwiGLU branches. Backward also sums axes introduced by broadcasting to restore parameter shapes.

Leaf gradients accumulate between calls. Training performs `opt.zero_grad()`, `loss.backward()`, gradient clipping and `opt.step()`. Backward alone does not change weights. `no_grad()` avoids building the graph; `GlassLM.generate` uses this context because generation does not update the model. See [Tensor and autograd](../../biblioteca/tensor-autograd/) and [training and optimization](../../biblioteca/treino-otimizacao/).

## Training and inference snapshots

`GlassLM.loss(..., trace=Tracer())` records scalar `loss` and `loss.grad`, with shape `(n,1024)` after removing the batch axis. `loss.grad` is the analytically computed logit gradient. Recording it does not execute backward or populate every `parameter.grad`.

The tracer accepts batch 1 only, following a single sequence. A default forward has 75 events; loss adds two for a total of 77. Generation also has 77, but its last two events are `probs` and `sample`, not loss. The catalog combines 25 distinct IDs across training and generation paths. The current interface shows inference: its probability panel and slow replay do not visualize the complete backward pass.

## Reference and scope

Baydin et al. discuss automatic differentiation and reverse mode in [Automatic differentiation in machine learning: a survey](https://arxiv.org/abs/1502.05767), first preprint in 2015 and published in JMLR in 2018. The project's engine is a small NumPy implementation with explicit rules for its operations, rather than a replacement for every feature of larger training frameworks. Lower loss measures improved prediction on the evaluated data; it does not certify understanding or the truth of generated text.
