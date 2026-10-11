---
title: "Logits, temperature and generation"
description: "How the final vector becomes a distribution over 1,024 candidates and how the selected token returns to context."
---

## The last position predicts the next token

After four blocks, `GlassLM.forward` applies the final RMSNorm and computes `logits = H @ E.T`. The output has shape `(B,n,1024)`: each position contains one score per vocabulary token. During training, each position predicts its successor. During generation, only the **last** position's logits feed the next selection.

```text
Context → embedding → 4 blocks → final RMSNorm → last-position logits
   ↑                                                    ↓
   └──────── append selected token ← selection ← softmax(logits/T)
```

Logits need not be positive or sum to 1. `next_token_probs` in `src/glassllm/model.py` normalizes them into probabilities using `softmax_np` from `src/glassllm/autograd.py`.

## Temperature changes the distribution

```text
p_i = exp(z_i/T) / Σ_j exp(z_j/T)
p_i / p_j = exp((z_i−z_j)/T)
```

For positive temperature, decreasing T sharpens score differences and increasing T spreads the distribution. The implementation subtracts the maximum logit before exponentiating. For finite logits, the limit `T→∞` is uniform. As `T→0+`, mass concentrates on the maxima; with tied maxima, the limit shares mass between them.

The engine requires `temperature > 0`, including with `greedy=True`. For deterministic selection, use that argument rather than temperature zero.

## Three-candidate example

```python
import numpy as np
from glassllm.model import next_token_probs

logits = np.array([2., 1., 0.])
for temperature in (0.5, 1., 2.):
    print(temperature, np.round(next_token_probs(logits, temperature), 6))
```

Verified output:

| Temperature | Token 0 | Token 1 | Token 2 |
| --- | --- | --- | --- |
| 0.5 | 0.866813 | 0.117310 | 0.015876 |
| 1.0 | 0.665241 | 0.244728 | 0.090031 |
| 2.0 | 0.506480 | 0.307196 | 0.186324 |

At T=1, the ratio between the first two candidates is `exp(1) ≈ 2.71828`. These probabilities do not guarantee factual correctness: they predict continuations from a small trained model.

## Sampling, argmax and seeds

`GlassLM.generate(ids, n_new, temperature=1.0, greedy=False, seed=None, trace=False)` yields `GenStep` instances.

| GenStep field | Contents |
| --- | --- |
| `token` | Selected integer ID |
| `probs` | NumPy array of 1,024 probabilities in the default configuration |
| `trace` | A fresh `Tracer` for this prediction, or `None` |

With `greedy=False`, `rng.choice` samples an ID from the distribution. A fixed seed reproduces random choices for the same call, weights, input and numerical configuration. With `greedy=True`, `np.argmax` selects the maximum and chooses the first ID in a tie. Positive temperatures preserve score ordering mathematically, so temperature does not change greedy selection under usual numerical conditions, although it changes displayed probabilities.

## Two actual predictions

```python
from glassllm import load_pretrained

model, tokenizer = load_pretrained("pt")
steps = list(model.generate(tokenizer.encode("Era uma vez"),
                            n_new=2, greedy=True, trace=True))
print([s.token for s in steps])
print(tokenizer.decode([s.token for s in steps]))
print([s.trace.get("embed").shape for s in steps])
print([s.trace.get("attn.softmax", layer=3, head=0).shape for s in steps])
```

Using the project's packaged weights, the verified run produced:

```text
[136, 118]
 que a
[(4, 128), (1, 128)]
[(4, 4), (1, 5)]
```

The first pass processed four prompt tokens. The second processed only token 136 but queried five positions through the cache. Exact token choices may change if model assets are replaced; these shape relationships remain.

## Stopping, context and snapshots

The iterator generates up to `n_new` tokens and does not automatically stop at EOS. Special tokens belong to the vocabulary: `decode` omits PAD/BOS/EOS and renders UNK as `⁇`. Not every selected token therefore adds visible text.

The prompt must contain at least one ID. `generate` retains only its last `max_len` tokens and rebuilds the cache when the window fills. The interface's limit of 128 **new tokens per run** is a lab policy, distinct from the engine's 128-position context window.

With `trace=True`, each prediction records 77 events in the four-layer configuration: 75 from forward, followed by `probs` and `sample`. `logits` contains all positions processed in that pass; `probs` contains only the distribution used for selection; `sample` is a scalar. [Slow motion](../../interface/slow-motion/) replays these actual snapshots after the prediction's calculations have completed.

## References

The autoregressive decoder uses attention organization presented in [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al., 2017. Sharing `E` with output follows the concept studied in [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859), Press and Wolf, first preprint in 2016. See complete contracts in [GlassLM and GenStep](../../biblioteca/glasslm/).
