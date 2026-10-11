---
title: Glossary and notation
description: Terms and symbols used by the interface, equations and library signatures.
---

## Text and vocabulary

| Term | Meaning in glassllm |
| --- | --- |
| Token | A tokenizer unit: character, subword or special token, rather than necessarily a word |
| ID | Integer identifying an entry in a particular tokenizer vocabulary |
| Vocabulary | Mapping between token representations and IDs; each distributed model has 1024 entries |
| BPE | Byte Pair Encoding: this implementation learns frequent-pair merges over text symbols, rather than covering every byte |
| `▁` | Visible representation of a space within tokens |
| `↵` | Visible representation of a newline in the interface |
| UNK | Unknown token for symbols not represented by the vocabulary; decode uses the tokenizer's representation |
| BOS / EOS | Start/end special tokens included in the vocabulary; the API neither inserts them automatically nor stops on generated EOS |
| Prompt | Initial text supplied to generation |
| Context | IDs used to predict the next token, bounded by the model window |

An ID belongs to a particular vocabulary. Do not combine PT tokenizer IDs with the EN model. Tokenization also does not fix each vector's meaning; embeddings are learned parameters. See [tokens and BPE](../../fundamentos/tokens-bpe/).

## Dimensions and axes

| Symbol | Meaning | Default or relationship |
| --- | --- | --- |
| `B` | Batch sequence count | Tracing `GlassLM.forward` requires `B = 1` |
| `n` | New positions processed in this forward | Initial prefix, or 1 token when advancing with cache |
| `m` | Available key/value positions | Previous cache + new positions |
| `d` / `d_model` | Dimension of each token vector | 128 |
| `H_Q` | Query heads | 4 |
| `H_KV` | Key/value heads | 2 |
| `d_k` | Dimension per head | `d / H_Q = 32` |
| `d_ff` | FFN hidden dimension | `floor(8d/3) = 341` |
| `V` | Vocabulary size when used as a dimension | 1024 |
| `L` | Number of blocks when used as a count | 4 |
| `N` | Number of targets in mean loss | `B × n` |

A `B × n × d` array contains a `d`-dimensional vector for every position of each sequence. An `n × m` attention matrix relates query positions to available context positions. The tracer removes the batch axis; selecting a head removes its axis too. The inspector's shape can therefore have fewer axes than the variable inside a layer.

Vocabulary dimension `V` and attention value matrix `V` share a conventional letter but denote different objects. Similarly, `L` can denote block count while `𝓛` denotes loss; the equation's context identifies the meaning.

## Matrices, operations and parameters

| Symbol | Meaning |
| --- | --- |
| `E` | `V × d` embedding matrix, also transposed for output projection |
| `X`, `H` | Text-position representations inside the model |
| `Q`, `K`, `V` | Attention queries, keys and values |
| `W_Q`, `W_K`, `W_V`, `W_O` | Attention projection matrices |
| `S` | `QKᵀ/√d_k` compatibility scores before masking |
| `M` | Causal mask: 0 for allowed positions, `−∞` for future positions |
| `A` | Attention weights after softmax |
| `γ` | Learned RMSNorm scale, one per dimension |
| `ε` | RMSNorm numerical stabilizer, default `1e-6` |
| `T` | Positive temperature applied before output softmax |
| `ᵀ` | Transpose: swaps matrix axes |
| `⊙` | Elementwise multiplication |
| `@` / `·` | Matrix multiplication in code / equation notation |

A projection such as `Q = X @ W_Q` combines each vector's dimensions; attention combines positions through `A @ V`. The FFN applies its projections independently to each position. [Causal attention](../../fundamentos/atencao-causal/) explains the distinction.

## Output, learning and inspection

A **logit** is an unnormalized score for a candidate token. **Softmax** turns scores into probabilities summing to 1 along the chosen axis. **Attention** also uses softmax, but its distribution covers context positions rather than vocabulary tokens. **Greedy** selection uses `argmax`; **sampling** draws from the distribution.

**Loss** measures disagreement between prediction and target. A **gradient** is the derivative of loss with respect to a variable or parameter. **Backpropagation** applies the chain rule in reverse; **autograd** records operations and organizes that propagation. A **parameter** is a model `Tensor` receiving gradients during training; an **activation** is an intermediate value computed with those parameters.

A **K/V cache** stores rotated keys and previously computed values to avoid recomputing the whole prefix for every token. A **checkpoint** stores configuration and weights, rather than tokenizer or AdamW state. A **trace** is the sequence of execution snapshots. A **snapshot** is a numerical copy captured at a marked point. **Slow motion** is the site's visual replay of that sequence.

## Numerical precision and samples

`float16`, `float32` and `float64` use different bit counts to represent numbers. Loading a float16 checkpoint as float32 does not guarantee all intermediates remain float32. `−∞` is intentional in the causal mask; `NaN` indicates an undefined result. The inspector counts non-finite values and excludes them from finite statistics.

`itemsize` gives an array's bytes per element; `nbytes` gives the bytes occupied by its elements. Use them to measure parameters, cache and snapshots without assuming a shared dtype. Python object and runtime storage are not included in `nbytes`.

The last 8 × 8 entries are a reading sample, rather than the complete tensor. Minimum, mean and maximum refer to finite elements across the entire selected tensor. Visible head-selector numbering starts at 1, while Python indices and the compact attention caption start at 0. Consult [the inspector](../../interface/inspetor/) for labels and [the API](../../biblioteca/api-publica/) for call conventions.
