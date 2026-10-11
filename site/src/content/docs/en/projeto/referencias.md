---
title: Papers and reference map
description: Primary sources, their connection to the code and differences between the papers and this educational Transformer.
---

## Using this bibliography

The papers below explain concepts used by the engine. Each record includes a primary link, full authors, first arXiv submission year, related pages and a preprint BibTeX citation. Publication notes distinguish venues and journals when listed by the primary record. Adam, for example, has a 2014 preprint and an ICLR 2015 publication; citing the preprint year does not change the publication date.

Use related-page links to move from an idea to its executable operation. The bibliography identifies conceptual provenance; it does not mean the project reproduces every paper's architecture, dataset or experimental results.

## Mapping papers to code

| Source | Used concept | Code location |
| --- | --- | --- |
| Attention Is All You Need | Scaled attention and Transformer structure | `nn.Attention`, `nn.Block`, `model.GlassLM` |
| Neural Machine Translation of Rare Words with Subword Units | Subword BPE merges | `tokenizer.Tokenizer.train`, `Tokenizer.encode` |
| SentencePiece | Tokenization and space representation context | `tokenizer.pre_tokenize`, `SPACE` |
| RoFormer | Positional rotations of queries and keys | `nn.rope_angles`, `autograd.rope` |
| GQA | Shared K/V heads | `nn.Attention`, `autograd.repeat_kv` |
| Root Mean Square Layer Normalization | Root mean square normalization | `nn.RMSNorm` |
| On Layer Normalization in the Transformer Architecture | Pre-norm placement | `nn.Block` |
| Deep Residual Learning for Image Recognition | Residual-addition context | `nn.Block`, `resid.attn` and `resid.ffn` |
| GLU Variants Improve Transformer | SwiGLU gated FFN | `nn.SwiGLU`, `autograd.silu` |
| Using the Output Embedding to Improve Language Models | Tied input/output embedding weights | `GlassLM.E`, `GlassLM.forward` |
| Automatic differentiation in machine learning: a survey | Reverse mode and chain rule | `autograd.Tensor.backward` |
| Adam: A Method for Stochastic Optimization | Moments and adaptive updates | `optim.AdamW.step` |
| Decoupled Weight Decay Regularization | Decoupled weight decay | `optim.AdamW.step` |
| SGDR: Stochastic Gradient Descent with Warm Restarts | Cosine part of the schedule | `optim.lr_schedule` |

Module prefixes in this table are relative to `glassllm`. The [public API reference](../../biblioteca/api-publica/) distinguishes root exports from submodule functions.

## A custom combination of architecture choices

The original [Vaswani et al. Transformer](https://arxiv.org/abs/1706.03762) uses an encoder and decoder for sequence transduction. glassllm implements a small causal decoder for next-token prediction, with no encoder or cross-attention. It uses RoPE, RMSNorm, GQA and SwiGLU, choices introduced or studied in different papers.

GQA shares two K/V heads across four query heads. The project does not run the checkpoint conversion and continued-training procedure from the [GQA paper](https://arxiv.org/abs/2305.13245). Its tokenizer implements BPE over normalized text; SentencePiece is contextual background rather than a dependency or a format-compatibility guarantee.

## Normalization and residuals

The [pre-norm paper](https://arxiv.org/abs/2002.04745) analyzes LayerNorm; this code uses RMSNorm before both sublayers and at the output. The relationship informs normalization placement, without automatically transferring all theoretical conclusions between distinct operations.

The [RMSNorm paper](https://arxiv.org/abs/1910.07467) also discusses pRMSNorm; this project computes the mean square of all dimensions. With `ε > 0`, scale invariance is not an exact equality for every vector. The [residual-network paper](https://arxiv.org/abs/1512.03385) supplies historical vision context rather than proving that this model keeps arbitrary gradients constant.

## Training and optimization

AdamW uses gradient moments and decay separate from adaptive updates. The project's schedule combines warmup with cosine decay down to a fraction of the initial rate. It **does not implement SGDR warm restarts**, and its optimizer is AdamW rather than the SGD evaluated in that paper.

Autograd is implemented locally over NumPy arrays. The [automatic-differentiation survey](https://arxiv.org/abs/1502.05767) provides terminology and a methodological overview; accumulation, broadcasting and traversal details belong to the implementation described by the [Tensor reference](../../biblioteca/tensor-autograd/).

## Citing and checking sources

Open each arXiv record to inspect versions, PDF and publication details. This page's BibTeX entries consistently cite preprints with authors, title, year, eprint, arXiv DOI and URL. The downloadable file contains all 14 records. For academic work requiring the published version, use the corresponding proceedings or journal reference and identify the version consulted.

Foundation equations also state the actual code's shapes and choices. Start with [causal attention](../../fundamentos/atencao-causal/) and [loss and gradients](../../fundamentos/perda-gradientes/) to connect a paper to an executed numerical calculation.
