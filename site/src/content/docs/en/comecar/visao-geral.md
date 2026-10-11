---
title: "What glassllm is"
description: "Meet the NumPy Transformer, the visual lab and the actual calculations connecting a prompt to its next prediction."
---

## A prompt, a prediction, visible calculations

glassllm is a small Transformer written in Python and NumPy to study how a language model works. You supply the beginning of a sentence; the model computes a distribution over its vocabulary, chooses an ID and appends that token to the context. Repeating this process produces a continuation.

The lab displays the same engine used by the Python library. The matrices, probabilities and diagram steps come from running the model on your prompt. They are actual values. Each marked calculation has an equation, a snapshot of its result and the location of the expression in the source code.

## The three parts of the project

| Part | What it does | How to explore it |
| --- | --- | --- |
| `glassllm` library | BPE tokenization, autograd, Transformer, generation, tracing and training | Python and CLI |
| PT and EN models | Learned weights and a separate tokenizer for each language | `load_pretrained("pt")` or `load_pretrained("en")` |
| Lab | Runs the Python wheel in a Pyodide worker and presents the results | Editor, attention, candidates and architecture |

In the browser, the sequence is text → BPE IDs → embedding → four blocks → final normalization → logits → probabilities → selected token. The selected ID returns directly to the context without being tokenized again. On first use, the browser downloads the runtime, NumPy and wheel; calculations for your prompt run in the local worker without calling an inference service.

## The model you are observing

The bundled assets use a vocabulary of 1,024 tokens, 128-dimensional vectors, four blocks, four Q heads and two K/V heads. Each head has 32 dimensions. The context window holds 128 tokens and the SwiGLU FFN has a hidden dimension of 341. The embedding matrix also projects the output, for a total of **852,608 learned parameters**.

This is a causal decoder: each position can observe only the positions permitted by its mask. It uses RMSNorm before attention and the FFN, RoPE on Q and K, GQA and a K/V cache. The library exposes these operations individually; the map brings their relationships together.

The models are small and trained on literary texts. They may repeat words, lose coherence and generate special tokens with no visible text. They are not instruction assistants or tools for factual answers. Their purpose is to make calculations observable and let you test hypotheses about the mechanism.

## Choose your first path

To explore without programming, open the [lab](../../../), select **Most likely** and use **Step once**. Compare the selected token with the candidate panel, then open the architecture. The [first experiment](../primeiro-experimento/) walks through this sequence and reproduces the same prediction in Python.

To use the engine directly, follow [local installation](../instalacao/) and the [library reference](../../biblioteca/api-publica/). To learn to read the diagram, start with the [component map](../../interface/mapa-componentes/) and [attention foundations](../../fundamentos/atencao-causal/).

## What a record represents

A prediction with four blocks records 77 events: embedding; 18 operations in each block; final normalization; logits; probabilities; selection. This does not mean 77 independent functions or 77 events per head. An attention record can store all heads and let you select one afterwards.

The lab retains the trace of the most recent prediction. [Slow motion](../../interface/slow-motion/) walks through snapshots that have already been computed, selecting the corresponding equation, values and node. It provides reading time, but does not change the NumPy calculations or keep a history of every prediction.
