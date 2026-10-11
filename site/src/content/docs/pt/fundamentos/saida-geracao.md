---
title: "Logits, temperatura e geração"
description: "Como o vetor final vira uma distribuição de 1.024 candidatos e como a escolha retorna ao contexto."
---

## A última posição prevê o próximo token

Depois dos quatro blocos, `GlassLM.forward` aplica a RMSNorm final e calcula `logits = H @ E.T`. A saída tem shape `(B,n,1024)`: cada posição contém uma pontuação para cada token do vocabulário. No treinamento, cada posição prevê seu sucessor; na geração, somente os logits da **última** posição alimentam a próxima escolha.

```text
Contexto → embedding → 4 blocos → RMSNorm final → logits da última posição
    ↑                                                        ↓
    └────────── acrescentar token escolhido ← escolha ← softmax(logits/T)
```

Os logits não precisam ser positivos nem somar 1. A normalização em probabilidades ocorre em `next_token_probs`, de `src/glassllm/model.py`, usando `softmax_np` de `src/glassllm/autograd.py`.

## Temperatura muda a distribuição

```text
p_i = exp(z_i/T) / Σ_j exp(z_j/T)
p_i / p_j = exp((z_i−z_j)/T)
```

Para temperatura positiva, diminuir T acentua a diferença entre scores; aumentar T espalha a distribuição. A implementação subtrai o maior logit antes de exponenciar. Para logits finitos, o limite `T→∞` é uniforme. Quando `T→0+`, a massa concentra-se nos máximos; se houver vários máximos iguais, o limite divide a massa entre eles.

O motor exige `temperature > 0`, inclusive com `greedy=True`. Para escolha determinística, use esse argumento em vez de passar temperatura zero.

## Exemplo com três candidatos

```python
import numpy as np
from glassllm.model import next_token_probs

logits = np.array([2., 1., 0.])
for temperature in (0.5, 1., 2.):
    print(temperature, np.round(next_token_probs(logits, temperature), 6))
```

Saída verificada:

| Temperatura | Token 0 | Token 1 | Token 2 |
| --- | --- | --- | --- |
| 0,5 | 0,866813 | 0,117310 | 0,015876 |
| 1,0 | 0,665241 | 0,244728 | 0,090031 |
| 2,0 | 0,506480 | 0,307196 | 0,186324 |

Com T=1, a razão entre os dois primeiros candidatos é `exp(1) ≈ 2,71828`. Nenhuma probabilidade representa uma garantia de correção factual: são previsões de continuação do pequeno modelo treinado.

## Sorteio, argmax e semente

`GlassLM.generate(ids, n_new, temperature=1.0, greedy=False, seed=None, trace=False)` é um iterador de `GenStep`.

| Campo de GenStep | Conteúdo |
| --- | --- |
| `token` | ID inteiro escolhido |
| `probs` | Array NumPy com 1.024 probabilidades na configuração padrão |
| `trace` | `Tracer` novo para esta previsão, ou `None` |

Com `greedy=False`, `rng.choice` sorteia o ID usando a distribuição. Uma semente fixa reproduz a sequência aleatória para a mesma chamada, pesos, entrada e configuração numérica. Com `greedy=True`, `np.argmax` seleciona o máximo; em empate, o primeiro ID. Temperaturas positivas preservam a ordem dos scores na expressão matemática, portanto o controle não altera a escolha greedy em condições numéricas usuais, embora altere as probabilidades exibidas.

## Duas previsões reais

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

Com os pesos distribuídos no projeto, a execução conferida produziu:

```text
[136, 118]
 que a
[(4, 128), (1, 128)]
[(4, 4), (1, 5)]
```

O primeiro passe processou quatro tokens do prompt. O segundo processou apenas o token 136, mas consultou cinco posições pelo cache. Esses valores podem mudar se os assets do modelo forem substituídos; as relações entre os shapes permanecem.

## Parada, contexto e snapshots

O iterador gera até `n_new` tokens e não para automaticamente em EOS. Tokens especiais fazem parte do vocabulário; `decode` omite PAD/BOS/EOS e mostra UNK como `⁇`. Assim, nem todo token escolhido acrescenta texto visível.

O prompt deve conter pelo menos um ID. `generate` mantém somente seus últimos `max_len` tokens e reconstrói o cache quando a janela enche. O limite de 128 tokens **novos por rodada** da interface é uma política do laboratório, distinta da janela de 128 posições do motor.

Com `trace=True`, cada previsão registra 77 eventos na configuração de quatro camadas: 75 do forward, mais `probs` e `sample`. `logits` contém todas as posições processadas nessa passada; `probs` contém somente a distribuição usada para a escolha; `sample` é um escalar. O [slow motion](../../interface/slow-motion/) reproduz esses snapshots reais, depois que os cálculos da previsão já ocorreram.

## Referências

O decoder autorregressivo usa a organização de atenção apresentada em [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al., 2017. O compartilhamento de `E` com a saída segue o conceito estudado em [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859), Press e Wolf, primeiro preprint de 2016. Veja os contratos completos em [GlassLM e GenStep](../../biblioteca/glasslm/).
