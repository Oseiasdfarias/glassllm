---
title: "Embedding e pesos compartilhados"
description: "A consulta de vetores por ID e o uso da mesma matriz na projeção dos logits."
---

## Um ID escolhe uma linha

Os IDs do tokenizer são índices, não medidas: o token 200 não é semanticamente o dobro do token 100. A matriz de embedding `E` associa cada entrada a um vetor aprendido. No modelo padrão, suas dimensões são `1.024 × 128` e ela possui 131.072 parâmetros.

```text
ids: (B, n)          E: (V, d)
X = E[ids]           X: (B, n, d)

B = tamanho do batch; n = posições recebidas
V = 1024; d = 128
```

Uma representação one-hot permitiria escrever `X = one_hot(ids) @ E`, mas construir essa matriz seria desnecessário. `embedding(weight, ids)` usa indexação direta de `Tensor`. A função está em `src/glassllm/autograd.py`; a chamada está em `GlassLM.forward`, no snapshot `embed`.

## Exemplo com três tokens

Uma matriz pequena torna a consulta e seu gradiente visíveis:

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

Saída verificada:

```text
Vetores consultados: [[1, 0], [1, 1], [1, 0]]
Gradiente de E:      [[2, 2], [0, 0], [1, 1]]
Logits para h=[2,1]: [[2, 1, 3]]
```

O ID 0 apareceu duas vezes, então sua linha recebeu duas contribuições. A linha 1 não foi consultada e recebeu zero nesse exemplo. O backward da indexação usa `np.add.at`: índices repetidos somam gradientes corretamente.

## A mesma matriz na saída

Depois dos quatro blocos e da RMSNorm final, cada posição tem um vetor `H`. `GlassLM.forward` calcula:

```text
H: (B, n, 128)
logits = H @ E.T
logits: (B, n, 1024)
```

O logit de um token é o produto escalar entre `H` e sua linha de `E`. O exemplo anterior mostra por que o terceiro token obtém logit 3: `[2,1] · [1,1] = 3`. Um logit é uma pontuação real; ainda não é uma probabilidade.

Esse compartilhamento, chamado *weight tying*, evita uma segunda matriz de 131.072 parâmetros. `E` é um único `Tensor`, reaproveitado na entrada e através de `transpose(self.E, (1, 0))` na saída. Não há cópia treinável independente nem bias na projeção dos logits.

No treinamento, o gradiente que chega pelo lookup se soma ao gradiente que chega pela projeção de saída. A linha de um token ausente da entrada ainda pode receber gradiente pela saída, porque a entropia cruzada considera todo o vocabulário. Os blocos, as normalizações e `E` totalizam **852.608 parâmetros** no modelo padrão.

## O que observar no laboratório

Selecione `embed` no mapa para ler `(n, 128)` no snapshot: o `Tracer` remove o eixo de batch quando `B=1`. O inspetor mostra um recorte dos valores, não o vetor inteiro quando há mais colunas que o limite de exibição. Selecione `logits` para acompanhar `(n, 1024)`. A ligação entre entrada e saída no grafo indica o compartilhamento de `E`.

No segundo token gerado com cache, o novo snapshot `embed` pode ter somente uma posição. O contexto antigo permanece em K/V; ele não reaparece como novos embeddings nessa passada. Veja [GQA e cache](../gqa-cache/) e [logits e geração](../saida-geracao/).

## Interpretação e referência

As coordenadas de um embedding não possuem rótulos humanos fixos. A geometria resulta do treinamento; um produto escalar alto não prova uma relação linguística específica. O modelo também não aplica uma camada de posição diretamente a `E`: a posição entra por [RoPE](../rope/) em Q e K.

Press e Wolf estudam o compartilhamento entre embeddings de entrada e saída em [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859), primeiro preprint de 2016. O projeto implementa esse compartilhamento, sem reproduzir todos os experimentos ou propostas de regularização do artigo.
