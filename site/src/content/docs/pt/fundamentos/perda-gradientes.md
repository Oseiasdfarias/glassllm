---
title: "Perda e backpropagation"
description: "Targets deslocados, entropia cruzada média e o gradiente (p − y)/N no motor de modo reverso."
---

## Aprender a prever o sucessor

O treino apresenta uma sequência de entrada e uma sequência alvo deslocada um token para a direita. Para um trecho de IDs `[4,5,6,7]`, a entrada pode ser `[4,5,6]` e o alvo `[5,6,7]`. A máscara causal impede que uma posição consulte o sucessor que deve prever, embora todas as posições sejam processadas em paralelo.

`get_batch` em `src/glassllm/train.py` amostra janelas desse tipo. `GlassLM.loss(ids, targets)` recebe os dois arrays; não faz o deslocamento automaticamente. O forward produz logits `(B,n,V)` e a perda os reorganiza como `(N,V)`, com `N=B×n` e um target para cada linha.

## Entropia cruzada média

```text
p[t,:] = softmax(z[t,:])
L = −(1/N) · Σ_t log(p[t, target[t]])

Para uma posição com alvo c:
−log p_c = −z_c + log Σ_j exp(z_j)
```

O código usa uma expressão estável de log-softmax, subtraindo o maior logit antes da exponencial. O resultado de `cross_entropy(logits, targets)`, em `src/glassllm/autograd.py`, é um `Tensor` escalar. É uma média sobre todas as posições; não há mecanismo de ignorar PAD ou uma máscara de padding nessa função.

Quando a distribuição é exatamente uniforme, cada alvo tem probabilidade `1/V` e a perda é `ln(V)`. Para `V=1024`, isso vale aproximadamente **6,931472 nats**. Uma inicialização aleatória pode ficar perto dessa referência, mas não é necessariamente uniforme. A unidade nats decorre do logaritmo natural usado no código.

## De onde vem p menos y

Derivando a perda de uma posição em relação ao logit i:

```text
∂(−log p_c)/∂z_i = p_i − 1[i=c]
∂L/∂z[t,i] = (p[t,i] − 1[i=target[t]]) / N
```

`y` é o vetor one-hot do target. O gradiente no alvo é negativo; os demais são positivos. Em uma atualização por descida de gradiente, isso tende a elevar o logit do alvo e reduzir os concorrentes. A divisão por N é essencial porque a perda foi definida como média, não soma.

## Exemplo com dois alvos

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

Saída verificada:

```text
0.407606
[[-0.167380, 0.122364, 0.045015],
 [ 0.045015, 0.122364,-0.167380]]
```

Os alvos nas duas linhas têm probabilidade aproximadamente 0,665241. A perda é `−ln(0,665241) ≈ 0,407606`. O gradiente de cada linha soma aproximadamente zero, consequência de a distribuição softmax somar 1.

## Modo reverso no Tensor

O forward diferenciável constrói um grafo de operações: cada `Tensor` guarda seus pais e uma função que distribui gradientes para eles. `Tensor.backward()` percorre esse grafo em ordem topológica reversa, começando por gradiente 1 para uma perda escalar.

```text
ids → embedding → blocos → logits → perda
          ↑          ↑       ↑        │
          └──────────┴───────┴────────┘ backward
```

Em cada operação, a regra da cadeia transforma o gradiente de saída em contribuições para as entradas. Caminhos compartilhados se somam: isso inclui a matriz E usada na entrada e na saída, tokens repetidos no lookup e os dois ramos de SwiGLU. O backward também soma eixos introduzidos por broadcasting para devolver o shape correto aos parâmetros.

Os gradientes das folhas acumulam entre chamadas. O treino executa `opt.zero_grad()`, `loss.backward()`, clipping e `opt.step()`. Rodar apenas backward não muda os pesos. `no_grad()` evita construir o grafo; `GlassLM.generate` utiliza esse contexto porque geração não atualiza o modelo. Consulte [Tensor e autograd](../../biblioteca/tensor-autograd/) e [treino e otimização](../../biblioteca/treino-otimizacao/).

## Snapshots de treino e de inferência

`GlassLM.loss(..., trace=Tracer())` registra `loss`, escalar, e `loss.grad`, com shape `(n,1024)` após remover o batch. `loss.grad` é o gradiente calculado analiticamente em relação aos logits; seu registro não executa backward nem preenche todos os `parameter.grad`.

O tracer aceita somente batch 1 para acompanhar uma sequência. Um forward padrão tem 75 eventos; loss acrescenta dois e totaliza 77. A geração também totaliza 77, mas seus dois últimos eventos são `probs` e `sample`, não perda. O catálogo reúne 25 IDs distintos entre os caminhos de treino e geração. A interface atual mostra inferência; o painel de probabilidades e a reprodução lenta não constituem uma visualização completa do backward.

## Referência e alcance

Baydin et al. discutem diferenciação automática e modo reverso em [Automatic differentiation in machine learning: a survey](https://arxiv.org/abs/1502.05767), primeiro preprint de 2015 e publicação em JMLR em 2018. O motor do projeto é uma implementação pequena sobre NumPy, com regras explícitas para suas operações; não substitui todos os recursos de frameworks de treino maiores. Uma perda menor mede melhor previsão nos dados avaliados, não certifica compreensão ou veracidade do texto gerado.
