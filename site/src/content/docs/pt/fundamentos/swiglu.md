---
title: "FFN e SwiGLU"
description: "Dois ramos de projeção, a ativação SiLU e a volta das 341 coordenadas internas para 128."
---

## Transformar cada token separadamente

A atenção mistura posições do contexto. A rede feed-forward (FFN) transforma cada vetor de posição sem consultar outros tokens diretamente. Ela recebe informação contextual que já passou pela atenção, mas sua operação é a mesma em cada posição.

`SwiGLU` em `src/glassllm/nn.py` possui três matrizes aprendidas: `W1` para o ramo *gate*, `W3` para o ramo *up* e `W2` para a projeção *down*. Não há bias nessas projeções.

```text
                 ┌─ @W1 → gate → SiLU ─┐
X normalizado ───┤                     ⊙ → @W2 → saída
                 └─ @W3 → up ──────────┘
```

Os ramos são multiplicados coordenada a coordenada, não por produto matricial. A ativação do gate modula o conteúdo de up. Esse gate não é uma probabilidade: `SiLU(z)` pode ser negativa e não fica limitada ao intervalo `[0,1]`.

## Equações e dimensões

```text
G = X @ W1
U = X @ W3
SiLU(z) = z · sigmoid(z)
A = SiLU(G) ⊙ U
Y = A @ W2
```

| Valor | Shape padrão |
| --- | --- |
| X normalizado | `(B,n,128)` |
| W1 e W3 | `(128,341)` |
| Gate, up e ativação | `(B,n,341)` |
| W2 | `(341,128)` |
| Saída Y | `(B,n,128)` |

O código calcula `hidden = int(8*d/3)`: com `d=128`, o valor é **341**, não 342 nem 512. As três matrizes somam `3 × 128 × 341 = 130.944` parâmetros por bloco. A regra aproxima o orçamento de `8d²` de uma FFN tradicional com largura `4d`, usando arredondamento para inteiro.

## Exemplo verificável da ativação

O exemplo abaixo começa após as projeções para focar no produto dos dois ramos:

```python
import numpy as np
from glassllm import Tensor
from glassllm.autograd import silu

gate = Tensor([-1., 0., 1.], requires_grad=True)
up = Tensor([2., 3., 4.], requires_grad=True)
act = silu(gate) * up
print(np.round(act.data, 6))
print(np.round(act.data @ np.ones((3, 1)), 6))
act.sum().backward()
print(np.round(gate.grad, 6))
print(np.round(up.grad, 6))
```

Saída verificada:

```text
Ativação:            [-0.537883, 0.000000, 2.924234]
Down com pesos 1:    [2.386351]
Gradiente do gate:   [0.144659, 1.500000, 3.710682]
Gradiente de up:     [-0.268941, 0.000000, 0.731059]
```

Mesmo quando `gate=0`, a derivada em relação ao gate não é zero: `SiLU'(0)=0,5`, multiplicada por `up=3`, resulta em 1,5. Já o gradiente em relação a up é zero nessa coordenada porque `SiLU(0)=0`.

## Como o gradiente percorre os ramos

Se `D` é o gradiente recebido por A:

```text
∂L/∂U = D ⊙ SiLU(G)
∂L/∂G = D ⊙ U ⊙ sigmoid(G) ⊙ (1 + G ⊙ (1−sigmoid(G)))
```

O backward das projeções segue a regra de `Y=XW`: o gradiente de X usa `grad_Y @ W.T`, e o de W soma os produtos entre entradas e gradientes de todas as posições e do batch. As contribuições dos ramos W1 e W3 voltam e se somam na entrada comum. O caminho residual oferece ainda uma contribuição separada.

`silu` em `src/glassllm/autograd.py` usa uma forma estável do sigmoid baseada em `exp(−abs(x))`. A derivada é implementada diretamente, enquanto as multiplicações e projeções usam as regras gerais do autograd.

## Snapshots, limites e referência

Na ordem de reprodução: `norm.ffn` apresenta a entrada normalizada; `ffn.gate` e `ffn.up` mostram os dois ramos; `ffn.act` mostra **SiLU(gate) vezes up**, e não SiLU isolada; `ffn.down` mostra a projeção para 128; `resid.ffn` mostra sua soma com o fluxo residual. O `Tracer` apresenta shapes sem batch.

Os valores negativos são válidos. Uma coordenada alta não identifica por si só um conceito ou palavra. O modelo não inclui dropout nessa FFN. Shazeer apresenta variantes com portões em [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202), 2020; o projeto implementa a variante SwiGLU com a largura definida acima. Continue em [normalização e resíduos](../normalizacao-residuos/) ou [saída e geração](../saida-geracao/).
