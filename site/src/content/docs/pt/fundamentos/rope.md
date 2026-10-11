---
title: "Posição e RoPE"
description: "Rotações de pares de coordenadas em Q/K, frequências, posições relativas e offset do cache."
---

## A posição entra pelas rotações

A atenção precisa distinguir a posição de cada token, além de seu conteúdo. O `glassllm` usa *Rotary Position Embedding* (RoPE): gira pares adjacentes de coordenadas de Q e K, com ângulos que dependem da posição. V permanece sem rotação, e não há uma tabela de embeddings posicionais somada à entrada.

Cada cabeça tem `dₖ=32`, portanto contém 16 pares. `rope_angles(d_k, max_len, base)` em `src/glassllm/nn.py` calcula as tabelas de seno e cosseno; `rope(x, cos, sin)` em `src/glassllm/autograd.py` aplica a transformação e sua derivada.

## Frequências e equação

Para o par de índice `i`, começando em zero:

```text
θ_i = 10000^(−2i/32)                i = 0, ..., 15
ângulo na posição p = p · θ_i

x'[2i]   = x[2i] · cos(pθ_i) − x[2i+1] · sin(pθ_i)
x'[2i+1] = x[2i] · sin(pθ_i) + x[2i+1] · cos(pθ_i)
```

O primeiro par tem frequência angular 1 radiano por posição; o segundo tem aproximadamente 0,562341; o último, 0,000177828. Os pares rápidos mudam mais entre posições próximas, enquanto os lentos variam em uma escala maior. As tabelas padrão possuem shape `(128,16)` e são buffers, não parâmetros treináveis.

Uma rotação mantém a norma de cada par, salvo arredondamento. Seu backward usa o ângulo oposto: a transposta de uma matriz de rotação é a rotação inversa. Por isso o código aplica `_rotate(g, cos, -sin)` ao gradiente.

## Por que aparecem posições relativas

Se `R_p` é a rotação na posição p, o produto escalar entre uma consulta na posição p e uma chave na posição q pode ser reescrito:

```text
dot(R_p · Q, R_q · K) = Qᵀ · R_(q−p) · K
```

A diferença entre posições aparece no produto escalar. Isso não significa que todo o modelo só depende de distâncias: os conteúdos, a máscara causal, as fronteiras da janela e as demais camadas também afetam o resultado.

## Uma conta em duas dimensões

```python
import numpy as np
from glassllm import Tensor
from glassllm.autograd import rope
from glassllm.nn import rope_angles

cos, sin = rope_angles(d_k=2, max_len=2, base=10000.)
x = Tensor([[[[1., 0.], [1., 0.]]]])
y = rope(x, cos, sin).data
print(np.round(y[0, 0], 6))
print(round(float(y[0, 0, 0] @ y[0, 0, 1]), 6))
```

Saída verificada:

```text
[[1.000000, 0.000000],
 [0.540302, 0.841471]]
0.540302
```

Os dois vetores começaram iguais. Na posição zero, o ângulo é zero; na posição um, é um radiano. Seu produto escalar passa a ser `cos(1)`, mantendo norma 1 para ambos.

## RoPE durante a geração com cache

`Attention.__call__` seleciona `cos[start:start+n]` e `sin[start:start+n]`. No primeiro passe, as posições do prompt são `0…n−1`. No passe incremental, a nova consulta e a nova chave usam a posição `start`, enquanto as chaves antigas já estão rotacionadas dentro do cache.

É incorreto rotacionar novamente as chaves armazenadas ou sempre usar posição zero para o token novo. `GlassLM.generate` mantém `start` enquanto a janela comporta o histórico. Quando ela enche, reconstrói K/V a partir dos últimos 128 tokens e reinicia as posições da janela em zero. Esse comportamento corresponde a um forward completo sobre a janela recortada, não a posições absolutas ilimitadas.

## Conferir no inspetor e referência

Compare `attn.q` com `attn.rope_q`, ou `attn.k` com `attn.rope_k`, na mesma camada e cabeça. Seus shapes permanecem iguais; mudam os números. Q tem quatro cabeças e K tem duas. No token de posição zero, a rotação é a identidade, então a igualdade dos valores é esperada.

O modelo exige `dₖ` par e recusa posições fora de `max_len`. A presença de RoPE não oferece extrapolação automática para contextos maiores nessa implementação. Su et al. apresentam o mecanismo em [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), primeiro preprint de 2021. Veja também [GQA e cache](../gqa-cache/).
