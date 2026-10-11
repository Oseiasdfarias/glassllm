---
title: "RMSNorm, pre-norm e resíduos"
description: "Normalização da escala e duas somas residuais por bloco, com o papel de epsilon e dos gradientes."
---

## Controlar a escala de cada posição

Cada token chega aos blocos como um vetor de 128 coordenadas. A RMSNorm ajusta a escala desse vetor a partir da média de seus quadrados. Ela opera sobre o último eixo e não mistura tokens nem elementos do batch.

```text
r(x) = sqrt((x₀² + … + x₁₂₇²)/128 + ε)
RMSNorm(x) = γ ⊙ x/r(x)

γ: (128,), aprendido e iniciado em 1
ε: 0,000001 na configuração padrão
```

`RMSNorm.__call__` em `src/glassllm/nn.py` escreve a operação diretamente com `Tensor`: quadrado, média, soma de epsilon, raiz, divisão e multiplicação por `gamma`. Não há um backward especial separado; o autograd deriva cada componente dessa expressão.

Ao contrário de LayerNorm, RMSNorm não subtrai a média nem adiciona um parâmetro beta. Um vetor constante diferente de zero continua não centralizado. Com epsilon positivo, um vetor de zeros permanece zero, sem divisão por zero.

## Exemplo numérico e efeito de epsilon

```python
import numpy as np
from glassllm import Tensor
from glassllm.nn import RMSNorm

norm = RMSNorm(d=2, eps=1e-6, dtype=np.float64)
a = norm(Tensor([[3., 4.]])).data
b = norm(Tensor([[6., 8.]])).data
print(np.round(a, 6))
print(b - a)
```

Saída verificada:

```text
[[0.848528, 1.131371]]
[[2.54558422e-08, 3.39411230e-08]]
```

A média dos quadrados de `[3,4]` é 12,5, então o divisor é `sqrt(12,5 + 10⁻⁶)`. Multiplicar a entrada por 2 produz quase a mesma saída, mas não exatamente a mesma. Para `c>0`:

```text
RMSNorm(c·x) = γ ⊙ x / sqrt(mean(x²) + ε/c²)
```

A invariância de escala é exata na expressão ideal com `ε=0`, para vetores não nulos. Com o epsilon utilizado no código, ela é aproximada quando a energia do vetor domina epsilon. Se gamma não for uniforme, a escala de saída também muda por coordenada; não se deve afirmar que toda saída tem norma fixa ou variância um.

## Duas somas residuais em cada bloco

`Block.__call__` usa a normalização **antes** de cada ramo, posição conhecida como pre-norm:

```text
H  = X + Attention(RMSNorm₁(X))
X' = H + SwiGLU(RMSNorm₂(H))
```

O fluxo residual mantém shape `(B,n,128)`. A atenção escreve informação obtida do contexto; [SwiGLU](../swiglu/) escreve uma transformação por posição. Cada ramo devolve o mesmo shape de sua entrada para permitir a soma.

```text
X ─────────────────────── + ── H ────────────────── + ── X'
 └─ RMSNorm₁ → Attention ─┘      └─ RMSNorm₂ → FFN ─┘
```

Cada bloco tem dois vetores gamma de 128 parâmetros. Quatro blocos contribuem com oito RMSNorms e existe uma RMSNorm final antes dos logits: nove normalizações, 1.152 parâmetros no total. As somas residuais não acrescentam parâmetros.

## O caminho do gradiente

Para uma transformação residual `y=x+F(x)`, o Jacobiano é `I+J_F`. Existe uma contribuição direta de identidade no backward, além da contribuição do ramo. Isso ajuda a entender por que resíduos facilitam otimização.

Esse caminho **não garante** que o gradiente nunca desapareça ou exploda. Os termos podem se cancelar, como no exemplo `F(x)=−x`, e produtos de Jacobianos podem amplificar valores. Pre-norm também não elimina a necessidade de uma taxa de aprendizado adequada, inicialização e controle de gradientes no treinamento.

## Ler no mapa e referências

Compare `norm.attn` com `resid.attn` e `norm.ffn` com `resid.ffn` na mesma camada. Os snapshots de normalização representam a entrada do ramo; os de resíduo representam a soma concluída. `norm.final` pertence à saída do modelo, depois do quarto bloco. O inspetor remove o batch e apresenta `(n,128)`.

Zhang e Sennrich apresentam RMSNorm em [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467), 2019. Xiong et al. analisam a posição pre-norm em [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745), 2020; o artigo trata de LayerNorm, enquanto este projeto usa RMSNorm. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), He et al., primeiro preprint de 2015, é uma referência histórica para conexões residuais em visão, não uma descrição deste decoder.
