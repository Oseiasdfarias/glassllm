---
title: "Operações diferenciáveis"
description: "Assinaturas, formas, derivadas e estabilidade das funções em glassllm.autograd."
---

## Convenção comum

Todas as funções desta página são importadas de `glassllm.autograd`, não da raiz. Exceto `softmax_np` e `cross_entropy_grad`, elas devolvem `Tensor`. Inputs numéricos ou arrays são convertidos para Tensor constante; um resultado participa do grafo quando algum input requer gradiente e `no_grad` não está ativo. Erros de broadcasting, eixos, formas e domínio seguem principalmente as regras NumPy.

O backward aplica a regra da cadeia e soma dimensões que foram expandidas por broadcasting. O gradiente é em relação aos Tensors, não aos seletores de índice, máscaras, expoentes constantes, número de repetições ou tabelas trigonométricas. As operações têm o contrato de dados reais e derivadas de primeira ordem.

## Aritmética e produto matricial

| Assinatura | Forward e backward |
| --- | --- |
| `add(a, b)` | `a+b`; envia `g` a ambos, reduzindo broadcast |
| `neg(a)` | `-a`; envia `-g` |
| `mul(a, b)` | `a*b`; envia `g*b` e `g*a` |
| `div(a, b)` | `a/b`; envia `g/b` e `-g*a/b²` |
| `power(a, exponent)` | `a**exponent`; envia `g*exponent*a**(exponent-1)`; expoente zero envia zero |
| `matmul(a, b)` | Produto `@`, com vetores e batch; usa produtos com os últimos eixos transpostos e desfaz promoções |
| `exp(a)` | `exp(a)`; envia `g*exp(a)` |
| `log(a)` | Logaritmo natural; envia `g/a` |

Divisão por zero, log de não positivos e potências fora do domínio real não têm política própria de erro: podem emitir warnings ou produzir inf/NaN. `exp` puro não aplica clipping. A proteção numérica das distribuições é implementada nas funções específicas abaixo.

## Reduções, forma e seleção

| Assinatura | Contrato |
| --- | --- |
| `tsum(a, axis=None, keepdims=False)` | Soma; backward expande e difunde `g` à forma original |
| `mean(a, axis=None, keepdims=False)` | `tsum / count`; aceita eixo ou tupla de eixos |
| `max(a, axis=None, keepdims=False)` | Máximo; em empates distribui `g` igualmente entre vencedores |
| `reshape(a, shape)` | Nova forma com mesmo número de elementos; backward volta à forma original |
| `transpose(a, axes)` | Permutação; `axes=None` inverte eixos; backward usa permutação inversa |
| `concat(tensors, axis=0)` | Concatena sequência não vazia de Tensors; backward separa os trechos de cada input |
| `where(condition, a, b)` | Seleciona `a` onde condição é True e `b` no restante; só o ramo escolhido recebe gradiente |

`tsum` tem esse nome para evitar confusão com o `sum` embutido; `max` existe no submódulo e também como `Tensor.max`. `condition` deve ser uma máscara NumPy booleana compatível com as formas. O eixo e o número de elementos não são diferenciáveis.

## Embedding e máscara

`embedding(weight, ids)` consulta linhas diretamente, sem construir one-hot. Com peso `(V,d)` e IDs `(B,n)`, retorna `(B,n,d)`; o backward soma gradientes nas linhas repetidas usando scatter-add. IDs precisam ser inteiros e válidos; a indexação mantém as regras NumPy, incluindo índices negativos.

`masked_fill(a, mask, value)` equivale a `where(mask, value, a)`. `mask=True` significa **substituir/bloquear**, não permitir. Valores mascarados não recebem gradiente de `a`. Com atenção, `value=-np.inf` produz peso zero para posições futuras após softmax, desde que haja pelo menos uma chave não mascarada em cada linha.

```python
import numpy as np
from glassllm import Tensor
from glassllm.autograd import embedding, masked_fill, softmax

weight = Tensor(np.zeros((4, 2)), requires_grad=True)
embedding(weight, np.array([1, 2, 1])).sum().backward()
np.testing.assert_array_equal(weight.grad[1], [2.0, 2.0])
np.testing.assert_array_equal(weight.grad[2], [1.0, 1.0])
scores = Tensor(np.zeros((3, 3)), requires_grad=True)
mask = np.triu(np.ones((3, 3), dtype=bool), k=1)
p = softmax(masked_fill(scores, mask, -np.inf))
assert np.all(p.data[mask] == 0)
np.testing.assert_allclose(p.data.sum(-1), 1.0)
```

## Distribuições e ativações

| Assinatura | Retorno e regra |
| --- | --- |
| `softmax_np(x, axis=-1)` | Array `exp(x-max) / sum(exp(x-max))`; sem grafo |
| `softmax(a, axis=-1)` | Tensor mesma forma; backward `p*(g-sum(g*p))` no eixo escolhido |
| `log_softmax(a, axis=-1)` | Tensor de log-probabilidades estáveis; backward `g-exp(y)*sum(g)` |
| `sigmoid(a)` | `1/(1+exp(-a))`; implementação estável usando `exp(-abs(a))`; backward `g*s*(1-s)` |
| `silu(a)` | `a*sigmoid(a)`; backward `g*s*(1+a*(1-s))` |

Softmax e log-softmax subtraem o máximo para evitar overflow em logits finitos grandes. Isso não resolve uma linha toda `-inf` ou dados NaN/+inf: essas entradas podem gerar NaN. Sigmoid aceita extremos finitos como ±10.000 sem overflow da exponencial; SiLU preserva esse cuidado, mas não promete resultados finitos para dados não finitos.

```python
import numpy as np
from glassllm import Tensor
from glassllm.autograd import log_softmax, sigmoid, silu, softmax_np

np.testing.assert_allclose(softmax_np([[1000.0, 1000.0]]), [[0.5, 0.5]])
np.testing.assert_allclose(log_softmax(Tensor([[1000.0, 1000.0]])).data,
                           [[-np.log(2.0), -np.log(2.0)]])
np.testing.assert_allclose(sigmoid(Tensor([-10000.0, 0.0, 10000.0])).data,
                           [0.0, 0.5, 1.0])
x = Tensor([0.0], requires_grad=True)
silu(x).sum().backward()
np.testing.assert_allclose(x.grad, [0.5])
```

## Cross-entropy

`cross_entropy(logits, targets)` exige logits `(N,V)` e targets inteiros `(N,)`; retorna Tensor escalar da média `-log p[target]`. Não aceita uma distribuição one-hot como target e não oferece `ignore_index`, suavização, pesos de classe ou redução configurável. Não aplique softmax antes: o input é o logit não normalizado.

`cross_entropy_grad(logits, targets) -> np.ndarray` retorna `(softmax(logits)-one_hot(targets))/N`, com forma `(N,V)`, sem grafo nem alteração dos logits. A loss usa essa função em seu backward. A implementação supõe pelo menos uma amostra e índices válidos; incompatibilidades geram erros NumPy e batch vazio pode produzir NaN.

```python
import numpy as np
from glassllm import Tensor
from glassllm.autograd import cross_entropy, cross_entropy_grad

logits = Tensor([[0.0, 0.0]], requires_grad=True)
targets = np.array([1], dtype=np.int64)
loss = cross_entropy(logits, targets)
np.testing.assert_allclose(loss.data, np.log(2.0))
loss.backward()
np.testing.assert_allclose(logits.grad, [[0.5, -0.5]])
np.testing.assert_allclose(cross_entropy_grad(logits.data, targets), logits.grad)
```

## RoPE e repetição de K/V

`rope(x, cos, sin)` retorna Tensor com a mesma forma de `x`; o último eixo precisa ter pares de coordenadas. Tabelas `cos/sin` de forma `(n,dk/2)` são difundidas sobre batch e cabeças de `x` `(B,H,n,dk)`. Cada par adjacente `(even,odd)` é rotacionado; backward usa a rotação transposta, isto é, troca o sinal do seno. Apenas `x` é diferenciável. A função não gera ângulos nem valida todos os casos manualmente; use `rope_angles` da [referência de camadas](../camadas-cache/).

`repeat_kv(x, repeats)` exige `x` `(B,Hkv,n,dk)` e repete cabeças adjacentes, produzindo `(B,Hkv*repeats,n,dk)`. Backward reagrupa o eixo de cabeças e soma repetições. `repeats` deve ser inteiro positivo para o uso no modelo; não há derivada nem validação específica de GQA nesta função.

```python
import numpy as np
from glassllm import Tensor
from glassllm.autograd import repeat_kv, rope

x = Tensor([[1.0, 2.0, 3.0, 4.0]])
rotated = rope(x, np.zeros((1, 2)), np.ones((1, 2)))
np.testing.assert_allclose(rotated.data, [[-2.0, 1.0, -4.0, 3.0]])
heads = Tensor(np.array([1.0, 2.0]).reshape(1, 2, 1, 1), requires_grad=True)
repeated = repeat_kv(heads, 2)
np.testing.assert_array_equal(repeated.data.ravel(), [1, 1, 2, 2])
repeated.sum().backward()
np.testing.assert_array_equal(heads.grad.ravel(), [2, 2])
```

As bases são [diferenciação automática](https://arxiv.org/abs/1502.05767), [RoPE](https://arxiv.org/abs/2104.09864) e [SwiGLU](https://arxiv.org/abs/2002.05202). O código fornece essas peças matemáticas, sem tentar reproduzir todas as arquiteturas ou experimentos dos artigos.
