---
title: "Tensor e motor de autograd"
description: "Arrays, operadores, diferenciação reversa, acumulação de gradientes e no_grad."
---

## Construir um Tensor

`from glassllm import Tensor, no_grad` é a importação pública. A assinatura `Tensor(data, parents=(), backward=None, requires_grad=False)` converte `data` com `np.asarray`, sem impor dtype ou criar sempre uma cópia. O uso normal fornece apenas `data` e `requires_grad`; `parents` e `backward` são o mecanismo interno de construção das operações.

| Atributo | Comportamento |
| --- | --- |
| `data` | Array NumPy com os valores atuais; pode compartilhar memória com a entrada |
| `grad` | `None` inicialmente; array com gradiente acumulado após backward |
| `requires_grad` | True se solicitado ou herdado de um pai, desde que o rastreamento esteja habilitado |
| `shape`, `ndim` | Propriedades que consultam o array |

Use arrays reais float32 ou float64 para derivadas. A classe aceita inteiros, mas isso não constitui um sistema de diferenciação discreta nem treinamento em parâmetros inteiros. Não existe `.detach()`, `.item()`, `.T` ou conversão implícita para um tensor PyTorch; leia `.data`, `float(t.data)` para escalares e `.transpose()` para eixos.

## Operadores e métodos

| Sintaxe | Operação e limite |
| --- | --- |
| `a+b`, `a-b`, `a*b`, `a/b` | Elemento a elemento, com broadcasting e operações reversas como `2-a` |
| `-a`, `a**exponent` | Negação e potência com expoente constante; sem derivada do expoente |
| `a @ b` | Produto matricial NumPy, vetores promovidos e batch broadcast |
| `a[index]` | Indexação NumPy; backward usa soma nos índices repetidos |
| `a.sum(axis=None, keepdims=False)` | Redução por eixo, tupla de eixos ou todos |
| `a.mean(axis=None, keepdims=False)` | Soma dividida pelo número de elementos reduzidos |
| `a.max(axis=None, keepdims=False)` | Máximo; divide gradiente entre empates |
| `a.reshape(*shape)` | Aceita dimensões separadas ou tupla/lista, incluindo `-1` |
| `a.transpose(*axes)` | Sem argumentos inverte eixos; aceita eixos separados ou tupla/lista |

Operações devolvem novos Tensors; incompatibilidades de formas ou índices produzem erros NumPy. O operador `@` está implementado em Tensor à esquerda; não há `__rmatmul__` dedicado. Compare valores com operações NumPy em `.data`: a classe não oferece operadores de comparação diferenciáveis nem atribuição por índice do tipo `a[i] = ...`.

## backward e gradiente inicial

`backward(grad=None) -> None` visita o grafo em ordem topológica reversa. Sem `grad`, usa `ones_like(data)`: para saída não escalar, isso diferencia a **soma de suas entradas**. Para calcular outro produto vetor-Jacobiano, forneça um gradiente com a forma da saída. Uma saída sem `requires_grad` não faz nada.

O algoritmo soma contribuições de caminhos compartilhados. Folhas acumulam gradientes entre chamadas; intermediários são limpos a cada percurso para impedir a reutilização de valores antigos. Antes de uma nova atualização de treinamento, faça `p.grad = None` ou `optimizer.zero_grad()`. `backward` não zera gradientes de todas as folhas e não atualiza parâmetros.

```python
import numpy as np
from glassllm import Tensor

x = Tensor([3.0], requires_grad=True)
y = x * x + x
y.backward()
np.testing.assert_allclose(x.grad, [7.0])
y.backward()
np.testing.assert_allclose(x.grad, [14.0])
x.grad = None
y.backward(np.array([2.0]))
np.testing.assert_allclose(x.grad, [14.0])
```

O broadcasting replica valores no forward, portanto o backward **soma** as contribuições das réplicas para voltar à forma original. A indexação tem o mesmo cuidado quando um ID de embedding aparece várias vezes.

```python
import numpy as np
from glassllm import Tensor

x = Tensor(np.ones((2, 3)), requires_grad=True)
bias = Tensor([1.0, 2.0, 3.0], requires_grad=True)
(x + bias).sum().backward()
np.testing.assert_allclose(bias.grad, [2.0, 2.0, 2.0])
np.testing.assert_allclose(x.grad, np.ones((2, 3)))
ties = Tensor([2.0, 2.0, 1.0], requires_grad=True)
ties.max().backward()
np.testing.assert_allclose(ties.grad, [0.5, 0.5, 0.0])
```

## no_grad

`no_grad()` é um context manager: resultados criados dentro do bloco não guardam pais nem funções de backward. Ele não apaga `requires_grad` de um parâmetro já existente nem zera gradientes. Blocos aninhados e exceções restauram o estado anterior ao sair.

```python
from glassllm import Tensor, no_grad

x = Tensor([3.0], requires_grad=True)
with no_grad():
    y = x * 2
    assert not y.requires_grad
    y.backward()
assert x.grad is None
assert (x * 2).requires_grad
```

O estado de rastreamento é uma variável global do módulo, não um contexto isolado por thread ou coroutine. Não sobreponha treinamento e inferência concorrentes no mesmo processo supondo isolamento automático.

## Limites do grafo

Backward produz arrays NumPy, sem construir um grafo das próprias derivadas: não há higher-order autograd. Operações fora das funções documentadas, como `np.sin(t.data)`, interrompem a cadeia. Mutar `.data` depois do forward pode invalidar valores que closures do backward ainda usam; termine o backward antes de atualizar parâmetros. Dados complexos não têm regras específicas de conjugação e não são o contrato deste motor.

O projeto implementa o modo reverso descrito na [revisão de diferenciação automática de Baydin et al.](https://arxiv.org/abs/1502.05767), com um conjunto pequeno de operações vetorizadas e sem a infraestrutura de um framework completo. A [referência funcional](../operacoes-autograd/) detalha derivadas e estabilidade.
